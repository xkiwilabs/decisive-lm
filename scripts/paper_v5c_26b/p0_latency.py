"""Request time of the prompted baselines (P0), for comparison with the heads' 102 ms (E4B) and 473 ms (26B A4B).

The request matches the heads' latency request: a state of about 1,000 tokens and four Choice questions of five options.
P0 answers each question by reading the answer label's log-probs once per option order (five rotations of five
options), through the vLLM servers in configs/teachers.yaml. Two ways of sending the 20 prompts:
- "as the labeller sends them": decisivelm.synth.teacher.question_labels, one question after another;
- "all at once": the 20 prompts in parallel, the fastest a client could go.
Each run uses a different state, so the state is never in vLLM's prefix cache from an earlier run; the 20 prompts of
one request share it, as they would in use. Run at night (the hub is shared); vllm:num_requests_running is recorded.

    uv run python scripts/paper_v5c_26b/p0_latency.py --text README.md docs/guide/*.md
"""
import argparse, statistics, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

from decisivelm.api_types import ChoiceQuestion
from decisivelm.model import load_tokenizer
from decisivelm.synth.label import choice_orders, choice_prompt
from decisivelm.synth.teacher import Gate, Schedule, VLLMTeacher, load_teacher_config, question_labels, server_waiting


def running(base_url: str, model: str) -> int:
    with urllib.request.urlopen(base_url.rsplit("/v1", 1)[0] + "/metrics", timeout=5) as r:
        text = r.read().decode()
    return int(sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
                   if line.startswith("vllm:num_requests_running{") and f'model_name="{model}"' in line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs="+", required=True, help="long English text that states are cut from")
    ap.add_argument("--teachers", nargs="+", default=["gemma4_e4b", "gemma4_26b_a4b"])
    ap.add_argument("--words", type=int, default=450)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args()
    words = " ".join(open(f).read() for f in a.text).split()
    state = lambda k: " ".join(words[300 * k: 300 * k + a.words])
    qs = [ChoiceQuestion(type="choice", instructions=f"Which of these best describes part {j} of the text? "
                         + " ".join(words[50 * j: 50 * j + 6]),
                         criteria={f"o{i}": " ".join(words[1000 + 97 * j + 13 * i: 1000 + 97 * j + 13 * i + 6])
                                   for i in range(5)}) for j in range(4)]
    for name in a.teachers:
        cfg = load_teacher_config(name)
        tok = load_tokenizer(cfg["tokenizer_backbone"])
        teacher = VLLMTeacher(cfg, tok, Gate(Schedule(**cfg["schedule"]), server_waiting(cfg["base_url"], cfg["model"])))
        enc = lambda text: tok.encode(text, add_special_tokens=False)
        pool = ThreadPoolExecutor(max_workers=20)
        for mode in ("as the labeller sends them", "all at once"):
            times, others, n_prompts, n_tokens = [], [], 0, 0
            for k in range(a.warmup + a.runs):
                s = state(k + (0 if mode.startswith("as") else a.warmup + a.runs))
                prompts = [enc(sh + it) for q in qs for sh, it in
                           (choice_prompt(q, s, order, tok) for order in choice_orders(len(q.criteria), f"lat-{k}"))]
                others.append(running(cfg["base_url"], cfg["model"]))
                t0 = time.perf_counter()
                if mode.startswith("as"):
                    for q in qs:
                        question_labels(q, [s], [f"lat-{k}"], teacher, cfg.get("max_logprobs", 20))
                else:
                    list(pool.map(teacher.top_logprobs, prompts))
                if k >= a.warmup:
                    times.append(time.perf_counter() - t0)
                n_prompts, n_tokens = len(prompts), statistics.mean(map(len, prompts))
            times.sort()
            print(f"{name} | {mode} | {n_prompts} prompts of {n_tokens:.0f} tokens on average | "
                  f"p50 {statistics.median(times) * 1000:.0f} ms | p95 {times[int(0.95 * len(times)) - 1] * 1000:.0f} ms | "
                  f"other requests running at the start: max {max(others)}", flush=True)


if __name__ == "__main__":
    main()
