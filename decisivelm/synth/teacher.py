"""Teacher labelling over an OpenAI-compatible vLLM server: method (a) for every question type (docs/guide/12).

- Choice: options lettered under every rotation (n ≤ 8) or 8 seeded permutations, averaged (`label.choice_labels`).
- Score: levels lettered in natural and in reversed order, averaged. Shuffling an ordered scale would set the teacher
  a different task from the student's.
- Noul: P(Yes) / (P(Yes) + P(No)), averaged over the phrasings in prompts/noul_v1.yaml.

Prompts are rendered with the teacher's own chat template, thinking off, the assistant turn pre-filled with "Answer:"
and sent as token ids, so the next token is the label. Each call records the probability on allowed labels.

`LocalTeacher` runs the same prompts on a local transformers model, with a row's images and audio where the state
starts: the media baseline (P0 on image and audio rows), which the server path does not send.

The server is shared (configs/teachers.yaml `schedule`): a `Gate` caps the requests in flight by time of day and, in
the day, holds new requests while other users' requests are queued on the server.
"""
import datetime
import json
import math
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import torch
import yaml

from decisivelm import CONFIGS
from decisivelm.api_types import ChoiceQuestion
from decisivelm.calibrate import question_names
from decisivelm.media import (AudioTokens, EncodedAudio, EncodedImage, ImageTokens, encode_audio, encode_image,
                              load_audio, load_image, stack, stack_audio)
from decisivelm.render import MEDIA_SEPARATOR, as_text
from decisivelm.synth.label import (LETTERS, MAX_LETTERED, PROMPTS, SENTINEL, choice_labels, digit_scorer, label_ids,
                                    label_probs, numbered, prompt_template)

from decisivelm.synth.label import MIN_ALLOWED

NOUL_WORDS = ("Yes", "No")
EPS = 1e-6


def kept(masses: list[float]) -> list[bool]:
    """Which calls count: those with allowed mass ≥ MIN_ALLOWED, or all of them if none has (the item is then
    flagged by its low mean allowed mass)."""
    ok = [m >= MIN_ALLOWED for m in masses]
    return ok if any(ok) else [True] * len(masses)


def teacher_logits(q, ans) -> torch.Tensor:
    """Teacher probabilities as logits for decisivelm.calibrate: log p (Choice/Score), logit P(yes) (Noul)."""
    if q.type == "noul":
        p = min(max(ans["p"], EPS), 1 - EPS)
        return torch.tensor([math.log(p / (1 - p))])
    return torch.tensor([math.log(max(ans["probabilities"][k], EPS)) for k in question_names(q)])


def load_teacher_config(name: str) -> dict:
    from decisivelm.model import check_origin
    return check_origin(yaml.safe_load((CONFIGS / "teachers.yaml").read_text())["teachers"][name], f"teacher {name}")


class Schedule:
    def __init__(self, day_hours=(6, 23), day_max_in_flight=4, night_max_in_flight=64, yield_when_others_waiting=True):
        self.day_hours, self.day, self.night = tuple(day_hours), day_max_in_flight, night_max_in_flight
        self.yield_when_others_waiting = yield_when_others_waiting

    def is_day(self, now: datetime.datetime | None = None) -> bool:
        h = (now or datetime.datetime.now()).hour
        return self.day_hours[0] <= h < self.day_hours[1]

    def limit(self, now=None) -> int:
        return self.day if self.is_day(now) else self.night


class Gate:
    """Requests in flight ≤ the schedule's limit, re-read on every acquire, so the limit follows the clock. In the day,
    while the server reports queued requests, nothing new is sent: those are other users' requests, since ours stay
    within a few in flight."""

    def __init__(self, schedule: Schedule, waiting=lambda: 0, poll: float = 5.0):
        self.schedule, self._waiting, self.poll = schedule, waiting, poll
        self.in_flight, self._cond = 0, threading.Condition()
        self._checked, self._others = 0.0, 0

    def others_waiting(self) -> int:
        if time.monotonic() - self._checked > self.poll:
            try:
                self._others = self._waiting()
            except Exception:
                self._others = 0
            self._checked = time.monotonic()
        return self._others

    def limit(self) -> int:
        if self.schedule.is_day() and self.schedule.yield_when_others_waiting and self.others_waiting() > 0:
            return 0
        return self.schedule.limit()

    def __enter__(self):
        with self._cond:
            while self.in_flight >= self.limit():
                self._cond.wait(timeout=1.0)
            self.in_flight += 1

    def __exit__(self, *exc):
        with self._cond:
            self.in_flight -= 1
            self._cond.notify_all()


def server_waiting(base_url: str, model: str):
    """A callable reading vllm:num_requests_waiting for `model` from the server's /metrics."""
    url = base_url.rsplit("/v1", 1)[0] + "/metrics"

    def read() -> int:
        with urllib.request.urlopen(url, timeout=5) as r:
            text = r.read().decode()
        return int(sum(float(line.rsplit(" ", 1)[1]) for line in text.splitlines()
                       if line.startswith("vllm:num_requests_waiting{") and f'model_name="{model}"' in line))
    return read


class VLLMTeacher:
    def __init__(self, cfg: dict, tokenizer, gate: Gate, timeout: float = 300, retries: int = 6, workers: int = 128):
        self.cfg, self.tokenizer, self.gate = cfg, tokenizer, gate
        self.timeout, self.retries = timeout, retries
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.requests, self.prompt_tokens, self.completion_tokens = 0, 0, 0
        self._lock = threading.Lock()

    def _post(self, endpoint: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if (key := os.environ.get(self.cfg.get("api_key_env") or "")):
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.cfg["base_url"].rstrip("/") + endpoint, data=json.dumps(body).encode(),
                                     headers=headers)
        for attempt in range(self.retries):
            try:
                with self.gate, urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                break
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(min(60, 2 ** attempt))
        with self._lock:
            self.requests += 1
            self.prompt_tokens += out["usage"]["prompt_tokens"]
            self.completion_tokens += out["usage"].get("completion_tokens", 0)
        return out

    def chat_json(self, messages: list[dict], schema: dict, seed: int, temperature: float = 1.0,
                  max_tokens: int = 4096) -> tuple[dict | None, str]:
        """Sampled JSON under `schema` (structured outputs), thinking off: (parsed or None, finish reason)."""
        out = self._post("/chat/completions", {
            "model": self.cfg["model"], "messages": messages, "temperature": temperature, "top_p": 0.95,
            "seed": seed, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}})
        choice = out["choices"][0]
        try:
            return json.loads(choice["message"]["content"]), choice.get("finish_reason", "")
        except (json.JSONDecodeError, TypeError):
            return None, f"{choice.get('finish_reason', '')}: unparsable {str(choice['message'].get('content'))[-200:]!r}"

    def top_logprobs(self, ids: list[int]) -> dict[int, float]:
        out = self._post("/completions", {"model": self.cfg["model"], "prompt": ids, "max_tokens": 1, "temperature": 0,
                                          "logprobs": self.cfg.get("max_logprobs", 20),
                                          "return_tokens_as_token_ids": True})
        top = out["choices"][0]["logprobs"]["top_logprobs"][0]
        return {int(k.split(":", 1)[1]): v for k, v in top.items()}

    def next_token_logprobs(self, shared: str, items: list[str], extra: list[list[int]] | None = None) -> list[dict[int, float]]:
        """Top log-probs after `shared + item`, then the token ids `extra[i]` if given."""
        enc = lambda text: self.tokenizer.encode(text, add_special_tokens=False)   # the text starts with <bos>
        extra = extra or [[] for _ in items]
        return list(self.pool.map(lambda ie: self.top_logprobs(enc(shared + ie[0]) + ie[1]), zip(items, extra)))

    def scorer(self, labels: list[list[int]]):
        """(shared, items) → [(probabilities over `labels`, allowed mass)]; a label absent from the top-k has p = 0."""
        def score(shared, items):
            out = []
            for lp in self.next_token_logprobs(shared, items):
                mass = torch.tensor([sum(math.exp(lp[i]) for i in ids if i in lp) for ids in labels])
                allowed = float(mass.sum())
                out.append((mass / allowed if allowed > 0 else torch.full((len(labels),), 1 / len(labels)), allowed))
            return out
        return score


class LocalTeacher:
    """VLLMTeacher's contract on a local transformers model with its lm_head (decisivelm.model.load_causal_lm), for
    rows with images or audio, which the vLLM path does not send. `bound(media)` returns a view for one row: its
    images and audio are encoded as the student encodes them (decisivelm.media) and placed where the state starts, so
    the prompt reads "State:\\n<media>\\n<state text>", as the student's prefix does. One prompt per forward pass, full
    next-token distribution."""

    def __init__(self, model, tokenizer, image_processor=None, feature_extractor=None, image_budget: int = 280,
                 media_root=None, media=()):
        self.model, self.tokenizer = model, tokenizer
        self.image_processor, self.feature_extractor = image_processor, feature_extractor
        self.image_budget, self.media_root, self.media = image_budget, media_root, list(media)
        self.image_tokens, self.audio_tokens = ImageTokens.from_config(model.config), AudioTokens.from_config(model.config)
        self.device = next(model.parameters()).device
        self.requests = 0

    def bound(self, media) -> "LocalTeacher":
        encoded = []
        for m in media:
            if m.type == "audio":
                encoded.append(encode_audio(self.feature_extractor,
                                            load_audio(m, self.media_root, self.feature_extractor.sampling_rate),
                                            self.audio_tokens))
            else:
                encoded.append(encode_image(self.image_processor, load_image(m, self.media_root), self.image_tokens,
                                            self.image_budget))
        return LocalTeacher(self.model, self.tokenizer, self.image_processor, self.feature_extractor, self.image_budget,
                            self.media_root, encoded)

    def prompt_ids(self, shared: str, item: str, extra: list[int] = ()) -> list[int]:
        enc = lambda text: self.tokenizer.encode(text, add_special_tokens=False)   # the text starts with <bos>
        media = [t for m in self.media for t in m.token_ids + enc(MEDIA_SEPARATOR)]
        return enc(shared) + media + enc(item) + list(extra)

    def _media_kwargs(self) -> dict:
        images = [m for m in self.media if isinstance(m, EncodedImage)]
        clips = [m for m in self.media if isinstance(m, EncodedAudio)]
        kw = {}
        if images:
            pix, pos = stack(images)
            kw |= {"pixel_values": pix.to(self.device, next(self.model.model.vision_tower.parameters()).dtype),
                   "image_position_ids": pos.to(self.device)}
        if clips:
            feats, mask = stack_audio(clips)
            kw |= {"input_features": feats.to(self.device, next(self.model.model.audio_tower.parameters()).dtype),
                   "input_features_mask": mask.to(self.device)}
        return kw

    @torch.no_grad()
    def next_token_logprobs(self, shared: str, items: list[str], extra: list[list[int]] | None = None) -> list[torch.Tensor]:
        extra = extra or [[] for _ in items]
        kw, out = self._media_kwargs(), []
        for item, e in zip(items, extra):
            ids = torch.tensor([self.prompt_ids(shared, item, e)], device=self.device)
            types = (ids == self.image_tokens.image).long() if self.image_tokens else None
            logits = self.model(input_ids=ids, mm_token_type_ids=types, logits_to_keep=1, **kw).logits
            out.append(logits[0, -1].float().log_softmax(-1).cpu())
            self.requests += 1
        return out

    def scorer(self, labels: list[list[int]]):
        return lambda shared, items: [label_probs(lp, labels) for lp in self.next_token_logprobs(shared, items)]


def _chat(tokenizer, system: str, user: str, prefill: str) -> tuple[str, str]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user},
                {"role": "assistant", "content": prefill}]
    shared, tail = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True).split(SENTINEL)
    return shared, tail


def score_labels(q, states: list[str], teacher: VLLMTeacher) -> list[tuple[list[float], dict]]:
    """[(probability per level, low to high), info] per state: natural and reversed lettering, averaged."""
    t, _ = prompt_template("score_v1")
    n = len(q.criteria)
    score = teacher.scorer(label_ids(teacher.tokenizer, n))
    calls = [[] for _ in states]   # per state: (probabilities by level, allowed mass) per ordering
    for order in (list(range(n)), list(range(n))[::-1]):
        options = "\n".join(f"{LETTERS[k]}. {as_text(q.criteria[i])}" for k, i in enumerate(order))
        shared, tail = _chat(teacher.tokenizer, t["system"],
                             t["user"].format(instructions=as_text(q.instructions), options=options, state=SENTINEL),
                             t["prefill"])
        for i, (p, allowed) in enumerate(score(shared, [s.strip() + tail for s in states])):
            by_level = torch.zeros(n)
            by_level[order] = p.float()
            calls[i].append((by_level, allowed))
    out = []
    for c in calls:
        use = [x for x, k in zip(c, kept([m for _, m in c])) if k]
        out.append(((sum(p for p, _ in use) / len(use)).tolist(),
                    {"method": "logprob", "n_orderings": len(c), "n_kept": len(use),
                     "allowed_mass": sum(m for _, m in c) / len(c)}))
    return out


def noul_criteria_text(q) -> str:
    c = q.criteria
    if c is None:
        return ""
    if isinstance(c, str):
        return "\n" + c
    parts = [f"Yes means: {as_text(c.yes)}" if c.yes is not None else "", f"No means: {as_text(c.no)}" if c.no is not None else ""]
    return "".join("\n" + p for p in parts if p)


def noul_labels(q, states: list[str], teacher: VLLMTeacher) -> list[tuple[float, dict]]:
    """[(P(yes), info)] per state, averaged over the phrasings."""
    raw = (PROMPTS / "noul_v1.yaml").read_text()
    t = yaml.safe_load(raw)
    score = teacher.scorer(label_ids_words(teacher.tokenizer, NOUL_WORDS))
    calls = [[] for _ in states]   # per state: (P(yes), allowed mass) per phrasing
    for phrasing in t["phrasings"]:
        shared, tail = _chat(teacher.tokenizer, t["system"],
                             phrasing.format(instructions=as_text(q.instructions), criteria=noul_criteria_text(q),
                                             state=SENTINEL), t["prefill"])
        for i, (p, allowed) in enumerate(score(shared, [s.strip() + tail for s in states])):
            calls[i].append((float(p[0]), allowed))
    out = []
    for c in calls:
        use = [x for x, k in zip(c, kept([m for _, m in c])) if k]
        out.append((sum(p for p, _ in use) / len(use),
                    {"method": "logprob", "n_orderings": len(c), "n_kept": len(use),
                     "allowed_mass": sum(m for _, m in c) / len(c)}))
    return out


def label_ids_words(tokenizer, words) -> list[list[int]]:
    """Per word, the ids of "Yes" and " Yes"; both must be single tokens."""
    out = []
    for w in words:
        ids = [tokenizer.encode(v, add_special_tokens=False) for v in (w, " " + w)]
        if any(len(x) != 1 for x in ids):
            raise ValueError(f"label {w!r} is not a single token: {ids}")
        out.append([x[0] for x in ids])
    return out


def question_labels(q, states: list[str], ids: list[str], teacher: VLLMTeacher, max_labels: int):
    """[(answer, info)] per state, as prediction answers: Choice/Score {"type", "probabilities"}, Noul {"type", "p"}.
    A Choice with more than MAX_LETTERED options is numbered and read digit by digit (per-option scoring, docs/guide/12);
    lettered options must all fit in the server's top log-probs."""
    if q.type == "choice":
        n = len(q.criteria)
        if numbered(n):
            scorer = digit_scorer(teacher.next_token_logprobs, teacher.tokenizer, n)
        elif n > max_labels:
            return [(None, {"skipped": f"{n} options > {max_labels} top log-probs"}) for _ in states]
        else:
            scorer = teacher.scorer(label_ids(teacher.tokenizer, n))
        out = choice_labels(ChoiceQuestion.model_validate(q.model_dump()), states, scorer, teacher.tokenizer, ids)
        return [({"type": "choice", "probabilities": p}, info) for p, info in out]
    if q.type == "score":
        return [({"type": "score", "probabilities": {str(k): v for k, v in enumerate(p)}}, info)
                for p, info in score_labels(q, states, teacher)]
    return [({"type": "noul", "p": p}, info) for p, info in noul_labels(q, states, teacher)]
