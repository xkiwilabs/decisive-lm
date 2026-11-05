# Request time: the heads against the prompted base models (P0)

Measured 2026-11-06 at 01:00 by `scripts/paper_v5c_26b/p0_latency.py`, with nothing else running on either vLLM
server when each request started (`vllm:num_requests_running` was 0 throughout).

**The request.** It matches the heads' latency request: a state of about 450 words (about 1,000 tokens) and four
Choice questions of five options each.
- *How P0 answers it:* P0 reads the answer label's log-probabilities once per option order, five orders per question,
  so the request is 20 prompts of about 900 tokens each. It runs through vLLM's OpenAI-compatible server, with prefix
  caching on, so the 20 prompts share the state's cache.
- *Fresh state each run:* every run uses a different state, so no state is cached from an earlier run.
- *Runs:* 5 warm-up runs, then 20 timed runs per line below.

| Method | How the 20 prompts are sent | p50 | p95 |
|---|---|---|---|
| P0 E4B (vLLM) | one question after another, as `decisivelm.synth.teacher` labels | 803 ms | 966 ms |
| P0 E4B (vLLM) | all 20 at once, the fastest a client can go | 613 ms | 668 ms |
| P0 26B A4B (vLLM) | one question after another | 1,109 ms | 1,223 ms |
| P0 26B A4B (vLLM) | all 20 at once | 577 ms | 632 ms |
| E4B head | one forward pass, in process (`results/paper_v5c_e4b.md` §6) | 102 ms | 102 ms |
| 26B A4B head | one forward pass, in process (float32 computation) | 451 ms | 462 ms |
| 26B A4B head | one forward pass, over HTTP through `decisivelm.serve` | 473 ms | 475 ms |

**Reading it:**
- **The E4B head** answers the request 6 to 8 times faster than the prompted E4B.
- **The 26B A4B head** is 1.2 to 2.3 times faster than the prompted 26B A4B. Its float32 computation costs most of the
  gain: P0 runs in bf16.
- **One pass for the whole request.** The head's time is one forward pass however the options are ordered. P0 needs
  one prompt per option order, so its time grows with the number of orders it averages over (up to 8 per question).

**The setups differ.**
- *Hardware:* all runs were on RTX PRO 6000 Blackwell Max-Q cards.
- *Software:* P0 ran on vLLM, an optimised inference server with continuous batching. The heads ran in PyTorch through
  Hugging Face transformers.
- *Interface:* the heads' rows are in process except the last. The P0 rows include an HTTP call per prompt to a server
  on the same machine.
