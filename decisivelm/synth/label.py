"""Labeller method (a): direct label log-probs (docs/guide/12). Run on the student's own backbone it is P0 (docs/guide/12 §12.8).

The prompt is pre-filled with "Answer:" and thinking is off, so the next token
is the label. Option letters are single tokens; "A" and " A" are merged. A Choice
with more than MAX_LETTERED options is numbered 01, 02, … instead (choice_num_v1)
and its label read one digit at a time (`digit_scorer`): per-option scoring, so
no option needs to be among a server's top log-probs. Choice questions are asked
under every rotation (n ≤ 8) or 8 seeded permutations, and the per-option
probabilities are averaged, which cancels position bias.

A *scorer* is any callable `(shared, items) -> [(label_probs, allowed_mass), ...]`
where each prompt is `shared + item`. `LocalLM` runs one on a transformers
model; a vLLM client can implement the same contract (M4a).
"""
import copy
import hashlib
import math
import pathlib
import random
from collections import defaultdict

import torch
import yaml

from decisivelm.api_types import ChoiceQuestion
from decisivelm.render import as_text

PROMPTS = pathlib.Path(__file__).parent / "prompts"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_LETTERED = 20   # above this a Choice's options are numbered (the server returns 20 top log-probs)
DIGITS = "0123456789"
PRUNE = 1e-3        # a label prefix with less probability is not expanded; its mass is spread over its labels
MAX_ROTATED = 8
SENTINEL = "\x00STATE\x00"
MIN_ALLOWED = 0.5   # calls with less probability on allowed labels are left out of the average (docs/guide/12)


def prompt_template(name: str = "choice_v1") -> tuple[dict, str]:
    """(template, sha256 of the file) — the hash is recorded with every teacher target (docs/guide/12)."""
    raw = (PROMPTS / f"{name}.yaml").read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def choice_orders(n: int, item_id: str) -> list[list[int]]:
    if n <= MAX_ROTATED:
        return [[(start + k) % n for k in range(n)] for start in range(n)]
    rng = random.Random(int(hashlib.sha256(item_id.encode()).hexdigest()[:16], 16))
    orders = []
    for _ in range(MAX_ROTATED):
        order = list(range(n))
        rng.shuffle(order)
        orders.append(order)
    return orders


def numbered(n: int) -> bool:
    return n > MAX_LETTERED


def number_labels(n: int) -> list[str]:
    """"01" … "60": zero-padded to the width of n, so every label has the same number of digits."""
    return [f"{k + 1:0{len(str(n))}d}" for k in range(n)]


def choice_prompt(q: ChoiceQuestion, state: str, order: list[int], tokenizer, template: str | None = None):
    """(shared, item): the chat-formatted prompt split where the state starts."""
    labels = number_labels(len(order)) if numbered(len(order)) else LETTERS
    t, _ = prompt_template(template or ("choice_num_v1" if numbered(len(order)) else "choice_v1"))
    names = list(q.criteria)
    describe = lambda name: f"{name}: {as_text(q.criteria[name])}" if as_text(q.criteria[name]).strip() else name
    options = "\n".join(f"{labels[k]}. {describe(names[i])}" for k, i in enumerate(order))
    user = t["user"].format(instructions=as_text(q.instructions), options=options, state=SENTINEL)
    messages = [{"role": "system", "content": t["system"]}, {"role": "user", "content": user},
                {"role": "assistant", "content": t["prefill"]}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    text += t["prefill"][len(t["prefill"].rstrip()):]   # the template trims it; a numbered answer's space is its own token
    shared, tail = text.split(SENTINEL)
    return shared, state.strip() + tail


def label_ids(tokenizer, n: int) -> list[list[int]]:
    """Per label, the ids of "A" and " A". Both must be single tokens (checked)."""
    out = []
    for letter in LETTERS[:n]:
        ids = [tokenizer.encode(v, add_special_tokens=False) for v in (letter, " " + letter)]
        if any(len(x) != 1 for x in ids):
            raise ValueError(f"label {letter!r} is not a single token: {ids}")
        out.append([x[0] for x in ids])
    return out


def digit_ids(tokenizer) -> list[int]:
    """The ids of "0" … "9"; each must be a single token (Gemma 4 splits numbers into digits)."""
    ids = [tokenizer.encode(d, add_special_tokens=False) for d in DIGITS]
    if any(len(x) != 1 for x in ids):
        raise ValueError(f"digits are not single tokens: {ids}")
    return [x[0] for x in ids]


def _lp(logprobs, token: int) -> float:
    """A token's log-prob from a full log-softmax tensor or a {token: log-prob} top-k dict (absent: -inf)."""
    if isinstance(logprobs, dict):
        return logprobs.get(token, -math.inf)
    return float(logprobs[token])


def digit_scorer(next_token_logprobs, tokenizer, n: int):
    """A scorer over the numbered labels of an n-option Choice. `next_token_logprobs(shared, items, extra)` returns
    the next-token log-probs after `shared + item` followed by the token ids in `extra`. Digit by digit: one call
    for every item, then one more for each label prefix holding at least PRUNE of the item's probability; at each
    step only digits that continue a real label count, renormalised. A pruned prefix's mass is spread evenly over its
    labels. The allowed mass is the first digit's (how much the model answers with a label at all)."""
    labels, ids = number_labels(n), digit_ids(tokenizer)
    width = len(labels[0])

    def score(shared, items):
        probs, allowed = [torch.zeros(n) for _ in items], [0.0] * len(items)
        frontier = [(i, "", 1.0) for i in range(len(items))]
        for pos in range(width):
            lps = next_token_logprobs(shared, [items[i] for i, _, _ in frontier],
                                      [[ids[int(c)] for c in prefix] for _, prefix, _ in frontier])
            grow = []
            for (i, prefix, mass), lp in zip(frontier, lps):
                ok = sorted({x[pos] for x in labels if x.startswith(prefix)})
                p = torch.tensor([math.exp(_lp(lp, ids[int(d)])) for d in ok])
                total = float(p.sum())
                if pos == 0:
                    allowed[i] = total
                p = p / total if total > 0 else torch.full((len(ok),), 1 / len(ok))
                for d, pd in zip(ok, p.tolist()):
                    m, nxt = mass * pd, prefix + d
                    if pos == width - 1:
                        probs[i][labels.index(nxt)] += m
                    elif m >= PRUNE:
                        grow.append((i, nxt, m))
                    else:
                        under = [k for k, x in enumerate(labels) if x.startswith(nxt)]
                        for k in under:
                            probs[i][k] += m / len(under)
            frontier = grow
            if not frontier:
                break
        return list(zip(probs, allowed))
    return score


def label_probs(logprobs: torch.Tensor, ids: list[list[int]]) -> tuple[torch.Tensor, float]:
    """(probabilities over the labels, renormalised; total probability on allowed labels)."""
    mass = torch.stack([logprobs[v].exp().sum() for v in ids])
    allowed = float(mass.sum())
    return mass / allowed, allowed


def choice_labels(q: ChoiceQuestion, states: list[str], scorer, tokenizer, item_ids: list[str]):
    """[(probabilities by option name, info), ...] for each state."""
    n, names = len(q.criteria), list(q.criteria)
    by_order = defaultdict(list)
    for i, item_id in enumerate(item_ids):
        for order in choice_orders(n, item_id):
            by_order[tuple(order)].append(i)
    calls = [[] for _ in states]   # per state: (probabilities by option, allowed mass) per ordering
    for order, members in by_order.items():
        shared = None
        items = []
        for i in members:
            s, item = choice_prompt(q, states[i], list(order), tokenizer)
            shared = shared or s
            items.append(item)
        for i, (p, allowed) in zip(members, scorer(shared, items)):
            by_option = torch.zeros(n)
            by_option[list(order)] = p.float()
            calls[i].append((by_option, allowed))
    out = []
    for c in calls:
        # Orderings with less than MIN_ALLOWED on the labels are left out unless all are.
        ok = [m >= MIN_ALLOWED for _, m in c]
        use = [x for x, k in zip(c, ok if any(ok) else [True] * len(c)) if k]
        a = sum(p for p, _ in use) / len(use)
        out.append(({name: float(x) for name, x in zip(names, a)},
                    {"method": "logprob", "n_orderings": len(c), "n_kept": len(use),
                     "allowed_mass": sum(m for _, m in c) / len(c)}))
    return out


class LocalLM:
    """Next-token log-probs from a transformers causal LM, reusing the shared prefix's KV cache."""

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.device = next(model.parameters()).device

    def _last_logprobs(self, ids, cache=None):
        out = self.model(input_ids=torch.tensor([ids], device=self.device), past_key_values=cache,
                         use_cache=cache is not None, logits_to_keep=1)
        return out.logits[0, -1].float().log_softmax(-1).cpu()

    @torch.no_grad()
    def next_token_logprobs(self, shared: str, items: list[str], extra: list[list[int]] | None = None) -> list[torch.Tensor]:
        """After `shared + item`, then the token ids `extra[i]` if given."""
        enc = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        sids = enc(shared)
        prefix = self.model(input_ids=torch.tensor([sids], device=self.device), use_cache=True,
                            logits_to_keep=1).past_key_values
        out = []
        for i, item in enumerate(items):
            ids = enc(shared + item) + (extra[i] if extra else [])
            if ids[:len(sids)] == sids:
                out.append(self._last_logprobs(ids[len(sids):], copy.deepcopy(prefix)))
            else:   # the tokenizer merged across the boundary: run the whole prompt
                out.append(self._last_logprobs(ids))
        return out

    def scorer(self, n_labels: int):
        if numbered(n_labels):
            return digit_scorer(self.next_token_logprobs, self.tokenizer, n_labels)
        ids = label_ids(self.tokenizer, n_labels)
        return lambda shared, items: [label_probs(lp, ids) for lp in self.next_token_logprobs(shared, items)]
