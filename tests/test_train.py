"""Adapter training loop (docs/guide/09 §9.5) on a tiny random Gemma 4."""
import random

import torch

from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.head import Scorer
from decisivelm.infer import QUERY_IDS, DecisiveLM
from decisivelm.lora import apply_lora
from decisivelm.model import backbone_profile
from decisivelm.train import make_items, micro_batches, train_adapter
from tiny_gemma import tiny_decoder, tokenizer

DEPT = {"type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "Payment issues", "technical": "Bugs or outages", "sales": "Pricing"}}
URGENT = {"type": "noul", "instructions": "The message is urgent"}


def record(i, state, dept, urgent=None):
    targets = {"d": QuestionTargets(human=HumanTarget(probabilities=dept))}
    if urgent is not None:
        targets["u"] = QuestionTargets(human=HumanTarget(noul=urgent))
    return Record(id=f"r{i}", lineage=Lineage(source_dataset="toy", source_id=str(i), family_id="toy"),
                  license="CC0-1.0", tier="R", modality=["text"], language="en", state=state,
                  questions={"u": URGENT, "d": DEPT}, targets=targets,
                  preprocess=Preprocess(render_version=1, max_state_tokens=8192, truncated=False), split="train")


ROWS = [record(0, "I was charged twice for my plan", {"billing": 1.0, "technical": 0.0, "sales": 0.0}, 0.2),
        record(1, "The app crashes on login since the update", {"billing": 0.0, "technical": 1.0, "sales": 0.0}, 0.9),
        record(2, "How much is the enterprise tier?", {"billing": 0.0, "technical": 0.0, "sales": 1.0}),
        record(3, "Refund the duplicate payment please", {"billing": 0.75, "technical": 0.0, "sales": 0.25}, 0.5)]


def dlm():
    m = tiny_decoder("dense", "sdpa", vocab=len(tokenizer()))
    apply_lora(m, backbone_profile("e4b")["lora_target_regex"], rank=4, alpha=8, query_ids=list(QUERY_IDS.values()))
    torch.manual_seed(1)
    return DecisiveLM(m, tokenizer(), Scorer(m.config.hidden_size))


def test_items_follow_the_request_question_order():
    item = make_items(dlm(), ROWS[:1])[0]
    assert [t for t, _ in item.questions] == ["noul", "choice"]
    assert item.questions[0][1].item() == torch.tensor(0.2).item()
    assert item.questions[1][1].tolist() == [1.0, 0.0, 0.0]


def test_missing_targets_are_none():
    item = make_items(dlm(), ROWS[2:3])[0]
    assert item.questions[0][1] is None


def test_micro_batches_respect_the_token_budget():
    items = make_items(dlm(), ROWS)
    budget = max(i.n_tokens for i in items) + 10
    batches = micro_batches(items, budget, random.Random(0))
    assert sorted(i.id for b in batches for i in b) == sorted(r.id for r in ROWS)
    assert all(sum(i.n_tokens for i in b) <= budget for b in batches)


def test_training_reduces_validation_loss():
    model = dlm()
    items = make_items(model, ROWS)
    result = train_adapter(model, items, items, epochs=15, rows_per_step=2, max_tokens=4096,
                           lr_lora=5e-3, lr_head=5e-3, eval_every=2, patience=100, seed=0, checkpointing=False)
    assert result["history"][-1]["val_loss"] < result["history"][0]["val_loss"] * 0.7
    assert set(result["scorer"]) == set(model.scorer.state_dict())
    assert result["adapter"] and all(k.endswith(("lora_A", "lora_B", "query_rows.delta")) for k in result["adapter"])


def test_val_metrics_and_selection_criteria():
    from decisivelm.train import val_metrics
    model = dlm()
    items = make_items(model, ROWS)
    m = val_metrics(model, items, 4096)
    assert set(m) == {"loss", "cal_loss", "acc"}
    assert m["cal_loss"] <= m["loss"] + 1e-6          # T = 1 is on the grid
    assert 0 <= m["acc"] <= 1
    result = train_adapter(model, items, items, epochs=4, rows_per_step=2, max_tokens=4096, lr_lora=5e-3,
                           lr_head=5e-3, eval_every=2, patience=100, seed=0, checkpointing=False, select="acc")
    assert set(result["best"]) == {"loss", "cal_loss", "acc"}
    assert result["best_step"] == result["best"]["acc"]["step"]
    assert result["best_val"] == max(h["val_acc"] for h in result["history"]) or result["best_step"] == 0


def test_training_skips_a_micro_batch_with_no_targets():
    # A row whose questions all lack targets, alone in its micro-batch (here every row is: the budget fits one), must
    # not stop training (General v5c_nts crashed on one at step ~3,700).
    model = dlm()
    empty = ROWS[0].model_copy(update={"id": "r_empty", "targets": {}})
    items = make_items(model, ROWS + [empty])
    budget = max(i.n_tokens for i in items)
    result = train_adapter(model, items, items, epochs=2, rows_per_step=2, max_tokens=budget, lr_lora=5e-3,
                           lr_head=5e-3, eval_every=2, patience=100, seed=0, checkpointing=False)
    assert len(result["history"]) > 1


def test_harden_takes_the_majority_and_breaks_ties_by_a_seeded_pick():
    from decisivelm.train import harden
    h = harden(ROWS[3])
    assert h.targets["d"].human.probabilities == {"billing": 1.0, "technical": 0.0, "sales": 0.0}
    assert h.targets["u"].human.noul in (0.0, 1.0)                    # exactly 0.5: a coin flip
    assert harden(ROWS[0]).targets["u"].human.noul == 0.0 and harden(ROWS[1]).targets["u"].human.noul == 1.0
    tie = record(9, "x", {"billing": 0.4, "technical": 0.4, "sales": 0.2})
    picks = harden(tie).targets["d"].human.probabilities
    assert sorted(picks.values()) == [0.0, 0.0, 1.0] and picks["sales"] == 0.0
    assert harden(tie) == harden(tie)                                    # reproducible
    assert ROWS[3].targets["d"].human.probabilities["billing"] == 0.75   # the input is not changed
