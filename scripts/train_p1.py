"""P1 (docs/guide/09 §9.4): frozen backbone, cached query-token states, train the scorer head only.

    uv run python scripts/train_p1.py --features data/features/llm_response_coding_e4b.pt \
        --data data/processed/llm_response_coding.jsonl [--seed 0] [--standardize]

Every question of every row is trained, with the same per-type losses as adapter training (docs/guide/06 §6.7: Choice and
Score soft cross-entropy over log n, Score plus EMD², Noul BCE; averaged per question, then per row), on human soft
targets from `train`, early stopping on `val`. Temperatures per (type, option-count bucket) and the Noul bias are
fitted on `gold_calib` (docs/guide/10, decisivelm.calibrate). Writes calibrated predictions for gold_calib and gold_test
to data/predictions/ and the head + calibration to bundles/.
"""
import argparse, json, pathlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from decisivelm.calibrate import calibration_json, fit_calibration, predictions, question_logits, question_names, \
    write_predictions
from decisivelm.data.dataset import load_records
from decisivelm.head import Scorer
from decisivelm.losses import choice_loss, score_loss
from decisivelm.train import _target

REPO = pathlib.Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "gold_calib", "gold_test")


@dataclass
class Table:
    """One split's cached leaf states, flattened, with each row's questions as (type, first leaf, n, target)."""
    rows: list
    X: torch.Tensor
    off: list[int]
    questions: list[list[tuple]]


def table(rows, states) -> Table:
    X, off, questions = [], [0], []
    for r in rows:
        x = states[r.id].float()
        X.append(x)
        off.append(off[-1] + len(x))
        qs, start = [], 0
        for qid, q in r.questions.items():
            n = len(question_names(q))
            qs.append((q.type, start, n, _target(q, r.targets.get(qid))))
            start += n
        if start != len(x):
            raise SystemExit(f"{r.id}: {len(x)} cached states for {start} leaves")
        questions.append(qs)
    return Table(rows, torch.cat(X), off, questions)


def batch_logits(head, t: Table, idx) -> tuple[torch.Tensor, list[int]]:
    leaves = torch.cat([torch.arange(t.off[i], t.off[i + 1]) for i in idx])
    base = [0]
    for i in idx:
        base.append(base[-1] + t.off[i + 1] - t.off[i])
    return head(t.X[leaves]), base


def batch_loss(head, t: Table, idx) -> torch.Tensor:
    """Mean over rows of each row's mean question loss, questions grouped by (type, n) to vectorise."""
    logits, base = batch_logits(head, t, idx)
    groups: dict[tuple, tuple[list, list, list]] = {}
    for j, i in enumerate(idx):
        for qtype, start, n, target in t.questions[i]:
            if target is not None:
                g = groups.setdefault((qtype, n), ([], [], []))
                g[0].append(base[j] + start); g[1].append(target); g[2].append(j)
    total, count = torch.zeros(len(idx)), torch.zeros(len(idx))
    for (qtype, n), (starts, targets, js) in groups.items():
        lg = logits[torch.tensor(starts)[:, None] + torch.arange(n)]
        tg = torch.stack(targets).float()
        if qtype == "noul":
            loss = F.binary_cross_entropy_with_logits(lg.squeeze(-1), tg, reduction="none")
        else:
            loss = (choice_loss if qtype == "choice" else score_loss)(lg, tg)
        js = torch.tensor(js)
        total = total.index_add(0, js, loss)
        count = count.index_add(0, js, torch.ones(len(js)))
    has = count > 0
    return (total[has] / count[has]).mean()


@torch.no_grad()
def split_loss(head, t: Table, batch: int = 512) -> float:
    n = len(t.rows)
    return float(sum(batch_loss(head, t, list(range(s, min(s + batch, n)))) * (min(s + batch, n) - s)
                     for s in range(0, n, batch)) / n)


@torch.no_grad()
def split_questions(head, t: Table, batch: int = 512):
    out = []
    for s in range(0, len(t.rows), batch):
        idx = list(range(s, min(s + batch, len(t.rows))))
        logits, base = batch_logits(head, t, idx)
        for j, i in enumerate(idx):
            out += question_logits(t.rows[i], logits[base[j]:base[j + 1]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=pathlib.Path, required=True)
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64, help="rows per step")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--max-train", type=int, help="use only the first N train rows (subsampling large seeds)")
    ap.add_argument("--standardize", action="store_true",
                    help="z-score each feature dim on train before the head (Scorer.standardize; needed on 26B A4B)")
    ap.add_argument("--name", help="bundle and predictions name (default: from the features file and options)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    cache = torch.load(a.features, weights_only=True)
    rows = [r for r in load_records(a.data) if r.id in cache["states"]]
    by = {s: [r for r in rows if r.split == s] for s in SPLITS}
    by["train"] = by["train"][:a.max_train]
    T = {s: table(v, cache["states"]) for s, v in by.items()}
    meta = {"backbone": cache["backbone"], "dtype": cache.get("dtype", "bfloat16"),
            "fp32_router": cache.get("fp32_router", False), "readout": cache.get("readout", "final query token")}
    del cache
    print({s: len(v) for s, v in by.items()}, flush=True)

    head = Scorer(T["train"].X.shape[-1])
    if a.standardize:
        head.standardize(T["train"].X)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(500):
        head.train()
        for idx in torch.randperm(len(by["train"])).split(a.batch):
            loss = batch_loss(head, T["train"], idx.tolist())
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        val = split_loss(head, T["val"])
        if val < best - 1e-4:
            best, best_state, wait = val, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            wait += 1
            if wait == a.patience:
                break
    head.load_state_dict(best_state)
    head.eval()

    qs = {s: split_questions(head, T[s]) for s in ("gold_calib", "gold_test")}
    calib = fit_calibration(qs["gold_calib"])
    print(f"best val loss {best:.4f} (epoch {epoch - wait}); gold_test loss {split_loss(head, T['gold_test']):.4f}; "
          f"calibration {calibration_json(calib)}", flush=True)

    name = a.name or f"P1_{a.features.stem}{'_z' if a.standardize else ''}{f'_n{a.max_train}' if a.max_train else ''}_seed{a.seed}"
    out = REPO / "data" / "predictions" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(out, predictions(qs["gold_calib"] + qs["gold_test"], calib))
    bundle = REPO / "bundles" / name
    bundle.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), bundle / "scorer.pt")
    (bundle / "calibration.json").write_text(json.dumps(
        {**calibration_json(calib), **meta, "standardize": a.standardize, "data": str(a.data), "seed": a.seed, "max_train": a.max_train,
         "best_val_loss": best}, indent=1))
    print(f"wrote {out.relative_to(REPO)} and {bundle.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
