"""TF-IDF + logistic regression baseline for every question of any dataset, on soft human targets.

    uv run --with scikit-learn python scripts/baseline_tfidf.py --data data/processed/helpsteer2.jsonl

scikit-learn is not a project dependency: run it in uv's overlay. One word 1–2-gram TF-IDF over the rendered state
(decisivelm.render.as_text, the text the model sees), fitted on `train`; per question, a logistic regression on the
human shares (each row expanded into one example per option or level, weighted by its share; Noul as yes/no
weighted by P(yes)). C is picked by soft loss on `val`, then temperatures are fitted on `gold_calib` exactly as for
the model (decisivelm.calibrate). Writes data/predictions/tfidf_<stem>.jsonl in the format eval_dataset.py reads.
"""
import argparse, math, pathlib

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from decisivelm.calibrate import QuestionLogits, fit_calibration, predictions, question_names, write_predictions
from decisivelm.data.dataset import load_records
from decisivelm.render import as_text
from decisivelm.train import _target

REPO = pathlib.Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "gold_calib", "gold_test")


def expand(X, T):
    """Rows x soft targets [N, k] -> one weighted example per (row, class) with positive share."""
    i, c = np.nonzero(T > 0)
    return X[i], c, T[i, c]


def fit(X, T, C):
    Xe, y, w = expand(X, T)
    return LogisticRegression(C=C, max_iter=2000).fit(Xe, y, sample_weight=w)


def log_probs(model, X, k):
    """[N, k] log-probabilities, classes never seen in train get a floor."""
    out = np.full((X.shape[0], k), math.log(1e-6))
    out[:, model.classes_] = model.predict_log_proba(X)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--grid", type=float, nargs="+", default=[0.3, 1, 3, 10, 30, 100])
    a = ap.parse_args()

    rows = load_records(a.data)
    by = {s: [r for r in rows if r.split == s] for s in SPLITS}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    vec.fit([as_text(r.state) for r in by["train"]])
    X = {s: vec.transform([as_text(r.state) for r in v]) for s, v in by.items()}

    qs = {s: [] for s in ("gold_calib", "gold_test")}
    for qid, q0 in rows[0].questions.items():
        names = question_names(q0)
        k = 2 if q0.type == "noul" else len(names)

        def targets(split):
            t = [_target(r.questions[qid], r.targets.get(qid)) for r in by[split]]
            t = np.stack([x.numpy() for x in t]).astype(float)
            return np.stack([1 - t, t], 1) if q0.type == "noul" else t

        T = {s: targets(s) for s in SPLITS}
        best = None
        for C in a.grid:
            m = fit(X["train"], T["train"], C)
            loss = -(T["val"] * log_probs(m, X["val"], k)).sum(1).mean()
            if best is None or loss < best[0]:
                best = (loss, C, m)
        print(f"{qid}: C={best[1]} val soft loss {best[0]:.4f}", flush=True)
        for s in qs:
            lp = log_probs(best[2], X[s], k)
            for r, l, t in zip(by[s], lp, T[s]):
                logits = torch.tensor([l[1] - l[0]] if q0.type == "noul" else l, dtype=torch.float32)
                target = torch.tensor(t[1] if q0.type == "noul" else t, dtype=torch.float32)
                qs[s].append(QuestionLogits(r.id, s, qid, q0.type, names, logits, target))

    calib = fit_calibration(qs["gold_calib"])
    out = REPO / "data" / "predictions" / f"tfidf_{a.data.stem}.jsonl"
    write_predictions(out, predictions(qs["gold_calib"] + qs["gold_test"], calib))
    print(f"calibration {calib.temperatures} b_noul {calib.b_noul:.3f}; wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
