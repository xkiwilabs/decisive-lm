import pandas as pd

from decisivelm.data.converters import ledgar

README = """---
dataset_info:
- config_name: ecthr_a
  features:
  - name: text
    dtype: string
- config_name: ledgar
  features:
  - name: text
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': Adjustments
          '1': Governing Laws
          '2': Notices
---
# LexGLUE
"""


def test_ledgar_names_targets_and_splits(tmp_path):
    (tmp_path / "README.md").write_text(README)
    for split, n in (("train", 5), ("validation", 2), ("test", 20)):
        pd.DataFrame({"text": [f"{split} clause {i}" for i in range(n)], "label": [i % 3 for i in range(n)]}
                     ).to_parquet(tmp_path / f"{split}.parquet")
    assert ledgar.label_names(tmp_path / "README.md") == ["Adjustments", "Governing Laws", "Notices"]
    rows = ledgar.convert(tmp_path, n_train=None, n_val=None, n_test=None)
    by = {r.lineage.source_id: r for r in rows}
    assert {r.split for r in rows if r.lineage.source_id.startswith("train")} == {"train"}
    assert {r.split for r in rows if r.lineage.source_id.startswith("validation")} == {"val"}
    assert {r.split for r in rows if r.lineage.source_id.startswith("test")} == {"gold_calib", "gold_test"}
    assert by["train-00001"].targets["provision"].human.probabilities == {
        "Adjustments": 0.0, "Governing Laws": 1.0, "Notices": 0.0}
    assert list(by["train-00001"].questions["provision"].criteria) == ["Adjustments", "Governing Laws", "Notices"]
