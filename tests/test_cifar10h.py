"""CIFAR-10H: about 50 people's labels per CIFAR-10 test image, held out as a zero-shot image calibration check."""
import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from decisivelm.data.converters import cifar10h
from decisivelm.data.dataset import check_licence, load_registry


def _png(colour):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(cifar10h, "N", 3)
    labels = [3, 5, 0]
    pd.DataFrame({"img": [{"bytes": _png((i, 0, 0)), "path": None} for i in range(3)], "label": labels}) \
        .to_parquet(tmp_path / "cifar10_test.parquet")
    counts = np.zeros((3, 10), dtype=np.int64)
    counts[0, 3], counts[0, 5] = 30, 20     # cat 30, dog 20
    counts[1, 5] = 51
    counts[2, 0], counts[2, 8] = 45, 5      # airplane, ship
    np.save(tmp_path / "cifar10h-counts.npy", counts)
    return tmp_path


def test_targets_are_the_raters_shares_and_images_are_written(source):
    rows = {r.lineage.source_id: r for r in cifar10h.convert(source)}
    t = rows["00000"].targets["label"].human
    assert t.n_raters == 50 and t.probabilities["cat"] == pytest.approx(0.6) and t.probabilities["dog"] == 0.4
    assert rows["00001"].media[0].path.endswith("test_png/00001.png") and (source / "test_png" / "00001.png").exists()
    assert {r.split for r in rows.values()} <= {"gold_calib", "gold_test"}


def test_images_out_of_order_with_the_counts_are_refused(source):
    pd.DataFrame({"img": [{"bytes": _png((0, 0, 0)), "path": None}] * 3, "label": [9, 9, 9]}) \
        .to_parquet(source / "cifar10_test.parquet")
    with pytest.raises(ValueError, match="wrong order"):
        cifar10h.convert(source)


def test_rows_match_the_registry_and_are_held_out(source):
    registry = load_registry()
    for r in cifar10h.convert(source):
        check_licence(r, registry)
    assert "cifar10h" in open("configs/holdout.yaml").read()
