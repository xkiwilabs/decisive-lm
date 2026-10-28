"""Stanford Politeness Corpus (Stack Exchange): a two-sided 5-level Score from 5 raters' 1–25 slider ratings."""
import json
import os
import pickle
import zipfile

import numpy as np
import pytest

from decisivelm.data.converters import politeness
from decisivelm.data.dataset import check_licence, load_registry


class _Boom:
    def __reduce__(self):
        return os.system, ("true",)


def _write(tmp_path, ratings_pickle: bytes):
    texts = ["Could you please share the code?", "Why would you do that?", "Could you please share the code?"]
    with zipfile.ZipFile(tmp_path / f"{politeness.CORPUS}.zip", "w") as z:
        z.writestr(f"{politeness.CORPUS}/utterances.jsonl",
                   "".join(json.dumps({"id": str(i), "text": t}) + "\n" for i, t in enumerate(texts)))
        z.writestr(f"{politeness.CORPUS}/Annotations-bin.p", ratings_pickle)


@pytest.fixture
def source(tmp_path):
    ratings = [{f"w{j}": np.int64(v) for j, v in enumerate(vs)}
               for vs in ([25, 20, 13, 14, 19], [1, 6, 7, 12, 13], [13, 13, 13, 13, 13])]
    _write(tmp_path, pickle.dumps(ratings, protocol=3))
    return tmp_path


def test_ratings_are_binned_symmetrically_about_the_middle():
    assert [politeness.level(v) for v in (1, 6, 7, 12, 13, 14, 19, 20, 25)] == [0, 0, 1, 1, 2, 3, 3, 4, 4]


def test_target_is_the_five_raters_shares(source):
    rows = {r.lineage.source_id: r for r in politeness.convert(source)}
    t = rows["0"].targets["politeness"].human
    assert t.probabilities == {"0": 0.0, "1": 0.0, "2": 0.2, "3": 0.4, "4": 0.4} and t.n_raters == 5
    assert rows["1"].targets["politeness"].human.probabilities == {"0": 0.4, "1": 0.4, "2": 0.2, "3": 0.0, "4": 0.0}


def test_a_repeated_request_keeps_one_split_and_family(source):
    rows = {r.lineage.source_id: r for r in politeness.convert(source)}
    assert rows["0"].split == rows["2"].split and rows["0"].lineage.family_id == rows["2"].lineage.family_id


@pytest.mark.parametrize("payload", [[_Boom()] * 3, [{"w0": np.float64(1.5)}] * 3])
def test_the_ratings_pickle_admits_nothing_but_numpy_integers(tmp_path, payload):
    _write(tmp_path, pickle.dumps(payload, protocol=3))
    with pytest.raises(pickle.UnpicklingError, match="refused"):
        politeness.convert(tmp_path)


def test_rows_match_the_registry(source):
    registry = load_registry()
    for r in politeness.convert(source):
        check_licence(r, registry)
