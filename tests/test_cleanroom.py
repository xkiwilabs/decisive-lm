"""Clean-room enforcement (docs/guide/16): whitelist, canary, scanner."""
import json
import pathlib

import pytest

from decisivelm.data.cleanroom import find_canaries, scan_state, whitelist_request

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
REPO = pathlib.Path(__file__).parent.parent
CANARY_LOG = json.loads((FIXTURES / "jev_canary_log.json").read_text())


def canaries():
    """Every CANARY string in the fixtures, plus the log's unique answer numbers."""
    found = set()
    for f in FIXTURES.rglob("*"):
        if f.is_file():
            found |= {w.strip('",') for w in f.read_text().replace(",", " ").split() if "CANARY-" in w}
    return sorted(found | {"0.8123456789", "0.0987654321", "0.9012345679"})


def test_whitelist_keeps_only_inputs():
    kept = whitelist_request(CANARY_LOG)
    assert set(kept) == {"state", "questions"}
    assert kept["questions"]["intent"]["criteria"]["Refund"] == "Wants their money back"


def test_whitelisted_log_contains_no_canary(tmp_path):
    (tmp_path / "rows.jsonl").write_text(json.dumps(whitelist_request(CANARY_LOG)) + "\n")
    assert find_canaries(tmp_path, canaries()) == []


def test_find_canaries_catches_a_leak(tmp_path):
    (tmp_path / "rows.jsonl").write_text(json.dumps(CANARY_LOG) + "\n")
    assert find_canaries(tmp_path, canaries()) == [tmp_path / "rows.jsonl"]


def test_scanner_flags_a_jev_answer_pasted_into_a_state():
    pasted = json.dumps(CANARY_LOG["response"]["answers"])
    assert scan_state(f"Previous result: {pasted}")
    assert scan_state({"history": CANARY_LOG["response"]["answers"]})


@pytest.mark.parametrize("state", ["how do I turn on the night mode",
                                   {"order": 17, "status": "late"},
                                   "Our confidence in the supplier is low"])
def test_scanner_passes_ordinary_states(state):
    assert not scan_state(state)


def test_no_canary_under_data_processed():
    """CI fails if a canary string appears anywhere under data/processed/."""
    processed = REPO / "data" / "processed"
    if not processed.exists():
        pytest.skip("no data/processed in this checkout")
    assert find_canaries(processed, canaries()) == []
