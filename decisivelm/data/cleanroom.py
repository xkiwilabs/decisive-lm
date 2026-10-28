"""Clean-room enforcement (docs/guide/16 §16.1).

- `whitelist_request`: from a Jev request/response log keep only the inputs
  (`state`, `questions`, `media`); every answer is dropped.
- `find_canaries`: files under a directory that contain any canary string.
- `scan_state`: flags states that contain Jev answer shapes, e.g. an earlier
  Jev answer pasted into a later request. Flagged items need a human decision.
"""
import pathlib
import re

INPUT_FIELDS = ("state", "questions", "media")
ANSWER_KEYS = {"probabilities", "noul", "confidence", "legend"}
_ANSWER_KEY_IN_TEXT = re.compile(r'["\']?(probabilities|noul|confidence|legend)["\']?\s*:\s*[{\[\d.]')


def whitelist_request(log: dict) -> dict:
    request = log.get("request", log)
    return {k: request[k] for k in INPUT_FIELDS if k in request}


def scan_state(state) -> bool:
    if isinstance(state, dict):
        return bool(ANSWER_KEYS & state.keys()) or any(scan_state(v) for v in state.values())
    if isinstance(state, list):
        return any(scan_state(v) for v in state)
    if isinstance(state, str):
        return bool(_ANSWER_KEY_IN_TEXT.search(state))
    return False


def find_canaries(root: pathlib.Path, canaries: list[str]) -> list[pathlib.Path]:
    hits = []
    for f in sorted(pathlib.Path(root).rglob("*")):
        if f.is_file():
            text = f.read_text(errors="ignore")
            if any(c in text for c in canaries):
                hits.append(f)
    return hits
