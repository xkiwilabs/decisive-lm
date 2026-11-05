"""ODAQ (Torcoli et al. 2024, Fraunhofer IIS and Netflix; docs/guide/14), audio quality judged by 26 expert listeners,
held out as the frozen audio suite for 1.0.

- media: two clips, as the listeners heard them in the MUSHRA test: the original excerpt (reference.wav), then a
  processed version (a codec, a band limit, source separation, or the hidden reference itself); both extracted from
  the archive to listening_test/<item>/<file> (relative to the repository root)
- one Score, how good the processed version sounds against the original, on MUSHRA's five labelled bands
  (0-20 bad … 80-100 excellent)
- targets: the shares of the 26 listeners whose 0-100 score falls in each band (ODAQ_results.csv, one row per
  listener and condition)
- only excerpts whose audio is CC BY or CC0 (_detailed_license.csv, read for each item's source file); the CC BY-NC
  excerpts are left out. The scores themselves are CC BY (_license_disclaimer.txt)
- splits by excerpt, so all conditions of one excerpt share a split: 40% gold_calib, 60% gold_test
"""
import csv
import hashlib
import io
import pathlib
import zipfile
from collections import Counter

from decisivelm.api_types import Media
from decisivelm.data.converters.crema_d import media_path
from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

SOURCE = "odaq"
ARCHIVE = "ODAQ.zip"
TEST = "ODAQ/ODAQ_listening_test/"
BANDS = ["Bad: the processed version is badly degraded", "Poor: clearly degraded",
         "Fair: noticeably degraded", "Good: slightly degraded", "Excellent: no audible difference from the original"]
QUESTION = {"type": "score", "criteria": BANDS,
            "instructions": "The first recording is an original; the second is a processed version of it. How good "
                            "does the processed version sound compared with the original?"}
FREE = ("creativecommons.org/licenses/by/", "creativecommons.org/publicdomain/zero/")
METHODS = ("DE_", "LP_", "PE_", "SH_", "TM_", "UN_")


def _split(item: str) -> str:
    u = int(hashlib.sha256(f"{SOURCE}:{item}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return "gold_calib" if u < 0.4 else "gold_test"


def licences(text: str) -> dict[str, str]:
    """Source file name -> licence URL, from _detailed_license.csv (semicolon-separated; its own commas are not
    delimiters)."""
    out = {}
    for line in text.splitlines()[1:]:
        f = line.split(";")
        if len(f) > 5 and f[0].endswith(".wav"):
            out[f[0]] = (f[5].split() or [""])[0]
    return out


def source_file(item: str) -> str:
    return (item[3:] if item.startswith(METHODS) else item) + ".wav"


def band(score: float) -> int:
    return min(4, int(score // 20))


def convert(source_dir: pathlib.Path) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    with zipfile.ZipFile(source_dir / ARCHIVE) as z:
        lic = licences(z.read("ODAQ/_detailed_license.csv").decode("utf-8", errors="replace"))
        votes: dict[tuple[str, str], Counter] = {}
        for r in csv.DictReader(io.TextIOWrapper(z.open(TEST + "ODAQ_results.csv"), encoding="utf-8")):
            votes.setdefault((r["item"], r["process"]), Counter())[band(float(r["score"]))] += 1
        names = set(z.namelist())

        def extract(item: str, process: str) -> pathlib.Path | None:
            member = f"{TEST}{item}/{process}.wav"
            if member not in names or "/" in process or "/" in item or ".." in item + process:
                return None
            out = source_dir / "listening_test" / item / f"{process}.wav"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(member))
            return out

        rows = []
        for (item, process), c in sorted(votes.items()):
            if not lic.get(source_file(item), "").split("://")[-1].startswith(FREE):
                continue
            ref, clip = extract(item, "reference"), extract(item, process)   # the hidden reference is reference.wav
            if ref is None or clip is None:
                continue
            n = sum(c.values())
            rows.append(Record(
                id=f"{SOURCE}-{item}-{process}",
                lineage=Lineage(source_dataset=SOURCE, source_id=f"{item}/{process}", family_id=f"{SOURCE}-{item}"),
                license="NOASSERTION", tier="X", modality=["audio", "text"], language="en",
                state="Two recordings of the same excerpt.",
                media=[Media(type="audio", path=media_path(ref)), Media(type="audio", path=media_path(clip))],
                questions={"quality": QUESTION},
                targets={"quality": QuestionTargets(human=HumanTarget(
                    probabilities={str(i): c[i] / n for i in range(5)}, n_raters=n))},
                preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS,
                                      truncated=False),
                split=_split(item)))
    return rows
