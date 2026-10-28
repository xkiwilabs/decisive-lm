"""Image example: THINGSplus (Hebart et al. 2019; Stoinski, Perkuhn & Hebart 2024; docs/guide/14), whether an object is
"something that lives", for 1,854 everyday objects. People split on many of them: of 38 raters, 28 agreed that an
apple lives and 7 disagreed, while a tree and a rock are near unanimous. That disagreement is what the confidence
should follow.

- media: the object's public-domain photo (object_images_CC0/<uniqueID>.jpg, relative to the repository root). The
  photos come as one zip; only entries named like object_images_CC0/<id>.jpg are written. The password-protected
  THINGS image set (research use only) is never used.
- state: the object's name, as raters saw it with a photo
- one Score, the study's 7-point agreement with "something that lives" (1 strongly disagree … 7 strongly agree; the
  study labelled only the ends, the middle levels are named here)
- targets: the share of the object's raters at each level (object-properties_trial-wise_thingsplus.tsv, about 37 per
  object). Only the object, its name and the rating are read; never the raters' age or gender.
- splits by object, by a hash of its id: 60% train, 10% val, 10% gold_calib, 20% gold_test
"""
import collections
import hashlib
import io
import pathlib
import re
import shutil
import zipfile

import pandas as pd

from decisivelm import CONFIGS
from decisivelm.api_types import Media
from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

SOURCE = "things_living"
QUESTION = {"type": "score",
            "instructions": 'How much do you agree that the object in the photo is "something that lives"?',
            "criteria": ["Strongly disagree", "Disagree", "Somewhat disagree", "Neither agree nor disagree",
                         "Somewhat agree", "Agree", "Strongly agree"]}
IMAGE = re.compile(r"object_images_CC0/([a-z0-9_]+)\.jpg")
SPLITS = [(0.6, "train"), (0.7, "val"), (0.8, "gold_calib"), (1.0, "gold_test")]


def unpack(source_dir: pathlib.Path) -> pathlib.Path:
    out = source_dir / "object_images_CC0"
    if out.is_dir() and any(out.iterdir()):
        return out
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(source_dir / "images_THINGSplus-CC0.zip") as z:
        for name in z.namelist():
            if (m := IMAGE.fullmatch(name)):
                with z.open(name) as src, open(out / f"{m.group(1)}.jpg", "wb") as dst:
                    shutil.copyfileobj(src, dst)
    return out


def ratings(path: pathlib.Path) -> pd.DataFrame:
    """The trial-wise table, one rater's ratings of one object per line. Each line of the file is wrapped in quotes as
    a whole; they are stripped before the tab-separated fields are read."""
    text = "\n".join(line.strip().strip('"') for line in open(path, encoding="utf-8"))
    d = pd.read_csv(io.StringIO(text), sep="\t", usecols=["Word", "uniqueID", "lives"])
    return d.dropna(subset=["lives"])


def _media_path(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(CONFIGS.parent.resolve()))
    except ValueError:
        return str(p.resolve())


def _split(uid: str) -> str:
    u = int(hashlib.sha256(f"{SOURCE}:{uid}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return next(s for t, s in SPLITS if u < t)


def convert(source_dir: pathlib.Path) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    images = unpack(source_dir)
    d = ratings(source_dir / "object-properties_trial-wise_thingsplus.tsv")
    out = []
    for uid, g in sorted(d.groupby("uniqueID")):
        path = images / f"{uid}.jpg"
        levels = [int(x) - 1 for x in g["lives"] if 1 <= int(x) <= 7]
        if len(levels) < 3 or not path.exists():
            continue
        counts = collections.Counter(levels)
        out.append(Record(
            id=f"{SOURCE}-{uid}",
            lineage=Lineage(source_dataset=SOURCE, source_id=uid, family_id=f"{SOURCE}-{uid}"),
            license="CC-BY-4.0", tier="R", modality=["image", "text"], language="en",
            state=f"The object: {g['Word'].iloc[0].strip()}.", media=[Media(type="image", path=_media_path(path))],
            questions={"lives": QUESTION},
            targets={"lives": QuestionTargets(human=HumanTarget(
                probabilities={str(i): counts[i] / len(levels) for i in range(7)}, n_raters=len(levels)))},
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS, truncated=False),
            split=_split(uid)))
    return out
