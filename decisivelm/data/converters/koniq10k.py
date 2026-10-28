"""Image example: KonIQ-10k (Hosu et al. 2020; docs/guide/14), the technical quality of 10,073 photos.

- media: the photo at 1024x768, the size raters saw (1024x768/<name>.jpg, relative to the repository root). The
  images come as a split zip (1024x768.zip + .z01), joined once with `zip -s 0` and unpacked here; only entries named
  like 1024x768/<digits>.jpg are written.
- one Score, the photo's technical quality on the study's five-point scale (bad … excellent)
- targets: the share of the photo's raters at each level (about 107 per photo; koniq10k_distributions_sets.csv)
- the authors' split: training → train, validation → val, test → gold_calib (40%) / gold_test (60%) by image name
"""
import hashlib
import pathlib
import re
import shutil
import subprocess
import zipfile

import pandas as pd

from decisivelm import CONFIGS
from decisivelm.api_types import Media
from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

SOURCE = "koniq10k"
QUESTION = {"type": "score",
            "instructions": "How good is the photo's technical quality (sharpness, noise, exposure, compression "
                            "artefacts), whatever its subject or how attractive it is?",
            "criteria": ["Bad", "Poor", "Fair", "Good", "Excellent"]}
IMAGE = re.compile(r"1024x768/\d+\.jpg")
CALIB_SHARE = 0.4


def unpack(source_dir: pathlib.Path) -> pathlib.Path:
    out = source_dir / "1024x768"
    if out.is_dir() and any(out.iterdir()):
        return out
    joined = source_dir / "1024x768_joined.zip"
    if not joined.exists():
        subprocess.run(["zip", "-s", "0", str(source_dir / "1024x768.zip"), "--out", str(joined)], check=True,
                       capture_output=True)
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(joined) as z:
        for name in z.namelist():
            if IMAGE.fullmatch(name):
                with z.open(name) as src, open(out / pathlib.PurePosixPath(name).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    joined.unlink()
    return out


def _media_path(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(CONFIGS.parent.resolve()))
    except ValueError:
        return str(p.resolve())


def _split(name: str, source_set: str) -> str:
    if source_set != "test":
        return {"training": "train", "validation": "val"}[source_set]
    u = int(hashlib.sha256(f"{SOURCE}:{name}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return "gold_calib" if u < CALIB_SHARE else "gold_test"


def convert(source_dir: pathlib.Path) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    images = unpack(source_dir)
    d = pd.read_csv(source_dir / "koniq10k_distributions_sets.csv")
    out = []
    for r in d.to_dict("records"):
        path = images / r["image_name"]
        if not path.exists():
            continue
        shares = [float(r[f"c{i}"]) for i in range(1, 6)]
        target = HumanTarget(probabilities={str(i): s / sum(shares) for i, s in enumerate(shares)},
                             n_raters=int(r["c_total"]))
        name = r["image_name"].removesuffix(".jpg")
        out.append(Record(
            id=f"{SOURCE}-{name}",
            lineage=Lineage(source_dataset=SOURCE, source_id=name, family_id=f"{SOURCE}-{name}"),
            license="NOASSERTION", tier="X", modality=["image", "text"], language="en", state="A photo.",
            media=[Media(type="image", path=_media_path(path))],
            questions={"quality": QUESTION}, targets={"quality": QuestionTargets(human=target)},
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS, truncated=False),
            split=_split(r["image_name"], r["set"])))
    return out
