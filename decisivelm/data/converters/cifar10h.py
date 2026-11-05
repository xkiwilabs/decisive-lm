"""Image calibration check: CIFAR-10H (Peterson, Battleday, Griffiths & Russakovsky 2019; docs/guide/14), about 50
people's labels for each of the 10,000 CIFAR-10 test images. Held out from every training mix (configs/holdout.yaml):
it tests whether a model's confidence on images follows how people split, zero-shot.

- media: the 32 × 32 test image, decoded from the uoft-cs/cifar10 test parquet and re-saved as PNG
  (test_png/<index>.png, relative to the repository root). Its order is checked against the counts: the parquet's
  label must be the raters' most common answer on almost every image.
- state: what raters were told, that it is a small photo
- one Choice over the ten CIFAR-10 classes, with the dataset's own boundaries between car and truck
- targets: the share of the image's raters choosing each class (cifar10h-counts.npy, read without pickle)
- splits: 40% gold_calib, 60% gold_test by a hash of the image index; no training rows
"""
import hashlib
import io
import pathlib

import numpy as np
import pandas as pd
from PIL import Image

from decisivelm import CONFIGS
from decisivelm.api_types import Media
from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

SOURCE = "cifar10h"
CLASSES = {
    "airplane": "Airplane",
    "automobile": "Automobile: a car, such as a sedan or SUV, not a pickup and not a large truck",
    "bird": "Bird",
    "cat": "Cat",
    "deer": "Deer",
    "dog": "Dog",
    "frog": "Frog",
    "horse": "Horse",
    "ship": "Ship or boat",
    "truck": "Truck: a large truck, not a pickup and not a car",
}
QUESTION = {"type": "choice", "instructions": "What is the main object in the photo?", "criteria": CLASSES}
N, SIDE = 10_000, 32
CALIB_SHARE = 0.4


def unpack(source_dir: pathlib.Path) -> pathlib.Path:
    out = source_dir / "test_png"
    if out.is_dir() and len(list(out.glob("*.png"))) == N:
        return out
    d = pd.read_parquet(source_dir / "cifar10_test.parquet")
    if len(d) != N:
        raise ValueError(f"cifar10_test.parquet has {len(d)} rows, not {N}")
    out.mkdir(exist_ok=True)
    for i, img in enumerate(d["img"]):
        im = Image.open(io.BytesIO(img["bytes"]))
        if im.size != (SIDE, SIDE):
            raise ValueError(f"test image {i} is {im.size}, not {SIDE} × {SIDE}")
        im.convert("RGB").save(out / f"{i:05d}.png")
    np.save(source_dir / "test_labels.npy", d["label"].to_numpy())
    return out


def _media_path(p: pathlib.Path) -> str:
    try:
        return str(p.resolve().relative_to(CONFIGS.parent.resolve()))
    except ValueError:
        return str(p.resolve())


def _split(i: int) -> str:
    u = int(hashlib.sha256(f"{SOURCE}:{i}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return "gold_calib" if u < CALIB_SHARE else "gold_test"


def convert(source_dir: pathlib.Path) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    images = unpack(source_dir)
    counts = np.load(source_dir / "cifar10h-counts.npy", allow_pickle=False)
    if counts.shape != (N, len(CLASSES)):
        raise ValueError(f"cifar10h-counts.npy has shape {counts.shape}, not {(N, len(CLASSES))}")
    match = (counts.argmax(1) == np.load(source_dir / "test_labels.npy", allow_pickle=False)).mean()
    if match < 0.9:
        raise ValueError(f"the images' labels match the raters' most common answer on only {match:.0%}: wrong order?")
    out = []
    for i, c in enumerate(counts):
        n = int(c.sum())
        target = HumanTarget(probabilities={k: float(x) / n for k, x in zip(CLASSES, c)}, n_raters=n)
        out.append(Record(
            id=f"{SOURCE}-{i:05d}",
            lineage=Lineage(source_dataset=SOURCE, source_id=f"{i:05d}", family_id=f"{SOURCE}-{i:05d}"),
            license="CC-BY-NC-SA-4.0", tier="X", modality=["image", "text"], language="en",
            state="A small photo (32 × 32 pixels).",
            media=[Media(type="image", path=_media_path(images / f"{i:05d}.png"))],
            questions={"label": QUESTION}, targets={"label": QuestionTargets(human=target)},
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS, truncated=False),
            split=_split(i)))
    return out
