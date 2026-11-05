"""TextVQA (Singh et al. 2019, Facebook AI; docs/guide/14), questions that need reading the text in a photo, ten crowd
answers each. Held out as the frozen image suite for 1.0 (configs/holdout.yaml): evaluation only.

- the validation questions only (4,985 of the 5,000 kept, on 3,161 photos); the training questions are never
  converted
- media: the photo, extracted from train_val_images.zip to images/<image_id>.jpg (relative to the repository root);
  state: the question
- one Choice, the answer, built by decisivelm.data.converters.vqa from the ten answers: targets the shares of the
  answerers who gave each option
- splits by photo: 40% gold_calib, 60% gold_test
"""
import hashlib
import json
import pathlib
import re
import zipfile

from decisivelm.api_types import Media
from decisivelm.data.converters import vqa
from decisivelm.data.converters.crema_d import media_path
from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

SOURCE = "textvqa"
ARCHIVE = "train_val_images.zip"
ANNOTATIONS = "TextVQA_0.5.1_val.json"
IMAGE_ID = re.compile(r"[0-9a-f]{16}")


def _split(image: str) -> str:
    u = int(hashlib.sha256(f"{SOURCE}:{image}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return "gold_calib" if u < 0.4 else "gold_test"


def convert(source_dir: pathlib.Path) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    data = json.loads((source_dir / ANNOTATIONS).read_text(encoding="utf-8"))["data"]
    pool = sorted({vqa.normalise(a) for d in data for a in d["answers"]} - vqa.NOT_ANSWERS - {""})
    images = source_dir / "images"
    images.mkdir(exist_ok=True)
    out = []
    with zipfile.ZipFile(source_dir / ARCHIVE) as z:
        names = set(z.namelist())
        for d in data:
            image, member = d["image_id"], f"train_images/{d['image_id']}.jpg"
            built = vqa.choice(d["answers"], pool, f"{SOURCE}:{d['question_id']}")
            if built is None or not IMAGE_ID.fullmatch(image) or member not in names:
                continue
            path = images / f"{image}.jpg"
            if not path.exists():
                path.write_bytes(z.read(member))
            question, probabilities, n = built
            out.append(Record(
                id=f"{SOURCE}-{d['question_id']}",
                lineage=Lineage(source_dataset=SOURCE, source_id=str(d["question_id"]),
                                family_id=f"{SOURCE}-{image}"),
                license="CC-BY-4.0", tier="X", modality=["image", "text"], language="en",
                state=f"Question: {d['question']}", media=[Media(type="image", path=media_path(path))],
                questions={"answer": question},
                targets={"answer": QuestionTargets(human=HumanTarget(probabilities=probabilities, n_raters=n))},
                preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS,
                                      truncated=False),
                split=_split(image)))
    return out
