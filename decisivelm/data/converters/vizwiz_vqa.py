"""VizWiz-VQA (Gurari et al. 2018; docs/guide/14), photos taken by blind people on their phones, each with the spoken
question they asked about it and ten crowd answers.

- media: the photo, extracted from train.zip or val.zip to images/<file> (relative to the repository root); state: the
  question. The photos are never viewed by us and never re-hosted
- a Noul, whether the photo shows enough to answer the question: the share of the ten answerers who gave an answer
  rather than "unanswerable"
- a Choice, the answer, built by decisivelm.data.converters.vqa from the answers given (two options for yes/no
  questions, else four), when at least three answerers gave one of its options (`MIN_ANSWERS`)
- splits by photo: the train annotations → train, a seeded 5% of them → val; the val annotations → gold_calib (40%)
  and gold_test (60%). The test split has no public answers and is not fetched
"""
import hashlib
import json
import pathlib
import zipfile

from decisivelm.api_types import Media
from decisivelm.data.converters import vqa
from decisivelm.data.converters.crema_d import media_path
from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

SOURCE = "vizwiz_vqa"
MIN_ANSWERS = 3
ANSWERABLE = {"type": "noul", "instructions": "The photo shows enough to answer the question"}


def _u(key: str) -> float:
    return int(hashlib.sha256(f"{SOURCE}:{key}".encode()).hexdigest()[:15], 16) / 16 ** 15


def split(part: str, image: str) -> str:
    if part == "train":
        return "val" if _u(f"val:{image}") < 0.05 else "train"
    return "gold_calib" if _u(f"gold:{image}") < 0.4 else "gold_test"


def convert(source_dir: pathlib.Path) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    images = source_dir / "images"
    images.mkdir(exist_ok=True)
    with zipfile.ZipFile(source_dir / "Annotations.zip") as z:
        annotations = {part: json.loads(z.read(f"{part}.json")) for part in ("train", "val")}
    out = []
    for part, data in annotations.items():
        pool = sorted({vqa.normalise(a["answer"]) for d in data for a in d["answers"]} - vqa.NOT_ANSWERS - {""})
        with zipfile.ZipFile(source_dir / f"{part}.zip") as z:
            members = {pathlib.PurePosixPath(n).name: n for n in z.namelist() if n.endswith(".jpg")}
            for d in data:
                name = d["image"]
                if name not in members or pathlib.PurePosixPath(name).name != name:
                    continue
                path = images / name                       # a bare file name, so nothing is written outside images/
                if not path.exists():
                    path.write_bytes(z.read(members[name]))
                answers = [a["answer"] for a in d["answers"]]
                answered = sum(vqa.normalise(a) not in vqa.NOT_ANSWERS for a in answers)
                questions = {"answerable": ANSWERABLE}
                targets = {"answerable": QuestionTargets(human=HumanTarget(noul=answered / len(answers),
                                                                           n_raters=len(answers)))}
                built = vqa.choice(answers, pool, f"{SOURCE}:{name}", yes_no=d["answer_type"] == "yes/no")
                if built is not None and built[2] >= MIN_ANSWERS:
                    question, probabilities, n = built
                    questions["answer"] = question
                    targets["answer"] = QuestionTargets(human=HumanTarget(probabilities=probabilities, n_raters=n))
                out.append(Record(
                    id=f"{SOURCE}-{pathlib.PurePosixPath(name).stem}",
                    lineage=Lineage(source_dataset=SOURCE, source_id=name, family_id=f"{SOURCE}-{name}"),
                    license="CC-BY-4.0", tier="R", modality=["image", "text"], language="en",
                    state=f"Question: {d['question']}", media=[Media(type="image", path=media_path(path))],
                    questions=questions, targets=targets,
                    preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS,
                                          truncated=False),
                    split=split(part, name)))
    return out
