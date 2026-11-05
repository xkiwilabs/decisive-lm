"""Emozionalmente (Catania, Politecnico di Milano; docs/guide/14), the emotion listeners hear in acted Italian speech.

- media: the clip's WAV, extracted from the archive to audio/<file_name> (relative to the repository root); state: the
  Italian sentence the actor says, one of 18 shared by every emotion, so the emotion must come from the voice
- one Choice, which emotion the speaker's voice expresses, over the seven the study used (the Big Six plus neutrality)
- targets: the shares of the 5 listeners per clip who guessed the emotion (metadata/evaluations.csv, one row per
  listener). The emotion the actor meant is not the target; listeners agreed with it 66% of the time, and that
  disagreement is what the confidence should follow. metadata/users.csv (demographics) is never opened.
- splits by actor (431), so no voice is in two splits: 60% train, 10% val, 10% gold_calib, 20% gold_test
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

SOURCE = "emozionalmente"
ARCHIVE = "emozionalmente_dataset.zip"
ROOT = "emozionalmente_dataset/"
EMOTIONS = {"anger": "Angry, annoyed or irritated", "disgust": "Disgusted or repelled",
            "fear": "Afraid, anxious or nervous", "joy": "Happy, joyful or cheerful",
            "neutrality": "No particular emotion: a calm, flat delivery", "sadness": "Sad, unhappy or downcast",
            "surprise": "Surprised or astonished"}
QUESTION = {"type": "choice", "instructions": "Which emotion does the speaker's voice express?", "criteria": EMOTIONS}
SPLITS = [(0.6, "train"), (0.7, "val"), (0.8, "gold_calib"), (1.0, "gold_test")]


def _split(actor: str) -> str:
    u = int(hashlib.sha256(f"{SOURCE}:{actor}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return next(s for t, s in SPLITS if u < t)


def _csv(z: zipfile.ZipFile, name: str) -> list[dict]:
    return list(csv.DictReader(io.TextIOWrapper(z.open(ROOT + "metadata/" + name), encoding="utf-8")))


def convert(source_dir: pathlib.Path) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    audio = source_dir / "audio"
    audio.mkdir(exist_ok=True)
    with zipfile.ZipFile(source_dir / ARCHIVE) as z:
        samples = {r["file_name"]: r for r in _csv(z, "samples.csv")}
        votes: dict[str, Counter] = {}
        for r in _csv(z, "evaluations.csv"):
            if r["emotion_recognized"] in EMOTIONS:
                votes.setdefault(r["file_name"], Counter())[r["emotion_recognized"]] += 1
        members = {pathlib.PurePosixPath(n).name: n for n in z.namelist()
                   if n.startswith(ROOT + "audio/") and n.endswith(".wav")}
        out = []
        for name in sorted(votes):
            s, n = samples.get(name), sum(votes[name].values())
            if s is None or n < 2 or name not in members or pathlib.PurePosixPath(name).name != name:
                continue
            wav = audio / name                       # a bare file name, so nothing is written outside audio/
            if not wav.exists():
                wav.write_bytes(z.read(members[name]))
            target = HumanTarget(probabilities={e: votes[name][e] / n for e in EMOTIONS}, n_raters=n)
            out.append(Record(
                id=f"{SOURCE}-{pathlib.PurePosixPath(name).stem}",
                lineage=Lineage(source_dataset=SOURCE, source_id=name, family_id=f"{SOURCE}-actor-{s['actor']}"),
                license="CC-BY-4.0", tier="R", modality=["audio", "text"], language="it",
                state=f'An actor says, in Italian: "{s["sentence"]}"', media=[Media(type="audio", path=media_path(wav))],
                questions={"emotion": QUESTION}, targets={"emotion": QuestionTargets(human=target)},
                preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS,
                                      truncated=False),
                split=_split(s["actor"])))
    return out
