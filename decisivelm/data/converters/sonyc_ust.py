"""SONYC Urban Sound Tagging v2 (Cartwright et al. 2020, NYU; docs/guide/14), the sounds three volunteers hear in a
ten-second recording from a New York City street sensor.

- media: the clip's WAV, extracted from the audio-*.tar.gz archives to audio/<audio_filename> (relative to the
  repository root); state: one fixed sentence, so the answer must come from the sound. The sensor's location and the
  time of recording are never put in the state
- one Noul per coarse class of the taxonomy (dcase-ust-taxonomy.yaml): engine, machinery impact, non-machinery impact,
  powered saw, alert signal, music, human voice, dog
- targets: the share of the clip's volunteers (annotator_id > 0; three per clip) who marked any fine class under that
  coarse class present. The SONYC team's rows (annotator_id ≤ 0) are not used
- splits: SONYC's own; train → train, validate → val, test (recorded in 2019) → gold_calib and gold_test by
  sensor, 40% of sensors to gold_calib
"""
import csv
import hashlib
import pathlib
import tarfile

from decisivelm.api_types import Media
from decisivelm.data.converters.crema_d import media_path
from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

SOURCE = "sonyc_ust"
STATE = "A ten-second recording from an acoustic sensor on a New York City street."
CLASSES = {
    "1": ("engine", "An engine can be heard: a car, truck, bus, motorbike or other vehicle or machine engine, "
                    "of any size"),
    "2": ("machinery_impact", "Impact machinery can be heard: a rock drill, jackhammer, hoe ram or pile driver"),
    "3": ("non_machinery_impact", "A non-machinery impact can be heard, such as something banging, slamming, "
                                  "dropping or being hit"),
    "4": ("powered_saw", "A powered saw can be heard: a chainsaw or a rotating saw"),
    "5": ("alert_signal", "An alert signal can be heard: a car horn, car alarm, siren or reversing beeper"),
    "6": ("music", "Music can be heard: from a stationary or moving source, such as a car stereo or an ice-cream "
                   "truck"),
    "7": ("human_voice", "A human voice can be heard: talking, shouting, a crowd or amplified speech"),
    "8": ("dog", "A dog can be heard barking or whining"),
}
SPLITS = {"train": "train", "validate": "val"}


def _u(key: str) -> float:
    return int(hashlib.sha256(f"{SOURCE}:{key}".encode()).hexdigest()[:15], 16) / 16 ** 15


def split(source_split: str, sensor: str) -> str:
    return SPLITS.get(source_split) or ("gold_calib" if _u(f"sensor:{sensor}") < 0.4 else "gold_test")


def extract(source_dir: pathlib.Path, wanted: set[str]) -> None:
    """The wanted clips from every audio-*.tar.gz, by bare file name into audio/; regular files only."""
    audio = source_dir / "audio"
    audio.mkdir(exist_ok=True)
    missing = {n for n in wanted if not (audio / n).exists()}
    for archive in sorted(source_dir.glob("audio-*.tar.gz")):
        if not missing:
            return
        with tarfile.open(archive, "r|gz") as t:
            for m in t:
                name = pathlib.PurePosixPath(m.name).name
                if m.isfile() and name in missing:
                    (audio / name).write_bytes(t.extractfile(m).read())
                    missing.discard(name)


def convert(source_dir: pathlib.Path) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    clips: dict[str, dict] = {}
    with open(source_dir / "annotations.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["annotator_id"]) <= 0:
                continue
            name = r["audio_filename"]
            c = clips.setdefault(name, {"split": r["split"], "sensor": r["sensor_id"], "votes": []})
            c["votes"].append({k: any(v == "1" for col, v in r.items()
                                      if col.endswith("_presence") and col.split("-")[0] == k) for k in CLASSES})
    extract(source_dir, {n for n in clips if pathlib.PurePosixPath(n).name == n})
    out = []
    for name in sorted(clips):
        c, wav = clips[name], source_dir / "audio" / name
        n = len(c["votes"])
        if n < 2 or not wav.exists():
            continue
        questions = {key: {"type": "noul", "instructions": text} for key, text in CLASSES.values()}
        targets = {key: QuestionTargets(human=HumanTarget(noul=sum(v[k] for v in c["votes"]) / n, n_raters=n))
                   for k, (key, _) in CLASSES.items()}
        out.append(Record(
            id=f"{SOURCE}-{pathlib.PurePosixPath(name).stem}",
            lineage=Lineage(source_dataset=SOURCE, source_id=name, family_id=f"{SOURCE}-sensor-{c['sensor']}"),
            license="CC-BY-4.0", tier="R", modality=["audio", "text"], language="en", state=STATE,
            media=[Media(type="audio", path=media_path(wav))], questions=questions, targets=targets,
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS, truncated=False),
            split=split(c["split"], c["sensor"])))
    return out
