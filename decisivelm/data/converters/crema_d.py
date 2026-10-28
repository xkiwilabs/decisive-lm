"""Audio example: CREMA-D (Cao et al. 2014; docs/guide/14), perceived emotion in acted speech, from the voice alone.

- media: the clip's WAV (AudioWAV/<clip>.wav, relative to the repository root); state: the sentence the actor
  says, which is the same in every emotion, so the emotion must come from the voice
- one Choice, which emotion the speaker expresses, over the six the study asked about
- targets: the shares of the listeners who heard the clip without the video (queryType 1 in finishedResponses.csv,
  about 10 per clip). The emotion the actor was asked to show is not the target: listeners often hear something
  else, and that disagreement is what the confidence should follow. The actors' demographics are never read.
- splits by actor (91), so no voice is in two splits: 60% train, 10% val, 10% gold_calib, 20% gold_test
"""
import hashlib
import pathlib

import pandas as pd

from decisivelm import CONFIGS
from decisivelm.api_types import Media
from decisivelm.data.schema import HumanTarget, Lineage, Preprocess, QuestionTargets, Record
from decisivelm.preprocess import MAX_STATE_TOKENS
from decisivelm.render import RENDER_VERSION

SOURCE = "crema_d"
SENTENCES = {"IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.",
             "IOM": "I'm on my way to the meeting.", "IWW": "I wonder what this is about.",
             "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
             "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.",
             "DFA": "Don't forget a jacket.", "ITS": "I think I've seen this before.",
             "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes."}
EMOTIONS = {"A": ("anger", "Angry, annoyed or irritated"), "D": ("disgust", "Disgusted or repelled"),
            "F": ("fear", "Afraid, anxious or nervous"), "H": ("happy", "Happy, cheerful or amused"),
            "N": ("neutral", "No particular emotion: a calm, flat delivery"), "S": ("sad", "Sad, unhappy or downcast")}
QUESTION = {"type": "choice", "instructions": "Which emotion does the speaker's voice express?",
            "criteria": {name: desc for name, desc in EMOTIONS.values()}}
SPLITS = [(0.6, "train"), (0.7, "val"), (0.8, "gold_calib"), (1.0, "gold_test")]


def _split(actor: str) -> str:
    u = int(hashlib.sha256(f"{SOURCE}:{actor}".encode()).hexdigest()[:15], 16) / 16 ** 15
    return next(s for t, s in SPLITS if u < t)


def media_path(p: pathlib.Path) -> str:
    """Relative to the repository when the file is inside it (run from the repository root), else absolute."""
    try:
        return str(p.resolve().relative_to(CONFIGS.parent.resolve()))
    except ValueError:
        return str(p.resolve())


def convert(source_dir: pathlib.Path) -> list[Record]:
    source_dir = pathlib.Path(source_dir)
    d = pd.read_csv(source_dir / "finishedResponses.csv", dtype=str,
                    usecols=["localid", "queryType", "clipName", "respEmo"])
    d = d[(d.queryType == "1") & d.respEmo.isin(list(EMOTIONS))]
    out = []
    for clip, g in sorted(d.groupby("clipName")):
        wav = source_dir / "AudioWAV" / f"{clip}.wav"
        if len(g) < 2 or not wav.exists():
            continue
        actor, sentence = clip.split("_")[:2]
        shares = g.respEmo.value_counts(normalize=True)
        target = HumanTarget(probabilities={name: float(shares.get(code, 0.0)) for code, (name, _) in EMOTIONS.items()},
                             n_raters=len(g))
        out.append(Record(
            id=f"{SOURCE}-{clip}",
            lineage=Lineage(source_dataset=SOURCE, source_id=clip, family_id=f"{SOURCE}-actor-{actor}"),
            license="ODbL-1.0", tier="S", modality=["audio", "text"], language="en",
            state=f'An actor says: "{SENTENCES[sentence]}"', media=[Media(type="audio", path=media_path(wav))],
            questions={"emotion": QUESTION}, targets={"emotion": QuestionTargets(human=target)},
            preprocess=Preprocess(render_version=RENDER_VERSION, max_state_tokens=MAX_STATE_TOKENS, truncated=False),
            split=_split(actor)))
    return out
