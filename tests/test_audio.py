"""Audio in the prefix (docs/guide/06 §6.10): decoding, soft-token counts, packing, and the request path on a tiny
Gemma4Model with an audio tower (E4B has one; 26B A4B does not)."""
import base64
import io
import types

import numpy as np
import pytest
import soundfile
import torch
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor

from decisivelm.api_types import Media, SystemOneRequest
from decisivelm.media import MAX_AUDIO_SECONDS, AudioTokens, audio_soft_tokens, encode_audio, load_audio, stack_audio
from tiny_gemma import tiny_multimodal

SR = 16_000


def tone(seconds: float, hz: float = 440.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def wav_b64(wave: np.ndarray, sr: int = SR) -> str:
    buf = io.BytesIO()
    soundfile.write(buf, wave, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("seconds", [0.5, 1.0, 2.37, 7.0])
def test_soft_token_count_matches_the_processor(seconds):
    fe = Gemma4AudioFeatureExtractor()
    wave = tone(seconds)
    mask = fe([wave], sampling_rate=SR, return_tensors="pt")["input_features_mask"][0].bool()
    proc = types.SimpleNamespace(feature_extractor=fe, audio_seq_length=750)
    assert audio_soft_tokens(mask) == Gemma4Processor._compute_audio_num_tokens(proc, wave, SR)
    assert audio_soft_tokens(mask) == pytest.approx(seconds * 1000 / 40, abs=2)   # one token per 40 ms


def test_load_audio_resamples_to_mono_16k_and_cuts_at_30_seconds(tmp_path):
    stereo = np.stack([tone(1.0, sr=44_100), tone(1.0, 220, sr=44_100)], axis=1)
    soundfile.write(tmp_path / "a.flac", stereo, 44_100)
    wave = load_audio(Media(type="audio", path="a.flac"), tmp_path)
    assert wave.ndim == 1 and wave.dtype == np.float32 and len(wave) == SR
    long = load_audio(Media(type="audio", data=wav_b64(tone(MAX_AUDIO_SECONDS + 2))))
    assert len(long) == MAX_AUDIO_SECONDS * SR


def test_clips_pad_to_the_longest_with_masked_frames():
    tokens, fe = AudioTokens(boa=1, audio=2, eoa=3), Gemma4AudioFeatureExtractor()
    a, b = encode_audio(fe, tone(0.5), tokens), encode_audio(fe, tone(1.2), tokens)
    feats, mask = stack_audio([a, b])
    assert feats.shape[:2] == mask.shape == (2, len(b.features))
    assert mask[0].sum() == a.mask.sum() and not mask[0, len(a.mask):].any()
    assert a.token_ids[0] == 1 and a.token_ids[-1] == 3 and a.n_soft == a.token_ids.count(2)


def test_requests_accept_audio_but_not_urls_or_too_many_clips():
    q = {"w": {"type": "noul", "instructions": "Someone is speaking"}}
    SystemOneRequest.model_validate({"state": "x", "media": [{"type": "audio", "data": "AAAA"}], "questions": q})
    with pytest.raises(ValueError):
        SystemOneRequest.model_validate({"state": "x", "media": [{"type": "audio", "url": "https://x/a.wav"}],
                                         "questions": q})
    with pytest.raises(ValueError):
        SystemOneRequest.model_validate({"state": "x", "media": [{"type": "audio", "data": "AAAA"}] * 5,
                                         "questions": q})


@pytest.fixture(scope="module")
def audio_dlm():
    from decisivelm.head import Scorer
    from decisivelm.infer import DecisiveLM
    from tiny_gemma import tokenizer
    backbone = tiny_multimodal("dense", vocab=len(tokenizer()), audio=True)
    torch.manual_seed(1)
    scorer = Scorer(backbone.config.get_text_config().hidden_size)
    with torch.no_grad():
        for p in scorer.parameters():
            p.normal_(0.0, 0.5)
    return DecisiveLM(backbone, tokenizer(), scorer, feature_extractor=Gemma4AudioFeatureExtractor())


EMOTION = {"type": "choice", "instructions": "Which emotion does the speaker express?",
           "criteria": {"neutral": "No particular emotion", "happy": "Happy or amused", "angry": "Angry or annoyed"}}
SPEECH = {"type": "noul", "instructions": "The clip contains speech"}


def test_system_one_with_audio(audio_dlm):
    clip = lambda hz: [{"type": "audio", "data": wav_b64(tone(1.0, hz))}]
    req = lambda media, **qs: audio_dlm.system_one({"state": "Call 17, second 3.", "media": media, "questions": qs})
    both = req(clip(440), emotion=EMOTION, speech=SPEECH)
    p = both["answers"]["emotion"]["probabilities"]
    assert list(p) == list(EMOTION["criteria"]) and sum(p.values()) == pytest.approx(1)
    tree = audio_dlm.tree_for({"state": "x", "media": clip(440), "questions": {"s": SPEECH}})
    assert tree.prefix.count(audio_dlm.audio_tokens.audio) == tree.audios[0].n_soft > 20
    # Independence holds with audio in the prefix; the audio itself matters.
    assert req(clip(440), emotion=EMOTION)["answers"]["emotion"]["probabilities"] == pytest.approx(p, abs=1e-5)
    logits = lambda hz: audio_dlm.logits([audio_dlm.tree_for(
        {"state": "Call 17, second 3.", "media": clip(hz), "questions": {"emotion": EMOTION}})])[1]
    assert ((logits(1500) - logits(440)).norm() / logits(440).norm()).item() > 1e-2


def test_packed_requests_with_audio_match_each_alone(audio_dlm):
    trees = [audio_dlm.tree_for({"state": f"Clip {i}.", "media": [{"type": "audio", "data": wav_b64(tone(s, hz))}],
                                 "questions": {"emotion": EMOTION}})
             for i, (s, hz) in enumerate([(0.7, 300), (1.6, 900)])]
    _, packed = audio_dlm.logits(trees)
    alone = torch.cat([audio_dlm.logits([t])[1] for t in trees])
    assert torch.allclose(packed, alone, atol=1e-4)


def test_dropped_features_reload_identically(audio_dlm):
    req = {"state": "x", "media": [{"type": "audio", "data": wav_b64(tone(0.8))}], "questions": {"s": SPEECH}}
    keep, drop = audio_dlm.tree_for(req), audio_dlm.tree_for(req)
    drop.audios[0].drop_features()
    assert drop.audios[0].features is None
    assert torch.equal(audio_dlm.logits([keep])[1], audio_dlm.logits([drop])[1])


def test_tree_pass_with_audio_matches_the_stock_model_on_each_path(audio_dlm):
    """Each leaf's hidden state equals the stock Gemma4Model's (its own causal masks, positions 0..n) on that leaf's
    root-to-leaf path, with the audio tower's output in the same slots."""
    req = {"state": "Call 17, second 3: " + "the caller repeats the order number. " * 3,
           "media": [{"type": "audio", "data": wav_b64(tone(1.3, 600))}], "questions": {"emotion": EMOTION}}
    tree = audio_dlm.tree_for(req)
    _, states = audio_dlm.query_states([tree])
    q = tree.questions[0]
    for leaf, got in zip(q.leaves, states):
        ids = torch.tensor([tree.prefix + q.segment + leaf])
        feats, mask = stack_audio(tree.audios)
        with torch.no_grad():
            want = audio_dlm.backbone(input_ids=ids, input_features=feats, input_features_mask=mask).last_hidden_state[0, -1]
        assert ((got - want).norm() / want.norm()).item() < 1e-4


def test_unusual_sample_rates_are_refused_before_decoding():
    with pytest.raises(ValueError, match="unsupported audio"):
        load_audio(Media(type="audio", data=wav_b64(tone(0.2, sr=12_345), sr=12_345)))
