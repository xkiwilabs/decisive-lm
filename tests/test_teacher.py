"""VLLMTeacher's requests (decisivelm/synth/teacher.py)."""
import io
import json

import pytest

from decisivelm.synth import teacher as T


def captured_post(monkeypatch, cfg):
    seen = {}

    def urlopen(req, timeout):
        seen["headers"] = dict(req.header_items())
        return io.BytesIO(json.dumps({"usage": {"prompt_tokens": 1}}).encode())
    monkeypatch.setattr(T.urllib.request, "urlopen", urlopen)
    teacher = T.VLLMTeacher(cfg, tokenizer=None, gate=T.Gate(T.Schedule()))
    teacher._post("/completions", {"model": "m"})
    return seen["headers"]


@pytest.mark.parametrize("set_env", [False, True])
def test_the_api_key_named_in_the_config_is_sent_when_set(monkeypatch, set_env):
    key = "-".join(["test", "value", "123"])
    monkeypatch.delenv("DLM_TEST_KEY", raising=False)
    if set_env:
        monkeypatch.setenv("DLM_TEST_KEY", key)
    headers = captured_post(monkeypatch, {"base_url": "http://localhost:1/v1", "api_key_env": "DLM_TEST_KEY"})
    assert headers.get("Authorization") == (f"Bearer {key}" if set_env else None)


def test_no_api_key_env_sends_no_authorization(monkeypatch):
    assert "Authorization" not in captured_post(monkeypatch, {"base_url": "http://localhost:1/v1", "api_key_env": None})


@pytest.fixture(scope="module")
def local():
    """LocalTeacher on a tiny Gemma 4 with an lm_head, a vision tower and an audio tower."""
    import torch
    from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
    from tiny_gemma import tiny_multimodal, tokenizer
    base = tiny_multimodal("dense", vocab=len(tokenizer()), audio=True)
    model = Gemma4ForConditionalGeneration._from_config(base.config, dtype=torch.float32).eval()
    model.model.load_state_dict(base.state_dict())
    return T.LocalTeacher(model, tokenizer(), Gemma4ImageProcessor(), Gemma4AudioFeatureExtractor(), image_budget=70)


def png(seed: int):
    import base64
    from decisivelm.api_types import Media
    from tiny_gemma import random_image
    buf = io.BytesIO()
    random_image(seed).save(buf, format="PNG")
    return Media(type="image", data=base64.b64encode(buf.getvalue()).decode())


def wav(hz: float):
    from decisivelm.api_types import Media
    from test_audio import tone, wav_b64
    return Media(type="audio", data=wav_b64(tone(0.6, hz)))


def test_local_teacher_puts_the_media_where_the_state_starts(local):
    enc = lambda text: local.tokenizer.encode(text, add_special_tokens=False)
    view = local.bound([png(0), wav(440)])
    ids = view.prompt_ids("Question: x\n\nState:\n", "A photo.\nAnswer:")
    want = enc("Question: x\n\nState:\n")
    for m in view.media:
        want += m.token_ids + enc("\n")
    assert ids == want + enc("A photo.\nAnswer:") and local.media == []


def test_local_teacher_answers_depend_on_the_media(local):
    lp = lambda media: local.bound(media).next_token_logprobs("State:\n", ["A photo.\nAnswer:"])[0]
    assert not torch_close(lp([png(0)]), lp([png(1)]))
    assert not torch_close(lp([wav(300)]), lp([wav(1200)]))
    assert torch_close(lp([png(0)]), lp([png(0)]))


def torch_close(a, b) -> bool:
    import torch
    return torch.allclose(a, b, atol=1e-5)


def test_local_teacher_labels_a_score_question_with_media(local):
    from decisivelm.api_types import ScoreQuestion
    q = ScoreQuestion.model_validate({"type": "score", "instructions": "How clear is the recording?",
                                      "criteria": ["Bad", "Fair", "Good"]})
    [(ans, info)] = T.question_labels(q, ["A clip."], ["r1"], local.bound([wav(440)]), max_labels=20)
    assert sum(ans["probabilities"].values()) == pytest.approx(1) and info["n_orderings"] == 2
