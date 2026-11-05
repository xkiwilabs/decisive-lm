"""A saved bundle (train_p1.py / train_p3.py layout) reattached to a fresh backbone answers exactly as the model
that was saved (docs/guide/06 §6.11)."""
import json

import pytest
import torch

from decisivelm.bundle import attach_bundle, bundle_backbone
from decisivelm.head import Calibration, Scorer, bucket
from decisivelm.infer import QUERY_IDS, DecisiveLM
from decisivelm.lora import apply_lora, lora_state
from decisivelm.model import backbone_profile
from tiny_gemma import tiny_decoder, tokenizer

REGEX = backbone_profile("e4b")["lora_target_regex"]
REQ = {"state": "Our API started returning 500s 20 minutes ago.",
       "questions": {"d": {"type": "choice", "instructions": "Which team should handle this?",
                           "criteria": {"billing": "Payment issues", "technical": "Bugs or outages", "sales": "Pricing"}}}}


def test_a_p3_bundle_reattaches_to_a_fresh_backbone(tmp_path):
    trained = tiny_decoder("dense", "sdpa", vocab=len(tokenizer()))
    apply_lora(trained, REGEX, rank=4, alpha=8, query_ids=list(QUERY_IDS.values()))
    with torch.no_grad():
        for name, p in trained.named_parameters():
            if name.endswith(("lora_B", "query_rows.delta")):
                p.normal_(0.0, 0.05)
    torch.manual_seed(1)
    scorer = Scorer(trained.config.hidden_size).eval()
    temps = {f"choice:{bucket(3)}": 1.7}
    expected = DecisiveLM(trained, tokenizer(), scorer, calibration=Calibration(temperatures=temps)).system_one(REQ)

    torch.save(lora_state(trained), tmp_path / "adapter.pt")
    torch.save(scorer.state_dict(), tmp_path / "scorer.pt")
    (tmp_path / "calibration.json").write_text(json.dumps(
        {"temperatures": temps, "lora": {"rank": 4, "alpha": 8, "dropout": 0.05, "target_regex": REGEX}}))

    got = attach_bundle(tmp_path, tiny_decoder("dense", "sdpa", vocab=len(tokenizer())), tokenizer()).system_one(REQ)
    exp, out = expected["answers"]["d"]["probabilities"], got["answers"]["d"]["probabilities"]
    assert max(abs(exp[c] - out[c]) for c in exp) < 1e-6


def test_bundle_keeps_the_noul_bias(tmp_path):
    """calibration.json's b_noul reaches serving: a Noul answer depends on it."""
    scorer = Scorer(64)
    torch.save(scorer.state_dict(), tmp_path / "scorer.pt")
    (tmp_path / "calibration.json").write_text(json.dumps({"temperatures": {"noul": 1.3}, "b_noul": -0.8}))
    dlm = attach_bundle(tmp_path, tiny_decoder("dense", "sdpa", vocab=len(tokenizer())), tokenizer())
    assert dlm.calibration.b_noul == -0.8 and dlm.calibration.temperatures == {"noul": 1.3}


def test_a_bundle_refuses_a_backbone_revision_it_was_not_trained_on(tmp_path):
    p = backbone_profile("e4b")
    (tmp_path / "calibration.json").write_text(json.dumps(
        {"temperatures": {}, "backbone": {"model_id": p["model_id"], "revision": p["revision"]}}))
    backbone = tiny_decoder("dense", "sdpa", vocab=len(tokenizer()))
    backbone.decisivelm_backbone = {"name": "e4b", "model_id": p["model_id"], "revision": "0" * 40}
    with pytest.raises(ValueError, match="was trained on"):
        attach_bundle(tmp_path, backbone, tokenizer())


def test_a_bundle_names_the_backbone_profile_it_was_trained_on(tmp_path):
    p = backbone_profile("26b-a4b")
    (tmp_path / "calibration.json").write_text(json.dumps({"backbone": {"model_id": p["model_id"],
                                                                        "revision": p["revision"]}}))
    assert bundle_backbone(tmp_path) == "26b-a4b"
    (tmp_path / "calibration.json").write_text(json.dumps({"backbone": {"model_id": "x", "revision": "y"}}))
    with pytest.raises(ValueError, match="no backbone profile"):
        bundle_backbone(tmp_path)


def test_the_package_exports_decisivelm_with_its_loader():
    import decisivelm
    from decisivelm.bundle import load_bundle
    assert decisivelm.DecisiveLM is DecisiveLM and DecisiveLM.load.__doc__ and load_bundle


def test_a_question_type_the_bundle_never_trained_on_is_answered_with_a_warning(tmp_path):
    torch.save(Scorer(64).state_dict(), tmp_path / "scorer.pt")
    (tmp_path / "calibration.json").write_text(json.dumps({"temperatures": {"choice:*": 1.0},
                                                           "question_types": {"choice": 40}}))
    dlm = attach_bundle(tmp_path, tiny_decoder("dense", "sdpa", vocab=len(tokenizer())), tokenizer())
    dlm.system_one(REQ)
    assert dlm.warnings == []
    req = {**REQ, "questions": {**REQ["questions"], "u": {"type": "noul", "instructions": "The API is down"}}}
    with pytest.warns(UserWarning, match="never trained on noul"):
        dlm.system_one(req)
    assert len(dlm.warnings) == 1 and dlm.warnings[0].startswith("u: ")


def test_an_older_bundle_takes_its_trained_types_from_its_calibration():
    from decisivelm.bundle import trained_types
    assert trained_types({"temperatures": {"choice:*": 1.0, "choice:3-5": 1.1}}) == {"choice"}
    assert trained_types({"temperatures": {"noul": 1.0, "score:*": 1.0}, "question_types": {"noul": 3}}) == {"noul"}
    assert trained_types({"temperatures": {}}) is None
