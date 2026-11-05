"""The model-origin rule in code (docs/guide/16 §16.2): every configured backbone and teacher is of approved origin,
and a model of any other origin is refused at load time."""
import pytest
import yaml

from decisivelm import CONFIGS
from decisivelm.model import backbone_profile, check_origin
from decisivelm.synth.teacher import load_teacher_config


def test_every_configured_backbone_and_teacher_loads():
    for name in yaml.safe_load((CONFIGS / "model.yaml").read_text())["backbones"]:
        assert backbone_profile(name)["developer"] == "Google DeepMind"
    for name in yaml.safe_load((CONFIGS / "teachers.yaml").read_text())["teachers"]:
        assert load_teacher_config(name)["base_model_origin"] == "Google DeepMind"


@pytest.mark.parametrize("profile", [{"developer": "Other Lab", "base_model_origin": "Google DeepMind"},
                                     {"developer": "Google DeepMind", "base_model_origin": "Other Lab"},
                                     {"developer": "Google DeepMind"}])
def test_an_unapproved_or_missing_origin_is_refused(profile):
    with pytest.raises(ValueError, match="approved_origins"):
        check_origin(profile, "test model")
