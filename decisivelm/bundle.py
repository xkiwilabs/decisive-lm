"""Adapter bundles (docs/guide/06 §6.11): what train_p1.py and train_p3.py write, reattached for inference.

Layout: calibration.json (temperatures, backbone profile, and a `lora` block for P3), scorer.pt, and for P3
adapter.pt (LoRA A/B and query-token rows).

    dlm = load_bundle("bundles/decisivelm-e4b-general")            # loads the backbone the bundle was trained on
    dlm = attach_bundle(bundle, backbone, tokenizer)                 # onto a backbone from decisivelm.model.load_backbone
"""
import json
import pathlib

import torch
import yaml

from decisivelm import CONFIGS
from decisivelm.head import Calibration, Scorer
from decisivelm.infer import QUERY_IDS, DecisiveLM
from decisivelm.lora import apply_lora, load_lora_state


def bundle_backbone(bundle: pathlib.Path) -> str:
    """The configs/model.yaml profile whose model id and revision the bundle was trained on."""
    trained = json.loads((pathlib.Path(bundle) / "calibration.json").read_text())["backbone"]
    for name, p in yaml.safe_load((CONFIGS / "model.yaml").read_text())["backbones"].items():
        if (p["model_id"], p["revision"]) == (trained["model_id"], trained["revision"]):
            return name
    raise ValueError(f"{bundle}: no backbone profile in configs/model.yaml is "
                     f"{trained['model_id']}@{trained['revision']}, the backbone it was trained on")


def load_bundle(bundle: pathlib.Path, dtype=None, device: str | None = None,
                media_root: pathlib.Path | None = None) -> DecisiveLM:
    """Load the backbone a bundle was trained on (in the dtype it was trained in, unless given) with its image
    processor and audio feature extractor, and attach the bundle. Relative media paths resolve against `media_root`."""
    from decisivelm.model import backbone_profile, load_backbone, load_feature_extractor, load_image_processor
    bundle = pathlib.Path(bundle)
    calib = json.loads((bundle / "calibration.json").read_text())
    name = bundle_backbone(bundle)
    backbone, tokenizer = load_backbone(name, dtype=dtype or getattr(torch, calib.get("dtype", "bfloat16")),
                                        device=device)
    dlm = attach_bundle(bundle, backbone, tokenizer)
    if "image" in backbone_profile(name).get("modalities", []):
        dlm.image_processor = load_image_processor(name)
        dlm.image_budget = calib.get("image_budget") or backbone_profile(name)["image_max_soft_tokens"]
        dlm.media_root = pathlib.Path(media_root) if media_root else None
    if "audio" in backbone_profile(name).get("modalities", []):
        dlm.feature_extractor = load_feature_extractor(name)
        dlm.media_root = pathlib.Path(media_root) if media_root else None
    return dlm


def attach_bundle(bundle: pathlib.Path, backbone, tokenizer) -> DecisiveLM:
    """DecisiveLM for a saved bundle on an already-loaded backbone. A backbone from decisivelm.model.load_backbone
    is checked against the model id and revision the bundle records; any other backbone is the caller's to check."""
    bundle = pathlib.Path(bundle)
    calib = json.loads((bundle / "calibration.json").read_text())
    loaded, trained = getattr(backbone, "decisivelm_backbone", None), calib.get("backbone")
    if loaded and trained and (loaded["model_id"], loaded["revision"]) != (trained["model_id"], trained["revision"]):
        raise ValueError(f"{bundle.name} was trained on {trained['model_id']}@{trained['revision']}, "
                         f"not {loaded['model_id']}@{loaded['revision']}")
    device = next(backbone.parameters()).device
    if (bundle / "adapter.pt").exists():
        lora = calib["lora"]
        apply_lora(backbone, lora["target_regex"], rank=lora["rank"], alpha=lora["alpha"],
                   query_ids=list(QUERY_IDS.values()))
        load_lora_state(backbone, torch.load(bundle / "adapter.pt", weights_only=True, map_location=device))
    scorer = Scorer(backbone.config.get_text_config().hidden_size)
    scorer.load_state_dict(torch.load(bundle / "scorer.pt", weights_only=True, map_location="cpu"))
    return DecisiveLM(backbone.eval(), tokenizer, scorer.to(device).eval(),
                      calibration=Calibration(temperatures=calib["temperatures"], b_noul=calib.get("b_noul", 0.0),
                                              tasks=calib.get("tasks", {})), adapter=bundle.name)
