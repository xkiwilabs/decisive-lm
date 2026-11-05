"""DecisiveLM.system_one: one System One request in, one response out (decisivelm.api_types).

Images and audio go into the prefix before the state text, in the request's media order (docs/guide/04 §4.5,
decisivelm.media).
"""
import json
import pathlib
import time
import warnings

import torch

from decisivelm import render
from decisivelm.api_types import ChoiceQuestion, ScoreQuestion, SystemOneRequest, SystemOneResponse, lint_request
from decisivelm.confidence import METHODS
from decisivelm.head import Calibration, Scorer, readout, task_key
from decisivelm.media import (DEFAULT_BUDGET, AudioTokens, ImageTokens, encode_audio, encode_image, load_audio,
                              load_image)
from decisivelm.preprocess import MAX_STATE_TOKENS, truncate_state
from decisivelm.tree import Question, TreeRequest, attention_masks, hf_masks, pack

QUERY_IDS = {"choice": 6, "score": 7, "noul": 8}   # <unused0..2>, configs/model.yaml


class RequestTooLarge(ValueError):
    """A request whose packed tree is longer than the caller's `max_tokens` (decisivelm.serve answers 413)."""


def tree_tokens(tree: TreeRequest) -> int:
    """Tokens in a request's packed sequence: the prefix (media included), each question segment and each leaf."""
    return len(tree.prefix) + sum(len(q.segment) + sum(len(leaf) for leaf in q.leaves) for q in tree.questions)


def serialize_state(state) -> str:
    """Strings as they are; JSON compactly in insertion order, identically for teacher and student (docs/guide/04 §4.6)."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, separators=(",", ":"))


class DecisiveLM:
    def __init__(self, backbone, tokenizer, scorer: Scorer, calibration: Calibration | None = None,
                 query_ids: dict[str, int] = QUERY_IDS, confidence_method: str = "peak", adapter: str | None = None,
                 max_state_tokens: int = MAX_STATE_TOKENS, image_processor=None, image_budget: int = DEFAULT_BUDGET,
                 media_root: pathlib.Path | None = None, feature_extractor=None, trained_types: set[str] | None = None):
        """`backbone` is a Gemma4Model or Gemma4TextModel: anything that returns last_hidden_state
        and accepts per-layer-type masks plus position_ids. Never the lm_head wrapper (docs/guide/06 §6.4).
        Images need a Gemma4Model and its processor's `image_processor`, audio its `feature_extractor` (E4B only);
        relative media paths resolve against `media_root`. A question of a type outside `trained_types` (the bundle's,
        decisivelm.bundle.trained_types) is answered with a warning: the head has never learned that type."""
        self.backbone, self.tokenizer, self.scorer = backbone, tokenizer, scorer
        self.calibration = calibration or Calibration()
        self.query_ids, self.confidence_method, self.adapter = query_ids, confidence_method, adapter
        self.max_state_tokens = max_state_tokens
        text_config = backbone.config.get_text_config()
        self.sliding_window = text_config.sliding_window
        self.bidirectional_images = getattr(text_config, "use_bidirectional_attention", None) == "vision"
        self.image_tokens = ImageTokens.from_config(backbone.config)
        self.image_processor, self.image_budget, self.media_root = image_processor, image_budget, media_root
        self.audio_tokens, self.feature_extractor = AudioTokens.from_config(backbone.config), feature_extractor
        self.trained_types = trained_types

    @staticmethod
    def load(bundle, **kwargs) -> "DecisiveLM":
        """A saved bundle on the backbone it was trained on (decisivelm.bundle.load_bundle)."""
        from decisivelm.bundle import load_bundle
        return load_bundle(bundle, **kwargs)

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False) if text else []

    def _tree(self, req: SystemOneRequest) -> TreeRequest:
        state, _ = truncate_state(serialize_state(req.state), self.tokenizer, self.max_state_tokens)
        prefix, images, starts, audios, audio_starts = [self.tokenizer.bos_token_id], [], [], [], []
        for m in req.media:
            if m.type == "audio":
                if self.audio_tokens is None or self.feature_extractor is None:
                    raise ValueError("this model takes no audio: its backbone has no audio input (26B A4B takes text "
                                     "and images only), or it was loaded without an audio feature_extractor")
                def encode(m=m):
                    return encode_audio(self.feature_extractor,
                                        load_audio(m, self.media_root, self.feature_extractor.sampling_rate),
                                        self.audio_tokens)
                item, items, offsets = encode(), audios, audio_starts
            else:
                if self.image_tokens is None or self.image_processor is None:
                    raise ValueError("this DecisiveLM has no vision path: pass a Gemma4Model and an image_processor")
                def encode(m=m):
                    return encode_image(self.image_processor, load_image(m, self.media_root), self.image_tokens,
                                        self.image_budget)
                item, items, offsets = encode(), images, starts
            item.reload = encode
            offsets.append(len(prefix) + 1)
            prefix += item.token_ids + self._ids(render.MEDIA_SEPARATOR)
            items.append(item)
        prefix += self._ids(render.prefix_text(state))
        questions = []
        for q in req.questions.values():
            qid = self.query_ids[q.type]
            questions.append(Question(segment=self._ids(render.question_segment(q)),
                                      leaves=[self._ids(t + " ") + [qid] for t in render.leaf_texts(q)]))
        return TreeRequest(prefix=prefix, questions=questions, images=images, image_starts=starts, audios=audios,
                           audio_starts=audio_starts)

    def tree_for(self, request) -> TreeRequest:
        req = request if isinstance(request, SystemOneRequest) else SystemOneRequest.model_validate(request)
        return self._tree(req)

    def query_states(self, trees: list[TreeRequest], grad: bool = False):
        """(packed tree, hidden state at every query token). The backbone's last hidden state only (docs/guide/06 §6.4)."""
        tree = pack(trees)
        device = next(self.backbone.parameters()).device
        dtype = self.backbone.get_input_embeddings().weight.dtype
        masks = {k: m.to(device) for k, m in hf_masks(
            attention_masks(tree, self.sliding_window, self.bidirectional_images), dtype).items()}
        mm = {}
        if tree.pixel_values is not None:
            vision_dtype = next(self.backbone.vision_tower.parameters()).dtype
            mm = {"pixel_values": tree.pixel_values.to(device, vision_dtype),
                  "image_position_ids": tree.image_position_ids.to(device)}
        if tree.input_features is not None:
            audio_dtype = next(self.backbone.audio_tower.parameters()).dtype
            mm |= {"input_features": tree.input_features.to(device, audio_dtype),
                   "input_features_mask": tree.input_features_mask.to(device)}
        with torch.set_grad_enabled(grad):
            states = self.backbone(input_ids=tree.input_ids[None].to(device),
                                   position_ids=tree.position_ids[None].to(device),
                                   attention_mask=masks, **mm).last_hidden_state[0]
        return tree, states[tree.query_index.to(device)]

    @torch.no_grad()
    def logits(self, trees: list[TreeRequest]):
        tree, q = self.query_states(trees)
        scorer_dtype = next(self.scorer.parameters()).dtype
        return tree, self.scorer(q.to(scorer_dtype)).float().cpu()

    def system_one(self, request, max_tokens: int | None = None) -> dict:
        t0 = time.perf_counter()
        req = request if isinstance(request, SystemOneRequest) else SystemOneRequest.model_validate(request)
        self.warnings = lint_request(req)
        if self.trained_types is not None:
            for qid, q in req.questions.items():
                if q.type not in self.trained_types:
                    trained = ", ".join(sorted(self.trained_types))
                    message = (f"{qid}: {self.adapter or 'this head'} was never trained on {q.type} questions, so its "
                               f"answer and confidence are untested; trained on {trained}")
                    self.warnings.append(message)
                    warnings.warn(message.split(": ", 1)[1], stacklevel=2)
        request_tree = self._tree(req)
        if max_tokens is not None and (n := tree_tokens(request_tree)) > max_tokens:
            raise RequestTooLarge(f"the request packs to {n:,} tokens, more than this server's {max_tokens:,}: send "
                                  f"fewer or shorter questions or a shorter state (states are cut at "
                                  f"{MAX_STATE_TOKENS:,})")
        tree, logits = self.logits([request_tree])
        conf = METHODS[self.confidence_method]
        answers = {}
        for i, (qid, q) in enumerate(req.questions.items()):
            names = list(q.criteria) if isinstance(q, ChoiceQuestion) else None
            p = readout(q.type, logits[tree.leaf_question == i], self.calibration, task_key(q), names).tolist()
            if isinstance(q, ChoiceQuestion):
                answers[qid] = {"type": "choice", "choice": names[max(range(len(p)), key=p.__getitem__)],
                                "confidence": conf(p), "probabilities": dict(zip(names, p))}
            elif isinstance(q, ScoreQuestion):
                answers[qid] = {"type": "score", "score": sum(k * x for k, x in enumerate(p)),
                                "confidence": conf(p),
                                "legend": {str(k): c for k, c in enumerate(q.criteria)},
                                "probabilities": {str(k): x for k, x in enumerate(p)}}
            else:
                answers[qid] = {"type": "noul", "noul": p[0]}
        resp = {"model": req.model, "answers": answers,
                "usage": {"input_tokens": len(tree.input_ids), "blocks": len(tree.query_index),
                          "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                          "confidence_method": self.confidence_method, "adapter": self.adapter}}
        return SystemOneResponse.model_validate(resp).model_dump()
