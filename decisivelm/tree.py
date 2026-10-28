"""Block tree -> packed token ids, logical positions and attention masks (docs/guide/06 §6.2, §6.3).

A request is a tree of token segments:

    prefix ─┬─ question segment ─┬─ leaf
            │                    └─ leaf
            └─ (empty segment, Noul) ── leaf

Tokens are laid out physically as prefix, then each question's segment followed
by its leaves. Logical positions follow the tree instead: each segment continues
from its parent's last position, so sibling leaves reuse the same positions.

A token may attend to every token of its ancestor segments and to earlier
tokens of its own segment. Gemma 4 has two layer kinds, so there are two masks:
the global one, and a sliding one that also requires the *logical* distance to
be under the window W. Reusing the global mask on sliding layers silently turns
them into full attention.

Images sit in the prefix as soft-token runs (decisivelm.media). A request lists
where each run starts, and the packed tree carries the pixels in token order plus
each token's image block, for backbones that let an image attend to itself in both
directions on sliding layers (26B A4B: `bidirectional_images`).

This module is token-level only; rendering text into segments is render.py's job.
"""
from dataclasses import dataclass, field

import torch

from decisivelm.media import EncodedAudio, EncodedImage, stack, stack_audio


@dataclass
class Question:
    segment: list[int]                  # question instructions; empty for Noul
    leaves: list[list[int]]             # one per option / level / statement, each ending in its query token


@dataclass
class TreeRequest:
    prefix: list[int]                   # system header, media placeholders, state
    questions: list[Question]
    images: list[EncodedImage] = field(default_factory=list)
    image_starts: list[int] = field(default_factory=list)   # prefix offset of each image's first soft token
    audios: list[EncodedAudio] = field(default_factory=list)
    audio_starts: list[int] = field(default_factory=list)   # prefix offset of each clip's first soft token


@dataclass
class PackedTree:
    input_ids: torch.Tensor             # [L]
    position_ids: torch.Tensor          # [L] logical positions, restarting at 0 per request
    node_of_token: torch.Tensor         # [L] segment index of each token
    ancestor: torch.Tensor              # [N, N] ancestor[a, b]: segment b is an ancestor of, or is, segment a
    query_index: torch.Tensor           # [n_leaves] physical index of each leaf's last (query) token
    leaf_question: torch.Tensor         # [n_leaves] packed question index of each leaf
    request_of_question: torch.Tensor   # [n_questions] request index of each question
    leaf_spans: list[tuple[int, int]] = field(default_factory=list)   # physical [start, end) of each leaf
    image_block: torch.Tensor | None = None       # [L] index of the image a soft token belongs to, else -1
    pixel_values: torch.Tensor | None = None      # [n_images, P, d] in token order (media.stack)
    image_position_ids: torch.Tensor | None = None
    input_features: torch.Tensor | None = None    # [n_audio, T, n_mels] in token order (media.stack_audio)
    input_features_mask: torch.Tensor | None = None


def pack(requests: list[TreeRequest]) -> PackedTree:
    ids, pos, node = [], [], []
    parents: list[int] = []
    query_index, leaf_question, request_of_question, leaf_spans = [], [], [], []
    images, block_spans, audios = [], [], []

    def add_segment(tokens, parent, start_pos):
        n = len(parents)
        parents.append(parent)
        begin = len(ids)
        ids.extend(tokens)
        pos.extend(range(start_pos, start_pos + len(tokens)))
        node.extend([n] * len(tokens))
        return n, begin

    for r, req in enumerate(requests):
        if len(req.images) != len(req.image_starts):
            raise ValueError("one start offset per image")
        root, root_begin = add_segment(req.prefix, -1, 0)
        for im, start in zip(req.images, req.image_starts):
            if req.prefix[start - 1:start + im.n_soft + 1] != im.token_ids:
                raise ValueError("an image's tokens are not at its start offset in the prefix")
            block_spans.append((root_begin + start, root_begin + start + im.n_soft))
            images.append(im)
        if len(req.audios) != len(req.audio_starts):
            raise ValueError("one start offset per audio clip")
        for clip, start in zip(req.audios, req.audio_starts):
            if req.prefix[start - 1:start + clip.n_soft + 1] != clip.token_ids:
                raise ValueError("an audio clip's tokens are not at its start offset in the prefix")
            audios.append(clip)
        for q in req.questions:
            qi = len(request_of_question)
            request_of_question.append(r)
            if q.segment:
                qnode, _ = add_segment(q.segment, root, len(req.prefix))
            else:
                qnode = root
            leaf_pos = len(req.prefix) + len(q.segment)
            for leaf in q.leaves:
                if not leaf:
                    raise ValueError("a leaf needs at least its query token")
                _, begin = add_segment(leaf, qnode, leaf_pos)
                leaf_spans.append((begin, begin + len(leaf)))
                query_index.append(begin + len(leaf) - 1)
                leaf_question.append(qi)

    n = len(parents)
    ancestor = torch.zeros(n, n, dtype=torch.bool)
    for a in range(n):
        b = a
        while b != -1:
            ancestor[a, b] = True
            b = parents[b]

    block = torch.full((len(ids),), -1, dtype=torch.long)
    for b, (s, e) in enumerate(block_spans):
        block[s:e] = b
    pix, pix_pos = stack(images) if images else (None, None)
    feats, feats_mask = stack_audio(audios) if audios else (None, None)
    return PackedTree(
        input_ids=torch.tensor(ids, dtype=torch.long),
        position_ids=torch.tensor(pos, dtype=torch.long),
        node_of_token=torch.tensor(node, dtype=torch.long),
        ancestor=ancestor,
        query_index=torch.tensor(query_index, dtype=torch.long),
        leaf_question=torch.tensor(leaf_question, dtype=torch.long),
        request_of_question=torch.tensor(request_of_question, dtype=torch.long),
        leaf_spans=leaf_spans,
        image_block=block,
        pixel_values=pix,
        image_position_ids=pix_pos,
        input_features=feats,
        input_features_mask=feats_mask,
    )


def attention_masks(tree: PackedTree, sliding_window: int, bidirectional_images: bool = False) -> dict[str, torch.Tensor]:
    """Boolean [L, L] masks (True = may attend), keyed by Gemma 4 layer type.

    `bidirectional_images` reproduces Gemma 4's `use_bidirectional_attention="vision"` (26B A4B): on sliding layers
    the soft tokens of one image also see each other's later tokens, still inside the window; global layers stay
    causal (transformers' create_masks_for_vision_model)."""
    node = tree.node_of_token
    idx = torch.arange(len(node))
    same_node = node[:, None] == node[None, :]
    allowed = tree.ancestor[node[:, None], node[None, :]] & (~same_node | (idx[None, :] <= idx[:, None]))
    pos = tree.position_ids
    in_window = (pos[:, None] - pos[None, :]) < sliding_window
    sliding = allowed
    if bidirectional_images and tree.image_block is not None:
        b = tree.image_block
        sliding = allowed | ((b[:, None] == b[None, :]) & (b[:, None] >= 0))
    return {"full_attention": allowed, "sliding_attention": sliding & in_window}


def hf_masks(masks: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Additive [1, 1, L, L] float masks in the per-layer-type mapping Gemma4TextModel accepts.

    Float rather than boolean so the same masks work for both the SDPA and the
    eager attention paths.
    """
    neg = torch.finfo(dtype).min
    return {k: torch.where(m, 0.0, neg).to(dtype)[None, None] for k, m in masks.items()}
