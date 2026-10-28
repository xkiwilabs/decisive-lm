"""Mask equivalence on a tiny random Gemma 4 (docs/guide/06 §6.3; results/M0.md).

The packed tree pass must give every token the same hidden state it gets when
its root-to-leaf path runs alone as an ordinary sequence. The tree is built so
that the window matters: the prefix is longer than W, and the branches under
one question together exceed W.
"""
import pytest
import torch

from decisivelm.tree import Question, TreeRequest, attention_masks, hf_masks, pack
from tiny_gemma import SLIDING_WINDOW, VOCAB, tiny_decoder

TOL = 1e-3   # relative error on hidden states, fp32 (results/M0.md)


def random_tree(seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = lambda n: torch.randint(3, VOCAB, (n,), generator=g).tolist()
    return TreeRequest(
        prefix=ids(12),                                        # > W = 8
        questions=[
            Question(segment=ids(3), leaves=[ids(4), ids(5), ids(3)]),   # leaves sum to 12 > W
            Question(segment=[], leaves=[ids(6)]),                       # Noul shape
            Question(segment=ids(2), leaves=[ids(2), ids(7)]),
        ],
    )


def paths(req):
    """Every root-to-leaf path as its own token list, with the leaf's span in it."""
    for q in req.questions:
        for leaf in q.leaves:
            head = req.prefix + q.segment
            yield head + leaf, len(head)


def packed_states(model, tree, masks):
    out = model(input_ids=tree.input_ids[None], position_ids=tree.position_ids[None],
                attention_mask=masks)
    return out.last_hidden_state[0]


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("variant", ["dense", "moe"])
@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_packed_tree_matches_each_path_run_alone(variant, attn):
    model = tiny_decoder(variant, attn)
    req = random_tree()
    tree = pack([req])
    with torch.no_grad():
        packed = packed_states(model, tree, hf_masks(attention_masks(tree, SLIDING_WINDOW), torch.float32))
        leaf_spans = iter(tree.leaf_spans)
        for path, leaf_start in paths(req):
            alone = model(input_ids=torch.tensor([path])).last_hidden_state[0]
            start, end = next(leaf_spans)
            assert rel_err(packed[start:end], alone[leaf_start:]) < TOL
            # and the shared ancestors agree too
            assert rel_err(packed[:len(req.prefix)], alone[:len(req.prefix)]) < TOL


@pytest.mark.parametrize("variant", ["dense", "moe"])
def test_one_global_mask_for_all_layers_is_detected(variant):
    """The failure docs/guide/06 §6.3 warns about: the global mask reused on sliding layers.

    If this test ever passes the equivalence check, the tiny config no longer
    exercises the window and the equivalence test above proves nothing.
    """
    model = tiny_decoder(variant, "sdpa")
    req = random_tree()
    tree = pack([req])
    masks = attention_masks(tree, SLIDING_WINDOW)
    wrong = hf_masks({"full_attention": masks["full_attention"],
                      "sliding_attention": masks["full_attention"]}, torch.float32)
    with torch.no_grad():
        packed = packed_states(model, tree, wrong)
        path, leaf_start = next(paths(req))
        alone = model(input_ids=torch.tensor([path])).last_hidden_state[0]
        start, end = tree.leaf_spans[0]
        assert rel_err(packed[start:end], alone[leaf_start:]) > 100 * TOL


@pytest.mark.parametrize("variant", ["dense", "moe"])
def test_packing_two_requests_matches_running_them_separately(variant):
    model = tiny_decoder(variant, "sdpa")
    r1, r2 = random_tree(1), random_tree(2)
    with torch.no_grad():
        both = pack([r1, r2])
        joint = packed_states(model, both, hf_masks(attention_masks(both, SLIDING_WINDOW), torch.float32))
        for req, offset in [(r1, 0), (r2, len(pack([r1]).input_ids))]:
            single = pack([req])
            alone = packed_states(model, single, hf_masks(attention_masks(single, SLIDING_WINDOW), torch.float32))
            assert rel_err(joint[offset:offset + len(single.input_ids)], alone) < TOL
