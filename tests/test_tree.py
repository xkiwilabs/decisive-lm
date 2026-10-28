"""Block tree layout: logical positions, global and sliding masks, packing (docs/guide/06 §6.2, §6.3)."""
import torch

from decisivelm.tree import Question, TreeRequest, attention_masks, pack

# Token ids are arbitrary labels here; tree.py never interprets them.
A, B, C, D, E, F, G, H, I, J = range(10, 20)


def one_question_tree():
    # prefix [A B C] -> question [D E] -> leaves [F G] and [H]
    return TreeRequest(prefix=[A, B, C], questions=[Question(segment=[D, E], leaves=[[F, G], [H]])])


def idx(tree, token):
    return tree.input_ids.tolist().index(token)


def test_physical_order_is_prefix_then_segment_then_leaves():
    tree = pack([one_question_tree()])
    assert tree.input_ids.tolist() == [A, B, C, D, E, F, G, H]


def test_siblings_reuse_the_same_logical_positions():
    tree = pack([one_question_tree()])
    assert tree.position_ids.tolist() == [0, 1, 2, 3, 4, 5, 6, 5]


def test_query_index_is_the_last_token_of_each_leaf():
    tree = pack([one_question_tree()])
    assert tree.query_index.tolist() == [idx(tree, G), idx(tree, H)]
    assert tree.leaf_question.tolist() == [0, 0]


def test_leaf_attends_to_ancestors_and_itself_only():
    tree = pack([one_question_tree()])
    allowed = attention_masks(tree, sliding_window=100)["full_attention"]
    seen_by_h = {tok for tok, ok in zip(tree.input_ids.tolist(), allowed[idx(tree, H)].tolist()) if ok}
    assert seen_by_h == {A, B, C, D, E, H}


def test_attention_within_a_segment_is_causal():
    tree = pack([one_question_tree()])
    allowed = attention_masks(tree, sliding_window=100)["full_attention"]
    assert allowed[idx(tree, G), idx(tree, F)]
    assert not allowed[idx(tree, F), idx(tree, G)]


def test_questions_do_not_see_each_other():
    tree = pack([TreeRequest(prefix=[A], questions=[Question(segment=[B], leaves=[[C]]),
                                                     Question(segment=[D], leaves=[[E]])])])
    allowed = attention_masks(tree, sliding_window=100)["full_attention"]
    for later, earlier in [(D, B), (D, C), (E, B), (E, C)]:
        assert not allowed[idx(tree, later), idx(tree, earlier)]


def test_noul_with_empty_segment_hangs_its_leaf_on_the_prefix():
    tree = pack([TreeRequest(prefix=[A, B], questions=[Question(segment=[], leaves=[[C, D]])])])
    assert tree.position_ids.tolist() == [0, 1, 2, 3]
    allowed = attention_masks(tree, sliding_window=100)["full_attention"]
    assert allowed[idx(tree, D)].all()


def test_sliding_window_uses_logical_not_physical_distance():
    # H sits at physical index 7 but logical position 5. With W = 3 it must see
    # logical positions 3..5 (D, E, H). D is 4 apart physically, so a mask built
    # from physical offsets would wrongly drop it.
    tree = pack([one_question_tree()])
    allowed = attention_masks(tree, sliding_window=3)["sliding_attention"]
    seen_by_h = {tok for tok, ok in zip(tree.input_ids.tolist(), allowed[idx(tree, H)].tolist()) if ok}
    assert seen_by_h == {D, E, H}


def test_global_mask_ignores_the_window():
    tree = pack([one_question_tree()])
    allowed = attention_masks(tree, sliding_window=3)["full_attention"]
    assert allowed[idx(tree, H), idx(tree, A)]


def test_packed_requests_are_block_diagonal_and_restart_positions():
    tree = pack([TreeRequest(prefix=[A, B], questions=[Question(segment=[], leaves=[[C]])]),
                 TreeRequest(prefix=[D], questions=[Question(segment=[E], leaves=[[F], [G]])])])
    assert tree.position_ids.tolist() == [0, 1, 2, 0, 1, 2, 2]
    allowed = attention_masks(tree, sliding_window=100)["full_attention"]
    first, second = slice(0, 3), slice(3, 7)
    assert not allowed[second, first].any()
    assert not allowed[first, second].any()
    assert tree.leaf_question.tolist() == [0, 1, 1]
    assert tree.request_of_question.tolist() == [0, 1]


def test_masks_are_boolean_square_matrices():
    tree = pack([one_question_tree()])
    for mask in attention_masks(tree, sliding_window=4).values():
        assert mask.dtype == torch.bool
        assert mask.shape == (len(tree.input_ids), len(tree.input_ids))
