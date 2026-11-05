"""Images in the prefix (docs/guide/06 §6.10): placeholder tokens, pixel packing, and mask equivalence with images.

The packed tree pass with images must give every leaf the hidden states it gets when its root-to-leaf path runs alone
through the stock Gemma4Model with the stock masks, on both variants: E4B-like (images causal) and 26B-A4B-like
(an image attends to itself in both directions on sliding layers).
"""
import base64
import io

import pytest
import torch
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor

from decisivelm.api_types import Media
from decisivelm.media import ImageTokens, encode_image, load_image, stack
from decisivelm.tree import Question, TreeRequest, attention_masks, hf_masks, pack
from tiny_gemma import SLIDING_WINDOW, VOCAB, random_image, tiny_multimodal

TOL = 1e-3


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def image_request(model, seed, budgets):
    """A request whose prefix holds one image per budget (prefix and images both far longer than W = 8)."""
    g = torch.Generator().manual_seed(seed)
    ids = lambda n: torch.randint(3, VOCAB - 20, (n,), generator=g).tolist()
    tokens, proc = ImageTokens.from_config(model.config), Gemma4ImageProcessor()
    prefix, images, starts = [2], [], []
    for i, b in enumerate(budgets):
        im = encode_image(proc, random_image(seed * 10 + i), tokens, b)
        starts.append(len(prefix) + 1)
        prefix += im.token_ids + ids(1)
        images.append(im)
    prefix += ids(5)
    return TreeRequest(prefix=prefix, images=images, image_starts=starts, questions=[
        Question(segment=ids(3), leaves=[ids(4), ids(5), ids(3)]),
        Question(segment=[], leaves=[ids(6)]),
    ])


def alone(model, req, path, bidirectional):
    pix, pos = stack(req.images)
    ids = torch.tensor([path])
    kw = {"mm_token_type_ids": (ids == model.config.image_token_id).long()} if bidirectional else {}
    return model(input_ids=ids, pixel_values=pix, image_position_ids=pos, **kw).last_hidden_state[0]


def packed(model, tree, bidirectional):
    masks = hf_masks(attention_masks(tree, SLIDING_WINDOW, bidirectional), torch.float32)
    return model(input_ids=tree.input_ids[None], position_ids=tree.position_ids[None], attention_mask=masks,
                 pixel_values=tree.pixel_values, image_position_ids=tree.image_position_ids).last_hidden_state[0]


def worst_error(model, reqs, bidirectional_masks, bidirectional_stock):
    tree = pack(reqs)
    with torch.no_grad():
        out = packed(model, tree, bidirectional_masks)
        spans, worst = iter(tree.leaf_spans), 0.0
        for req in reqs:
            for q in req.questions:
                for leaf in q.leaves:
                    head = req.prefix + q.segment
                    s, e = next(spans)
                    ref = alone(model, req, head + leaf, bidirectional_stock)
                    worst = max(worst, rel_err(out[s:e], ref[len(head):]))
    return worst


@pytest.mark.parametrize("variant", ["dense", "moe"])
def test_packed_images_match_each_path_run_alone(variant):
    model = tiny_multimodal(variant)
    bidir = variant == "moe"
    # Two requests in one pack, three images at two budgets: pixels must line up with their soft tokens in order.
    reqs = [image_request(model, 1, [70, 140]), image_request(model, 2, [70])]
    assert worst_error(model, reqs, bidir, bidir) < TOL


def test_bidirectional_image_attention_is_needed_on_26b_like_models():
    """Without the image block on sliding layers the 26B-A4B-like model no longer matches the stock forward: the
    check above would catch a missing or misplaced block."""
    model = tiny_multimodal("moe")
    reqs = [image_request(model, 1, [70])]
    assert worst_error(model, reqs, bidirectional_masks=False, bidirectional_stock=True) > 1e-2


def test_image_tokens_must_sit_at_their_offsets():
    model = tiny_multimodal("dense")
    req = image_request(model, 1, [70])
    req.image_starts = [req.image_starts[0] + 1]
    with pytest.raises(ValueError):
        pack([req])


def test_load_image_from_path_and_base64(tmp_path):
    img = random_image(3)
    img.save(tmp_path / "a.png")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    a = load_image(Media(type="image", path="a.png"), root=tmp_path)
    b = load_image(Media(type="image", data=base64.b64encode(buf.getvalue()).decode()))
    assert a.size == b.size == img.size and a.tobytes() == b.tobytes()
    with pytest.raises(ValueError):
        load_image(Media(type="image", url="https://example.com/a.png"))


def test_placeholders_match_the_gemma4_processor():
    """Our image tokens and pixels are exactly what Gemma 4's processor produces for a lone image placeholder."""
    from transformers import AutoProcessor
    from decisivelm.model import backbone_profile, hub_kwargs
    p = backbone_profile("e4b")
    proc = AutoProcessor.from_pretrained(p["model_id"], revision=p["revision"], **hub_kwargs())
    tokens = ImageTokens(proc.tokenizer.convert_tokens_to_ids(proc.boi_token), proc.image_token_id,
                         proc.tokenizer.convert_tokens_to_ids(proc.eoi_token))
    for size, budget in [((40, 56), 70), ((1280, 720), 280)]:
        img = random_image(5, size)
        ref = proc(text=[proc.image_token], images=[[img]], return_tensors="pt", images_kwargs={"max_soft_tokens": budget})
        im = encode_image(proc.image_processor, img, tokens, budget)
        assert im.token_ids == ref["input_ids"][0].tolist()
        assert torch.equal(im.pixel_values, ref["pixel_values"][0])
        assert torch.equal(im.position_ids, ref["image_position_ids"][0])


@pytest.fixture(scope="module", params=["dense", "moe"])
def vision_dlm(request):
    from decisivelm.head import Scorer
    from decisivelm.infer import DecisiveLM
    from tiny_gemma import tokenizer
    backbone = tiny_multimodal(request.param, vocab=len(tokenizer()))
    torch.manual_seed(1)
    scorer = Scorer(backbone.config.get_text_config().hidden_size)
    with torch.no_grad():
        for p in scorer.parameters():
            p.normal_(0.0, 0.5)
    return DecisiveLM(backbone, tokenizer(), scorer, image_processor=Gemma4ImageProcessor(), image_budget=70)


ROLE = {"type": "choice", "instructions": "What is inside the marked box?",
        "criteria": {"parcel": "A parcel or package", "envelope": "A letter or envelope",
                     "other": "Something that is neither a parcel nor an envelope"}}
WHOLE = {"type": "noul", "instructions": "The box holds exactly one whole item"}


def test_system_one_with_an_image(vision_dlm, tmp_path):
    for i in range(2):
        random_image(10 + i, (64, 48)).save(tmp_path / f"{i}.png")
    vision_dlm.media_root = tmp_path
    req = lambda media, **qs: vision_dlm.system_one({"state": "Photo 12, box 3.", "media": media, "questions": qs})
    one = [{"type": "image", "path": "0.png"}]
    both = req(one, role=ROLE, whole=WHOLE)
    p = both["answers"]["role"]["probabilities"]
    assert list(p) == list(ROLE["criteria"]) and sum(p.values()) == pytest.approx(1)
    n_soft = sum(t == vision_dlm.image_tokens.image for t in vision_dlm.tree_for(
        {"state": "x", "media": one, "questions": {"w": WHOLE}}).prefix)
    assert both["usage"]["input_tokens"] > n_soft > 0
    # Independence holds with images in the prefix; the image itself matters.
    alone = req(one, role=ROLE)
    assert alone["answers"]["role"]["probabilities"] == pytest.approx(p, abs=1e-5)
    logits = lambda path: vision_dlm.logits([vision_dlm.tree_for(
        {"state": "Photo 12, box 3.", "media": [{"type": "image", "path": path}], "questions": {"role": ROLE}})])[1]
    assert rel_err(logits("1.png"), logits("0.png")) > 1e-2     # raw scores: the random head saturates the softmax


def test_text_only_dlm_refuses_media():
    from decisivelm.head import Scorer
    from decisivelm.infer import DecisiveLM
    from tiny_gemma import tiny_decoder, tokenizer
    dlm = DecisiveLM(tiny_decoder("dense", "sdpa", vocab=len(tokenizer())), tokenizer(), Scorer(64))
    with pytest.raises(ValueError):
        dlm.tree_for({"state": "x", "media": [{"type": "image", "data": "AAAA"}], "questions": {"w": WHOLE}})


def test_dropped_pixels_reload_identically(vision_dlm, tmp_path):
    """Training keeps trees without pixels and reloads them per micro-batch; the packed result is unchanged."""
    random_image(20, (64, 48)).save(tmp_path / "0.png")
    vision_dlm.media_root = tmp_path
    req = {"state": "Photo 1.", "media": [{"type": "image", "path": "0.png"}], "questions": {"role": ROLE}}
    keep = vision_dlm.tree_for(req)
    drop = vision_dlm.tree_for(req)
    drop.images[0].drop_pixels()
    assert drop.images[0].pixel_values is None
    assert torch.equal(vision_dlm.logits([keep])[1], vision_dlm.logits([drop])[1])


def test_data_urls_are_accepted_and_unreadable_images_are_value_errors():
    buf = io.BytesIO()
    random_image(3).save(buf, format="PNG")
    raw = base64.b64encode(buf.getvalue()).decode()
    plain = load_image(Media(type="image", data=raw))
    assert load_image(Media(type="image", data="data:image/png;base64," + raw)).tobytes() == plain.tobytes()
    with pytest.raises(ValueError, match="not valid base64"):
        load_image(Media(type="image", data="not base64!"))
    with pytest.raises(ValueError, match="not a readable image"):
        load_image(Media(type="image", data=base64.b64encode(b"%PDF-1.4 not an image").decode()))
