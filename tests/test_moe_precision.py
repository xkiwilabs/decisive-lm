"""MoE numerics (docs/guide/06 §6.9): in bf16, near-tied expert scores make 26B A4B's routing depend on the packing
(results/invariance.md). These run parts or all of the backbone in fp32."""
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts, Gemma4TextModel, Gemma4TextRouter

from decisivelm.model import fp32_compute, fp32_routers
from tiny_gemma import VOCAB, tiny_decoder


def test_fp32_routers_choose_experts_from_full_precision_scores():
    """bf16 rounds the 128 router scores coarsely, so near-ties between the 8th and 9th expert are common and the
    choice can depend on packing (results/invariance.md). The patched router decides from fp32 scores."""
    model = tiny_decoder("moe", "sdpa").to(torch.bfloat16)
    router = next(m for m in model.modules() if isinstance(m, Gemma4TextRouter))
    x = torch.randn(64, model.config.hidden_size).to(torch.bfloat16)
    h = x.float()
    h = h * (h.pow(2).mean(-1, keepdim=True) + router.eps).pow(-0.5) * router.scale.float() * router.scalar_root_size
    expected = torch.topk(torch.softmax(h @ router.proj.weight.float().T, -1), router.config.top_k_experts, -1).indices

    assert fp32_routers(model) == model.config.num_hidden_layers
    _, weights, index = router(x)
    assert torch.equal(index, expected) and weights.dtype == torch.float32


def test_fp32_compute_matches_a_full_fp32_load_with_experts_stored_in_bf16():
    """The released checkpoint is bf16, so upcasting it is exact: expert weights (~90% of 26B A4B) can stay bf16 in
    memory while every matmul runs in fp32. The result must equal a model loaded wholly in fp32."""
    ref = tiny_decoder("moe", "sdpa")
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(p.to(torch.bfloat16).float())          # bf16-representable weights, like the real checkpoint
    model = Gemma4TextModel._from_config(ref.config, attn_implementation="sdpa", dtype=torch.bfloat16)
    model.load_state_dict(ref.state_dict())                 # what from_pretrained(dtype=bfloat16) gives
    model.eval()

    assert fp32_compute(model) == ref.config.num_hidden_layers
    experts = {id(p) for m in model.modules() if isinstance(m, Gemma4TextExperts) for p in m.parameters()}
    assert all(p.dtype == (torch.bfloat16 if id(p) in experts else torch.float32) for p in model.parameters())

    ids = torch.randint(0, VOCAB, (1, 40), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected, got = ref(input_ids=ids).last_hidden_state, model(input_ids=ids).last_hidden_state
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_fp32_compute_backpropagates_through_the_frozen_experts():
    """P3 trains LoRA in the layers below each expert block, so gradients must pass through the patched experts."""
    ref = tiny_decoder("moe", "sdpa")
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    model = Gemma4TextModel._from_config(ref.config, attn_implementation="sdpa", dtype=torch.bfloat16)
    model.load_state_dict(ref.state_dict())
    fp32_compute(model)

    g = torch.Generator().manual_seed(0)
    x, direction = torch.randn(2, 1, 24, ref.config.hidden_size, generator=g)
    grads = []
    for m in (ref, model):
        e = x.clone().requires_grad_()
        (m(inputs_embeds=e).last_hidden_state * direction).sum().backward()   # not .square(): the final norm fixes that
        grads.append(e.grad)
    torch.testing.assert_close(grads[1], grads[0], rtol=1e-4, atol=1e-4 * grads[0].abs().max().item())
