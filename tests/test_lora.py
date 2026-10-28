"""LoRA on decoder projections only, plus trainable query-token rows (docs/guide/06 §6.6)."""
import re

import pytest
import torch

from decisivelm.head import Scorer
from decisivelm.infer import QUERY_IDS, DecisiveLM
from decisivelm.lora import LoRALinear, QueryRows, apply_lora, lora_state, load_lora_state
from decisivelm.model import backbone_profile
from tiny_gemma import tiny_decoder, tokenizer

REGEX = backbone_profile("e4b")["lora_target_regex"]
STATE = "Our API started returning 500s 20 minutes ago; we can't process orders."
DEPT = {"type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payment issues", "technical": "Bugs or outages", "sales": "Pricing"}}
URGENT = {"type": "noul", "instructions": "The message conveys urgency"}


def model(variant="dense"):
    return tiny_decoder(variant, "sdpa", vocab=len(tokenizer()))


def dlm_for(backbone, seed=1):
    torch.manual_seed(seed)
    return DecisiveLM(backbone, tokenizer(), Scorer(backbone.config.hidden_size))


def perturb(backbone):
    """Give the LoRA B matrices and query rows non-zero values, as after training."""
    with torch.no_grad():
        for name, p in backbone.named_parameters():
            if name.endswith("lora_B") or name.endswith("query_rows.delta"):
                p.normal_(0.0, 0.05)


@pytest.mark.parametrize("variant", ["dense", "moe"])
def test_lora_wraps_exactly_the_decoder_projections(variant):
    m = model(variant)
    expected = {n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and re.search(REGEX, n)}
    apply_lora(m, REGEX, rank=4, alpha=8, query_ids=list(QUERY_IDS.values()))
    wrapped = {n for n, mod in m.named_modules() if isinstance(mod, LoRALinear)}
    assert wrapped == expected and len(wrapped) > 0


def test_only_adapter_parameters_train():
    m = model()
    apply_lora(m, REGEX, rank=4, alpha=8, query_ids=list(QUERY_IDS.values()))
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(("lora_A", "lora_B", "query_rows.delta")) for n in trainable)


def test_fresh_adapter_leaves_outputs_unchanged():
    m = model()
    dlm = dlm_for(m)
    tree = [dlm.tree_for({"state": STATE, "questions": {"d": DEPT}})]
    _, before = dlm.logits(tree)
    apply_lora(m, REGEX, rank=4, alpha=8, query_ids=list(QUERY_IDS.values()))
    _, after = dlm.logits(tree)
    torch.testing.assert_close(after, before, atol=0, rtol=0)


def test_query_rows_change_only_query_positions():
    m = model()
    rows = QueryRows(m.embed_tokens, list(QUERY_IDS.values()))
    with torch.no_grad():
        rows.delta.fill_(1.0)
    ids = torch.tensor([[2, 100, QUERY_IDS["choice"], 200, QUERY_IDS["noul"]]])
    out = m.embed_tokens(ids)
    rows.remove()
    plain = m.embed_tokens(ids)
    changed = (out - plain).abs().sum(-1)[0]
    assert changed[[0, 1, 3]].eq(0).all() and changed[[2, 4]].gt(0).all()


@pytest.mark.parametrize("variant", ["dense", "moe"])
def test_independence_holds_with_a_trained_adapter(variant):
    m = model(variant)
    apply_lora(m, REGEX, rank=4, alpha=8, query_ids=list(QUERY_IDS.values()))
    perturb(m)
    dlm = dlm_for(m)
    alone = dlm.system_one({"state": STATE, "questions": {"d": DEPT}})["answers"]["d"]["probabilities"]
    both = dlm.system_one({"state": STATE, "questions": {"u": URGENT, "d": DEPT}})["answers"]["d"]["probabilities"]
    assert max(abs(alone[k] - both[k]) for k in alone) < 1e-5


def test_adapter_state_round_trips():
    m1, m2 = model(), model()
    for m in (m1, m2):
        apply_lora(m, REGEX, rank=4, alpha=8, query_ids=list(QUERY_IDS.values()))
    perturb(m1)
    load_lora_state(m2, lora_state(m1))
    d1, d2 = dlm_for(m1), dlm_for(m2)
    req = {"state": STATE, "questions": {"d": DEPT, "u": URGENT}}
    assert d1.system_one(req)["answers"] == d2.system_one(req)["answers"]


def test_training_steps_reduce_the_loss():
    m = model()
    apply_lora(m, REGEX, rank=4, alpha=8, query_ids=list(QUERY_IDS.values()))
    dlm = dlm_for(m)
    trees = [dlm.tree_for({"state": STATE, "questions": {"d": DEPT}})]
    target = torch.tensor([0.0, 1.0, 0.0])
    params = [p for p in m.parameters() if p.requires_grad] + list(dlm.scorer.parameters())
    opt = torch.optim.AdamW(params, lr=1e-2)
    losses = []
    for _ in range(10):
        _, q = dlm.query_states(trees, grad=True)
        loss = -(target * torch.log_softmax(dlm.scorer(q), -1)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_moe_profile_adapts_attention_and_the_dense_mlp_and_freezes_experts_and_router():
    """26B A4B layers run a dense MLP beside 128 fused experts (3-D parameters, not nn.Linear) and a router."""
    m = model("moe")
    apply_lora(m, backbone_profile("26b-a4b")["lora_target_regex"], rank=4, alpha=8,
               query_ids=list(QUERY_IDS.values()))
    wrapped = {n for n, mod in m.named_modules() if isinstance(mod, LoRALinear)}
    assert {n.split(".", 2)[2] for n in wrapped} >= {"self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"}
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert not [n for n in trainable if ".experts." in n or ".router." in n]
