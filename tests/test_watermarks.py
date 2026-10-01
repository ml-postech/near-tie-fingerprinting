"""Properties of the teachers on a small synthetic vocabulary (CPU)."""
import pytest
import torch

from robust_fp.watermarks import KGWWatermark, NearTieWrapper

V = 512
VOCAB = {f"t{i}": i for i in range(V)}


class Tok:
    eos_token_id, bos_token_id, pad_token_id, unk_token_id = 0, 1, None, None


def make(**kw):
    return KGWWatermark(vocab=VOCAB, tokenizer=Tok(), device="cpu", **kw)


@pytest.fixture(scope="module")
def batch():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(2, V, (3, 12), generator=g)
    logits = torch.randn(3, 12, V, generator=g) * 3.0
    return ids, logits


def test_greenlist_fraction_and_special_tokens():
    wm = make(gamma=0.25, delta=2.0)
    frac = wm.greenlist_masks.float().mean(dim=-1)
    assert torch.all(frac[2:] > 0.20) and torch.all(frac[2:] < 0.30)
    assert not wm.greenlist_masks[:, 0].any() and not wm.greenlist_masks[0, :].any()
    assert not wm.greenlist_masks[:, 1].any() and not wm.greenlist_masks[1, :].any()


def test_dense_kgw_boosts_every_green_token(batch):
    ids, logits = batch
    wm = make(gamma=0.25, delta=2.0)
    diff = wm.watermark_logits_diff(ids, logits)
    green = wm.greenlist_masks[ids]
    assert torch.equal(diff > 0, green)
    assert torch.all(diff[green] == 2.0)
    assert torch.allclose(wm.watermark_logits(ids, logits), logits + diff)
    assert torch.equal(logits, batch[1]), "the input logits must not be modified"


def test_neartie_boosts_green_tokens_inside_the_window(batch):
    ids, logits = batch
    tau = 1.5
    wm = make(gamma=0.25, delta=4.0, tie_restricted=True, tau_gap=tau)
    diff = wm.watermark_logits_diff(ids, logits)
    sl = logits
    in_window = sl > (sl.max(dim=-1, keepdim=True).values - tau)
    green = wm.greenlist_masks[ids]
    assert torch.equal(diff > 0, green & in_window)
    assert (diff > 0).sum() < (make(gamma=0.25, delta=4.0).watermark_logits_diff(ids, logits) > 0).sum()


def test_neartie_requires_tau():
    with pytest.raises(ValueError):
        make(tie_restricted=True)


def test_sweet_gates_on_entropy(batch):
    ids, logits = batch
    wm = make(gamma=0.25, delta=4.0, entropy_restricted=True, entropy_threshold=5.0)
    diff = wm.watermark_logits_diff(ids, logits)
    p = torch.softmax(logits, dim=-1)
    entropy = -(p * p.log()).sum(-1)
    boosted_positions = (diff > 0).any(-1)
    assert torch.equal(boosted_positions, entropy > 5.0)


def test_topk_gates_on_rank(batch):
    ids, logits = batch
    wm = make(gamma=0.25, delta=4.0, topk_restricted=5)
    diff = wm.watermark_logits_diff(ids, logits)
    kth = logits.topk(5, dim=-1).values[..., -1:]
    green = wm.greenlist_masks[ids]
    assert torch.equal(diff > 0, green & (logits >= kth))


def test_morphmark_returns_a_distribution_that_favours_green(batch):
    ids, logits = batch
    wm = make(gamma=0.5, delta=0.0, morphmark=True, morphmark_config=dict(type="exp", k_exp=2.0, p_0=0.05))
    out = wm.watermark_logits(ids, logits)
    p_new = out.exp()
    assert torch.allclose(p_new.sum(-1), torch.ones(3, 12), atol=1e-4)
    green = wm.greenlist_masks[ids]
    p_old = torch.softmax(logits, dim=-1)
    assert torch.all((p_new * green).sum(-1) >= (p_old * green).sum(-1) - 1e-6)


def test_neartie_wrapper_masks_outside_the_support(batch):
    ids, logits = batch
    tau = 1.0
    base = make(gamma=0.25, delta=4.0, entropy_restricted=True, entropy_threshold=0.5)
    wrapped = NearTieWrapper(base, tau)
    diff = wrapped.watermark_logits_diff(ids, logits)
    support = logits > (logits.max(dim=-1, keepdim=True).values - tau)
    assert torch.all(diff[~support] == 0)
    assert torch.equal(diff[support], base.watermark_logits_diff(ids, logits)[support])
    assert wrapped.gate_name == "sweet+neartie"
    assert wrapped.k == base.k and wrapped.gamma == base.gamma


def test_gate_names():
    assert make().gate_name == "kgw"
    assert make(tie_restricted=True, tau_gap=1.0).gate_name == "neartie"
    assert make(entropy_restricted=True).gate_name == "sweet"
    assert make(topk_restricted=5).gate_name == "topk"
    with pytest.raises(ValueError):
        make(tie_restricted=True, tau_gap=1.0, topk_restricted=5)
