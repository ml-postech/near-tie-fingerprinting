# Copyright (c) 2026 Jeongyeon Hwang, Anshul Nasery, Sewoong Oh and Jungseul Ok
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

"""Properties of the teachers on a small synthetic vocabulary (CPU)."""
import pytest
import torch

from robust_fp.watermarks import (KGW, EntropySelector, GreenList, MorphMark, NearTieSelector, NearTieWrapper,
                                  TopKSelector)

V = 512
VOCAB = {f"t{i}": i for i in range(V)}


class Tok:
    eos_token_id, bos_token_id, pad_token_id, unk_token_id = 0, 1, None, None


@pytest.fixture(scope="module")
def greenlist():
    return GreenList(VOCAB, gamma=0.25, tokenizer=Tok(), device="cpu")


@pytest.fixture(scope="module")
def batch():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(2, V, (3, 12), generator=g)
    logits = torch.randn(3, 12, V, generator=g) * 3.0
    return ids, logits


def test_greenlist_fraction_and_special_tokens(greenlist):
    frac = greenlist.masks.float().mean(dim=-1)
    assert torch.all(frac[2:] > 0.20) and torch.all(frac[2:] < 0.30)
    for t in (0, 1):
        assert not greenlist.masks[:, t].any() and not greenlist.masks[t, :].any()
    assert greenlist.k == 1 and greenlist.gamma == 0.25


def test_dense_kgw_boosts_every_green_token(greenlist, batch):
    ids, logits = batch
    wm = KGW(greenlist, delta=2.0)
    diff = wm.watermark_logits_diff(ids, logits)
    green = greenlist[ids]                      # with k = 1 the green list of position t is seeded by token t
    assert torch.equal(diff > 0, green)
    assert torch.all(diff[green] == 2.0)
    assert torch.allclose(wm.watermark_logits(ids, logits), logits + diff)
    assert torch.equal(logits, batch[1]), "the input logits must not be modified"
    assert wm.name == "kgw" and not wm.restricts_positions and wm.eligibility(logits[0]) is None


def test_neartie_boosts_green_tokens_inside_the_window(greenlist, batch):
    ids, logits = batch
    tau = 1.5
    wm = KGW(greenlist, delta=4.0, selector=NearTieSelector(tau))
    diff = wm.watermark_logits_diff(ids, logits)
    in_window = logits > (logits.max(dim=-1, keepdim=True).values - tau)
    assert torch.equal(diff > 0, greenlist[ids] & in_window)
    assert (diff > 0).sum() < (KGW(greenlist, 4.0).watermark_logits_diff(ids, logits) > 0).sum()
    # detection scores the positions whose top-1/top-2 gap is below tau
    logp = torch.log_softmax(logits[0], dim=-1)
    top2 = logits[0].topk(2, dim=-1).values
    assert torch.equal(wm.eligibility(logp), (top2[:, 0] - top2[:, 1]) < tau)
    assert wm.name == "neartie" and wm.restricts_positions


def test_neartie_requires_tau():
    with pytest.raises(ValueError):
        NearTieSelector(None)


def test_sweet_gates_on_entropy(greenlist, batch):
    ids, logits = batch
    wm = KGW(greenlist, delta=4.0, selector=EntropySelector(5.0))
    diff = wm.watermark_logits_diff(ids, logits)
    p = torch.softmax(logits, dim=-1)
    entropy = -(p * p.log()).sum(-1)
    assert torch.equal((diff > 0).any(-1), entropy > 5.0)
    assert torch.equal(wm.eligibility(torch.log_softmax(logits[0], -1)), entropy[0] > 5.0)
    assert wm.name == "sweet"


def test_topk_gates_on_rank(greenlist, batch):
    ids, logits = batch
    wm = KGW(greenlist, delta=4.0, selector=TopKSelector(5))
    diff = wm.watermark_logits_diff(ids, logits)
    kth = logits.topk(5, dim=-1).values[..., -1:]
    assert torch.equal(diff > 0, greenlist[ids] & (logits >= kth))
    assert wm.name == "topk" and not wm.restricts_positions


def test_morphmark_returns_a_distribution_that_favours_green(greenlist, batch):
    ids, logits = batch
    wm = MorphMark(greenlist, variant="exp", k_exp=2.0, p_0=0.05)
    out = wm.watermark_logits(ids, logits)
    p_new = out.exp()
    assert torch.allclose(p_new.sum(-1), torch.ones(3, 12), atol=1e-4)
    green = greenlist[ids]
    p_old = torch.softmax(logits, dim=-1)
    assert torch.all((p_new * green).sum(-1) >= (p_old * green).sum(-1) - 1e-6)
    assert torch.allclose(wm.watermark_logits_diff(ids, logits), out - logits, atol=1e-5)
    assert wm.name == "morphmark" and not wm.restricts_positions


def test_neartie_wrapper_masks_outside_the_support(greenlist, batch):
    ids, logits = batch
    tau = 1.0
    base = KGW(greenlist, delta=4.0, selector=EntropySelector(0.5))
    wrapped = NearTieWrapper(base, tau)
    diff = wrapped.watermark_logits_diff(ids, logits)
    support = logits > (logits.max(dim=-1, keepdim=True).values - tau)
    assert torch.all(diff[~support] == 0)
    assert torch.equal(diff[support], base.watermark_logits_diff(ids, logits)[support])
    assert wrapped.name == "sweet+neartie" and wrapped.restricts_positions
    logp = torch.log_softmax(logits[0], -1)
    assert torch.equal(wrapped.eligibility(logp), base.eligibility(logp) & NearTieSelector(tau).eligibility(logp))
    assert wrapped.greenlist is greenlist


def test_neartie_wrapper_on_morphmark_keeps_a_distribution(greenlist, batch):
    ids, logits = batch
    wrapped = NearTieWrapper(MorphMark(greenlist, k_exp=2.0, p_0=0.05), 1.5)
    out = wrapped.watermark_logits(ids, logits)
    assert torch.allclose(torch.softmax(out, -1).sum(-1), torch.ones(3, 12), atol=1e-4)
    support = logits > (logits.max(dim=-1, keepdim=True).values - 1.5)
    diff = wrapped.watermark_logits_diff(ids, logits)
    assert torch.all(diff[~support] <= 1e-5)                  # probability is only ever added to near-tie tokens
    assert wrapped.name == "morphmark+neartie"


def test_a_custom_selector_plugs_in(greenlist, batch):
    """A new biasing rule is one class: here, green tokens with an even id."""
    from robust_fp.watermarks import TokenSelector

    class EvenSelector(TokenSelector):
        name = "even"

        def select(self, logits, green):
            even = (torch.arange(green.shape[-1]) % 2 == 0).to(green.device)
            return green & even

    ids, logits = batch
    wm = KGW(greenlist, delta=1.0, selector=EvenSelector())
    diff = wm.watermark_logits_diff(ids, logits)
    assert wm.name == "even"
    assert torch.equal(diff > 0, greenlist[ids] & (torch.arange(V) % 2 == 0))


def test_padded_logit_width_is_handled(greenlist, batch):
    """Models whose logit width exceeds the tokenizer vocabulary (Qwen) pad the width with red columns."""
    ids, logits = batch
    wide = torch.cat([logits, torch.full((3, 12, 16), -30.0)], dim=-1)
    for wm in (KGW(greenlist, 2.0), KGW(greenlist, 4.0, NearTieSelector(1.5)), MorphMark(greenlist, k_exp=2.0, p_0=0.05),
               NearTieWrapper(MorphMark(greenlist, k_exp=2.0, p_0=0.05), 1.5)):
        diff = wm.watermark_logits_diff(ids, wide)
        assert diff.shape == wide.shape
        assert torch.all(diff[..., V:] <= 1e-5), f"{wm.name}: padding columns must never gain probability"
        assert torch.allclose(wm.watermark_logits(ids, wide), wide + diff, atol=1e-5)


def test_morphmark_variants_and_threshold(greenlist, batch):
    ids, logits = batch
    for variant in ("exp", "linear", "log"):
        wm = MorphMark(greenlist, variant=variant, p_0=0.0)
        p_new = torch.softmax(wm.watermark_logits(ids, logits), dim=-1)
        assert torch.all((p_new * greenlist[ids]).sum(-1) >= (torch.softmax(logits, -1) * greenlist[ids]).sum(-1) - 1e-6)
    with pytest.raises(ValueError):
        MorphMark(greenlist, variant="quadratic")
    # below p_0 nothing is redistributed: the output is the base distribution (as log-probabilities)
    assert torch.allclose(torch.softmax(MorphMark(greenlist, p_0=1.0).watermark_logits(ids, logits), -1),
                          torch.softmax(logits, -1), atol=1e-5)


def test_topk_requires_a_positive_k():
    with pytest.raises(ValueError):
        TopKSelector(0)


def test_neartie_wrapper_on_morphmark_restricts_the_redistribution(greenlist, batch):
    ids, logits = batch
    tau = 1.5
    base = MorphMark(greenlist, k_exp=2.0, p_0=0.05)
    wrapped = NearTieWrapper(base, tau)
    p_old = torch.softmax(logits, -1)
    p_full = torch.softmax(base.watermark_logits(ids, logits), -1)
    p_new = torch.softmax(wrapped.watermark_logits(ids, logits), -1)
    support = logits > (logits.max(dim=-1, keepdim=True).values - tau)
    gained = p_new > p_old + 1e-6
    assert torch.all(support[gained]), "only near-tie tokens gain probability"
    assert torch.all(greenlist[ids][gained]), "only green tokens gain probability"
    # green tokens inside the support keep MorphMark's increase; everything outside is frozen or loses mass
    inside = support & greenlist[ids] & (p_full > p_old)
    assert torch.allclose(p_new[inside], p_full[inside], atol=1e-4)
    assert torch.all(p_new[~support] <= p_old[~support] + 1e-6)
