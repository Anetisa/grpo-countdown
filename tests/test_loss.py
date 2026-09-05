"""Tests for the GRPO loss.

Covers the algebra (ratio, KL, clipping, masking) and — most importantly — the
behavioral property that a gradient step raises the log-prob of high-advantage
completions and lowers it for low-advantage ones. All on CPU.
"""

import torch

from grpo_countdown.loss import compute_token_logprobs, grpo_loss


def test_token_logprobs_match_manual_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    ids = torch.randint(0, 5, (2, 3))
    lp = compute_token_logprobs(logits, ids)
    manual = torch.log_softmax(logits, dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(lp, manual, atol=1e-6, rtol=0)


def test_ratio_one_and_kl_zero_at_init():
    """If current == old == ref, ratio=1, kl=0, loss = -mean(advantage)."""
    N, T = 3, 4
    lp = torch.randn(N, T)
    adv = torch.tensor([1.0, 0.0, -1.0])
    mask = torch.ones(N, T)
    loss, m = grpo_loss(lp, lp.clone(), lp.clone(), adv, mask, beta=0.04)
    assert abs(m["ratio"] - 1.0) < 1e-5
    assert abs(m["kl"]) < 1e-6
    # loss = -mean over seqs of (1 * adv) = -mean(adv) = 0 here
    assert abs(loss.item() - (-adv.mean().item())) < 1e-5


def test_kl_positive_when_policy_differs_from_ref():
    N, T = 2, 3
    cur = torch.zeros(N, T)
    ref = torch.full((N, T), -1.0)   # different from current
    adv = torch.zeros(N)             # isolate the KL term
    mask = torch.ones(N, T)
    _, m = grpo_loss(cur, cur.clone(), ref, adv, mask, beta=0.04)
    assert m["kl"] > 0.0


def test_masking_ignores_padding():
    N, T = 1, 4
    lp = torch.randn(N, T)
    adv = torch.tensor([1.0])
    full = torch.ones(N, T)
    half = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    # different padded logprobs must not change the loss when masked out
    lp_pad = lp.clone()
    lp_pad[0, 2:] += 100.0
    loss_a, _ = grpo_loss(lp, lp.clone(), lp.clone(), adv, half)
    loss_b, _ = grpo_loss(lp_pad, lp.clone(), lp.clone(), adv, half)
    torch.testing.assert_close(loss_a, loss_b, atol=1e-5, rtol=0)


def test_clipping_engages_for_large_ratio_positive_adv():
    """With positive advantage and a big policy shift up, the clip should bind."""
    N, T = 1, 3
    old = torch.zeros(N, T)
    cur = torch.full((N, T), 1.0)     # ratio = e^1 ≈ 2.7 >> 1+eps
    ref = torch.zeros(N, T)
    adv = torch.tensor([1.0])
    mask = torch.ones(N, T)
    _, m = grpo_loss(cur, old, ref, adv, mask, beta=0.0, clip_eps=0.2)
    assert m["clip_frac"] > 0.99      # essentially all tokens clipped


def test_gradient_raises_good_lowers_bad():
    """The core behavioral guarantee: one step increases the taken-token logprob
    of the positive-advantage completion and decreases the negative one."""
    torch.manual_seed(0)
    N, T, V = 2, 5, 16
    logits = torch.randn(N, T, V, requires_grad=True)
    ids = torch.randint(0, V, (N, T))
    mask = torch.ones(N, T)
    adv = torch.tensor([1.0, -1.0])   # seq0 good, seq1 bad

    lp0 = compute_token_logprobs(logits, ids)
    # at init old = ref = current (detached)
    loss, _ = grpo_loss(lp0, lp0.detach(), lp0.detach(), adv, mask, beta=0.0)
    loss.backward()

    with torch.no_grad():
        new_logits = logits - 0.5 * logits.grad   # one SGD step
    lp1 = compute_token_logprobs(new_logits, ids)

    good_delta = (lp1[0] - lp0[0].detach()).mean().item()
    bad_delta = (lp1[1] - lp0[1].detach()).mean().item()
    assert good_delta > 0, f"good completion logprob should rise, got {good_delta}"
    assert bad_delta < 0, f"bad completion logprob should fall, got {bad_delta}"


def test_kl_has_zero_gradient_at_reference():
    """k3 KL is minimized (grad 0) when the policy equals the reference, so at
    init the beta term adds no gradient regardless of its size."""
    torch.manual_seed(0)
    N, T, V = 2, 4, 16
    ids = torch.randint(0, V, (N, T))
    mask = torch.ones(N, T)
    adv = torch.tensor([1.0, -1.0])

    def step_delta(beta):
        logits = torch.randn(N, T, V, generator=torch.Generator().manual_seed(1),
                             requires_grad=True)
        lp0 = compute_token_logprobs(logits, ids)
        ref = lp0.detach()
        loss, _ = grpo_loss(lp0, lp0.detach(), ref, adv, mask, beta=beta)
        loss.backward()
        return logits.grad.norm().item()

    # with adv fixed, KL term adds no gradient at init (cur==ref) -> grads equal;
    # so instead check kl term contributes 0 grad at init but > 0 loss curvature.
    g0 = step_delta(0.0)
    gb = step_delta(0.5)
    # at init cur==ref so kl grad is zero; gradients should match (sanity, not drift)
    assert abs(g0 - gb) < 1e-4


def test_chunked_logprobs_match_single_chunk():
    """Memory-efficient chunking must give identical values to one big pass."""
    torch.manual_seed(0)
    N, T, V = 20, 6, 128   # N > chunk_size to exercise chunking
    logits = torch.randn(N, T, V)
    ids = torch.randint(0, V, (N, T))
    a = compute_token_logprobs(logits, ids, chunk_size=4)
    b = compute_token_logprobs(logits, ids, chunk_size=N)  # single chunk
    torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


def test_chunked_logprobs_preserve_gradients():
    """Gradients must still flow through the chunked path (current policy needs them)."""
    torch.manual_seed(0)
    N, T, V = 10, 4, 64
    logits = torch.randn(N, T, V, requires_grad=True)
    ids = torch.randint(0, V, (N, T))
    lp = compute_token_logprobs(logits, ids, chunk_size=3)
    lp.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
