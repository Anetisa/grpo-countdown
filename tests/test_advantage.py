"""Tests for group-relative advantage (the GRPO baseline)."""

import torch

from grpo_countdown.advantage import (
    broadcast_advantage_to_tokens,
    group_relative_advantage,
)


def test_advantage_is_mean_centered_per_group():
    """Within each group the advantages must sum (and average) to ~0."""
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    adv = group_relative_advantage(rewards)
    torch.testing.assert_close(adv.mean(dim=1), torch.zeros(2), atol=1e-5, rtol=0)


def test_higher_reward_gets_higher_advantage():
    rewards = torch.tensor([[0.0, 0.1, 1.0, 0.5]])
    adv = group_relative_advantage(rewards)[0]
    # ordering of advantages must follow ordering of rewards
    assert torch.argmax(adv).item() == 2   # reward 1.0
    assert torch.argmin(adv).item() == 0   # reward 0.0


def test_degenerate_group_gives_zero_no_nan():
    """All-equal rewards (all wrong or all right) -> zero advantage, no NaN."""
    for val in [0.0, 1.0, 0.1]:
        rewards = torch.full((3, 4), val)
        adv = group_relative_advantage(rewards)
        assert torch.isfinite(adv).all()
        torch.testing.assert_close(adv, torch.zeros_like(adv), atol=1e-3, rtol=0)


def test_unit_variance_after_normalization():
    """With std normalization, each group's advantages have ~unit std."""
    torch.manual_seed(0)
    rewards = torch.randn(5, 8)
    adv = group_relative_advantage(rewards)
    # population std of advantages per group ~= 1 (eps makes it slightly under)
    std = adv.std(dim=1, unbiased=False)
    torch.testing.assert_close(std, torch.ones(5), atol=1e-3, rtol=0)


def test_groups_normalized_independently():
    """A group with big rewards and one with tiny rewards should both standardize."""
    rewards = torch.tensor([[100.0, 0.0], [0.01, 0.0]])
    adv = group_relative_advantage(rewards)
    # both groups: first element positive, second negative, symmetric
    torch.testing.assert_close(adv[0], torch.tensor([1.0, -1.0]), atol=1e-3, rtol=0)
    torch.testing.assert_close(adv[1], torch.tensor([1.0, -1.0]), atol=1e-3, rtol=0)


def test_mean_center_only_option():
    rewards = torch.tensor([[2.0, 0.0]])
    adv = group_relative_advantage(rewards, normalize_std=False)
    torch.testing.assert_close(adv[0], torch.tensor([1.0, -1.0]), atol=1e-6, rtol=0)


def test_broadcast_to_tokens_respects_mask():
    adv = torch.tensor([2.0, -1.0])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])   # completion lengths 2 and 1
    tok = broadcast_advantage_to_tokens(adv, mask)
    expected = torch.tensor([[2.0, 2.0, 0.0], [-1.0, 0.0, 0.0]])
    torch.testing.assert_close(tok, expected, atol=1e-6, rtol=0)
