"""Group-relative advantage — the core of GRPO.

PPO needs a learned value network to get a baseline for the advantage. GRPO's
key simplification: for each prompt, sample a *group* of G completions and use
the group's own reward statistics as the baseline. The advantage of completion i
is its reward standardized within its group:

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

Intuition: answers above the group average get a positive push, below-average a
negative one. No critic, no extra network. A scalar advantage per completion is
then broadcast to every token of that completion in the loss.

Degenerate group (all rewards equal, e.g. all wrong or all right): std = 0, so
every advantage is 0 — the group carries no learning signal, which is exactly
right (there's nothing to prefer within it). We handle this without NaNs.
"""

from __future__ import annotations

import torch


def group_relative_advantage(
    rewards: torch.Tensor, eps: float = 1e-6, normalize_std: bool = True
) -> torch.Tensor:
    """Standardize rewards within each group.

    Args:
        rewards: [num_prompts, group_size] — reward of each completion, grouped
                 by prompt.
        eps: numerical floor added to std.
        normalize_std: if False, only mean-center (divide by 1) — some GRPO
                 variants skip the std division; kept as an option.

    Returns:
        advantages: same shape as `rewards`; within each row (group), mean ~= 0.
    """
    assert rewards.dim() == 2, "rewards must be [num_prompts, group_size]"
    rewards = rewards.float()
    mean = rewards.mean(dim=1, keepdim=True)
    centered = rewards - mean
    if not normalize_std:
        return centered
    # population std (unbiased=False); a degenerate group -> std 0 -> adv 0 via eps
    std = rewards.std(dim=1, unbiased=False, keepdim=True)
    return centered / (std + eps)


def broadcast_advantage_to_tokens(
    advantages: torch.Tensor, completion_mask: torch.Tensor
) -> torch.Tensor:
    """Expand a per-completion scalar advantage to per-token, zeroing padding.

    Args:
        advantages: [N] — one scalar per completion (already flattened over the
                    group dimension, N = num_prompts * group_size).
        completion_mask: [N, T] — 1 for real completion tokens, 0 for padding /
                    prompt tokens that shouldn't receive gradient.

    Returns:
        [N, T] advantage per token (advantage where mask==1, else 0).
    """
    assert advantages.dim() == 1 and completion_mask.dim() == 2
    assert advantages.shape[0] == completion_mask.shape[0]
    return advantages[:, None] * completion_mask.float()
