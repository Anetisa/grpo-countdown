"""The GRPO loss: clipped policy-gradient with a KL penalty to a reference.

Per token, GRPO maximizes

    min( ρ·A , clip(ρ, 1-ε, 1+ε)·A )  -  β·KL(π_θ ‖ π_ref)

where
  ρ = exp(logπ_θ(token) − logπ_old(token))     # importance ratio
  A = the completion's group-relative advantage  (same for all its tokens)
  ε = PPO clip range, β = KL coefficient.

The advantage tells each completion which way to move; the clip keeps the update
from moving too far when the policy has already shifted (this is what lets you
take several gradient steps per batch of rollouts); the KL term keeps the policy
close to the frozen reference so it doesn't collapse or forget.

KL estimator: the low-variance, non-negative **k3** estimator (Schulman),
    KL ≈ exp(ref − cur) − (ref − cur) − 1,
computed per token — the same one DeepSeek/TRL use.

Reduction: per-sequence token-mean (divide by completion length), then mean over
completions — the original GRPO formulation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_token_logprobs(
    logits: torch.Tensor, target_ids: torch.Tensor, chunk_size: int = 8
) -> torch.Tensor:
    """Log-prob of each taken token, computed memory-efficiently.

    Args:
        logits: [N, T, V] — model logits predicting each position's token.
        target_ids: [N, T] — the actually-taken token id at each position.
        chunk_size: process this many rows at a time so we never materialize a
            full [N, T, V] fp32 tensor. With LLM vocabularies (~150k) the fp32
            cast of the whole batch is many GB and OOMs even large GPUs; chunking
            caps peak memory to chunk_size rows while giving identical values.
    Returns:
        [N, T] log π(target_id) under a softmax over V.
    """
    N = logits.shape[0]
    outs = []
    for i in range(0, N, chunk_size):
        lg = logits[i : i + chunk_size].float()
        lp = F.log_softmax(lg, dim=-1)
        outs.append(
            torch.gather(lp, dim=-1, index=target_ids[i : i + chunk_size].unsqueeze(-1)).squeeze(-1)
        )
    return torch.cat(outs, dim=0)


def _masked_seq_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence mean over masked tokens, then mean over sequences."""
    mask = mask.float()
    tok_per_seq = mask.sum(dim=1).clamp(min=1.0)
    seq_mean = (x * mask).sum(dim=1) / tok_per_seq
    return seq_mean.mean()


def grpo_loss(
    logprobs: torch.Tensor,       # [N, T] current policy, requires grad
    old_logprobs: torch.Tensor,   # [N, T] policy at rollout time (detached)
    ref_logprobs: torch.Tensor,   # [N, T] reference policy (detached)
    advantages: torch.Tensor,     # [N] per-completion advantage
    completion_mask: torch.Tensor,  # [N, T] 1 for real completion tokens
    beta: float = 0.04,
    clip_eps: float = 0.2,
):
    """Return (scalar loss, metrics dict). Minimizing this performs GRPO."""
    old_logprobs = old_logprobs.detach()
    ref_logprobs = ref_logprobs.detach()
    adv = advantages[:, None]                          # broadcast to tokens

    ratio = torch.exp(logprobs - old_logprobs)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy = torch.minimum(unclipped, clipped)         # PPO pessimistic bound

    # k3 KL estimator (>= 0), per token
    diff = ref_logprobs - logprobs
    kl = torch.exp(diff) - diff - 1.0

    per_token = policy - beta * kl
    loss = -_masked_seq_mean(per_token, completion_mask)

    with torch.no_grad():
        mask = completion_mask.float()
        clip_frac = _masked_seq_mean((unclipped != clipped).float(), completion_mask)
        metrics = {
            "loss": loss.item(),
            "kl": _masked_seq_mean(kl, completion_mask).item(),
            "ratio": _masked_seq_mean(ratio, completion_mask).item(),
            "clip_frac": clip_frac.item(),
            "adv_mean": advantages.mean().item(),
        }
    return loss, metrics
