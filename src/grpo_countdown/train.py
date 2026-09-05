"""GRPO training loop.

Ties the pieces together: sample completions -> reward -> group-relative
advantage -> GRPO loss -> optimizer step.

Design for testability: the fiddly, bug-prone parts (padding, completion masks,
the logit/label shift) are pure tensor logic in `build_rollout_batch` and
`sequence_logprobs`, and the update in `grpo_update` — all exercisable on CPU
with a toy model. Only `generate_rollouts` and `train` need `transformers` + a
real model + a GPU, and they import transformers lazily so this module loads on
a CPU-only box.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .advantage import group_relative_advantage
from .countdown import compute_reward, format_prompt, generate_problem
from .loss import compute_token_logprobs, grpo_loss


# --------------------------------------------------------------------------- #
# Rollout bookkeeping (pure logic — no model, fully CPU-testable)             #
# --------------------------------------------------------------------------- #
@dataclass
class RolloutBatch:
    input_ids: torch.Tensor        # [N, T] prompt + completion, left prompt / right pad
    attention_mask: torch.Tensor   # [N, T] 1 for real tokens
    completion_mask: torch.Tensor  # [N, T-1] 1 for completion tokens (shifted frame)


def build_rollout_batch(
    prompt_ids: list[list[int]],
    completion_ids: list[list[int]],
    pad_id: int,
) -> RolloutBatch:
    """Assemble padded tensors + a completion mask aligned to the shifted logprobs.

    Each row is [prompt tokens][completion tokens][pad]. Log-probs are computed on
    the causal shift (logits at position j predict token j+1), so the completion
    mask lives in the [T-1] shifted frame: index j is 1 iff token j+1 is a real
    completion token.
    """
    assert len(prompt_ids) == len(completion_ids)
    seqs = [p + c for p, c in zip(prompt_ids, completion_ids)]
    T = max(len(s) for s in seqs)
    N = len(seqs)

    input_ids = torch.full((N, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((N, T), dtype=torch.long)
    completion_mask = torch.zeros((N, T - 1), dtype=torch.long)

    for i, (p, c) in enumerate(zip(prompt_ids, completion_ids)):
        s = p + c
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attention_mask[i, : len(s)] = 1
        # completion token at absolute position q (len(p) <= q < len(s)) has its
        # logprob at shifted index q-1
        start = max(len(p) - 1, 0)
        end = len(s) - 1
        completion_mask[i, start:end] = 1

    return RolloutBatch(input_ids, attention_mask, completion_mask)


def sequence_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token logprobs on the causal shift: logits[:, :-1] predict ids[:, 1:].
    Returns [N, T-1]."""
    return compute_token_logprobs(logits[:, :-1, :], input_ids[:, 1:])


def _forward_logprobs(model, input_ids, attention_mask):
    """Run a model and return shifted per-token logprobs. Accepts a transformers
    model (returns .logits) or a plain callable returning logits."""
    out = model(input_ids, attention_mask=attention_mask)
    logits = out.logits if hasattr(out, "logits") else out
    return sequence_logprobs(logits, input_ids)


# --------------------------------------------------------------------------- #
# The GRPO update (CPU-testable with a toy model)                             #
# --------------------------------------------------------------------------- #
def grpo_update(
    model,
    ref_model,
    batch: RolloutBatch,
    advantages: torch.Tensor,     # [N]
    old_logprobs: torch.Tensor,   # [N, T-1] detached, from rollout time
    optimizer,
    beta: float = 0.04,
    clip_eps: float = 0.2,
    max_grad_norm: float = 1.0,
):
    """One optimizer step of GRPO. Returns the metrics dict."""
    logprobs = _forward_logprobs(model, batch.input_ids, batch.attention_mask)
    with torch.no_grad():
        ref_logprobs = _forward_logprobs(ref_model, batch.input_ids, batch.attention_mask)

    loss, metrics = grpo_loss(
        logprobs, old_logprobs, ref_logprobs, advantages,
        batch.completion_mask, beta=beta, clip_eps=clip_eps,
    )
    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return metrics


# --------------------------------------------------------------------------- #
# Rollout generation + full training loop (needs transformers + GPU)          #
# --------------------------------------------------------------------------- #
def generate_rollouts(
    model, tokenizer, problems, group_size, max_new_tokens=256,
    temperature=1.0, top_p=1.0, device="cuda",
):
    """Sample `group_size` completions per problem and score them.

    Returns (batch, advantages, old_logprobs, rewards, texts). Uses the model's
    `.generate`; requires a real model. The masking/padding is delegated to
    `build_rollout_batch` (unit-tested separately).
    """
    model.eval()
    prompt_ids, completion_ids, rewards, texts, group_of = [], [], [], [], []
    for gi, prob in enumerate(problems):
        user_msg = format_prompt(prob)
        # Instruct models need their chat template to follow instructions/format.
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True,
            )
        else:
            prompt = user_msg
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        plen = enc.input_ids.shape[1]
        gen = model.generate(
            **enc, do_sample=True, temperature=temperature, top_p=top_p,
            num_return_sequences=group_size, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        for row in gen:
            p = row[:plen].tolist()
            c = row[plen:].tolist()
            prompt_ids.append(p)
            completion_ids.append(c)
            text = tokenizer.decode(row[plen:], skip_special_tokens=True)
            texts.append(text)
            rewards.append(compute_reward(text, prob))
            group_of.append(gi)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    batch = build_rollout_batch(prompt_ids, completion_ids, pad_id)
    batch = RolloutBatch(
        batch.input_ids.to(device), batch.attention_mask.to(device),
        batch.completion_mask.to(device),
    )

    rewards_t = torch.tensor(rewards, device=device).view(len(problems), group_size)
    advantages = group_relative_advantage(rewards_t).view(-1)

    model.train()
    with torch.no_grad():
        old_logprobs = _forward_logprobs(model, batch.input_ids, batch.attention_mask)
    return batch, advantages, old_logprobs, rewards_t, texts


def train(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    iterations: int = 200,
    prompts_per_iter: int = 4,
    group_size: int = 8,
    inner_epochs: int = 1,
    lr: float = 1e-6,
    beta: float = 0.04,
    clip_eps: float = 0.2,
    max_new_tokens: int = 256,
    n_numbers: int = 4,
    seed: int = 0,
    device: str = "cuda",
    log_every: int = 10,
    log_file: str | None = None,
):
    """Full GRPO training on Countdown. Needs transformers + a GPU.

    Kept intentionally simple/readable over maximally efficient. For memory, the
    reference model is a frozen copy; on tight budgets swap it for a LoRA-disabled
    pass of the same model. Every `log_every` iters it prints a couple of sample
    completions so you can *see* what the policy is producing; metrics are also
    appended to `log_file` (JSONL) for plotting.
    """
    import json
    from copy import deepcopy

    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import

    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
    ref_model = deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    logf = open(log_file, "w") if log_file else None
    for it in range(iterations):
        problems = [generate_problem(rng, n_numbers=n_numbers) for _ in range(prompts_per_iter)]
        batch, adv, old_lp, rewards, texts = generate_rollouts(
            model, tok, problems, group_size, max_new_tokens=max_new_tokens, device=device,
        )
        for _ in range(inner_epochs):
            metrics = grpo_update(model, ref_model, batch, adv, old_lp, optim,
                                  beta=beta, clip_eps=clip_eps)
        acc = (rewards >= 1.0).float().mean().item()
        rec = {"iter": it, "acc": acc, "reward": rewards.mean().item(), **metrics}
        print(f"iter {it:4d} | acc {acc:.3f} | reward {rec['reward']:.3f} "
              f"| kl {metrics['kl']:.4f} | loss {metrics['loss']:.4f}")
        if logf:
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

        # visibility: peek at what the policy is actually generating
        if it % log_every == 0:
            for t in texts[:2]:
                snippet = t.replace("\n", " ")[:160]
                print(f"    sample: {snippet!r}")

    if logf:
        logf.close()
    return model


def main():
    import argparse

    p = argparse.ArgumentParser(description="GRPO training on Countdown")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--prompts-per-iter", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--inner-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--n-numbers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--log-file", default="results_train.jsonl")
    a = p.parse_args()
    train(
        model_name=a.model, iterations=a.iterations, prompts_per_iter=a.prompts_per_iter,
        group_size=a.group_size, inner_epochs=a.inner_epochs, lr=a.lr, beta=a.beta,
        clip_eps=a.clip_eps, max_new_tokens=a.max_new_tokens, n_numbers=a.n_numbers,
        seed=a.seed, device=a.device, log_every=a.log_every, log_file=a.log_file,
    )


if __name__ == "__main__":
    main()
