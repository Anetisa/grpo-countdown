# GRPO on Countdown: the math

## The setup

We have a policy (a language model) `π_θ` and a task where answers are
**verifiable**: given a Countdown problem, we can check an answer exactly. RL
turns that reward signal into gradient updates that make the model reason its way
to correct answers more often.

## Why GRPO (vs PPO)

PPO estimates, for each token, how much better an action was than "average" — the
**advantage** — using a separate **value network** (a critic) trained alongside
the policy. That critic doubles the memory and adds its own instability.

**GRPO** removes the critic. For each prompt it samples a **group** of `G`
completions and uses the group's own rewards as the baseline:

    A_i = (r_i − mean(r_group)) / (std(r_group) + ε)

An answer better than its group's average gets a positive advantage, worse gets
negative. That's the whole baseline — no critic, no extra network. This is what
makes GRPO cheap enough to study on a single GPU.

## The objective

Per completion token, GRPO maximizes the PPO-clipped, KL-regularized objective:

    min( ρ·A , clip(ρ, 1−ε, 1+ε)·A )  −  β·KL(π_θ ‖ π_ref)

- `ρ = exp(logπ_θ − logπ_old)` — how much more/less likely this token is now
  than at rollout time. The **clip** stops a single update from moving too far,
  which is what lets you take several gradient steps per batch of rollouts.
- `A` — the group-relative advantage above, the same scalar for every token in
  the completion, broadcast across its tokens.
- `β·KL` — a leash to a frozen **reference** policy so the model improves at the
  task without drifting into gibberish or forgetting its pretraining. The KL uses
  the low-variance, non-negative **k3** estimator
  `exp(ref−cur) − (ref−cur) − 1`.

## The loop

1. Draw a batch of Countdown problems.
2. For each, sample `G` completions from the current policy.
3. Score every completion with the verifiable reward (0 / format / correct).
4. Standardize rewards within each group → advantages.
5. Compute per-token log-probs under the current, old, and reference policies.
6. GRPO loss → backprop → optimizer step. (Optionally repeat 5–6 a few times on
   the same rollouts, which is where the clip earns its keep.)

## The reward

Exact and cheap, so it needs no learned reward model:

- `0.0` — no `<answer>` tag, unparseable, or uses numbers not on offer
- `0.1` — well-formed answer that evaluates but misses the target (a format
  shaping signal so early learning isn't all-or-nothing)
- `1.0` — evaluates to the target using each allowed number at most once

Because model output is untrusted, expressions are parsed with `ast` and
evaluated over a strict arithmetic whitelist — never `eval`.

## What's tested on CPU vs GPU

Everything except the training run is validated on CPU with tiny tensors and a
toy model: the reward (incl. exploit/security cases), the advantage (incl.
degenerate groups), the loss (incl. the gradient-direction guarantee), and the
loop plumbing (padding, masks, the logit/label shift, a real update that makes a
toy model prefer the high-advantage completion). Only the final run — a real
0.5–1.5B model learning Countdown — needs a GPU.
