# grpo-countdown

**GRPO (the R1-style RL algorithm) implemented from scratch, trained on the verifiable Countdown task.**

Large reasoning models like DeepSeek-R1 are trained with **GRPO** (Group Relative
Policy Optimization): sample a group of answers per prompt, score each with a
reward, and push the policy toward the above-average answers — no value network,
no learned reward model. This repo implements that loop from scratch on
**Countdown** (reach a target number from a few given numbers using arithmetic),
a task whose reward is *verifiable*: you can check an answer exactly by evaluating
it. A small model (0.5–1.5B) learns to reason its way to correct expressions.

> Companion to [triton-llm-kernels](https://github.com/Anetisa/triton-llm-kernels).
> Built the same way: every component is validated on CPU with tiny tensors/models
> before spending a minute of GPU time — only the final training run needs a GPU.

## Status

Built incrementally; each piece is tested on CPU as it lands.

- [x] **Countdown task + verifiable reward** — safe (`ast`-based, no `eval`) reward,
      guaranteed-solvable problem generation, exploit tests
- [ ] **Group-relative advantage** — per-group reward normalization (the GRPO core)
- [ ] **GRPO loss** — advantage-weighted log-probs + KL to a reference policy
- [ ] **Rollout + training loop** — sample G completions, score, update
- [ ] **Training run** on a 0.5–1.5B model, accuracy curve on Countdown
- [ ] Write-up: the math of GRPO and what the reward/advantage do

## Why Countdown

The reward is exact and cheap: parse the model's expression, check it uses the
allowed numbers and evaluates to the target. No reward model to train, no human
labels — the perfect setting to study RL on reasoning at small scale (à la
TinyZero). Because model output is untrusted, expressions are evaluated over a
strict arithmetic whitelist with `ast`, never `eval`.

## Quickstart (the reward, so far)

```python
import random
from grpo_countdown import generate_problem, format_prompt, compute_reward

prob = generate_problem(random.Random(0))     # numbers + target, guaranteed solvable
print(format_prompt(prob))                     # the instruction shown to the model

# a model completion is scored 0.0 / 0.1 (format) / 1.0 (correct):
compute_reward("<answer>11 + 7 * 2</answer>", prob)
```

## Install

```bash
pip install -e ".[dev]"     # core + tests
pip install -e ".[train]"   # adds transformers/datasets for real training
```

## Test

```bash
make test    # runs on CPU; no GPU needed for the logic
```

## References

- Shao et al., *DeepSeekMath* (2024) — introduces GRPO
- DeepSeek-AI, *DeepSeek-R1* (2025)
- TinyZero — minimal Countdown RL reproduction

## License

MIT — see [LICENSE](LICENSE).
