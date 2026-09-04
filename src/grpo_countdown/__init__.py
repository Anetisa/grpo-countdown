"""grpo-countdown: GRPO (R1-style RL) on the verifiable Countdown task, from scratch."""

from .countdown import (
    CountdownProblem,
    compute_reward,
    extract_answer,
    format_prompt,
    generate_problem,
    safe_eval_expr,
)
from .advantage import (
    broadcast_advantage_to_tokens,
    group_relative_advantage,
)
from .loss import (
    compute_token_logprobs,
    grpo_loss,
)
from .train import (
    build_rollout_batch,
    grpo_update,
    sequence_logprobs,
)

__all__ = [
    "CountdownProblem",
    "generate_problem",
    "compute_reward",
    "extract_answer",
    "safe_eval_expr",
    "format_prompt",
    "group_relative_advantage",
    "broadcast_advantage_to_tokens",
    "grpo_loss",
    "compute_token_logprobs",
    "build_rollout_batch",
    "sequence_logprobs",
    "grpo_update",
]

__version__ = "0.1.0"
