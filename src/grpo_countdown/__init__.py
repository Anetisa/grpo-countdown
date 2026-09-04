"""grpo-countdown: GRPO (R1-style RL) on the verifiable Countdown task, from scratch."""

from .countdown import (
    CountdownProblem,
    compute_reward,
    extract_answer,
    format_prompt,
    generate_problem,
    safe_eval_expr,
)

__all__ = [
    "CountdownProblem",
    "generate_problem",
    "compute_reward",
    "extract_answer",
    "safe_eval_expr",
    "format_prompt",
]

__version__ = "0.1.0"
