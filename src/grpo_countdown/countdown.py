"""The Countdown task and its (verifiable) reward.

Countdown: given a small set of numbers and a target, write an arithmetic
expression using each number **at most once** that evaluates to the target,
e.g. numbers [3, 7, 11, 2], target 25 -> "3 * 7 + 2 + ... ".

Why this task: the reward is *verifiable* — we can check an answer exactly by
parsing and evaluating it, no learned reward model needed. That makes it ideal
for GRPO/R1-style RL on a small budget.

Security: model output is untrusted text, so we NEVER `eval()` it. Expressions
are parsed with `ast` and evaluated over a strict whitelist of arithmetic nodes.
"""

from __future__ import annotations

import ast
import operator
import random
import re
from dataclasses import dataclass

# Whitelisted binary operators for the safe evaluator.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


@dataclass
class CountdownProblem:
    numbers: list[int]
    target: int


# --------------------------------------------------------------------------- #
# Problem generation (guaranteed solvable)                                     #
# --------------------------------------------------------------------------- #
def generate_problem(
    rng: random.Random, n_numbers: int = 4, max_number: int = 20
) -> CountdownProblem:
    """Generate a solvable Countdown problem.

    We build a random arithmetic combination of the drawn numbers and use its
    value as the target, so a solution is guaranteed to exist (integer, positive).
    Division that isn't exact is simply retried.
    """
    for _ in range(1000):
        numbers = [rng.randint(1, max_number) for _ in range(n_numbers)]
        # random left-to-right combination with random ops
        value = float(numbers[0])
        ok = True
        for x in numbers[1:]:
            op = rng.choice(["+", "-", "*", "/"])
            if op == "+":
                value += x
            elif op == "-":
                value -= x
            elif op == "*":
                value *= x
            else:
                if x == 0 or value % x != 0:
                    ok = False
                    break
                value /= x
        if ok and value == int(value) and 1 <= value <= 1000:
            return CountdownProblem(numbers=numbers, target=int(value))
    # fallback: trivially solvable (sum)
    numbers = [rng.randint(1, max_number) for _ in range(n_numbers)]
    return CountdownProblem(numbers=numbers, target=sum(numbers))


# --------------------------------------------------------------------------- #
# Safe expression evaluation                                                   #
# --------------------------------------------------------------------------- #
def _safe_eval(node) -> float:
    """Evaluate an AST arithmetic expression over the whitelist; raise otherwise."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    # numeric literal (py3.8+: ast.Constant)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    raise ValueError(f"disallowed expression node: {ast.dump(node)}")


def safe_eval_expr(expr: str) -> float:
    """Parse and evaluate an arithmetic expression string safely. Raises on
    anything outside +,-,*,/ , unary +/- and numeric literals."""
    tree = ast.parse(expr.strip(), mode="eval")
    return _safe_eval(tree)


def _numbers_in_expr(expr: str) -> list[int]:
    """Extract integer literals used in the expression (for the 'use each number
    at most once' check)."""
    return [int(tok) for tok in re.findall(r"\d+", expr)]


# --------------------------------------------------------------------------- #
# Answer extraction + reward                                                    #
# --------------------------------------------------------------------------- #
def extract_answer(text: str) -> str | None:
    """Pull the expression from the model output. Convention: the answer is
    wrapped in <answer>...</answer> (R1-style). Returns None if absent."""
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if m is None:
        return None
    return m.group(1).strip()


def compute_reward(
    text: str,
    problem: CountdownProblem,
    format_reward: float = 0.05,
    numbers_reward: float = 0.1,
    tol: float = 1e-6,
) -> float:
    """Graded, verifiable reward — shaped for a soft cold start, but not hackable.

    - 0.0            : no <answer> tag, OR the contents don't parse as arithmetic
                       (garbage and exploits land here — the safe evaluator raises,
                       nothing runs, nothing is rewarded)
    - format_reward  : a *parseable* arithmetic expression in <answer> (a foothold:
                       the model learned the output format), even if it used
                       numbers it shouldn't
    - numbers_reward : ... using only the allowed numbers (each at most once),
                       but the wrong value
    - 1.0            : ... and it evaluates to the target

    Nothing above format_reward is reachable without valid arithmetic, and the
    top reward needs the exact target — so the shaping guides learning without
    being farmable by emitting empty or junk tags.
    """
    answer = extract_answer(text)
    if answer is None:
        return 0.0
    try:
        value = safe_eval_expr(answer)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
        return 0.0  # tag present but not valid arithmetic (incl. any exploit)

    reward = format_reward  # parseable arithmetic present

    used = _numbers_in_expr(answer)
    allowed = list(problem.numbers)
    for n in used:
        if n in allowed:
            allowed.remove(n)
        else:
            return reward  # valid arithmetic, but illegal/reused numbers

    reward = numbers_reward
    if abs(value - problem.target) < tol:
        return 1.0
    return reward


def format_prompt(problem: CountdownProblem) -> str:
    """Render a problem into an instruction prompt (R1-style: think, then answer).

    Includes a tiny worked example so a small instruct model reliably produces
    the <answer> format. This is the *user message*; the training loop wraps it
    in the model's chat template.
    """
    nums = ", ".join(str(n) for n in problem.numbers)
    return (
        "You solve Countdown puzzles. Using the given numbers and the operators "
        "+, -, *, / (each number at most once), write an expression equal to the "
        "target. Think briefly, then put ONLY the final expression inside "
        "<answer> </answer> tags.\n\n"
        "Example:\n"
        "Numbers: [2, 3, 4], target: 10\n"
        "Reasoning: 2 * 3 = 6, and 6 + 4 = 10.\n"
        "<answer>2 * 3 + 4</answer>\n\n"
        f"Numbers: [{nums}], target: {problem.target}\n"
    )
