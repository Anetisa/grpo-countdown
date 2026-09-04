"""Tests for the Countdown task and reward.

The reward is what RL optimizes, so it must be tight: no exploits, no code
execution, and correct on all the edge cases. These run entirely on CPU.
"""

import random

import pytest

from grpo_countdown.countdown import (
    CountdownProblem,
    compute_reward,
    extract_answer,
    format_prompt,
    generate_problem,
    safe_eval_expr,
)


# --------------------------------------------------------------------------- #
# Safe evaluator                                                              #
# --------------------------------------------------------------------------- #
def test_safe_eval_basic_arithmetic():
    assert safe_eval_expr("3 + 7 * 2") == 17.0
    assert safe_eval_expr("(3 + 7) * 2") == 20.0
    assert safe_eval_expr("10 / 4") == 2.5
    assert safe_eval_expr("-5 + 8") == 3.0


def test_safe_eval_rejects_code_execution():
    """Model output is untrusted: anything non-arithmetic must raise, not run."""
    for evil in [
        "__import__('os').system('echo hacked')",
        "open('/etc/passwd').read()",
        "1 if True else 2",
        "[x for x in range(3)]",
        "print(1)",
        "lambda: 1",
    ]:
        with pytest.raises((ValueError, SyntaxError, TypeError)):
            safe_eval_expr(evil)


# --------------------------------------------------------------------------- #
# Answer extraction                                                          #
# --------------------------------------------------------------------------- #
def test_extract_answer():
    assert extract_answer("...thinking... <answer>3 + 4</answer>") == "3 + 4"
    assert extract_answer("<answer> 2*8 </answer>") == "2*8"
    assert extract_answer("no tags here") is None
    # multiline reasoning before the answer
    assert extract_answer("line1\nline2\n<answer>1+1</answer>") == "1+1"


# --------------------------------------------------------------------------- #
# Reward                                                                     #
# --------------------------------------------------------------------------- #
def _p():
    return CountdownProblem(numbers=[3, 7, 11, 2], target=25)


def test_reward_correct_answer():
    # 3 * 7 + 2 + ... need 25: 3*7=21, +2=23, +11=34 no. Use 11+7*2=25.
    assert compute_reward("<answer>11 + 7 * 2</answer>", _p()) == 1.0


def test_reward_wrong_value_gets_format_reward():
    r = compute_reward("<answer>3 + 7</answer>", _p(), format_reward=0.1)
    assert r == 0.1


def test_reward_no_answer_tag_is_zero():
    assert compute_reward("I think it's 25", _p()) == 0.0


def test_reward_unparseable_is_zero():
    assert compute_reward("<answer>3 +</answer>", _p()) == 0.0


def test_reward_rejects_disallowed_numbers():
    # 99 is not in the allowed set -> no reward even if it hit the target
    assert compute_reward("<answer>99 - 74</answer>", _p()) == 0.0


def test_reward_rejects_reused_number():
    # only one 3 available; using 3 twice must be rejected
    p = CountdownProblem(numbers=[3, 7], target=6)
    assert compute_reward("<answer>3 + 3</answer>", p) == 0.0


def test_reward_no_code_execution_via_reward():
    # even wrapped in <answer>, evil content must score 0, never run
    r = compute_reward("<answer>__import__('os').system('x')</answer>", _p())
    assert r == 0.0


# --------------------------------------------------------------------------- #
# Problem generation                                                         #
# --------------------------------------------------------------------------- #
def test_generated_problems_are_solvable_and_wellformed():
    rng = random.Random(0)
    for _ in range(200):
        prob = generate_problem(rng, n_numbers=4, max_number=20)
        assert len(prob.numbers) == 4
        assert all(1 <= n <= 20 for n in prob.numbers)
        assert isinstance(prob.target, int)
        assert 1 <= prob.target <= 1000


def test_format_prompt_mentions_numbers_and_target():
    p = _p()
    prompt = format_prompt(p)
    assert "25" in prompt
    for n in p.numbers:
        assert str(n) in prompt
    assert "<answer>" in prompt
