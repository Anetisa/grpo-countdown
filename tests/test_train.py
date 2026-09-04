"""Tests for the training-loop plumbing, on CPU with a toy model.

Validates the bug-prone parts end-to-end without transformers or a GPU:
  * build_rollout_batch: padding, attention + completion masks, the shift frame
  * grpo_update: a real optimizer step that makes the model prefer the
    high-advantage completion (the whole point).
"""

import torch

from grpo_countdown.train import (
    RolloutBatch,
    build_rollout_batch,
    grpo_update,
    sequence_logprobs,
)


class ToyLM(torch.nn.Module):
    """Minimal causal LM: embed -> linear to vocab logits. Enough to test the
    GRPO update plumbing (mirrors the transformers `.logits` interface)."""

    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None):
        return self.head(self.emb(input_ids))   # [N, T, V] (plain logits)


# --------------------------------------------------------------------------- #
# build_rollout_batch                                                         #
# --------------------------------------------------------------------------- #
def test_build_rollout_batch_shapes_and_masks():
    prompts = [[1, 2, 3], [1, 2]]        # different prompt lengths
    comps = [[4, 5], [6, 7, 8]]          # different completion lengths
    b = build_rollout_batch(prompts, comps, pad_id=0)

    assert b.input_ids.shape == (2, 5)   # max total length = 3+2 = 2+3 = 5
    # row 0: [1,2,3,4,5], row 1: [1,2,6,7,8]
    assert b.input_ids[0].tolist() == [1, 2, 3, 4, 5]
    assert b.input_ids[1].tolist() == [1, 2, 6, 7, 8]
    assert b.attention_mask.sum().item() == 10   # all real here


def test_build_rollout_batch_padding():
    prompts = [[1, 2, 3], [1]]
    comps = [[4, 5], [6]]
    b = build_rollout_batch(prompts, comps, pad_id=0)
    assert b.input_ids.shape == (2, 5)
    assert b.input_ids[1].tolist() == [1, 6, 0, 0, 0]     # padded
    assert b.attention_mask[1].tolist() == [1, 1, 0, 0, 0]


def test_completion_mask_marks_only_completion_tokens():
    # prompt len 3, completion len 2 -> completion tokens at abs pos 3,4
    # shifted frame index j corresponds to predicting token j+1, so indices 2,3
    b = build_rollout_batch([[1, 2, 3]], [[4, 5]], pad_id=0)
    assert b.completion_mask.shape == (1, 4)
    assert b.completion_mask[0].tolist() == [0, 0, 1, 1]


def test_sequence_logprobs_shape_and_shift():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 16)
    ids = torch.randint(0, 16, (2, 5))
    lp = sequence_logprobs(logits, ids)
    assert lp.shape == (2, 4)   # T-1


# --------------------------------------------------------------------------- #
# grpo_update end-to-end on the toy model                                     #
# --------------------------------------------------------------------------- #
def test_grpo_update_runs_and_changes_params():
    torch.manual_seed(0)
    model = ToyLM()
    ref = ToyLM()
    ref.load_state_dict(model.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)

    b = build_rollout_batch([[1, 2, 3], [1, 2, 3]], [[4, 5], [6, 7]], pad_id=0)
    adv = torch.tensor([1.0, -1.0])
    with torch.no_grad():
        old_lp = sequence_logprobs(model(b.input_ids), b.input_ids)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    before = model.head.weight.clone()
    metrics = grpo_update(model, ref, b, adv, old_lp, opt, beta=0.0)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(before, model.head.weight)   # a step happened


def test_grpo_update_increases_good_completion_prob():
    """Repeated GRPO updates should raise the log-prob of the +advantage
    completion relative to the −advantage one — RL actually learning."""
    torch.manual_seed(0)
    model = ToyLM()
    ref = ToyLM()
    ref.load_state_dict(model.state_dict())
    for p in ref.parameters():
        p.requires_grad_(False)

    # same prompt, two different completions; prefer the first
    b = build_rollout_batch([[1, 2], [1, 2]], [[3, 4, 5], [6, 7, 8]], pad_id=0)
    adv = torch.tensor([1.0, -1.0])
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    def good_minus_bad():
        with torch.no_grad():
            lp = sequence_logprobs(model(b.input_ids), b.input_ids)
            g = (lp[0] * b.completion_mask[0]).sum()
            bad = (lp[1] * b.completion_mask[1]).sum()
        return (g - bad).item()

    start = good_minus_bad()
    for _ in range(20):
        with torch.no_grad():
            old_lp = sequence_logprobs(model(b.input_ids), b.input_ids)
        grpo_update(model, ref, b, adv, old_lp, opt, beta=0.0)
    end = good_minus_bad()

    assert end > start, f"good-minus-bad logprob should rise: {start:.3f} -> {end:.3f}"
