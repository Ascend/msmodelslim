#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from msmodelslim.core.train.loss import (
    ChunkedCrossEntropyLoss,
    LossOnlyOutput,
    capture_backbone_hidden,
    chunked_shifted_cross_entropy,
    count_shifted_ce_tokens,
)
from msmodelslim.utils.exception import SchemaValidateError

_HID = 8
_VOC = 7
_BATCH = 2
_SEQ = 5


def _stub_logger(monkeypatch) -> MagicMock:
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.loss.get_logger", lambda: fake_logger)
    return fake_logger


def _make_lm_head(seed=0, hidden=_HID, vocab=_VOC, bias=True) -> nn.Linear:
    gen = torch.Generator().manual_seed(seed)
    head = nn.Linear(hidden, vocab, bias=bias)
    with torch.no_grad():
        head.weight.copy_(torch.randn(vocab, hidden, generator=gen))
        if head.bias is not None:
            head.bias.copy_(torch.randn(vocab, generator=gen))
    return head


def _make_hidden_and_labels(seed=1, batch=_BATCH, seq=_SEQ, hidden=_HID, vocab=_VOC):
    gen = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(batch, seq, hidden, generator=gen)
    labels = torch.randint(0, vocab, (batch, seq), generator=gen)
    return hidden_states, labels


def _with_ignore_index(labels):
    mixed = labels.clone()
    mixed[:, 0] = -100
    mixed[0, -1] = -100
    return mixed


def _reference_ce(lm_head, hidden, labels, num_items_in_batch=None, ignore_index=-100):
    """全量 [B, S, V] logits 的等价实现（同 shift 语义）。"""
    logits = F.linear(hidden, lm_head.weight, getattr(lm_head, "bias", None)).float()
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = labels[:, 1:].reshape(-1)
    total = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index, reduction="sum")
    if num_items_in_batch is None:
        denom = shift_labels.ne(ignore_index).sum().clamp(min=1).to(dtype=total.dtype)
    elif isinstance(num_items_in_batch, torch.Tensor):
        denom = num_items_in_batch.to(dtype=total.dtype)
    else:
        denom = total.new_tensor(float(num_items_in_batch))
    return total / denom


class _NormBackbone(nn.Module):
    """model.norm + lm_head 结构：final-norm 走 forward hook 捕获。"""

    def __init__(self, hidden=_HID, vocab=_VOC):
        super().__init__()
        inner = nn.Module()
        inner.norm = nn.LayerNorm(hidden)
        self.model = inner
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, inputs_embeds):
        return self.lm_head(self.model.norm(inputs_embeds))


class _LnFBackbone(nn.Module):
    """GPT-2 风格 transformer.ln_f 结构：final-norm 走 forward hook 捕获。"""

    def __init__(self, hidden=_HID, vocab=_VOC):
        super().__init__()
        inner = nn.Module()
        inner.ln_f = nn.LayerNorm(hidden)
        self.transformer = inner
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, inputs_embeds):
        return self.lm_head(self.transformer.ln_f(inputs_embeds))


class _HeadOnlyBackbone(nn.Module):
    """无 final-norm 结构：走替换 lm_head.forward 捕获。"""

    def __init__(self, hidden=_HID, vocab=_VOC):
        super().__init__()
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, inputs_embeds):
        return self.lm_head(inputs_embeds)


class _IdleBackbone(nn.Module):
    """有 lm_head 但 forward 不触达：hidden 捕获失败。"""

    def __init__(self, hidden=_HID, vocab=_VOC):
        super().__init__()
        self.lm_head = nn.Linear(hidden, vocab)

    def forward(self, inputs_embeds):
        return None


# ========== chunked_shifted_cross_entropy：数值对齐 ==========


def test_chunked_shifted_cross_entropy_aligns_with_full_ce_when_chunk_divides_tokens():
    head, (hidden, labels) = _make_lm_head(), _make_hidden_and_labels()
    result = chunked_shifted_cross_entropy(head, hidden, labels, chunk_size=4)
    assert torch.allclose(result, _reference_ce(head, hidden, labels))


def test_chunked_shifted_cross_entropy_aligns_with_full_ce_when_chunk_has_remainder():
    head, (hidden, labels) = _make_lm_head(), _make_hidden_and_labels()
    result = chunked_shifted_cross_entropy(head, hidden, labels, chunk_size=3)
    assert torch.allclose(result, _reference_ce(head, hidden, labels))


def test_chunked_shifted_cross_entropy_aligns_with_full_ce_when_chunk_exceeds_tokens():
    head, (hidden, labels) = _make_lm_head(), _make_hidden_and_labels()
    result = chunked_shifted_cross_entropy(head, hidden, labels, chunk_size=100)
    assert torch.allclose(result, _reference_ce(head, hidden, labels))


def test_chunked_shifted_cross_entropy_aligns_with_full_ce_when_chunk_size_is_one():
    head, (hidden, labels) = _make_lm_head(), _make_hidden_and_labels()
    result = chunked_shifted_cross_entropy(head, hidden, labels, chunk_size=1)
    assert torch.allclose(result, _reference_ce(head, hidden, labels))


def test_chunked_shifted_cross_entropy_aligns_with_full_ce_when_head_has_no_bias():
    head = _make_lm_head(seed=3, bias=False)
    hidden, labels = _make_hidden_and_labels(seed=4)
    result = chunked_shifted_cross_entropy(head, hidden, labels, chunk_size=3)
    assert head.bias is None
    assert torch.allclose(result, _reference_ce(head, hidden, labels))


# ========== chunked_shifted_cross_entropy：分母形态 ==========


def test_chunked_shifted_cross_entropy_averages_valid_tokens_when_num_items_is_none():
    head, (hidden, labels) = _make_lm_head(), _make_hidden_and_labels()
    labels = _with_ignore_index(labels)
    result = chunked_shifted_cross_entropy(head, hidden, labels, num_items_in_batch=None, chunk_size=4)
    assert torch.allclose(result, _reference_ce(head, hidden, labels))
    n_valid = labels[:, 1:].ne(-100).sum()
    assert n_valid.item() == 7
    assert torch.allclose(result, _reference_ce(head, hidden, labels, num_items_in_batch=n_valid))


def test_chunked_shifted_cross_entropy_divides_by_int_when_num_items_is_int():
    head, (hidden, labels) = _make_lm_head(), _make_hidden_and_labels()
    result = chunked_shifted_cross_entropy(head, hidden, labels, num_items_in_batch=16, chunk_size=4)
    assert torch.allclose(result, _reference_ce(head, hidden, labels, num_items_in_batch=16))
    mean = chunked_shifted_cross_entropy(head, hidden, labels, chunk_size=4)
    assert torch.allclose(result, mean * _BATCH * (_SEQ - 1) / 16)


def test_chunked_shifted_cross_entropy_divides_by_float_when_num_items_is_float():
    head, (hidden, labels) = _make_lm_head(), _make_hidden_and_labels()
    result = chunked_shifted_cross_entropy(head, hidden, labels, num_items_in_batch=3.5, chunk_size=4)
    assert torch.allclose(result, _reference_ce(head, hidden, labels, num_items_in_batch=3.5))


def test_chunked_shifted_cross_entropy_casts_tensor_when_num_items_is_tensor():
    head, (hidden, labels) = _make_lm_head(), _make_hidden_and_labels()
    denom = torch.tensor(5, dtype=torch.long)
    result = chunked_shifted_cross_entropy(head, hidden, labels, num_items_in_batch=denom, chunk_size=4)
    assert torch.allclose(result, _reference_ce(head, hidden, labels, num_items_in_batch=5))


def test_chunked_shifted_cross_entropy_skips_ignored_labels_when_labels_contain_ignore_index():
    head, (hidden, labels) = _make_lm_head(), _make_hidden_and_labels()
    labels = _with_ignore_index(labels)
    for chunk_size in (1, 3, 4, 100):
        result = chunked_shifted_cross_entropy(head, hidden, labels, chunk_size=chunk_size)
        assert torch.allclose(result, _reference_ce(head, hidden, labels))
        assert torch.isfinite(result)


def test_chunked_shifted_cross_entropy_returns_zero_when_all_labels_ignored():
    head, (hidden, labels) = _make_lm_head(), _make_hidden_and_labels()
    labels[:] = -100
    result = chunked_shifted_cross_entropy(head, hidden, labels, chunk_size=4)
    assert torch.isfinite(result)
    assert result.item() == pytest.approx(0.0, abs=1e-7)


# ========== chunked_shifted_cross_entropy：校验分支 ==========


def test_chunked_shifted_cross_entropy_raises_when_chunk_size_below_one():
    head, (hidden, labels) = _make_lm_head(), _make_hidden_and_labels()
    with pytest.raises(SchemaValidateError, match="train.qat.ce_chunk_size"):
        chunked_shifted_cross_entropy(head, hidden, labels, chunk_size=0)


def test_chunked_shifted_cross_entropy_raises_when_hidden_not_rank_three():
    head, (hidden, labels) = _make_lm_head(), _make_hidden_and_labels()
    with pytest.raises(SchemaValidateError, match="expects hidden"):
        chunked_shifted_cross_entropy(head, hidden[:, 0, :], labels)


def test_chunked_shifted_cross_entropy_raises_when_labels_shape_mismatches():
    head, (hidden, _) = _make_lm_head(), _make_hidden_and_labels()
    bad_labels = torch.randint(0, _VOC, (3, _SEQ))
    with pytest.raises(SchemaValidateError, match="does not match hidden"):
        chunked_shifted_cross_entropy(head, hidden, bad_labels, chunk_size=4)


# ========== count_shifted_ce_tokens ==========


def test_count_shifted_ce_tokens_returns_zero_when_labels_not_tensor():
    assert count_shifted_ce_tokens(None) == 0
    assert count_shifted_ce_tokens([1, 2, 3]) == 0


def test_count_shifted_ce_tokens_returns_zero_when_labels_empty():
    assert count_shifted_ce_tokens(torch.empty(2, 0, dtype=torch.long)) == 0
    assert count_shifted_ce_tokens(torch.empty(0, dtype=torch.long)) == 0


def test_count_shifted_ce_tokens_counts_flat_when_labels_rank_one():
    labels = torch.tensor([1, -100, 3, -100])
    assert count_shifted_ce_tokens(labels) == 2


def test_count_shifted_ce_tokens_shifts_when_labels_rank_two():
    labels = torch.tensor([[1, 2, -100], [4, -100, 6]])
    assert count_shifted_ce_tokens(labels) == 2


def test_count_shifted_ce_tokens_counts_all_when_sequence_length_is_one():
    labels = torch.tensor([[1], [-100]])
    assert count_shifted_ce_tokens(labels) == 1


def test_count_shifted_ce_tokens_returns_zero_when_all_labels_ignored():
    labels = torch.full((2, 3), -100, dtype=torch.long)
    assert count_shifted_ce_tokens(labels) == 0


# ========== capture_backbone_hidden ==========


def test_capture_backbone_hidden_captures_norm_output_when_model_has_norm():
    model = _NormBackbone()
    orig_forward = model.lm_head.forward
    hidden_in, _ = _make_hidden_and_labels(seed=5)

    with capture_backbone_hidden(model, model.lm_head) as captured:
        out = model(inputs_embeds=hidden_in)
        assert "hidden" in captured

    assert out.shape == (_BATCH, _SEQ, 1)
    assert torch.count_nonzero(out).item() == 0
    assert torch.allclose(captured["hidden"], model.model.norm(hidden_in))
    # bound method 每次访问都是新对象，恢复语义断言用 ==（同 __self__ + __func__）
    assert model.lm_head.forward == orig_forward
    assert model.lm_head.__dict__["forward"] == orig_forward
    assert len(model.model.norm._forward_hooks) == 0
    assert model.lm_head(hidden_in).shape == (_BATCH, _SEQ, _VOC)


def test_capture_backbone_hidden_captures_ln_f_output_when_transformer_backbone():
    model = _LnFBackbone()
    orig_forward = model.lm_head.forward
    hidden_in, _ = _make_hidden_and_labels(seed=12)

    with capture_backbone_hidden(model, model.lm_head) as captured:
        out = model(inputs_embeds=hidden_in)
        assert "hidden" in captured

    assert out.shape == (_BATCH, _SEQ, 1)
    assert torch.count_nonzero(out).item() == 0
    assert torch.allclose(captured["hidden"], model.transformer.ln_f(hidden_in))
    assert model.lm_head.forward == orig_forward
    assert model.lm_head.__dict__["forward"] == orig_forward
    assert len(model.transformer.ln_f._forward_hooks) == 0
    assert model.lm_head(hidden_in).shape == (_BATCH, _SEQ, _VOC)


def test_capture_backbone_hidden_captures_head_input_when_norm_missing():
    model = _HeadOnlyBackbone()
    orig_forward = model.lm_head.forward
    hidden_in, _ = _make_hidden_and_labels(seed=6)

    with capture_backbone_hidden(model, model.lm_head) as captured:
        out = model(inputs_embeds=hidden_in)
        assert "hidden" in captured

    assert out.shape == (_BATCH, _SEQ, 1)
    assert torch.count_nonzero(out).item() == 0
    assert torch.equal(captured["hidden"], hidden_in)
    assert model.lm_head.forward == orig_forward
    assert model.lm_head.__dict__["forward"] == orig_forward
    assert model.lm_head(hidden_in).shape == (_BATCH, _SEQ, _VOC)


# ========== ChunkedCrossEntropyLoss ==========


def test_ChunkedCrossEntropyLoss_stores_chunk_size_when_constructed():
    assert ChunkedCrossEntropyLoss(chunk_size=3).chunk_size == 3
    assert ChunkedCrossEntropyLoss().chunk_size == 512


def test_ChunkedCrossEntropyLoss_raises_when_chunk_size_below_one():
    with pytest.raises(SchemaValidateError, match="train.qat.ce_chunk_size"):
        ChunkedCrossEntropyLoss(chunk_size=0)
    with pytest.raises(SchemaValidateError, match="train.qat.ce_chunk_size"):
        ChunkedCrossEntropyLoss(chunk_size=-2)


def test_ChunkedCrossEntropyLoss_returns_none_when_labels_missing():
    hidden_in, _ = _make_hidden_and_labels(seed=7)
    loss_fn = ChunkedCrossEntropyLoss(chunk_size=4)
    assert loss_fn(MagicMock(), {"inputs_embeds": hidden_in}, unwrap=lambda m: m) is None


def test_ChunkedCrossEntropyLoss_warns_when_unwrapped_model_lacks_lm_head(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _, labels = _make_hidden_and_labels(seed=8)
    unwrapped = nn.Module()
    loss_fn = ChunkedCrossEntropyLoss(chunk_size=4)
    result = loss_fn(
        MagicMock(),
        {"inputs_embeds": torch.zeros(_BATCH, _SEQ, _HID), "labels": labels},
        unwrap=lambda m: unwrapped,
    )
    assert result is None
    fake_logger.warning.assert_called_once()
    assert "no lm_head" in fake_logger.warning.call_args.args[0]


def test_ChunkedCrossEntropyLoss_warns_when_hidden_not_captured(monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    model = _IdleBackbone()
    hidden_in, labels = _make_hidden_and_labels(seed=9)
    loss_fn = ChunkedCrossEntropyLoss(chunk_size=4)
    result = loss_fn(model, {"inputs_embeds": hidden_in, "labels": labels}, unwrap=lambda m: m)
    assert result is None
    fake_logger.warning.assert_called_once()
    assert "final-norm hidden" in fake_logger.warning.call_args.args[0]


def test_ChunkedCrossEntropyLoss_returns_loss_and_output_when_hidden_captured():
    model = _NormBackbone()
    hidden_in, labels = _make_hidden_and_labels(seed=10)
    loss_fn = ChunkedCrossEntropyLoss(chunk_size=3)
    result = loss_fn(model, {"inputs_embeds": hidden_in, "labels": labels}, unwrap=lambda m: m)
    assert isinstance(result, tuple) and len(result) == 2
    loss, output = result
    assert torch.isfinite(loss) and loss.dim() == 0
    assert isinstance(output, LossOnlyOutput)
    assert output.loss is loss
    assert output.logits is None
    expected = chunked_shifted_cross_entropy(model.lm_head, model.model.norm(hidden_in), labels, chunk_size=3)
    assert torch.allclose(loss, expected)


def test_ChunkedCrossEntropyLoss_forwards_num_items_when_provided():
    model = _NormBackbone()
    hidden_in, labels = _make_hidden_and_labels(seed=11)
    loss_fn = ChunkedCrossEntropyLoss(chunk_size=4)
    loss, _ = loss_fn(
        model,
        {"inputs_embeds": hidden_in, "labels": labels},
        unwrap=lambda m: m,
        num_items_in_batch=16,
    )
    expected = chunked_shifted_cross_entropy(
        model.lm_head, model.model.norm(hidden_in), labels, num_items_in_batch=16, chunk_size=4
    )
    assert torch.allclose(loss, expected)
    mean = chunked_shifted_cross_entropy(model.lm_head, model.model.norm(hidden_in), labels, chunk_size=4)
    assert torch.allclose(loss, mean * _BATCH * (_SEQ - 1) / 16)


# ========== LossOnlyOutput ==========


def test_LossOnlyOutput_returns_items_when_indexed():
    loss = torch.tensor(2.5)
    output = LossOnlyOutput(loss)
    assert output[0] is loss
    assert output[1] is None
    assert output[-1] is None


def test_LossOnlyOutput_len_is_two_when_queried():
    assert len(LossOnlyOutput(torch.tensor(1.0))) == 2


def test_LossOnlyOutput_pairs_loss_and_none_when_to_tuple_called():
    loss = torch.tensor(1.5)
    pair = LossOnlyOutput(loss).to_tuple()
    assert len(pair) == 2
    assert pair[0] is loss
    assert pair[1] is None
