#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pytest

from msmodelslim.core.train.train_dataset_builder import (
    TrainDatasetBuilder,
    TrainDatasetConfig,
    _MAX_MAP_NUM_PROC,
    build_text_extractor,
)
from msmodelslim.utils.exception import SchemaValidateError


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        if not isinstance(text, list):
            return {
                "input_ids": [ord(char) for char in text],
                "attention_mask": [1] * len(text),
            }
        texts = text
        return {
            "input_ids": [[ord(char) for char in item] for item in texts],
            "attention_mask": [[1] * len(item) for item in texts],
        }


class _FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(self.rows[0]) if self.rows else []

    def __len__(self):
        return len(self.rows)

    def train_test_split(self, test_size, seed, shuffle):
        eval_size = max(1, int(round(len(self.rows) * test_size)))
        return {
            "train": _FakeDataset(self.rows[:-eval_size]),
            "test": _FakeDataset(self.rows[-eval_size:]),
        }

    def map(self, function, batched, remove_columns, **kwargs):
        batch = {key: [row.get(key) for row in self.rows] for key in self.column_names}
        mapped = function(batch)
        return _FakeDataset(
            [{key: values[index] for key, values in mapped.items()} for index in range(len(mapped["input_ids"]))]
        )


def _builder(
    tmp_path,
    *,
    ratio=0.0,
    rows=None,
    eval_rows=None,
    num_proc=1,
    use_eval_path=False,
    tokenizer=None,
):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    eval_path = None
    if use_eval_path:
        eval_file = tmp_path / "eval.jsonl"
        eval_file.write_text("{}\n", encoding="utf-8")
        eval_path = str(eval_file)
    config = TrainDatasetConfig(
        train_path=str(data),
        eval_path=eval_path,
        ratio_eval_samples=ratio,
    )
    builder = TrainDatasetBuilder(
        config,
        tokenizer=_FakeTokenizer() if tokenizer is None else tokenizer,
        max_length=32,
        num_proc=num_proc,
    )

    def _load(*args, **kwargs):
        data_files = kwargs.get("data_files") or (args[1] if len(args) > 1 else None)
        if use_eval_path and data_files == eval_path:
            return _FakeDataset(eval_rows or [])
        return _FakeDataset(rows or [])

    builder._import_load_dataset = lambda: _load
    return builder


def test_build_text_extractor_prefers_chat_template():
    class _Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "rendered:" + messages[0]["content"]

    row = {"messages": [{"role": "user", "content": "hello"}]}
    assert build_text_extractor(_Tokenizer()).extract(row) == "rendered:hello"


def test_build_text_extractor_falls_back_to_text_key():
    row = {"text": "plain"}
    assert build_text_extractor(object()).extract(row) == "plain"


def test_build_text_extractor_falls_back_to_single_column():
    row = {"only": "solo"}
    assert build_text_extractor(object()).extract(row) == "solo"


def test_TrainDatasetBuilder_builds_train_without_eval_when_ratio_zero(tmp_path):
    builder = _builder(tmp_path, rows=[{"text": "a"}, {"text": ""}, {"text": "bc"}])

    train_dataset, eval_dataset = builder.build()

    assert len(train_dataset) == 2
    assert eval_dataset is None


def test_TrainDatasetBuilder_splits_and_tokenizes_eval(tmp_path):
    builder = _builder(
        tmp_path,
        ratio=0.5,
        rows=[{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}],
    )

    train_dataset, eval_dataset = builder.build()

    assert len(train_dataset) == 2
    assert eval_dataset is not None
    assert len(eval_dataset) == 2
    assert train_dataset.rows[0]["input_ids"]


def test_TrainDatasetBuilder_raises_when_tokenize_drops_all_rows(tmp_path):
    builder = _builder(tmp_path, rows=[{"text": ""}])

    with pytest.raises(SchemaValidateError, match="No usable text sample"):
        builder.build()


def test_TrainDatasetBuilder_uses_eval_path_and_ignores_ratio(tmp_path):
    builder = _builder(
        tmp_path,
        ratio=0.5,
        rows=[{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}],
        eval_rows=[{"text": "e"}, {"text": "f"}],
        use_eval_path=True,
    )

    train_dataset, eval_dataset = builder.build()

    assert len(train_dataset) == 4
    assert eval_dataset is not None
    assert len(eval_dataset) == 2


def test_TrainDatasetBuilder_adds_labels_for_plain_text(tmp_path):
    builder = _builder(tmp_path, rows=[{"text": "ab"}])

    train_dataset, _ = builder.build()

    assert train_dataset.rows[0]["labels"] == train_dataset.rows[0]["input_ids"]


def test_TrainDatasetBuilder_keeps_full_labels_for_chat_messages(tmp_path):
    class _FakeChatTokenizer(_FakeTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
            return "".join(message["content"] for message in messages)

    builder = _builder(
        tmp_path,
        rows=[{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}],
        tokenizer=_FakeChatTokenizer(),
    )

    train_dataset, _ = builder.build()

    assert train_dataset.rows[0]["input_ids"] == [ord("u"), ord("a")]
    assert train_dataset.rows[0]["labels"] == train_dataset.rows[0]["input_ids"]


def test_TrainDatasetBuilder_caps_map_num_proc(tmp_path, monkeypatch):
    builder = _builder(tmp_path, num_proc=1)
    monkeypatch.setattr(
        "msmodelslim.core.train.train_dataset_builder._available_cpus",
        lambda: 256,
    )
    assert builder._effective_num_proc() == _MAX_MAP_NUM_PROC
