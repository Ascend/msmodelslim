from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from msmodelslim.utils.exception import InvalidDatasetError, SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.validation.conversion import convert_to_readable_file

_TEXT_KEYS = ("text", "inputs_pretokenized", "prompt", "content")
_DATASET_SPLIT_SEED = 42
_DEFAULT_NUM_PROC = 1
# Cap datasets.map workers; full CPU affinity on large machines can fork hundreds
# of processes per rank and thrash / hang under Ctrl+C.
_MAX_MAP_NUM_PROC = 8


class TrainDatasetConfig(BaseModel):
    """QAT 训练数据集配置与静态校验。"""

    model_config = ConfigDict(extra="forbid")

    train_path: str = Field(
        ...,
        min_length=1,
        description="本地 json/jsonl 训练集文件路径（必填，且须为可读文件）。",
    )
    eval_path: Optional[str] = Field(
        default=None,
        description="独立评估集 json/jsonl。设置后优先使用该文件，忽略 ratio_eval_samples。",
    )
    ratio_eval_samples: float = Field(
        default=0.0,
        description="未设置 eval_path 时，从训练集中划分评估样本的比例，取值 [0, 1)。",
    )

    @field_validator("train_path")
    @classmethod
    def _validate_train_path_readable(cls, value: str) -> str:
        return str(convert_to_readable_file(value.strip(), param_name="dataset.train_path"))

    @field_validator("eval_path")
    @classmethod
    def _validate_eval_path_readable(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not str(value).strip():
            return None
        return str(convert_to_readable_file(str(value).strip(), param_name="dataset.eval_path"))

    @field_validator("ratio_eval_samples")
    @classmethod
    def _validate_ratio_eval_samples(cls, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaValidateError(
                f"dataset.ratio_eval_samples must be a number in [0, 1), got {value!r}",
                action="Please set ratio_eval_samples to a float in [0, 1).",
            )
        ratio = float(value)
        if not 0.0 <= ratio < 1.0:
            raise SchemaValidateError(
                f"dataset.ratio_eval_samples must be in [0, 1), got {ratio}",
                action="Please set ratio_eval_samples to a value in [0, 1); use 0 to disable eval split.",
            )
        return ratio


class TextExtractor(Protocol):
    """Strategy for turning one row-like mapping into a training string."""

    def extract(self, row: Mapping[str, Any]) -> Optional[str]: ...


@dataclass(frozen=True, slots=True)
class ChatMessageTextExtractor:
    """Render ``messages`` through the tokenizer chat template."""

    tokenizer: Any

    def extract(self, row: Mapping[str, Any]) -> Optional[str]:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            return None

        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            try:
                rendered = apply_template(messages, tokenize=False, add_generation_prompt=False)
                if isinstance(rendered, str) and rendered.strip():
                    return rendered
            except Exception as exc:
                get_logger().debug("apply_chat_template failed, falling back: %s", exc)

        parts = [
            message["content"]
            for message in messages
            if isinstance(message, dict) and isinstance(message.get("content"), str) and message["content"]
        ]
        return "\n".join(parts) if parts else None


@dataclass(frozen=True, slots=True)
class KeyTextExtractor:
    """Use the first non-empty string field from a configured key priority."""

    keys: Tuple[str, ...] = _TEXT_KEYS

    def extract(self, row: Mapping[str, Any]) -> Optional[str]:
        for key in self.keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None


@dataclass(frozen=True, slots=True)
class SingleColumnTextExtractor:
    """Support JSON loaders that expose a row as one bare string column."""

    def extract(self, row: Mapping[str, Any]) -> Optional[str]:
        if len(row) != 1:
            return None
        only = next(iter(row.values()))
        if isinstance(only, str) and only.strip():
            return only
        return None


@dataclass(frozen=True, slots=True)
class CompositeTextExtractor:
    """Apply extractor strategies in order and return the first non-empty result."""

    extractors: Tuple[TextExtractor, ...]

    def extract(self, row: Mapping[str, Any]) -> Optional[str]:
        for extractor in self.extractors:
            text = extractor.extract(row)
            if text:
                return text
        return None


def build_text_extractor(tokenizer: Any) -> TextExtractor:
    """Create the default extractor chain used by QAT datasets."""
    return CompositeTextExtractor(
        (
            ChatMessageTextExtractor(tokenizer),
            KeyTextExtractor(),
            SingleColumnTextExtractor(),
        )
    )


def _available_cpus() -> int:
    """Return the number of CPUs available to this process."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    try:
        import psutil

        return len(psutil.Process().cpu_affinity())
    except (ImportError, AttributeError, NotImplementedError, OSError):
        pass
    return os.cpu_count() or 1


class TrainDatasetBuilder:
    """Build tokenized QAT train/eval datasets from a validated config."""

    def __init__(
        self,
        config: TrainDatasetConfig,
        *,
        tokenizer: Any,
        max_length: int,
        num_proc: int = _DEFAULT_NUM_PROC,
        seed: int = _DATASET_SPLIT_SEED,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self._num_ranks = int(num_proc)
        self.seed = int(seed)
        self._text_extractor = build_text_extractor(tokenizer)

    def build(self) -> Tuple[Any, Optional[Any]]:
        """Load, split and tokenize the configured dataset.

        Returns ``(train_dataset, eval_dataset)``; ``eval_dataset`` is ``None``
        when neither ``eval_path`` nor ``ratio_eval_samples`` provides eval data.
        When ``eval_path`` is set it is used directly and ``ratio_eval_samples``
        is ignored.
        """

        load_dataset = self._import_load_dataset()
        train_raw = self._load_split(
            load_dataset,
            self.config.train_path,
            split_name="train",
        )
        if self.config.eval_path:
            if self.config.ratio_eval_samples > 0:
                get_logger().warning(
                    "dataset.eval_path is set; dataset.ratio_eval_samples=%s is ignored.",
                    self.config.ratio_eval_samples,
                )
            eval_raw = self._load_split(
                load_dataset,
                self.config.eval_path,
                split_name="eval",
            )
        else:
            train_raw, eval_raw = self._split(train_raw, self.config.ratio_eval_samples)

        num_proc = self._effective_num_proc()
        train_dataset = self._tokenize(
            train_raw,
            num_proc,
            split_name="train",
        )
        eval_dataset = (
            self._tokenize(
                eval_raw,
                num_proc,
                split_name="eval",
            )
            if eval_raw is not None
            else None
        )

        get_logger().info(
            "QAT datasets ready: train_path=%s eval_path=%s train=%d eval=%d "
            "ratio_eval_samples=%s model_max_length=%s num_proc=%s",
            self.config.train_path,
            self.config.eval_path or "<ratio-split>",
            len(train_dataset),
            len(eval_dataset) if eval_dataset is not None else 0,
            self.config.ratio_eval_samples,
            self.max_length,
            num_proc if num_proc is not None else 1,
        )
        return train_dataset, eval_dataset

    def _load_split(self, load_dataset: Any, path: str, *, split_name: str) -> Any:
        try:
            dataset = load_dataset("json", data_files=path, split="train")
        except Exception as exc:
            raise InvalidDatasetError(
                f"Failed to load {split_name} dataset from {path}",
                action="Please ensure the file is valid JSON/JSONL readable by HuggingFace datasets.",
            ) from exc
        if len(dataset) == 0:
            raise InvalidDatasetError(
                f"No samples loaded from {path}",
                action="Ensure json/jsonl entries are strings or objects with a text-like field.",
            )
        return dataset

    @staticmethod
    def _import_load_dataset() -> Any:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise UnsupportedError(
                "datasets is required to build QAT train/eval sets",
                action="Please install the HuggingFace datasets package.",
            ) from exc
        return load_dataset

    def _split(self, dataset: Any, ratio: float) -> Tuple[Any, Optional[Any]]:
        if ratio <= 0.0:
            return dataset, None
        if len(dataset) < 2:
            raise InvalidDatasetError(
                f"Need at least 2 samples to split by ratio_eval_samples={ratio}",
                action="Provide more samples or set ratio_eval_samples to 0.",
            )
        split = dataset.train_test_split(test_size=ratio, seed=self.seed, shuffle=True)
        return split["train"], split["test"]

    def _effective_num_proc(self) -> Optional[int]:
        """Pick map workers: share CPUs across ranks, hard-capped at ``_MAX_MAP_NUM_PROC``."""
        cpu = _available_cpus()
        if self._num_ranks > 1:
            workers = max(1, cpu // self._num_ranks)
        else:
            workers = cpu
        workers = min(workers, _MAX_MAP_NUM_PROC)
        return workers if workers > 1 else None

    def _tokenize_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        if not batch:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        batch_size = len(next(iter(batch.values())))
        texts: List[str] = []
        for index in range(batch_size):
            row = {key: values[index] for key, values in batch.items()}
            text = self._text_extractor.extract(row)
            if text:
                texts.append(text)

        if not texts:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        raw_ids = encoded.get("input_ids") or []
        raw_masks = encoded.get("attention_mask") or []
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []
        for index, values in enumerate(raw_ids):
            ids = list(values or [])
            if not ids:
                continue
            mask = list(raw_masks[index] or []) if index < len(raw_masks) else []
            input_ids.append(ids)
            attention_mask.append(mask if len(mask) == len(ids) else [1] * len(ids))
            labels.append(list(ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def _tokenize(
        self,
        dataset: Any,
        num_proc: Optional[int],
        *,
        split_name: str,
    ) -> Any:
        columns = list(getattr(dataset, "column_names", []) or [])
        map_kwargs: Dict[str, Any] = {
            "function": self._tokenize_batch,
            "batched": True,
            "remove_columns": columns or None,
        }
        if num_proc is not None and num_proc > 1:
            map_kwargs["num_proc"] = num_proc

        tokenized = dataset.map(**map_kwargs)
        if len(tokenized) == 0:
            raise SchemaValidateError(
                f"No usable text sample after tokenize in the {split_name} split",
                action=f"Please provide rows with a text column {list(_TEXT_KEYS)} or a chat 'messages' list.",
            )
        return tokenized
