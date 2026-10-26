# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from unittest.mock import MagicMock, patch

import pytest

from testing_utils.mock import mock_init_config

mock_init_config()

from msmodelslim.app.eval import application as app_module  # noqa: E402
from msmodelslim.app.eval.application import InferenceApplication  # noqa: E402
from msmodelslim.core.const import DeviceType  # noqa: E402
from msmodelslim.core.context import ContextManager  # noqa: E402
from msmodelslim.core.infer_engine.interface import FakeQuantInferenceInterface, InferenceResult  # noqa: E402
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError, UnsupportedError  # noqa: E402


class _FakeAdapter(FakeQuantInferenceInterface):
    def build_meta_model(self):
        return None

    def handle_dataset(self, dataset, device=DeviceType.NPU):
        return dataset


def _make_application(adapter, samples, context):
    model_factory = MagicMock()
    model_factory.create.return_value = adapter
    dataset_loader = MagicMock()
    dataset_loader.get_dataset_by_name.return_value = samples
    context_factory = MagicMock()
    context_factory.create.return_value = context
    result_displayer = MagicMock()
    application = InferenceApplication(
        model_factory=model_factory,
        dataset_loader=dataset_loader,
        context_factory=context_factory,
        result_displayer=result_displayer,
    )
    return application, model_factory, dataset_loader, context_factory, result_displayer


def test_run_raises_schema_validate_error_when_model_type_is_not_string():
    application, _, _, _, _ = _make_application(_FakeAdapter(), [], object())

    with pytest.raises(SchemaValidateError, match="model_type must be a string"):
        application.run(model_type=123, model_path="/model", prompt_file="prompts.jsonl")


def test_run_raises_schema_validate_error_when_max_new_tokens_is_not_positive(tmp_path):
    application, _, _, _, _ = _make_application(_FakeAdapter(), [], object())

    with (
        patch.object(app_module, "convert_to_readable_dir", return_value=tmp_path),
        patch.object(app_module, "check_read_permission"),
    ):
        with pytest.raises(SchemaValidateError, match="max_new_tokens must be positive"):
            application.run(
                model_type="Qwen3-32B",
                model_path=str(tmp_path),
                prompt_file="prompts.jsonl",
                max_new_tokens=0,
            )


def test_run_raises_invalid_model_error_when_model_factory_creation_fails(tmp_path):
    application, model_factory, _, _, _ = _make_application(_FakeAdapter(), [], object())
    model_factory.create.side_effect = RuntimeError("missing tokenizer")

    with (
        patch.object(app_module, "convert_to_readable_dir", return_value=tmp_path),
        patch.object(app_module, "check_read_permission"),
    ):
        with pytest.raises(InvalidModelError, match="missing tokenizer"):
            application.run(
                model_type="Qwen3-32B",
                model_path=str(tmp_path),
                prompt_file="prompts.jsonl",
            )


def test_run_raises_unsupported_error_when_adapter_lacks_fake_quant_interface(tmp_path):
    application, _, _, _, _ = _make_application(object(), [], object())

    with (
        patch.object(app_module, "convert_to_readable_dir", return_value=tmp_path),
        patch.object(app_module, "check_read_permission"),
    ):
        with pytest.raises(UnsupportedError, match="does not implement FakeQuantInferenceInterface"):
            application.run(
                model_type="Qwen3-32B",
                model_path=str(tmp_path),
                prompt_file="prompts.jsonl",
            )


def test_run_raises_schema_validate_error_when_prompt_dataset_is_empty(tmp_path):
    application, _, dataset_loader, _, _ = _make_application(_FakeAdapter(), [], object())

    # _load_samples runs after the format chain resolves; stub the chain so the
    # empty-prompt validation is the behavior under test here.
    with (
        patch.object(app_module, "convert_to_readable_dir", return_value=tmp_path),
        patch.object(app_module, "check_read_permission"),
        patch.object(app_module, "build_default_format_chain"),
    ):
        with pytest.raises(SchemaValidateError, match="Prompt file 'prompts.jsonl' is empty"):
            application.run(
                model_type="Qwen3-32B",
                model_path=str(tmp_path),
                prompt_file="prompts.jsonl",
            )

    dataset_loader.get_dataset_by_name.assert_called_once_with("prompts.jsonl")


def test_run_executes_single_card_inference_when_device_indices_are_absent(tmp_path):
    adapter = _FakeAdapter()
    context = object()
    application, model_factory, _, context_factory, result_displayer = _make_application(
        adapter, ["first", "second"], context
    )
    inference_result = InferenceResult(generated_token_ids=[[101, 102]], generated_texts=["hello"])
    engine = MagicMock()
    format_handler = MagicMock()
    format_loader = object()
    format_handler.handle.return_value = format_loader

    def run_engine(**kwargs):
        assert ContextManager.get_current() is context
        assert kwargs["inputs"] == ["first", "second"]
        return inference_result

    engine.run.side_effect = run_engine

    with (
        patch.object(app_module, "convert_to_readable_dir", return_value=tmp_path),
        patch.object(app_module, "check_read_permission"),
        patch.object(app_module, "build_default_format_chain", return_value=format_handler),
        patch.object(app_module, "FakeQuantInferenceEngine", return_value=engine) as engine_class,
    ):
        result = application.run(
            model_type="Qwen3-32B",
            model_path=str(tmp_path),
            prompt_file="prompts.jsonl",
            device=DeviceType.CPU,
            max_new_tokens=3,
        )

    model_factory.create.assert_called_once_with(
        model_type="Qwen3-32B",
        model_path=tmp_path,
        trust_remote_code=True,
    )
    format_handler.handle.assert_called_once_with(str(tmp_path), device="cpu")
    engine_class.assert_called_once_with()
    engine.run.assert_called_once()
    engine_kwargs = engine.run.call_args.kwargs
    assert engine_kwargs["adapter"] is adapter
    assert engine_kwargs["format_loader"] is format_loader
    assert engine_kwargs["inference_config"].max_new_tokens == 3
    assert engine_kwargs["device"] == DeviceType.CPU
    assert engine_kwargs["device_indices"] is None
    context_factory.create.assert_called_once_with(is_distributed=False)
    result_displayer.display_result.assert_called_once_with(inference_result, device_indices=None)
    assert ContextManager.get_current() is None
    assert result == {
        "samples_ok": 2,
        "model_path": str(tmp_path),
        "mode": "layer_wise",
        "device_indices": None,
        "max_new_tokens": 3,
        "generated_token_ids": [[101, 102]],
        "generated_texts": ["hello"],
    }
