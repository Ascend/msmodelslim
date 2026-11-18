#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from msmodelslim.core.train.rank_worker import RankWorker
from msmodelslim.core.train.task import CoreTrainTaskConfig
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError


def _minimal_task(tmp_path) -> CoreTrainTaskConfig:
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    return CoreTrainTaskConfig.model_validate(
        {
            "algorithm": [{"type": "paretoq"}],
            "dataset": {"train_path": str(data)},
            "train": {"training_arguments": {}, "qat": {"model_max_length": 16}},
        }
    )


def _worker(tmp_path, hooks=None) -> RankWorker:
    return RankWorker(
        _minimal_task(tmp_path),
        MagicMock(),
        tmp_path / "out",
        hooks=[] if hooks is None else hooks,
    )


def _stub_logger(monkeypatch) -> MagicMock:
    fake_logger = MagicMock()
    monkeypatch.setattr("msmodelslim.core.train.rank_worker.get_logger", lambda: fake_logger)
    return fake_logger


def _install_fake_module(monkeypatch, module_name: str, **attrs):
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def _install_fake_accelerate(monkeypatch, state=None) -> None:
    if state is None:
        state = SimpleNamespace(local_process_index=0, process_index=0, num_processes=1, device="cpu")
    _install_fake_module(monkeypatch, "accelerate", PartialState=MagicMock(return_value=state))


def _install_fake_dataset_builder(monkeypatch, build_result):
    builder_cls = MagicMock()
    builder_cls.return_value.build.return_value = build_result
    _install_fake_module(
        monkeypatch,
        "msmodelslim.core.train.train_dataset_builder",
        TrainDatasetBuilder=builder_cls,
    )
    return builder_cls


def _install_fake_trainer_modules(monkeypatch):
    callback_cls = MagicMock()
    trainer_cls = MagicMock()
    _install_fake_module(monkeypatch, "msmodelslim.core.train.callback", QATCallback=callback_cls)
    _install_fake_module(monkeypatch, "msmodelslim.core.train.trainer", QATTrainer=trainer_cls)
    return callback_cls, trainer_cls


def _assert_stage_logged(fake_logger: MagicMock, stage: str) -> None:
    fake_logger.exception.assert_called()
    assert fake_logger.exception.call_args.args[1] == stage


def _setup_running_worker(tmp_path, monkeypatch, hook):
    """组装 run() 全链路所需的边界替身，返回可断言的对象集合。"""
    fake_logger = _stub_logger(monkeypatch)
    _install_fake_accelerate(monkeypatch)
    builder = _install_fake_dataset_builder(monkeypatch, (["s1"], ["s2"]))
    callback_cls, trainer_cls = _install_fake_trainer_modules(monkeypatch)
    fake_tok = MagicMock()
    fake_model = MagicMock()
    fake_model.config.use_cache = True
    worker = _worker(tmp_path, hooks=[hook])
    worker.model_adapter.load_tokenizer_for_qat.return_value = fake_tok
    worker.model_adapter.load_model_for_qat.return_value = fake_model
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    return {
        "worker": worker,
        "logger": fake_logger,
        "builder": builder,
        "callback_cls": callback_cls,
        "trainer_cls": trainer_cls,
        "tokenizer": fake_tok,
        "model": fake_model,
    }


# ========== 构造 / entry ==========


def test_RankWorker_stores_save_path_when_constructed(tmp_path):
    assert _worker(tmp_path).save_path == tmp_path / "out"


def test_RankWorker_stores_model_max_length_when_constructed(tmp_path):
    assert _worker(tmp_path).model_max_length == 16


def test_RankWorker_stores_hooks_when_constructed(tmp_path):
    hooks = [MagicMock()]
    assert _worker(tmp_path, hooks=hooks).hooks == hooks


def test_RankWorker_entry_runs_worker_when_called(tmp_path, monkeypatch):
    run_calls = []

    def _record_run(self):
        run_calls.append(self)

    monkeypatch.setattr(RankWorker, "run", _record_run)
    RankWorker.entry(_minimal_task(tmp_path), MagicMock(), tmp_path / "out", [])
    assert len(run_calls) == 1


# ========== run()：环境与进程组校验 ==========


def test_RankWorker_run_raises_when_world_size_invalid(tmp_path, monkeypatch):
    _stub_logger(monkeypatch)
    worker = _worker(tmp_path, hooks=[MagicMock()])
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "0")
    with pytest.raises(SchemaValidateError, match="Invalid ranks"):
        worker.run()


def test_RankWorker_run_raises_when_local_rank_not_int(tmp_path, monkeypatch):
    _stub_logger(monkeypatch)
    worker = _worker(tmp_path, hooks=[MagicMock()])
    monkeypatch.setenv("LOCAL_RANK", "x")
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(SchemaValidateError, match="Invalid distributed env"):
        worker.run()


def test_RankWorker_run_raises_when_accelerate_missing(tmp_path, monkeypatch):
    _stub_logger(monkeypatch)
    worker = _worker(tmp_path, hooks=[MagicMock()])
    monkeypatch.setitem(sys.modules, "accelerate", None)
    with pytest.raises(UnsupportedError, match="accelerate is required"):
        worker.run()


def test_RankWorker_run_raises_when_process_group_not_initialized(tmp_path, monkeypatch):
    _stub_logger(monkeypatch)
    _install_fake_accelerate(
        monkeypatch,
        state=SimpleNamespace(local_process_index=0, process_index=0, num_processes=2, device="cpu"),
    )
    worker = _worker(tmp_path, hooks=[MagicMock()])
    with pytest.raises(UnsupportedError, match="Process group was not initialized"):
        worker.run()


# ========== run()：阶段失败日志 ==========


def test_RankWorker_run_logs_stage_when_tokenizer_load_fails(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _install_fake_accelerate(monkeypatch)
    worker = _worker(tmp_path, hooks=[MagicMock()])
    worker.model_adapter.load_tokenizer_for_qat.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        worker.run()
    _assert_stage_logged(fake_logger, "load_tokenizer")


def test_RankWorker_build_datasets_logs_stage_when_builder_fails(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    worker = _worker(tmp_path)
    worker.tokenizer = MagicMock()
    worker.dist_info = {"world_size": 1}
    builder = MagicMock()
    builder.return_value.build.side_effect = RuntimeError("boom")
    _install_fake_module(
        monkeypatch,
        "msmodelslim.core.train.train_dataset_builder",
        TrainDatasetBuilder=builder,
    )
    with pytest.raises(RuntimeError, match="boom"):
        worker.run()
    _assert_stage_logged(fake_logger, "build_datasets")


def test_RankWorker_run_logs_stage_when_model_load_fails(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _install_fake_accelerate(monkeypatch)
    _install_fake_dataset_builder(monkeypatch, ([], []))
    worker = _worker(tmp_path, hooks=[MagicMock()])
    worker.model_adapter.load_model_for_qat.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        worker.run()
    _assert_stage_logged(fake_logger, "load_model")


def test_RankWorker_run_logs_stage_when_pre_train_hook_fails(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _install_fake_accelerate(monkeypatch)
    _install_fake_dataset_builder(monkeypatch, ([], []))
    hook = MagicMock()
    hook.pre_train.side_effect = RuntimeError("boom")
    worker = _worker(tmp_path, hooks=[hook])
    with pytest.raises(RuntimeError, match="boom"):
        worker.run()
    _assert_stage_logged(fake_logger, "pre_train")


def test_RankWorker_run_raises_when_hooks_empty(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _install_fake_accelerate(monkeypatch)
    _install_fake_dataset_builder(monkeypatch, ([], []))
    worker = _worker(tmp_path, hooks=[])
    with pytest.raises(SchemaValidateError, match="empty pre_train hooks"):
        worker.run()
    fake_logger.exception.assert_not_called()


def test_RankWorker_run_logs_stage_when_trainer_build_fails(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _install_fake_accelerate(monkeypatch)
    _install_fake_dataset_builder(monkeypatch, ([], []))
    _, trainer_cls = _install_fake_trainer_modules(monkeypatch)
    trainer_cls.side_effect = RuntimeError("boom")
    worker = _worker(tmp_path, hooks=[MagicMock()])
    with pytest.raises(RuntimeError, match="boom"):
        worker.run()
    _assert_stage_logged(fake_logger, "train")


# ========== run()：成功路径 ==========


def test_RankWorker_run_completes_pipeline_when_stages_succeed(tmp_path, monkeypatch):
    ctx = _setup_running_worker(tmp_path, monkeypatch, MagicMock())
    worker, fake_logger = ctx["worker"], ctx["logger"]
    worker.run()
    assert worker.tokenizer is ctx["tokenizer"]
    assert worker.model is ctx["model"]
    assert ctx["model"].config.use_cache is False
    assert worker.train_dataset == ["s1"]
    assert worker.eval_dataset == ["s2"]
    assert worker.core_task.qparam_names == []
    assert worker.core_task.dist_info["world_size"] == 1
    assert worker.core_task.dist_info["local_rank"] == 0
    assert worker.core_task.dist_info["dist_initialized"] is False
    fake_logger.exception.assert_not_called()
    log_args = fake_logger.info.call_args.args
    assert log_args[1] == []
    assert log_args[3] == 1
    assert log_args[4] == 1


def test_RankWorker_run_passes_task_context_to_trainer_when_built(tmp_path, monkeypatch):
    ctx = _setup_running_worker(tmp_path, monkeypatch, MagicMock())
    worker = ctx["worker"]
    worker.run()
    builder_kwargs = ctx["builder"].call_args.kwargs
    assert ctx["builder"].call_args.args == (worker.task_cfg.dataset,)
    assert builder_kwargs == {
        "tokenizer": ctx["tokenizer"],
        "max_length": 16,
        "num_proc": 1,
    }
    trainer_kwargs = ctx["trainer_cls"].call_args.kwargs
    assert trainer_kwargs["model"] is ctx["model"]
    assert trainer_kwargs["train_cfg"] is worker.task_cfg.train
    assert trainer_kwargs["output_dir"] == str(tmp_path / "out")
    assert trainer_kwargs["train_dataset"] == ["s1"]
    assert trainer_kwargs["eval_dataset"] == ["s2"]
    assert trainer_kwargs["tokenizer"] is ctx["tokenizer"]
    assert trainer_kwargs["algorithm_cfg"][0]["type"] == "paretoq"
    assert trainer_kwargs["source_model_path"] is None
    ctx["trainer_cls"].return_value.train.assert_called_once()


def test_RankWorker_run_falls_back_to_default_qparam_names_when_hook_silent(tmp_path, monkeypatch):
    ctx = _setup_running_worker(tmp_path, monkeypatch, MagicMock())
    ctx["worker"].run()
    trainer_kwargs = ctx["trainer_cls"].call_args.kwargs
    assert trainer_kwargs["qparam_names"] == ("weight_clip_val",)
    assert trainer_kwargs["callbacks"] == [ctx["callback_cls"].return_value]


def test_RankWorker_run_reports_qparam_names_when_hook_reports(tmp_path, monkeypatch):
    def _hook_pre_train(ctx):
        ctx.extra["qparam_names"] = ["alpha_1", "alpha_2"]

    hook = MagicMock()
    hook.pre_train.side_effect = _hook_pre_train
    ctx = _setup_running_worker(tmp_path, monkeypatch, hook)
    worker = ctx["worker"]
    worker.run()
    assert worker.core_task.qparam_names == ["alpha_1", "alpha_2"]
    trainer_kwargs = ctx["trainer_cls"].call_args.kwargs
    assert trainer_kwargs["qparam_names"] == ("alpha_1", "alpha_2")


def test_RankWorker_run_accepts_plain_dict_algorithm_when_task_carries_raw_dicts(tmp_path, monkeypatch):
    fake_logger = _stub_logger(monkeypatch)
    _install_fake_accelerate(monkeypatch)
    _install_fake_dataset_builder(monkeypatch, (["s1"], None))
    _, trainer_cls = _install_fake_trainer_modules(monkeypatch)
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    task = SimpleNamespace(
        algorithm=[{"type": "plain"}],
        dataset=SimpleNamespace(path=str(data), ratio_eval_samples=0.0),
        train=SimpleNamespace(
            qat=SimpleNamespace(model_max_length=16, fake_quant_after_n_steps=None),
        ),
        source_model_path=None,
    )
    worker = RankWorker(task, MagicMock(), tmp_path / "out", [MagicMock()])
    worker.run()
    assert trainer_cls.call_args.kwargs["algorithm_cfg"] == [{"type": "plain"}]
    fake_logger.exception.assert_not_called()
