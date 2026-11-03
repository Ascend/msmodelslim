#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.app.train.proxy.application import TrainProxyApplication
from msmodelslim.app.train.qat.application import get_plugin as get_qat_plugin
from msmodelslim.core.const import DeviceType
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.plugin import register_plugin

# 测试环境显式注册；生产路径由 config.ini entry_point 加载。
register_plugin(get_qat_plugin)


def test_proxy_dispatches_by_type(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    yaml_path = tmp_path / "qat.yaml"
    yaml_path.write_text(
        "type: qat\n"
        "algorithm:\n"
        "  - type: paretoq\n"
        "    w_bits: 4\n"
        "dataset:\n"
        f"  path: {data.as_posix()}\n"
        "train:\n"
        "  hyperparameters:\n"
        "    model_max_length: 16\n",
        encoding="utf-8",
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    save_path = tmp_path / "save"
    save_path.mkdir()

    train_app = MagicMock()
    factory = MagicMock()
    factory.create.return_value = train_app
    with patch(
        "msmodelslim.app.train.proxy.application._TRAIN_APP_FACTORY",
        factory,
    ):
        proxy = TrainProxyApplication(
            model_factory=MagicMock(),
            train_launcher=MagicMock(),
        )
        proxy.run(
            config_path=yaml_path,
            model_path=model_path,
            save_path=save_path,
            device=DeviceType.CPU,
            model_type="Dummy",
        )
    factory.create.assert_called_once()
    train_app.run.assert_called_once()


def test_proxy_rejects_nested_spec(tmp_path):
    yaml_path = tmp_path / "nested.yaml"
    yaml_path.write_text(
        "type: qat\nspec:\n  algorithm:\n    - type: paretoq\n",
        encoding="utf-8",
    )
    app = TrainProxyApplication(model_factory=MagicMock(), train_launcher=MagicMock())
    with pytest.raises(SchemaValidateError):
        app.run(
            config_path=yaml_path,
            model_path=tmp_path,
            save_path=tmp_path,
            device=DeviceType.CPU,
            model_type="Dummy",
        )


def test_proxy_rejects_unknown_type(tmp_path):
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("type: distill\n", encoding="utf-8")
    app = TrainProxyApplication(model_factory=MagicMock(), train_launcher=MagicMock())
    with pytest.raises((UnsupportedError, Exception)):
        app.run(
            config_path=yaml_path,
            model_path=tmp_path,
            save_path=tmp_path,
            device=DeviceType.CPU,
            model_type="Dummy",
        )
