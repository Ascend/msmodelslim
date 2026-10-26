#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# pylint: disable=no-member

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

msmodelslim.infra.vllm_ascend_server 模块的单元测试
"""

from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateContext
from msmodelslim.infra.vllm_ascend_server import VllmAscendConfig, VllmAscendServer
from msmodelslim.utils.exception import ConfigError


def _make_context(tmp_path):
    return EvaluateContext(evaluate_id="bench-1", working_dir=tmp_path)


class TestVllmAscendServer:
    """测试 vLLM-Ascend 进程参数、启动和健康检查行为。"""

    @patch("msmodelslim.infra.vllm_ascend_server.build_safe_url", return_value="http://localhost:1234/health")
    @patch("msmodelslim.infra.vllm_ascend_server.AsyncProcess")
    def test_init_builds_safe_process_options_and_env(self, mock_process_cls, mock_build_url, tmp_path):
        """正常路径：进程参数包含基础项、布尔项、JSON 项和环境变量。"""
        model_path = tmp_path / "model"
        model_path.mkdir()
        config = VllmAscendConfig(
            port=1234,
            health_check_endpoint="/health",
            env_vars={"VLLM_FLAG": "1"},
            args={
                "trust-remote-code": True,
                "enable-prefix-caching": False,
                "ignored": None,
                "additional_config": {"enable": True},
                "dtype": "float16",
            },
        )
        server = VllmAscendServer(_make_context(tmp_path), config, model_path, tmp_path / "vllm.log")

        mock_build_url.assert_called_once_with(host="localhost", port=1234, endpoint="/health", scheme="http")
        assert server.health_check_url == "http://localhost:1234/health"
        mock_process_cls.assert_called_once_with(
            binary="python",
            log_file=str(tmp_path / "vllm.log"),
            options={
                "-m": "vllm.entrypoints.openai.api_server",
                "--model": str(model_path),
                "--host": "localhost",
                "--port": "1234",
                "--trust-remote-code": None,
                "--additional_config": '{"enable":true}',
                "--dtype": "float16",
            },
            env={"VLLM_FLAG": "1"},
        )
        assert server.process is mock_process_cls.return_value

    @patch("msmodelslim.infra.vllm_ascend_server.build_safe_url", return_value="http://localhost:1234/health")
    @patch("msmodelslim.infra.vllm_ascend_server.AsyncProcess")
    @patch.object(VllmAscendServer, "_wait_for_ready", return_value=False)
    def test_start_stops_process_when_server_is_not_ready(
        self, mock_wait_for_ready, mock_process_cls, mock_build_url, tmp_path
    ):
        """异常路径：健康检查超时后返回 False 并停止已启动进程。"""
        model_path = tmp_path / "model"
        model_path.mkdir()
        server = VllmAscendServer(_make_context(tmp_path), VllmAscendConfig(), model_path, tmp_path / "vllm.log")

        assert server.start() is False
        server.process.start.assert_called_once_with()
        server.process.stop.assert_called_once_with()
        mock_wait_for_ready.assert_called_once_with()

    @patch("msmodelslim.infra.vllm_ascend_server.time.sleep")
    @patch("msmodelslim.infra.vllm_ascend_server.time.time", side_effect=[0.0, 0.0])
    @patch("msmodelslim.infra.vllm_ascend_server.safe_get")
    @patch("msmodelslim.infra.vllm_ascend_server.build_safe_url", return_value="http://localhost:1234/health")
    @patch("msmodelslim.infra.vllm_ascend_server.AsyncProcess")
    def test_wait_for_ready_returns_true_on_http_200(
        self, mock_process_cls, mock_build_url, mock_safe_get, mock_time, mock_sleep, tmp_path
    ):
        """正常路径：健康检查返回 200 时立即就绪，不额外休眠。"""
        model_path = tmp_path / "model"
        model_path.mkdir()
        mock_safe_get.return_value = MagicMock(status_code=200)
        server = VllmAscendServer(_make_context(tmp_path), VllmAscendConfig(), model_path, tmp_path / "vllm.log")

        assert server._wait_for_ready() is True
        mock_safe_get.assert_called_once_with("http://localhost:1234/health", timeout=3)
        mock_sleep.assert_not_called()

    @patch("msmodelslim.infra.vllm_ascend_server.build_safe_url")
    @patch("msmodelslim.infra.vllm_ascend_server.AsyncProcess")
    def test_init_raises_when_model_path_missing(self, mock_process_cls, mock_build_url, tmp_path):
        """异常路径：模型路径不存在时拒绝创建服务。"""
        with pytest.raises(ConfigError):
            VllmAscendServer(
                _make_context(tmp_path),
                VllmAscendConfig(),
                tmp_path / "missing-model",
                tmp_path / "vllm.log",
            )
