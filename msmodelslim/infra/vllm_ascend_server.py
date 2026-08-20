#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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

import json
import time
from pathlib import Path
from typing import Literal, Dict, Annotated

import requests
from pydantic import BaseModel, Field, AfterValidator

from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateContext
from msmodelslim.utils.exception import ConfigError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import AsyncProcess, build_safe_url, safe_get
from msmodelslim.utils.validation.pydantic import (
    is_safe_host,
    is_safe_endpoint,
    is_port,
    greater_than_zero,
    non_empty_string,
    validate_str_length,
)


class VllmAscendConfig(BaseModel):
    """vLLM-Ascend 推理引擎配置：用于拉起 OpenAI 兼容的服务端并做健康检查。"""

    type: Literal['vllm-ascend'] = Field(default='vllm-ascend', description="推理引擎类型，固定为 `vllm-ascend`")
    entrypoint: Annotated[str, AfterValidator(non_empty_string), AfterValidator(validate_str_length())] = Field(
        default="vllm.entrypoints.openai.api_server", description="vLLM 服务启动入口，默认 OpenAI API server"
    )
    env_vars: Dict = Field(default_factory=dict, description="传递给 vLLM 进程的额外环境变量字典")
    served_model_name: Annotated[str, AfterValidator(non_empty_string), AfterValidator(validate_str_length())] = Field(
        default='served_model_name', description="已部署/对外暴露的模型名称"
    )
    host: Annotated[str, AfterValidator(is_safe_host)] = Field(default="localhost", description="服务监听地址")
    port: Annotated[int, AfterValidator(is_port)] = Field(default=1234, description="服务监听端口")
    health_check_endpoint: Annotated[str, AfterValidator(is_safe_endpoint)] = Field(
        default="/v1/models", description="健康检查接口路径（vLLM OpenAI 兼容）"
    )
    startup_timeout: Annotated[
        int,
        AfterValidator(greater_than_zero),
    ] = Field(default=600, description="服务启动超时（秒），必须 > 0")
    args: Dict = Field(default_factory=dict, description="追加的 vLLM 启动命令行参数（键值对）")


# 健康检查轮询间隔（秒）
HEALTH_CHECK_INTERVAL = 5
# HTTP 请求超时时间（秒）
HTTP_REQUEST_TIMEOUT = 3


@logger_setter()
class VllmAscendServer:
    """
    配置驱动的 VLLM-Ascend 服务器启动器。

    职责:
    1. 从配置中构建环境变量 (env_vars)。
    2. 从配置中构建命令行参数 (args)，处理 bool, str, dict。
    3. 启动服务并等待其就绪 (health check)。
    4. 停止服务。
    """

    def __init__(
        self, context: EvaluateContext, server_config: VllmAscendConfig, model_path: Path, log_file_path: Path
    ):
        """
        Args:
            context: 评估上下文
            server_config: VLLM 服务器配置
            model_path: 量化后模型的路径
            log_file_path: 本次运行的 vllm 日志文件路径
        """
        if not model_path.exists():
            raise ConfigError(f"Model path does not exist: {model_path}")

        self.config = server_config
        self.model_path = model_path
        self.log_file = log_file_path

        # 安全地构造健康检查 URL
        self.health_check_url = self._build_health_check_url()
        self.startup_timeout = self.config.startup_timeout

        # 构建命令选项和环境变量
        cmd_options = self._build_command_options()

        # 初始化异步进程管理器（使用安全接口）
        self.process = AsyncProcess(
            binary="python", log_file=str(self.log_file), options=cmd_options, env=self.config.env_vars or None
        )
        get_logger().debug("VLLM command options: %s", cmd_options)

    def start(self):
        """启动 VLLM 进程并等待其就绪。"""
        get_logger().info("Starting VLLM server for model: %s", self.model_path)
        self.process.start()

        get_logger().info("Waiting for server to be ready at %s ...", self.health_check_url)
        if not self._wait_for_ready():
            get_logger().error("VLLM server failed to start. Check log: %s", self.log_file)
            # 尝试停止僵尸进程
            try:
                self.stop()
            except Exception as e:
                get_logger().warning("Failed to stop process during cleanup: %s", e)
            return False

        get_logger().info("VLLM server started successfully.")
        return True

    def stop(self):
        """停止 VLLM 进程。"""
        get_logger().info("Stopping VLLM server...")
        self.process.stop()
        get_logger().info("VLLM server stopped.")

    def _build_health_check_url(self) -> str:
        """
        安全地构建健康检查 URL，防止 URL 注入攻击。
        使用安全模块的 URL 构建函数。

        Returns:
            健康检查 URL
        """
        return build_safe_url(
            host=self.config.host, port=self.config.port, endpoint=self.config.health_check_endpoint, scheme='http'
        )

    def _build_command_options(self) -> dict:
        """
        构建命令选项字典，用于安全命令执行。

        Returns:
            选项字典，格式为 {option_name: value}
        """
        options = {
            "-m": self.config.entrypoint,
            "--model": str(self.model_path),
            "--host": self.config.host,
            "--port": str(self.config.port),
        }

        # 遍历配置中的 'args' 来构建其他参数
        for key, value in self.config.args.items():
            if value is True:
                # e.g., trust-remote-code: true -> --trust-remote-code
                options[f"--{key}"] = None
            elif value is False or value is None:
                # e.g., enable-prefix-caching: false -> (被忽略)
                continue
            elif isinstance(value, dict):
                # e.g., additional_config: {...} -> --additional_config='{"...":...}'
                # 序列化为紧凑的 JSON（AsyncProcess 会进行安全验证）
                json_str = json.dumps(value, separators=(',', ':'))
                options[f"--{key}"] = json_str
            else:
                # e.g., tp: 2 -> --tp 2
                options[f"--{key}"] = str(value)

        return options

    def _wait_for_ready(self) -> bool:
        """
        轮询健康检查接口，等待服务器就绪。
        使用安全请求模块防止 SSRF 和其他网络攻击。

        Returns:
            True 如果服务器成功启动，False 如果超时
        """
        start_time = time.time()
        while time.time() - start_time < self.startup_timeout:
            try:
                # 使用安全请求函数，自动应用安全配置
                resp = safe_get(self.health_check_url, timeout=HTTP_REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    return True
                get_logger().debug("Health check returned status %s, retrying...", resp.status_code)
            except requests.ConnectionError:
                # 服务器尚未监听端口，继续等待
                pass
            except requests.Timeout:
                get_logger().debug("Health check request timed out, retrying...")
            except requests.TooManyRedirects:
                # 重定向过多，可能是攻击尝试
                get_logger().warning("Too many redirects for health check URL: %s", self.health_check_url)
            except requests.SSLError as e:
                # SSL 错误（如果使用 HTTPS）
                get_logger().warning("SSL verification failed: %s", e)
            except requests.RequestException as e:
                # 其他请求错误（如 DNS 解析失败等）
                get_logger().debug("Health check request error: %s", e)

            time.sleep(HEALTH_CHECK_INTERVAL)

        get_logger().error("Server startup timed out after %s seconds.", self.startup_timeout)
        return False
