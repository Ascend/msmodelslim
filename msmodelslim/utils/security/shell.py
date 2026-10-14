#!/usr/bin/env python
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
# pylint: disable=consider-using-with,no-member,subprocess-run-check,unspecified-encoding

import os
import re
import shlex
import signal
import subprocess  # nosec B404
from typing import List, Optional, Union

from msmodelslim.utils.exception import SecurityError
from msmodelslim.utils.logging import get_logger

# 允许的标识符字符：字母、数字、下划线、连字符、点、斜杠（用于路径）、花括号、冒号、双引号、逗号
SAFE_IDENTIFIER_REGEX = re.compile(r'^[a-zA-Z0-9_\-./{}:",]+$')


def validate_safe_identifier(value: str, field_name: str) -> str:
    """
    验证标识符只包含安全字符，防止命令注入。

    Args:
        value: 要验证的字符串值
        field_name: 字段名称，用于错误消息

    Returns:
        验证后的字符串值

    Raises:
        SecurityError: 如果值为空或包含不安全字符
    """
    if not value:
        raise SecurityError(
            f"{field_name} cannot be empty.", action=f"Please provide a non-empty value for {field_name}."
        )

    if not SAFE_IDENTIFIER_REGEX.match(value):
        raise SecurityError(
            f"{field_name} contains invalid characters: {value}",
            action=f"Please ensure {field_name} contains only valid characters [a-zA-Z0-9_\\-./{{}}:\\\",]",
        )

    return value


def sanitize_extra_args(extra_args: Optional[Union[str, List[str]]]) -> List[str]:
    """
    安全地处理额外的命令行参数，防止命令注入。

    Args:
        extra_args: 额外参数，可以是字符串或字符串列表

    Returns:
        验证后的参数列表（列表形式，不需要 shell 转义）
    """
    if not extra_args:
        return []

    sanitized = []
    if isinstance(extra_args, str):
        # 对于字符串，使用 shlex.split 安全地分割参数（正确处理引号和转义）
        # 然后验证每个部分，防止命令注入
        parts = shlex.split(extra_args)
        for part in parts:
            if part:
                # 验证参数安全性
                part = validate_safe_identifier(part, "extra_arg")
                sanitized.append(part)
    elif isinstance(extra_args, (list, tuple)):
        # 对于列表，验证每个元素
        for item in extra_args:
            if item:
                item_str = str(item).strip()
                item_str = validate_safe_identifier(item_str, "extra_arg")
                sanitized.append(item_str)

    return sanitized


def build_safe_command(binary: str, *args: str, extra_args: Optional[Union[str, List[str]]] = None) -> List[str]:
    """
    安全地构建命令行列表，所有参数都会被验证。

    Args:
        binary: 可执行文件路径或命令名称（会被验证）
        *args: 位置参数，每个参数都会被验证
        extra_args: 额外的命令行参数（可选）

    Returns:
        验证后的命令行列表（用于 shell=False）

    Raises:
        SecurityError: 如果任何参数包含不安全字符
    """
    # 验证 binary
    safe_binary = validate_safe_identifier(binary, "binary")
    cmd_parts = [safe_binary]

    # 验证位置参数
    for arg in args:
        safe_arg = validate_safe_identifier(arg, "argument")
        cmd_parts.append(safe_arg)

    # 处理额外参数
    if extra_args:
        sanitized_extra = sanitize_extra_args(extra_args)
        cmd_parts.extend(sanitized_extra)

    return cmd_parts


def build_safe_command_with_options(
    binary: str, options: dict, extra_args: Optional[Union[str, List[str]]] = None
) -> List[str]:
    """
    安全地构建带选项的命令行列表。

    Args:
        binary: 可执行文件路径或命令名称（会被验证）
        options: 选项字典，格式为 {option_name: value}，例如 {"--models": "model_name"}
        extra_args: 额外的命令行参数（可选）

    Returns:
        验证后的命令行列表（用于 shell=False）

    Raises:
        SecurityError: 如果任何参数包含不安全字符
    """
    # 验证 binary
    safe_binary = validate_safe_identifier(binary, "binary")
    cmd_parts = [safe_binary]

    # 验证选项
    for option_name, option_value in options.items():
        # 选项名称允许包含 "--" 前缀，所以需要特殊处理
        # 验证选项名称（允许 -- 前缀和字母、数字、下划线、连字符）
        if not option_name:
            raise SecurityError("option_name cannot be empty.", action="Please provide a non-empty option name.")
        # 选项名称可以是 --option 或 -o 格式
        if not re.match(r'^-?[a-zA-Z0-9_\-]+$', option_name):
            raise SecurityError(
                f"option_name contains invalid characters: {option_name}",
                action="Please ensure option_name contains only valid characters [-a-zA-Z0-9_\\-]",
            )
        cmd_parts.append(option_name)

        # 验证选项值
        if option_value is not None:
            safe_option_value = validate_safe_identifier(str(option_value), f"option_value({option_name})")
            cmd_parts.append(safe_option_value)

    # 处理额外参数
    if extra_args:
        sanitized_extra = sanitize_extra_args(extra_args)
        cmd_parts.extend(sanitized_extra)

    return cmd_parts


class ShellRunner:
    """安全的命令执行器"""

    @staticmethod
    def run_safe_cmd(
        binary: str,
        options: Optional[dict] = None,
        extra_args: Optional[Union[str, List[str]]] = None,
        timeout: Optional[int] = None,
    ):
        """
        安全地执行命令，所有参数都会被验证和转义，防止命令注入。

        Args:
            binary: 可执行文件路径或命令名称
            options: 选项字典，格式为 {option_name: value}，例如 {"--models": "model_name"}
            extra_args: 额外的命令行参数（可选）
            timeout: 超时时间（秒）

        Returns:
            (success, stdout, stderr) 元组

        Raises:
            SecurityError: 如果任何参数包含不安全字符
        """
        if options:
            cmd = build_safe_command_with_options(binary, options, extra_args)
        else:
            cmd = build_safe_command(binary, extra_args=extra_args)

        return ShellRunner._run_cmd(cmd, timeout=timeout)

    @staticmethod
    def _run_cmd(cmd: List[str], timeout: Optional[int] = None):
        """
        内部方法：执行命令列表。

        注意：此方法为内部使用，不对外暴露。所有外部调用应使用 run_safe_cmd。

        Args:
            cmd: 已验证的命令列表
            timeout: 超时时间（秒）

        Returns:
            (success, stdout, stderr) 元组
        """
        get_logger().debug("[ShellRunner] Executing: %r", ' '.join(cmd))
        try:
            result = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace',  # nosec B603
            )
            if result.returncode != 0:
                get_logger().error("[ShellRunner] Command failed: %r", result.stderr)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            get_logger().error("[ShellRunner] Command timed out")
            return False, "", "Timeout"


class AsyncProcess:
    """用于管理 vllm 这种需要长期运行的服务进程"""

    def __init__(
        self,
        binary: str,
        log_file: str,
        options: Optional[dict] = None,
        extra_args: Optional[Union[str, List[str]]] = None,
        env: Optional[dict] = None,
    ):
        """
        初始化异步进程（使用安全命令构建）。

        Args:
            binary: 可执行文件路径或命令名称
            log_file: 日志文件路径
            options: 选项字典，格式为 {option_name: value}，例如 {"--port": "8000"}
            extra_args: 额外参数（可选）
            env: 环境变量字典（可选），会合并到当前环境变量中

        Raises:
            SecurityError: 如果任何参数包含不安全字符
        """
        if options:
            self.cmd = build_safe_command_with_options(binary, options, extra_args)
        else:
            self.cmd = build_safe_command(binary, extra_args=extra_args)
        self.process = None
        self.log_file = open(log_file, 'w')  # noqa: SIM115
        # 合并环境变量
        if env:
            self.env = {**os.environ, **env}
        else:
            self.env = None

    def start(self):
        get_logger().debug("[AsyncProcess] Starting Async Process: %r", ' '.join(self.cmd))
        # 使用 start_new_session=True 创建新的进程组，方便后续能够杀掉整个进程树
        popen_kwargs = {
            'args': self.cmd,
            'shell': False,
            'stdout': self.log_file,
            'stderr': subprocess.STDOUT,
            'start_new_session': True,  # 创建新的进程组
        }
        if self.env:
            # 确保所有环境变量值均为字符串类型
            popen_kwargs['env'] = {k: str(v) for k, v in self.env.items()}
        self.process = subprocess.Popen(**popen_kwargs)  # nosec B603

    def stop(self):
        if self.process:
            get_logger().debug("[AsyncProcess] Stopping process PID: %r", self.process.pid)
            try:
                # 发送 SIGTERM 给整个进程组，确保子进程也能被杀掉
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except Exception as e:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                finally:
                    get_logger().warning("[AsyncProcess] Normal stop failed, forcing kill: %r", e)
        self.log_file.close()
