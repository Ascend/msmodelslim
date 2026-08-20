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

import importlib.util
import os
import re
from decimal import Decimal
import shlex
import time
from pathlib import Path
from typing import Annotated
from typing import Dict, Literal, List, Optional

from pydantic import BaseModel, Field, AfterValidator

from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateContext
from msmodelslim.core.tune_strategy import EvaluateAccuracy
from msmodelslim.infra.evaluation.precheck import model_output_precheck
from msmodelslim.infra.evaluation.precheck.base import BasePrecheckConfigList
from msmodelslim.utils.exception import SpecError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import ShellRunner
from msmodelslim.utils.security.path import json_safe_load
from msmodelslim.utils.security.shell import build_safe_command_with_options
from msmodelslim.utils.validation.pydantic import (
    is_safe_host,
    is_port,
    greater_than_zero,
    non_empty_string,
    validate_str_length,
)


class ModelConfigMeta(BaseModel):
    """模型配置元数据"""

    directory: Annotated[str, AfterValidator(validate_str_length())] = Field(
        default="", description="模型配置目录的显式路径，空字符串表示使用默认路径"
    )
    subdir: Annotated[str, AfterValidator(validate_str_length())] = Field(
        default="vllm_api", description="模型配置子目录"
    )
    base_name: Annotated[str, AfterValidator(validate_str_length())] = Field(
        default="vllm_api_general_chat", description="模型配置基础名称"
    )
    name_suffix: Annotated[str, AfterValidator(validate_str_length())] = Field(
        default="auto", description="模型配置名称后缀，'auto'表示自动生成"
    )
    abbr: Annotated[str, AfterValidator(validate_str_length())] = Field(
        default="vllm-api-general-chat", description="模型配置缩写"
    )
    attr: Annotated[str, AfterValidator(validate_str_length())] = Field(default="service", description="模型配置属性")


class AisbenchConfig(BaseModel):
    """AISBench 评测配置"""

    binary: Literal['ais_bench'] = Field(default="ais_bench", description="aisbench 启动命令，固定为 'ais_bench'")
    mode: Annotated[str, AfterValidator(validate_str_length())] = Field(default="all", description="评测模式")
    timeout: Annotated[int, AfterValidator(greater_than_zero)] = Field(
        default=7200, description="命令执行超时时间（秒），默认2小时"
    )
    cleanup_model_config: bool = Field(default=True, description="是否清理生成的模型配置文件")
    model_meta: ModelConfigMeta = Field(default_factory=ModelConfigMeta, description="模型配置元数据")
    request_rate: Annotated[float, AfterValidator(greater_than_zero)] = Field(
        default=1.0, description="默认请求速率，必须 > 0"
    )
    pred_postprocessor: Annotated[str, AfterValidator(validate_str_length())] = Field(
        default="extract_non_reasoning_content", description="预测后处理器名称"
    )
    retry: int = Field(default=2, ge=0, description="请求重试次数，必须 >= 0")
    batch_size: Annotated[int, AfterValidator(greater_than_zero)] = Field(default=1, description="批处理大小，必须 > 0")
    max_out_len: Annotated[int, AfterValidator(greater_than_zero)] = Field(
        default=512, description="最大输出长度，必须 > 0"
    )
    trust_remote_code: bool = Field(default=False, description="是否信任远程代码")
    generation_kwargs: Dict = Field(default_factory=dict, description="生成参数配置字典")
    extra_args: List[str] = Field(default_factory=list, description="额外的命令行参数列表，默认为空列表")
    log_dir: Annotated[str, AfterValidator(validate_str_length())] = Field(
        default="", description="日志目录路径，空字符串表示使用默认路径"
    )


class DatasetConfig(BaseModel):
    """单个数据集的评测配置"""

    config_name: Annotated[
        str,
        AfterValidator(non_empty_string),
        AfterValidator(validate_str_length()),
    ] = Field(..., description="数据集在 ais_bench 中的配置名称（必需）")
    mode: Annotated[
        str,
        AfterValidator(validate_str_length()),
    ] = Field(default="", description="该数据集的评测模式，空字符串表示使用全局模式")
    request_rate: float = Field(default=0.0, ge=0.0, description="该数据集的请求速率，0.0 表示使用全局默认值")
    max_out_len: Annotated[
        Optional[int],
        AfterValidator(greater_than_zero),
    ] = Field(default=None, description="该数据集的最大输出长度，None 表示使用全局默认值")
    returns_tool_calls: Optional[bool] = Field(default=None, description="是否返回工具调用，None 表示不写入该字段")
    api_chat_type: Annotated[
        str,
        AfterValidator(non_empty_string),
        AfterValidator(validate_str_length()),
    ] = Field(default="VLLMCustomAPIChat", description="该数据集使用的 API Chat 类型")
    chat_template_kwargs: Dict = Field(
        default_factory=dict, description="chat_template 的额外参数，例如 aime25 需要 {\"thinking\": True}"
    )
    extra_args: List[str] = Field(default_factory=list, description="该数据集额外的命令行参数列表，默认为空列表")


class AisbenchServerConfig(BaseModel):
    """AISBench 评测服务配置"""

    type: Literal['aisbench'] = Field(default='aisbench', description="评测服务类型，固定为 `aisbench`")
    aisbench: AisbenchConfig = Field(default_factory=AisbenchConfig, description="AISBench 评测配置")
    datasets: Dict[str, DatasetConfig] = Field(default_factory=dict, description="数据集配置字典，键为数据集名称")
    host: Annotated[str, AfterValidator(is_safe_host)] = Field(
        default="localhost", description="评测服务监听地址，须为合法主机名/IP"
    )
    port: Annotated[int, AfterValidator(is_port)] = Field(default=1234, description="评测服务监听端口，须为合法端口号")
    served_model_name: Annotated[str, AfterValidator(non_empty_string), AfterValidator(validate_str_length())] = Field(
        default='served_model_name', description="已部署的模型名称"
    )
    precheck: BasePrecheckConfigList = Field(
        default_factory=list,
        description="模型预检配置列表，每个元素是一个字典，包含 'type' 字段（'garbled_text' 或 'expected_answer'）",
    )


class AisBenchServer:
    """
    负责生成 ais_bench 所需的 model 配置文件，并对每个数据集执行评测。
    """

    def __init__(
        self,
        context: EvaluateContext,
        eval_config: AisbenchServerConfig,
        datasets: List[str],
        quantized_model_path: Path,
        current_run_dir: Path,
    ):
        """
        Args:
            eval_config: 评测配置
            datasets: 数据集列表
            quantized_model_path: 本轮量化权重的保存目录/文件
            current_run_dir: 工作区当前运行目录，用于保存日志
        """
        self.context = context
        self.eval_config = eval_config
        self.ais_config = eval_config.aisbench
        self.quantized_model_path = quantized_model_path
        self.current_run_dir = current_run_dir
        self.datasets = datasets
        self.dataset_configs = eval_config.datasets

        self.model_config_dir: Path = Path()
        self.model_config_name: str = ""
        self.model_config_path: Path = Path()

    @model_output_precheck
    def run(self) -> List[EvaluateAccuracy]:
        results: List[EvaluateAccuracy] = []
        if not self.datasets:
            get_logger().warning("[AISBench] Dataset is empty. Nothing to benchmark.")
            return []

        try:
            self._prepare_model_config_handle()
        except Exception as exc:
            raise SpecError("Failed to prepare ais_bench model config directory") from exc

        get_logger().debug("[AISBench] Starting AISBench evaluation for %r", self.datasets)

        def fail_to_evaluate_dataset(dataset_type: str, log_content: str, *variables):
            get_logger().error(log_content, *variables)
            results.append(EvaluateAccuracy(dataset=dataset_type, accuracy=Decimal('0')))

        try:
            for dataset_name in self.datasets:
                dataset_key = str(dataset_name)
                if dataset_key not in self.dataset_configs:
                    fail_to_evaluate_dataset(dataset_name, "[AISBench] No dataset config for %r", dataset_name)
                    continue

                dataset_cfg = self.dataset_configs[dataset_key]
                get_logger().debug("[AISBench] Dataset: %s", dataset_name)

                try:
                    if dataset_cfg.request_rate > 0.0:
                        request_rate = dataset_cfg.request_rate
                    else:
                        request_rate = self.ais_config.request_rate
                    max_out_len = (
                        dataset_cfg.max_out_len if dataset_cfg.max_out_len is not None else self.ais_config.max_out_len
                    )
                    self._write_model_config(
                        max_out_len=max_out_len,
                        request_rate=request_rate,
                        returns_tool_calls=dataset_cfg.returns_tool_calls,
                        api_chat_type=dataset_cfg.api_chat_type,
                        chat_template_kwargs=dataset_cfg.chat_template_kwargs,
                    )
                    cmd_options = self._build_command_options(dataset_cfg)
                except Exception as exc:
                    fail_to_evaluate_dataset(
                        dataset_name, "[AISBench] Failed to prepare evaluation for %r: %r", dataset_name, exc
                    )
                    continue

                success, stdout, stderr = ShellRunner.run_safe_cmd(
                    binary=self.ais_config.binary,
                    options=cmd_options['options'],
                    extra_args=cmd_options['extra_args'],
                    timeout=self.ais_config.timeout,
                )
                cmd = self._build_command_string(cmd_options)
                log_file = self._write_log(dataset_name, cmd, stdout, stderr)

                if not success:
                    fail_to_evaluate_dataset(
                        dataset_name, "AISBench command failed for %r. See log: %r", dataset_name, log_file
                    )
                    continue

                accuracy = self._parse_accuracy(stdout + "\n" + stderr, dataset_name)
                get_logger().debug("AISBench result for %r: %r", dataset_name, accuracy)
                results.append(EvaluateAccuracy(dataset=dataset_name, accuracy=accuracy))
        finally:
            self._cleanup_model_config()

        return results

    # ------------------------------------------------------------------ #
    # 配置生成
    # ------------------------------------------------------------------ #
    def _prepare_model_config_handle(self):
        """定位 ais_bench 的 models 配置目录，并生成当前 run 专属的配置名。"""
        model_meta = self.ais_config.model_meta

        if model_meta.directory:
            self.model_config_dir = Path(model_meta.directory)
        else:
            spec = importlib.util.find_spec("ais_bench")
            if not spec or not spec.submodule_search_locations:
                raise ImportError("Cannot locate ais_bench package on PYTHONPATH.")
            base_dir = Path(spec.submodule_search_locations[0])
            self.model_config_dir = base_dir / "benchmark" / "configs" / "models" / model_meta.subdir

        if not self.model_config_dir.exists():
            raise FileNotFoundError(f"Model config directory not found: {self.model_config_dir}")

        base_name = model_meta.base_name.replace('.py', '')
        # 确保 base_name 只包含安全字符
        base_name = re.sub(r'[^0-9A-Za-z_\-]+', '_', base_name)

        suffix = model_meta.name_suffix
        if suffix in ('', 'auto'):
            suffix = f"trial_{self.context.evaluate_id}_{int(time.time())}"
        # 确保 suffix 只包含安全字符
        suffix = re.sub(r'[^0-9A-Za-z_\-]+', '_', suffix)

        self.model_config_name = f"{base_name}_{suffix}"
        self.model_config_path = self.model_config_dir / f"{self.model_config_name}.py"
        get_logger().debug("AISBench model config will be written to: %s", self.model_config_path)

    def _write_model_config(
        self,
        max_out_len: int,
        request_rate: float = 0.0,
        returns_tool_calls: Optional[bool] = None,
        api_chat_type: str = "VLLMCustomAPIChat",
        chat_template_kwargs: Optional[Dict] = None,
    ):
        """根据当前量化结果生成 vllm_api model config。"""
        if not self.model_config_path or str(self.model_config_path) == ".":
            raise RuntimeError("Model config path has not been prepared.")

        cfg = self.ais_config
        model_meta = cfg.model_meta

        postproc_field = ""
        import_line = ""
        if cfg.pred_postprocessor:
            import_line = f"from ais_bench.benchmark.utils.model_postprocessors import {cfg.pred_postprocessor}\n"
            postproc_field = f"        pred_postprocessor=dict(type={cfg.pred_postprocessor})\n"

        request_rate_value = request_rate if request_rate > 0.0 else cfg.request_rate
        max_out_len_value = max_out_len
        host_port_value = self.eval_config.port

        chat_template_kwargs_field = ""
        if chat_template_kwargs:
            # 如果后面还有 postproc_field，需要加逗号
            comma = "," if postproc_field else ""
            chat_template_kwargs_field = f"        chat_template_kwargs={repr(chat_template_kwargs)}{comma}\n"

        # Only include returns_tool_calls if explicitly specified in yaml (not None)
        returns_tool_calls_field = ""
        if returns_tool_calls is not None:
            returns_tool_calls_field = f"        returns_tool_calls={returns_tool_calls},\n"

        content = (
            f"from ais_bench.benchmark.models import {api_chat_type}\n"
            f"{import_line}"
            "models = [\n"
            "    dict(\n"
            f"        attr={repr(model_meta.attr)},\n"
            f"        type={api_chat_type},\n"
            f"        abbr={repr(model_meta.abbr)},\n"
            f"        path={repr(str(self.quantized_model_path))},\n"
            f"        model={repr(self.eval_config.served_model_name)},\n"
            f"        request_rate={request_rate_value},\n"
            f"        retry={cfg.retry},\n"
            f"        host_ip={repr(self.eval_config.host)},\n"
            f"        host_port={repr(host_port_value)},\n"
            f"        max_out_len={max_out_len_value},\n"
            f"        batch_size={cfg.batch_size},\n"
            f"{returns_tool_calls_field}"
            f"        trust_remote_code={cfg.trust_remote_code},\n"
            f"        generation_kwargs={repr(cfg.generation_kwargs)},\n"
            f"{chat_template_kwargs_field}{postproc_field}\n"
            "    )\n"
            "]\n"
        )

        self.model_config_path.write_text(content, encoding='utf-8')
        get_logger().debug("[AISBench] Written model config to: %s", self.model_config_path)

    def _cleanup_model_config(self):
        if self.ais_config.cleanup_model_config and self.model_config_path and self.model_config_path.exists():
            try:
                self.model_config_path.unlink()
                get_logger().debug("Removed temporary AISBench model config: %s", self.model_config_path)
            except Exception as exc:
                get_logger().warning("Failed to remove AISBench model config %s: %s", self.model_config_path, exc)

    # ------------------------------------------------------------------ #
    # 运行 & 解析
    # ------------------------------------------------------------------ #
    def _build_command_options(self, dataset_cfg: DatasetConfig) -> dict:
        """构建命令选项和额外参数，用于安全执行。"""
        if not dataset_cfg.config_name:
            raise KeyError("Dataset config must provide 'config_name' for ais_bench.")

        mode = dataset_cfg.mode if dataset_cfg.mode else self.ais_config.mode

        # 合并全局和数据集特定的额外参数
        combined_extra_args = []
        for candidate in (self.ais_config.extra_args, dataset_cfg.extra_args):
            if candidate:
                combined_extra_args.extend(candidate)

        options = {
            "--models": self.model_config_name,
            "--datasets": dataset_cfg.config_name,
            "--mode": mode,
            "--work-dir": self.context.working_dir / 'aisbench_output',
        }

        return {'options': options, 'extra_args': combined_extra_args if combined_extra_args else None}

    def _build_command_string(self, cmd_options: dict) -> str:
        """构建命令字符串用于日志记录。"""
        cmd_list = build_safe_command_with_options(
            binary=self.ais_config.binary, options=cmd_options['options'], extra_args=cmd_options['extra_args']
        )
        # 将列表转换为字符串用于日志记录
        return shlex.join(cmd_list)

    def _write_log(self, dataset_alias: str, cmd: str, stdout: str, stderr: str) -> str:
        log_dir = (
            self.ais_config.log_dir if self.ais_config.log_dir else os.path.join(self.current_run_dir, "aisbench_logs")
        )
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{dataset_alias}_trial{self.context.evaluate_id}.log")

        Path(log_path).write_text(f"[COMMAND]\n{cmd}\n\n[STDOUT]\n{stdout}\n\n[STDERR]\n{stderr}\n", encoding='utf-8')

        get_logger().info("AISBench logs for %s saved to: %s", dataset_alias, log_path)
        return log_path

    def _parse_accuracy(self, logs: str, dataset_alias: str) -> Decimal:
        """
        从 AISBench 日志中解析精度。

        解析流程：
        1. 从日志中找到 "Current exp folder: " 后面的路径
        2. 在该路径下的 results/{abbr}/ 目录下查找 JSON 文件
        3. 从 JSON 文件中读取 accuracy 字段
        """
        # 1. 从日志中提取工作路径
        exp_folder_pattern = r"Current exp folder:\s*(.+?)(?:\r?\n|$)"
        match = re.search(exp_folder_pattern, logs, re.IGNORECASE | re.MULTILINE)
        if not match:
            get_logger().warning(
                "[AISBench] Could not find 'Current exp folder': in AISBench logs for %s. Defaulting to 0.0.",
                dataset_alias,
            )
            return Decimal('0')

        exp_folder = Path(match.group(1).strip())
        get_logger().debug("Found AISBench exp folder: %s", exp_folder)

        # 2. 构建 results/{abbr} 目录路径
        abbr = self.ais_config.model_meta.abbr
        results_dir = exp_folder / "results" / abbr

        if not results_dir.exists():
            error_msg = (
                f"[AISBench] Results directory not found: {results_dir} for {dataset_alias}. "
                f"AISBench evaluation may have failed. Please check if the configuration is correct. "
                f"Please check the AISBench logs in directory: {exp_folder}"
            )
            get_logger().error(error_msg)
            raise SpecError(error_msg)

        # 3. 查找 JSON 文件（通常只有一个 JSON 文件）
        json_files = list(results_dir.glob("*.json"))
        if not json_files:
            get_logger().warning(
                "[AISBench] No JSON file found in %s for %s. Defaulting to 0.0.", results_dir, dataset_alias
            )
            return Decimal('0')

        # 使用第一个找到的 JSON 文件
        json_file = json_files[0]
        get_logger().debug("Reading accuracy from JSON file: %s", json_file)

        try:
            # 4. 使用安全的 JSON 加载函数读取文件并提取 accuracy
            data = json_safe_load(str(json_file))
        except Exception as e:
            get_logger().warning(
                "[AISBench] Failed to load JSON file %s for %s: %s. Defaulting to 0.0.", json_file, dataset_alias, e
            )
            return Decimal('0')

        if 'accuracy' not in data:
            get_logger().warning(
                "[AISBench] No 'accuracy' field found in %s for %s. Defaulting to 0.0.", json_file, dataset_alias
            )
            return Decimal('0')

        # 使用字符串构造 Decimal，避免 JSON 浮点与二进制浮点误差
        accuracy = Decimal(str(data['accuracy']))
        get_logger().debug("[AISBench] Parsed accuracy from JSON: %s for %s", accuracy, dataset_alias)
        return accuracy

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
