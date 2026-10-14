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
# pylint: disable=logging-fstring-interpolation

from functools import wraps
from typing import List, Optional, Any, Callable

from typing import Dict  # noqa: F401
from msmodelslim.core.tune_strategy import EvaluateAccuracy
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.plugin.plugin_utils import get_entry_points
from msmodelslim.utils.plugin.typed_factory import TypedFactory

from .base import BasePrecheckRule, BasePrecheckConfig, BasePrecheckConfigList  # noqa: F401

PRECHECK_PLUGIN_PATH = "msmodelslim.precheck_rule.plugins"


class PrecheckerFactory:
    """预检查规则工厂类，使用 TypedFactory 根据配置类型创建对应的预检查规则实例"""

    def __init__(self):
        """初始化预检查规则工厂，使用 TypedFactory 管理规则类的动态加载和实例化"""
        self._factory = TypedFactory[BasePrecheckRule](config_base_class=BasePrecheckConfig)

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """获取支持的所有预检查类型"""
        try:
            entry_points_list = get_entry_points(PRECHECK_PLUGIN_PATH)
            return [ep.name for ep in entry_points_list]
        except Exception:
            get_logger().warning("Failed to get supported precheck types from entry_points")
            return []

    @classmethod
    def is_supported_type(cls, precheck_type: str) -> bool:
        """检查预检查类型是否支持"""
        return precheck_type in cls.get_supported_types()

    def create(self, config: BasePrecheckConfig) -> BasePrecheckRule:
        """根据配置字典创建预检查规则实例"""
        return self._factory.create(config)


class ModelOutputPrechecker:
    """
    模型输出预检查器管理器。

    统一管理多个预检查器，支持多种预检类型。
    """

    def __init__(self, configs: Optional[List[BasePrecheckConfig]] = None):
        self.precheckers: List[BasePrecheckRule] = []
        self._factory = PrecheckerFactory()
        if configs:
            for config in configs:
                try:
                    self.precheckers.append(self._factory.create(config))
                except Exception as e:
                    get_logger().warning(f"Failed to create prechecker from config {config}: {e}. Skipping.")

    def check(
        self, host: str, port: int, served_model_name: str, datasets: List[str]
    ) -> Optional[List[EvaluateAccuracy]]:
        """
        执行所有预检查。

        依次执行所有配置的预检查器，如果任何一个预检查失败，则返回失败结果。

        如果检查失败，返回不达标的结果列表；如果检查通过，返回 None
        """
        if not self.precheckers:
            get_logger().debug("No precheckers configured. Skipping precheck.")
            return None

        get_logger().info(f"Running {len(self.precheckers)} prechecker(s)...")

        for idx, prechecker in enumerate(self.precheckers, 1):
            get_logger().debug(f"Running prechecker {idx}/{len(self.precheckers)}: {type(prechecker).__name__}")
            result = prechecker.check(host=host, port=port, served_model_name=served_model_name, datasets=datasets)
            if result is not None:
                get_logger().warning(f"Prechecker {idx} failed. Skipping evaluation stage.")
                return result

        get_logger().info("All precheckers passed. Proceeding with evaluation.")
        return None


def model_output_precheck(func: Callable) -> Callable:
    """
    模型输出预检查装饰器。

    如果配置中启用了预检查，则在执行测评前进行检测。

    使用示例：
        @model_output_precheck
        def run(self):
            ...

    Returns:
        装饰后的函数
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        # 从实例中获取预检相关配置
        eval_config = getattr(self, 'eval_config', None)
        if not eval_config:
            return func(self, *args, **kwargs)

        precheck_config = getattr(eval_config, 'precheck', None)

        if precheck_config is None:
            return func(self, *args, **kwargs)

        # 只支持列表格式
        if not isinstance(precheck_config, list):
            get_logger().warning(
                f"Precheck config must be a list format, got {type(precheck_config)}. Skipping precheck."
            )
            return func(self, *args, **kwargs)

        if not precheck_config:
            return func(self, *args, **kwargs)

        # 从配置中获取服务器信息
        host = getattr(eval_config, 'host', None)
        port = getattr(eval_config, 'port', None)
        served_model_name = getattr(eval_config, 'served_model_name', None)

        # 如果缺少必要的服务器信息，跳过预检查
        if not all([host, port, served_model_name]):
            get_logger().warning(
                "Precheck enabled but missing server info (host, port, or served_model_name). Skipping precheck."
            )
            return func(self, *args, **kwargs)

        # 获取数据集列表（用于构建返回结果）
        datasets = getattr(self, 'datasets', [])

        # 使用 ModelOutputPrechecker 执行预检查
        prechecker = ModelOutputPrechecker(configs=precheck_config)
        result = prechecker.check(host=host, port=port, served_model_name=served_model_name, datasets=datasets)

        if result is not None:
            return result

        return func(self, *args, **kwargs)

    return wrapper
