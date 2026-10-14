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

import os
import unittest
from dataclasses import dataclass, replace
from typing import List, Any, Generator, Literal
from typing import Optional
from unittest.mock import patch, MagicMock

from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1 import AscendV1Config
from msmodelslim.core.quant_service.modelslim_v1.save.ascendv1_distributed import DistributedAscendV1Config
from msmodelslim.core.runner.dp_layer_wise_runner import DPLayerWiseRunner
from msmodelslim.core.runner.pipeline_interface import PipelineInterface
from msmodelslim.processor.base import AutoProcessorConfig, AutoSessionProcessor
from msmodelslim.utils.exception import UnsupportedError


def is_npu_available():
    """检查NPU是否可用"""
    try:
        import torch_npu

        return torch_npu.npu.is_available()
    except ImportError:
        return False


@dataclass
class WorkerParams:
    """Parameters for the distributed_worker method."""

    rank: int
    world_size: int
    device_indices: List[int]
    model: Optional[nn.Module]
    calib_data: Optional[List[Any]]
    device: DeviceType
    master_port: int = 29500


class MockPipelineInterface(PipelineInterface):
    """Mock PipelineInterface for testing"""

    def __init__(self):
        self._model_path = "/mock/model/path"
        self._model_type = "mock_model"
        self._trust_remote_code = False
        self._init_model_called = False

    @property
    def model_path(self):
        return self._model_path

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def trust_remote_code(self) -> bool:
        return self._trust_remote_code

    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        return dataset if dataset else []

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        self._init_model_called = True
        return nn.Linear(10, 5)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        yield ProcessRequest(name="test", module=model, args=(), kwargs={})

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        yield ProcessRequest(name="test", module=model, args=inputs, kwargs={})

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        pass


class MockProcessorConfig(AutoProcessorConfig):
    """Mock processor config for testing"""

    type: Literal['mock_processor'] = "mock_processor"


@unittest.skipIf(not is_npu_available(), "NPU not available")
class TestDPLayerWiseRunnerInit(unittest.TestCase):
    """测试 DPLayerWiseRunner 初始化"""

    def test_init_with_default_and_custom_params(self):
        """测试默认和自定义参数初始化"""
        adapter = MockPipelineInterface()

        # 默认参数
        runner = DPLayerWiseRunner(adapter)
        self.assertEqual(runner.backend, 'hccl')
        self.assertEqual(runner.offload_device, 'meta')

        # 自定义参数
        runner = DPLayerWiseRunner(adapter, offload_device='cpu', backend='nccl')
        self.assertEqual(runner.backend, 'nccl')
        self.assertEqual(runner.offload_device, 'cpu')


@unittest.skipIf(not is_npu_available(), "NPU not available")
class TestConvertToDistributedConfig(unittest.TestCase):
    """测试 convert_to_distributed_config_if_needed 静态方法"""

    def test_convert_ascendv1_config(self):
        """测试将 AscendV1Config 转换为 DistributedAscendV1Config"""
        original_config = AscendV1Config(
            type="ascendv1_saver", save_directory="/test/path", part_file_size=8, ext={"key": "value"}
        )

        converted = DPLayerWiseRunner.convert_to_distributed_config_if_needed(original_config)

        self.assertIsInstance(converted, DistributedAscendV1Config)
        self.assertEqual(converted.save_directory, "/test/path")
        self.assertEqual(converted.part_file_size, 8)
        self.assertEqual(converted.type, "ascendv1_saver_distributed")

    def test_no_convert_already_distributed_config(self):
        """测试不转换已经是 DistributedAscendV1Config 的配置"""
        original_config = DistributedAscendV1Config(type="ascendv1_saver_distributed", save_directory="/test/path")

        converted = DPLayerWiseRunner.convert_to_distributed_config_if_needed(original_config)
        self.assertIs(converted, original_config)

    def test_no_convert_other_config(self):
        """测试不转换其他类型的配置"""
        original_config = MockProcessorConfig(type="mock_processor")
        converted = DPLayerWiseRunner.convert_to_distributed_config_if_needed(original_config)
        self.assertIs(converted, original_config)


@unittest.skipIf(not is_npu_available(), "NPU not available")
class TestAddProcessor(unittest.TestCase):
    """测试 add_processor 方法"""

    def test_add_processor_append_and_insert(self):
        """测试追加和插入处理器"""
        adapter = MockPipelineInterface()
        runner = DPLayerWiseRunner(adapter)

        config1 = MockProcessorConfig(type="mock_processor")
        config2 = MockProcessorConfig(type="mock_processor")

        runner.add_processor(config1, append=True)
        runner.add_processor(config2, append=False)

        self.assertEqual(len(runner.process_config_list), 2)
        self.assertIs(runner.process_config_list[0], config2)
        self.assertIs(runner.process_config_list[1], config1)

    def test_add_processor_auto_convert_ascendv1(self):
        """测试自动转换 AscendV1Config 为 DistributedAscendV1Config"""
        adapter = MockPipelineInterface()
        runner = DPLayerWiseRunner(adapter)

        config = AscendV1Config(type="ascendv1_saver", save_directory="/test/path")
        runner.add_processor(config, append=True)

        self.assertIsInstance(runner.process_config_list[0], DistributedAscendV1Config)


@unittest.skipIf(not is_npu_available(), "NPU not available")
class TestCheckDistributedSupport(unittest.TestCase):
    """测试 _check_distributed_support 方法"""

    @patch.object(AutoSessionProcessor, 'from_config')
    def test_check_distributed_support(self, mock_from_config):
        """测试检查处理器分布式支持"""
        adapter = MockPipelineInterface()
        runner = DPLayerWiseRunner(adapter)
        model = nn.Linear(10, 5)

        # 测试所有支持
        mock_processor = MagicMock()
        mock_processor.support_distributed.return_value = True
        mock_from_config.return_value = mock_processor

        processor_list = [MockProcessorConfig(type="mock_processor")]
        unsupported = runner._check_distributed_support(processor_list, model)
        self.assertEqual(len(unsupported), 0)

        # 测试部分不支持
        mock_processor1 = MagicMock()
        mock_processor1.support_distributed.return_value = True
        mock_processor2 = MagicMock()
        mock_processor2.support_distributed.return_value = False
        mock_from_config.side_effect = [mock_processor1, mock_processor2]

        processor_list = [MockProcessorConfig(type="mock_processor")] * 2
        unsupported = runner._check_distributed_support(processor_list, model)
        self.assertEqual(len(unsupported), 1)


@unittest.skipIf(not is_npu_available(), "NPU not available")
class TestRunFallbackToSingleDevice(unittest.TestCase):
    """测试 run 方法回退到单设备执行"""

    @patch('msmodelslim.core.runner.layer_wise_runner.LayerWiseRunner.run')
    @patch('msmodelslim.core.runner.generated_runner.get_input_datas')
    def test_run_fallback_to_single_device(self, mock_get_input_datas, mock_parent_run):
        """测试 device_indices 为 None 或单设备时回退到单设备"""
        adapter = MockPipelineInterface()
        runner = DPLayerWiseRunner(adapter)
        model = nn.Linear(10, 5)
        mock_get_input_datas.return_value = []

        # None
        runner.run(model=model, device_indices=None)
        self.assertEqual(mock_parent_run.call_count, 1)

        # 单设备
        runner.run(model=model, device_indices=[0])
        self.assertEqual(mock_parent_run.call_count, 2)

        # 空列表
        runner.run(model=model, device_indices=[])
        self.assertEqual(mock_parent_run.call_count, 3)


@unittest.skipIf(not is_npu_available(), "NPU not available")
class TestRunDistributed(unittest.TestCase):
    """测试 run 方法的分布式执行"""

    @patch('torch.multiprocessing.spawn')
    @patch('torch.multiprocessing.set_start_method')
    @patch.object(AutoSessionProcessor, 'from_config')
    @patch('msmodelslim.core.runner.generated_runner.get_input_datas')
    def test_run_distributed_execution(self, mock_get_input_datas, mock_from_config, mock_set_start_method, mock_spawn):
        """测试多设备分布式执行"""
        adapter = MockPipelineInterface()
        runner = DPLayerWiseRunner(adapter)
        model = nn.Linear(10, 5)

        mock_get_input_datas.return_value = []
        mock_processor = MagicMock()
        mock_processor.support_distributed.return_value = True
        mock_from_config.return_value = mock_processor

        runner.add_processor(MockProcessorConfig(type="mock_processor"))
        runner.run(model=model, device_indices=[0, 1])

        mock_set_start_method.assert_called_once_with('spawn', force=True)
        mock_spawn.assert_called_once()

    @patch.object(AutoSessionProcessor, 'from_config')
    @patch('msmodelslim.core.runner.generated_runner.get_input_datas')
    def test_run_with_unsupported_processors_raises_error(self, mock_get_input_datas, mock_from_config):
        """测试存在不支持分布式的处理器时抛出异常"""
        adapter = MockPipelineInterface()
        runner = DPLayerWiseRunner(adapter)
        model = nn.Linear(10, 5)

        mock_get_input_datas.return_value = []
        mock_processor = MagicMock()
        mock_processor.support_distributed.return_value = False
        mock_from_config.return_value = mock_processor

        runner.add_processor(MockProcessorConfig(type="mock_processor"))

        with self.assertRaises(UnsupportedError) as context:
            runner.run(model=model, device_indices=[0, 1])
        self.assertIn("do not support distributed quantization", str(context.exception))

    @patch('msmodelslim.core.runner.dp_layer_wise_runner.find_free_port')
    @patch('torch.multiprocessing.spawn')
    @patch('torch.multiprocessing.set_start_method')
    @patch.object(AutoSessionProcessor, 'from_config')
    @patch('msmodelslim.core.runner.generated_runner.get_input_datas')
    def test_run_port_handling(
        self, mock_get_input_datas, mock_from_config, mock_set_start_method, mock_spawn, mock_find_free_port
    ):
        """测试端口处理逻辑"""
        adapter = MockPipelineInterface()
        runner = DPLayerWiseRunner(adapter)
        model = nn.Linear(10, 5)

        mock_get_input_datas.return_value = []
        mock_find_free_port.return_value = 29501
        mock_processor = MagicMock()
        mock_processor.support_distributed.return_value = True
        mock_from_config.return_value = mock_processor

        runner.add_processor(MockProcessorConfig(type="mock_processor"))

        # 测试自动查找端口
        if 'MASTER_PORT' in os.environ:
            del os.environ['MASTER_PORT']
        runner.run(model=model, device_indices=[0, 1])
        mock_find_free_port.assert_called_once()
        self.assertEqual(os.environ['MASTER_PORT'], '29501')

        # 测试使用已有端口
        os.environ['MASTER_PORT'] = '12345'
        mock_find_free_port.reset_mock()
        runner.run(model=model, device_indices=[0, 1])
        call_args = mock_spawn.call_args
        self.assertEqual(call_args[1]['args'][5], 12345)

        # 清理
        del os.environ['MASTER_PORT']


@unittest.skipIf(not is_npu_available(), "NPU not available")
class TestDistributedWorker(unittest.TestCase):
    """测试 distributed_worker 方法"""

    def setUp(self):
        """设置测试的默认参数"""
        self.adapter = MockPipelineInterface()
        self.runner = DPLayerWiseRunner(self.adapter)
        self.model = nn.Linear(10, 5)
        # 默认的 worker 参数
        self.default_params = WorkerParams(
            rank=0,
            world_size=2,
            device_indices=[0, 1],
            model=self.model,
            calib_data=None,
            device=DeviceType.NPU,
            master_port=29500,
        )

    def test_distributed_worker_execution(self):
        """测试 distributed_worker 执行流程"""
        runner_path = 'msmodelslim.core.runner.dp_layer_wise_runner'
        layer_runner_path = 'msmodelslim.core.runner.layer_wise_runner'
        with (
            patch('torch.distributed.get_rank', return_value=0),
            patch('torch.distributed.is_initialized', return_value=False),
            patch(f'{runner_path}.dist.is_initialized') as mock_is_initialized,
            patch(f'{runner_path}.setup_distributed') as mock_setup,
            patch.object(DPLayerWiseRunner, 'generated_schedule') as mock_schedule,
            patch.object(DPLayerWiseRunner, 'build_process_unit') as mock_build,
            patch(f'{layer_runner_path}.torch.npu', create=True) as mock_npu,
        ):
            mock_is_initialized.return_value = True
            mock_build.return_value = []
            mock_npu.current_device.return_value = 0

            # 使用默认参数
            worker_params = self.default_params
            self.runner.distributed_worker(**vars(worker_params))

            mock_setup.assert_called_once_with(
                worker_params.rank,
                worker_params.world_size,
                'hccl',
                device_index=worker_params.device_indices[worker_params.rank],
                master_port=worker_params.master_port,
            )
            mock_build.assert_called_once()
            mock_schedule.assert_called_once()

    def test_distributed_worker_device_index_mapping(self):
        """测试 distributed_worker 设备索引映射"""
        runner_path = 'msmodelslim.core.runner.dp_layer_wise_runner'
        layer_runner_path = 'msmodelslim.core.runner.layer_wise_runner'
        with (
            patch('torch.distributed.get_rank', return_value=0),
            patch('torch.distributed.is_initialized', return_value=False),
            patch(f'{runner_path}.dist.destroy_process_group'),
            patch(f'{runner_path}.dist.is_initialized') as mock_is_initialized,
            patch(f'{runner_path}.setup_distributed') as mock_setup,
            patch.object(DPLayerWiseRunner, 'generated_schedule'),
            patch.object(DPLayerWiseRunner, 'build_process_unit') as mock_build,
            patch(f'{layer_runner_path}.torch.npu', create=True) as mock_npu,
        ):
            mock_is_initialized.return_value = True
            mock_build.return_value = []
            mock_npu.current_device.return_value = 3

            # 修改 rank 和 device_indices
            worker_params = replace(self.default_params, rank=1, device_indices=[0, 3])

            # rank=1 对应 device_indices[1]=3
            self.runner.distributed_worker(**vars(worker_params))

            mock_setup.assert_called_once_with(
                worker_params.rank,
                worker_params.world_size,
                'hccl',
                device_index=worker_params.device_indices[worker_params.rank],
                master_port=worker_params.master_port,
            )

    def test_distributed_worker_cleanup_behavior(self):
        """测试 distributed_worker 清理行为"""
        runner_path = 'msmodelslim.core.runner.dp_layer_wise_runner'
        layer_runner_path = 'msmodelslim.core.runner.layer_wise_runner'
        with (
            patch('torch.distributed.get_rank', return_value=0),
            patch('torch.distributed.is_initialized', return_value=False),
            patch(f'{runner_path}.dist.destroy_process_group') as mock_destroy,
            patch(f'{runner_path}.dist.is_initialized') as mock_is_initialized,
            patch(f'{runner_path}.setup_distributed'),
            patch.object(DPLayerWiseRunner, 'generated_schedule'),
            patch.object(DPLayerWiseRunner, 'build_process_unit') as mock_build,
            patch(f'{layer_runner_path}.torch.npu', create=True) as mock_npu,
        ):
            mock_build.return_value = []
            mock_npu.current_device.return_value = 0

            # 使用默认参数
            worker_params = self.default_params

            # 未初始化时不调用 destroy
            mock_is_initialized.return_value = False
            self.runner.distributed_worker(**vars(worker_params))
            mock_destroy.assert_not_called()


if __name__ == '__main__':
    unittest.main()
