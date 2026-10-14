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
# pylint: disable=consider-using-from-import,no-value-for-parameter

import unittest
from pathlib import Path
from typing import List
from unittest.mock import patch, Mock, MagicMock

import torch
import torch.nn as nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model.deepseek_v3_2.model_adapter import DeepSeekV32ModelAdapter
from msmodelslim.utils.exception import InvalidModelError


class DummyModelArgs:
    """模拟ModelArgs配置类"""

    def __init__(self):
        self.num_hidden_layers = 2
        self.qk_nope_head_dim = 64
        self.v_head_dim = 64
        self.num_attention_heads = 8
        self.num_key_value_heads = 4
        self.rms_norm_eps = 1e-6
        self.vocab_size = 1000
        self.hidden_size = 128
        self.first_k_dense_replace = 3  # 前k层使用Dense FFN，之后使用MoE
        self.n_routed_experts = 8  # MoE路由专家数量
        self.n_shared_experts = 1  # MoE共享专家数量


class DummyDeepSeekV3RMSNorm(nn.Module):
    """模拟DeepseekV3RMSNorm归一化层"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return hidden_states * self.weight


class DummySharedHead(nn.Module):
    """模拟SharedHead类（MTP层依赖）"""

    def __init__(self, config):
        super().__init__()
        self.norm = DummyDeepSeekV3RMSNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.head(self.norm(hidden_states))


class DummyMTPLayer(nn.Module):
    """模拟MTPLayer类"""

    def __init__(self, config):
        super().__init__()
        self.enorm = DummyDeepSeekV3RMSNorm(config.hidden_size)
        self.hnorm = DummyDeepSeekV3RMSNorm(config.hidden_size)
        self.shared_head = DummySharedHead(config)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


class DummyDecoderLayer(nn.Module):
    """模拟解码器层"""

    def __init__(self, layer_id=0, args=None):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.shared_head = None  # 默认无MTP相关属性
        self.hook_id = 0
        self._forward_hooks = {}
        self._forward_pre_hooks_with_kwargs = {}  # 接收kwargs的钩子

    def get_submodule(self, name):
        if name == "shared_head" and hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"No submodule named {name}")

    def register_forward_pre_hook(self, hook, with_kwargs=True, prepend=True):
        current_id = self.hook_id
        self.hook_id += 1
        self._forward_pre_hooks_with_kwargs[current_id] = hook

        def remove(*args, **kwargs):
            if current_id in self._forward_pre_hooks_with_kwargs:
                del self._forward_pre_hooks_with_kwargs[current_id]

        return type('', (), {'remove': remove})()

    def forward(self, hidden_states, **kwargs):
        # 处理需要接收kwargs的钩子
        for _, hook in self._forward_pre_hooks_with_kwargs.items():
            args_kwargs_result = hook(self, (hidden_states,), kwargs)
            if args_kwargs_result is not None:
                if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
                    (hidden_states,), kwargs = args_kwargs_result
                else:
                    raise RuntimeError(
                        "forward pre-hook must return None or a tuple "
                        f"of (new_args, new_kwargs), but got {args_kwargs_result}."
                    )

        return hidden_states


class DummyModelInner(nn.Module):
    """模拟模型内部的model对象（含layers、norm、freqs_cis）"""

    def __init__(self, num_layers=2, config=None):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer(layer_id=i, args=config) for i in range(num_layers)])
        self.norm = DummyDeepSeekV3RMSNorm(config.hidden_size if config else 128)
        self.freqs_cis = torch.randn(100, 128)

    def forward(self, hidden_states, **kwargs):
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)  # 执行当前层
        return self.norm(hidden_states)

    def get_all_param_names(self) -> List[str]:
        return [name for name, _ in self.named_parameters()]


class DummyModel(nn.Module):
    """模拟整体模型（含model、lm_head）"""

    def __init__(self, config=None):
        super().__init__()
        self.model = DummyModelInner(num_layers=config.num_hidden_layers if config else 2, config=config)
        self.lm_head = nn.Linear(
            config.hidden_size if config else 128,
            config.vocab_size if config else 1000,
            bias=True,  # 匹配lm_head.bias参数
        )

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden_states = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        return self.lm_head(hidden_states)

    def generate_full_state_dict(self):
        """生成完整state_dict，避免加载缺失键"""
        state_dict = {}
        for name, param in self.model.named_parameters():
            state_dict[f"model.{name}"] = param.data.clone()
        for name, param in self.lm_head.named_parameters():
            state_dict[f"lm_head.{name}"] = param.data.clone()
        return state_dict


class TestDeepSeekV32ModelAdapter(unittest.TestCase):
    def setUp(self):
        """初始化测试环境（统一配置，避免重复）"""
        self.model_path = Path(".")
        self.model_type = "DeepSeek-V3.2-Exp"
        self.dummy_config = DummyModelArgs()
        self.dummy_config.num_hidden_layers = 62  # 匹配真实模型层数
        self.test_device = "cpu"
        self.dummy_full_state_dict = DummyModel(config=self.dummy_config).generate_full_state_dict()
        self.adapter_patcher = patch.object(DeepSeekV32ModelAdapter, "__init__", lambda x, model_path, model_type: None)

    def create_adapter(self, **kwargs):
        """创建并配置适配器实例的通用方法"""
        with self.adapter_patcher:
            adapter = DeepSeekV32ModelAdapter(model_path=self.model_path, model_type=self.model_type)

            for key, value in kwargs.items():
                setattr(adapter, key, value)

            if 'config' not in kwargs:
                adapter.config = self.dummy_config
            if 'model_path' not in kwargs:
                adapter.model_path = self.model_path

            return adapter

    def test_get_model_pedigree(self):
        """测试get_model_pedigree返回固定谱系"""
        try:
            adapter = self.create_adapter()
            self.assertEqual(adapter.get_model_pedigree(), "deepseek_v3_2")
        except Exception as e:
            self.fail(f"test_get_model_pedigree failed: {e}")

    def test_get_model_type(self):
        """测试get_model_type返回初始化的模型类型"""
        adapter = self.create_adapter(model_type=self.model_type)
        self.assertEqual(adapter.get_model_type(), self.model_type)

    def test_enable_kv_cache(self):
        """测试enable_kv_cache方法的调用和参数传递"""
        adapter = self.create_adapter(model_type=self.model_type)
        adapter.enable_kv_cache(Mock(), True)
        assert True  # 方法无异常执行即通过

    def test_handle_dataset(self):
        """测试handle_dataset方法调用_get_tokenized_data"""
        adapter = self.create_adapter()
        mock_dataset = Mock()
        mock_tokenized_data = [{"input_ids": torch.tensor([1, 2, 3])}]

        # Mock _get_tokenized_data方法
        adapter._get_tokenized_data = Mock(return_value=mock_tokenized_data)

        # 测试默认device参数
        result = adapter.handle_dataset(mock_dataset)
        adapter._get_tokenized_data.assert_called_once_with(mock_dataset, DeviceType.NPU)
        self.assertEqual(result, mock_tokenized_data)

        # 测试自定义device参数
        adapter._get_tokenized_data.reset_mock()
        result = adapter.handle_dataset(mock_dataset, device=DeviceType.CPU)
        adapter._get_tokenized_data.assert_called_once_with(mock_dataset, DeviceType.CPU)

    def test_generate_model_visit(self):
        """测试generate_model_visit方法"""
        adapter = self.create_adapter()
        dummy_model = DummyModel(config=self.dummy_config)

        # Mock generate_decoder_layer
        mock_blocks = [('model.layers.0', Mock()), ('model.layers.1', Mock())]
        adapter.generate_decoder_layer = Mock(return_value=mock_blocks)

        # 执行测试
        with patch(
            'msmodelslim.model.deepseek_v3_2.model_adapter.generated_decoder_layer_visit_func'
        ) as mock_visit_func:
            mock_visit_func.return_value = iter([])

            result = adapter.generate_model_visit(dummy_model)  # noqa: F841

            # 验证调用了generated_decoder_layer_visit_func
            mock_visit_func.assert_called_once()
            call_args = mock_visit_func.call_args
            self.assertEqual(call_args[0][0], dummy_model)  # model参数
            # transformer_blocks参数应该是generate_decoder_layer的返回值

    @patch('msmodelslim.model.deepseek_v3_2.model_adapter.ModelArgs')
    def test_load_config(self, mock_model_args):
        """测试_load_config方法是否正确返回ModelArgs实例"""
        mock_args_instance = Mock()
        mock_model_args.return_value = mock_args_instance

        adapter = self.create_adapter(model_type=self.model_type)
        result = adapter._load_config()

        mock_model_args.assert_called_once_with()
        self.assertEqual(result, mock_args_instance)

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.Transformer", new=DummyModel)
    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.auto_convert_module_fp8_to_bf16")
    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.get_logger")
    @patch("torch.set_default_dtype")
    def test_init_model(self, mock_set_dtype: Mock, mock_get_logger: Mock, mock_auto_convert: Mock):
        """测试init_model完整流程"""
        adapter = self.create_adapter()
        adapter.get_state_dict = Mock(return_value=self.dummy_full_state_dict)
        original_num_layers = adapter.config.num_hidden_layers  # noqa: F841

        # 执行初始化并Mock load_state_dict
        result_model = adapter.init_model(device=self.test_device)
        result_model.load_state_dict = MagicMock()
        result_model.load_state_dict(self.dummy_full_state_dict)

        # 验证核心逻辑
        self.assertIsInstance(result_model, DummyModel)
        self.assertEqual(adapter.config.num_hidden_layers, 62)  # 恢复到62层
        result_model.load_state_dict.assert_called_once_with(self.dummy_full_state_dict)
        mock_auto_convert.assert_called_once_with("", result_model, str(self.model_path))
        mock_get_logger.return_value.info.assert_any_call("Model with 62 layers totally")

        # 验证设置了默认dtype为bfloat16
        mock_set_dtype.assert_called_with(torch.bfloat16)

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.json_safe_load")
    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.os.path.join")
    def test_get_weight_map(self, mock_path_join: Mock, mock_json_load: Mock):
        """测试get_weight_map加载权重映射"""
        adapter = self.create_adapter()

        # Mock依赖返回值
        mock_json_load.return_value = {
            "weight_map": {
                "model.layers.0.self_attn.q_a_proj.weight": "model-00001.safetensors",
                "model.layers.1.self_attn.q_a_proj.weight": "model-00002.safetensors",
            }
        }
        mock_index_path = self.model_path / "model.safetensors.index.json"
        mock_path_join.return_value = mock_index_path

        # 执行测试并验证
        weight_map = adapter.get_weight_map()
        mock_path_join.assert_called_once_with(self.model_path, "model.safetensors.index.json")
        mock_json_load.assert_called_once_with(mock_index_path)
        self.assertEqual(weight_map["model.layers.0.self_attn.q_a_proj.weight"], "model-00001.safetensors")

        # 验证LRU缓存
        adapter.get_weight_map()
        self.assertEqual(mock_json_load.call_count, 1)

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.get_mtp_layer", return_value=DummyMTPLayer(DummyModelArgs()))
    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.wrap_mtp_decoder")
    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.get_logger")
    def test_load_mtp_if_not_load_missing(self, mock_get_logger: Mock, mock_wrap_mtp: Mock, mock_get_mtp: Mock):
        """测试MTP层缺失时加载"""
        adapter = self.create_adapter()

        # 准备无shared_head属性的decoder
        dummy_decoder = DummyDecoderLayer()
        if hasattr(dummy_decoder, 'shared_head'):
            del dummy_decoder.shared_head
        with self.assertRaises(AttributeError):
            _ = dummy_decoder.shared_head

        # 执行加载并验证
        adapter.load_mtp_if_not_load(mtp_decoder=dummy_decoder)
        mock_get_mtp.assert_called_once_with(config=self.dummy_config, model_path=self.model_path)
        mock_wrap_mtp.assert_called_once_with(mtp_decoder=dummy_decoder, mtp_layer=mock_get_mtp.return_value)
        mock_get_logger.return_value.info.assert_any_call("Creating MTP layer")

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.get_mtp_layer")
    def test_load_mtp_if_not_load_exist(self, mock_get_mtp: Mock):
        """测试MTP层已存在时跳过加载"""
        adapter = self.create_adapter()

        # 准备带shared_head的decoder
        dummy_decoder = DummyDecoderLayer()
        dummy_decoder.shared_head = DummySharedHead(DummyModelArgs())

        # 执行加载并验证
        adapter.load_mtp_if_not_load(mtp_decoder=dummy_decoder)
        mock_get_mtp.assert_not_called()

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.auto_convert_module_fp8_to_bf16")
    def test_load_decoder_if_not_exist_missing(self, mock_auto_convert: Mock):
        """测试decoder缺失时创建"""
        # 生成匹配的state_dict（过滤不存在的参数）
        dummy_decoder = DummyDecoderLayer(layer_id=1, args=self.dummy_config)
        actual_param_names = [name for name, _ in dummy_decoder.named_parameters()]
        mock_state_dict = {name: torch.ones(1) for name in actual_param_names if "input_layernorm.weight" not in name}

        adapter = self.create_adapter(get_state_dict=Mock(return_value=mock_state_dict))

        # 准备仅含1层的模型
        dummy_model = DummyModel(config=self.dummy_config)
        dummy_model.model.layers = nn.ModuleList([DummyDecoderLayer(layer_id=0)])

        # 执行创建并验证
        result_decoder = adapter.load_decoder_if_not_exist(model=dummy_model, name="model.layers.1", idx=1)
        self.assertIsInstance(result_decoder, DummyDecoderLayer)
        self.assertEqual(len(dummy_model.model.layers), 2)
        mock_auto_convert.assert_called_once_with("model.layers.1", result_decoder, str(self.model_path))

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.auto_convert_module_fp8_to_bf16")
    def test_load_decoder_if_exist(self, mock_auto_convert: Mock):
        """测试decoder已存在时直接返回（覆盖try分支成功路径）"""
        adapter = self.create_adapter()

        # 准备已有decoder的模型
        dummy_model = DummyModel(config=self.dummy_config)
        existing_decoder = dummy_model.model.layers[0]

        # 尝试加载已存在的decoder
        result_decoder = adapter.load_decoder_if_not_exist(model=dummy_model, name="model.layers.0", idx=0)

        # 验证返回的是已存在的decoder，且没有调用auto_convert
        self.assertEqual(result_decoder, existing_decoder)
        mock_auto_convert.assert_not_called()

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.DeepSeekV32ModelAdapter.load_mtp_if_not_load")
    def test_generate_decoder_layer(self, mock_load_mtp: Mock):
        """测试generate_decoder_layer生成所有层（最后一层加载MTP）"""
        mock_decoders = [DummyDecoderLayer(0), DummyDecoderLayer(1), DummyDecoderLayer(2)]
        adapter = self.create_adapter(
            config=Mock(num_hidden_layers=3), load_decoder_if_not_exist=Mock(side_effect=mock_decoders)
        )

        # 执行生成并验证
        dummy_model = DummyModel(config=self.dummy_config)
        layers = list(adapter.generate_decoder_layer(model=dummy_model))

        self.assertEqual(len(layers), 3)
        self.assertEqual([name for name, _ in layers], ["model.layers.0", "model.layers.1", "model.layers.2"])
        mock_load_mtp.assert_called_once_with(mock_decoders[2])

    def test_generate_model_forward_raise_error(self):
        """测试无法获取first_block_input时抛异常"""
        adapter = self.create_adapter()
        adapter.generate_model_forward.__globals__["dist"] = Mock(is_initialized=lambda: False)

        # 1. 正常创建模型
        dummy_model = DummyModel(config=self.dummy_config)
        mock_inputs = torch.randint(0, 1000, (1, 128)).float()  # 形状为(1, 128)

        # 2. 获取第一层并保存原始的钩子注册方法
        first_layer = dummy_model.model.layers[0]

        # 3. 定义无操作的钩子注册方法（不实际注册钩子）
        def no_op_register_forward_pre_hook(self, hook, with_kwargs=True, prepend=True):
            class DummyRemove:
                @staticmethod
                def remove():
                    pass

            return DummyRemove()

        # 4. 动态替换第一层的注册方法
        first_layer.register_forward_pre_hook = no_op_register_forward_pre_hook.__get__(first_layer, DummyDecoderLayer)

        # 5. 验证异常（因钩子未注册导致无法获取输入）
        with self.assertRaises(InvalidModelError) as cm:
            gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
            next(gen)

        self.assertIn("Can't get first block input", str(cm.exception))

    @patch("msmodelslim.model.deepseek_v3_2.model_adapter.dist")
    def test_generate_model_forward_with_dist_initialized(self, mock_dist):
        """测试分布式环境已初始化时的逻辑"""
        mock_dist.is_initialized.return_value = True  # 模拟分布式已初始化

        adapter = self.create_adapter()
        adapter.generate_decoder_layer = Mock(return_value=[])  # 空迭代器

        dummy_model = DummyModel(config=self.dummy_config)
        mock_inputs = torch.randint(0, 1000, (1, 10)).float()  # 正确的数据类型

        gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
        try:
            next(gen)
        except StopIteration:
            pass  # 预期的生成器结束

        mock_dist.barrier.assert_called_once()  # 验证分布式屏障被调用

    def test_generate_model_forward_processes_decoder_layers(self):
        """测试处理transformer decoder层的逻辑：最后一层调用mtp_preprocess，非最后一层不调用"""
        adapter = self.create_adapter()
        adapter.generate_model_forward.__globals__["dist"] = Mock(is_initialized=lambda: False)
        adapter.mtp_preprocess = Mock(return_value=((torch.tensor([1]), torch.tensor([2])), {}))

        # 测试1: 最后一层调用mtp_preprocess
        adapter.config.num_hidden_layers = 2
        mock_blocks = [
            ('model.layers.0', Mock()),
            ('model.layers.1', Mock()),  # 最后一层
        ]
        adapter.generate_decoder_layer = Mock(return_value=mock_blocks)

        dummy_model = DummyModel(config=self.dummy_config)
        mock_inputs = torch.randint(0, 1000, (1, 10))

        gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
        request = next(gen)
        self.assertEqual(request.name, 'model.layers.0')

        try:
            gen.send((torch.tensor([1]), torch.tensor([2])))
        except StopIteration:
            pass

        # 验证最后一层调用了mtp_preprocess
        adapter.mtp_preprocess.assert_called_once()

        # 测试2: 非最后一层不调用mtp_preprocess
        adapter.mtp_preprocess.reset_mock()
        adapter.config.num_hidden_layers = 3
        mock_blocks = [
            ('model.layers.0', Mock()),
            ('model.layers.1', Mock()),  # 不是最后一层
        ]
        adapter.generate_decoder_layer = Mock(return_value=mock_blocks)

        gen = adapter.generate_model_forward(model=dummy_model, inputs=mock_inputs)
        request1 = next(gen)
        self.assertEqual(request1.name, 'model.layers.0')

        try:
            request2 = gen.send((torch.tensor([1]), torch.tensor([2])))
            self.assertEqual(request2.name, 'model.layers.1')
        except StopIteration:
            pass

        # 验证mtp_preprocess没有被调用
        adapter.mtp_preprocess.assert_not_called()

    @patch('msmodelslim.model.deepseek_v3_2.model_adapter.remove_zero_and_shift')
    def test_mtp_preprocess_dict_and_list_inputs(self, mock_remove_zero):
        """测试mtp_preprocess处理dict和list输入的分支"""
        adapter = self.create_adapter()

        # 准备测试数据
        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, self.dummy_config.hidden_size)
        residual = torch.randn(batch_size, seq_len, self.dummy_config.hidden_size)
        input_ids = torch.randint(0, self.dummy_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Mock remove_zero_and_shift
        mock_remove_zero.return_value = input_ids

        # Mock transformers的attention mask函数
        with patch('transformers.modeling_attn_mask_utils._prepare_4d_causal_attention_mask') as mock_prepare:
            mock_prepare.return_value = torch.randn(batch_size, 1, seq_len, seq_len)

            # Mock .to() 方法避免 npu 设备错误
            original_to = nn.Module.to

            def mock_to(self, device=None, **kwargs):
                if device == 'npu' or (isinstance(device, str) and 'npu' in device.lower()):
                    return self
                if device is None:
                    return self
                try:
                    return original_to(self, device, **kwargs)
                except (RuntimeError, ValueError):
                    return self

            def mock_tensor_to(self, *args, **kwargs):
                if args:
                    device = args[0]
                    if device == 'npu' or (isinstance(device, str) and 'npu' in device.lower()):
                        return self
                device = kwargs.get('device', None)
                if device == 'npu' or (isinstance(device, str) and 'npu' in device.lower()):
                    return self
                return self

            mtp_decoder = DummyMTPLayer(self.dummy_config)
            dummy_model = DummyModel(config=self.dummy_config)
            dummy_model.model.freqs_cis = torch.randn(100, 32)
            args = (hidden_states, residual)
            kwargs = {'start_pos': 2, 'freqs_cis': torch.randn(8, 32)}

            with patch.object(nn.Module, 'to', mock_to):
                with patch.object(torch.Tensor, 'to', mock_tensor_to):
                    # 测试1: dict输入
                    inputs_dict = {'input_ids': input_ids, 'attention_mask': attention_mask}
                    new_args_dict, new_kwargs_dict = adapter.mtp_preprocess(
                        model=dummy_model, mtp_decoder=mtp_decoder, inputs=inputs_dict, args=args, kwargs=kwargs
                    )

                    # 验证dict输入的结果
                    self.assertEqual(len(new_args_dict), 2)
                    self.assertIn('mask', new_kwargs_dict)
                    self.assertIn('freqs_cis', new_kwargs_dict)

                    # 测试2: list输入
                    inputs_list = [input_ids, attention_mask]
                    new_args_list, new_kwargs_list = adapter.mtp_preprocess(
                        model=dummy_model, mtp_decoder=mtp_decoder, inputs=inputs_list, args=args, kwargs=kwargs
                    )

                    # 验证list输入的结果
                    self.assertEqual(len(new_args_list), 2)
                    self.assertIn('mask', new_kwargs_list)

                    # 验证两次都调用了remove_zero_and_shift和_prepare_4d_causal_attention_mask
                    self.assertEqual(mock_remove_zero.call_count, 2)
                    self.assertEqual(mock_prepare.call_count, 2)

    def create_mock_setup(self, prefix, weight_map, params):
        """创建get_state_dict测试的公共模拟设置"""
        adapter = self.create_adapter()
        adapter.get_weight_map = Mock()
        adapter.get_weight_map.return_value = weight_map

        mock_module = Mock(spec=nn.Module)
        mock_module.named_parameters.return_value = params

        return mock_module, adapter

    def test_get_state_dict_with_prefix_and_multiple_files(self):
        """测试get_state_dict：带前缀、不带前缀、多文件等情况"""
        test_cases = [
            # 带prefix的情况
            {
                "prefix": "prefix",
                "weight_map": {"prefix.layer.weight": "file1.safetensors", "prefix.layer.bias": "file1.safetensors"},
                "params": [("layer.weight", Mock()), ("layer.bias", Mock())],
                "expected_calls": ["prefix.layer.weight", "prefix.layer.bias"],
                "expected_file_count": 1,
            },
            # 不带prefix的情况
            {
                "prefix": "",
                "weight_map": {"layer.weight": "file2.safetensors", "layer.bias": "file2.safetensors"},
                "params": [("layer.weight", Mock()), ("layer.bias", Mock())],
                "expected_calls": ["layer.weight", "layer.bias"],
                "expected_file_count": 1,
            },
            # 多文件的情况
            {
                "prefix": "",
                "weight_map": {
                    "layer1.weight": "file1.safetensors",
                    "layer1.bias": "file1.safetensors",
                    "layer2.weight": "file2.safetensors",
                    "layer2.bias": "file2.safetensors",
                },
                "params": [
                    ("layer1.weight", Mock()),
                    ("layer1.bias", Mock()),
                    ("layer2.weight", Mock()),
                    ("layer2.bias", Mock()),
                ],
                "expected_calls": ["layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"],
                "expected_file_count": 2,
            },
        ]

        for case in test_cases:
            with self.subTest(case=case):
                try:
                    mock_module, adapter = self.create_mock_setup(case["prefix"], case["weight_map"], case["params"])
                except Exception as e:
                    self.fail(f"test_get_state_dict_with_prefix_and_multiple_files failed: {e}")

                with (
                    patch(
                        'msmodelslim.model.deepseek_v3_2.model_adapter.get_valid_read_path',
                        side_effect=lambda x, **kwargs: x,
                    ),
                    patch('msmodelslim.model.deepseek_v3_2.model_adapter.safe_open') as mock_safe_open,
                ):
                    if case["expected_file_count"] > 1:
                        # 多文件情况
                        mock_file1, mock_file2 = MagicMock(), MagicMock()
                        mock_file1.get_tensor.return_value = "tensor1"
                        mock_file2.get_tensor.return_value = "tensor2"

                        def create_file_side_effect(file1, file2):
                            def file_side_effect(path, **kwargs):
                                mock_context = MagicMock()
                                mock_context.__enter__.return_value = file1 if "file1" in path else file2
                                return mock_context

                            return file_side_effect

                        mock_safe_open.side_effect = create_file_side_effect(mock_file1, mock_file2)

                    else:
                        # 单文件情况
                        mock_file = MagicMock()
                        mock_file.get_tensor.return_value = "dummy_tensor"
                        mock_safe_open.return_value.__enter__.return_value = mock_file

                    result = adapter.get_state_dict(mock_module, case["prefix"])

                    self.assertEqual(len(result), len(case["params"]))
                    for name, _ in case["params"]:
                        self.assertIn(name, result)
                    self.assertEqual(mock_safe_open.call_count, case["expected_file_count"])

    def test_get_state_dict_file_not_found(self):
        """测试文件不存在时的异常处理"""
        weight_map = {"layer.weight": "missing.safetensors"}
        params = [("layer.weight", Mock())]

        mock_module, adapter = self.create_mock_setup("", weight_map, params)

        with patch(
            'msmodelslim.model.deepseek_v3_2.model_adapter.get_valid_read_path',
            side_effect=FileNotFoundError("File not found"),
        ):
            with self.assertRaises(FileNotFoundError):
                adapter.get_state_dict(mock_module)

    def test_get_ln_fuse_map(self):
        """测试get_ln_fuse_map生成LayerNorm融合映射"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 2

        # Mock get_ln_fuse_map from quarot module
        with patch('msmodelslim.model.deepseek_v3_2.model_adapter.get_ln_fuse_map') as mock_get_ln_fuse:
            # 返回包含所有需要的key的字典
            mock_get_ln_fuse.return_value = {
                "model.layers.0.input_layernorm": ["model.layers.0.self_attn.q_a_proj"],
                "model.layers.0.self_attn.q_a_layernorm": [],  # 添加这个key
                "model.layers.1.input_layernorm": ["model.layers.1.self_attn.q_a_proj"],
                "model.layers.1.self_attn.q_a_layernorm": [],  # 添加这个key
            }

            empty_dict, ln_linear_map = adapter.get_ln_fuse_map()

            # 验证返回的第一个字典为空
            self.assertEqual(empty_dict, {})

            # 验证为每层添加了indexer相关的映射
            self.assertIn("model.layers.0.input_layernorm", ln_linear_map)
            self.assertIn("model.layers.0.self_attn.indexer.wk", ln_linear_map["model.layers.0.input_layernorm"])
            self.assertIn("model.layers.0.self_attn.q_a_layernorm", ln_linear_map)
            self.assertIn(
                "model.layers.0.self_attn.indexer.wq_b", ln_linear_map["model.layers.0.self_attn.q_a_layernorm"]
            )

    def test_get_bake_names(self):
        """测试get_bake_names返回空列表"""
        adapter = self.create_adapter()

        result1, result2 = adapter.get_bake_names()

        self.assertEqual(result1, [])
        self.assertEqual(result2, [])

    def test_get_rotate_map(self):
        """测试get_rotate_map生成旋转映射"""
        adapter = self.create_adapter()
        adapter.config.num_hidden_layers = 2
        block_size = 128

        # Mock get_rotate_map from quarot module
        with patch('msmodelslim.model.deepseek_v3_2.model_adapter.get_rotate_map') as mock_get_rotate:
            mock_pre_run = Mock()
            mock_rot_pair = Mock()
            mock_rot_pair.right_rot = {}
            mock_rot_b_proj_pair = Mock()
            mock_rot_b_proj_pair.right_rot = {}

            mock_rotate_matrix = {'rot': torch.randn(128, 128), 'rot_b_proj': torch.randn(128, 128)}

            mock_get_rotate.return_value = (
                mock_pre_run,
                {'rot': mock_rot_pair, 'rot_b_proj': mock_rot_b_proj_pair},
                mock_rotate_matrix,
            )

            pre_run_list, rot_pairs_list = adapter.get_rotate_map(block_size)

            # 验证调用了quarot的get_rotate_map
            mock_get_rotate.assert_called_once_with(adapter.config, block_size, num_hidden_layers=2)

            # 验证返回值
            self.assertEqual(pre_run_list, [mock_pre_run])
            self.assertEqual(len(rot_pairs_list), 2)

            # 验证为每层的indexer添加了旋转映射
            self.assertIn("model.layers.0.self_attn.indexer.wk", mock_rot_pair.right_rot)
            self.assertIn("model.layers.1.self_attn.indexer.wk", mock_rot_pair.right_rot)
            self.assertIn("model.layers.0.self_attn.indexer.wq_b", mock_rot_b_proj_pair.right_rot)
            self.assertIn("model.layers.1.self_attn.indexer.wq_b", mock_rot_b_proj_pair.right_rot)
