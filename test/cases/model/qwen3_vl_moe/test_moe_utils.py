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
# pylint: disable=attribute-defined-outside-init,consider-using-from-import,pointless-string-statement

"""
Unit tests for `msmodelslim.model.qwen3_vl_moe.moe_utils`.

这些用例主要覆盖：
- UnstackedQwen3VLMoeTextMLP 的初始化和 forward
- UnstackedQwen3VLMoeSparseMoeBlock 的初始化、forward、权重转换
- convert_qwen3_moe_to_linear 的转换逻辑和各种边界情况
"""

from unittest.mock import Mock, patch  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from msmodelslim.model.qwen3_vl_moe.moe_utils import (  # noqa: E402
    UnstackedQwen3VLMoeTextMLP,
    UnstackedQwen3VLMoeSparseMoeBlock,
    convert_qwen3_moe_to_linear,
)


class MockConfig:
    """模拟 Qwen3VLMoeTextConfig"""

    def __init__(
        self,
        hidden_size=2048,
        moe_intermediate_size=1408,
        hidden_act="silu",
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=4,
        decoder_sparse_step=2,
        mlp_only_layers=None,
    ):
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.hidden_act = hidden_act
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = mlp_only_layers or []


class MockOriginalMoeBlock(nn.Module):
    """模拟 Qwen3VLMoeTextSparseMoeBlock"""

    def __init__(self, num_experts=8, hidden_size=2048, expert_dim=1408, dtype=torch.float32):
        super().__init__()
        # gate 会自动注册参数
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)
        # 模拟 3D 权重 - 使用 nn.Module 以便 PrepareWeight 能正确处理
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.randn(num_experts, hidden_size, 2 * expert_dim, dtype=dtype))
        self.experts.down_proj = nn.Parameter(torch.randn(num_experts, expert_dim, hidden_size, dtype=dtype))
        # 将 experts 注册为子模块，确保 parameters() 能正确工作
        self.add_module('experts', self.experts)

        # 确保至少有一个参数可以被访问（用于获取 dtype）
        # gate.weight 应该已经存在，但为了确保，我们验证一下
        assert self.gate.weight is not None, "gate.weight should not be None"


@pytest.fixture
def mock_act2fn():
    """Mock ACT2FN"""

    def silu(x):
        return x * torch.sigmoid(x)

    with patch('msmodelslim.model.qwen3_vl_moe.moe_utils.ACT2FN', {'silu': silu}):
        yield


class TestUnstackedQwen3VLMoeTextMLP:
    """测试 UnstackedQwen3VLMoeTextMLP"""

    @pytest.mark.usefixtures("mock_act2fn")
    def test_init_creates_linear_layers(self):
        """验证初始化时创建了正确的 Linear 层"""
        config = MockConfig(hidden_size=512, moe_intermediate_size=256)
        mlp = UnstackedQwen3VLMoeTextMLP(config, dtype=torch.float32)

        assert mlp.hidden_size == 512
        assert mlp.intermediate_size == 256
        assert mlp.expert_dim == 256
        assert isinstance(mlp.gate_proj, nn.Linear)
        assert isinstance(mlp.up_proj, nn.Linear)
        assert isinstance(mlp.down_proj, nn.Linear)
        assert mlp.gate_proj.weight.shape == (256, 512)
        assert mlp.up_proj.weight.shape == (256, 512)
        assert mlp.down_proj.weight.shape == (512, 256)

    @pytest.mark.usefixtures("mock_act2fn")
    def test_forward_computes_correctly(self):
        """验证 forward 方法正确计算 MLP 输出"""
        config = MockConfig(hidden_size=4, moe_intermediate_size=8)
        mlp = UnstackedQwen3VLMoeTextMLP(config, dtype=torch.float32)

        # 设置已知权重以便验证
        torch.nn.init.ones_(mlp.gate_proj.weight)
        torch.nn.init.ones_(mlp.up_proj.weight)
        torch.nn.init.ones_(mlp.down_proj.weight)

        x = torch.ones(2, 4)  # batch_size=2, hidden_size=4
        output = mlp(x)

        assert output.shape == (2, 4)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.usefixtures("mock_act2fn")
    def test_init_with_dtype(self):
        """验证使用不同 dtype 初始化"""
        config = MockConfig()
        mlp_float16 = UnstackedQwen3VLMoeTextMLP(config, dtype=torch.float16)
        mlp_float32 = UnstackedQwen3VLMoeTextMLP(config, dtype=torch.float32)

        assert mlp_float16.gate_proj.weight.dtype == torch.float16
        assert mlp_float32.gate_proj.weight.dtype == torch.float32


class TestUnstackedQwen3VLMoeSparseMoeBlock:
    """测试 UnstackedQwen3VLMoeSparseMoeBlock"""

    @pytest.mark.usefixtures("mock_act2fn")
    def test_init_creates_gate_and_experts(self):
        """验证初始化时创建了 gate 和 experts"""
        config = MockConfig(num_experts=4, hidden_size=512)
        original_moe = MockOriginalMoeBlock(num_experts=4, hidden_size=512, dtype=torch.float32)

        # 确保 original_moe 有参数
        params = list(original_moe.parameters())
        assert len(params) > 0, "MockOriginalMoeBlock should have parameters"

        moe_block = UnstackedQwen3VLMoeSparseMoeBlock(config, original_moe, copy_weights=False)

        assert moe_block.hidden_size == 512
        assert moe_block.num_experts == 4
        assert moe_block.top_k == 2
        assert isinstance(moe_block.gate, nn.Linear)
        assert len(moe_block.experts) == 4
        assert all(isinstance(expert, UnstackedQwen3VLMoeTextMLP) for expert in moe_block.experts)

    @pytest.mark.usefixtures("mock_act2fn")
    def test_init_with_copy_weights_calls_transform(self):
        """验证 copy_weights=True 时调用权重转换"""
        config = MockConfig(num_experts=2, hidden_size=4)
        original_moe = MockOriginalMoeBlock(num_experts=2, hidden_size=4, expert_dim=8, dtype=torch.float32)

        # 确保 original_moe 有参数
        assert len(list(original_moe.parameters())) > 0

        with patch.object(UnstackedQwen3VLMoeSparseMoeBlock, '_transform_weights_from_original') as mock_transform:
            moe_block = UnstackedQwen3VLMoeSparseMoeBlock(config, original_moe, copy_weights=True)  # noqa: F841
            mock_transform.assert_called_once_with(original_moe, in_place=False)

    @pytest.mark.usefixtures("mock_act2fn")
    def test_forward_inference_mode(self):
        """验证推理模式下的 forward 计算"""
        config = MockConfig(num_experts=2, hidden_size=4, num_experts_per_tok=1)
        original_moe = MockOriginalMoeBlock(num_experts=2, hidden_size=4, expert_dim=8, dtype=torch.float32)

        # 确保 original_moe 有参数
        assert len(list(original_moe.parameters())) > 0

        moe_block = UnstackedQwen3VLMoeSparseMoeBlock(config, original_moe, copy_weights=False)
        moe_block.eval()  # 设置为评估模式

        # 设置 gate 权重以便路由可预测
        torch.nn.init.ones_(moe_block.gate.weight)

        hidden_states = torch.randn(2, 3, 4)  # batch=2, seq=3, hidden=4
        output = moe_block(hidden_states)

        assert output.shape == (2, 3, 4)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.usefixtures("mock_act2fn")
    def test_forward_training_mode_raises_error(self):
        """验证训练模式下 forward 抛出 NotImplementedError"""
        config = MockConfig(num_experts=2, hidden_size=4)
        original_moe = MockOriginalMoeBlock(num_experts=2, hidden_size=4, dtype=torch.float32)

        # 确保 original_moe 有参数
        assert len(list(original_moe.parameters())) > 0

        moe_block = UnstackedQwen3VLMoeSparseMoeBlock(config, original_moe, copy_weights=False)
        moe_block.train()  # 设置为训练模式

        hidden_states = torch.randn(1, 2, 4)

        with pytest.raises(NotImplementedError) as exc_info:
            moe_block(hidden_states)

        assert "Training mode" in str(exc_info.value)

    @pytest.mark.usefixtures("mock_act2fn")
    def test_forward_handles_device_mismatch(self):
        """验证 forward 处理设备不匹配的情况"""
        config = MockConfig(num_experts=2, hidden_size=4, num_experts_per_tok=1)
        original_moe = MockOriginalMoeBlock(num_experts=2, hidden_size=4, dtype=torch.float32)

        # 确保 original_moe 有参数
        assert len(list(original_moe.parameters())) > 0

        moe_block = UnstackedQwen3VLMoeSparseMoeBlock(config, original_moe, copy_weights=False)
        moe_block.eval()

        # 将模型移到 CPU，输入在 CPU
        moe_block = moe_block.cpu()
        hidden_states = torch.randn(1, 2, 4)

        output = moe_block(hidden_states)
        assert output.shape == (1, 2, 4)

    @pytest.mark.usefixtures("mock_act2fn")
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.PrepareWeight')
    def test_transform_weights_from_original(self, mock_prepare_weight):
        """验证权重转换逻辑"""
        config = MockConfig(num_experts=2, hidden_size=4, num_experts_per_tok=1)
        original_moe = MockOriginalMoeBlock(num_experts=2, hidden_size=4, expert_dim=8, dtype=torch.float32)

        # 确保 original_moe 有参数
        assert len(list(original_moe.parameters())) > 0

        # 设置已知权重
        torch.nn.init.ones_(original_moe.gate.weight)
        torch.nn.init.ones_(original_moe.experts.gate_up_proj)
        torch.nn.init.ones_(original_moe.experts.down_proj)

        moe_block = UnstackedQwen3VLMoeSparseMoeBlock(config, original_moe, copy_weights=False)

        # Mock PrepareWeight context manager
        mock_prepare_weight.return_value.__enter__ = Mock(return_value=None)
        mock_prepare_weight.return_value.__exit__ = Mock(return_value=None)

        moe_block._transform_weights_from_original(original_moe, in_place=False)

        # 验证权重被转换
        assert moe_block.gate.weight is not None
        assert moe_block.experts[0].gate_proj.weight is not None
        assert moe_block.experts[0].up_proj.weight is not None
        assert moe_block.experts[0].down_proj.weight is not None

    @pytest.mark.usefixtures("mock_act2fn")
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.PrepareWeight')
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.gc')
    def test_transform_weights_in_place_deletes_original(self, mock_gc, mock_prepare_weight):
        """验证 in_place=True 时删除原始权重"""
        config = MockConfig(num_experts=2, hidden_size=4)
        original_moe = MockOriginalMoeBlock(num_experts=2, hidden_size=4, expert_dim=8, dtype=torch.float32)

        # 确保 original_moe 有参数
        assert len(list(original_moe.parameters())) > 0

        moe_block = UnstackedQwen3VLMoeSparseMoeBlock(config, original_moe, copy_weights=False)

        mock_prepare_weight.return_value.__enter__ = Mock(return_value=None)
        mock_prepare_weight.return_value.__exit__ = Mock(return_value=None)

        moe_block._transform_weights_from_original(original_moe, in_place=True)

        # 验证原始权重被删除
        assert not hasattr(original_moe.experts, 'gate_up_proj')
        assert not hasattr(original_moe.experts, 'down_proj')
        mock_gc.collect.assert_called()


class TestConvertQwen3MoeToLinear:
    """测试 convert_qwen3_moe_to_linear"""

    def create_mock_model(self, num_layers=4, decoder_sparse_step=2, num_experts=2):
        """创建模拟模型"""
        model = Mock()
        language_model = nn.Module()  # 使用真实的 nn.Module 而不是 Mock
        layers = []

        for i in range(num_layers):
            layer = nn.Module()  # 使用真实的 nn.Module
            if (i + 1) % decoder_sparse_step == 0:
                moe_block = MockOriginalMoeBlock(num_experts=num_experts, hidden_size=4, dtype=torch.float32)
                # 确保 moe_block 有参数
                assert len(list(moe_block.parameters())) > 0
                layer.mlp = moe_block
            else:
                layer.mlp = nn.Linear(4, 4)  # 普通 MLP
            layers.append(layer)

        language_model.layers = nn.ModuleList(layers)
        model.model = Mock()
        model.model.language_model = language_model

        return model

    @pytest.mark.usefixtures("mock_act2fn")
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.Qwen3VLMoeTextSparseMoeBlock', MockOriginalMoeBlock)
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.get_logger')
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.gc')
    def test_convert_identifies_target_layers(self, mock_gc, mock_logger):
        """验证转换函数正确识别目标层"""
        config = MockConfig(
            num_hidden_layers=4,
            decoder_sparse_step=2,
            mlp_only_layers=[],
            num_experts=2,  # 确保与 MockOriginalMoeBlock 的 num_experts 一致
        )
        model = self.create_mock_model(num_layers=4, decoder_sparse_step=2, num_experts=2)

        # 确保模型有4层
        assert len(model.model.language_model.layers) == 4

        convert_qwen3_moe_to_linear(model, config, in_place=True, verbose=True)

        # 确保转换后仍有4层
        assert len(model.model.language_model.layers) == 4

        # 验证只有第 1 和第 3 层（索引从 0 开始，第 2 和第 4 层）被转换
        # decoder_sparse_step=2 意味着第 2、4 层（索引 1、3）会被转换
        assert isinstance(model.model.language_model.layers[1].mlp, UnstackedQwen3VLMoeSparseMoeBlock)
        assert isinstance(model.model.language_model.layers[3].mlp, UnstackedQwen3VLMoeSparseMoeBlock)
        # 第 0 和第 2 层应该是普通 MLP（索引 0、2）
        assert isinstance(model.model.language_model.layers[0].mlp, nn.Linear)
        assert isinstance(model.model.language_model.layers[2].mlp, nn.Linear)

    @pytest.mark.usefixtures("mock_act2fn")
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.Qwen3VLMoeTextSparseMoeBlock', MockOriginalMoeBlock)
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.get_logger')
    def test_convert_skips_mlp_only_layers(self, mock_logger):
        """验证转换函数跳过 mlp_only_layers"""
        config = MockConfig(
            num_hidden_layers=4,
            decoder_sparse_step=1,  # 每层都应该是 MoE
            mlp_only_layers=[1, 3],  # 但第 1 和第 3 层被跳过
            num_experts=2,  # 确保与 MockOriginalMoeBlock 的 num_experts 一致
        )
        model = self.create_mock_model(num_layers=4, decoder_sparse_step=1, num_experts=2)

        convert_qwen3_moe_to_linear(model, config, in_place=True, verbose=False)

        # 只有第 0 和第 2 层被转换
        assert isinstance(model.model.language_model.layers[0].mlp, UnstackedQwen3VLMoeSparseMoeBlock)
        assert isinstance(model.model.language_model.layers[2].mlp, UnstackedQwen3VLMoeSparseMoeBlock)
        # 第 1 和第 3 层保持原样（虽然创建时是 MoE，但应该被跳过）

    @pytest.mark.usefixtures("mock_act2fn")
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.Qwen3VLMoeTextSparseMoeBlock', MockOriginalMoeBlock)
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.get_logger')
    def test_convert_handles_config_with_text_config(self, mock_logger):
        """验证处理包含 text_config 的配置"""
        outer_config = Mock()
        outer_config.text_config = MockConfig(
            num_hidden_layers=2,
            decoder_sparse_step=1,
            num_experts=2,  # 确保与 MockOriginalMoeBlock 的 num_experts 一致
        )
        model = self.create_mock_model(num_layers=2, decoder_sparse_step=1, num_experts=2)

        convert_qwen3_moe_to_linear(model, outer_config, in_place=True, verbose=False)

        assert isinstance(model.model.language_model.layers[0].mlp, UnstackedQwen3VLMoeSparseMoeBlock)
        assert isinstance(model.model.language_model.layers[1].mlp, UnstackedQwen3VLMoeSparseMoeBlock)

    @pytest.mark.usefixtures("mock_act2fn")
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.Qwen3VLMoeTextSparseMoeBlock', MockOriginalMoeBlock)
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.get_logger')
    def test_convert_skips_non_moe_layers(self, mock_logger):
        """验证转换函数跳过非 MoE 层"""
        config = MockConfig(num_hidden_layers=2, decoder_sparse_step=1, num_experts=2)
        model = self.create_mock_model(num_layers=2, decoder_sparse_step=1, num_experts=2)

        # 将第 0 层改为普通 MLP
        model.model.language_model.layers[0].mlp = nn.Linear(4, 4)

        convert_qwen3_moe_to_linear(model, config, in_place=True, verbose=False)

        # 第 0 层保持为普通 MLP
        assert isinstance(model.model.language_model.layers[0].mlp, nn.Linear)
        # 第 1 层被转换
        assert isinstance(model.model.language_model.layers[1].mlp, UnstackedQwen3VLMoeSparseMoeBlock)

    @pytest.mark.usefixtures("mock_act2fn")
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.Qwen3VLMoeTextSparseMoeBlock', MockOriginalMoeBlock)
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.get_logger')
    def test_convert_handles_model_without_model_attribute(self, mock_logger):
        """验证处理没有 model.model 属性的模型"""
        config = MockConfig(num_hidden_layers=2, decoder_sparse_step=1, num_experts=2)

        # 创建一个简单的对象而不是 Mock，以便可以设置属性
        class SimpleModel:
            pass

        model = SimpleModel()
        language_model = nn.Module()  # 使用真实的 nn.Module
        layers = [
            nn.Module()  # 使用真实的 nn.Module 而不是 Mock
            for _ in range(2)
        ]
        # 为每个 layer 设置 mlp，确保每个 mlp 都有参数
        for layer in layers:
            moe_block = MockOriginalMoeBlock(num_experts=2, hidden_size=4, dtype=torch.float32)
            assert len(list(moe_block.parameters())) > 0
            layer.mlp = moe_block
        language_model.layers = nn.ModuleList(layers)
        model.language_model = language_model

        convert_qwen3_moe_to_linear(model, config, in_place=True, verbose=False)

        assert isinstance(model.language_model.layers[0].mlp, UnstackedQwen3VLMoeSparseMoeBlock)

    @pytest.mark.usefixtures("mock_act2fn")
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.Qwen3VLMoeTextSparseMoeBlock', MockOriginalMoeBlock)
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.get_logger')
    def test_convert_logs_progress_when_verbose(self, mock_logger):
        """验证 verbose=True 时记录日志"""
        config = MockConfig(num_hidden_layers=2, decoder_sparse_step=1, num_experts=2)
        model = self.create_mock_model(num_layers=2, decoder_sparse_step=1, num_experts=2)

        convert_qwen3_moe_to_linear(model, config, in_place=True, verbose=True)

        # 验证日志被调用
        assert mock_logger.return_value.info.called

    @pytest.mark.usefixtures("mock_act2fn")
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.Qwen3VLMoeTextSparseMoeBlock', MockOriginalMoeBlock)
    @patch('msmodelslim.model.qwen3_vl_moe.moe_utils.get_logger')
    def test_convert_with_in_place_false(self, mock_logger):
        """验证 in_place=False 时的转换"""
        config = MockConfig(num_hidden_layers=2, decoder_sparse_step=1, num_experts=2)
        model = self.create_mock_model(num_layers=2, decoder_sparse_step=1, num_experts=2)

        convert_qwen3_moe_to_linear(model, config, in_place=False, verbose=False)

        # 验证转换成功
        assert isinstance(model.model.language_model.layers[0].mlp, UnstackedQwen3VLMoeSparseMoeBlock)
        assert isinstance(model.model.language_model.layers[1].mlp, UnstackedQwen3VLMoeSparseMoeBlock)
