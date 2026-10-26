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

# pylint: disable=abstract-method

import re

import torch
from torch import nn

from msmodelslim.processor.flat_quant.flat_quant_utils.utils import (
    clone_module_hooks,
    convert_outputs_to_inputs,
    get_decompose_dim,
    get_init_scale,
    get_init_weight,
    get_inverse,
    get_module_by_name,
    get_n_set_parameters_byname,
    get_para_names,
    get_random_orthg,
    get_trainable_parameters,
    match_pattern,
    move_tensors_to_device,
    remove_after_substring,
    set_module_by_name,
    set_require_grad_all,
    stat_input_hook,
    stat_tensor,
)


class TestGetInitScale:
    """Test suite for get_init_scale — 缩放因子计算。"""

    def test_get_init_scale_returns_clamped_value_when_default_alpha(self):
        """主路径：默认 alpha=0.5，输出应 > 0（受 clamp 保护）。"""
        w_smax = torch.tensor([1.0, 2.0])
        x_smax = torch.tensor([2.0, 4.0])

        scale = get_init_scale(w_smax, x_smax)

        assert torch.all(scale >= 1e-5)
        assert scale.shape == w_smax.shape

    def test_get_init_scale_returns_smaller_value_when_alpha_increases(self):
        """边界：alpha→1 时 scale 趋近 w_smax / x_smax（应较小）。"""
        w_smax = torch.tensor([2.0])
        x_smax = torch.tensor([2.0])

        scale_low = get_init_scale(w_smax, x_smax, alpha=0.0)
        scale_high = get_init_scale(w_smax, x_smax, alpha=0.9)

        # alpha 越大，x_smax 权重越大，scale 越小
        assert scale_low[0] >= scale_high[0]


class TestGetDecomposeDim:
    """Test suite for get_decompose_dim — 整数分解为平方差。"""

    def test_get_decompose_dim_returns_pair_of_positive_ints_when_n_is_perfect_difference(self):
        """主路径：n=5=3²-2²=9-4 应返回 (a-b=1, a+b=5)，即 a=3, b=2。"""
        a_minus_b, a_plus_b = get_decompose_dim(5)

        a = (a_minus_b + a_plus_b) // 2
        b = (a_plus_b - a_minus_b) // 2
        assert a == 3
        assert b == 2

    def test_get_decompose_dim_returns_pair_for_zero(self):
        """边界：n=0 应返回 (a-b, a+b) 使得 a²- b² = 0 → 最小 a=0, b=0。"""
        a_minus_b, a_plus_b = get_decompose_dim(0)

        assert a_minus_b == 0
        assert a_plus_b == 0

    def test_get_decompose_dim_returns_pair_for_large_n(self):
        """边界：较大 n 仍应能找到 (a-b, a+b)。"""
        a_minus_b, a_plus_b = get_decompose_dim(100)

        a = (a_minus_b + a_plus_b) // 2
        b = (a_plus_b - a_minus_b) // 2
        assert a * a - b * b == 100


class TestGetRandomOrthg:
    """Test suite for get_random_orthg — 随机正交矩阵生成。"""

    def test_get_random_orthg_returns_tensor_with_correct_shape_when_size_given(self):
        """主路径：返回 shape=(size, size) 的 tensor。"""
        mat = get_random_orthg(8)

        assert mat.shape == (8, 8)

    def test_get_random_orthg_returns_orthogonal_matrix_when_size_given(self):
        """主路径：Q @ Q.T ≈ I（允许 float32 误差）。"""
        mat = get_random_orthg(16)
        product = (mat.double() @ mat.T.double()).float()

        assert torch.allclose(product, torch.eye(16), atol=1e-4)


class TestGetInitWeight:
    """Test suite for get_init_weight — 包装 get_random_orthg。"""

    def test_get_init_weight_returns_orthogonal_matrix_of_given_dim(self):
        """主路径：返回 dim×dim 的正交矩阵。"""
        w = get_init_weight(10)
        product = (w.double() @ w.T.double()).float()

        assert w.shape == (10, 10)
        assert torch.allclose(product, torch.eye(10), atol=1e-4)


class TestGetInverse:
    """Test suite for get_inverse — 矩阵求逆。"""

    def test_get_inverse_returns_correct_inverse_when_matrix_is_invertible(self):
        """主路径：A @ A_inv ≈ I（统一为 float32 比较）。"""
        a = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        a_inv = get_inverse(a)

        product = (a @ a_inv).float()
        assert torch.allclose(product, torch.eye(2), atol=1e-4)


class TestMatchPattern:
    """Test suite for match_pattern — 名称匹配。"""

    def test_match_pattern_returns_true_when_str_prefix_matches(self):
        """主路径：字符串前缀匹配。"""
        assert match_pattern("model.layer.weight", "model.layer") is True

    def test_match_pattern_returns_false_when_str_prefix_does_not_match(self):
        """主路径：字符串前缀不匹配。"""
        assert match_pattern("model.other", "model.layer") is False

    def test_match_pattern_returns_true_when_regex_matches(self):
        """主路径：正则表达式匹配。"""
        assert match_pattern("model.layer.0.weight", re.compile(r".*layer\.\d+.*")) is not None


class TestMoveTensorsToDevice:
    """Test suite for move_tensors_to_device — 递归搬设备。"""

    def test_move_tensors_to_device_returns_same_tensor_when_already_on_target(self):
        """主路径：tensor 已在目标 device 应原样返回。"""
        t = torch.zeros(3)
        result = move_tensors_to_device(t, "cpu")

        assert torch.equal(result, t)

    def test_move_tensors_to_device_recursively_handles_dict_when_nested_structure(self):
        """边界：嵌套 dict 中的所有 tensor 都被搬设备。"""
        t1 = torch.zeros(2)
        t2 = torch.ones(2)
        data = {"a": t1, "b": {"c": t2}}

        result = move_tensors_to_device(data, "cpu")

        assert isinstance(result, dict)
        assert torch.equal(result["a"], t1)
        assert torch.equal(result["b"]["c"], t2)

    def test_move_tensors_to_device_recursively_handles_list_when_nested_structure(self):
        """边界：嵌套 list 中的 tensor 都被搬设备。"""
        data = [torch.zeros(2), [torch.ones(3)]]

        result = move_tensors_to_device(data, "cpu")

        assert isinstance(result, list)
        assert torch.equal(result[0], torch.zeros(2))

    def test_move_tensors_to_device_recursively_handles_tuple_when_nested_structure(self):
        """边界：嵌套 tuple 也能被搬。"""
        data = (torch.zeros(2), (torch.ones(3), "str"))

        result = move_tensors_to_device(data, "cpu")

        assert isinstance(result, tuple)
        assert torch.equal(result[0], torch.zeros(2))

    def test_move_tensors_to_device_returns_non_tensor_as_is(self):
        """边界：非 tensor 直接返回。"""
        assert move_tensors_to_device(42, "cpu") == 42
        assert move_tensors_to_device("hello", "cpu") == "hello"


class TestSetRequireGradAll:
    """Test suite for set_require_grad_all — 设置 requires_grad。"""

    def test_set_require_grad_all_sets_all_params_to_true_when_called_with_true(self):
        """主路径：所有参数 requires_grad=True。"""
        model = nn.Linear(3, 3)
        set_require_grad_all(model, True)

        for _, p in model.named_parameters():
            assert p.requires_grad is True

    def test_set_require_grad_all_sets_all_params_to_false_when_called_with_false(self):
        """主路径：所有参数 requires_grad=False。"""
        model = nn.Linear(3, 3)
        for _, p in model.named_parameters():
            p.requires_grad = True  # 先设 True
        set_require_grad_all(model, False)

        for _, p in model.named_parameters():
            assert p.requires_grad is False


class TestGetTrainableParameters:
    """Test suite for get_trainable_parameters — 收集优化器参数。"""

    def test_get_trainable_parameters_returns_empty_groups_when_no_named_params(self):
        """主路径：无匹配名称的模型应返回 6 个空组，need_train=False。"""
        model = nn.Linear(3, 3)

        params, trainable, need_train = get_trainable_parameters(model)

        assert all(len(v) == 0 for v in params.values())
        assert need_train is False
        # trainable 仍是 6 个 dict（每种参数类型一个），只是 params 都是空
        assert len(trainable) == 6

    def test_get_trainable_parameters_returns_matched_params_when_names_match(self):
        """主路径：参数名含 'linear_u' 时应被收集。"""
        # 构造一个带 linear_u 后缀的参数
        p = nn.Parameter(torch.zeros(3))
        p.requires_grad = False
        model = nn.Linear(3, 3)
        # 用 register_parameter 加一个 "linear_u" 参数
        model.register_parameter("linear_u", p)
        # 把 linear 自己的参数 requires_grad 设回原状
        for _, existing in model.named_parameters():
            if existing is not p:
                existing.requires_grad = False

        params, _, need_train = get_trainable_parameters(model)

        assert len(params["linear_u"]) == 1
        assert need_train is True
        assert params["linear_u"][0].requires_grad is True


class TestGetParaNames:
    """Test suite for get_para_names — 固定参数名列表。"""

    def test_get_para_names_returns_six_expected_names(self):
        """主路径：返回 6 个预期名称。"""
        names = get_para_names()

        assert "linear_u" in names
        assert "linear_v" in names
        assert "trans_linear" in names
        assert "linear_diag" in names
        assert "diag_scale" in names
        assert "clip_factor" in names
        assert len(names) == 6


class TestRemoveAfterSubstring:
    """Test suite for remove_after_substring — 字符串截断。"""

    def test_remove_after_substring_truncates_when_substring_found(self):
        """主路径：找到子串则截断到子串末尾。"""
        result = remove_after_substring("model.layer.weight.bias", "weight")

        assert result == "model.layer.weight"

    def test_remove_after_substring_returns_unchanged_when_substring_not_found(self):
        """主路径：未找到子串时原样返回。"""
        text = "model.other"

        result = remove_after_substring(text, "weight")

        assert result == text


class TestConvertOutputsToInputs:
    """Test suite for convert_outputs_to_inputs — 输出转输入格式。"""

    def test_convert_outputs_to_inputs_wraps_each_in_list(self):
        """主路径：每个 output 包裹成 [output] 子列表。"""
        result = convert_outputs_to_inputs([1, 2, 3])

        assert result == [[1], [2], [3]]

    def test_convert_outputs_to_inputs_returns_empty_list_for_empty_input(self):
        """边界：空输入返回空列表。"""
        assert not convert_outputs_to_inputs([])


class TestGetModuleByName:
    """Test suite for get_module_by_name / set_module_by_name — 按路径访问子模块。"""

    def test_get_module_by_name_returns_nested_module_when_path_is_valid(self):
        """主路径：按点分路径返回嵌套子模块。"""

        class _Sub(nn.Module):
            pass

        class _Top(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = _Sub()

        model = _Top()

        result = get_module_by_name(model, "layer1")

        assert isinstance(result, _Sub)

    def test_get_module_by_name_strips_prefix_when_prefix_provided(self):
        """边界：prefix 应在查找前从 key 中去除。"""

        class _Sub(nn.Module):
            pass

        class _Top(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = _Sub()

        model = _Top()

        result = get_module_by_name(model, "model.layer1", prefix="model")

        assert isinstance(result, _Sub)

    def test_set_module_by_name_replaces_nested_module_when_path_valid(self):
        """主路径：按路径替换子模块。"""

        class _SubA(nn.Module):
            pass

        class _SubB(nn.Module):
            pass

        class _Top(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = _SubA()

        model = _Top()

        set_module_by_name(model, "layer1", _SubB())

        assert isinstance(model.layer1, _SubB)


class TestGetNSetParametersByname:
    """Test suite for get_n_set_parameters_byname — 按名收集参数并启用 grad。"""

    def test_get_n_set_parameters_byname_returns_matching_params_when_name_substring(self):
        """主路径：name 含 required_names 子串的参数应被收集。"""
        p = nn.Parameter(torch.zeros(3))
        p.requires_grad = False
        model = nn.Linear(3, 3)
        model.register_parameter("linear_u", p)
        for _, existing in model.named_parameters():
            if existing is not p:
                existing.requires_grad = False

        result = get_n_set_parameters_byname(model, ["linear_u"])

        assert len(result) == 1
        assert result[0].requires_grad is True

    def test_get_n_set_parameters_byname_returns_empty_list_when_no_match(self):
        """主路径：无匹配时返回空列表。"""
        model = nn.Linear(3, 3)
        for _, p in model.named_parameters():
            p.requires_grad = False

        result = get_n_set_parameters_byname(model, ["nonexistent_name"])

        assert not result


class TestStatTensor:
    """Test suite for stat_input_hook / stat_tensor — 激活统计。"""

    def test_stat_tensor_initializes_input_max_when_key_missing(self):
        """主路径：首次调用应写入 input_max。"""
        act_stats = {"layer": {}}
        x = torch.tensor([[0.5, -0.3], [0.2, 0.8]])

        stat_tensor(act_stats, "layer", x)

        assert "input_max" in act_stats["layer"]
        assert torch.allclose(act_stats["layer"]["input_max"], torch.tensor([0.5, 0.8]))

    # 注：stat_tensor 的 update 分支强依赖 torch_npu (.npu())，CPU 上无法运行，故跳过

    def test_stat_input_hook_unwraps_tuple_input_when_called(self):
        """主路径：传入 tuple 时应取第一个元素作为 x。"""
        act_stats = {"layer": {}}
        x = torch.tensor([[0.5, 0.3]])
        hook_input = (x, None)  # forward hook 第二个是 None

        stat_input_hook(m=None, x=hook_input, y=None, name="layer", act_stats=act_stats)

        assert "input_max" in act_stats["layer"]


class TestCloneModuleHooks:
    """Test suite for clone_module_hooks — 复制模块 hook。"""

    def test_clone_module_hooks_copies_forward_hooks_when_source_has_hooks(self):
        """主路径：源模块的 forward_hooks 应被克隆到目标。"""

        class _M(nn.Module):
            def forward(self, x):
                return x

        src = _M()
        tgt = _M()
        called = []

        def hook(module, input_data, output):
            called.append(input_data)

        src.register_forward_hook(hook)

        clone_module_hooks(src, tgt)

        # 触发 target forward，应触发 hook
        tgt(torch.zeros(2, 2))
        assert len(called) == 1
