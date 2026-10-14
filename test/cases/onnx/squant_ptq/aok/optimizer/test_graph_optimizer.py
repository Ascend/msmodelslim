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
# pylint: disable=attribute-defined-outside-init,unspecified-encoding

import os
import pytest  # noqa: F401
import numpy as np
from unittest.mock import patch, MagicMock  # noqa: F401
import tempfile
from tempfile import TemporaryDirectory

from msmodelslim.onnx.squant_ptq.aok.optimizer import architectures  # noqa: F401
from msmodelslim.onnx.squant_ptq.aok.optimizer.graph_optimizer import _InferenceRunInfo
from msmodelslim.onnx.squant_ptq.aok.optimizer.graph_optimizer import GraphOptimizer


class TestInferenceRunInfo:
    def test_normal_initialization(self):
        """测试正常初始化"""
        model_name = "test_model"
        latency = 0.123
        output = np.array([1, 2, 3])

        run_info = _InferenceRunInfo(model_name, latency, output)

        assert run_info.model_name == model_name
        assert run_info.latency == latency
        assert np.array_equal(run_info.output, output)


class TestGraphOptimizer:
    def setup_method(self):
        """测试前的初始化"""
        self.logger = MagicMock()
        self.optimizer = GraphOptimizer(self.logger)

    def test_normal_initialization(self):
        """测试正常初始化"""
        logger = MagicMock()  # 使用 Mock 对象模拟 logger
        optimizer = GraphOptimizer(logger)

        # 验证所有属性是否被正确初始化
        assert optimizer._logger == logger
        assert optimizer._opset_version is None
        assert optimizer._ir_version is None
        assert optimizer._soc_version is None
        assert optimizer._simplify is False
        assert optimizer._check_model is False
        assert optimizer._check_output_threshold is None
        assert optimizer._arch is None
        assert optimizer._debug is None
        assert optimizer._auto_quant_enabled is False
        assert optimizer._runner is None

    def test_delete_model(self):
        # 创建临时目录和文件
        with TemporaryDirectory() as temp_dir:
            model_name = "test_model"
            onnx_path = os.path.join(temp_dir, f"{model_name}.onnx")
            om_path = os.path.join(temp_dir, f"{model_name}.om")

            mode = 0o640

            # 创建测试文件
            fd = os.open(onnx_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
            with open(fd, "w") as f:
                f.write("ONNX model content")
            fd = os.open(om_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
            with open(fd, "w") as f:
                f.write("OM model content")

            # 测试正常删除
            GraphOptimizer.delete_model(temp_dir, model_name, delete_onnx=True)
            assert not os.path.exists(onnx_path)
            assert not os.path.exists(om_path)

            # 重新创建文件，测试删除 ONNX 文件
            fd = os.open(onnx_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
            with open(fd, "w") as f:
                f.write("ONNX model content")
            fd = os.open(om_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
            with open(fd, "w") as f:
                f.write("OM model content")

            GraphOptimizer.delete_model(temp_dir, model_name, delete_onnx=False)
            assert os.path.exists(onnx_path)
            assert not os.path.exists(om_path)

    def test_get_architectures(self):
        """测试 _get_architectures 方法"""
        # 获取架构字典
        archs = GraphOptimizer._get_architectures()

        # 验证返回值类型
        assert isinstance(archs, dict)

        # 验证预期的架构类是否都被加载
        expected_arch_names = [
            'default',
            'dummy',
            'bert',
            'roberta',
            'efficientnet',
            'mobilenetv2',
            'mobilenetv3',
            'shufflenetv2',
            'swin',
            'seresnet',
            'densenet',
            'u2net',
            'yolov5',
            'yolov5s1',
            'yolov7',
        ]

        # 验证字典中包含所有预期的架构名称
        assert set(archs.keys()) == set(expected_arch_names)

        # 验证每个架构类的实例化对象是否正确
        for arch_name, arch_instance in archs.items():
            assert arch_instance.get_name() == arch_name

    def test_delete_model_and_its_variations_normal(self):
        """测试正常删除模型及其变体"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_name = "test_model"
            onnx_path = os.path.join(temp_dir, f"{model_name}.onnx")
            om_path = os.path.join(temp_dir, f"{model_name}.om")
            quant_onnx_path = os.path.join(temp_dir, f"{model_name}q.onnx")
            quant_om_path = os.path.join(temp_dir, f"{model_name}q.om")

            mode = 0o640

            # 创建测试文件
            fd = os.open(onnx_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
            with open(fd, "w") as f:
                f.write("ONNX model content")
            fd = os.open(om_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
            with open(fd, "w") as f:
                f.write("OM model content")
            fd = os.open(quant_onnx_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
            with open(fd, "w") as f:
                f.write("Quantized ONNX model content")
            fd = os.open(quant_om_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
            with open(fd, "w") as f:
                f.write("Quantized OM model content")

            # 测试 _auto_quant_enabled 为 False
            self.optimizer._auto_quant_enabled = False
            self.optimizer.delete_model_and_its_variations(temp_dir, model_name, delete_onnx=True)
            assert not os.path.exists(onnx_path)
            assert not os.path.exists(om_path)
            assert os.path.exists(quant_onnx_path)
            assert os.path.exists(quant_om_path)

            # 重新创建文件
            fd = os.open(onnx_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
            with open(fd, "w") as f:
                f.write("ONNX model content")
            fd = os.open(om_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, mode)
            with open(fd, "w") as f:
                f.write("OM model content")

            # 测试 _auto_quant_enabled 为 True
            self.optimizer._auto_quant_enabled = True
            self.optimizer.delete_model_and_its_variations(temp_dir, model_name, delete_onnx=True)
            assert not os.path.exists(onnx_path)
            assert not os.path.exists(om_path)
            assert not os.path.exists(quant_onnx_path)
            assert not os.path.exists(quant_om_path)

    def test_set_methods(self):
        """测试所有设置方法"""
        # 测试 set_opset_version
        op_version = 12
        assert self.optimizer.set_opset_version(op_version)._opset_version == op_version

        # 测试 set_ir_version
        ir_version = 7
        assert self.optimizer.set_ir_version(ir_version)._ir_version == ir_version

        # 测试 set_soc_version
        soc_version = 1
        assert self.optimizer.set_soc_version(soc_version)._soc_version == soc_version

        # 测试 set_simplify
        simplify = True
        assert self.optimizer.set_simplify(simplify)._simplify == simplify

        # 测试 set_check_model
        check_model = True
        assert self.optimizer.set_check_model(check_model)._check_model == check_model

        # 测试 set_check_output_threshold
        threshold = 0.5
        assert self.optimizer.set_check_output_threshold(threshold)._check_output_threshold == threshold

        # 测试 set_architecture
        arch = "arm64"
        assert self.optimizer.set_architecture(arch)._arch == arch

        # 测试 set_debug
        debug = "verbose"
        assert self.optimizer.set_debug(debug)._debug == debug

        # 测试 set_runner
        runner = "custom_runner"
        assert self.optimizer.set_runner(runner)._runner == runner

        # 测试 set_auto_quantization
        auto_quant = True
        assert self.optimizer.set_auto_quantization(auto_quant)._auto_quant_enabled == auto_quant

    def test_set_methods_chaining(self):
        """测试方法链式调用"""
        op_version = 12
        ir_version = 7
        soc_version = 1
        simplify = True
        check_model = True
        threshold = 0.5
        arch = "arm64"
        debug = "verbose"
        runner = "custom_runner"
        auto_quant = True

        optimizer = (
            self.optimizer.set_opset_version(op_version)
            .set_ir_version(ir_version)
            .set_soc_version(soc_version)
            .set_simplify(simplify)
            .set_check_model(check_model)
            .set_check_output_threshold(threshold)
            .set_architecture(arch)
            .set_debug(debug)
            .set_runner(runner)
            .set_auto_quantization(auto_quant)
        )

        assert optimizer._opset_version == op_version
        assert optimizer._ir_version == ir_version
        assert optimizer._soc_version == soc_version
        assert optimizer._simplify == simplify
        assert optimizer._check_model == check_model
        assert optimizer._check_output_threshold == threshold
        assert optimizer._arch == arch
        assert optimizer._debug == debug
        assert optimizer._runner == runner
        assert optimizer._auto_quant_enabled == auto_quant
