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

from abc import ABC

from msmodelslim.core.analysis_service.pipeline_analysis.pipeline_loader_infra import (
    AnalysisPipelineLoaderInfra,
    PipelineBuilderInfra,
)


class _StubBuilder(PipelineBuilderInfra):
    """Minimal concrete builder for ABC contract tests."""

    def __init__(self):
        self._modules = []

    def template_modules(self, modules):
        self._modules = list(modules)
        return self

    def create(self):
        return list(self._modules) if self._modules else []


class _StubLoader(AnalysisPipelineLoaderInfra):
    def get_pipeline_builder(self, metrics: str) -> PipelineBuilderInfra:
        return _StubBuilder()


class TestPipelineBuilderInfra:
    """Tests for PipelineBuilderInfra chain."""

    def test_create_returns_configs_when_modules_provided(self):
        """场景：template_modules 后 create。预期：非空配置列表。"""
        builder = _StubBuilder()
        result = builder.template_modules(["layer.*"]).create()
        assert result == ["layer.*"]

    def test_create_returns_empty_list_when_no_modules(self):
        """场景：未 template_modules。预期：空列表。"""
        builder = _StubBuilder()
        assert not builder.create()


class TestAnalysisPipelineLoaderInfra:
    """Tests for AnalysisPipelineLoaderInfra."""

    def test_get_pipeline_builder_returns_builder_when_loader_called(self):
        """场景：stub loader。预期：返回 PipelineBuilderInfra 实例。"""
        loader = _StubLoader()
        builder = loader.get_pipeline_builder("std")
        assert isinstance(builder, PipelineBuilderInfra)

    def test_analysis_pipeline_loader_infra_is_abstract(self):
        """场景：直接实例化 ABC。预期：无法实例化。"""
        assert issubclass(AnalysisPipelineLoaderInfra, ABC)
