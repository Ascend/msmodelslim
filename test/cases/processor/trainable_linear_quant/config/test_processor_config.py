#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pytest

from msmodelslim.core.quantizer.base import QConfig
from msmodelslim.core.quantizer.linear import LinearQConfig
from msmodelslim.ir.qal import QDType, QScope
from msmodelslim.processor.trainable_linear_quant.config import (
    BlockTrainConfig,
    QuantStrategyConfig,
    TrainableLinearQuantProcessorConfig,
)
from msmodelslim.utils.exception import SchemaValidateError


def _int8_qconfig() -> LinearQConfig:
    return LinearQConfig(
        act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
        weight=QConfig(dtype=QDType.INT8, scope=QScope.PER_CHANNEL, symmetric=True, method="minmax"),
    )


class TestTLQConfigValidator:
    def test_valid_default_config(self):
        config = TrainableLinearQuantProcessorConfig(
            strategies=[QuantStrategyConfig(qconfig=_int8_qconfig())],
        )
        assert config.type == "trainable_linear_quant"

    def test_invalid_operation_type_raises(self):
        with pytest.raises(SchemaValidateError, match="not a registered TLQ op"):
            TrainableLinearQuantProcessorConfig(
                operations=[{"type": "not_a_real_op"}],
                strategies=[QuantStrategyConfig(qconfig=_int8_qconfig())],
            )

    def test_invalid_select_best_mode_raises(self):
        with pytest.raises(Exception):
            TrainableLinearQuantProcessorConfig(
                train_config=BlockTrainConfig(select_best={"mode": "invalid"}),
                strategies=[QuantStrategyConfig(qconfig=_int8_qconfig())],
            )

    def test_per_group_without_group_size_raises(self):
        weight = QConfig(
            dtype=QDType.INT8,
            scope=QScope.PER_GROUP,
            symmetric=True,
            method="minmax",
        )
        qconfig = LinearQConfig(
            act=QConfig(dtype=QDType.FLOAT, scope=QScope.PER_TENSOR, symmetric=True, method="none"),
            weight=weight,
        )
        with pytest.raises(SchemaValidateError, match="group_size"):
            TrainableLinearQuantProcessorConfig(
                strategies=[QuantStrategyConfig(qconfig=qconfig)],
            )

    def test_empty_strategies_raises(self):
        with pytest.raises(SchemaValidateError, match="strategies"):
            TrainableLinearQuantProcessorConfig(strategies=[])

    def test_pipeline_fields_on_processor_config(self):
        config = TrainableLinearQuantProcessorConfig(
            strategies=[QuantStrategyConfig(qconfig=_int8_qconfig())],
            train_with_act_quant=True,
            enable_quanted_input=True,
        )
        assert config.train_with_act_quant is True
        assert config.enable_quanted_input is True
        assert "enable_quanted_input" not in BlockTrainConfig.model_fields
        assert "train_with_act_quant" not in BlockTrainConfig.model_fields
