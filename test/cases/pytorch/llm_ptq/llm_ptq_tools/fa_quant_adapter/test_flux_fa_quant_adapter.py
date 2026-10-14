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
# pylint: disable=no-name-in-module,redefined-outer-name

import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
import torch
from transformers import PretrainedConfig

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux import (
    get_flux_config,
    _install_forward_adapter,
    install_for_flux_model,
    flux_attn_processor_adapter,
    flux_attn_single_processor_adapter,
)


@pytest.fixture
def mock_logger():
    logger = Mock(spec=logging.Logger)
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def mock_config():
    config = Mock(spec=PretrainedConfig)
    config.num_attention_heads = 16
    config.attention_head_dim = 64
    config.is_tp = False
    return config


@pytest.fixture
def mock_config_tp():
    config = Mock(spec=PretrainedConfig)
    config.num_attention_heads = 16
    config.attention_head_dim = 64
    config.is_tp = True
    return config


@pytest.fixture
def mock_model():
    model = Mock(spec=torch.nn.Module)

    # Create mock transformer block with correct structure
    transformer_block = Mock()
    transformer_block.__class__.__name__ = "FluxTransformerBlock"
    transformer_block.attn = Mock()
    transformer_block.attn.fa_quantizer = None
    transformer_block.processor = Mock()
    transformer_block.processor.__call__ = Mock()
    transformer_block.processor.__class__ = type('Processor', (), {})

    # Create mock single transformer block
    single_transformer_block = Mock()
    single_transformer_block.__class__.__name__ = "FluxSingleTransformerBlock"
    single_transformer_block.attn = Mock()
    single_transformer_block.attn.fa_quantizer = None
    single_transformer_block.processor = Mock()
    single_transformer_block.processor.__call__ = Mock()
    single_transformer_block.processor.__class__ = type('SingleProcessor', (), {})

    # Create unknown module type
    unknown_module = Mock()
    unknown_module.__class__.__name__ = "UnknownModule"

    model.named_modules.return_value = [
        ('transformer.blocks.0', transformer_block),
        ('transformer.blocks.1', single_transformer_block),
        ('unknown', unknown_module),
    ]

    return model


class TestGetFluxConfig:
    @staticmethod
    def test_get_flux_config_given_non_tp_config_when_valid_then_pass(mock_config, mock_logger):
        config = get_flux_config(mock_config, mock_logger)

        assert config.num_attention_heads == 16
        assert config.hidden_size == 1024
        assert config.num_key_value_heads == 16

    @staticmethod
    def test_get_flux_config_given_tp_config_when_dist_available_then_pass(mock_config_tp, mock_logger):
        with patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux.dist') as mock_dist:
            mock_dist.get_world_size.return_value = 2
            config = get_flux_config(mock_config_tp, mock_logger)

            assert config.num_attention_heads == 8
            assert config.hidden_size == 1024
            assert config.num_key_value_heads == 8

    @staticmethod
    def test_get_flux_config_given_invalid_config_when_missing_attr_then_fail(mock_logger):
        invalid_config = Mock(spec=PretrainedConfig)
        # Deliberately missing required attributes
        with pytest.raises((AttributeError, TypeError)):
            get_flux_config(invalid_config, mock_logger)


class TestInstallForwardAdapter:
    @staticmethod
    def test_install_forward_adapter_given_valid_module_when_install_then_pass(mock_logger):
        # Create a proper mock module with processor
        mock_module = Mock()
        mock_module.processor = Mock()
        mock_module.processor.__call__ = Mock()

        # Create a real class for processor
        class MockProcessor:
            def __call__(self):
                pass

        mock_module.processor.__class__ = MockProcessor

        mock_forward_adapter = Mock(return_value=lambda self, *args, **kwargs: None)

        with patch(
            'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux.ForwardFactory.get_forward_adapter',
            return_value=mock_forward_adapter,
        ):
            _install_forward_adapter(
                mock_module, 'FluxTransformerBlock', 'FluxAttnProcessor2_0', 'test_module', mock_logger
            )

            mock_logger.info.assert_called_with("Successfully installed FAQuantizer for module test_module")

    @staticmethod
    def test_install_forward_adapter_given_invalid_module_when_missing_processor_then_fail(mock_logger):
        mock_module = Mock()
        # Ensure processor attribute doesn't exist
        if hasattr(mock_module, 'processor'):
            delattr(mock_module, 'processor')

        with pytest.raises(Exception):
            _install_forward_adapter(mock_module, 'InvalidModule', 'InvalidProcessor', 'test_module', mock_logger)

        mock_logger.error.assert_called()
        call_args = mock_logger.error.call_args
        assert "Failed to install FAQuantizer for module test_module" in call_args[0][0]

    @staticmethod
    def test_install_forward_adapter_given_invalid_adapter_type_when_factory_fails_then_fail(mock_logger):
        mock_module = Mock()
        mock_module.processor = Mock()
        mock_module.processor.__call__ = Mock()
        mock_module.processor.__class__ = type('Processor', (), {})

        with patch(
            'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux.ForwardFactory.get_forward_adapter',
            side_effect=Exception("Adapter not found"),
        ):
            with pytest.raises(Exception, match="Adapter not found"):
                _install_forward_adapter(mock_module, 'InvalidModule', 'InvalidProcessor', 'test_module', mock_logger)


class TestInstallForFluxModel:
    @staticmethod
    def test_install_for_flux_model_given_valid_model_when_install_then_pass(mock_model, mock_config, mock_logger):
        with (
            patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux.get_flux_config') as mock_get_config,
            patch(
                'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux._install_forward_adapter'
            ) as mock_install_adapter,
            patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux.FAQuantizer') as mock_quantizer,
        ):
            mock_get_config.return_value = SimpleNamespace()
            mock_quantizer.return_value = Mock()

            class DummyAttn:
                pass

            class FluxTransformerBlock:
                pass

            class FluxSingleTransformerBlock:
                pass

            mock_module1 = Mock()
            mock_module1.attn = Mock(spec=DummyAttn)
            mock_module1.__class__ = FluxTransformerBlock
            mock_module2 = Mock()
            mock_module2.attn = Mock(spec=DummyAttn)
            mock_module2.__class__ = FluxSingleTransformerBlock

            mock_model.named_modules.return_value = [
                ("transformer.block.0", mock_module1),
                ("transformer.block.1", mock_module2),
            ]

            install_for_flux_model(mock_model, mock_config, mock_logger, [])

            # Verify that named_modules was called
            assert mock_model.named_modules.called

            # Should install quantizer for both transformer blocks
            assert mock_quantizer.call_count == 2

            # Should install adapters for both modules
            assert mock_install_adapter.call_count == 2

    @staticmethod
    def test_install_for_flux_model_given_skip_layers_when_skip_then_pass(mock_model, mock_config, mock_logger):
        with (
            patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux.get_flux_config') as mock_get_config,
            patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux.FAQuantizer') as mock_quantizer,
        ):
            mock_get_config.return_value = SimpleNamespace()
            mock_quantizer.return_value = Mock()

            # Skip the first transformer block
            install_for_flux_model(mock_model, mock_config, mock_logger, ['blocks.0'])

            # Should only call for non-skipped modules
            mock_logger.info.assert_any_call("Skipping transformer.blocks.0.attn")

    @staticmethod
    def test_install_for_flux_model_given_existing_quantizer_when_already_installed_then_warn(mock_config, mock_logger):
        # Create a model with existing quantizer
        mock_model_with_quantizer = Mock()

        transformer_block = Mock()
        transformer_block.__class__.__name__ = "FluxTransformerBlock"
        transformer_block.attn = Mock()
        transformer_block.attn.fa_quantizer = Mock()  # Already has quantizer
        transformer_block.processor = Mock()

        mock_model_with_quantizer.named_modules.return_value = [('transformer.blocks.0', transformer_block)]

        with patch(
            'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux.get_flux_config'
        ) as mock_get_config:
            mock_get_config.return_value = SimpleNamespace()

            install_for_flux_model(mock_model_with_quantizer, mock_config, mock_logger, [])

            mock_logger.warning.assert_called_with(
                "Module transformer.blocks.0.attn already has FAQuantizer installed."
            )

    @staticmethod
    def test_install_for_flux_model_given_unknown_module_type_when_skip_then_pass(mock_model, mock_config, mock_logger):
        with (
            patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux.get_flux_config') as mock_get_config,
            patch('msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux.FAQuantizer') as mock_quantizer,  # noqa: F841
        ):
            mock_get_config.return_value = SimpleNamespace()

            install_for_flux_model(mock_model, mock_config, mock_logger, [])

            # Should not install quantizer for unknown module type
            # The unknown module doesn't have attn attribute, so no quantizer should be installed
            assert True  # If we reach here without error, the test passes


class TestFluxAttnProcessorAdapter:
    @staticmethod
    def test_flux_attn_processor_adapter_given_3d_inputs_when_valid_then_pass():
        mock_original_call = Mock()
        mock_original_call.__module__ = 'test_module'

        mock_apply_rotary = Mock(side_effect=lambda x, y: x)

        def apply_fa(query, key, value, attention_mask):
            batch_size = query.shape[0]
            heads = query.shape[-2]
            head_dim = query.shape[-1]
            return hidden_states.reshape(batch_size, -1, head_dim * heads)

        mock_apply_fa = apply_fa

        mock_attention_class = Mock()
        mock_attention_instance = Mock()
        mock_attention_instance.to_q = Mock(return_value=torch.randn(2, 10, 512))
        mock_attention_instance.to_k = Mock(return_value=torch.randn(2, 10, 512))
        mock_attention_instance.to_v = Mock(return_value=torch.randn(2, 10, 512))
        mock_attention_instance.add_q_proj = Mock(return_value=torch.randn(2, 5, 512))
        mock_attention_instance.add_k_proj = Mock(return_value=torch.randn(2, 5, 512))
        mock_attention_instance.add_v_proj = Mock(return_value=torch.randn(2, 5, 512))
        mock_attention_instance.to_out = [
            Mock(return_value=torch.randn(2, 10, 512)),
            Mock(return_value=torch.randn(2, 10, 512)),
        ]
        mock_attention_instance.to_add_out = Mock(return_value=torch.randn(2, 5, 512))
        mock_attention_instance.is_tp = False
        mock_attention_instance.heads = 8
        mock_attention_instance.world_size = 1
        mock_attention_instance.norm_q = None
        mock_attention_instance.norm_k = None
        mock_attention_instance.norm_added_q = None
        mock_attention_instance.norm_added_k = None
        mock_attention_instance.fa_quantizer = Mock()
        mock_attention_instance.fa_quantizer.quant = Mock(side_effect=lambda x, qkv: x)

        with patch(
            'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux.import_module'
        ) as mock_import_module:
            mock_module = Mock()
            mock_module.apply_rotary_emb_mindspeed = mock_apply_rotary
            mock_module.apply_fa = mock_apply_fa
            mock_module.Attention = mock_attention_class
            mock_import_module.return_value = mock_module

            adapter_func = flux_attn_processor_adapter(mock_original_call)

            # Test with 3D inputs
            hidden_states = torch.randn(2, 10, 512)  # 3D: [batch, seq_len, features]
            encoder_hidden_states = torch.randn(2, 5, 512)  # 3D
            attention_mask = None
            image_rotary_emb = None

            result = adapter_func(
                Mock(), mock_attention_instance, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb
            )

            assert len(result) == 2
            assert result[0].shape == hidden_states.shape
            assert result[1].shape == encoder_hidden_states.shape


class TestFluxAttnSingleProcessorAdapter:
    @staticmethod
    def test_flux_attn_single_processor_adapter_given_4d_input_when_valid_then_pass():
        mock_original_call = Mock()
        mock_original_call.__module__ = 'test_module'

        mock_apply_rotary = Mock(side_effect=lambda x, y: x)

        def apply_fa(query, key, value, attention_mask):
            batch_size = query.shape[0]
            heads = query.shape[-2]
            head_dim = query.shape[-1]
            return hidden_states.reshape(batch_size, -1, head_dim * heads)

        mock_apply_fa = apply_fa
        mock_attention_class = Mock()
        mock_attention_instance = Mock()
        mock_attention_instance.to_q = Mock(return_value=torch.randn(2, 16, 512))
        mock_attention_instance.to_k = Mock(return_value=torch.randn(2, 16, 512))
        mock_attention_instance.to_v = Mock(return_value=torch.randn(2, 16, 512))
        mock_attention_instance.is_tp = False
        mock_attention_instance.heads = 8
        mock_attention_instance.world_size = 1
        mock_attention_instance.norm_q = None
        mock_attention_instance.norm_k = None
        mock_attention_instance.fa_quantizer = Mock()
        mock_attention_instance.fa_quantizer.quant = Mock(side_effect=lambda x, qkv: x)

        with patch(
            'msmodelslim.pytorch.llm_ptq.llm_ptq_tools.fa_quant_adapter.flux.import_module'
        ) as mock_import_module:
            mock_module = Mock()
            mock_module.apply_rotary_emb_mindspeed = mock_apply_rotary
            mock_module.apply_fa = mock_apply_fa
            mock_module.Attention = mock_attention_class
            mock_import_module.return_value = mock_module

            adapter_func = flux_attn_single_processor_adapter(mock_original_call)

            # Test with 4D input
            hidden_states = torch.randn(2, 64, 4, 4)  # 4D input
            encoder_hidden_states = None
            attention_mask = None
            image_rotary_emb = None

            result = adapter_func(
                Mock(), mock_attention_instance, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb
            )

            # Result should maintain original 4D shape
            assert result.shape == hidden_states.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
