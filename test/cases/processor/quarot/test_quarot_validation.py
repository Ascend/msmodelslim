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
# pylint: disable=pointless-string-statement

"""
Test validation logic for QuaRotProcessorConfig
"""

import pytest  # noqa: E402

from msmodelslim.processor.quarot.offline_quarot.quarot import QuaRotProcessorConfig  # noqa: E402
from msmodelslim.utils.exception import SchemaValidateError  # noqa: E402


class TestQuaRotProcessorConfigValidation:
    """Test validation logic for QuaRotProcessorConfig"""

    def test_block_size_valid_values(self):
        """Test block_size with valid values"""
        # Test -1 (special value)
        config = QuaRotProcessorConfig(block_size=-1)
        assert config.block_size == -1

        # Test powers of 2
        valid_powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for power in valid_powers:
            config = QuaRotProcessorConfig(block_size=power)
            assert config.block_size == power

    def test_block_size_invalid_negative(self):
        """Test block_size with invalid negative values"""
        invalid_values = [-2, -3, -10, -100]
        for value in invalid_values:
            with pytest.raises(SchemaValidateError) as exc_info:
                QuaRotProcessorConfig(block_size=value)

            assert f"block_size must be -1 or a positive power of 2, got {value}" in str(exc_info.value)

    def test_block_size_invalid_zero(self):
        """Test block_size with zero"""
        with pytest.raises(SchemaValidateError) as exc_info:
            QuaRotProcessorConfig(block_size=0)

        assert "block_size must be -1 or a positive power of 2, got 0" in str(exc_info.value)

    def test_block_size_invalid_non_power_of_two(self):
        """Test block_size with non-power-of-two positive values"""
        invalid_values = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 24, 30, 33, 40, 50, 100]
        for value in invalid_values:
            with pytest.raises(SchemaValidateError) as exc_info:
                QuaRotProcessorConfig(block_size=value)

            assert f"block_size must be -1 or a positive power of 2, got {value}" in str(exc_info.value)

    def test_max_tp_size_valid_values(self):
        """Test max_tp_size with valid values"""
        valid_powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        for power in valid_powers:
            config = QuaRotProcessorConfig(max_tp_size=power)
            assert config.max_tp_size == power

    def test_max_tp_size_invalid_negative(self):
        """Test max_tp_size with invalid negative values"""
        invalid_values = [-1, -2, -3, -10, -100]
        for value in invalid_values:
            with pytest.raises(SchemaValidateError) as exc_info:
                QuaRotProcessorConfig(max_tp_size=value)

            assert f"max_tp_size must be a positive power of 2 or equal to 1, got {value}" in str(exc_info.value)

    def test_max_tp_size_invalid_zero(self):
        """Test max_tp_size with zero"""
        with pytest.raises(SchemaValidateError) as exc_info:
            QuaRotProcessorConfig(max_tp_size=0)

        assert "max_tp_size must be a positive power of 2 or equal to 1, got 0" in str(exc_info.value)

    def test_max_tp_size_invalid_non_power_of_two(self):
        """Test max_tp_size with non-power-of-two values"""
        invalid_values = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 24, 30, 33, 40, 50, 100]
        for value in invalid_values:
            with pytest.raises(SchemaValidateError) as exc_info:
                QuaRotProcessorConfig(max_tp_size=value)

            assert f"max_tp_size must be a positive power of 2 or equal to 1, got {value}" in str(exc_info.value)

    def test_combined_valid_config(self):
        """Test combined valid configuration"""
        config = QuaRotProcessorConfig(block_size=64, max_tp_size=4, online=True, down_proj_online_layers=[0, 1, 2])

        assert config.block_size == 64
        assert config.max_tp_size == 4
        assert config.online
        assert config.down_proj_online_layers == [0, 1, 2]

    def test_combined_invalid_config(self):
        """Test combined configuration with invalid values"""
        # Test with invalid block_size and valid max_tp_size
        with pytest.raises(SchemaValidateError) as exc_info:
            QuaRotProcessorConfig(
                block_size=3,  # Invalid: not a power of 2
                max_tp_size=4,  # Valid
                online=True,
            )

        assert "block_size must be -1 or a positive power of 2, got 3" in str(exc_info.value)

        # Test with valid block_size and invalid max_tp_size
        with pytest.raises(SchemaValidateError) as exc_info:
            QuaRotProcessorConfig(
                block_size=64,  # Valid
                max_tp_size=3,  # Invalid: not a power of 2
                online=True,
            )

        assert "max_tp_size must be a positive power of 2 or equal to 1, got 3" in str(exc_info.value)
