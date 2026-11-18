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
# pylint: disable=logging-fstring-interpolation,not-callable

import os
import re
from typing import Any, Optional, Union, List, Tuple, Dict

from ascend_utils.common import security
from ascend_utils.common.security.type import check_element_type
from msmodelslim.common.low_rank_decompose import RankMethods
from msmodelslim import logger


class Decompose:
    def __init__(self, model: Any, config_file: str = None):
        self.model = model
        self.decompose_config = None
        self.get_decomposed_config_backend = None
        self.decompose_network_backend = None

        if config_file is not None and (not isinstance(config_file, str) or not config_file.endswith(".json")):
            raise ValueError("Parameter config_file is not a valid json file name.")
        self.config_file = config_file

    def from_fixed(self, channel_fixed: int, excludes: Optional[Union[List, Tuple]] = None, divisor: int = 64):
        """
        Init model layers decompose channels config by fixed int value.

        Args:
          channel_fixed: int value as fixed target decomposed channels, for each supported
              layer in model except those in `excludes`.
          excludes: list or tuple, layer names which will be not decompose. Mostly it's the input and output layers.
          divisor: int value for keeping target decomposed channels being divisible by `divisor`.
              Set `1` for disable.

        Raise ValueError if
          - first parameter is not an int, or <= 0.
          - excludes is not None or list or tuple.
          - divisor is not valid int.
          - initialized `model` is not a PyTorch or MindSpore model.
        """
        if not isinstance(channel_fixed, int) or channel_fixed <= 0:
            raise ValueError("Parameter channel_fixed is not a valid int value.")
        return self._get_decomposed_config_(channel_fixed, excludes, divisor)

    def from_ratio(self, channel_ratio: float, excludes: Optional[Union[List, Tuple]] = None, divisor: int = 64):
        """
        Init model layers decompose channels config by float ratio value.

        Args:
          channel_ratio: float value as ratio of original channels, calculated as target decomposed channels,
              for each supported layer in model except those in `excludes`.
          excludes: list or tuple, layer names which will be not decompose. Mostly it's the input and output layers.
          divisor: int value for keeping target decomposed channels being divisible by `divisor`.
              Set `1` for disable.

        Raise ValueError if
          - first parameter is not a float, or not in (0, 1).
          - excludes is not None or list or tuple.
          - divisor is not valid int.
          - initialized `model` is not a PyTorch or MindSpore model.
        """
        if not isinstance(channel_ratio, float) or channel_ratio <= 0 or channel_ratio >= 1:
            raise ValueError("Parameter channel_ratio is not a valid float value.")
        return self._get_decomposed_config_(channel_ratio, excludes, divisor)

    def from_dict(self, channel_dict: Dict, excludes: Optional[Union[List, Tuple]] = None, divisor: int = 64):
        """
        Init model layers decompose channels config by dict value.

        Args:
          channel_dict: float value as ratio of original channels, calculated as target decomposed channels,
              for each supported layer in model except those in `excludes`.
              Dict key can be a full name or regex matching model layer name, like ".*.output.fc".
              Dict value can be:
              - int for `from_fixed`.
              - float for `from_ratio`.
              - "vbmf" for `from_vbmf`.
          excludes: list or tuple, layer names which will be not decompose. Mostly it's the input and output layers.
          divisor: int value for keeping target decomposed channels being divisible by `divisor`.
              Set `1` for disable.

        Raise ValueError if
          - first parameter is not a dict.
          - excludes is not None or list or tuple.
          - divisor is not valid int.
          - initialized `model` is not a PyTorch or MindSpore model.
        """
        if not isinstance(channel_dict, dict) or len(channel_dict) == 0:
            raise ValueError("Parameter channel_dict is not a valid dict value.")

        try:
            channel_dict = {re.compile(kk): vv for kk, vv in channel_dict.items()}
        except re.error as compile_error:
            raise ValueError("Parameter channel_dict contains invalid key") from compile_error

        return self._get_decomposed_config_(channel_dict, excludes, divisor)

    def from_vbmf(self, excludes: Optional[Union[List, Tuple]] = None, divisor: int = 64):
        """
        Init model layers decompose channels config by VBMF (Variational Bayes Matrix Factorization) method.
        Better if model has pretrained weights.
        VBMF Paper: Global analytic solution of fully-observed variational Bayesian matrix factorization.

        Args:
          excludes: list or tuple, layer names which will be not decompose. Mostly it's the input and output layers.
          divisor: int value for keeping target decomposed channels being divisible by `divisor`.
              Set `1` for disable.

        Raise ValueError if
          - excludes is not None or list or tuple.
          - divisor is not valid int.
          - initialized `model` is not a PyTorch or MindSpore model.
        """
        return self._get_decomposed_config_(RankMethods.VBMF, excludes, divisor)

    def from_file(self):
        """
        Restore model layers decompose channels config from previous saved file.
        Only if initialized `config_file` is valid, and already called other from_xxx function.

        Raise ValueError if initialized `config_file` is not valid json file name, or not exists
        """
        if self.config_file is None or not os.path.exists(self.config_file):
            raise ValueError("Initialized config_file is not valid, or not initialized by other from_xxx functions.")
        self.decompose_config = security.json_safe_load(self.config_file)
        return self

    def decompose_network(self, do_decompose_weight: bool = True, datasets: Any = None, max_iter: int = -1):
        """
        Decompose model after initialized by any from_xxx function.

        Args:
          do_decompose_weight: boolean value if decompose model weights.
             - If False, will just convert model as a decomposed one, with weights being randomly initialized.
             - If True, will apply decomposition on model weights, and set to the decomposed model.
             Better True if model has pretrained weights.
          datasets: None or an iterative datasets with elements can be feed into model directly.
              If not None, will apply `data aware` decompose method while decomposing model Dense/Linear weights.
          max_iter: int value for datasets, max iteration steps. Default -1 for the entire datasets

        Returns: decomposed model.

        Raise ValueError if
          - `decompose_config` has not been initialized by calling `from_xxx` function.
          - initialized `model` is not a PyTorch or MindSpore model.
        """
        if self.decompose_config is None:
            raise ValueError("decompose_config not initialized, call from_xxx functions first.")
        if not isinstance(do_decompose_weight, bool):
            raise TypeError("parameter do_decompose_weight need to be a boolean value.")
        if not isinstance(max_iter, int):
            raise TypeError("parameter max_iter need to be an int value.")

        return self.decompose_network_backend(
            network=self.model,
            decompose_config=self.decompose_config,
            do_decompose_weight=do_decompose_weight,
            datasets=datasets,
            max_iter=max_iter,
        )

    def _get_decomposed_config_(self, hidden_channels, excludes=None, divisor=64):
        if excludes is not None and not isinstance(excludes, (list, tuple)):
            raise ValueError("Parameter excludes is not a valid list or tuple value.")
        elif not isinstance(divisor, (int, float)) or divisor <= 0:
            raise ValueError("Parameter divisor is not a valid int or float value.")
        else:
            if excludes is not None:
                check_element_type(excludes, str, value_type=(list, tuple), param_name="excludes")
            self.decompose_config = self.get_decomposed_config_backend(
                network=self.model,
                hidden_channels=hidden_channels,
                excludes=excludes,
                divisor=divisor,
            )
            if self.decompose_config is None or len(self.decompose_config) == 0:
                raise ValueError("Generated decompose_config is not valid, check model structure and parameters.")

            if self.config_file is not None:
                security.json_safe_dump(self.decompose_config, self.config_file)
                logger.info(f"Write decompose_config to {self.config_file}")

        return self
