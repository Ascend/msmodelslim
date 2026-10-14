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

from abc import abstractmethod

from msmodelslim.model import IModel


class ModelInfoInterface(IModel):
    """
    Interface for getting model information.
    """

    @abstractmethod
    def get_model_pedigree(self) -> str:
        """
        Get the pedigree of the model, e.g. "qwen3", "deepseek_v3", etc.
        Best practices categorized by pedigree in the repository.

        Returns:
            str: The pedigree of the model.
        """
        ...

    @abstractmethod
    def get_model_type(self) -> str:
        """
        Get the type of the model, e.g. "Qwen3-32B", "DeepSeek-R1", etc.
        Best practices labeled by model type, which is used to limit the best practice candidates.

        Returns:
            str: The type of the model.
        """
        ...
