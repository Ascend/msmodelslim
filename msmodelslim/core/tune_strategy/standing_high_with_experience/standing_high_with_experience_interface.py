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

from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.model import IModel
from msmodelslim.utils.exception import UnsupportedError


class StandingHighWithExperienceInterface(IModel):
    """
    Interface for standing high with experience strategy.

    Only ``load_model`` is required here (anti-outlier capability probe).
    Sensitivity analysis uses ``PipelineInterface`` / ``ModelSlimPipelineInterfaceV1`` separately.
    """

    @abstractmethod
    def load_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """
        Load the model to specified device.
        After loading, the model should be ready for inference.

        Returns:
            nn.Module: The loaded model.
        """
        raise UnsupportedError(
            "This model does not support load model to specified device and torch dtype.",
            action="Please implement load_model in StandingHighWithExperienceInterface.",
        )
