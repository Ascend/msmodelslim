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

from typing import Any, List, Tuple, Dict
from typing import Callable  # noqa: F401
from enum import Enum
from abc import abstractmethod
from dataclasses import dataclass
import torch

from ..common.quarot_utils import QuaRotMode, create_rot, GLOBAL_DTYPE


class RotSide(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class RotateCommand:
    target: str
    rot: Any
    side: RotSide


@dataclass
class RotatePair:
    left_rot: Dict[str, Any]
    right_rot: Dict[str, Any]


def _get_rotate_command(rotate_pair):
    commands = []
    for target, rot in rotate_pair.left_rot.items():
        commands.append(RotateCommand(target=target, rot=rot, side=RotSide.LEFT))
    for target, rot in rotate_pair.right_rot.items():
        commands.append(RotateCommand(target=target, rot=rot, side=RotSide.RIGHT))
    return commands


def get_rotate_command(rotate_pair):
    commands = []
    if isinstance(rotate_pair, List):
        for pair in rotate_pair:
            commands.extend(_get_rotate_command(pair))
    else:
        commands.extend(_get_rotate_command(rotate_pair))
    return commands


class QuaRotInterface:
    QuaRotMode = QuaRotMode
    RotatePair = RotatePair

    @staticmethod
    def get_rotate_command(
        mode: QuaRotMode, size: int, block_size: int = -1, rot_step: int = 1, eye_step: tuple = (-1,)
    ):
        rot = create_rot(mode, size, block_size, rot_step, eye_step)
        if hasattr(torch, 'npu') and torch.npu.is_available():
            rot = rot.to(dtype=GLOBAL_DTYPE, device=torch.device(f"npu:{torch.npu.current_device()}"))
        return rot

    @abstractmethod
    def get_ln_fuse_map(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Get the fusion mapping between LayerNorm layers and Linear layers.

        Returns:
            A tuple containing two dictionaries:
            - pre_run_fused_ln (Dict[str, List[str]]): The Norm fusion mapping for the pre-run phase.
            - fused_map (Dict[str, List[str]]): The Norm fusion mapping for the preprocess phase.
        """
        return {}, {}

    @abstractmethod
    def get_bake_names(self) -> Tuple[List[str], List[str]]:
        """
        Get a list of Linear layer names that require mean fusion.

        When the model uses nn.LayerNorm, mean preprocessing is required for the Linear layers
        before LayerNorm. This is usually not required to be configured.

        Returns:
            A tuple containing two lists of strings:
            - pre_run_bake_names (List[str]): The names for mean fusion in the pre-run phase.
            - bake_names (List[str]): The names for local mean fusion in the preprocess phase.
        """
        return [], []

    @abstractmethod
    def get_rotate_map(self, block_size: int) -> Tuple[List[RotatePair], List[RotatePair]]:
        """
        Get the rotation mapping, including configurations for left and right rotations.

        Args:
            block_size (int): The block size for rotation.

        Returns:
            A tuple containing two lists of RotatePair objects:
            - pre_run_pairs (List[RotatePair]): The rotation mapping for the pre-run phase,
              typically for embedding layer rotation.
            - rotate_pairs (List[RotatePair]): The rotation mapping for the preprocess phase.
        """
        return [], []


class LAOSOnlineRotationInterface:
    @abstractmethod
    def get_head_dim(self):
        pass

    @abstractmethod
    def get_num_attention_heads(self):
        pass

    @abstractmethod
    def get_layer_wise_ov_pair(self, decoder_module):
        pass

    @abstractmethod
    def get_layer_wise_up_down_pair(self, decoder_module):
        pass
