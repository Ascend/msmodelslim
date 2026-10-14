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
# pylint: disable=consider-using-from-import,useless-object-inheritance

from ascend_utils.common.security import check_type
from ascend_utils.common.security import check_element_type
from ascend_utils.common.security import check_int

from ascend_utils.common.utils import check_model_backend
from msmodelslim import logger


class KnowledgeDistillConfig(object):
    """
    Configuration for distillation. It defines the hard label and the soft label of teacher/student.
    The total Loss to be optimized is calculated by the hard label and the soft label.

    Total_loss = student_loss * hard_label_loss_weight + soft_loss * (1 - hard_label_loss_weight)
    soft_loss = sum(loss_func(t_module, s_module) * func_weight)

    Parameters of hard_label_loss_weight/loss_func/t_module/s_module/func_weight can be defined in set_hard_label
    and add_inter_soft_label.

    Example:
        distill_config = KnowledgeDistillConfig()
        distill_config.set_hard_label(0.5, 0) \
            .add_inter_soft_label({
                "t_module": "uniter.encoder.encoder.blocks.11.output",
                "s_module": "uniter.encoder.encoder.blocks.5.output",
                "t_output_idx": 0,
                "s_output_idx": 0,
                "loss_func": [{"func_name": "HiddenMse",
                               "func_weight": 1}],
                "shape": [2048]  # only for MindSpore
            })
    """

    def __init__(self):
        self.train_teacher = False
        self.model_parallel = False

        self.hard_label_loss_weight = 0.5
        self.output_replace_idx = 0
        self.inter_matches = []
        self.output_matches = []

        self.inter_match_keys = ["t_module", "s_module", "t_output_idx", "s_output_idx", "loss_func"]
        self.output_match_keys = ["t_output_idx", "s_output_idx", "loss_func"]

        self.custom_loss_func = {}

    @staticmethod
    def generate_loss_instance(config, distill_loss_func, func_type):
        for inter_match in config.inter_matches:
            KnowledgeDistillConfig.get_loss_instance(inter_match, distill_loss_func, func_type)
        for output_match in config.output_matches:
            KnowledgeDistillConfig.get_loss_instance(output_match, distill_loss_func, func_type)

    @staticmethod
    def get_loss_instance(match, distill_loss_func, func_type):
        for loss_func in match["loss_func"]:
            loss_name = loss_func["func_name"]
            loss_param = loss_func["func_param"]
            if isinstance(distill_loss_func[loss_name], func_type):
                func_instance = distill_loss_func[loss_name]
            else:
                try:
                    func_instance = distill_loss_func[loss_name](*loss_param)
                except TypeError as ex:
                    logger.error(
                        "Please check loss_param and loss_name, current loss_param is: %r, loss_name is: %r",
                        loss_param,
                        loss_name,
                    )
                    raise ex
            loss_func["func_instance"] = func_instance

    @staticmethod
    def check_config(config, distill_loss_func, func_type, is_mindspore=True):
        check_type(config, KnowledgeDistillConfig, param_name="config")
        KnowledgeDistillConfig.check_loss_func_name(config.inter_matches, distill_loss_func)
        KnowledgeDistillConfig.check_loss_func_name(config.output_matches, distill_loss_func)
        KnowledgeDistillConfig.check_and_add_custom_loss_func(config.custom_loss_func, distill_loss_func, func_type)
        if is_mindspore:
            KnowledgeDistillConfig.check_ms_extra_config(config)

    @staticmethod
    def check_loss_func_name(matches, distill_loss_func):
        for match in matches:
            for loss_func in match.get("loss_func"):
                func_name = loss_func.get("func_name")
                if func_name not in distill_loss_func:
                    raise ValueError("The func name \"{}\" not exist. please check config.".format(func_name))

    @staticmethod
    def check_and_add_custom_loss_func(custom_loss_func, distill_loss_func, func_type):
        for name, loss_func in custom_loss_func.items():
            if not isinstance(loss_func, func_type):
                raise TypeError(
                    "Custom loss func \"{}\" must be a {}, not {}".format(name, type(func_type), type(loss_func))
                )
            if distill_loss_func.get(name):
                loss_func_list = list(distill_loss_func.keys())
                logger.warning("loss function name in %r. default %r will be overwritten.", str(loss_func_list), name)
            distill_loss_func[name] = custom_loss_func[name]

    @staticmethod
    def check_ms_extra_config(config):
        if config.output_replace_idx is None:
            raise ValueError("must use method \"set_hard_label\" to set weight and index")
        for inter_config in config.inter_matches:
            if not inter_config.get("shape"):
                raise ValueError("config in \"add_inter_soft_label\" must has key: \"shape\"")
            check_element_type(inter_config["shape"], int, value_type=list, param_name='inter_config["shape"]')

    def add_custom_loss_func(self, name: str, instance):
        """
        For PyTorch:
            Add a new loss function to DISTILL_LOSS_FUNC_Torch. And use it in config.
            DISTILL_LOSS_FUNC_Torch default provides KDMse/HardKDCrossEntropy/KDCrossEntropy/HiddenMse/MMD.

        For MindSpore:
            Add a new loss function to DISTILL_LOSS_FUNC_MS. And use it in config.
            DISTILL_LOSS_FUNC_MS default provides KDMse/KLDivLoss/KDCrossEntropy/HiddenMse.

        Args:
            name(str): the name of loss function.
            instance: the instance of loss function. nn.Module for PyTorch, nn.Cell for MindSpore.
        """
        check_type(name, str, param_name="name")

        # check instance
        is_valid = False
        try:
            import torch.nn as nn

            if isinstance(instance, nn.Module):
                is_valid = True
        except ImportError:  # PyTorch未安装时跳过
            pass

        if not is_valid:  # 如果PyTorch校验未通过，检查MindSpore
            try:
                import mindspore.nn as nn_ms

                if isinstance(instance, nn_ms.Cell):
                    is_valid = True
            except ImportError:  # MindSpore未安装时跳过
                pass

        if not is_valid:
            raise TypeError("`instance` must be a PyTorch `nn.Module` or MindSpore `nn.Cell`. ")

        self.custom_loss_func[name] = instance

    def set_hard_label(self, weight: float, index: int = None):
        """
        Set the config to create a hard label.

        Args:
            weight(float): the weight of student's loss
            index(int): The index of student's output(only for MindSpore). if student has several outputs, one output
                can be chosen to calculate the loss. Most time index can be 0.
        """
        check_type(weight, float, param_name="weight")
        if index < 0:
            raise ValueError("index must be greater than 0, but input {}".format(index))
        if weight >= 1 or weight <= 0:
            raise ValueError("weight must be between 0 and 1, but input {}".format(weight))
        self.hard_label_loss_weight = weight
        if index:
            check_type(index, int, param_name="index")
            self.output_replace_idx = index
        return self

    def add_inter_soft_label(self, config: dict):
        """
        Add a config to create a soft label for specific layer.
        You can choose a layer of teacher and student to calculate loss.

        Config example:
                {
                    "t_module": "uniter.encoder.encoder.blocks.11.output",
                    "s_module": "uniter.encoder.encoder.blocks.5.output",
                    "t_output_idx": 0,
                    "s_output_idx": 0,
                    "loss_func": [{"func_name": "HiddenMse",
                                   "func_weight": 1,
                                   "temperature": 1,  # default None.
                                   "func_param": []}],  # default [].
                    "shape": [2048]  # only for MindSpore
                }

        Meaning of fields:
            t_module: name of teacher layer.
            s_module: name of student layer.
            t_output_idx: index of the t_module output. if t_module has several outputs, one output can be chosen to
                calculate the loss. Most time t_output_idx can be 0.
            s_output_idx: index of the s_module output. if s_module has several outputs, one output can be chosen to
                calculate the loss. Most time s_output_idx can be 0.
            loss_func: a list of loss function.
            func_name: please choose a loss function name in DISTILL_LOSS_FUNC_MS/DISTILL_LOSS_FUNC_TORCH. Custom loos
                func can be added by add_custom_loss_func().
            func_weight: weight for loss. You can control the weight of every loss.
            temperature: the parameter for KDMse/KLDivLoss/KDCrossEntropy.
            func_param: reserve fields. Initialization parameters for loss function.
            shape: the shape of t_module's output[t_output_idx]
        """
        self._check_inter_match(config)
        self.inter_matches.append(config)
        return self

    def add_output_soft_label(self, config: dict):
        """
        Add a config to create a soft label for the last layer.
        You must use t_module/s_module/shape to create config in add_inter_soft_label(). But in add_output_soft_label(),
        default use the last layer of model without specifying t_module/s_module/shape.

        config example:
            {
                "t_output_idx": 0,
                "s_output_idx": 0,
                "loss_func": [{"func_name": "HiddenMse",
                               "func_weight": 1,
                               "temperature": 1,  # default None
                               "func_param": []}],  # default []
            }

        Meaning of fields:
            Please refer to add_inter_soft_label()
        """
        self._check_output_match(config)
        self.output_matches.append(config)
        return self

    def set_teacher_train(self):
        """
        Do not use this method unless you need to train the teacher.
        """
        self.train_teacher = True
        return self

    def get_soft_label_shape(self):
        soft_label_shape = {}
        for inter_match in self.inter_matches:
            soft_label_shape[inter_match["t_module"]] = inter_match["shape"]
            soft_label_shape[inter_match["s_module"]] = inter_match["shape"]
        return soft_label_shape

    def _check_config_key(self, config, keys):
        lost_key = []
        for key in keys:
            if key not in config:
                lost_key.append(key)
        if len(lost_key) > 0:
            raise ValueError("config must have key: {}".format(str(lost_key)))

    def _check_output_match(self, config):
        check_type(config, dict, param_name="config")
        self._check_config_key(config, self.output_match_keys)
        check_int(config["t_output_idx"], min_value=0, param_name='config["t_output_idx"]')
        check_int(config["s_output_idx"], min_value=0, param_name='config["s_output_idx"]')
        for loss_func in config["loss_func"]:
            self._check_loss_func(loss_func)

    def _check_inter_match(self, config):
        check_type(config, dict, param_name="config")
        self._check_config_key(config, self.inter_match_keys)
        check_int(config["t_output_idx"], min_value=0, param_name='config["t_output_idx"]')
        check_int(config["s_output_idx"], min_value=0, param_name='config["s_output_idx"]')
        check_type(config["t_module"], str, param_name='config["t_module"]')
        check_type(config["s_module"], str, param_name='config["s_module"]')
        for loss_func in config["loss_func"]:
            self._check_loss_func(loss_func)

    def _check_loss_func(self, loss_func):
        if not loss_func.get("func_name"):
            raise ValueError("loss_func must have func_name")
        if not loss_func.get("func_weight"):
            raise ValueError("loss_func must have func_weight")

        if loss_func.get("temperature"):
            check_type(
                loss_func.get("temperature"),
                (int, float),
                param_name="temperature",
                additional_check_func=lambda x: x > 0,
            )
        elif not loss_func.get("temperature"):
            loss_func["temperature"] = None

        if loss_func.get("func_param") and not isinstance(loss_func.get("func_param"), list):
            raise TypeError("func_param must be list")
        elif not loss_func.get("func_param"):
            loss_func["func_param"] = []


def get_distill_model(teacher, student, config: KnowledgeDistillConfig):
    """
    Build a model for knowledge distillation that contains teacher, student, and loss functions.
    And you can get fine-tuned student model from this model after training.

    Args:
        teacher: teacher model.
        student: student model.
        config(KnowledgeDistillConfig): Configuration for knowledge distillation.

    Returns:
        a model contains teacher and student.

    Examples:
        >>> distill_config = KnowledgeDistillConfig()
        >>> # set distill_config
        >>> distill_model = get_distill_model(teacher, student, distill_config)
        >>> # train distill_model
        >>> distilled_student_model = distill_model.get_student_model()
    """

    backend = check_model_backend(teacher)
    if backend == "mindspore":
        from msmodelslim.mindspore.knowledge_distill.knowledge_distill_ms import get_distill_model_ms

        return get_distill_model_ms(teacher, student, config)
    elif backend == "pytorch":
        from msmodelslim.pytorch.knowledge_distill.knowledge_distill_torch import get_distill_model_torch

        return get_distill_model_torch(teacher, student, config)
    raise TypeError("Only support MindSpore and PyTorch!")
