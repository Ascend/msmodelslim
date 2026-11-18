#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2025 Huawei Technologies Co.,Ltd.

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
import os

import numpy as np

from ascend_utils.common.security import get_valid_read_path, check_type, check_int
from msmodelslim import logger


MAX_PIXEL_VALUE = 255.0
THREE_DIMS = 3
SECOND_DIM = 2
THREE_CHANNELS = 3
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def _base_process_func_(data_path, image_mean=0.0, image_std=1.0, height=224, width=224):
    import cv2

    check_type(data_path, value_type=str, param_name="data_path")
    check_int(height, min_value=1, param_name="height")
    check_int(width, min_value=1, param_name="width")
    data_path = get_valid_read_path(data_path, is_dir=True)

    if np.sum(image_std) == 0:
        raise ValueError("image_std cannot be all zero")

    image_list = os.listdir(data_path)
    data_list = []
    for image_name in image_list:
        if not os.path.splitext(image_name)[-1].lower() in IMAGE_EXTENSIONS:
            continue
        image_filepath = get_valid_read_path(os.path.join(data_path, image_name))
        img = cv2.imread(image_filepath)
        if img is None or img.ndim != THREE_DIMS or img.shape[SECOND_DIM] < THREE_CHANNELS:
            logger.warning("File %s is invalid.", image_filepath)
            continue

        img_data = cv2.resize(img[:, :, :3], (width, height))
        img_data = img_data[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB -> channels_first
        norm_img_data = (img_data - image_mean) / image_std  # normalize
        data_list.append(norm_img_data.astype(np.float32))
    return data_list


def _get_all_batch_(data_list, batch_size):
    check_int(batch_size, min_value=1, param_name="batch_size")

    all_batch_data = []
    data_size = len(data_list)
    if data_size < batch_size:
        raise ValueError("The number of calibrated images is smaller than the batch size, please add some pictures.")
    if batch_size == 0:
        raise ValueError("Batch size can not be zero.")
    for batch_start in range(0, data_size // batch_size * batch_size, batch_size):
        per_batch_data = data_list[batch_start: batch_start + batch_size]
        per_batch_data = np.array(per_batch_data)
        all_batch_data.append(per_batch_data)
    return all_batch_data


def preprocess_func_imagenet(data_path, height=224, width=224, batch_size=1):
    """
    Loads a batch of images and preprocess them, mean=[0.485, 0.456, 0.406] * 255, std=[0.229, 0.224, 0.225] * 255
    parameter data_path: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    return: list of preprocess image data
    """
    image_mean = np.array([0.485, 0.456, 0.406])[:, None, None] * MAX_PIXEL_VALUE  # -> channels_first format
    image_std = np.array([0.229, 0.224, 0.225])[:, None, None] * MAX_PIXEL_VALUE  # -> channels_first format
    data_list = _base_process_func_(data_path, image_mean, image_std, height=height, width=width)
    return _get_all_batch_(data_list, batch_size)


def preprocess_func_coco(data_path, height=320, width=320, batch_size=1):
    """
    Loads a batch of images and preprocess them, mean=0, std=255
    parameter data_path: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    return: list of preprocess image data
    """
    image_mean = 0.0
    image_std = MAX_PIXEL_VALUE
    data_list = _base_process_func_(data_path, image_mean, image_std, height=height, width=width)
    return _get_all_batch_(data_list, batch_size)
