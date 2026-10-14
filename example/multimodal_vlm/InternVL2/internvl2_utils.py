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
# pylint: disable=undefined-variable,unspecified-encoding

import os
import json
import random
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from example.common.security.path import get_valid_read_path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    try:
        aspect_ratio = orig_width / orig_height
    except ZeroDivisionError as ex:
        logging.error('orig_height can not be zero. %s', str(ex))  # noqa: F821
        raise ex

    # calculate the existing image aspect ratio
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if len(processed_images) != blocks:
        raise ValueError("The number of processed images does not match the expected number of blocks.")
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image_file = get_valid_read_path(image_file, is_dir=False)
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_textvqa_calibration(textvqa_path, calib_num=30, get_all_calib=False):
    val_json = 'textvqa_val.jsonl'
    calibration_dataset = []

    val_json_path = os.path.join(textvqa_path, val_json)
    val_json_path = get_valid_read_path(val_json_path)
    with open(val_json_path, 'r') as file:
        for line in file:
            line_dict = json.loads(line.strip())
            line_dict['text'] = line_dict['question']
            line_dict['image_url'] = line_dict['image']
            calibration_dataset.append(line_dict)

    if not get_all_calib:
        calibration_dataset = random.sample(calibration_dataset, calib_num)  # nosec B311

    return calibration_dataset


def get_tokenized_data(tokenizer, inputs, dtype=torch.float16):
    tokenization_data = []
    for _, input_item in tqdm(enumerate(inputs), total=len(inputs)):
        question = input_item.get('text')
        query = (
            '<|im_start|>system\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|im_end|>'
            + '<|im_start|>user\n<image>\n'
            + question
            + '<|im_end|><|im_start|>assistant\n'
        )
        image_url = input_item['image_url']
        pixel_values = load_image(image_url, max_num=12).to('npu').to(dtype)
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        tokenization_data.append([tokenizer, pixel_values, query, generation_config])
    return tokenization_data


def cmd_bool(cmd_arg):
    if cmd_arg == "True":
        return True
    elif cmd_arg == "False":
        return False
    raise ValueError(f"{cmd_arg} should be True or False")
