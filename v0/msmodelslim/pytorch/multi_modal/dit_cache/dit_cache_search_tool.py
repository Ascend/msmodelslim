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

from dataclasses import dataclass, fields, replace
import multiprocessing
import time
import os
import logging
from functools import wraps
from typing import Optional

import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def get_local_rank():
    if 'LOCAL_RANK' not in os.environ:
        return 0

    local_rank_str = os.environ['LOCAL_RANK']

    # 1. check whether LOCAL_RANK is empty
    if not local_rank_str.strip():
        raise ValueError("LOCAL_RANK environment variable is empty")

    # 2. check whether LOCAL_RANK contains non-digit characters
    if not local_rank_str.isdigit():
        raise ValueError(f"LOCAL_RANK includes non-digit characters: {local_rank_str}")

    # 3. check whether LOCAL_RANK is a valid integer
    try:
        local_rank = int(local_rank_str)
    except ValueError as e:
        raise ValueError(f"LOCAL_RANK conversion to integer failed: {local_rank_str}") from e

    # 4. check whether LOCAL_RANK is negative
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK cannot be negative: {local_rank}")

    return local_rank


def log_if_rank_0(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if get_local_rank() == 0:
            func(*args, **kwargs)

    return wrapper


@log_if_rank_0
def logger_info(*args, **kwargs):
    logger.info(*args, **kwargs)


@log_if_rank_0
def logger_debug(*args, **kwargs):
    logger.debug(*args, **kwargs)


@dataclass
class DitCacheSearcherConfig:
    # 传入的
    dit_block_num: int = None
    prompts_num: int = 1
    num_sampling_steps: int = None
    cache_ratio: float = 1.5
    search_cache_path: str = None
    cache_step_interval: int = 2

    # 搜索中用到的参数
    use_cache: Optional[bool] = False
    cache_step_start: Optional[int] = 0
    cache_block_start: Optional[int] = 0
    cache_num_blocks: Optional[int] = 0

    def __post_init__(self):
        if not isinstance(self.cache_ratio, (int, float)):
            raise ValueError("cache_ratio must be a number")
        if not (1.0 <= self.cache_ratio <= 2.0):
            raise ValueError(f"cache_ratio must be between 1.0 and 2.0 (inclusive), got {self.cache_ratio}")

        if self.dit_block_num is not None:
            if not isinstance(self.dit_block_num, int):
                raise ValueError("dit_block_num must be an integer")
            if self.dit_block_num <= 0:
                raise ValueError("dit_block_num must be positive")

        if self.num_sampling_steps is None:
            raise ValueError("num_sampling_steps must be set to search")
        elif not isinstance(self.num_sampling_steps, int):
            raise ValueError("num_sampling_steps must be an integer")
        elif self.num_sampling_steps <= 0:
            raise ValueError("num_sampling_steps must be positive")

        if not isinstance(self.cache_step_interval, int):
            raise ValueError("cache_step_interval must be an integer")
        elif self.cache_step_interval < 2:
            raise ValueError("cache_step_interval for searching must be larger or equal to 2")

    def __repr__(self) -> str:
        field_reprs = []
        for field in fields(self):
            value = getattr(self, field.name)

            # 对特殊类型进行格式化处理
            if isinstance(value, str):
                # 转译字符串中的特殊字符
                value_repr = repr(value)
            else:
                value_repr = str(value)

            field_reprs.append(f"{field.name}={value_repr}")

        return f"DitCacheSearcherConfig({', '.join(field_reprs)})"


class DitCacheSearcher:

    def __init__(self, config: DitCacheSearcherConfig, pipeline, generate_videos: callable):
        self.config = config
        self.pipeline = pipeline
        self.generate_videos = generate_videos

    @staticmethod
    def cal_videos_mse(no_cache_video_paths, cache_video_paths):
        all_mse = []
        for no_cache_path, cache_path in zip(no_cache_video_paths, cache_video_paths):
            for _ in range(50):
                cap_no_cache = cv2.VideoCapture(no_cache_path)
                cap_cache = cv2.VideoCapture(cache_path)
                if not cap_no_cache.isOpened() or not cap_cache.isOpened():
                    time.sleep(1)
                else:
                    break
            if not cap_no_cache.isOpened() or not cap_cache.isOpened():
                raise FileNotFoundError("Cannot open video file")

            mse_sum = []
            # 逐帧读取视频
            while True:
                # 读取帧
                ret1, frame1 = cap_no_cache.read()
                ret2, frame2 = cap_cache.read()
                # 如果视频结束，则退出循环
                if not ret1 or not ret2:
                    break
                if frame1.shape != frame2.shape:
                    raise ValueError("Video shapes do not match")
                mse_sum.append(
                    np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2)
                )
            average_mse = np.mean(mse_sum)
            all_mse.append(average_mse)
            # 释放视频文件
            cap_no_cache.release()
            cap_cache.release()

        return np.mean(all_mse)

    def eval_config(self, block_start, num_blocks, step_start, baseline_paths):
        config = replace(
            self.config,
            cache_step_start=step_start,
            cache_block_start=block_start,
            cache_num_blocks=num_blocks,
            use_cache=True,
        )
        video_paths = self.generate_videos(config, self.pipeline)
        mean_mse = [None]
        if get_local_rank() == 0:
            mean_mse[0] = self.cal_videos_mse(baseline_paths, video_paths)
        if dist.is_initialized():
            dist.broadcast_object_list(mean_mse, src=0)
        return mean_mse[0]

    def search(self):
        # 主进度条
        main_progress = tqdm(total=2, desc="Config Search", position=0, leave=True)

        # 缓存开始timestep范围
        cache_start_step_list = [0, 5]

        # 1. 生成不用缓存加速的基线
        baseline_paths = self.generate_videos(self.config, self.pipeline)

        # 2. 根据期望加速比计算最小的cache len
        start = cache_start_step_list[-1]
        avail_step = self.config.num_sampling_steps - start
        update_steps = avail_step // self.config.cache_step_interval
        reuse_steps = avail_step - update_steps

        if avail_step <= 0 and reuse_steps <= 0:
            raise ValueError(
                f"Cache ratio not possible with cache start step list: {cache_start_step_list}. "
                f"Num sampling steps is too small!"
            )

        min_l = int(
            self.config.dit_block_num * (1 - (
                    self.config.num_sampling_steps / self.config.cache_ratio - start - update_steps) / reuse_steps)
        )

        logger_debug(f"min_l: {min_l}")

        if min_l <= 0 or min_l >= self.config.dit_block_num:
            raise ValueError(
                f"Cache ratio not possible with cache start step list: {cache_start_step_list}. "
                f"Cache ratio may be too large, num sampling steps may be too small, or cache step interval may be too "
                f"small."
            )

        # 3. 在最小cache情况下（生成质量最高），选择质量前三的block start
        ## 第一阶段进度条
        first_stage_num_runs = max(0, self.config.dit_block_num - min_l)
        first_stage_progress = tqdm(total=first_stage_num_runs, desc="First-Stage Search", position=1, leave=True)

        ## 第一阶段搜索正文
        self.config.use_cache = True
        mse_list, cache_list = [], []
        for start in range(1, self.config.dit_block_num):
            step = cache_start_step_list[-1]
            if (start + min_l) > self.config.dit_block_num:
                break
            mse_score = self.eval_config(start, min_l, step, baseline_paths)

            mse_list.append(mse_score)
            cache_list.append([start, self.config.cache_step_interval, min_l, step])

            first_stage_progress.update(1)

        main_progress.update(1)

        ## 取前三的block start作为第一阶段的结果
        sorted_indices = sorted(range(len(mse_list)), key=lambda i: mse_list[i])[:3]
        cache_list_3 = [cache_list[i] for i in sorted_indices]
        mse_list_3 = [mse_list[i] for i in sorted_indices]
        cache_start_list = [i[0] for i in cache_list_3]

        logger_debug(f"First Stage: mse_list: {mse_list}, cache_list: {cache_list}")
        logger_info(
            f"The top 3 cache config in the first stage are: \n{cache_list_3}\n" \
            + f"Their corresponding mse scores are {mse_list_3}"
        )

        # 4. 基于前三的block start，进行cache_num_blocks和cache_step_start的搜索
        ## 第二阶段进度条
        second_stage_num_runs = 0
        for start in cache_start_list:
            for cache_len in range(min_l, min_l + 8):
                if (start + cache_len) < self.config.dit_block_num:
                    for step in cache_start_step_list:
                        # 跳过第三步已经生成过的配置
                        if (cache_len == min_l) and (step == cache_start_step_list[-1]):
                            continue
                        second_stage_num_runs += 1
        second_stage_progress = tqdm(total=second_stage_num_runs, desc="Second-Stage Search", position=2, leave=True)

        ## 第二阶段搜索正文
        mse_list, cache_list = mse_list_3, cache_list_3
        for start in cache_start_list:
            for cache_len in range(min_l, min_l + 8):
                if (start + cache_len) < self.config.dit_block_num:
                    for step in cache_start_step_list:
                        # 跳过第三步已经生成过的配置
                        if (cache_len == min_l) and (step == cache_start_step_list[-1]):
                            continue
                        mse_score = self.eval_config(start, cache_len, step, baseline_paths)

                        mse_list.append(mse_score)
                        cache_list.append([start, self.config.cache_step_interval, cache_len, step])

                        second_stage_progress.update(1)

        main_progress.update(1)

        # 5. 取前三的结果作为搜索结果
        sorted_indices = sorted(range(len(mse_list)), key=lambda i: mse_list[i])[:3]
        cache_final_list = [cache_list[i] for i in sorted_indices]
        cache_final_score_list = [mse_list[i] for i in sorted_indices]

        logger_debug(f"Second Stage: mse_list: {mse_list}, cache_list: {cache_list}")
        logger_info(
            f"The top 3 cache config are: \n{cache_final_list}\n" \
            + f"Their corresponding mse scores are {cache_final_score_list}"
        )

        return cache_final_list
