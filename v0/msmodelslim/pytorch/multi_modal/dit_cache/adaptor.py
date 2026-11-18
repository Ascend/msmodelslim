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
from datetime import datetime, timezone

from typing import Optional
from dataclasses import dataclass
import logging

import imageio
import torch
import torch.nn as nn
import torch.distributed as dist

import ascend_utils.common.security.path as path_checker

from . import dit_cache_search_tool

START_HIDDEN_KEY = 'start_hidden'
DELTA_HIDDEN_KEY = 'delta_hidden'

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


def logger_info(*args, **kwargs):
    if get_local_rank() == 0:
        logger.info(*args, **kwargs)


def logger_debug(*args, **kwargs):
    if get_local_rank() == 0:
        logger.debug(*args, **kwargs)


# ----------------- Cache-related Data Structures -----------------
@dataclass
class DitCacheConfig:
    """Configuration for DiT caching mechanism.

    Attributes:
        cache_step_start: Starting timestep for caching
        cache_step_interval: Interval between timesteps for cache computation
        cache_block_start: Starting block index for caching region
        cache_num_blocks: Number of blocks in caching region
    """
    cache_step_start: int = None
    cache_step_interval: int = None  # Compute every n timesteps, reuse cache for others
    cache_block_start: int = None  # Starting block index (0 means from first block)
    cache_num_blocks: int = None  # Number of blocks to cache

    def __post_init__(self):
        if not isinstance(self.cache_step_start, int) or self.cache_step_start < 0:
            raise ValueError("cache_step_start must be a non-negative integer")

        if not isinstance(self.cache_step_interval, int) or self.cache_step_interval <= 0:
            raise ValueError("cache_step_interval must be a positive integer")

        if not isinstance(self.cache_block_start, int) or self.cache_block_start < 0:
            raise ValueError("cache_block_start must be a non-negative integer")

        if not isinstance(self.cache_num_blocks, int) or self.cache_num_blocks < 0:
            raise ValueError("cache_num_blocks must be a non-negative integer")

    def __iter__(self):
        """Enables dict(config) calls"""
        return iter(self.to_dict().items())

    def to_dict(self):
        return {
            "cache_step_start": self.cache_step_start,
            "cache_step_interval": self.cache_step_interval,
            "cache_block_start": self.cache_block_start,
            "cache_num_blocks": self.cache_num_blocks,
        }


@dataclass
class DitCacheSearchConfig:
    """Configuration for DiT cache search parameters.

    Attributes:
        cache_ratio: Speedup ratio to control cache application
        dit_block_num: Total number of DiT blocks
        num_sampling_steps: Total number of sampling steps
        num_hidden_states: Total number of hidden states to keep track of (like double/triple stream blocks)
    """
    cache_ratio: float = 1.3
    dit_block_num: Optional[int] = None
    num_sampling_steps: int = None
    cache_step_interval: int = 2
    num_hidden_states: int = 1

    def __post_init__(self):
        """Validates configuration parameters"""

        if not isinstance(self.cache_ratio, (int, float)):
            raise ValueError("cache_ratio must be a number")
        if not (1.0 < self.cache_ratio <= 2.0):
            raise ValueError("cache_ratio should be in the range of (1.0, 2.0]")

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
            raise ValueError("cache_step_interval for searching must be larger or equal to 2`")

        if not isinstance(self.num_hidden_states, int):
            raise ValueError("num_hidden_states must be an integer")
        elif self.num_hidden_states < 1:
            raise ValueError("num_hidden_states must be positive")


class DitCacheDummy:
    """Placeholder object returned during forward pass carrying DiT cache information"""

    def __init__(self, info="dit-cache: cached output placeholder, not used"):
        self.info = info

    def __repr__(self):
        return f"<DitCacheDummy: {self.info}>"


# ----------------- Adaptor Implementation -----------------
class DitCacheAdaptor:
    """DiT (Diffusion Transformer) Cache Adaptor for optimizing inference performance.

    This adaptor implements a caching mechanism for DiT blocks to reduce computation
    during the diffusion process. It requires setting the timestep index before each
    DiT block forward pass.

    Key Features:
        - Caches and reuses DiT block outputs for similar timesteps
        - Supports automatic cache configuration search
        - Requires explicit timestep tracking via set_timestep_idx()

    Example:
        ```python
        # Initialize adaptor
        config = DitCacheSearchConfig(cache_ratio=1.3, num_sampling_steps=100)
        adaptor = DitCacheAdaptor(pipeline, config)

        # Search for optimal cache configuration
        cache_config = adaptor.search(run_pipeline_and_save_videos)

        # ----------- Requires explicit set_timestep_idx() -------------
        # In diffusion loop, set timestep before each forward pass
        for t in timesteps:
            DitCacheAdaptor.set_timestep_idx(t)
            model_output = pipeline(...)
        # --------------------------------------------------------------
        ```

    Important:
        Always call set_timestep_idx() before running the DiT block forward pass.
    """

    _timestep_idx = None
    _init_timestep_idx = None
    _cur_block = 0

    def __init__(self, pipeline,
                 config: DitCacheSearchConfig = None,
                 dit_block_path: str = "transformer.transformer_blocks",
                 ):
        """Initialize the DiT Cache Adaptor.

        Args:
            pipeline: The pipeline object (expected to be OpenSoraPipelineV1x2 type)
                     that contains transformer blocks at the specified path.
            config (DitCacheSearchConfig): Configuration object for cache behavior
                     and search parameters.
            dit_block_path (str, optional): Access path to transformer blocks within
                     the pipeline. Defaults to "transformer.transformer_blocks".

        Raises:
            ValueError: If config is not DitCacheSearchConfig or if dit_block_path
                       is invalid.
        """
        if config is not None:
            if not isinstance(config, DitCacheSearchConfig):
                raise ValueError("config must be DitCacheSearchConfig")

        self.encoder_cache = None
        self.pipeline = pipeline
        self.search_config = config
        self.cache = {}  # Stores block outputs from previous timestep, keyed by block index

        # Validate path format during initialization
        if not isinstance(dit_block_path, str) or not dit_block_path:
            raise ValueError("dit_block_path must be a non-empty string")

        if not all(part.isidentifier() for part in dit_block_path.split('.')):
            raise ValueError(
                f"Invalid dit_block_path format: '{dit_block_path}'. "
                "Path should be dot-separated valid Python identifiers (e.g., 'transformer.transformer_blocks')"
            )

        self.pipeline = pipeline

        try:
            obj_dit_blocks = self.get_and_check_blocks(self.pipeline, dit_block_path)
        except (ValueError, TypeError, AttributeError) as e:
            raise ValueError(
                f"Failed to access transformer blocks at path '{dit_block_path}'. "
                f"Please ensure the pipeline has the correct structure and the path is valid. "
                f"Error: {str(e)}"
            ) from None

        if self.search_config is not None:
            if self.search_config.dit_block_num is not None:
                if self.search_config.dit_block_num != len(obj_dit_blocks):
                    raise ValueError('dit_block_num not equal to the number of blocks in the dit_block_path')
            self.search_config.dit_block_num = len(obj_dit_blocks)

        self.dit_cache_config = None
        self.enable_cache = True

        self._add_cache_to_dit_block(obj_dit_blocks)

        self._temp_cache_dir = None

    @staticmethod
    def get_and_check_blocks(obj, block_path: str) -> nn.ModuleList:
        """Retrieves and validates transformer blocks from the model.

        Args:
            obj: Model object
            block_path: Access path to blocks object, supports nested format like 'transformer.transformer_blocks'

        Returns:
            nn.ModuleList: Validated blocks object

        Raises:
            ValueError: When access path or blocks object doesn't exist
            TypeError: When retrieved object is of incorrect type
        """
        # Get nested attributes
        current_obj = obj
        path_parts = block_path.split('.')

        for i, part in enumerate(path_parts):
            current_path = '.'.join(path_parts[:i + 1])
            current_obj = getattr(current_obj, part, None)

            if current_obj is None:
                raise ValueError(
                    f"Failed to access '{current_path}': "
                    f"Attribute '{part}' not found in {type(current_obj).__name__}"
                )

        blocks = current_obj

        # Validate type
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError(
                f"Object at path '{block_path}' must be type nn.ModuleList, "
                f"but got {type(blocks).__name__}"
            )

        # Validate non-empty
        if len(blocks) == 0:
            raise ValueError(f"ModuleList at path '{block_path}' is empty")

        return blocks

    @staticmethod
    def replace_obj_class(obj, new_obj_class):
        """Replace object's class with a new class."""
        obj.__class__ = new_obj_class
        return obj

    @classmethod
    def set_timestep_idx(cls, t_idx):
        """Set the current timestep index for DiT cache mechanism.

        This method MUST be called before running the DiT block forward pass.
        The timestep index is used to determine whether
        to use cached results or compute new ones.

        Example:
            ```python
            # In your diffusion loop:
            for t in timesteps:
                DitCacheAdaptor.set_timestep_idx(t)
                # Now you can run the DiT forward pass
                model_output = pipeline(...)
            ```

        Args:
            t_idx: Current timestep index in the diffusion process

        Raises:
            ValueError: If not called before DiT block forward pass
        """
        if not isinstance(t_idx, int) or t_idx < 0:
            raise ValueError("t_idx must be non-negative integer")
        cls._timestep_idx = t_idx
        cls._init_timestep_idx = t_idx
        logger_debug('set timestep idx: %r', t_idx)

    def get_timestep_idx(self):
        """Get current timestep index.

        Raises:
            ValueError: If timestep_idx is not set via set_timestep_idx()
        """
        if self._timestep_idx is None:
            raise ValueError(
                "timestep_idx is not set, "
                "Please call DitCacheAdaptor.set_timestep_idx(t_idx) at the beginning of timestep.")
        to_return_timestep_idx = self._timestep_idx
        self._counter()
        return to_return_timestep_idx

    def update_cache_config(self, cache_block_start: int, cache_num_blocks: int,
                            cache_step_start: int, cache_step_interval: int):
        """Update cache configuration parameters."""
        self.dit_cache_config = DitCacheConfig(
            cache_block_start=cache_block_start,
            cache_num_blocks=cache_num_blocks,
            cache_step_start=cache_step_start,
            cache_step_interval=cache_step_interval,
        )

    @torch.no_grad()
    def search(self, run_pipeline_and_save_videos: callable, prompts_num: int = 1, num_videos: int = 1):
        """Execute search process to optimize model forward pass using caching mechanism.

        Args:
            run_pipeline_and_save_videos (callable): A closure function to run model and save generated videos.
                The function should accept one parameter:
                    - `pipeline` (OpenSoraPipeline): Current model pipeline.
                It should return a list of generated videos with length equal to prompts_num,
                where each video is an `ndarray` with shape (num_frames, h, w, c).
            prompts_num (int, optional): Number of videos to generate. Defaults to 1.

        Returns:
            DitCacheConfig: Object containing cache configuration:
                - `cache_step_start` (int): Starting timestep for caching
                - `cache_step_interval` (int): Compute every n timesteps, reuse cache for others
                - `cache_block_start` (int): Starting block index for cache region
                - `cache_num_blocks` (int): Number of blocks in cache region
        """
        if self.search_config is None:
            raise ValueError("search_config must be set before calling search()")
        if not callable(run_pipeline_and_save_videos):
            raise ValueError("run_pipeline_and_save_videos must be callable")
        if not isinstance(prompts_num, int) or prompts_num <= 0:
            raise ValueError("prompts_num must be a positive integer")

        def generate_videos(config_, pipeline_):
            # Log function entry
            logger_debug("Entering generate_videos function with config_: %r", config_)

            # Copy key values from config_ to local variables for fixed snapshot
            use_cache = getattr(config_, 'use_cache', True) and config_.cache_num_blocks != 0
            cache_dit_block_start = config_.cache_block_start
            cache_num_dit_blocks = config_.cache_num_blocks
            cache_step_start = config_.cache_step_start
            cache_step_interval = config_.cache_step_interval

            # Sync configuration to current object
            self.enable_cache = use_cache
            self.dit_cache_config = DitCacheConfig(
                cache_block_start=cache_dit_block_start,
                cache_num_blocks=cache_num_dit_blocks,
                cache_step_start=cache_step_start,
                cache_step_interval=cache_step_interval,
            )

            # Generate videos using external function
            videos = run_pipeline_and_save_videos(pipeline_)
            logger_debug("Video generation complete, number of videos: %d", len(videos))
            if get_local_rank() == 0 and (not isinstance(videos, list) or len(videos) != prompts_num * num_videos):
                raise ValueError(
                    "run_pipeline_and_save_videos must return a video list with length equal to prompts_num:"
                    " {}!={}".format(len(videos), prompts_num))

            # Define template function for video file paths using local variables
            def get_video_path_tpl() -> str:
                if not use_cache:
                    vid_name = "sample_{sample_idx:04d}_no_cache.mp4"
                else:
                    vid_name = ("sample_{sample_idx:04d}_{cache_dit_block_start}_{cache_step_interval}"
                                "_{cache_num_dit_blocks}_{cache_step_start}.mp4")
                vid_path = os.path.join(search_cache_path, vid_name)
                logger_debug("Generated video file path template: %r", vid_path)
                return vid_path

            video_paths = []
            # Save and verify generated videos
            if get_local_rank() == 0:
                for sample_idx, video in enumerate(videos):
                    video_path = get_video_path_tpl().format(
                        sample_idx=sample_idx,
                        cache_dit_block_start=cache_dit_block_start,
                        cache_step_interval=cache_step_interval,
                        cache_num_dit_blocks=cache_num_dit_blocks,
                        cache_step_start=cache_step_start
                    )
                    logger_debug("Saving video #%d to path: %r", sample_idx, video_path)
                    fps = getattr(self.pipeline, 'config', {}).get('fps', 24)
                    path_checker.get_valid_write_path(video_path)
                    imageio.mimwrite(
                        video_path,
                        video,
                        fps=fps,
                        quality=6,
                        codec='libx264',
                        output_params=['-threads', '20']
                    )
                    if os.path.exists(video_path):
                        logger_debug("Video file successfully generated: %r", video_path)
                    else:
                        logger.error("Video file not generated: %r", video_path)
                    video_paths.append(video_path)

            logger_debug("Exiting generate_videos function")
            return video_paths

        try:
            search_cache_path = self._setup_temp_cache_dir()

            # ----------------- Search Interface Integration -----------------
            config = dit_cache_search_tool.DitCacheSearcherConfig(
                dit_block_num=self.search_config.dit_block_num,
                prompts_num=prompts_num,
                num_sampling_steps=self.search_config.num_sampling_steps,
                cache_ratio=self.search_config.cache_ratio,
                search_cache_path=search_cache_path,
                cache_step_interval=self.search_config.cache_step_interval
            )

            logger_info("***** Start searching for dit cache with config %r", self.search_config)
            searcher = dit_cache_search_tool.DitCacheSearcher(
                config=config,
                pipeline=self.pipeline,
                generate_videos=generate_videos
            )

            # Start search
            [cache_block_start, cache_step_interval, cache_num_blocks, cache_step_start] = searcher.search()[0]
            if dist.is_initialized():
                dist.barrier()
        finally:
            # Clean up temporary cache directory
            self._cleanup_temp_cache_dir()

        searched_config = DitCacheConfig(
            cache_step_start=cache_step_start,
            cache_step_interval=cache_step_interval,
            cache_block_start=cache_block_start,
            cache_num_blocks=cache_num_blocks
        )

        self.dit_cache_config = searched_config
        self.enable_cache = True
        logger_info("***** Finish searching for dit cache with result: %r", searched_config)

        return searched_config

    def _counter(self):
        # 内部计数 + 1
        self._cur_block += 1
        if self._cur_block == self.search_config.dit_block_num:
            self._timestep_idx += 1  # 满足blocks_count时，timestep_idx + 1
            self._cur_block = 0  # 重置block计数
            if self._timestep_idx == self._init_timestep_idx + self.search_config.num_sampling_steps:
                self._timestep_idx = self._init_timestep_idx  # 满足到num_sampling_steps时，恢复到初始timestep_idx

    def _add_cache_to_dit_block(self, dit_blocks: nn.ModuleList):
        """Replace transformer block forward method with caching logic.

        Cache logic flow:
        - When t_idx < cache_step_start or cache disabled: call original forward
        - Base block (index equals cache_block_start): call forward and update cache
        - Blocks between base and reuse (if any): return DitCacheDummy placeholder
        - Reuse block (index equals cache_block_start+1): compute output using
          out(t, reuse) = out(t, base) + (cached_reuse_{t-1} - cached_base_{t-1})
          and update cache
        - Other blocks: call original forward
        """

        for block_idx, dit_block in enumerate(dit_blocks):
            _orig_forward = dit_block.forward

            def forward_with_cache(blk_obj, hidden_states, *args, orig_forward=_orig_forward, _block_idx=block_idx,
                                   **kwargs):
                use_cache = self.enable_cache
                # Direct forward when cache disabled
                if not use_cache:
                    return orig_forward(hidden_states, *args, **kwargs)

                sr = self.dit_cache_config
                if self.dit_cache_config is None:
                    raise ValueError("You must set the dit cache config to enable forward with cache")

                t_start = sr.cache_step_start
                t_interval = sr.cache_step_interval
                blk_start = sr.cache_block_start
                blk_end = sr.cache_block_start + sr.cache_num_blocks - 1
                t_idx = self.get_timestep_idx()

                if use_cache and self._timestep_idx is None:
                    raise ValueError(
                        "You must call DitCacheAdaptor.set_timestep_idx(t_idx) at the beginning of timestep.")

                # Direct forward when t_idx < cache_step_start
                if t_idx < t_start:
                    return orig_forward(hidden_states, *args, **kwargs)

                is_step_to_store_cache = (t_idx - t_start) % t_interval == 0

                # Between base block and reuse block
                if blk_start <= _block_idx < blk_end:
                    # Base block: call forward and update cache
                    if _block_idx == blk_start:
                        logger_debug(
                            "cache: t_idx=%r, block=%r: store cache for input hidden states",
                            t_idx, _block_idx)
                        if self.search_config.num_hidden_states == 1:
                            self.cache[START_HIDDEN_KEY] = hidden_states
                        else:
                            self.cache[START_HIDDEN_KEY] = (hidden_states,
                                                            *args[:self.search_config.num_hidden_states - 1])

                    if is_step_to_store_cache:
                        return orig_forward(hidden_states, *args, **kwargs)
                    else:
                        logger_debug("cache: t_idx=%r, block=%r: cache skipped block",
                                     t_idx, _block_idx)
                        return DitCacheDummy() if self.search_config.num_hidden_states == 1 else tuple(
                            DitCacheDummy() for _ in range(self.search_config.num_hidden_states))

                elif _block_idx == blk_end:
                    # Get the hidden states of block_start's input
                    if START_HIDDEN_KEY not in self.cache:
                        raise ValueError(
                            f"cache: t_idx={t_idx}, block={_block_idx}: cache for block_start input not found")
                    last_block_hidden_states = self.cache.pop(START_HIDDEN_KEY)

                    if is_step_to_store_cache:
                        hidden_states = orig_forward(hidden_states, *args, **kwargs)

                        # Calculate delta hidden states and save to cache
                        if self.search_config.num_hidden_states == 1:
                            delta_hidden = hidden_states - last_block_hidden_states
                        else:
                            delta_hidden = tuple(hidden_states[i] - last_block_hidden_states[i]
                                                 for i in range(self.search_config.num_hidden_states))
                        self.cache[DELTA_HIDDEN_KEY] = delta_hidden

                        logger_debug("cache: t_idx=%r, block=%r: calculate delta hidden states and save to cache",
                                     t_idx, _block_idx)
                        return hidden_states
                    else:
                        if DELTA_HIDDEN_KEY not in self.cache:
                            raise ValueError(
                                f"cache: t_idx={t_idx}, block={_block_idx}: cache for block_delta not found")

                        if self.search_config.num_hidden_states == 1:
                            hidden_states_reuse = self.cache[DELTA_HIDDEN_KEY] + last_block_hidden_states
                        else:
                            hidden_states_reuse = tuple(self.cache[DELTA_HIDDEN_KEY][i] + last_block_hidden_states[i]
                                                        for i in range(self.search_config.num_hidden_states))

                        logger_debug("cache: t_idx=%r, block=%r: reuse the cached delta hidden",
                                     t_idx, _block_idx)
                        return hidden_states_reuse

                # Other blocks: normal forward
                else:
                    return orig_forward(hidden_states, *args, **kwargs)

            # Use closure to fix current block_idx and orig_forward, then replace forward method
            dit_block.forward = forward_with_cache.__get__(dit_block)

    def _setup_temp_cache_dir(self):
        """Create temporary cache directory."""
        # 所有进程都初始化一个空列表用于广播
        temp_dir_list = [None]  # 使用列表包装，因为broadcast_object_list需要列表

        if get_local_rank() == 0:
            timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            temp_dir = './__cache_for_dit_adaptor_{}'.format(timestamp)
            path_checker.safe_delete_path_if_exists(temp_dir)  # 清理旧目录
            path_checker.get_write_directory(temp_dir)
            logger_debug(f"Created shared cache directory: {temp_dir}")
            temp_dir_list[0] = temp_dir

        if dist.is_initialized():
            # 所有进程都参与广播，使用相同大小的列表
            dist.broadcast_object_list(temp_dir_list, src=0)
        else:
            pass
        self._temp_cache_dir = temp_dir_list[0]
        return self._temp_cache_dir

    def _cleanup_temp_cache_dir(self):
        """Clean up temporary cache directory."""
        if get_local_rank() == 0 and self._temp_cache_dir and os.path.exists(self._temp_cache_dir):
            try:
                path_checker.safe_delete_path_if_exists(self._temp_cache_dir)
                logger_debug(f"Cleaned up temporary cache directory: {self._temp_cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary cache directory: %r", e)
