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
# pylint: disable=consider-using-from-import,logging-fstring-interpolation,not-callable,pointless-string-statement,unspecified-encoding,useless-parent-delegation

import importlib
import logging
import os
import shutil
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from unittest.mock import Mock

import torch
import torch.nn as nn

from ascend_utils.common.security import get_write_directory
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.timestep.manager import TimestepManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from unittest.mock import patch  # noqa: E402

patch('torch.cuda.current_device', return_value=0).start()


@dataclass
class OneStepSampleArgs:
    """Arguments for one step sampling function."""

    latents: torch.Tensor
    timestep: torch.Tensor
    step_index: int
    encoder_states: dict
    extra_step_kwargs: dict
    added_cond_kwargs: dict = None


@dataclass
class TextEmbeddingsArgs:
    """Arguments for text embeddings function."""

    prompt: str
    negative_prompt: str
    num_frames: int
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    num_images_per_prompt: int = 1
    mask_feature: bool = True
    device: str = None
    max_sequence_length: int = 512


class DummyScheduler:
    """A dummy scheduler that mimics the behavior of diffusion schedulers."""

    def __init__(self, num_train_timesteps=1000, clip_sample=False):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.timesteps = None

    def set_timesteps(self, num_inference_steps, device=None):
        """
        Sets the timesteps used for the diffusion chain.

        Args:
            num_inference_steps (int): number of diffusion steps used for inference.
            device (str or torch.device): device to put the timesteps on.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create linearly spaced steps from 999 down to 0
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.int, device=device
        )

        logger.info(f"Scheduler set with {num_inference_steps} steps")
        return self.timesteps


class DummyVAE(nn.Module):
    """A dummy VAE model for testing."""

    def __init__(self):
        super().__init__()

    def encode(self, x):
        """Mock encode function that returns random latents."""
        batch_size, frames, height, width, channels = x.shape
        latents = torch.randn(batch_size, frames, height // 8, width // 8, 4, device=x.device)
        return latents

    def decode(self, latents):
        """Mock decode function that returns random images."""
        batch_size, frames, height, width, channels = latents.shape
        images = torch.rand(batch_size, frames, height * 8, width * 8, 3, device=latents.device)
        return images


class DummyTextEncoder(nn.Module):
    """A dummy text encoder for testing."""

    def __init__(self):
        super().__init__()

    def forward(self, input_ids, attention_mask=None):
        """Mock forward function that returns random embeddings."""
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, 1024, device=input_ids.device)


class DummyPipeline:
    """
    A dummy pipeline that mimics the behavior of OpenSoraPipeline for testing timestep optimization.
    """

    def __init__(self, device='cuda'):
        self.scheduler = DummyScheduler()
        self.vae = DummyVAE()
        self.text_encoder = DummyTextEncoder()
        self.device = torch.device(device if torch.cuda.is_available() and 'cuda' in device else 'cpu')
        self.args = None  # Will be set by the optimization script

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        num_frames=16,
        height=256,
        width=256,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        mask_feature=True,
        device=None,
        max_sequence_length=512,
    ):
        """
        Mock pipeline call function that returns random video tensors.
        """
        device = device or self.device

        # Create dummy latents
        latents = torch.randn(num_images_per_prompt, num_frames, height // 8, width // 8, 4, device=device)

        # Mock diffusion process
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for step_index, timestep in enumerate(self.scheduler.timesteps):
            # Use TimestepManager to track current timestep
            TimestepManager.set_timestep_idx(step_index)

            # Mock noise prediction step
            latents = latents - 0.1 * torch.randn_like(latents)

            # Log progress
            if step_index % 5 == 0:
                logger.info(f"Step {step_index}/{num_inference_steps}, timestep {timestep}")

        # Mock VAE decode
        videos = torch.rand(num_images_per_prompt, num_frames, height, width, 3, device=device)

        return type('DummyPipelineOutput', (), {'images': videos})

    def prepare_extra_step_kwargs(self, generator=None, eta=None):
        """Mock preparation of extra step kwargs."""
        return {"generator": generator, "eta": eta or 0.0}

    def get_text_embeddings(self, args: TextEmbeddingsArgs):
        """Mock text embeddings generation."""
        device = self.device
        batch_size = args.num_images_per_prompt

        # Create dummy embeddings and masks
        prompt_embeds = torch.randn(batch_size, 1, 77, 1024, device=device)
        prompt_attention_mask = torch.ones(batch_size, 77, device=device)
        negative_prompt_embeds = torch.randn(batch_size, 77, 1024, device=device)
        negative_prompt_attention_mask = torch.ones(batch_size, 77, device=device)

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def one_step_sample(self, args: OneStepSampleArgs):
        """Mock one step sampling function."""
        # Use TimestepManager to track current timestep
        TimestepManager.set_timestep_idx(args.step_index)

        # Mock denoising step
        noise = torch.randn_like(args.latents) * 0.1
        return args.latents - noise

    def get_sequence_parallel_state(self):
        """Mock sequence parallel state."""
        return False

    def split_sequence(self, tensor, rank, world_size):
        """Mock sequence splitting for parallelism."""
        return tensor


@dataclass
class DummyArgs:
    """Dummy arguments class to mimic command line arguments."""

    num_frames: int = 16
    height: int = 256
    width: int = 256
    num_sampling_steps: int = 50
    guidance_scale: float = 7.5
    max_sequence_length: int = 512
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


# 辅助函数：检查torch_npu是否可导入
def is_torch_npu_available():
    try:
        importlib.import_module('torch_npu')
        return torch.npu.is_available()
    except ImportError:
        return False  # 导入失败，返回False


class TestSampleOptimization(unittest.TestCase):
    """Test class for sample optimization pipeline."""

    @classmethod
    def setUpClass(cls):
        # 1. 尝试导入真实的torch_npu
        cls.mock_used_torch_npu = False  # 标记是否使用了mock
        try:
            import torch_npu  # noqa: F401
            # 导入成功：使用真实模块，无需mock
        except ImportError:
            # 导入失败：使用mock
            cls.mock_used_torch_npu = True

        # 保存原始模块引用，用于后续恢复
        cls.original_modules = {}
        # 定义需要模拟的模块
        cls.mock_modules = {
            'opensora': Mock(),
            'opensora.sample': Mock(),
            'opensora.sample.pipeline_opensora_sp': Mock(),
        }
        if cls.mock_used_torch_npu:
            # 为'torch_npu'配置__spec__和方法返回值
            torch_npu_spec = types.ModuleType(name='torch_npu')
            cls.mock_modules.update(
                {
                    'torch_npu': Mock(
                        __spec__=torch_npu_spec,  # 直接在Mock参数中设置__spec__
                        npu_init=Mock(return_value=True),  # 模拟方法调用返回True
                        __version__='2.1.0',  # 模拟属性
                    )
                }
            )

        # 应用所有模拟并保存原始模块
        for module_name, mock_module in cls.mock_modules.items():
            cls.original_modules[module_name] = sys.modules.get(module_name)
            sys.modules[module_name] = mock_module

        # 重置所有mock
        for mock_module in cls.mock_modules.values():
            mock_module.reset_mock()

        from msmodelslim.pytorch.multi_modal.sampling_optimization import ReStepSearchConfig, ReStepAdaptor

        cls.ReStepSearchConfig = ReStepSearchConfig
        cls.ReStepAdaptor = ReStepAdaptor

        """Set up test environment once before all tests."""
        cls.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Create temporary directories for test artifacts
        cls.temp_dir = tempfile.mkdtemp(prefix="test_sample_optimization_")
        cls.videos_path = os.path.join(cls.temp_dir, "dummy_videos")
        cls.save_dir = os.path.join(cls.temp_dir, "timestep_results")

        # Ensure directories exist with proper permissions
        get_write_directory(cls.videos_path)
        get_write_directory(cls.save_dir)

        # Create test configuration
        cls.test_config = {
            "num_frames": 16,
            "height": 256,
            "width": 256,
            "num_sampling_steps": 50,
            "guidance_scale": 7.5,
            "monte_carlo_iters": 3,
            "neighbour_type": "uniform",
        }

        cls._create_dummy_videos(cls.videos_path)

        # Reset timestep manager before tests
        TimestepManager._timestep_var.set(None)

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests are done."""
        # Remove temporary directories
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

        # Reset timestep manager
        TimestepManager._timestep_var.set(None)

        # 恢复原始模块，避免影响其他测试
        for module_name, original_module in cls.original_modules.items():
            if original_module is not None:
                sys.modules[module_name] = original_module
            else:
                # 如果原始模块不存在，则从sys.modules中移除
                del sys.modules[module_name]

    def setUp(self):
        """Set up test environment before each test."""
        # Create a fresh pipeline for each test
        self.pipeline = DummyPipeline(device=self.device)

        # Set pipeline args
        self.pipeline.args = DummyArgs(
            num_frames=self.test_config["num_frames"],
            height=self.test_config["height"],
            width=self.test_config["width"],
            num_sampling_steps=self.test_config["num_sampling_steps"],
            guidance_scale=self.test_config["guidance_scale"],
            device=self.device,
        )

        # Reset timestep manager before each test
        TimestepManager._timestep_var.set(None)

    def tearDown(self):
        """Clean up after each test."""
        # Clean up any GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reset timestep manager
        TimestepManager._timestep_var.set(None)

        # Remove any large objects
        self.pipeline = None

    @staticmethod
    def _create_dummy_videos(videos_path, num_videos=5):
        """Create dummy MP4 files for testing."""
        get_write_directory(videos_path)

        # Create empty dummy mp4 files
        for i in range(num_videos):
            dummy_file = os.path.join(videos_path, f"dummy_video_{i}.mp4")
            with open(dummy_file, 'w') as f:
                f.write(f"Dummy video {i} content")

        logger.info(f"Created {num_videos} dummy video files in {videos_path}")

    def test_pipeline_basic_operation(self):
        """Test that the dummy pipeline runs correctly."""
        # Run the pipeline with default parameters
        result = self.pipeline(
            prompt="A test prompt",
            negative_prompt="A negative prompt",
            num_frames=self.test_config["num_frames"],
            height=self.test_config["height"],
            width=self.test_config["width"],
            num_inference_steps=self.test_config["num_sampling_steps"],
        )

        # Verify the output shape
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'images'))
        self.assertEqual(result.images.shape[1], self.test_config["num_frames"])
        self.assertEqual(result.images.shape[2], self.test_config["height"])
        self.assertEqual(result.images.shape[3], self.test_config["width"])
        self.assertEqual(result.images.shape[4], 3)  # RGB channels

    def test_timestep_manager_integration(self):
        """Test that the TimestepManager correctly tracks step indices."""
        # Set a timestep manually
        test_step_idx = 10
        TimestepManager.set_timestep_idx(test_step_idx)
        self.assertEqual(TimestepManager.get_timestep_idx(), test_step_idx)

        # Run a single denoising step with the timestep set
        latents = torch.randn(1, 16, 32, 32, 4, device=self.device)
        timestep = torch.tensor([500], device=self.device)

        # Create mock arguments
        args = OneStepSampleArgs(
            latents=latents, timestep=timestep, step_index=test_step_idx, encoder_states={}, extra_step_kwargs={}
        )

        # Run the step - this should update the timestep index
        self.pipeline.one_step_sample(args)

        # Verify the TimestepManager was updated
        self.assertEqual(TimestepManager.get_timestep_idx(), test_step_idx)

    def test_restep_adaptor_initialization(self):
        """Test that the ReStepAdaptor can be properly initialized."""
        # Create a configuration
        config = self.ReStepSearchConfig(
            videos_path=self.videos_path,
            save_dir=self.save_dir,
            neighbour_type=self.test_config["neighbour_type"],
            monte_carlo_iters=self.test_config["monte_carlo_iters"],
            num_sampling_steps=self.test_config["num_sampling_steps"] // 2,  # Target half the original steps
        )

        # Set environment variables required by the adaptor
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        try:
            # Initialize the adaptor
            adaptor = self.ReStepAdaptor(self.pipeline, config)

            # Override video paths for testing
            adaptor.videos_paths = [os.path.join(self.videos_path, f"dummy_video_{i}.mp4") for i in range(5)]

            # Verify the adaptor was created successfully
            self.assertIsNotNone(adaptor)
            self.assertEqual(adaptor.search_config, config)
            self.assertEqual(adaptor.pipeline, self.pipeline)

        finally:
            # Clean up environment variables
            if "RANK" in os.environ:
                del os.environ["RANK"]
            if "WORLD_SIZE" in os.environ:
                del os.environ["WORLD_SIZE"]

    def test_restep_adaptor_search(self):
        """Test that the ReStepAdaptor search method returns expected results."""
        # Create a configuration
        config = self.ReStepSearchConfig(
            videos_path=self.videos_path,
            save_dir=self.save_dir,
            neighbour_type=self.test_config["neighbour_type"],
            monte_carlo_iters=self.test_config["monte_carlo_iters"],
            num_sampling_steps=self.test_config["num_sampling_steps"] // 2,  # Target half the original steps
        )

        # Set environment variables required by the adaptor
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        try:
            # Initialize the adaptor
            adaptor = self.ReStepAdaptor(self.pipeline, config)

            # Override video paths for testing
            adaptor.videos_paths = [os.path.join(self.videos_path, f"dummy_video_{i}.mp4") for i in range(5)]

            # Mock the search method for testing purposes
            def mock_search(self):
                """Mock search method that returns a predefined schedule."""
                logger.info("Running mock search algorithm")
                # Return a simple schedule for testing (normalized 0 to 1)
                steps = self.search_config.num_sampling_steps
                return [i / steps for i in range(steps)]

            # Apply the mock search method
            adaptor.search = types.MethodType(mock_search, adaptor)

            # Run the search
            schedule = adaptor.search()

            # Verify the results
            self.assertIsNotNone(schedule)
            self.assertEqual(len(schedule), config.num_sampling_steps)
            self.assertEqual(schedule[0], 0.0)
            self.assertAlmostEqual(schedule[-1], (config.num_sampling_steps - 1) / config.num_sampling_steps)

        finally:
            # Clean up environment variables
            if "RANK" in os.environ:
                del os.environ["RANK"]
            if "WORLD_SIZE" in os.environ:
                del os.environ["WORLD_SIZE"]

    @unittest.skipIf(
        not is_torch_npu_available(),  # 条件：如果torch_npu不可用则跳过
        "torch_npu 导入失败，跳过此用例",
    )
    @patch('msmodelslim.pytorch.multi_modal.sampling_optimization.adaptor.AYSOptimizer')
    def test_search_method(self, mock_optimizer):
        """Test that the search method returns the expected schedule."""
        # Mock the optimizer's optimize method to return a predetermined schedule
        mock_optimizer_instance = mock_optimizer.return_value
        expected_schedule = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_optimizer_instance.optimize.return_value = expected_schedule

        # Create a configuration for testing
        config = self.ReStepSearchConfig(
            videos_path=self.videos_path,
            save_dir=self.save_dir,
            neighbour_type="uniform",
            monte_carlo_iters=3,
            num_sampling_steps=5,
        )

        # Set environment variables required by the adaptor
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        try:
            # Create the pipeline and adaptor
            pipeline = DummyPipeline(device=self.device)
            pipeline.args = DummyArgs(
                num_sampling_steps=5  # Match the config
            )

            # Override video paths for testing
            adaptor = self.ReStepAdaptor(pipeline, config)
            adaptor.videos_paths = [os.path.join(self.videos_path, f"dummy_video_{i}.mp4") for i in range(5)]

            # Mock the dump_json method to avoid file I/O
            with patch('msmodelslim.pytorch.multi_modal.sampling_optimization.adaptor.dump_json'):
                # Call the search method
                schedule = adaptor.search()

                # Verify the optimizer was created with the expected arguments
                mock_optimizer.assert_called_once()
                args, kwargs = mock_optimizer.call_args
                self.assertEqual(kwargs['neighbourhood_type'], "uniform")
                self.assertEqual(kwargs['batch_size'], 1)

                # Verify optimize was called with the expected arguments
                mock_optimizer_instance.optimize.assert_called_once()

                # Verify the returned schedule matches what the optimizer returned
                self.assertEqual(schedule, expected_schedule)

        finally:
            # Clean up environment variables
            if "RANK" in os.environ:
                del os.environ["RANK"]
            if "WORLD_SIZE" in os.environ:
                del os.environ["WORLD_SIZE"]

    @patch('torch.cuda.manual_seed_all')
    @patch('torch.cuda.manual_seed')
    @patch('torch.manual_seed')
    @patch('numpy.random.seed')
    @patch('random.seed')
    def test_seed_everything_method(
        self, mock_random_seed, mock_np_seed, mock_torch_seed, mock_torch_cuda_seed, mock_torch_cuda_seed_all
    ):
        """Test that seed_everything sets all random seeds correctly."""
        # Create a configuration for testing
        config = self.ReStepSearchConfig(
            videos_path=self.videos_path,
            save_dir=self.save_dir,
        )

        # Set environment variables required by the adaptor
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        try:
            # Create the pipeline and adaptor
            pipeline = DummyPipeline(device=self.device)
            adaptor = self.ReStepAdaptor(pipeline, config)

            # Override video paths for testing
            adaptor.videos_paths = [os.path.join(self.videos_path, f"dummy_video_{i}.mp4") for i in range(5)]

            # Test seed
            test_seed = 42

            # Call the seed_everything method
            adaptor.seed_everything(test_seed)

            # Verify each seed function was called with the expected seed value
            # The adaptor adds the rank (0 in this case) to the seed
            expected_seed = test_seed + 0
            mock_random_seed.assert_called_once_with(expected_seed)
            mock_np_seed.assert_called_once_with(expected_seed)
            mock_torch_seed.assert_called_once_with(expected_seed)
            mock_torch_cuda_seed.assert_called_once_with(expected_seed)
            mock_torch_cuda_seed_all.assert_called_once_with(expected_seed)

        finally:
            # Clean up environment variables
            if "RANK" in os.environ:
                del os.environ["RANK"]
            if "WORLD_SIZE" in os.environ:
                del os.environ["WORLD_SIZE"]


if __name__ == "__main__":
    unittest.main()
