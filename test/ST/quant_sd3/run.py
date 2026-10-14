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
# pylint: disable=dangerous-default-value,pointless-string-statement,redefined-outer-name,unspecified-encoding,useless-import-alias

"""
导入相关依赖
"""
import os  # noqa: E402
import numpy as np  # noqa: E402, F401
import json  # noqa: E402, F401
from torchvision.transforms import ToTensor  # noqa: E402, F401
import torch  # noqa: E402
from tqdm import tqdm as tqdm  # noqa: E402
from diffusers import StableDiffusion3Pipeline  # noqa: E402
from msmodelslim.pytorch.quant.ptq_tools.quant_modules import TensorQuantizer  # noqa: E402, F401

from msmodelslim.pytorch.quant.ptq_tools import Calibrator, QuantConfig  # noqa: E402
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier  # noqa: E402


torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

"""
导入相关模型
"""


def inference(save_path="imgs", categories=[]):
    os.makedirs(save_path, exist_ok=True)

    torch.manual_seed(42)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        f"{os.environ['PROJECT_PATH']}/resource/multi_modal/sd3_project/stable-diffusion-3-medium-diffusers/",
        torch_dtype=torch.float16,
    )
    pipe.to("npu")

    pipe.set_progress_bar_config(disable=True)

    model = pipe.transformer
    prompt_list = []
    # dataset
    with open(f"{os.environ['PROJECT_PATH']}/resource/multi_modal/sd3_project/PartiPrompts.tsv") as f:
        if categories == []:
            prompt_list = [sample.split("\t")[0] for sample in f][1:]
        else:
            prompt_list = [sample.split("\t")[0] for sample in f if sample.split("\t")[1] in categories]
    count = 0

    calib_dataset = torch.load(
        f"{os.environ['PROJECT_PATH']}/resource/multi_modal/sd3_project/sd3_calib_data_v3.pth", map_location="npu"
    )

    for data in tqdm(calib_dataset):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    data[key] = value.to(torch.float16)

    """
    对于linear算子中的激活值如果有表示范围过大，或者"尖刺"的异常值过多，
    需要使用anti outlier功能，使用方法如下
    """
    smooth_config = AntiOutlierConfig(
        anti_method='m4',
        dev_type="npu",
        dev_id=0,
    )
    anti_outlier = AntiOutlier(model, calib_dataset[:1], smooth_config, norm_class_name="layernorm")
    anti_outlier.process()

    # quantization
    q_config = QuantConfig(
        w_bit=8,
        a_bit=8,
        w_signed=True,
        a_signed=True,
        w_sym=True,
        a_sym=False,
        act_quant=True,
        act_method=1,
        quant_mode=1,
        disable_names=None,
        amp_num=0,
        keep_acc=None,
        sigma=25,
        device="npu",
    )
    calibrator = Calibrator(model, q_config, calib_dataset[:1])
    calibrator.run()

    calibrator.export_quant_safetensor(f"{os.environ['PROJECT_PATH']}/output/ptq-tools/quant_sd3")

    prompt_list = [
        "Portrait of a tiger wearing a train conductor's hat and holding a skateboard that has a yin-yang symbol on it"
    ]
    for prompt in tqdm(prompt_list):
        prompts = [prompt]
        neg_prompts = [""]
        images = pipe(
            prompt=prompts,
            negative_prompt=neg_prompts,
            num_inference_steps=28,
            height=1024,
            width=1024,
            guidance_scale=7.0,
        ).images
        for i, img in enumerate(images):
            img.save(os.path.join(save_path, str(count) + "_" + str(i) + ".jpg"))
        count += 1


if __name__ == '__main__':
    path_to_save = f"{os.environ['PROJECT_PATH']}/output/ptq-tools/quant_sd3/samples/"
    categories = []
    inference(path_to_save, categories)
