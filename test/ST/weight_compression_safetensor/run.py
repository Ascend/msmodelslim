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
# pylint: disable=unspecified-encoding

import json
import os
from safetensors.torch import load_file
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor

# 准备待压缩权重文件和相关压缩配置，请根据实际情况进行修改
LOAD_PATH = f"{os.environ['PROJECT_PATH']}/resource/weight_compression_safetensor"
weight_path = os.path.join(LOAD_PATH, "quant_model_weight_w8a8s.safetensors")  # 待压缩权重文件的路径
json_path = os.path.join(LOAD_PATH, "quant_model_description_w8a8s.json")  # 待压缩权重文件的描述文件的路径

# 使用CompressConfig接口，配置压缩参数，并返回配置实例
compress_config = CompressConfig(
    do_pseudo_sparse=False,
    sparse_ratio=1,
    is_debug=True,
    record_detail_root=f"{os.environ['PROJECT_PATH']}/output/weight_compression_safetensor",
    multiprocess_num=8,
)

sparse_weight = load_file(weight_path)
with open(json_path, 'r') as f:
    quant_model_description = json.load(f)

# 使用Compressor接口，输入加载的压缩配置和待压缩权重文件
compressor = Compressor(compress_config, weight=sparse_weight, quant_model_description=quant_model_description)
compress_weight, compress_index, compress_info = compressor.run()

# 使用export_safetensors()接口，保存压缩后的结果文件
compressor.export_safetensors(
    f"{os.environ['PROJECT_PATH']}/output/weight_compression_safetensor", safetensors_name=None, json_name=None
)
