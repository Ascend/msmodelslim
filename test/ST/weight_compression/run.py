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

import os
import numpy as np
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor
from msmodelslim import logger as msmodelslim_logger


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o750)
    return path


def main(root):
    weight_path = os.path.join(root, "quant_weight.npy")
    save_path = f"{os.environ['PROJECT_PATH']}/output/weight_compression"
    index_root = make_dir(os.path.join(save_path, 'index'))
    weight_root = make_dir(os.path.join(save_path, 'weight'))
    info_root = make_dir(os.path.join(save_path, 'info'))

    config = CompressConfig(
        do_pseudo_sparse=False, sparse_ratio=1, is_debug=True, record_detail_root=save_path, multiprocess_num=2
    )
    compressor = Compressor(config, weight_path)
    compress_weight, compress_index, compress_info = compressor.run()

    compressor.export(compress_weight, weight_root)
    compressor.export(compress_index, index_root)
    compressor.export(compress_info, info_root, dtype=np.int64)


src_root = f"{os.environ['PROJECT_PATH']}/resource/weight_compression"
main(src_root)
msmodelslim_logger.info("Weight Compression success!")
