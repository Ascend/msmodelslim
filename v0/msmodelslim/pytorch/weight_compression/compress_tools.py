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
from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm
from safetensors.torch import save_file

from ascend_utils.common.security import get_valid_read_path, get_valid_write_path, get_write_directory, \
    SafeWriteUmask, MAX_READ_FILE_SIZE_512G, safe_delete_path_if_exists, check_type, json_safe_dump
from msmodelslim import logger
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.llm_ptq_utils import QuantModelJsonDescription, QuantType
from .compress_config import CompressConfig
from .compress_utils import compress_weight_fun
from .compress_utils import pseudo_sparse
from .compress_utils import transform_nd2nz

SUPPORT_DTYPE = [np.int8, np.int64]


class Compressor:
    def __init__(self, config: CompressConfig, weight_path=None, weight=None, quant_model_description=None):
        if not isinstance(config, CompressConfig):
            raise ValueError("Invalid `config`: expected a CompressConfig instance, but received {}".format(config))

        self.config = config
        self.logger = logger
        self.weights = None
        self.quant_model_description = None
        self.load_weights(weight_path, weight, quant_model_description)
        self.compress_result_weight, self.compress_result_index, self.compress_result_info = {}, {}, {}
        self.sparse_type = None
        # 如果 do_pseudo_sparse 为 None，则默认使用 W8A8S 量化类型，适配 w8a8s 的 numpy 格式压缩
        if self.quant_model_description is None:
            self.sparse_type = QuantType.W8A8S

    @classmethod
    def export(cls, arr, path, dtype=np.int8):
        if dtype not in SUPPORT_DTYPE:
            raise ValueError("dtype must be numpy.int8, numpy.int64!")
        check_type(arr, dict, param_name="arr")
        check_type(path, str, param_name="path")
        get_write_directory(path, write_mode=0o750)
        logger.info(f"The output files will be saved in: {path}")

        for key in arr.keys():
            save_path = os.path.join(path, key + '.dat')
            safe_delete_path_if_exists(save_path, logger_level="debug")
            get_valid_write_path(save_path)
            with SafeWriteUmask(0o377):
                arr[key].astype(dtype).tofile(os.path.join(path, key + '.dat'))

        logger.info("Files saved successfully!")

    def export_safetensors(self, path, safetensors_name=None, json_name=None):
        if not self.quant_model_description:
            raise ValueError()
        compress_weight = {}
        compress_model_description = QuantModelJsonDescription(QuantType.W8A8SC)

        # 适配 w8a8s 的 numpy 格式压缩
        if self.quant_model_description is not None:
            self.sparse_type = self.quant_model_description.get("model_quant_type", None)

        if self.sparse_type == QuantType.W16A16S:
            compress_model_description = QuantModelJsonDescription(QuantType.W16A16SC)

        if not isinstance(safetensors_name, str) or not safetensors_name.endswith('.safetensors'):
            self.logger.warning("Invalid `safetensors_name` provided. Reverting `safetensors_name` to default.")
            safetensors_name = \
                f"quant_model_weight_{compress_model_description.model_quant_type.lower()}.safetensors"
        if not isinstance(json_name, str) or not json_name.endswith('.json'):
            self.logger.warning("Invalid `json_name` provided. Reverting `json_name` to default.")
            if self.quant_model_description.get(QuantModelJsonDescription.version_type_name, '0.0.0') == '0.0.0':
                json_name = f"quant_model_description_{compress_model_description.model_quant_type.lower()}.json"
            else:
                json_name = f'quant_model_description.json'

        compress_model_description.change_version_name(
            self.quant_model_description.get(QuantModelJsonDescription.version_type_name))
        compress_model_description.change_kvcache_type(
            self.quant_model_description.get(QuantModelJsonDescription.kv_quant_type_name))
        compress_model_description.change_fa_quant_type(
            self.quant_model_description.get(QuantModelJsonDescription.fa_quant_type_name))
        compress_model_description.change_group_size(
            self.quant_model_description.get(QuantModelJsonDescription.group_size_name, 0))
        compress_model_description.change_reduce_quant_type(
            self.quant_model_description.get(QuantModelJsonDescription.reduce_quant_type_name))

        for key, value in self.quant_model_description.items():
            if key in [
                QuantModelJsonDescription.model_quant_type_name,
                QuantModelJsonDescription.version_type_name,
                QuantModelJsonDescription.group_size_name,
                QuantModelJsonDescription.kv_quant_type_name,
                QuantModelJsonDescription.kv_cache_type_name,
                QuantModelJsonDescription.fa_quant_type_name,
                QuantModelJsonDescription.reduce_quant_type_name,
                QuantModelJsonDescription.metadata_name,
            ]:
                continue
            if value == QuantType.W8A8S and key.endswith('.weight'):
                key_short = '.'.join(key.split('.')[:-1])
                key_index = key_short + '.index'
                key_info = key_short + '.info'

                compress_model_description.change_weight_type(key, QuantType.W8A8SC)
                compress_model_description.change_weight_type(key_index, QuantType.W8A8SC)
                compress_model_description.change_weight_type(key_info, QuantType.W8A8SC)

                compress_weight[key] = torch.from_numpy(self.compress_result_weight.get(key))
                compress_weight[key_index] = torch.from_numpy(self.compress_result_index.get(key).astype(np.int8))
                compress_weight[key_info] = torch.from_numpy(self.compress_result_info.get(key).astype(np.int64))
            
            elif value == QuantType.W16A16S and key.endswith('.weight'):
                key_short = '.'.join(key.split('.')[:-1])
                key_index = key_short + '.index'
                key_info = key_short + '.info'

                compress_model_description.change_weight_type(key, QuantType.W16A16SC)
                compress_model_description.change_weight_type(key_index, QuantType.W16A16SC)
                compress_model_description.change_weight_type(key_info, QuantType.W16A16SC)

                compress_weight[key] = torch.from_numpy(self.compress_result_weight.get(key))
                compress_weight[key_index] = torch.from_numpy(self.compress_result_index.get(key).astype(np.int8))
                compress_weight[key_info] = torch.from_numpy(self.compress_result_info.get(key).astype(np.int64))
            
            else:
                compress_model_description.change_weight_type(key, QuantType(value))
                compress_weight[key] = self.weights.get(key)

        get_write_directory(path, write_mode=0o750)
        output_path = os.path.join(path, safetensors_name)
        self.logger.info(f"The compressed quant weight safetensors file will be saved in: {output_path}")
        output_path = get_valid_write_path(output_path, extensions='.safetensors')
        self.quant_model_description = compress_model_description
        QuantModelJsonDescription.check_description_match(
            quant_model_json_description=compress_model_description.quant_model_description,
            quant_model_safetensor=compress_weight)
        with SafeWriteUmask(umask=0o377):
            save_file(compress_weight, output_path)
        json_path = os.path.join(path, json_name)
        compress_model_description.save(json_path)
        self.logger.info("Files saved successfully!")

    def run(self, weight_transpose: bool = False):
        self.config.record_detail_root = get_write_directory(self.config.record_detail_root, write_mode=0o750)
        check_type(weight_transpose, bool, param_name="weight_transpose")

        # 适配 w8a8s 的 numpy 格式压缩
        if self.quant_model_description is not None:
            self.sparse_type = self.quant_model_description.get("model_quant_type", None)

        ori_total_length = 0
        now_total_length = 0

        p = Pool(self.config.multiprocess_num)
        result_list = []
        self.logger.info(f"The weight compressor will run with {self.config.multiprocess_num} processes.")

        if self.quant_model_description:
            keys_list = []
            for key in self.weights.keys():
                quant_type = self.quant_model_description.get(key)
                if key.endswith('.weight') and 'norm' not in key and \
                        quant_type in [QuantType.W8A8S, QuantType.W16A16S]:
                    keys_list.append(key)
        else:
            keys_list = sorted(self.weights.keys())

        if len(keys_list) == 0:
            raise ValueError("No sparse weight found in input weight. Please check the input weight.")

        # 初始化进度条
        pbar = tqdm(total=len(keys_list))
        pbar.set_description("Compression Process")

        # 进度条更新的内嵌函数
        def update(*args):
            # Pool.apply_async方法中，callback参数回接收任何返回值作为其唯一的参数，所以这里需要接收*args，不过并不会使用
            pbar.update()

        for key_index, key in enumerate(keys_list):
            each_weight = self.weights[key]
            ori_shape = each_weight.shape
            ori_length = np.prod(ori_shape)

            if isinstance(each_weight, torch.Tensor):
                each_weight = each_weight.cpu().data.numpy()

            if self.config.do_pseudo_sparse:
                each_weight = pseudo_sparse(each_weight, self.config.sparse_ratio)

            if each_weight.ndim != 2:
                raise ValueError(f"The number of dimensions (ndim) of input weights must be 2, "
                                 f"but received {each_weight.ndim}.")

            if weight_transpose:
                each_weight = each_weight.T

            n, k = each_weight.shape

            if self.quant_model_description is not None and self.quant_model_description.get(key) == QuantType.W16A16S:
                each_weight = each_weight.view(np.int8).reshape(n, k * 2)
                k = k * 2
                ori_length = ori_length * 2
                self.logger.debug(f"W16A16S weight shape: {each_weight.shape}, k: {k}, ori_length: {ori_length}")

            k0 = 32
            n0 = 16
            each_weight = transform_nd2nz(each_weight, block_size=[n0, k0])  # 矩阵在int8的时候，分型大小就是16 * 32

            save_key = key
            if key in self.config.compress_disable_layers:
                self.compress_result_weight[save_key] = each_weight
                self.compress_result_index[save_key] = np.empty((0, 0))  # 置空保证后面调用export时调用ndarray.astype时不报错
                continue

            self.compress_result_info[key] = np.array([8, 8, k, n, 1], dtype=np.int64)

            self.logger.debug("Compressing weight_part {}".format(key_index))
            res = p.apply_async(
                compress_weight_fun,
                args=(each_weight, self.config.record_detail_root, self.sparse_type),
                callback=update
            )
            result_list.append((save_key, ori_length, res))

        p.close()
        p.join()

        for key_index, (save_key, ori_length, res) in enumerate(result_list):

            compress_info, compress_output, compress_index = res.get()
            if compress_info is None:
                raise Exception("Error occurred when compressing weights")

            self.compress_result_weight[save_key] = compress_output
            self.compress_result_index[save_key] = compress_index

            tmp_num = len(compress_info)
            if tmp_num < 3:
                raise ValueError("`compress_info` should contains at least 3 elements, but only had {}".format(tmp_num))
            tiling_n, tiling_k, compress_length = compress_info[:3]  # 压缩算子返回的是 n k

            if len(self.compress_result_info[save_key]) >= 2:
                self.compress_result_info[save_key][0] = tiling_k  # 解压缩算子需要 k n
                self.compress_result_info[save_key][1] = tiling_n

            if len(self.compress_result_info[save_key]) >= 2:
                self.compress_result_info[save_key][0] = tiling_k
                self.compress_result_info[save_key][1] = tiling_n

            ori_total_length += ori_length
            now_total_length += compress_length

            if ori_length == 0:
                raise ValueError("Calculating {} length but got zero. Please check the input weight.".format(save_key))

            cur_compress_ratio = round(float(compress_length) / ori_length, 4)

            if self.config.is_debug:
                self.logger.info("[%d/%d]  %-80s%-20s" % (key_index + 1, len(keys_list), save_key, cur_compress_ratio))

        if ori_total_length == 0:
            raise ValueError("Calculating sparse weight size but got zero. Please check the input weight.")

        if self.config.is_debug:
            self.logger.info("The final compression ratio is {:.4f}\t(from {} to {})".format(
                float(now_total_length) / ori_total_length, ori_total_length, now_total_length))
        return self.compress_result_weight, self.compress_result_index, self.compress_result_info

    def load_from_file(self, weight_path):
        self.logger.info("Only load data from trusted sources. Avoid loading data from unknown or untrusted sources.")
        self.logger.info("Are you sure you want to load data from the following path: {}".format(weight_path))
        message = input("Please enter 'yes' or 'no'\n")
        if message != "yes":
            raise Exception("Loading is prohibited because the data may be from untrusted sources.")
        weight_path = get_valid_read_path(weight_path, extensions='.npy', size_max=MAX_READ_FILE_SIZE_512G)
        try:
            self.weights = np.load(weight_path, allow_pickle=False).item()
        except FileNotFoundError as e:
            self.logger.warning(f"Failed to load weights file: {e}. The file may not exist.")
        except Exception as e:
            self.logger.warning("Failed to load weights file with allow_pickle=False. This may be because " \
                                "the file contains Python objects. Will attempt to load with allow_pickle=True " \
                                "(Note: This could pose security risks as pickle format may contain malicious code)")
            self.weights = np.load(weight_path, allow_pickle=True).item()
        self.logger.info("Data loaded")

    def load_weights(self, weight_path, weight, quant_model_description):
        if weight_path:
            self.load_from_file(weight_path)
        elif weight and quant_model_description:
            check_type(weight, dict)
            check_type(quant_model_description, dict)
            self.weights = weight
            self.quant_model_description = quant_model_description
            QuantModelJsonDescription.check_description_match(quant_model_json_description=quant_model_description,
                                                              quant_model_safetensor=self.weights)
        else:
            raise ValueError("Weight Loading failed! Please provide valid quant weight input")
