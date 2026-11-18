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


import torch


def inter_category_distance(top_cate, bottom_cate):
    min_outlier = top_cate.min()
    max_outlier = bottom_cate.max()
    distance = (min_outlier-max_outlier).abs()
    return distance, min_outlier, max_outlier


def intra_category_variance(top_cate, bottom_cate):
    if top_cate.shape[0] == 1:
        top_variance = 0
    else:
        top_variance = torch.var(top_cate)
    if bottom_cate.shape[0] == 1:
        bottom_variance = 0
    else:
        bottom_variance = torch.var(bottom_cate)
    return top_variance, bottom_variance


def otsu(data, pos=True):
    "计算激活函数中异常值阈值"
    data = data.sort(descending=True)[0]
    metric = -1000
    out_num = 0
    outlier = None
    outlier_queue = []
    for idx, _ in enumerate(data):
        up_cate = data[:idx + 1]
        bottom_cate = data[idx + 1:]
        if bottom_cate.shape[0] == 0:
            continue 
        if pos is True:
            _inter, _, _ = inter_category_distance(up_cate, bottom_cate)

            _, bottom_intra = intra_category_variance(up_cate, bottom_cate)

            cur_metric = _inter-bottom_intra
            
            if cur_metric > metric:
                metric = cur_metric
                out_num = idx + 1
                outlier = data[:idx + 1]
                outlier_threshold = bottom_cate[0]
        else:
            outlier_queue.append(data[idx])
            inter, _, _ = inter_category_distance(up_cate, bottom_cate)
            top_intra, _ = intra_category_variance(up_cate, bottom_cate)
            cur_metric = inter - top_intra 

            if cur_metric > metric:
                metric = cur_metric
                out_num = data.shape[0] - idx - 1
                outlier = data[idx + 1:]
                outlier_threshold = up_cate[-1]
    return out_num, outlier, outlier_threshold


def calcu_outlier_mask(per_channel_min, per_channel_max, alpha=1.5, smooth_rate=0.7):
    total_channel = per_channel_max.size()[0]
    per_channel_max = per_channel_max
    per_channel_min = per_channel_min

    outlier_mask = torch.ones(total_channel).to(per_channel_min.device)
    outlier_mask_pos = torch.ones(total_channel).to(per_channel_min.device)
    outlier_mask_neg = torch.ones(total_channel).to(per_channel_min.device)

    per_channel_max = per_channel_max
    per_channel_min = per_channel_min

    # corse detect ## 
    q1_neg = torch.quantile(per_channel_min.float(), 0.20)
    q3_neg = torch.quantile(per_channel_min.float(), 0.80)
    iqr_neg = q3_neg - q1_neg
    outlier_neg = q1_neg - alpha * iqr_neg
    
    q1_pos = torch.quantile(per_channel_max.float(), 0.20)
    q3_pos = torch.quantile(per_channel_max.float(), 0.80)
    iqr_pos = q3_pos - q1_pos
    outlier_pos = q3_pos + alpha * iqr_pos

    # fine detect
    if per_channel_max[per_channel_max > outlier_pos].numel() > 1:
        _, _, outlier_pos = otsu(per_channel_max[per_channel_max > outlier_pos])
    if per_channel_min[per_channel_min < outlier_neg].numel() > 1:
        _, _, outlier_neg = otsu(per_channel_min[per_channel_min < outlier_neg], pos=False)

    outlier_mask_pos[per_channel_max > outlier_pos] = 0
    outlier_mask_neg[per_channel_min < outlier_neg] = 0

    max_value = torch.max(per_channel_max[outlier_mask_pos == 1])
    min_value = torch.min(per_channel_min[outlier_mask_neg == 1])

    outlier_mask_pos[outlier_mask_pos == 0] = \
    (per_channel_max.to(torch.float32)[outlier_mask_pos == 0] / max_value)**smooth_rate
    outlier_mask_neg[outlier_mask_neg == 0] = \
    (per_channel_min.to(torch.float32)[outlier_mask_neg == 0] / min_value)**smooth_rate

    if min_value != 0:
        outlier_mask = torch.max(outlier_mask_pos, outlier_mask_neg)
    else:
        outlier_mask = outlier_mask_pos
    
    outlier_mask[outlier_mask == 0] = 1

    return outlier_mask
