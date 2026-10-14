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
# pylint: disable=consider-using-from-import,super-with-arguments

import os
import torchvision  # noqa: F401
from msmodelslim.pytorch.quant.ptq_tools import QuantConfig, Calibrator

import torch
import torch.nn.functional as F  # noqa: F401
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class up_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Unet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(Unet, self).__init__()
        self.ndf = 64
        self.Maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = conv_block(ch_in=img_ch, ch_out=self.ndf)
        self.conv2 = conv_block(ch_in=self.ndf, ch_out=self.ndf * 2)
        self.conv3 = conv_block(ch_in=self.ndf * 2, ch_out=self.ndf * 2 * 2)
        self.conv4 = conv_block(ch_in=self.ndf * 2 * 2, ch_out=self.ndf * 2 * 2 * 2)
        self.conv5 = conv_block(ch_in=self.ndf * 2 * 2 * 2, ch_out=self.ndf * 2 * 2 * 2 * 2)

        self.up4 = up_block(ch_in=self.ndf * 2 * 2 * 2 * 2, ch_out=self.ndf * 2 * 2 * 2)
        self.up_conv4 = conv_block(ch_in=self.ndf * 2 * 2 * 2 * 2, ch_out=self.ndf * 2 * 2 * 2)

        self.up3 = up_block(ch_in=self.ndf * 2 * 2 * 2, ch_out=self.ndf * 2 * 2)
        self.up_conv3 = conv_block(ch_in=self.ndf * 2 * 2 * 2, ch_out=self.ndf * 2 * 2)

        self.up2 = up_block(ch_in=self.ndf * 2 * 2, ch_out=self.ndf * 2)
        self.up_conv2 = conv_block(ch_in=self.ndf * 2 * 2, ch_out=self.ndf * 2)

        self.up1 = up_block(ch_in=self.ndf * 2, ch_out=self.ndf)
        self.up_conv1 = conv_block(ch_in=self.ndf * 2, ch_out=self.ndf)

        self.conv1_1 = conv_block(ch_in=self.ndf, ch_out=output_ch)

    def forward(self, x):
        x1 = self.conv1(x)  # [none,3,256,256]

        x1_ = self.Maxpool(x1)  # [none,64,128,128]
        x2 = self.conv2(x1_)  # [none,128,128,128]

        x2_ = self.Maxpool(x2)  # [none,128,64,64]
        x3 = self.conv3(x2_)  # [none,256,64,64]

        x3_ = self.Maxpool(x3)  # [none,256,32,32]
        x4 = self.conv4(x3_)  # [none,512,32,32]

        x4_ = self.Maxpool(x4)  # [none,512,16,16]
        x5 = self.conv5(x4_)  # [none,1024,16,16]

        u4_ = self.up4(x5)  # [none,1024,32,32]
        u4 = self.up_conv4(torch.cat([x4, u4_], dim=1))  # [none,512,32,32]

        u3_ = self.up3(u4)  # [none,512,64,64]
        u3 = self.up_conv3(torch.cat([x3, u3_], dim=1))  # [none,256,64,64]

        u2_ = self.up2(u3)  # [none,256,128,128]
        u2 = self.up_conv2(torch.cat([x2, u2_], dim=1))  # [none,128,128,128]

        u1_ = self.up1(u2)  # [none,128,256,256]
        u1 = self.up_conv1(torch.cat([x1, u1_], dim=1))  # [none,64,256,256]

        out = self.conv1_1(u1)  # [none,1,256,256]
        out = torch.sigmoid(out)
        return out


if __name__ == '__main__':
    MODEL_ARCH = "unet"
    SAVE_PATH = f"{os.environ['PROJECT_PATH']}/output/ptq-tools/quant_param"
    INPUTS_NAMES = ["input.1"]

    os.makedirs(SAVE_PATH, exist_ok=True)

    model = Unet(3, output_ch=1)
    model.eval()

    disable_names = []
    input_shape = [1, 3, 224, 224]
    keep_acc = {'admm': [False, 1000], 'easy_quant': [False, 1000], 'round_opt': False}

    quant_config = QuantConfig(
        disable_names=disable_names,  # 手动回退的量化层名称，要求格式list[str]，如精度太差，推荐回退量化敏感层，如分类层、输入层、检测head层等
        amp_num=0,  # 混合精度量化回退层数，要求格式int；默认为0
        input_shape=input_shape,  # 模型输入的shape，用于data-free量化构造虚拟数据
        keep_acc=keep_acc,  # 精度保持策略
        sigma=25,  # 大于0使用sigma统计方法；传入0值使用min-max统计方法。
    )

    calibrator = Calibrator(model, quant_config)

    calibrator.run()  # 执行量化算法

    calibrator.export_param(SAVE_PATH)  # 用来导出量化参数
