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
# pylint: disable=consider-using-from-import,self-assigning-variable,super-with-arguments

import torch
import torch.nn as nn


def conv_bn_relu(in_channel, out_channel, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride), nn.BatchNorm2d(out_channel), nn.ReLU()
    )


class TestAscendQuantModel(nn.Module):
    def __init__(self):
        super(TestAscendQuantModel, self).__init__()
        self.first_conv = nn.Conv2d(1, 1, 3)
        self.left_conv = nn.Conv2d(1, 1, 3)
        self.right_conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        x = self.first_conv(x)
        x1 = self.left_conv(x)
        x2 = self.right_conv(x)
        y = torch.cat((x1, x2))
        return y


class TestNet(nn.Module):
    """
    TestNet
    """

    def __init__(self, class_num=10):
        super(TestNet, self).__init__()
        self.network = conv_bn_relu(3, 32, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, class_num)

    def forward(self, x):
        x = x
        x = self.network(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TestNet2(nn.Module):
    """
    TestNet
    """

    def __init__(self, class_num=10):
        super(TestNet2, self).__init__()
        self.backbone = conv_bn_relu(3, 32, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, class_num)

    def forward(self, x):
        x = x
        x = self.backbone(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TestNet3(nn.Module):
    """
    TestNet
    """

    def __init__(self, class_num=10):
        super().__init__()
        self.network = conv_bn_relu(3, 32, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, class_num)

    def forward(self, x):
        x2 = x
        x = self.network(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, x2


class TestOnnxQuantModel(nn.Module):
    def __init__(self, class_num=10):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [
                conv_bn_relu(3, 32, 3, 1),
                conv_bn_relu(32, 32, 3, 1),
                conv_bn_relu(32, 32, 3, 1),
                conv_bn_relu(32, 32, 3, 1),
            ]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(32, class_num)

    def forward(self, input_x):
        for conv in self.conv_list:
            input_x = conv(input_x)
        input_x = self.avg_pool(input_x)
        input_x = torch.flatten(input_x, 1)
        output = self.linear(input_x)
        return output


def get_model():
    return TestNet(class_num=10)


class LrdSampleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(16, 32),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )
        self.feature = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.inner = nn.Linear(64 * 5 * 5, 512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, 10),
        )

    def forward(self, inputs):
        shortcut = self.embedding(inputs)
        shortcut = shortcut.permute([0, 3, 1, 2])
        next_node = self.feature(shortcut)
        next_node = next_node + shortcut
        next_node = self.pool(next_node)
        next_node = torch.flatten(next_node, 1)
        next_node = self.inner(next_node)
        next_node = self.classifier(next_node)
        return next_node


class TorchTeacherModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher_fc = torch.nn.Linear(1, 1)

    def forward(self, inputs):
        output = self.teacher_fc(inputs)
        return output


class TorchStudentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.student_fc = torch.nn.Linear(1, 1)

    def forward(self, inputs):
        output = self.student_fc(inputs)
        return output


class GroupLinearTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.config = None
        self.dtype = torch.float16
        self.l1 = torch.nn.Linear(256, 256, bias=False)
        self.l2 = torch.nn.Linear(256, 256, bias=False)

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)

        return x


class TwoLinearTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.config = None
        self.dtype = torch.float16
        self.l1 = torch.nn.Linear(8, 8, bias=False)
        self.l2 = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)

        return x


class ThreeLinearTorchModel_for_Sparse(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.config = None
        self.dtype = torch.float16
        self.l1 = torch.nn.Linear(256, 256, bias=False)
        self.l2 = torch.nn.Linear(256, 256, bias=False)
        self.l3 = torch.nn.Linear(256, 256, bias=False)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim))

    def forward(self, x):
        variance = torch.mean(x * x, dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.g * x


class AttentionTorchModel(nn.Module):
    def __init__(self, embed_dim=32, num_heads=8):
        super().__init__()
        self.device = torch.device('cpu')
        self.config = None
        self.dtype = torch.float16
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Key, Query, Value 投影
        self.input_norm = RMSNorm(embed_dim)  # 使用RMSNorm替换LayerNorm
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o = nn.Linear(embed_dim, embed_dim)
        self.post_norm = RMSNorm(embed_dim)

        # Scaling factor
        self.scale = embed_dim**-0.5

    def forward(self, hidden_states, past_key_value=None, mask=None):
        # 投影
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = self.input_norm(hidden_states)
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 分割
        query = query.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = key.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = value.view(-1, self.num_heads, self.embed_dim // self.num_heads)

        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=0)
            value = torch.cat([past_key_value[1], value], dim=0)

        # 点积
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # 添加掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))

        # 归一化
        attn_weights = nn.functional.softmax(scores, dim=-1)

        # 加权和
        output = torch.matmul(attn_weights, value)

        # 合并
        output = output.view(-1, self.num_heads * self.embed_dim // self.num_heads)

        output = self.o(output)

        output = self.post_norm(output)

        past_key_value = (key, value)

        return output, past_key_value


class SophonRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        """
        SophonRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.to(self.weight.dtype)

        return self.weight * x


class SophonTorchAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale = self.embed_dim**-0.5

    def forward(self, hidden_states, past_key_value=None, mask=None, **kargs):
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 分割
        query = query.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = key.view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = value.view(-1, self.num_heads, self.embed_dim // self.num_heads)

        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=0)
            value = torch.cat([past_key_value[1], value], dim=0)

        past_key_value = (key, value)

        # 点积
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # 添加掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))

        # 归一化
        attn_weights = nn.functional.softmax(scores, dim=-1)

        # 加权和
        output = torch.matmul(attn_weights, value)

        # 合并
        output = output.view(-1, self.num_heads * self.embed_dim // self.num_heads)

        output = self.o_proj(output)

        return output, past_key_value


class SophonTorchMlp(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.gate_proj2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.down_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, hidden_states):
        return self.down_proj(
            (self.act2(self.gate_proj(hidden_states)) + self.act1(self.gate_proj2(hidden_states)))
            * self.up_proj(hidden_states)
        )


class SophonTorchDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.input_norm = SophonRMSNorm(self.embed_dim)  # 使用SophonRMSNorm替换LayerNorm
        self.attn = SophonTorchAttention(self.embed_dim, self.num_heads)
        self.mlp = SophonTorchMlp(self.embed_dim)
        self.post_norm = SophonRMSNorm(self.embed_dim)

    def forward(self, hidden_states, attention_mask, rotary_pos_emb_list, use_cache=False):
        residual = hidden_states

        hidden_states = self.input_norm(hidden_states)

        hidden_states, past_key_value = self.attn(hidden_states)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, past_key_value


class AttentionTorchSophonModel(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.device = torch.device('cpu')
        self.config = None
        self.dtype = torch.float16
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        layer_list = []
        layer_list.append(SophonTorchDecoder(self.embed_dim, self.num_heads))
        self.layers = nn.ModuleList(layer_list)
        self.norm = SophonRMSNorm(self.embed_dim)

    def forward(self, hidden_states, use_cache=False):
        for _, decoder_layer in enumerate(self.layers):
            attention_mask = 1
            rotary_pos_emb_list = 1
            hidden_states, past_key_value = decoder_layer(
                hidden_states, attention_mask, rotary_pos_emb_list, use_cache=use_cache
            )
        hidden_states = self.norm(hidden_states)

        return hidden_states, past_key_value


class ExpertFFN(nn.Module):
    def __init__(self, embed_dim=32):
        super(ExpertFFN, self).__init__()
        self.act_fn = RMSNorm(embed_dim)
        self.w1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MOEModel(nn.Module):
    """
    MOE-like model
    """

    def __init__(self, embed_dim=32):
        super(MOEModel, self).__init__()
        self.expert1 = ExpertFFN(embed_dim)
        self.expert2 = ExpertFFN(embed_dim)
        self.dtype = torch.float16
        self.device = torch.device('cpu')

    def forward(self, x):
        # 模拟MOE局部运行，不执行expert2
        output = self.expert1(x)
        return output
