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
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------
"""

from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F

from msmodelslim.processor.analysis.methods_base import AnalysisTargetMatcher
from msmodelslim.utils.exception import UnexpectedError
from msmodelslim.utils.logging import get_logger
from ..base import UnaryAnalysisMethod
from .interface import RaCompressAnalysisInterface

logger = get_logger()

DUMMY_INPUT_LENGTH = 2500
REPET_TIMES = 4

# 默认比例（可通过 YAML 配置覆盖）
_DEFAULT_INDUCTION_HEAD_RATIO = 0.14
_DEFAULT_ECHO_HEAD_RATIO = 0.01

# 默认名称模式（当 adapter 未实现 RaCompressAnalysisInterface 时使用）
_DEFAULT_Q_NAME_PATTERN = "q_proj"
_DEFAULT_K_NAME_PATTERN = "k_proj"
_DEFAULT_QKV_NAME_PATTERN = "qkv_proj"


class RaCompressAnalysisMethod(UnaryAnalysisMethod, AnalysisTargetMatcher):
    """RA Compress analysis method for attention head importance.

    Hooks Q/K/QKV projection Linear layers to capture Q and K
    outputs, then manually reconstructs the attention softmax matrix
    (Q @ K^T / sqrt(d)) to compute prefix matching and copying matching
    scores for each attention head.

    The scores identify induction heads (prefix matching) and echo heads
    (copying matching), which are critical for long-context repeat
    detection in RA compression.

    After all layers are processed, call :meth:`get_compress_heads` to
    select top heads by ratio and produce the head dict for saving.

    Q/K/QKV 名称模式通过 ``adapter`` 的 :class:`RaCompressAnalysisInterface`
    获取，未实现时回退到 ``q_proj`` / ``k_proj`` / ``qkv_proj``。
    """

    def __init__(self, adapter: Optional[object] = None):
        self.adapter = adapter

        # 从 adapter 获取名称模式，或使用默认值
        if adapter is not None and isinstance(adapter, RaCompressAnalysisInterface):
            patterns = adapter.get_ra_compress_proj_patterns()
            self._q_name_pattern = patterns.get("q", _DEFAULT_Q_NAME_PATTERN)
            self._k_name_pattern = patterns.get("k", _DEFAULT_K_NAME_PATTERN)
            self._qkv_name_pattern = patterns.get("qkv", _DEFAULT_QKV_NAME_PATTERN)
        else:
            self._q_name_pattern = _DEFAULT_Q_NAME_PATTERN
            self._k_name_pattern = _DEFAULT_K_NAME_PATTERN
            self._qkv_name_pattern = _DEFAULT_QKV_NAME_PATTERN

        self._num_attention_heads: int = 0
        self._num_key_value_heads: int = 0
        self._hidden_size: int = 0
        self._head_dim: int = 0
        self._config_extracted: bool = False

        # YAML 可配置的超参
        self._induction_head_ratio: float = _DEFAULT_INDUCTION_HEAD_RATIO
        self._echo_head_ratio: float = _DEFAULT_ECHO_HEAD_RATIO

        # 存储 Q 和 K 的 Linear 输出
        self._q_outputs: Dict[str, torch.Tensor] = {}
        self._k_outputs: Dict[str, torch.Tensor] = {}

        # 存储每层每个 head 的 prefix / copying 分数
        # key = 层索引(int, 从0开始), value = list[float] (每个 head 一个分数)
        self._prefix_scores: Dict[int, List[float]] = {}
        self._copying_scores: Dict[int, List[float]] = {}

        # 层名 -> 层索引 的映射
        self._layer_name_to_idx: Dict[str, int] = {}
        # 层索引 -> 层名 的映射（反向，用于 get_compress_heads 输出 name 作为 key）
        self._layer_idx_to_name: Dict[int, str] = {}
        self._next_layer_idx: int = 0

    @property
    def name(self) -> str:
        return "ra_compress"

    def _matches(self, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear)

    def get_target_layers(self, model: nn.Module, prefix: str = "") -> List[str]:
        """Return only Q, K, or QKV Linear layers as target layers."""
        target_layers = []
        for name, module in model.named_modules(prefix=prefix):
            if not self._matches(module):
                continue
            if self._is_target_layer(name):
                target_layers.append(name)
                if not self._config_extracted:
                    self._extract_attention_config(model, name)
        return target_layers

    def _is_target_layer(self, name: str) -> bool:
        return self._q_name_pattern in name or self._k_name_pattern in name or self._qkv_name_pattern in name

    def _extract_attention_config(self, model: nn.Module, layer_name: str) -> None:
        """从模型 config 提取注意力配置。

        LayerWiseRunner 传入的 model 是单个 decoder layer，其 self_attn 子模块
        持有 config 引用（Qwen2Attention.config）；完整模型则直接有 model.config。
        直接从 config 读取，不需要复杂的路径导航。
        """
        # 获取 config 对象：优先 attention 子模块的 config，其次 model 自身
        cfg = None
        if hasattr(model, 'self_attn') and hasattr(model.self_attn, 'config'):
            cfg = model.self_attn.config
        elif hasattr(model, 'config'):
            cfg = model.config

        if cfg is not None:
            self._num_attention_heads = int(
                getattr(cfg, 'num_attention_heads', 0) or getattr(cfg, 'num_heads', 0) or getattr(cfg, 'n_head', 0)
            )
            self._num_key_value_heads = int(
                getattr(cfg, 'num_key_value_heads', 0)
                or getattr(cfg, 'multi_query_group_num', 0)
                or getattr(cfg, 'num_kv_heads', 0)
                or self._num_attention_heads
            )
            self._hidden_size = int(getattr(cfg, 'hidden_size', 0) or getattr(cfg, 'embed_dim', 0))
            head_dim_cfg = getattr(cfg, 'head_dim', None)
            if head_dim_cfg:
                self._head_dim = int(head_dim_cfg)
            elif self._num_attention_heads > 0 and self._hidden_size > 0:
                self._head_dim = self._hidden_size // self._num_attention_heads

        if self._num_attention_heads == 0:
            logger.warning("RA compress: can not extract attention config from %s", layer_name)
        else:
            logger.info(
                "RA compress config: num_attention_heads=%d, num_key_value_heads=%d, hidden_size=%d, head_dim=%d",
                self._num_attention_heads,
                self._num_key_value_heads,
                self._hidden_size,
                self._head_dim,
            )
        self._config_extracted = True

    def get_hook(self) -> Callable:
        """Hook 注册在 q_proj / k_proj / qkv_proj 的 nn.Linear 上。

        捕获 Linear 的输出（即 Q 或 K 的投影结果），保留在 device 上
        （不搬到 CPU），供后续在 NPU/GPU 上直接重建 attention 矩阵。
        """

        def linear_output_hook(
            module: nn.Module,
            input_tensor: Any,
            output_tensor: Any,
            layer_name: str,
            stats_dict: Dict[str, Any],
        ):
            if isinstance(output_tensor, tuple):
                output_tensor = output_tensor[0]

            # 保留在原 device（NPU/GPU）上，不 .cpu()
            output_detached = output_tensor.detach()

            # 存入 stats_dict（processor 靠此判断是否有数据并触发 compute_score）
            if layer_name not in stats_dict:
                stats_dict[layer_name] = {'outputs': [], 'layer_name': layer_name}
            stats_dict[layer_name]['outputs'].append(output_detached)

            # 存入 method 自己的字典
            if self._qkv_name_pattern in layer_name:
                self._q_outputs[layer_name] = output_detached
                self._k_outputs[layer_name] = output_detached
            elif self._q_name_pattern in layer_name:
                self._q_outputs[layer_name] = output_detached
            elif self._k_name_pattern in layer_name:
                self._k_outputs[layer_name] = output_detached

        return linear_output_hook

    def compute_score(self, layer_data: Dict[str, Any]) -> float:
        """用 Q 和 K 的 Linear 输出重建 attention softmax 矩阵，计算分数。

        对 Q/QKV 层：取 Q 输出和对应 K 输出，手动计算
        softmax(Q @ K^T / sqrt(head_dim))，再算 prefix + copying 分数。
        同时记录每个 head 的单独分数，用于后续 head 筛选。
        对 K 层：返回 0（分数在 Q 层计算）。
        """
        layer_name = layer_data.get('layer_name', '')
        outputs = layer_data.get('outputs', [])

        if not outputs:
            return 0.0

        # K 层不计算分数
        if self._k_name_pattern in layer_name and self._q_name_pattern not in layer_name:
            return 0.0

        if self._q_name_pattern not in layer_name and self._qkv_name_pattern not in layer_name:
            return 0.0

        # 取 Q 输出
        q_output = self._q_outputs.get(layer_name)
        if q_output is None:
            return 0.0

        # 分配层索引
        if layer_name not in self._layer_name_to_idx:
            self._layer_name_to_idx[layer_name] = self._next_layer_idx
            self._layer_idx_to_name[self._next_layer_idx] = layer_name
            self._next_layer_idx += 1
        layer_idx = self._layer_name_to_idx[layer_name]

        # 取对应 K 输出并计算
        if self._qkv_name_pattern in layer_name:
            prefix_per_head, copying_per_head = self._compute_qkv_scores(layer_name, q_output)
        else:
            k_name = layer_name.replace(self._q_name_pattern, self._k_name_pattern)
            k_output = self._k_outputs.get(k_name)
            if k_output is None:
                logger.warning(
                    "RA compress: no K output for Q layer %s (expected K at %s)",
                    layer_name,
                    k_name,
                )
                return 0.0
            prefix_per_head, copying_per_head = self._compute_qk_scores(layer_name, q_output, k_output)

        # 存储每头分数
        self._prefix_scores[layer_idx] = prefix_per_head
        self._copying_scores[layer_idx] = copying_per_head

        # 返回所有头的平均 combined 分数
        combined_per_head = [p + c for p, c in zip(prefix_per_head, copying_per_head)]
        avg_combined = sum(combined_per_head) / len(combined_per_head) if combined_per_head else 0.0

        logger.debug(
            "RA compress: layer %s (idx=%d) prefix=%s copying=%s avg_combined=%.6f",
            layer_name,
            layer_idx,
            prefix_per_head,
            copying_per_head,
            avg_combined,
        )
        return float(avg_combined)

    def _compute_qkv_scores(
        self,
        layer_name: str,
        qkv_output: torch.Tensor,
    ) -> tuple:
        """从 qkv_proj 输出中切分 Q 和 K，计算每头分数。"""
        if self._num_attention_heads == 0 or self._head_dim == 0:
            return [], []

        q_dim = self._num_attention_heads * self._head_dim
        k_dim = self._num_key_value_heads * self._head_dim

        total_dim = qkv_output.shape[-1]
        if total_dim < q_dim + k_dim:
            logger.warning(
                "RA compress: qkv output dim %d < q_dim(%d) + k_dim(%d) for %s",
                total_dim,
                q_dim,
                k_dim,
                layer_name,
            )
            return [], []

        q_output = qkv_output[..., :q_dim]
        k_output = qkv_output[..., q_dim : q_dim + k_dim]

        return self._compute_qk_scores(layer_name, q_output, k_output)

    def _compute_qk_scores(
        self,
        layer_name: str,
        q_output: torch.Tensor,
        k_output: torch.Tensor,
    ) -> tuple:
        """用 Q 和 K 的输出手动重建 attention softmax 矩阵并计算每头分数。

        逐 head 在 device 上完成 Q@K^T / causal mask / softmax，
        配合向量化算分，避免 Python 双重循环。
        """
        if self._num_attention_heads == 0 or self._head_dim == 0:
            return [], []

        try:
            q_flat = self._flatten_to_2d(q_output)
            k_flat = self._flatten_to_2d(k_output)

            total_tokens = q_flat.shape[0]
            required_tokens = DUMMY_INPUT_LENGTH * REPET_TIMES
            if total_tokens < required_tokens:
                raise UnexpectedError(
                    f"RA compress requires at least {required_tokens} tokens "
                    f"(DUMMY_INPUT_LENGTH={DUMMY_INPUT_LENGTH} * REPET_TIMES={REPET_TIMES}), "
                    f"but got {total_tokens}. Please use calib_dummy.jsonl or ensure the "
                    f"calibration data has sufficient token length."
                )

            device = q_flat.device
            num_heads = self._num_attention_heads
            num_kv_heads = self._num_key_value_heads if self._num_key_value_heads > 0 else num_heads
            head_dim = self._head_dim
            scale = float(head_dim) ** 0.5
            repeats = num_heads // num_kv_heads if num_kv_heads < num_heads else 1

            # [total_tokens, num_heads, head_dim]
            q_reshaped = q_flat.reshape(total_tokens, num_heads, head_dim)
            # [total_tokens, num_kv_heads, head_dim]
            k_reshaped = k_flat.reshape(total_tokens, num_kv_heads, head_dim)

            # 预生成 causal mask: [total_tokens, total_tokens]，逐 head 使用
            causal_mask = torch.triu(
                torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device),
                diagonal=1,
            )

            prefix_per_head: List[float] = []
            copying_per_head: List[float] = []

            for h in range(num_heads):
                kv_idx = h // repeats
                q_h = q_reshaped[:, h, :]  # [seq, head_dim]
                k_h = k_reshaped[:, kv_idx, :]  # [seq, head_dim]

                # [seq, seq] — 在 device 上完成
                attn = torch.matmul(q_h, k_h.transpose(0, 1)) / scale
                attn = attn.masked_fill(causal_mask, float('-inf'))
                attn = F.softmax(attn, dim=-1)

                # 向量化算分
                prefix_per_head.append(self._prefix_score_for_matrix(attn))
                copying_per_head.append(self._copying_score_for_matrix(attn))

                del attn

            return prefix_per_head, copying_per_head

        except Exception as e:
            logger.warning("RA compress: error computing scores for %s: %s", layer_name, e)
            return [], []

    @staticmethod
    def _flatten_to_2d(tensor: torch.Tensor) -> torch.Tensor:
        """将 [batch, seq, dim] 或 [batch*seq, dim] 统一展平为 [total_tokens, dim]。"""
        if tensor.dim() == 2:
            return tensor
        if tensor.dim() == 3:
            batch, seq, dim = tensor.shape
            return tensor.reshape(batch * seq, dim)
        if tensor.dim() == 4:
            return tensor.reshape(-1, tensor.shape[-1])
        return tensor.reshape(-1, tensor.shape[-1])

    @staticmethod
    def _prefix_score_for_matrix(attn: torch.Tensor) -> float:
        """向量化计算单个 head 的 prefix matching 分数。

        attn: [seq_len, seq_len]，在 device 上。
        按 (段 k, 偏移 d) 分块 gather，完全避免 Python 双重循环。
        """
        if attn.dim() != 2:
            return 0.0
        seq_len = attn.shape[0]
        device = attn.device
        total_sum = 0.0
        num_rows = 0

        # 段数（基于 seq_len 和 DUMMY_INPUT_LENGTH 动态计算）
        num_segments = (seq_len + DUMMY_INPUT_LENGTH - 1) // DUMMY_INPUT_LENGTH

        for k in range(1, num_segments):  # 跳过第 0 段 (k=0)
            start = k * DUMMY_INPUT_LENGTH
            if start >= seq_len:
                break
            end = min((k + 1) * DUMMY_INPUT_LENGTH, seq_len)
            # 行索引
            i_vals = torch.arange(start, end, device=device)
            # d 从 -k 到 -1
            for d in range(-k, 0):
                col_vals = i_vals + d * DUMMY_INPUT_LENGTH + 1  # prefix: +1
                # 有效 mask
                valid = (col_vals >= 0) & (col_vals < seq_len)
                if not valid.any():
                    continue
                total_sum += attn[i_vals[valid], col_vals[valid]].sum().item()
                num_rows += valid.sum().item()

        return total_sum / num_rows if num_rows > 0 else 0.0

    @staticmethod
    def _copying_score_for_matrix(attn: torch.Tensor) -> float:
        """向量化计算单个 head 的 copying matching 分数。

        attn: [seq_len, seq_len]，在 device 上。
        """
        if attn.dim() != 2:
            return 0.0
        seq_len = attn.shape[0]
        device = attn.device
        total_sum = 0.0
        num_rows = 0

        num_segments = (seq_len + DUMMY_INPUT_LENGTH - 1) // DUMMY_INPUT_LENGTH

        for k in range(1, num_segments):
            start = k * DUMMY_INPUT_LENGTH
            if start >= seq_len:
                break
            end = min((k + 1) * DUMMY_INPUT_LENGTH, seq_len)
            i_vals = torch.arange(start, end, device=device)
            for d in range(-k, 0):
                col_vals = i_vals + d * DUMMY_INPUT_LENGTH  # copying: 不加 1
                valid = (col_vals >= 0) & (col_vals < seq_len)
                if not valid.any():
                    continue
                total_sum += attn[i_vals[valid], col_vals[valid]].sum().item()
                num_rows += valid.sum().item()

        return total_sum / num_rows if num_rows > 0 else 0.0

    def enrich_layer_scores(self, layer_scores: List[Dict[str, Any]]) -> None:
        """将 head 选择信息合并进 layer_scores 条目。

        遍历 get_compress_heads() 的结果，将 induction_heads / echo_heads
        按层名写入对应的 layer_scores 条目。
        """
        head_dict = self.get_compress_heads()
        prefix_map = head_dict.get('prefix_matching', {})
        copying_map = head_dict.get('copying', {})

        name_to_induction = {}
        name_to_echo = {}
        for layer_idx, heads in prefix_map.items():
            name = self._layer_idx_to_name.get(int(layer_idx), str(layer_idx))
            name_to_induction[name] = heads
        for layer_idx, heads in copying_map.items():
            name = self._layer_idx_to_name.get(int(layer_idx), str(layer_idx))
            name_to_echo[name] = heads

        for entry in layer_scores:
            name = entry['name']
            entry['induction_heads'] = name_to_induction.get(name, [])
            entry['echo_heads'] = name_to_echo.get(name, [])

    # ========== Head 选择逻辑（与 ra_rope_tools.py 对齐）==========

    def get_compress_heads(self) -> Dict[str, Dict[int, List[int]]]:
        """选择 top heads 并返回 head_dict。

        - induction head: prefix matching 分数前 14%
        - echo head: copying matching 分数前 1%

        key 为层索引（int），value 为需要保留的 KV 头索引列表。

        Returns:
            {
                'prefix_matching': {layer_idx: [kv_head_idx, ...]},
                'copying': {layer_idx: [kv_head_idx, ...]},
            }
        """
        # GQA: 每个 kv group 内取 max，分组后索引即为 KV 头索引
        num_kv_per_group = (
            int(self._num_attention_heads // self._num_key_value_heads) if self._num_key_value_heads > 0 else 1
        )

        prefix_grouped = self._max_every_group(self._prefix_scores, num_kv_per_group)
        copying_grouped = self._max_every_group(self._copying_scores, num_kv_per_group)

        selected_prefix = self._select_top_heads(prefix_grouped, self._induction_head_ratio)
        selected_copying = self._select_top_heads(copying_grouped, self._echo_head_ratio)

        # key 直接使用 layer_idx（int），value 为 KV 头索引列表
        head_dict = {
            'prefix_matching': self._remove_empty_list_keys(selected_prefix),
            'copying': self._remove_empty_list_keys(selected_copying),
        }
        return head_dict

    @staticmethod
    def _max_every_group(data: Dict[int, List[float]], n: int) -> Dict[int, List[float]]:
        """每个 n 个 head 一组，取组内 max（GQA 分组）。"""
        if n <= 1:
            return data
        result = {}
        for key, values in data.items():
            max_values = [max(values[i : i + n]) for i in range(0, len(values), n)]
            result[key] = max_values
        return result

    @staticmethod
    def _select_top_heads(data: Dict[int, List[float]], ratio: float) -> Dict[int, List[int]]:
        """选择分数前 ratio 比例的 head，返回每层的 head 索引列表。"""
        all_values = [value for key in data for value in data[key]]
        if not all_values:
            return {}

        sorted_values = sorted(all_values, reverse=True)
        percent_index = round(len(sorted_values) * ratio)
        percent_values = sorted_values[:percent_index]

        result = {}
        for key in data:
            indices = [i for i, value in enumerate(data[key]) if value in percent_values]
            result[key] = indices
        return result

    @staticmethod
    def _remove_empty_list_keys(dictionary: Dict) -> Dict:
        return {k: v for k, v in dictionary.items() if v != []}
