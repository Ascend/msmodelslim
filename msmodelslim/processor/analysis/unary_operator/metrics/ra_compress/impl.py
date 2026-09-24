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

import contextlib
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from msmodelslim.processor.analysis.methods_base import AnalysisTargetMatcher
from msmodelslim.utils.exception import SchemaValidateError, UnexpectedError
from msmodelslim.utils.logging import get_logger
from ..base import UnaryAnalysisMethod
from .calib_input import DUMMY_INPUT_LENGTH, RA_COMPRESS_METRIC, REPET_TIMES
from .interface import RaCompressAnalysisInterface

logger = get_logger()

# 默认比例（可通过 YAML 配置覆盖）
_DEFAULT_INDUCTION_HEAD_RATIO = 0.14
_DEFAULT_ECHO_HEAD_RATIO = 0.01

# 默认名称模式（当 adapter 未实现 RaCompressAnalysisInterface 时使用）
_DEFAULT_Q_NAME_PATTERN = "q_proj"
_DEFAULT_K_NAME_PATTERN = "k_proj"
_DEFAULT_QKV_NAME_PATTERN = "qkv_proj"

# ra_compress 支持的 metric_params 白名单
RA_COMPRESS_METRIC_PARAMS = ("induction_head_ratio", "echo_head_ratio")


def validate_metric_params(params: Dict[str, Any]) -> None:
    """校验 ra_compress 的专属超参（``metric_params``），非法时抛 ``SchemaValidateError``。

    比例类参数取值区间为 [0, 1]；未知参数直接报错，避免配置写错后被静默忽略。
    """
    unknown = sorted(set(params) - set(RA_COMPRESS_METRIC_PARAMS))
    if unknown:
        raise SchemaValidateError(
            f"Unsupported metric_params for metrics={RA_COMPRESS_METRIC!r}: {unknown}",
            action=f"Please use one of {list(RA_COMPRESS_METRIC_PARAMS)}.",
        )
    for key in RA_COMPRESS_METRIC_PARAMS:
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise SchemaValidateError(
                f"metric_params.{key} must be a number in [0, 1], but got {value!r}",
                action=f"Please set metric_params.{key} to a number in [0, 1].",
            )


def _pick_attr(obj: Any, names: Tuple[str, ...], default: Any = 0) -> Any:
    """按顺序返回第一个非空的属性值（兼容不同模型对同一配置的命名差异）。"""
    for name in names:
        value = getattr(obj, name, None)
        if value:
            return value
    return default


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
    获取（空串视为该投影层不存在），未实现该接口时回退到
    ``q_proj`` / ``k_proj`` / ``qkv_proj``。
    """

    def __init__(
        self,
        adapter: Optional[object] = None,
        induction_head_ratio: Optional[float] = None,
        echo_head_ratio: Optional[float] = None,
    ):
        self.adapter = adapter
        self._interface: Optional[RaCompressAnalysisInterface] = (
            adapter if isinstance(adapter, RaCompressAnalysisInterface) else None
        )

        # 名称模式是本接口的必需能力：实现了接口就取其返回值（空串视为该层不存在），
        # 未实现接口时用默认模式
        patterns = self._interface.get_proj_names() if self._interface is not None else {}
        self._q_name_pattern = patterns.get("q", _DEFAULT_Q_NAME_PATTERN)
        self._k_name_pattern = patterns.get("k", _DEFAULT_K_NAME_PATTERN)
        self._qkv_name_pattern = patterns.get("qkv", _DEFAULT_QKV_NAME_PATTERN)
        if not (self._q_name_pattern or self._k_name_pattern or self._qkv_name_pattern):
            logger.warning("RA compress: empty Q/K/QKV projection names, no layer will be analyzed")

        self._num_attention_heads: int = 0
        self._num_key_value_heads: int = 0
        self._hidden_size: int = 0
        self._head_dim: int = 0
        self._config_extracted: bool = False

        # YAML 可配置的超参（未指定时使用算法默认值）
        self._induction_head_ratio: float = (
            float(induction_head_ratio) if induction_head_ratio is not None else _DEFAULT_INDUCTION_HEAD_RATIO
        )
        self._echo_head_ratio: float = (
            float(echo_head_ratio) if echo_head_ratio is not None else _DEFAULT_ECHO_HEAD_RATIO
        )

        # 存储 Q 和 K 的 Linear 输出
        self._q_outputs: Dict[str, torch.Tensor] = {}
        self._k_outputs: Dict[str, torch.Tensor] = {}

        # 存储每层每个 head 的 prefix / copying 分数
        # key = 层索引(int, 从0开始), value = list[float] (每个 head 一个分数)
        self._prefix_scores: Dict[int, List[float]] = {}
        self._copying_scores: Dict[int, List[float]] = {}

        # 层名 -> self_attn 模块引用（用于提取 RoPE）
        self._layer_attn_modules: Dict[str, nn.Module] = {}
        # 块名 -> 该块的 forward kwargs（如 model.layers.0 -> kwargs），供 RoPE 解析按层取用
        self._rope_forward_kwargs_by_block: Dict[str, Dict[str, Any]] = {}
        # 最近一次收到的 forward kwargs（块名未知时的兜底）
        self._rope_latest_forward_kwargs: Optional[Dict[str, Any]] = None
        # RoPE 告警去重（按原因），避免逐层刷屏
        self._rope_warning_keys: Set[str] = set()

        # 层名 -> 层索引 的映射
        self._layer_name_to_idx: Dict[str, int] = {}
        # 层索引 -> 层名 的映射（反向，用于 get_compress_heads 输出 name 作为 key）
        self._layer_idx_to_name: Dict[int, str] = {}
        self._next_layer_idx: int = 0

    def set_forward_kwargs(self, kwargs: Dict[str, Any], layer_name: Optional[str] = None) -> None:
        """接收当前块（decoder layer）的 forward kwargs，供该块内的层解析 RoPE。

        位置编码由模型给出，方式随 transformers 版本变化：新版把 rotary embedding
        上移到顶层模型，算好后通过 ``position_embeddings=(cos, sin)`` 随 forward
        kwargs 下发；旧版则由 attention 模块内部持有 ``rotary_emb``。这里只做逐块缓存，
        具体解析交给内置兜底（见 :meth:`_generic_rope_cos_sin`）。

        ``layer_name`` 为该 kwargs 对应的块名（如 ``model.layers.0``）；分析分块进行，
        而算分可能推迟到 ``post_run``，故按块缓存以免张冠李戴。
        """
        if not kwargs:
            return
        self._rope_latest_forward_kwargs = kwargs
        if layer_name:
            self._rope_forward_kwargs_by_block[layer_name] = kwargs

    def _rope_kwargs_for(self, layer_name: str) -> Optional[Dict[str, Any]]:
        """取该层所属块的 forward kwargs；块名不可用时退回最近一次收到的 kwargs。"""
        for block_name, kwargs in self._rope_forward_kwargs_by_block.items():
            if layer_name == block_name or layer_name.startswith(f"{block_name}."):
                return kwargs
        return self._rope_latest_forward_kwargs

    @property
    def name(self) -> str:
        return "ra_compress"

    def _matches(self, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear)

    def get_target_layers(self, model: nn.Module, prefix: str = "") -> List[str]:
        """Return only Q, K, or QKV Linear layers as target layers."""
        target_layers = []
        for name, module in model.named_modules(prefix=prefix):
            if not self._matches(module) or not self._is_target_layer(name):
                continue
            target_layers.append(name)
            if not self._config_extracted:
                self._extract_attention_config(model, name)
            # 存储 self_attn 模块引用（用于提取 RoPE）
            attn_module = self._find_self_attn(model, name, prefix)
            if attn_module is not None:
                self._layer_attn_modules[name] = attn_module
        return target_layers

    def _find_self_attn(self, model: nn.Module, layer_name: str, prefix: str = "") -> Optional[nn.Module]:
        """从 layer_name（如 model.layers.0.self_attn.q_proj）反向找到 self_attn 模块。

        LayerWiseRunner 传入的 ``model`` 是当前 decoder layer，而 ``layer_name`` 是
        带 ``prefix`` 前缀的全名（如 ``model.layers.0``），必须先把 prefix 剥掉，
        否则 ``get_submodule`` 一定失败、RoPE 会被静默跳过。
        """
        relative_name = layer_name
        if prefix and relative_name.startswith(prefix):
            relative_name = relative_name[len(prefix) :].lstrip('.')
        if not relative_name:
            return None

        for proj_suffix in (self._q_name_pattern, self._k_name_pattern, self._qkv_name_pattern):
            idx = relative_name.rfind(proj_suffix) if proj_suffix else -1
            if idx > 0:
                with contextlib.suppress(Exception):
                    return model.get_submodule(relative_name[:idx].rstrip('.'))
        return None

    @staticmethod
    def _matches_pattern(name: str, pattern: str) -> bool:
        """名称匹配：空串表示该投影层未使用，不参与匹配（否则空串会命中所有 Linear）。"""
        return bool(pattern) and pattern in name

    def _is_target_layer(self, name: str) -> bool:
        return (
            self._matches_pattern(name, self._q_name_pattern)
            or self._matches_pattern(name, self._k_name_pattern)
            or self._matches_pattern(name, self._qkv_name_pattern)
        )

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
            self._num_attention_heads = int(_pick_attr(cfg, ('num_attention_heads', 'num_heads', 'n_head')))
            self._num_key_value_heads = int(
                _pick_attr(
                    cfg, ('num_key_value_heads', 'multi_query_group_num', 'num_kv_heads'), self._num_attention_heads
                )
            )
            self._hidden_size = int(_pick_attr(cfg, ('hidden_size', 'embed_dim')))
            self._head_dim = int(_pick_attr(cfg, ('head_dim',)))
            if not self._head_dim and self._num_attention_heads > 0 and self._hidden_size > 0:
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
            if self._matches_pattern(layer_name, self._qkv_name_pattern):
                self._q_outputs[layer_name] = output_detached
                self._k_outputs[layer_name] = output_detached
            elif self._matches_pattern(layer_name, self._q_name_pattern):
                self._q_outputs[layer_name] = output_detached
            elif self._matches_pattern(layer_name, self._k_name_pattern):
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
        if not layer_data.get('outputs', []):
            return 0.0

        # 分数只在 Q 侧计算：QKV 融合层走切分，Q 层配对同名 K 层，其余（含 K 层）跳过
        if self._matches_pattern(layer_name, self._qkv_name_pattern):
            is_qkv = True
        elif self._matches_pattern(layer_name, self._q_name_pattern):
            is_qkv = False
        else:
            return 0.0

        q_output = self._q_outputs.get(layer_name)
        if q_output is None:
            return 0.0

        layer_idx = self._layer_idx_of(layer_name)

        # 取对应 K 输出并计算
        if is_qkv:
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

    def _layer_idx_of(self, layer_name: str) -> int:
        """分配/复取层索引：首次出现的顺序即层序（与 V0 的 softmax 调用序一致）。"""
        if layer_name not in self._layer_name_to_idx:
            self._layer_name_to_idx[layer_name] = self._next_layer_idx
            self._layer_idx_to_name[self._next_layer_idx] = layer_name
            self._next_layer_idx += 1
        return self._layer_name_to_idx[layer_name]

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
                    f"but got {total_tokens}. Please ensure the calibration data "
                    f"has sufficient token length."
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

            # 对 Q/K 施加 RoPE：模型真实注意力是带位置编码的，缺少它重建的矩阵不对
            q_reshaped, k_reshaped = self._apply_rope(layer_name, q_reshaped, k_reshaped, total_tokens, device)

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
        return tensor.reshape(-1, tensor.shape[-1])

    def _apply_rope(
        self,
        layer_name: str,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> tuple:
        """对 Q/K 施加 RoPE，使重建的注意力矩阵与模型真实注意力一致。

        q: [seq, num_heads, head_dim]
        k: [seq, num_kv_heads, head_dim]
        """
        cos, sin = self._generic_rope_cos_sin(layer_name, seq_len, q.shape[-1], device)
        if cos is None or sin is None:
            self._warn_rope_once(
                'unavailable',
                layer_name,
                'can not obtain RoPE cos/sin, skip RoPE and the reconstructed attention loses position '
                'information; only standard HF rotary layouts are covered',
            )
            return q, k

        head_dim = q.shape[-1]
        if cos.shape[-1] != head_dim or sin.shape[-1] != head_dim:
            self._warn_rope_once(
                'dim_mismatch',
                layer_name,
                'RoPE dim %d does not match head_dim %d, skip RoPE' % (cos.shape[-1], head_dim),
            )
            return q, k
        if cos.shape[0] < seq_len or sin.shape[0] < seq_len:
            self._warn_rope_once(
                'seq_short',
                layer_name,
                'RoPE covers only %d positions but %d are needed, skip RoPE' % (cos.shape[0], seq_len),
            )
            return q, k

        # cos/sin 可能来自 CPU（模型首次前向在 CPU 上捕获），统一搬到计算设备
        # [seq, head_dim] -> [seq, 1, head_dim] 广播到 head 维
        cos = cos[:seq_len].to(device=device, dtype=q.dtype).unsqueeze(1)
        sin = sin[:seq_len].to(device=device, dtype=q.dtype).unsqueeze(1)

        return self._rotate_half(q, cos, sin), self._rotate_half(k, cos, sin)

    def _generic_rope_cos_sin(
        self,
        layer_name: str,
        seq_len: int,
        head_dim: int,
        device: torch.device,
    ) -> tuple:
        """内置兜底：按标准 HF 结构尽力获取 cos/sin。

        覆盖三种情况：

        1. 新版 transformers：rotary embedding 上移到顶层模型，算好后以
           ``position_embeddings`` 随 forward kwargs 下发，直接复用（含 rope_scaling）；
        2. 旧版 transformers：rotary embedding 在 attention 模块内部，调用其 forward
           取 cos/sin（``position_ids`` / ``seq_len`` / 无参三种签名依次尝试）；
        3. 前两者都拿不到时，按 config 的 rope_theta 手工推算（不含 rope_scaling）。

        该兜底只对标准 HF 结构成立；非标准旋转约定（interleaved、partial rotary 等）无法
        仅用 cos/sin 表达，需另行扩展分析能力。
        """
        # 1) 新版布局：复用模型算好的 position_embeddings
        kwargs = self._rope_kwargs_for(layer_name) or {}
        cos, sin = self._as_cos_sin_pair(kwargs.get('position_embeddings'))
        if cos is not None and sin is not None:
            return cos, sin

        attn_module = self._layer_attn_modules.get(layer_name)
        if attn_module is None:
            return None, None

        # 2) 旧版布局：attention 内自带 rotary_emb
        cos, sin = self._rotary_module_cos_sin(
            attn_module,
            seq_len,
            device,
            self._position_ids_for(layer_name, seq_len, device),
        )
        if cos is not None and sin is not None:
            return cos, sin

        # 3) 最后兜底：config 手工推算（config 未声明 RoPE 时 _compute_cos_sin_from_config 会放弃）
        cos, sin = self._compute_cos_sin_from_config(attn_module, head_dim, seq_len, device)
        if cos is not None and sin is not None:
            self._warn_rope_once(
                'manual_fallback',
                layer_name,
                'fallback to manually computed RoPE: rope_scaling in config is NOT applied, scores may be inaccurate',
            )
        return cos, sin

    @classmethod
    def _rotary_module_cos_sin(
        cls,
        attn_module: nn.Module,
        seq_len: int,
        device: torch.device,
        position_ids: torch.Tensor,
    ) -> tuple:
        """调用 attention 模块内的 ``rotary_emb`` 取 cos/sin（旧版 transformers）。

        交给 rotary_emb 自己算，rope_scaling（linear / dynamic / yarn 等）
        由模型处理；三种 forward 签名依次尝试，覆盖各版本差异。
        """
        rotary_emb = getattr(attn_module, 'rotary_emb', None)
        if rotary_emb is None:
            return None, None

        dummy_x = torch.zeros(1, seq_len, 1, device=device)
        call_kwargs_list = (
            {'position_ids': position_ids},
            {'seq_len': seq_len},
            {},
        )
        for call_kwargs in call_kwargs_list:
            try:
                result = rotary_emb(dummy_x, **call_kwargs)
            except Exception:  # nosec B112
                continue
            cos, sin = cls._as_cos_sin_pair(result)
            if cos is not None and sin is not None:
                return cos, sin

        # 旧版实现可能把 cos/sin 缓存在属性上（forward 不需要额外入参）
        cached = (getattr(rotary_emb, 'cos_cached', None), getattr(rotary_emb, 'sin_cached', None))
        return cls._as_cos_sin_pair(cached)

    @classmethod
    def _as_cos_sin_pair(cls, value: Any) -> tuple:
        """把 ``(cos, sin)`` 规整为 [seq, dim] 形态；任一侧不可用时返回 (None, None)。"""
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return None, None
        cos = cls._normalize_cos_sin(value[0])
        sin = cls._normalize_cos_sin(value[1])
        if cos is None or sin is None:
            return None, None
        return cos, sin

    @staticmethod
    def _normalize_cos_sin(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """把各种形状的 cos/sin 统一成 [seq, dim]，形状不符返回 None（宁可不加也不加错）。"""
        if not isinstance(value, torch.Tensor):
            return None
        if value.dim() > 3:
            value = value.squeeze()
        if value.dim() == 3:
            # [1, seq, dim] 或 [seq, 1, dim]
            if value.shape[0] == 1:
                value = value[0]
            elif value.shape[1] == 1:
                value = value.squeeze(1)
        if value.dim() != 2:
            return None
        return value

    def _position_ids_for(self, layer_name: str, seq_len: int, device: torch.device) -> torch.Tensor:
        """取该层位置编号：优先复用模型下发的 ``position_ids``，否则等距生成。

        位置编号不总是从 0 开始的连续序列（缓存推理、分段输入等），模型实际下发的更准。
        """
        kwargs = self._rope_kwargs_for(layer_name)
        if kwargs:
            position_ids = kwargs.get('position_ids')
            if isinstance(position_ids, torch.Tensor) and position_ids.dim() == 2:
                if position_ids.shape[-1] >= seq_len:
                    return position_ids.to(device)
        return torch.arange(seq_len, device=device).unsqueeze(0)

    def _warn_rope_once(self, key: str, layer_name: str, message: str) -> None:
        """RoPE 相关告警按原因去重（首个触发的层名会带在日志里），避免逐层刷屏。"""
        if key in self._rope_warning_keys:
            return
        self._rope_warning_keys.add(key)
        logger.warning("RA compress: %s (first at %s)", message, layer_name)

    @staticmethod
    def _compute_cos_sin_from_config(
        attn_module: nn.Module,
        head_dim: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple:
        """兜底方案：用 config 的 rope_theta 手工计算标准 rotary 的 cos/sin。

        只覆盖"各 head 共享同一套频率 + 全维旋转"的标准结构：config 未声明 RoPE
        （``rope_theta`` / ``rope_scaling`` / ``rope_parameters`` 均无）时直接放弃，
        避免给本来不用 RoPE 的模型（NoPE 等）硬加位置编码；不含 rope_scaling 的缩放，
        部分旋转（partial_rotary_factor != 1）也直接放弃，避免算错。
        """
        cfg = getattr(attn_module, 'config', None)
        if cfg is None or head_dim <= 0:
            return None, None
        if not any(getattr(cfg, name, None) is not None for name in ('rope_theta', 'rope_scaling', 'rope_parameters')):
            return None, None
        try:
            if float(getattr(cfg, 'partial_rotary_factor', 1.0) or 1.0) != 1.0:
                return None, None
            rope_theta = float(getattr(cfg, 'rope_theta', 10000.0) or 10000.0)
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos(), emb.sin()
        except Exception:
            return None, None

    @staticmethod
    def _rotate_half(
        q: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """标准 RoPE 旋转：(q * cos) + (rotate_half(q) * sin)。

        即"前后半维配对"的 HF 标准约定，要求 cos/sin 覆盖完整 head_dim。其它旋转约定
        （interleaved、partial rotary 等）无法仅用 cos/sin 表达，若后续出现这类模型，
        需另行扩展分析能力（例如由模型侧直接提供施加 RoPE 后的 Q/K）。

        q: [seq, num_heads, head_dim]
        cos/sin: [seq, 1, head_dim]
        """

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        return q * cos + rotate_half(q) * sin

    @staticmethod
    def _prefix_score_for_matrix(attn: torch.Tensor) -> float:
        """prefix 分数，等价 V0 ``_get_prefix_matching_score``（attn 为含 BOS 的单头矩阵）。"""
        return RaCompressAnalysisMethod._segment_repeat_score(attn, target_offset=1)

    @staticmethod
    def _copying_score_for_matrix(attn: torch.Tensor) -> float:
        """copying 分数，等价 V0 ``_get_copying_matching_score``（attn 为含 BOS 的单头矩阵）。"""
        return RaCompressAnalysisMethod._segment_repeat_score(attn, target_offset=0)

    @staticmethod
    def _segment_repeat_score(attn: torch.Tensor, target_offset: int) -> float:
        """向量化的 prefix（offset=1）/ copying（offset=0）分数。

        语义严格对齐 V0 的双层循环::

            for i, token_attn in enumerate(attn[1:, 1:]):      # 截断 BOS
                if i // SEG == 0: continue
                for j in range(i % SEG, i, SEG):                # j = i - m*SEG
                    score += token_attn[j + offset]
            return mean(score)                                   # 对 token 求均值

        注意与"按 (token, offset) 对求均值"不同：V0 是**每个 token 先求和、再对
        token 取平均**，若用 sum/count(pairs) 会给靠后的段加权，与 V0 不等价。

        实现上对每个段偏移 m 做一次 gather，全部留在 device 上，避免 V0 那种
        逐 token 回传 CPU 造成的海量同步。
        """
        if attn.dim() != 2:
            return 0.0

        matrix = attn[1:, 1:]  # 截断 BOS 行/列
        num_tokens = matrix.shape[0]
        if num_tokens <= DUMMY_INPUT_LENGTH:
            return 0.0

        index = torch.arange(num_tokens, device=matrix.device)
        score_acc = torch.zeros(num_tokens, device=matrix.device, dtype=torch.float32)
        counted = torch.zeros(num_tokens, device=matrix.device, dtype=torch.bool)

        for step in range(1, num_tokens // DUMMY_INPUT_LENGTH + 1):
            offset = step * DUMMY_INPUT_LENGTH
            rows = index[offset:]
            if rows.numel() == 0:
                break
            cols = rows - offset + target_offset
            score_acc[rows] += matrix[rows, cols].to(torch.float32)
            counted[rows] = True

        if not counted.any():
            return 0.0
        return float(score_acc[counted].mean())

    @property
    def supports_distributed(self) -> bool:
        return True

    def enrich_layer_scores(self, layer_scores: List[Dict[str, Any]]) -> None:
        """将 head 选择信息合并进 layer_scores。"""
        self._sync_head_scores_across_ranks()
        head_dict = self.get_compress_heads()
        prefix_map = head_dict.get('prefix_matching', {})
        copying_map = head_dict.get('copying', {})

        def _by_layer_name(heads_map: Dict[int, List[int]]) -> Dict[str, List[int]]:
            return {self._layer_idx_to_name.get(int(idx), str(idx)): heads for idx, heads in heads_map.items()}

        name_to_induction = _by_layer_name(prefix_map)
        name_to_echo = _by_layer_name(copying_map)

        for entry in layer_scores:
            entry['induction_heads'] = name_to_induction.get(entry['name'], [])
            entry['echo_heads'] = name_to_echo.get(entry['name'], [])

    def _export_head_scores_by_name(self) -> Dict[str, Tuple[List[float], List[float]]]:
        """Export local prefix/copying scores keyed by layer name."""
        exported: Dict[str, Tuple[List[float], List[float]]] = {}
        for name, idx in self._layer_name_to_idx.items():
            prefix = self._prefix_scores.get(idx)
            copying = self._copying_scores.get(idx)
            if prefix is None or copying is None:
                continue
            exported[name] = (list(prefix), list(copying))
        return exported

    def _sync_head_scores_across_ranks(self) -> None:
        """Average per-head scores across DP ranks, then rebuild local maps by name."""
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return

        payload = self._export_head_scores_by_name()
        gathered: List[Optional[Dict[str, Tuple[List[float], List[float]]]]] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, payload)

        # name -> [prefix 累加, copying 累加, 参与归并的 rank 数]
        merged: Dict[str, List[Any]] = {}
        for rank_payload in gathered:
            for name, (prefix, copying) in (rank_payload or {}).items():
                acc = merged.get(name)
                if acc is None:
                    merged[name] = [list(prefix), list(copying), 1]
                    continue
                if len(prefix) != len(acc[0]) or len(copying) != len(acc[1]):
                    logger.warning(
                        "RA compress DP: skip inconsistent head dims for layer %s",
                        name,
                    )
                    continue
                acc[0] = [a + b for a, b in zip(acc[0], prefix)]
                acc[1] = [a + b for a, b in zip(acc[1], copying)]
                acc[2] += 1

        self._prefix_scores = {}
        self._copying_scores = {}
        self._layer_name_to_idx = {}
        self._layer_idx_to_name = {}
        self._next_layer_idx = 0
        for name in sorted(merged):
            prefix_sum, copying_sum, count = merged[name]
            idx = self._layer_idx_of(name)
            self._prefix_scores[idx] = [value / count for value in prefix_sum]
            self._copying_scores[idx] = [value / count for value in copying_sum]

        logger.info(
            "RA compress: merged head scores across %d ranks for %d layers",
            dist.get_world_size(),
            len(self._prefix_scores),
        )

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
        if not self._prefix_scores and not self._copying_scores:
            logger.error(
                "RA compress: no head score was collected for any layer. "
                "The analysis will return empty head dict; please check the "
                "'error computing scores' warnings above (e.g. device mismatch, "
                "insufficient token length, or unsupported attention structure)."
            )
            return {'prefix_matching': {}, 'copying': {}}

        # GQA: 每个 kv group 内取 max，分组后索引即为 KV 头索引
        num_kv_per_group = 1
        if self._num_key_value_heads > 0:
            num_kv_per_group = max(1, self._num_attention_heads // self._num_key_value_heads)

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
        return {k: v for k, v in dictionary.items() if v}
