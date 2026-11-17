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

import random
from contextlib import contextmanager
from typing import Any, ContextManager, Generator, List, Optional, Tuple

import torch

from msmodelslim.utils.logging import get_logger

from .interface import RaCompressAnalysisInterface

# ---------------------------------------------------------------------------
# ra_compress（attn_head scope）随机段校准输入
#
# attn_head 的注意力头筛选依赖「[首 token] + 随机 token 段 × REPET_TIMES」这类重复输入才能
# 诱发出 induction head，而随机 token id 经文本 tokenize 无法精确还原。故校准集在此直接由
# token id 构造（配方即下方 ``build_ra_compress_dummy_dataset``），不依赖模型适配器按名提供数据集。
#
# 首 token 需与 V0 同口径：它虽在算分时被 ``attn[1:, 1:]`` 截断，但仍参与后续各行的 softmax
# 归一化，取值会改变落在选择阈值边界的头。取值配方（``resolve_v0_prefix_token_id``）留在本模块，
# 只向模型侧取 tokenizer（``RaCompressAnalysisInterface.get_tokenizer``）；取不到时回落兜底值并告警。
#
# 注入点是 ``generate_model_forward``（模型前向入口）：runner 先用 ``handle_dataset`` 把数据集
# 转成模型输入，再逐样本交给 ``generate_model_forward``，在后者把 token_id 输入整段换成随机段
# 即可，通用分词路径无需感知预分词输入，runner 侧只需一个能正常分词的占位样本驱动前向。
# 相较顶替 ``handle_dataset``，此处不受 ``get_input_datas`` 进程级缓存（key 固定为
# "data_loader"，命中即不再回调适配器）影响：即便数据集已被别的流程缓存，进入模型的仍是随机段。
# DP 子进程经 ``mp.spawn`` 反序列化适配器后同样生效，见 ``_RandomTokenForward``。
# ---------------------------------------------------------------------------
RA_COMPRESS_METRIC = "ra_compress"

# 占位样本：attn_head 的 token_id 输入在前向口被顶替，故只需一个任何 tokenizer 都能正常分词、
# 且必然非空的样本，用于驱动 runner 的「数据集 → 模型输入」流程。
PLACEHOLDER_DATASET: List[Any] = ["A"]

# ---------------------------------------------------------------------------
# 随机重复段校准输入配方（V0 口径）
#
# 单段长度、重复段数、token id 取值范围与随机种子均沿用 V0；首 token 由本模块按 V0 口径解析
# （见 ``resolve_prefix_token_id``），模型侧只提供 tokenizer。
# ---------------------------------------------------------------------------
DUMMY_INPUT_LENGTH = 2500
REPET_TIMES = 4

# 随机 token id 取值范围下界（沿用 V0 取值范围）
TOKEN_ID_LOW = 10000
RANDOM_SEED = 42

# 首 token 兜底取值：适配器无法按 V0 口径解析出首 token 时使用
PREFIX_TOKEN_ID = 0

# V0 在空输入下的兜底文本：``tokenizer('')`` 为空时改用该文本的末位 token
_V0_FALLBACK_TEXT = "A"

_INPUT_IDS_KEY = "input_ids"


def is_random_token_metric(metrics: str) -> bool:
    """该指标是否依赖随机重复段输入（当前仅 ra_compress）。"""
    return metrics == RA_COMPRESS_METRIC


def build_ra_compress_dummy_dataset(
    prefix_token_id: int = PREFIX_TOKEN_ID,
    dummy_input_length: int = DUMMY_INPUT_LENGTH,
    repet_times: int = REPET_TIMES,
    seed: int = RANDOM_SEED,
) -> List[List[torch.Tensor]]:
    """构造随机重复段校准输入，shape 为 ``[1, 1 + dummy_input_length * repet_times]``。

    Args:
        prefix_token_id: 首 token 的 token id（截断行/列，但仍参与后续行的 softmax 归一化），
            应取自适配器提供的 V0 口径取值或 ``PREFIX_TOKEN_ID`` 兜底值。
        dummy_input_length: 单段 token 长度。
        repet_times: 重复段数。
        seed: 随机种子（V0 未固定种子，此处固定以保证结果可复现）。

    Returns:
        已预分词的校准数据 ``[[input_ids, attention_mask]]``。
    """
    # 用独立 Random 实例，避免污染全局 random 状态
    rng = random.Random(seed)  # nosec B311
    rand_tokens = torch.tensor(
        [rng.randint(TOKEN_ID_LOW, TOKEN_ID_LOW + dummy_input_length) for _ in range(dummy_input_length)] * repet_times
    )
    input_ids = torch.cat((torch.tensor([prefix_token_id]), rand_tokens)).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return [[input_ids, attention_mask]]


def resolve_model_tokenizer(model_adapter: Any) -> Any:
    """取模型侧 tokenizer（模型事实）。

    模型适配器实现了 :class:`RaCompressAnalysisInterface` 才会被询问；未实现该接口、
    默认实现返回 ``None`` 或取值异常时返回 ``None``，由调用方决定兜底。
    """
    if not isinstance(model_adapter, RaCompressAnalysisInterface):
        return None
    try:
        return model_adapter.get_tokenizer()
    except Exception as error:  # pylint: disable=broad-exception-caught  适配器实现差异不应中断分析
        get_logger().warning(
            "RA compress: can not get tokenizer from %s: %s",
            type(model_adapter).__name__,
            error,
        )
        return None


def resolve_v0_prefix_token_id(tokenizer: Any) -> Optional[int]:
    """按 V0 口径解析 ra_compress 随机重复段首 token 的 token id。

    V0 的做法是取 ``tokenizer('')`` 的末位 token，若结果为空则退回 ``tokenizer('A')``
    的末位 token：Qwen2 系列不加 BOS，空串 tokenize 结果为空，取到的是 ``'A'`` 的 id。
    该取值经各行 softmax 归一化轻微影响得分，故需与 V0 同口径复现，而不是固定取 BOS 或 0。

    Args:
        tokenizer: 模型侧提供的 tokenizer，可为 ``None``。

    Returns:
        首 token 的 token id；tokenizer 缺失、解析异常或结果为空时返回 ``None``。
    """
    if tokenizer is None:
        return None
    try:
        input_ids = tokenizer("", return_tensors="pt")[_INPUT_IDS_KEY].tolist()[0]
        if not input_ids:
            input_ids = tokenizer(_V0_FALLBACK_TEXT, return_tensors="pt")[_INPUT_IDS_KEY].tolist()[0]
    except Exception:  # pylint: disable=broad-exception-caught  解析失败不应中断分析
        return None
    return int(input_ids[-1]) if input_ids else None


def resolve_prefix_token_id(model_adapter: Any) -> int:
    """按 V0 口径解析首 token id：模型侧只提供 tokenizer，配方留在本模块。

    取不到 tokenizer 或解析失败时回落兜底值并告警（不静默），以保证首 token 口径可追溯。
    """
    resolved = resolve_v0_prefix_token_id(resolve_model_tokenizer(model_adapter))
    if resolved is None:
        get_logger().warning(
            "RA compress: can not resolve the V0 prefix token id from %s, fallback to %s",
            type(model_adapter).__name__,
            PREFIX_TOKEN_ID,
        )
        return PREFIX_TOKEN_ID
    return int(resolved)


def _resolve_forward_device(inputs: Any, model: Any) -> torch.device:
    """取本次前向实际使用的设备：优先随行输入（runner 已搬运），退回模型参数所在设备。"""
    if isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], torch.Tensor):
        return inputs[0].device
    if isinstance(inputs, torch.Tensor):
        return inputs.device
    parameters = getattr(model, "parameters", None)
    first_parameter = next(parameters(), None) if callable(parameters) else None
    return first_parameter.device if first_parameter is not None else torch.device("cpu")


class _RandomTokenForward:
    """顶替 ``generate_model_forward`` 的前向入口：忽略随行输入，改喂随机重复段 token。

    实现为可调用对象而非闭包，以保证适配器被 ``mp.spawn`` 序列化到 DP 子进程后仍然可用；且不
    持有原绑定方法——属性名 ``generate_model_forward`` 反序列化后会指回本对象，持有绑定方法将
    形成无限递归，故原始实现按类获取。
    """

    def __init__(self, model_adapter: Any, token_inputs: List[Any]):
        self._model_adapter = model_adapter
        self._token_inputs = token_inputs

    def __call__(self, model: Any, inputs: Any = None) -> Any:
        device = _resolve_forward_device(inputs, model)
        token_inputs = [tensor.to(device) for tensor in self._token_inputs]
        original_forward = type(self._model_adapter).generate_model_forward
        return original_forward(self._model_adapter, model, token_inputs)


@contextmanager
def _random_token_forward(model_adapter: Any, token_inputs: List[Any]) -> Generator[None, None, None]:
    """运行期用随机重复段输入顶替适配器的 ``generate_model_forward``，退出时还原。

    ``LayerWiseRunner``（含 DP 子进程）先经 ``handle_dataset`` 把数据集转成模型输入，再按样本
    调用 ``generate_model_forward``，故在此处替换即可让随机 token 段进入前向。
    """
    original_forward = model_adapter.generate_model_forward
    model_adapter.generate_model_forward = _RandomTokenForward(model_adapter, token_inputs)
    get_logger().info(
        "Override %s.generate_model_forward with %s random repeat-segment tokens",
        model_adapter.__class__.__name__,
        RA_COMPRESS_METRIC,
    )
    try:
        yield
    finally:
        model_adapter.generate_model_forward = original_forward


def build_random_token_calib(model_adapter: Any) -> Tuple[List[Any], ContextManager[None]]:
    """构造 ra_compress 的校准输入，并返回顶替模型前向的上下文。

    Returns:
        ``(calib_data, forward_override)``：``calib_data`` 为驱动 runner「数据集 → 模型输入」
        流程的占位样本（真实 token_id 输入在 ``forward_override`` 生效期间注入）；
        ``forward_override`` 为可重入管理的上下文管理器，退出时还原适配器原前向。
    """
    prefix_token_id = resolve_prefix_token_id(model_adapter)
    token_inputs = build_ra_compress_dummy_dataset(prefix_token_id=prefix_token_id)[0]
    get_logger().info(
        "Built random repeat-segment inputs for %s: token shape %s, prefix token id %s",
        RA_COMPRESS_METRIC,
        tuple(token_inputs[0].shape),
        prefix_token_id,
    )
    return PLACEHOLDER_DATASET, _random_token_forward(model_adapter, token_inputs)
