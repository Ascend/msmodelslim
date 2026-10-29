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

DeepSeek-V4.1 对话渲染（文本路径的最小忠实实现）。
**不引用模型目录下的代码** —— 官方实现在 `<model>/encoding/encoding.py`，
按项目约定推理代码要集成进 msmodelslim，所以这里放一份对应实现。
改动时请对照官方更新。

覆盖范围：文本、单/多轮、system/user/assistant，无工具、无图片 —— eval 只用这一种。
官方 `encode_messages(..., thinking_mode, drop_thinking=True, add_default_bos_token=True)`
在"单轮 user + chat 模式"下渲染出的就是：

    BOS + USER_SP + content + ASSISTANT_SP + THINK_END

与官方 encoding/encoding.py 里的 render_message 逐分支对齐。
"""

from typing import Any, Dict, List

BOS = '<｜begin▁of▁sentence｜>'
EOS = '<｜end▁of▁sentence｜>'
THINK_START = '<think>'
THINK_END = '</think>'
USER_SP = '<｜User｜>'
ASSISTANT_SP = '<｜Assistant｜>'
SYSTEM_SP = '<｜System｜>'

_VALID_MODES = ('chat', 'thinking')
_KNOWN_ROLES = ('system', 'user', 'assistant')


def encode_messages(
    messages: List[Dict[str, Any]],
    thinking_mode: str = 'chat',
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
) -> str:
    """把 OpenAI 格式的 messages 渲染成 V4.1 的 prompt 字符串。

    Args:
        messages: `[{"role": "user", "content": "..."}]`；assistant 消息可带
            `reasoning_content`。
        thinking_mode: `"chat"` 或 `"thinking"`。
        drop_thinking: 是否丢弃最后一条 user 之前的历史 thinking（官方默认 True）。
        add_default_bos_token: 是否在开头加 BOS（官方默认 True）。

    Returns:
        渲染好的 prompt 字符串。
    """
    if thinking_mode not in _VALID_MODES:
        raise ValueError(f'invalid thinking_mode {thinking_mode!r}, expected one of {_VALID_MODES}')

    last_user_idx = max((i for i, m in enumerate(messages) if m.get('role') == 'user'), default=-1)
    out = BOS if add_default_bos_token else ''

    for index, msg in enumerate(messages):
        role = msg.get('role')
        content = msg.get('content') or ''
        if role not in _KNOWN_ROLES:
            raise NotImplementedError(
                f'unsupported role {role!r}; this renderer covers text-only {_KNOWN_ROLES}（无 tool / latest_reminder）'
            )

        if role == 'system':
            if index > 0:
                out += SYSTEM_SP
            out += content
        elif role == 'user':
            out += USER_SP + content
        else:  # assistant
            thinking = ''
            if thinking_mode == 'thinking' and (not drop_thinking or index > last_user_idx):
                thinking = (msg.get('reasoning_content') or '') + THINK_END
            out += thinking + content + EOS

        # 过渡 token：官方只在"下一条不是 assistant/latest_reminder"时才追加
        if index + 1 < len(messages) and messages[index + 1].get('role') not in ('assistant', 'latest_reminder'):
            continue
        if role == 'user' or (role == 'system' and index > 0):
            out += ASSISTANT_SP
            if thinking_mode == 'thinking' and (not drop_thinking or index >= last_user_idx):
                out += THINK_START
            else:
                out += THINK_END

    return out


def render_user_prompt(content: str, thinking_mode: str = 'chat') -> str:
    """单轮 user 消息的快捷入口（eval 用）。"""
    return encode_messages([{'role': 'user', 'content': content}], thinking_mode=thinking_mode)
