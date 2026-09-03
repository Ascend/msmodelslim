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

DeepSeek-V4-Flash-DSpark / DeepSeek-V4-Pro-DSpark 模型适配器。

在 DeepSeekV4ModelAdapter 基础上：
  - 完整主模型（layers.*）量化流程与对应 Flash/Pro backbone 一致
  - DSpark draft（mtp.*）其余模块不做 QuaRot；mtp.0.main_proj 右旋以抵消 Q 域 main_hidden
  - 在 QuaRot 前缓存主模型 embed/head；mtp.0.embed / mtp.{N-1}.head 复制并落盘为旋转前原始权重
"""

import os
from typing import Any, Dict, Generator, List, Optional, Tuple
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from msmodelslim.core.const import DeviceType
from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.processor.quarot import QuaRotInterface
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import logger_setter, get_logger
from msmodelslim.utils.security import json_safe_load

from ..deepseek_v4.model_adapter import DeepSeekV4ModelAdapter
from ..common.layer_wise_forward import TransformersForwardBreak
from .model import DSparkBlock, enable_dspark_moe_comm
from .mtp_quant_module import (
    ensure_config_n_mtp_layers,
    ensure_dedicated_mtp_embed,
    ensure_dedicated_mtp_head,
    load_dspark_mtp_state_dict,
    prune_dspark_mtp_stage_modules,
    refresh_mtp_embed_from_checkpoint,
    refresh_mtp_head_from_checkpoint,
    wrap_dspark_mtp_decoder,
)

_FLASH_MTP_LN_KEYS = ("enorm", "hnorm")
_FLASH_MTP_ROTATE_LEFT = ("e_proj", "h_proj")
_FLASH_MTP_ROTATE_RIGHT = ("head", "emb.tok_emb", "e_proj", "h_proj")


@logger_setter("msmodelslim.model.deepseek_v4_dspark")
class DeepSeekV4DSparkModelAdapter(DeepSeekV4ModelAdapter):  # pylint: disable=too-many-ancestors
    """DeepSeek-V4-Flash/Pro-DSpark：主模型 + 多层 DSpark draft 量化。"""

    def get_model_pedigree(self) -> str:
        return "deepseek_v4_dspark"

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        """加载主模型后立刻缓存旋转前的 embed/head，供 MTP 复制与落盘。"""
        model = super().init_model(device=device)
        # 主模型 layers 的 MoE 也需支持逐层 offload 后的 CPU 激活 + HCCL
        enable_dspark_moe_comm(model)
        self._cache_unrotated_shared_weights(model)
        return model

    def load_decoder_if_not_exist(self, model: nn.Module, layer_prefix: str, idx: int):
        decoder = super().load_decoder_if_not_exist(model, layer_prefix=layer_prefix, idx=idx)
        enable_dspark_moe_comm(decoder)
        return decoder

    def _cache_unrotated_shared_weights(self, model: nn.Module) -> None:
        embed = getattr(model, "embed", None)
        head = getattr(model, "head", None)
        if embed is None or not hasattr(embed, "weight") or embed.weight.is_meta:
            raise InvalidModelError(
                "Cannot cache unrotated embed.weight.",
                action="Ensure backbone embed is loaded before QuaRot.",
            )
        if head is None or not hasattr(head, "weight") or head.weight.is_meta:
            raise InvalidModelError(
                "Cannot cache unrotated head.weight.",
                action="Ensure backbone head is loaded before QuaRot.",
            )
        # pylint: disable=attribute-defined-outside-init
        self._unrotated_embed_weight = embed.weight.detach().cpu().contiguous().clone()
        self._unrotated_head_weight = head.weight.detach().cpu().contiguous().clone()
        # pylint: enable=attribute-defined-outside-init
        get_logger().info(
            "Cached unrotated embed/head for MTP copy & save, shapes=%s/%s",
            tuple(self._unrotated_embed_weight.shape),
            tuple(self._unrotated_head_weight.shape),
        )

    def _get_unrotated_embed_weight(self) -> Optional[torch.Tensor]:
        return getattr(self, "_unrotated_embed_weight", None)

    def _get_unrotated_head_weight(self) -> Optional[torch.Tensor]:
        return getattr(self, "_unrotated_head_weight", None)

    def _get_n_mtp_layers(self) -> int:
        return int(getattr(self.config, "n_mtp_layers", 0))

    def _get_last_mtp_idx(self) -> Optional[int]:
        n_mtp = self._get_n_mtp_layers()
        if n_mtp <= 0:
            return None
        return n_mtp - 1

    @staticmethod
    def _primary_rot_pair(rot_pairs: List[QuaRotInterface.RotatePair]) -> Optional[QuaRotInterface.RotatePair]:
        return rot_pairs[0] if rot_pairs else None

    def load_mtp_decoder_if_not_exist(self, model: nn.Module, layer_prefix: str, mtp_idx: int):
        try:
            mtp_block = model.mtp[mtp_idx]
        except (IndexError, AttributeError):
            with patch.object(nn.Linear, "reset_parameters", lambda _self: None):
                get_logger().info("Creating DSpark MTP decoder layer %s", mtp_idx)
                layer_id = self.config.num_hidden_layers + mtp_idx
                n_mtp = ensure_config_n_mtp_layers(self.config, str(self.model_path))
                mtp_block = DSparkBlock(layer_id, self.config)
                prune_dspark_mtp_stage_modules(mtp_block, mtp_idx, n_mtp)

                state_dict = load_dspark_mtp_state_dict(self.model_path, mtp_block, layer_prefix)
                mtp_block.load_state_dict(state_dict, strict=False)

                wrap_dspark_mtp_decoder(
                    mtp_decoder=mtp_block,
                    config=self.config,
                    model_path=str(self.model_path),
                    layer_prefix=layer_prefix,
                    mtp_idx=mtp_idx,
                    n_mtp_layers=n_mtp,
                    backbone_embed=model.embed,
                    backbone_head=model.head,
                    original_embed_weight=self._get_unrotated_embed_weight(),
                    original_head_weight=self._get_unrotated_head_weight(),
                )

                mtp_block.eval()
                model.mtp.append(mtp_block)
                get_logger().info("Create DSpark MTP decoder layer %s successfully", mtp_idx)
        else:
            # 已存在时勿反复在 CPU 重建 embed/head，否则会覆盖 LoadProcessor 放到 NPU 的权重。
            last = self._get_last_mtp_idx()
            device = self._prefer_module_device(mtp_block)
            if mtp_idx == 0:
                embed = getattr(mtp_block, "embed", None)
                if embed is None or self._module_has_meta_params(embed):
                    ensure_dedicated_mtp_embed(
                        mtp_block,
                        None,
                        model_path=str(self.model_path),
                        original_weight=self._get_unrotated_embed_weight(),
                        device=device,
                    )
            if last is not None and mtp_idx == last:
                head = getattr(mtp_block, "head", None)
                if head is None or self._module_has_meta_params(head):
                    ensure_dedicated_mtp_head(
                        mtp_decoder=mtp_block,
                        model_path=str(self.model_path),
                        layer_prefix=layer_prefix,
                        dim=self.config.dim,
                        vocab_size=self.config.vocab_size,
                        original_weight=self._get_unrotated_head_weight(),
                        device=device,
                    )
        return mtp_block

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        first_block_input: Optional[Tuple] = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs)
            raise TransformersForwardBreak()

        remove_handler = model.layers[0].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)

        try:
            if isinstance(inputs, (list, tuple)):
                model(inputs[0])
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except TransformersForwardBreak:
            pass
        except Exception as e:
            raise e
        finally:
            remove_handler.remove()

        if first_block_input is None:
            raise InvalidModelError("Can't get first block input.", action="Please check the model and input")

        if dist.is_initialized():
            dist.barrier()

        target_ids = set(getattr(self.config, "dspark_target_layer_ids", ()) or ())
        main_hiddens: List[torch.Tensor] = []
        main_x: Optional[torch.Tensor] = None
        mtp_decode_pos: Optional[int] = None
        draft_token_ids: Optional[torch.Tensor] = None

        args, kwargs = first_block_input
        h, start_pos, input_ids = args

        for name, block in self.generate_decoder_layer(model):
            if name.startswith("layers."):
                layer_idx = int(name.split(".")[1])
                h = yield ProcessRequest(name, block, args, kwargs)
                if layer_idx in target_ids:
                    main_hiddens.append(h.mean(dim=2))
                args = (h, start_pos, input_ids)
                continue

            if name.startswith("mtp."):
                mtp_idx = int(name.split(".")[1])
                if mtp_idx == 0:
                    if not main_hiddens:
                        raise InvalidModelError(
                            "DSpark main_hidden is empty.",
                            action="Check dspark_target_layer_ids and main model forward.",
                        )
                    main_hidden = torch.cat(main_hiddens, dim=-1)
                    mtp_decode_pos = self._resolve_mtp_decode_pos(start_pos, main_hidden.size(1))
                    args, kwargs = self.dspark_mtp_preprocess(
                        model=model,
                        mtp_decoder=block,
                        mtp_idx=mtp_idx,
                        main_hidden=main_hidden,
                        input_ids=input_ids,
                        start_pos=mtp_decode_pos,
                        kwargs=kwargs,
                    )
                    main_x = self._slice_main_x_for_decode(kwargs.pop("main_x"), mtp_decode_pos)
                    draft_token_ids = args[2]

                if draft_token_ids is None:
                    raise InvalidModelError(
                        "DSpark draft_token_ids is missing for MTP forward.",
                        action="Check mtp.0 preprocess and input_ids.",
                    )
                args = (args[0], mtp_decode_pos, draft_token_ids)
                kwargs = {**kwargs, "main_x": main_x}
                args, kwargs = self._align_mtp_request(args, kwargs)

            h = yield ProcessRequest(name, block, args, kwargs)
            args = (h, start_pos, input_ids)

    @staticmethod
    def _current_npu_device() -> torch.device:
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device(f"npu:{torch.npu.current_device()}")
        return torch.device("cpu")

    @classmethod
    def _align_tensors_to_device(cls, device: torch.device, *values: Any) -> Tuple[Any, ...]:
        aligned = []
        for value in values:
            if torch.is_tensor(value):
                aligned.append(value.to(device))
            else:
                aligned.append(value)
        return tuple(aligned)

    def _align_mtp_request(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        device = self._current_npu_device()
        x, start_pos, input_ids = args
        x, input_ids = self._align_tensors_to_device(device, x, input_ids)
        aligned_kwargs = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in kwargs.items()}
        return (x, start_pos, input_ids), aligned_kwargs

    @staticmethod
    def _resolve_mtp_decode_pos(backbone_start_pos: int, seq_len: int) -> int:
        if backbone_start_pos > 0:
            return backbone_start_pos
        return max(seq_len - 1, 1)

    @staticmethod
    def _slice_main_x_for_decode(main_x: torch.Tensor, decode_pos: int) -> torch.Tensor:
        if main_x.dim() != 3 or main_x.size(1) <= 1:
            return main_x
        idx = min(decode_pos, main_x.size(1) - 1)
        return main_x[:, idx : idx + 1, :]

    @staticmethod
    def _module_has_meta_params(module: nn.Module) -> bool:
        return any(p.is_meta for p in module.parameters(recurse=True))

    def _prefer_module_device(self, module: nn.Module) -> torch.device:
        for param in module.parameters(recurse=True):
            if not param.is_meta:
                return param.device
        return self._current_npu_device()

    def _materialize_mtp_decoder_if_needed(self, mtp_decoder: DSparkBlock, mtp_idx: int) -> None:
        """逐层 offload 到 meta 后，校准前向需从 checkpoint 重新加载 MTP 权重。"""
        if not self._module_has_meta_params(mtp_decoder):
            return
        device = self._current_npu_device()
        layer_prefix = f"mtp.{mtp_idx}"
        get_logger().info("Reloading %s weights from checkpoint for calibration forward", layer_prefix)
        state_dict = load_dspark_mtp_state_dict(self.model_path, mtp_decoder, layer_prefix)
        mtp_decoder.load_state_dict(state_dict, strict=False)
        if mtp_idx == 0:
            embed = getattr(mtp_decoder, "embed", None)
            if embed is None or self._module_has_meta_params(embed):
                ensure_dedicated_mtp_embed(
                    mtp_decoder,
                    None,
                    model_path=str(self.model_path),
                    original_weight=self._get_unrotated_embed_weight(),
                    device=device,
                )
        last = self._get_last_mtp_idx()
        if last is not None and mtp_idx == last:
            head = getattr(mtp_decoder, "head", None)
            if head is None or self._module_has_meta_params(head):
                ensure_dedicated_mtp_head(
                    mtp_decoder=mtp_decoder,
                    model_path=str(self.model_path),
                    layer_prefix=layer_prefix,
                    dim=self.config.dim,
                    vocab_size=self.config.vocab_size,
                    original_weight=self._get_unrotated_head_weight(),
                    device=device,
                )
        mtp_decoder.to(device)

    def dspark_mtp_preprocess(
        self,
        model: nn.Module,
        mtp_decoder: DSparkBlock,
        mtp_idx: int,
        main_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        start_pos: int,
        kwargs: Dict[str, Any],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        del model
        device = self._current_npu_device()
        self._materialize_mtp_decoder_if_needed(mtp_decoder, mtp_idx)
        # embed/head 可能刚在 CPU 重建，或 Load 后局部仍留在 CPU，前向前对齐设备。
        mtp_decoder.to(device)
        if mtp_idx == 0:
            embed = getattr(mtp_decoder, "embed", None)
            if embed is None or self._module_has_meta_params(embed):
                ensure_dedicated_mtp_embed(
                    mtp_decoder,
                    None,
                    model_path=str(self.model_path),
                    original_weight=self._get_unrotated_embed_weight(),
                    device=device,
                )
            elif next(embed.parameters()).device != device:
                embed.to(device)

        draft_token_ids = input_ids[:, -1] if input_ids.dim() == 2 else input_ids
        main_hidden = main_hidden.to(device)
        if torch.is_tensor(draft_token_ids):
            draft_token_ids = draft_token_ids.to(device)

        x, main_x = mtp_decoder.forward_embed(main_hidden, draft_token_ids)
        return (x, start_pos, draft_token_ids), {**kwargs, "main_x": main_x}

    def _strip_mtp_ln_fuse_entries(self, ln_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
        return {key: value for key, value in ln_map.items() if not key.startswith("mtp.")}

    def get_ln_fuse_map(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        pre_ln_map, ln_map = super().get_ln_fuse_map()
        return pre_ln_map, self._strip_mtp_ln_fuse_entries(ln_map)

    def _remove_flash_mtp_rotate_entries(self, pair: QuaRotInterface.RotatePair, n_mtp: int) -> None:
        for mtp_idx in range(n_mtp):
            for suffix in _FLASH_MTP_ROTATE_LEFT:
                pair.left_rot.pop(f"mtp.{mtp_idx}.{suffix}", None)
            for suffix in _FLASH_MTP_ROTATE_RIGHT:
                pair.right_rot.pop(f"mtp.{mtp_idx}.{suffix}", None)

    def _remove_mtp_rotate_entries(self, pair: QuaRotInterface.RotatePair) -> None:
        for key in list(pair.left_rot.keys()):
            if key.startswith("mtp."):
                pair.left_rot.pop(key, None)
        for key in list(pair.right_rot.keys()):
            if key.startswith("mtp.") and key != "mtp.0.main_proj":
                pair.right_rot.pop(key, None)

    def _add_dspark_main_proj_rotate(self, pair: QuaRotInterface.RotatePair, block_size: int) -> None:
        """main_hidden 来自 QuaRot 后的 backbone（Q 域），对 main_proj 右旋以抵消输入旋转。"""
        if self._get_n_mtp_layers() <= 0:
            return
        rot = QuaRotInterface.get_rotate_command(
            size=self.config.dim,
            mode=QuaRotInterface.QuaRotMode.HADAMARD,
            block_size=block_size,
        )
        pair.right_rot["mtp.0.main_proj"] = rot

    def get_rotate_map(
        self, block_size: int
    ) -> Tuple[List[QuaRotInterface.RotatePair], List[QuaRotInterface.RotatePair]]:
        pre_run_list, rot_pairs = super().get_rotate_map(block_size)

        n_mtp = self._get_n_mtp_layers()
        for pair in rot_pairs:
            self._remove_flash_mtp_rotate_entries(pair, n_mtp)
            self._remove_mtp_rotate_entries(pair)

        primary = self._primary_rot_pair(rot_pairs)
        if primary is not None:
            self._add_dspark_main_proj_rotate(primary, block_size)

        return pre_run_list, rot_pairs

    def ascendv1_save_module_preprocess(
        self, prefix: str, module: nn.Module, model: nn.Module
    ) -> Tuple[str, nn.Module]:
        model_path = str(self.model_path)
        if prefix == "mtp.0.embed" or (prefix.endswith(".embed") and prefix.startswith("mtp.0.")):
            if isinstance(module, nn.Module) and hasattr(module, "weight"):
                refresh_mtp_embed_from_checkpoint(
                    module,
                    model_path,
                    original_weight=self._get_unrotated_embed_weight(),
                )
            return "mtp.0.embed", module
        last = self._get_last_mtp_idx()
        if last is not None and prefix == f"mtp.{last}.head":
            if isinstance(module, nn.Linear):
                refresh_mtp_head_from_checkpoint(
                    module,
                    model_path,
                    original_weight=self._get_unrotated_head_weight(),
                )
            return prefix, module
        return super().ascendv1_save_module_preprocess(prefix, module, model)

    def _load_config(self, trust_remote_code=False) -> object:
        args = super()._load_config(trust_remote_code=trust_remote_code)
        config_data = json_safe_load(os.path.join(self.model_path, "config.json"))

        args.dspark_target_layer_ids = tuple(config_data.get("dspark_target_layer_ids", []))
        args.dspark_block_size = config_data.get("dspark_block_size", 0)
        args.dspark_noise_token_id = config_data.get("dspark_noise_token_id", 0)
        args.dspark_markov_rank = config_data.get("dspark_markov_rank", 0)
        args.vocab_size = config_data.get("vocab_size", getattr(args, "vocab_size", 129280))
        args.hc_mult = config_data.get("hc_mult", getattr(args, "hc_mult", 4))

        args.n_mtp_layers = ensure_config_n_mtp_layers(args, str(self.model_path), config_data)
        get_logger().info(
            "DSpark config: n_mtp_layers=%s, target_layer_ids=%s",
            args.n_mtp_layers,
            args.dspark_target_layer_ids,
        )
        return args
