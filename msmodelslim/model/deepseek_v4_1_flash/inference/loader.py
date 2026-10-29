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

Build executable reference blocks / Engram from a source checkpoint.
This module is the direct successor of the legacy calibration bridge
(the checkpoint's own ``inference/model.py``) and the adapter-side handle
(``msmodelslim/model/deepseek_v4_1_flash/inference/``).  It drives the *embedded* inference
graph (``msmodelslim.model.deepseek_v4_1_flash.inference.model``) — a straight import, no
``sys.path`` / ``sys.modules`` shim — and keeps the same public entry points the
adapter and the acceptance scripts rely on.

The graph's geometry comes from the checkpoint root's ``config.json``
(``read_model_config``); a real source checkpoint signals its forward capability
by shipping ``inference/config.json``.
"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from msmodelslim.model.deepseek_v4_1_flash.inference import kernel
from msmodelslim.model.deepseek_v4_1_flash.inference import model as reference_model
from msmodelslim.model.deepseek_v4_1_flash.checkpoint import checkpoint_index
from msmodelslim.utils.logging import get_logger
from safetensors import safe_open


def install_calibration_runtime(graph=None):
    """Set the eager/calibration runtime knobs on the embedded graph (idempotent)."""
    g = graph if graph is not None else reference_model
    if getattr(g, '_dsv41_integrated', False):
        return
    if not hasattr(g, 'calibration_device'):
        g.calibration_device = 'cpu'
    g.default_dtype = torch.bfloat16
    g.apply_rotary_emb = kernel.rotary
    for module_name in ('get_window_topk_idxs', 'get_dspark_topk_idxs'):
        original = getattr(g, module_name, None)
        if original is None:
            continue

        def _on_device(*args, _original=original, **kwargs):
            return _original(*args, **kwargs).to(g.calibration_device)

        setattr(g, module_name, _on_device)
    g._dsv41_integrated = True


#: Keys ``_build_model_args`` supplies itself, so a config need not carry them.
_ARGS_INJECTED = ('dtype', 'expert_dtype', 'max_batch_size', 'max_seq_len')

#: ``vision_config`` sub-keys and the ModelArgs names they fill.  The vision tower
#: geometry lives in its own object at the top level of ``config.json``, not in
#: ``text_config`` -- dropped, every one of these falls to a ModelArgs default and
#: the tower silently comes out with zero layers.
_VISION_ALIASES = {
    'num_hidden_layers': 'vision_n_layers',
    'hidden_size': 'vision_dim',
    'num_attention_heads': 'vision_n_heads',
    'intermediate_size': 'vision_inter_dim',
    'patch_size': 'vision_patch_size',
    'rope_theta': 'vision_rope_theta',
    'downsample_ratio': 'vision_downsample_ratio',
    'max_image_tokens': 'vision_max_n_token',
    'min_pixels': 'vision_min_pixels',
    'max_wh_ratio': 'vision_max_wh_ratio',
}

#: Top-level ``config.json`` keys that belong to the model.  ``image_token_id`` is
#: the one ModelArgs field that lives outside both ``text_config`` and
#: ``vision_config``.
_TOP_LEVEL_KEYS = ('image_token_id',)

#: ``rope_scaling`` sub-keys and the ModelArgs names they fill.  The HF side packs
#: these four into one dict, so the flat alias table above cannot reach them.
_ROPE_SCALING_KEYS = {
    'factor': 'rope_factor',
    'beta_fast': 'beta_fast',
    'beta_slow': 'beta_slow',
    'original_max_position_embeddings': 'original_seq_len',
}

_TEXT_CONFIG_ALIASES = {
    'hidden_size': 'dim',
    'num_hidden_layers': 'n_layers',
    'num_nextn_predict_layers': 'n_mtp_layers',
    'moe_intermediate_size': 'moe_inter_dim',
    'num_attention_heads': 'n_heads',
    'num_experts_per_tok': 'n_activated_experts',
    'scoring_func': 'score_func',
    'routed_scaling_factor': 'route_scale',
    'sliding_window': 'window_size',
    'qk_rope_head_dim': 'rope_head_dim',
    'rms_norm_eps': 'norm_eps',
    'kv_source_layer_ids': 'kv_source_layers',
    'index_source_layer_ids': 'index_source_layers',
    'engram_pad_token_id': 'engram_pad_id',  # nosec B105  (config key name, not a password)
    'candidate_source_layer_id': 'candidate_source_layer',
    'dspark_num_experts_per_tok': 'dspark_n_activated_experts',
}


def engram_layers_in_export(root):
    """Engram layers an AscendV1 export carries tables for, from its own index.

    The index is the authority: an export describes its own Engram layout even
    when its config omits ``engram_layer_ids`` (the HF-side config has no such
    key), and it is also what names the shard holding each table -- so nothing
    here depends on how the saver chose to name its files.
    """
    index = Path(root) / 'quant_model_weights.safetensors.index.json'
    if not index.is_file():
        return []
    suffix = '.engram.embed.weight'
    layers = set()
    for key in json.loads(index.read_text()).get('weight_map', {}):
        head = key.removesuffix(suffix)
        if head != key and head.startswith('layers.') and head[len('layers.') :].isdigit():
            layers.add(int(head[len('layers.') :]))
    return sorted(layers)


def engram_table_rows(root, layer: int):
    """Row count of ``layers.<layer>.engram.embed.weight`` under ``root``, or None.

    Read from the shard's own safetensors header, so it describes the bytes the
    loader will read rather than what a config claims about them.
    """
    root = Path(root)
    key = f'layers.{layer}.engram.embed.weight'
    for index_name in ('model.safetensors.index.json', 'quant_model_weights.safetensors.index.json'):
        index = root / index_name
        if not index.is_file():
            continue
        shard = json.loads(index.read_text()).get('weight_map', {}).get(key)
        if shard is None:
            continue
        try:
            header, _ = _safetensors_header(root / shard)
        except FileNotFoundError:
            # An export may name a shard that was not downloaded (the release's
            # engram tables are separate files).  Unknown is not zero.
            continue
        shape = header.get(key, {}).get('shape')
        if shape and len(shape) == 2:
            return int(shape[0])
    return None


def _with_engram_rows(root, data: dict) -> dict:
    """Replace ``engram_num_embeddings`` with the row counts the tables actually have.

    This checkpoint's ``inference/config.json`` says 384008192/384018432 while its
    own tables hold 384006168/384016682 (the HF-side ``config.json`` agrees with the
    tables).  Which config wins decides
    ``ParallelEngramEmbedding.part_num_embeddings = num_embeddings // world_size``,
    so the wrong one puts every rank but the first on the wrong row range -- 253
    rows off, and the last rank past the end of the table.

    Reading the tables makes both entry points agree by construction, whichever
    root each was pointed at.
    """
    layer_ids = data.get('engram_layer_ids') or []
    rows = [engram_table_rows(root, i) for i in layer_ids]
    if layer_ids and all(r is not None for r in rows):
        declared = data.get('engram_num_embeddings')
        if list(declared or []) != rows:
            get_logger().info('engram_num_embeddings %s -> %s (read from the tables under %s)', declared, rows, root)
        data['engram_num_embeddings'] = rows
    return data


def _audit_defaults(graph, data: dict) -> None:
    """Report ModelArgs fields the config did not supply, so their defaults stand.

    Every field has a default, so a missed rename or a missing key builds a
    wrong-shaped model (``dim`` alone defaults to 1024) rather than raising.  The
    quantization forward and the eval read the same config through this function,
    so naming the gaps once here is what keeps them from diverging silently.
    """
    fields = getattr(graph.ModelArgs, '__dataclass_fields__', None)
    if not fields:
        return
    absent = sorted(f for f in fields if f not in data and f not in _ARGS_INJECTED)
    if absent:
        get_logger().warning('ModelArgs fields not supplied by the config, using defaults: %s', absent)


def read_model_config(root):
    """ModelArgs kwargs for a checkpoint root or an AscendV1 export root.

    One source: ``config.json``.  ``text_config`` carries the text geometry, but
    three other places in the same file hold ModelArgs fields too -- the top level
    (``image_token_id``), ``vision_config`` (the whole vision tower), and
    ``rope_scaling`` (the YaRN parameters) -- so all four are read.  Taking only
    ``text_config`` drops them and every one of those fields falls to a default
    that builds a silently different model.
    The engram row counts are then replaced by what the tables under ``root``
    actually hold (see :func:`_with_engram_rows`).

    There is deliberately no fallback to a checkpoint's own ``inference/config.json``:
    this checkpoint's copy disagrees with its own tables (and with ``config.json``),
    and having two entry points read two different files is how the quantization
    forward and the eval came to disagree about the model they were building.
    """
    root = Path(root)
    cfg = root / 'config.json'
    if not cfg.is_file():
        raise RuntimeError(f'No config.json under {root}')
    raw = json.loads(cfg.read_text())
    text_config = raw.get('text_config')
    if isinstance(text_config, dict) and ('n_layers' in text_config or 'num_hidden_layers' in text_config):
        data = dict(text_config)
    else:
        data = dict(raw)
    for hf_name, native_name in _TEXT_CONFIG_ALIASES.items():
        if native_name not in data and hf_name in data:
            data[native_name] = data[hf_name]
    scaling = data.get('rope_scaling')
    if isinstance(scaling, dict):
        for hf_name, native_name in _ROPE_SCALING_KEYS.items():
            if scaling.get(hf_name) is not None:
                data[native_name] = scaling[hf_name]
    # The vision tower and the image token live outside ``text_config``; taking
    # only ``text_config`` silently drops them.  setdefault, not assignment: a
    # flat config that already carries these names keeps its own values.
    vision = raw.get('vision_config')
    if isinstance(vision, dict):
        for hf_name, native_name in _VISION_ALIASES.items():
            if native_name not in data and vision.get(hf_name) is not None:
                data[native_name] = vision[hf_name]
    for key in _TOP_LEVEL_KEYS:
        if key not in data and raw.get(key) is not None:
            data[key] = raw[key]
    if not data.get('engram_layer_ids'):
        data['engram_layer_ids'] = engram_layers_in_export(root)
    data = _with_engram_rows(root, data)
    _audit_defaults(reference_model, data)
    return data


def _build_model_args(graph, data, max_length):
    data.update(dtype='bf16', expert_dtype=None, max_batch_size=1, max_seq_len=max_length)
    fields = getattr(graph.ModelArgs, '__dataclass_fields__', None)
    if fields is not None:
        data = {k: v for k, v in data.items() if k in fields}
    return graph.ModelArgs(**data)


def model_args(root, max_length):
    """Standalone ModelArgs for ``root``.

    Delegates to :func:`read_model_config`, so this and the engine paths cannot
    disagree about the geometry.
    """
    if root is None:
        return None
    return _build_model_args(reference_model, read_model_config(root), max_length)


def make_identity_pre_mix(reference, hidden, hc_mult):
    """Initial per-token mixing state (one-hot on the first hc copy)."""
    if not reference.available:
        return None
    return reference.graph.make_identity_pre_mix(hidden, hc_mult).to(hidden)


# ---------------------------------------------------------------------------
# Engram table storage
# ---------------------------------------------------------------------------

#: safetensors dtype name -> torch dtype.  An Engram table is stored either as a
#: float that needs no scale, or as a quantized element that does.
_ENGRAM_TORCH_DTYPE = {
    'BF16': torch.bfloat16,
    'F16': torch.float16,
    'F32': torch.float32,
    'I8': torch.int8,
    'F8_E4M3': torch.float8_e4m3fn,
    'F8_E8M0': torch.float8_e8m0fnu,
}
#: Stored quantized, so a scale MUST accompany them.
_ENGRAM_QUANTIZED = ('I8', 'F8_E4M3')
#: Scale element types: per-group float, or the MX shared exponent.
_ENGRAM_SCALE_DTYPES = ('F32', 'F8_E8M0')


def _dtype_itemsize(name: str) -> int:
    """Bytes per element of a stored dtype."""
    return torch.empty(0, dtype=_ENGRAM_TORCH_DTYPE[name]).element_size()


def _byte_map(path: Path, spec: dict, offset: int):
    """Read-only byte view of one tensor, shaped ``[..., width * itemsize]``.

    numpy has neither float8 nor bf16, so the map holds raw bytes and ``view``
    reinterprets them once the rows are in hand -- which also keeps the map's
    byte length honest for the multi-byte dtypes.
    """
    shape = list(spec['shape'])
    shape[-1] *= _dtype_itemsize(spec['dtype'])
    return np.memmap(path, mode='r', dtype=np.uint8, offset=offset, shape=tuple(shape))


def _safetensors_header(path: Path):
    """``(header, base)``: tensor specs by name, and where the data starts."""
    with open(path, 'rb') as handle:
        size = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(size))
    return {k: v for k, v in header.items() if k != '__metadata__'}, 8 + size


@dataclass(frozen=True)
class EngramStorage:
    """What an Engram table's bytes ARE, read off the tensors themselves.

    Derived from the artifact -- the weight's dtype, whether a scale rides along,
    and that scale's dtype -- rather than from a file name or a config
    declaration.  The failure mode is silent otherwise: an ``I8`` table viewed as
    ``BF16`` still produces numbers, just not the right ones.
    """

    weight_dtype: str
    scale_dtype: Optional[str] = None

    @property
    def quantized(self) -> bool:
        return self.scale_dtype is not None

    @classmethod
    def of(cls, weight_spec: dict, scale_spec: Optional[dict] = None) -> 'EngramStorage':
        """Describe one table from the specs of the tensors actually present."""
        weight_dtype = weight_spec.get('dtype')
        if weight_dtype not in _ENGRAM_TORCH_DTYPE:
            raise ValueError(
                f'engram table: unsupported weight dtype {weight_dtype!r}; known: {sorted(_ENGRAM_TORCH_DTYPE)}'
            )
        if scale_spec is None:
            if weight_dtype in _ENGRAM_QUANTIZED:
                raise ValueError(f'engram table: {weight_dtype} weights need a scale tensor, none found')
            return cls(weight_dtype)
        scale_dtype = scale_spec.get('dtype')
        if scale_dtype not in _ENGRAM_SCALE_DTYPES:
            raise ValueError(
                f'engram table: unsupported scale dtype {scale_dtype!r}; known: {list(_ENGRAM_SCALE_DTYPES)}'
            )
        if weight_dtype not in _ENGRAM_QUANTIZED:
            raise ValueError(f'engram table: {weight_dtype} weights should not carry a scale')
        width, groups = weight_spec['shape'][-1], scale_spec['shape'][-1]
        if not groups or width % groups:
            raise ValueError(f'engram table: {width} values do not divide into {groups} scale groups')
        return cls(weight_dtype, scale_dtype)

    def decode(self, weight: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One block of rows, as raw bytes, -> bf16 by this description."""
        w = weight.view(_ENGRAM_TORCH_DTYPE[self.weight_dtype])
        if not self.quantized:
            return w.to(torch.bfloat16)
        s = scale.view(_ENGRAM_TORCH_DTYPE[self.scale_dtype]).float()
        # The group width is the scale's own last dim, so nothing is assumed
        # about it here -- it happens to be 32 in every layout shipping today.
        return (w.float().unflatten(-1, (s.shape[-1], -1)) * s.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


class MappedEngram(torch.nn.Module):
    """Read-only rows of a giant Engram table via safetensors.slice, decoded on CPU.

    The table is never materialised (384M rows x 256 is ~98 GB in BF16): the
    safetensors handle stays open and ``get_slice(key)[rows]`` reads only the
    requested rows.  The source stores MXFP8 (``F8_E4M3`` weights against
    ``F8_E8M0`` scales), but that is read off the tensors rather than assumed
    (see :class:`EngramStorage`), so a differently-quantized source loads
    through the same path.
    """

    def __init__(self, model_path, key, device):
        super().__init__()
        self.device = device
        self.key = key
        self.scale_key = _engram_scale_key(key)
        root = Path(model_path)
        index = checkpoint_index(root)
        self._fh = safe_open(str(root / index[key]), framework='pt')
        self._scale_fh = (
            safe_open(str(root / index[self.scale_key]), framework='pt') if self.scale_key in index else None
        )

        def _spec(fh, name):
            sl = fh.get_slice(name)
            return {'dtype': sl.get_dtype(), 'shape': list(sl.get_shape())}

        self.storage = EngramStorage.of(
            _spec(self._fh, key), _spec(self._scale_fh, self.scale_key) if self._scale_fh else None
        )

    def forward(self, ids):
        rows = ids.cpu().numpy().tolist()
        weight = self._fh.get_slice(self.key)[rows]
        scale = self._scale_fh.get_slice(self.scale_key)[rows] if self._scale_fh else None
        return self.storage.decode(weight, scale).to(device=ids.device)


def _engram_scale_key(key: str) -> str:
    """The scale that accompanies an Engram table weight."""
    return key.removesuffix('.weight') + '.scale'


class AscendV1EngramTable(torch.nn.Module):
    """Read-only engram table over an AscendV1 export (memmap, rows decode on
    demand).

    The shard that holds the table comes from the export's own index, and the
    FORMAT is read off the tensors themselves (see :class:`EngramStorage`).
    Neither the file name nor the encoding is assumed, so a table the saver
    decides to store elsewhere, or re-quantizes to something other than bf16,
    loads through this one path.
    """

    def __init__(self, root, layer):
        super().__init__()
        self.root = Path(root)
        index = self.root / 'quant_model_weights.safetensors.index.json'
        weight_map = json.loads(index.read_text()).get('weight_map', {})
        key = f'layers.{layer}.engram.embed.weight'
        if key not in weight_map:
            raise KeyError(f'{index}: no {key}')
        shard = self.root / weight_map[key]
        specs, base = _safetensors_header(shard)
        if key not in specs:
            raise KeyError(f'{shard}: index lists {key} but the file does not hold it')
        weight = specs[key]
        # The scale may sit in the same shard or in another one.
        scale_spec, scale_shard, scale_base = None, None, 0
        scale_key = _engram_scale_key(key)
        if scale_key in weight_map:
            scale_shard = self.root / weight_map[scale_key]
            if scale_shard == shard:
                scale_spec, scale_base = specs.get(scale_key), base
            else:
                other, other_base = _safetensors_header(scale_shard)
                scale_spec, scale_base = other.get(scale_key), other_base
        self.storage = EngramStorage.of(weight, scale_spec)
        self.maps = [_byte_map(shard, weight, base + weight['data_offsets'][0])]
        if scale_spec is not None:
            self.maps.append(_byte_map(scale_shard, scale_spec, scale_base + scale_spec['data_offsets'][0]))

    def forward(self, ids):
        rows = ids.cpu().numpy()
        weight = torch.from_numpy(self.maps[0][rows].copy())
        scale = torch.from_numpy(self.maps[1][rows].copy()) if len(self.maps) > 1 else None
        return self.storage.decode(weight, scale).to(device=ids.device)
