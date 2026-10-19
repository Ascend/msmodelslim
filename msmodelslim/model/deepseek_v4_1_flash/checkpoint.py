"""DeepSeek-V4.1-Flash checkpoint reading and Ascend weight export."""

import hashlib
import json
import math
import os
from pathlib import Path
import struct

import torch
from safetensors import safe_open

from msmodelslim.model.common.tensor_stream import TensorInput, TensorOutput, TensorStreamProcessor

if not __debug__:
    raise RuntimeError('Do not use Python -O: conversion requires validation assertions')

DTYPES = {
    'BF16': torch.bfloat16,
    'F32': torch.float32,
    'I8': torch.int8,
    'F8_E4M3': torch.float8_e4m3fn,
    'F8_E8M0': torch.float8_e8m0fnu,
}
DTYPE_NAMES = {torch.bfloat16: 'BF16', torch.float32: 'F32', torch.int8: 'I8', torch.int64: 'I64'}
ENGRAM_ROTATION_CONFIG = {
    'value_projection_rotated': True,
    'value_basis': 'quarot_global',
    'key_and_gate_basis': 'original',
    'runtime_delta_rotation': False,
}

#: Element type of a quantized Engram table's per-group scale.  E8M0 is the MX
#: shared exponent -- one byte holding a power of two -- which is the
#: representation the source tables already carry for their own scales, and a
#: quarter the bytes of F32.  It holds powers of two only, so the writer rounds
#: up rather than to nearest (see ``_e8m0_scale``).  The reader does not consult
#: this: it reads the storage off the tensor dtype (loader.py), so this constant
#: only says what the writer emits -- and what the description must therefore
#: declare.
ENGRAM_SCALE_DTYPE = 'F8_E8M0'


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def checkpoint_index(root) -> dict:
    """``{tensor_key: shard_filename}`` for the multi-shard source checkpoint."""
    return json.loads((Path(root) / 'model.safetensors.index.json').read_text())['weight_map']


def tensor_spec(model_path, key):
    """``(shape, dtype)`` of one tensor via safe_open slice metadata (no data read)."""
    root = Path(model_path)
    with safe_open(str(root / checkpoint_index(root)[key]), framework='pt') as f:
        sl = f.get_slice(key)
        return list(sl.get_shape()), str(sl.get_dtype())


def load_tensor(model_path, key) -> torch.Tensor:
    """Full-tensor read through safetensors.safe_open."""
    root = Path(model_path)
    with safe_open(str(root / checkpoint_index(root)[key]), framework='pt') as f:
        return f.get_tensor(key)


def decode_tensor(model_path, key) -> torch.Tensor:
    """Read one checkpoint weight and decode FP8/FP4-nibble/INT8 to BF16.

    The scale companion is read the same way; both tensors are loaded whole --
    no partial (row-sliced) IO.  FP4 experts are stored as packed uint8/int8
    nibbles and decoded through a byte-level lookup table (two bf16 per byte).
    """
    w = load_tensor(model_path, key)
    dtype = str(w.dtype)
    if dtype not in ('torch.float8_e4m3fn', 'torch.int8'):
        return w
    scale = load_tensor(model_path, key.removesuffix('.weight') + '.scale').float()
    if 'engram.embed.weight' in key:
        return (w.float().reshape(-1, 8, 32) * scale.unsqueeze(-1)).reshape(-1, 256).bfloat16()
    if dtype == 'torch.float8_e4m3fn':
        n, k = w.shape
        assert tuple(scale.shape) == (n // 32, k // 32), (key, w.shape, scale.shape)  # nosec
        return (w.float().reshape(n // 32, 32, k // 32, 32) * scale[:, None, :, None]).reshape(n, k).bfloat16()
    assert '.ffn.experts.' in key, key  # nosec
    data = w.view(torch.uint8)
    # Byte-level decode table: entry[b] = int32 packing two bf16 bit patterns
    # (low 16 bits = bf16(low nibble), high 16 bits = bf16(high nibble)).
    # One int32 gather per byte + view_as — no float LUT, no arithmetic;
    # ~2.2-2.9x faster than the nibble-LUT version and bit-exact with it.
    _b = torch.arange(256, dtype=torch.uint8)
    _lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    _lo = _lut[(_b & 15).long()].to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    _hi = (_lut[(_b >> 4).long()].to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF) << 16
    codes = (_lo | _hi)[data.long()]  # [out, in/2] int32
    values = codes.view(torch.int16).unflatten(-1, (data.shape[1], 2)).view(torch.bfloat16).reshape(data.shape[0], -1)
    assert tuple(scale.shape) == (values.shape[0], values.shape[1] // 32)  # nosec
    return (values.reshape(*scale.shape, 32) * scale.unsqueeze(-1)).reshape_as(values)


def write_shard(path, tensors):
    """Stream contiguous CPU tensors without a second shard-sized serialization copy."""
    header, offset = {'__metadata__': {'format': 'pt'}}, 0
    for k, v in tensors.items():
        nbytes = v.numel() * v.element_size()
        header[k] = {'dtype': DTYPE_NAMES[v.dtype], 'shape': list(v.shape), 'data_offsets': [offset, offset + nbytes]}
        offset += nbytes
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    temp = Path(str(path) + '.partial')
    digest = hashlib.sha256()
    with temp.open('wb') as f:
        for data in (struct.pack('<Q', len(raw)), raw):
            f.write(data)
            digest.update(data)
        for v in tensors.values():
            data = memoryview(v.contiguous().view(torch.uint8).numpy()).cast('B')
            f.write(data)
            digest.update(data)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)
    return {
        'sha256': digest.hexdigest(),
        'payload_bytes': offset,
        'file_bytes': Path(path).stat().st_size,
        'tensors': {k: v for k, v in header.items() if k != '__metadata__'},
    }


def engram_source_description(tensors):
    """Describe stored Engram tensors for ``optional.embedding_storage``.

    Optional: only emitted when the model adapter config
    (``spec.model_adapter.export_embedding_storage``) asks for it; the primary
    per-table description is ``layers.N.engram.embed.quant_type``.
    """
    modules, expected = {}, set()
    for key, spec in tensors.items():
        if not key.endswith('.engram.embed.weight'):
            continue
        prefix = key.removesuffix('.weight')
        shape = spec['shape']
        if spec['dtype'] not in ('BF16', 'I8') or len(shape) != 2 or shape[1] != 256:
            raise ValueError(f'{key}: expected BF16 or INT8 [rows, 256]')
        module = dict(
            format='bf16' if spec['dtype'] == 'BF16' else 'int8_sym',
            weight_dtype=spec['dtype'],
            weight_shape=list(shape),
            layout='row_major',
        )
        expected.add(key)
        if spec['dtype'] == 'I8':
            scale_key = prefix + '.scale'
            scale = tensors.get(scale_key, {})
            if scale.get('dtype') != ENGRAM_SCALE_DTYPE or scale.get('shape') != [shape[0], 8]:
                raise ValueError(f'{scale_key}: expected {ENGRAM_SCALE_DTYPE} [rows, 8]')
            expected.add(scale_key)
            module.update(
                group_size=32,
                axis=1,
                scale_suffix='scale',
                scale_dtype=scale['dtype'],
                scale_shape=list(scale['shape']),
            )
        modules[prefix] = module
    actual = {key for key in tensors if '.engram.embed.' in key}
    if actual != expected:
        raise ValueError('Unexpected Engram tensors: legacy parts or scales')
    return {'version': 2, 'modules': modules}


def _e8m0_scale(grouped: torch.Tensor) -> torch.Tensor:
    """Per-group int8 scale, a power of two, rounded UP.

    ``amax / 127`` is the smallest scale the group's int8 range needs.  E8M0
    stores powers of two only, so take the exponent up: rounding to nearest can
    land below what the group needs and clip its largest value, and on a table
    this size that is a silently wrong row rather than a rounding error.  Same
    direction the calibration kernel rounds its ue8m0 scales.

    ``clamp_min`` keeps an all-zero group at a representable exponent instead of
    ``-inf``; both passes call this, so what gets quantized and what gets stored
    are the same number by construction.
    """
    exponent = torch.ceil(torch.log2((grouped.abs().amax(dim=-1) / 127.0).clamp_min(1e-30)))
    assert exponent.max().item() <= 127, 'engram scale overflows E8M0'  # nosec
    return torch.exp2(exponent)


def _decode_engram_rows(w_block, s_block) -> torch.Tensor:
    """Decode one FP8 (E4M3 + E8M0 scale) or BF16 Engram row block to BF16."""
    dtype = str(w_block.dtype)
    if dtype in ('torch.float8_e4m3fn', 'torch.int8'):
        return (w_block.float().reshape(-1, 8, 32) * s_block.float().unsqueeze(-1)).reshape(-1, 256).bfloat16()
    return w_block.bfloat16()


def write_engram_int8(path, model_path, key, group=32, block_rows=1 << 20):
    """Write one int8 table plus its per-group scale, streaming row-blocks.

    The source stores MXFP8 -- E4M3 against an E8M0 scale per ``group`` values --
    so each row-block is decoded to BF16, re-quantized symmetrically to int8 over
    the same groups, and its bytes streamed straight to the file via
    :class:`~msmodelslim.model.common.tensor_stream.TensorStreamProcessor`.  Only one
    block's decode and the output scale bytes are held in memory at once, instead
    of the whole ~100GB table.  ``tqdm`` reports progress over rows.

    Out comes the pair the serving contract describes for ``int8_sym``: I8
    [rows, width] and an ``ENGRAM_SCALE_DTYPE`` [rows, width // group] scale --
    the same shared exponent the source carries, one byte per group, which keeps
    the export under half the bytes of the bf16 table.
    """
    shape, _ = tensor_spec(model_path, key)
    rows, width = shape
    assert width % group == 0, f'{key}: width {width} not divisible by group {group}'  # nosec
    groups = width // group
    scale_key = key.removesuffix('.weight') + '.scale'
    root = Path(model_path)
    index = checkpoint_index(root)
    inputs = [TensorInput(root / index[key], key, shape)]
    has_scale = scale_key in index
    if has_scale:
        inputs.append(TensorInput(root / index[scale_key], scale_key, (rows, groups)))

    def quant(w_block, s_block=None):
        values = _decode_engram_rows(w_block, s_block).float()
        assert torch.isfinite(values).all().item(), key  # nosec
        grouped = values.view(-1, groups, group)
        scale = _e8m0_scale(grouped)
        quant = (grouped / scale.clamp_min(1e-30).unsqueeze(-1)).round()
        weights = quant.clamp_(-127, 127).to(torch.int8).contiguous()
        scales = scale.to(DTYPES[ENGRAM_SCALE_DTYPE]).contiguous()
        return [weights, scales]

    outputs = [
        TensorOutput(path, key, 'I8', shape),
        TensorOutput(path, scale_key, ENGRAM_SCALE_DTYPE, (rows, groups)),
    ]
    result = TensorStreamProcessor(block_rows=block_rows).process(inputs, outputs, quant, desc=key)
    entry = result[str(Path(path))]
    return {
        'sha256': entry['sha256'],
        'payload_bytes': rows * width + rows * groups,
        'file_bytes': Path(path).stat().st_size,
        'tensors': entry['tensors'],
    }


def write_engram_bf16(path, model_path, key, block_rows=1 << 20):
    """Write one original-name BF16 table, streaming row-blocks.

    The source table is decoded block by block through row slices and re-emitted
    under the same name via :class:`~msmodelslim.model.common.tensor_stream.TensorStreamProcessor`,
    so only one block is in memory at once instead of the whole (potentially
    ~200GB) BF16 table.  ``tqdm`` reports progress over rows.
    """
    shape, _ = tensor_spec(model_path, key)
    rows, width = shape
    scale_key = key.removesuffix('.weight') + '.scale'
    root = Path(model_path)
    index = checkpoint_index(root)
    inputs = [TensorInput(root / index[key], key, shape)]
    has_scale = scale_key in index
    if has_scale:
        inputs.append(TensorInput(root / index[scale_key], scale_key, (rows, width // 32)))

    def passthrough(w_block, s_block=None):
        values = _decode_engram_rows(w_block, s_block).contiguous()
        assert values.dtype == torch.bfloat16 and torch.isfinite(values).all().item(), key  # nosec
        return [values]

    outputs = [TensorOutput(path, key, 'BF16', shape)]
    result = TensorStreamProcessor(block_rows=block_rows).process(inputs, outputs, passthrough, desc=key)
    entry = result[str(Path(path))]
    return {
        'sha256': entry['sha256'],
        'payload_bytes': math.prod(shape) * 2,
        'file_bytes': Path(path).stat().st_size,
        'tensors': entry['tensors'],
    }
