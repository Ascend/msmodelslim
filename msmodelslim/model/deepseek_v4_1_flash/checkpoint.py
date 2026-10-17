"""DeepSeek-V4.1-Flash checkpoint reading and Ascend weight export."""

import hashlib
import json
import math
import os
from pathlib import Path
import struct

import torch
from safetensors import safe_open

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
        'file_bytes': path.stat().st_size,
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


def write_engram_int8(path, model_path, key, group=32):
    """Write one int8 table plus its per-group scale.

    The source stores MXFP8 -- E4M3 against an E8M0 scale per ``group`` values --
    so the table is dequantized first (loaded whole through safe_open), then
    re-quantized symmetrically to int8 over the same groups.  Out comes the pair
    the serving contract describes for ``int8_sym``: I8 [rows, width] and an
    ``ENGRAM_SCALE_DTYPE`` [rows, width // group] scale -- the same shared
    exponent the source carries, one byte per group, which keeps the export
    under half the bytes of the bf16 table.

    The table is read, dequantized and re-quantized in full; safetensors wants
    each tensor's bytes contiguous, so the int8 weights are written first, then
    their scales.
    """
    shape, _ = tensor_spec(model_path, key)
    rows, width = shape
    assert width % group == 0, f'{key}: width {width} not divisible by group {group}'  # nosec
    groups = width // group
    scale_key = key.removesuffix('.weight') + '.scale'
    weight_bytes = rows * width * 1  # int8
    scale_bytes = rows * groups * 1  # E8M0, one byte per group
    header = {
        '__metadata__': {'format': 'pt'},
        key: {'dtype': 'I8', 'shape': [rows, width], 'data_offsets': [0, weight_bytes]},
        scale_key: {
            'dtype': ENGRAM_SCALE_DTYPE,
            'shape': [rows, groups],
            'data_offsets': [weight_bytes, weight_bytes + scale_bytes],
        },
    }
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    temp = Path(str(path) + '.partial')
    digest = hashlib.sha256()
    values = decode_tensor(model_path, key).float()
    assert torch.isfinite(values).all().item(), key  # nosec
    grouped = values.view(-1, groups, group)
    scale = _e8m0_scale(grouped)
    quant = (grouped / scale.clamp_min(1e-30).unsqueeze(-1)).round()
    weights = quant.clamp_(-127, 127).to(torch.int8).contiguous()
    # The exponent byte itself; ``view(uint8)`` is a bit cast, and every value
    # here is an exact power of two, so it round-trips exactly.
    scales = scale.to(DTYPES[ENGRAM_SCALE_DTYPE]).contiguous()
    with temp.open('wb') as stream:
        for data in (struct.pack('<Q', len(raw)), raw):
            stream.write(data)
            digest.update(data)
        for data in (weights.view(torch.uint8).numpy().tobytes(), scales.view(torch.uint8).numpy().tobytes()):
            stream.write(data)
            digest.update(data)
        stream.flush()
        os.fsync(stream.fileno())
    assert temp.stat().st_size == 8 + len(raw) + weight_bytes + scale_bytes  # nosec
    temp.replace(path)
    return {
        'sha256': digest.hexdigest(),
        'payload_bytes': weight_bytes + scale_bytes,
        'file_bytes': path.stat().st_size,
        'tensors': {key: header[key], scale_key: header[scale_key]},
    }


def write_engram_bf16(path, model_path, key):
    """Write one original-name BF16 table.

    The table is read in full through safe_open and re-emitted under the same
    name.  If an identical share already exists under $ENGRAM_SHARE_DIR (keyed
    by tensor name), symlink to it instead of re-streaming the (potentially
    ~200GB) BF16 table.  Engram memory tables are written unrotated and
    byte-for-byte identical across exports of the same source, so the symlink
    is exact.
    """
    shape, _ = tensor_spec(model_path, key)
    payload = math.prod(shape) * 2
    spec = {'dtype': 'BF16', 'shape': shape, 'data_offsets': [0, payload]}
    header = {'__metadata__': {'format': 'pt'}, key: spec}
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    temp = Path(str(path) + '.partial')
    digest = hashlib.sha256()
    values = decode_tensor(model_path, key).contiguous()
    assert values.dtype == torch.bfloat16 and torch.isfinite(values).all().item(), key  # nosec
    with temp.open('wb') as stream:
        for data in (struct.pack('<Q', len(raw)), raw):
            stream.write(data)
            digest.update(data)
        data = values.view(torch.uint8).numpy().tobytes()
        stream.write(data)
        digest.update(data)
        stream.flush()
        os.fsync(stream.fileno())
    assert temp.stat().st_size == 8 + len(raw) + payload  # nosec
    temp.replace(path)
    return {
        'sha256': digest.hexdigest(),
        'payload_bytes': payload,
        'file_bytes': path.stat().st_size,
        'tensors': {key: spec},
    }
