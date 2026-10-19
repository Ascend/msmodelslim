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

Stream large safetensors tensors in row-blocks through a custom function.

Typical use: a huge tensor (or several) is read one row-block at a time, each
block is transformed independently, and the results are written back to one or
more safetensors files.  Only one block's read + the trailing outputs' bytes are
held in memory at once, instead of the whole (often ~100GB) tensor.  Reads are
prefetched and IO runs on a thread executor, so the next block's disk read
overlaps the current block's compute and write.
"""

import asyncio
import hashlib
import json
import math
import os
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import torch
from safetensors import safe_open
from tqdm import tqdm


@dataclass
class TensorInput:
    """One source tensor to stream rows from."""

    shard: Union[str, Path]  # safetensors file holding the tensor
    key: str  # tensor key
    shape: Tuple[int, ...]  # (rows, *rest); the first dim is streamed


@dataclass
class TensorOutput:
    """One destination tensor to stream rows into."""

    shard: Union[str, Path]  # output safetensors file (one or more outputs per file)
    key: str  # tensor key
    dtype: str  # safetensors dtype name, e.g. 'I8', 'BF16', 'F8_E8M0', 'F32'
    shape: Tuple[int, ...]  # (rows, *rest); must equal the input rows count


#: Bytes per element for the safetensors dtype names this utility emits.
_DTYPE_BYTES = {
    'I8': 1,
    'U8': 1,
    'F8_E4M3': 1,
    'F8_E8M0': 1,
    'BF16': 2,
    'F16': 2,
    'I16': 2,
    'U16': 2,
    'F32': 4,
    'I32': 4,
    'U32': 4,
    'F64': 8,
    'I64': 8,
    'U64': 8,
}


class TensorStreamProcessor:
    """Row-block streaming processor over safetensors.

    ``process`` opens each input's row slice, walks the rows in ``block_rows``
    chunks, calls ``fn(*block_slices) -> [block_tensors]`` (one per output), and
    streams the results into the output files.  Outputs that share a file are
    laid out contiguously in the given order; the first such output is written
    block-by-block as it is produced, later ones are buffered and flushed at the
    end (safetensors requires each tensor's bytes contiguous).  ``tqdm`` reports
    row progress.  Returns per-output-file ``{'sha256': ..., 'tensors': {...}}``.
    """

    def __init__(self, block_rows: int = 1 << 20, prefetch: int = 2, max_workers: int = 4):
        self.block_rows = block_rows
        self.prefetch = prefetch
        self.max_workers = max_workers

    def process(
        self,
        inputs: List[TensorInput],
        outputs: List[TensorOutput],
        fn: Callable,
        desc: str = 'stream',
    ) -> Dict[str, Dict]:
        """Run the streaming pipeline to completion.

        Returns ``{shard: {'sha256': ..., 'tensors': {key: {'dtype', 'shape',
        'data_offsets'}}}}`` so callers can build weight maps / descriptions
        from exactly what was written.
        """
        return asyncio.run(self._process(inputs, outputs, fn, desc))

    # ------------------------------------------------------------------ setup

    def _group_outputs(self, outputs: List[TensorOutput]) -> Dict[Path, Dict]:
        """Group outputs per file and compute each file's safetensors header.

        The header bytes are held in ``raw`` (written once the stream opens);
        ``buffers`` is a bytearray per output after the first, written
        contiguously at the end so each tensor stays contiguous.
        """
        groups: Dict[Path, Dict] = {}
        for i, out in enumerate(outputs):
            groups.setdefault(Path(out.shard), {}).setdefault('indices', []).append(i)
        for path, group in groups.items():
            indices = sorted(group['indices'])
            header = {'__metadata__': {'format': 'pt'}}
            specs: Dict[str, Dict] = {}
            offset = 0
            for i in indices:
                out = outputs[i]
                nbytes = math.prod(out.shape) * _DTYPE_BYTES[out.dtype]
                spec = {'dtype': out.dtype, 'shape': list(out.shape), 'data_offsets': [offset, offset + nbytes]}
                header[out.key] = spec
                specs[out.key] = spec
                offset += nbytes
            raw = json.dumps(header, separators=(',', ':')).encode()
            raw += b' ' * (-len(raw) % 8)
            group.update(
                temp=Path(str(path) + '.partial'),
                raw=raw,
                digest=hashlib.sha256(),
                specs=specs,
                first=indices[0],
                buffers={i: bytearray() for i in indices[1:]},
            )
        return groups

    def _write_header(self, group) -> None:
        for data in (struct.pack('<Q', len(group['raw'])), group['raw']):
            group['stream'].write(data)
            group['digest'].update(data)

    async def _process(self, inputs, outputs, fn, desc) -> Dict[str, Dict]:
        rows = inputs[0].shape[0]
        for inp in inputs:
            if inp.shape[0] != rows:
                raise ValueError(f'inputs must share the row count: {inp.key} {inp.shape[0]} != {rows}')
        for out in outputs:
            if out.shape[0] != rows:
                raise ValueError(f'outputs must share the row count: {out.key} {out.shape[0]} != {rows}')

        with ExitStack() as stack:
            slices = []
            for inp in inputs:
                fh = stack.enter_context(safe_open(str(inp.shard), framework='pt'))
                slices.append(fh.get_slice(inp.key))
            groups = self._group_outputs(outputs)
            for group in groups.values():
                group['stream'] = stack.enter_context(group['temp'].open('wb'))
                self._write_header(group)
            await self._stream(slices, outputs, groups, fn, rows, desc)
            for group in groups.values():
                for i in sorted(group['buffers']):
                    data = bytes(group['buffers'][i])
                    group['stream'].write(data)
                    group['digest'].update(data)
                group['stream'].flush()
                os.fsync(group['stream'].fileno())
            for group in groups.values():
                group['temp'].replace(group['temp'].with_suffix(''))
        return {
            str(path): {'sha256': group['digest'].hexdigest(), 'tensors': dict(group['specs'])}
            for path, group in groups.items()
        }

    # ---------------------------------------------------------------- streaming

    async def _stream(self, slices, outputs, groups, fn, rows, desc):
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.prefetch)

        async def producer():
            for start in range(0, rows, self.block_rows):
                end = min(start + self.block_rows, rows)
                blocks = await asyncio.to_thread(self._read_blocks, slices, start, end)
                await queue.put((start, end, blocks))
            await queue.put(None)

        prod = asyncio.create_task(producer())
        with tqdm(total=rows, unit='row', desc=desc) as pbar:
            while True:
                item = await queue.get()
                if item is None:
                    break
                start, end, inblocks = item
                outblocks = await self._run_fn(fn, inblocks)
                await asyncio.to_thread(self._write_blocks, outputs, groups, outblocks)
                pbar.update(end - start)
        await prod

    def _read_blocks(self, slices, start, end) -> List:
        return [sl[start:end] for sl in slices]

    async def _run_fn(self, fn, inblocks) -> List:
        result = fn(*inblocks)
        if asyncio.iscoroutine(result):
            result = await result
        return list(result)

    def _write_blocks(self, outputs, groups, outblocks) -> None:
        """Write one output row-block per output: stream the first tensor of each
        file (its byte range is reached in row order), buffer the rest until the
        end so each tensor stays contiguous.
        """
        for i, block in enumerate(outblocks):
            group = groups[Path(outputs[i].shard)]
            data = block.contiguous().view(torch.uint8).numpy().tobytes()
            if i == group['first']:
                group['stream'].write(data)
                group['digest'].update(data)
            else:
                group['buffers'][i].extend(data)
