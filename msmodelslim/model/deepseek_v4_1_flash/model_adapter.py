# -*- coding: UTF-8 -*-
# pylint: disable=C0302,R0901,E1102  # large adapter, 13-mixin hierarchy, torch dynamic callables
"""DeepSeek-V4.1-Flash ModelAdapter — drives its quantization through the standard
``msmodelslim quant`` pipeline instead of the legacy hand-rolled converter.

The standard pipeline applies:
1. ``quarot``  processor — norm folding + QuaRot rotation, driven by this
   adapter's :class:`QuaRotInterface` implementation (``get_rotate_map`` /
   ``get_ln_fuse_map``).  The rotation matrices come from the same
   ``create_rot`` helper the legacy ``checkpoint.Converter`` used, so the
   exported ``optional/quarot`` matrices match the serving contract.
2. ``linear_quant`` / ``flex_awq_ssz`` processors — bit-width selection via the
   best-practice recipes.

Quant settings
--------------
* The best-practice recipes select which linears get quantized (W8A8 : W4A8 via
  ``linear_quant`` ``include``); this adapter only answers which tensor each
  module is and how it rotates.
* Data-free paths (per-token activation + per-channel weight minmax/ssz) run
  through ``generate_model_visit`` only — no forward is required.
* Calibration-based processors (``flex_awq_ssz``) need a live text backbone
  forward.  The checkpoint ships ``inference/model.py``; wiring it into
  ``generate_model_forward`` is the documented extension point.
"""

import collections
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
from pydantic import BaseModel, Field
from torch import nn

if TYPE_CHECKING:
    pass

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.core.quant_service.modelslim_v1.save.interface import AscendV1SaveInterface
from msmodelslim.model.deepseek_v4_1_flash.inference.model import (
    RotationPlan,
    RotationScheme,
)
from msmodelslim.model.deepseek_v4_1_flash.checkpoint import (
    ENGRAM_ROTATION_CONFIG,
    checkpoint_index,
    decode_tensor,
    engram_source_description,
    tensor_spec,
    write_engram_bf16,
    write_engram_int8,
    write_json,
    write_shard,
)
from msmodelslim.model.deepseek_v4_1_flash.inference import loader as dsv41_reference
from msmodelslim.model.common.pausable_forward import (
    PointConfig,
    pausable_forward,
)
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.interface_hub import (
    FakeQuantInferenceInterface,
    FlexSmoothQuantInterface,
    IterSmoothInterface,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    QuaRotInterface,
)
from msmodelslim.model.common.utils import resolve_expert_ep_range
from msmodelslim.utils.exception import SchemaValidateError, UnsupportedError
from msmodelslim.utils.logging import get_logger, logger_setter


class DeepSeekV41FlashModelConfig(BaseModel):
    """Control parameters declared in the practice ``spec.model_adapter``.

    Provided by ``DeepSeekV41FlashAdapterLoader.get_adapter_config_class`` and
    validated uniformly during the quantization pre-check; the adapter applies
    it via :meth:`apply_adapter_config`.
    """

    engram_quant_type: Literal['int8_sym', 'bf16'] = Field(
        default='int8_sym',
        description="Engram 记忆表导出格式：`int8_sym` 写 INT8 权重 + E8M0 每组 scale（体积减半）；"
        "`bf16` 原样写 BF16。",
    )
    dspark: bool = Field(
        default=True,
        description="推理/量化路径是否运行 DSpark 草稿。True（默认）运行草稿块；False 关闭。",
    )
    export_embedding_storage: bool = Field(
        default=False,
        description="是否在 quant_model_description.json 的 optional.embedding_storage 里附带 "
        "Engram 表存储描述（version 2 + 每表 format/scale 说明）。默认不导出——逐层 "
        "layers.N.engram.embed.quant_type 已是主描述；需要该块时置 True。",
    )


@logger_setter('msmodelslim.model.deepseek_v4_1_flash')
def _install_detached_weight(module, weight):
    """Bind ``weight`` as the module's OWN parameter (own storage, own object)."""
    data = weight.detach().to(device="cpu", dtype=module.weight.dtype).contiguous().clone()
    module.weight = torch.nn.Parameter(data)


def _separate_draft_leaves(model, embed_weight, head_weight):
    """Give the draft its own UNROTATED embed/head.

    The reference checkpoint shares both with the backbone (``Transformer``:
    ``mtp[i].embed = model.embed`` / ``mtp[i].head = model.head``), which is
    sound only while nothing transforms those leaves.  Under QuaRot the shared
    copies are rotated -- and the head additionally carries the backbone's
    final-norm diagonal, fused by ``get_ln_fuse_map``'s ``{'norm': ['head']}``
    entry -- while the draft blocks stay in the ORIGINAL basis (only
    ``mtp.0.main_proj`` is rotated, and only to cancel the rotated
    ``main_hidden``).  A draft that kept the shared leaves would have to undo
    both transforms on every forward; taking its own pre-rotation copies removes
    that boundary entirely, so the draft path needs no basis conversion at all.

    Only the first stage embeds (``forward_embed``) and only the last one heads
    (``forward_head``); the middle blocks touch neither, so they drop both shared
    references instead of carrying inert aliases (the reference export keeps no
    ``mtp.<middle>.embed`` / ``mtp.<middle>.head``).  Same split as
    ``deepseek_v4_dspark``'s stage0 / last-stage attach.
    """
    if not len(model.mtp):
        return
    vocab, dim = model.embed.vocab_size, model.embed.dim
    head_cls = type(model.head)
    head_attrs = (model.head.vocab_size, model.head.dim, model.head.norm_eps, model.head.hc_eps)
    first, last = model.mtp[0], model.mtp[-1]
    first.embed = type(model.embed)(vocab, dim)
    _install_detached_weight(first.embed, embed_weight)
    last.head = head_cls(*head_attrs)
    _install_detached_weight(last.head, head_weight)
    if first is not last:
        # mtp.0 only embeds (forward_embed) and mtp.2 only heads
        # (forward_head): drop the counterpart shared references so the draft
        # carries no rotated leaf it never uses.  Middle blocks never enter the
        # embed/head phases, so they drop both.
        first.head = None
        last.embed = None
        for block in model.mtp[1:-1]:
            block.embed = None
            block.head = None
    get_logger().info(
        'DeepSeek-V4.1-Flash: draft detached from the shared embed/head (mtp.0.embed + mtp.%d.head, unrotated copies)',
        len(model.mtp) - 1,
    )


def _load_dedicated_draft_leaves(model_path):
    """Read the draft's own (pre-rotation) embed/head out of an export.

    A shared leaf is memoised under whichever ``mtp.<i>`` block first referenced
    it, so BOTH names involved -- the one carrying the shared copy and the one
    carrying the draft's private copy -- depend on the saver's walk order, and
    the distributed merge reshuffles it.  So do not assume an index and do not
    take the first match: of the candidates, the dedicated copy is the one that
    does *not* match the shared leaf.
    """
    import re as _re

    root = Path(model_path)
    index = root / 'quant_model_weights.safetensors.index.json'
    if not index.is_file():
        return None
    weight_map = json.loads(index.read_text()).get('weight_map', {})
    embed_keys = [k for k in weight_map if _re.fullmatch(r'mtp\.\d+\.embed\.weight', k)]
    head_keys = [k for k in weight_map if _re.fullmatch(r'mtp\.\d+\.head\.weight', k)]
    if not embed_keys or not head_keys:
        return None
    try:
        from safetensors.torch import load_file as _load

        def _read(key):
            return _load(str(root / weight_map[key]))[key]

        def _dedicated(keys, shared_key):
            """The candidate that differs from the shared leaf, else None.

            None means every candidate IS the shared copy -- an export written
            before the draft had its own leaves.  Handing that to the draft as
            if it were pre-rotation silently produces a wrong draft, so the
            caller turns it into an error rather than a fallback.
            """
            shared = _read(shared_key) if shared_key in weight_map else None
            for key in keys:
                value = _read(key)
                if shared is None or tuple(value.shape) != tuple(shared.shape):
                    return key
                if float((value.float() - shared.float()).abs().mean()) >= 1e-4:
                    return key
            return None

        embed_key = _dedicated(embed_keys, 'embed.weight')
        head_key = _dedicated(head_keys, 'head.weight')
        if embed_key is None or head_key is None:
            return 'stale'
        return _read(embed_key), _read(head_key)
    except Exception:  # a malformed export must not abort a load
        return None


def _install_param(module: nn.Module, dotted: str, value: torch.Tensor) -> None:
    """Replace a (meta) parameter in place, addressed by its dotted name.

    ``get_submodule`` walks to the owning module and the new Parameter takes the
    placeholder's slot, so every reference to it -- including one an MTP block
    holds to a shared leaf -- sees the real tensor.
    """
    parent_name, _, leaf = dotted.rpartition('.')
    parent = module.get_submodule(parent_name) if parent_name else module
    parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)


_CALIB_FORWARD_LOGGED: set = set()


def _log_calibration_basis_once(native) -> None:
    """Say once, at INFO, which basis and engram switches the calibration forward runs with.

    The settings live in ``Transformer.rotation_plan`` and ``engram_hash``, which the
    scheduling edits from two directions -- the adapter sets the plan when quarot runs,
    and the model builds the hash state -- so "what was actually in force" is not
    readable from either site alone.  Printing it at the point of use is the evidence.
    """
    key = id(native)
    if key in _CALIB_FORWARD_LOGGED:
        return
    _CALIB_FORWARD_LOGGED.add(key)
    plan = getattr(native, 'rotation_plan', None)
    gate = getattr(plan, 'engram_gate_basis', None)
    get_logger().info(
        'DeepSeek-V4.1-Flash calibration forward: engram injection %s | engram gate basis %s | rotation scheme %s',
        'ON' if getattr(native, 'engram_hash', None) is not None else 'OFF',
        (
            'inverse-rotates by global_rotation (original basis)'
            if gate is not None
            else 'NONE -- the gate reads the stream as-is'
        ),
        getattr(plan, 'scheme', None),
    )


class DeepSeekV41FlashModelAdapter(
    BaseModelAdapter,
    ModelInfoInterface,
    ModelSlimPipelineInterfaceV1,
    AscendV1SaveInterface,
    QuaRotInterface,
    FlexSmoothQuantInterface,
    IterSmoothInterface,
    FakeQuantInferenceInterface,
):
    """Adapter for the internally-derived DeepSeek-V4.1-Flash checkpoint layout."""

    def __init__(self, model_type: str, model_path: str, trust_remote_code: bool = False):
        super().__init__(model_type=model_type, model_path=model_path, trust_remote_code=trust_remote_code)
        self._source_keys: Optional[set] = None
        self._model: Optional[nn.Module] = None
        self._engram_quant_type: Literal['int8_sym', 'bf16'] = 'int8_sym'
        self._dspark: bool = True
        self._export_embedding_storage: bool = False
        #: Cached ``named_parameters()`` names of the built model; see _in_model.
        self._model_param_names: Optional[set] = None
        #: Shell / device state filled by ``init_model``.
        self._native_shell: Optional[nn.Module] = None
        self._model_device: Optional[str] = None
        self._blocks: Optional[list] = None

    def apply_adapter_config(self, config: DeepSeekV41FlashModelConfig) -> None:
        """Apply the validated ``spec.model_adapter`` control parameters."""
        self._engram_quant_type = config.engram_quant_type
        self._dspark = config.dspark
        self._export_embedding_storage = config.export_embedding_storage

    def _layer_counts(self):
        """(n_backbone, n_mtp) from the native shell, else from the config."""
        native = getattr(self, '_native_shell', None)
        if native is not None:
            return len(native.layers), len(native.mtp)
        cfg = dsv41_reference.read_model_config(self.model_path)
        return cfg['n_layers'], cfg['n_mtp_layers']

    def _expert_counts(self):
        """(backbone experts, draft experts) from the native shell, else the config.

        Same reason as :meth:`_layer_counts`: the maps below name expert modules
        by index, and a checkpoint whose expert count differs from the released
        shape would otherwise name experts this rank does not own.  Those sit in
        the ModuleList as ``None``, and ``get_submodule`` rejects a ``None`` atom.

        The released shapes happen to hide this: with 384 experts and 4 ranks a
        rank owns 0..95, which is exactly the set the source carries, so nothing
        names a foreign expert.  A reduced checkpoint does not line up like that.
        """
        native = getattr(self, '_native_shell', None)
        if native is not None and len(native.layers):
            mtp = native.mtp[0].ffn.n_routed_experts if len(native.mtp) else 0
            return native.layers[0].ffn.n_routed_experts, mtp
        cfg = dsv41_reference.read_model_config(self.model_path)
        return cfg['n_routed_experts'], cfg.get('dspark_n_routed_experts', 0)

    def _local_experts(self, n_experts: int) -> range:
        """The routed experts THIS rank owns.

        Under EP the experts are sharded contiguously and ``MoE`` builds only
        the local slots (the rest stay ``None``), so any map that enumerated
        ``range(n_experts)`` would name modules that do not exist here -- the
        consumer's ``get_submodule`` would then walk into a ``None``.  Every
        expert enumeration must go through this.  ``resolve_expert_ep_range``
        also rejects an expert count that the world size cannot divide.
        """
        _, _, start, end = resolve_expert_ep_range(n_experts)
        return range(start, end)

    def _in_model(self, name: str) -> bool:
        """True when the *built model* carries this parameter.

        The model, not the source layout.  Under EP a rank holds only its own
        experts, so a key can exist in the source for a module this rank's
        ``get_submodule`` cannot reach -- and the consumers walk the model, so
        the model is the only oracle that cannot disagree with them.

        Raises when no model has been built.  The previous answer to that case
        ("assume present") turned a declaration error into a silently smaller
        map, which is how a wrong target survived to the twentieth layer.
        """
        if self._model is None:
            raise RuntimeError(
                'the fuse/rotate/subgraph maps are checked against the model; build it with init_model() first'
            )
        if self._model_param_names is None:
            self._model_param_names = {n for n, _ in self._model.named_parameters()}
        names = self._model_param_names
        return (name in names) or (name + '.weight') in names or (name + '.bias') in names

    # ==================== FakeQuantInferenceInterface ====================
    @property
    def tokenizer(self):
        return _load_tokenizer(self.model_path)

    def build_meta_model(self) -> nn.Module:
        """Build the DeepSeek-V4.1-Flash meta skeleton for fake-quant inference.

        The shell IS the checkpoint's own graph (``inference/model.py``), built on
        ``meta`` with ``include_buffers=False``: parameters are placeholders the
        layer-wise runner materialises one block at a time, while buffers (RoPE
        freqs, window KV caches) stay real CPU tensors that ``RuntimeBufferStore``
        can snapshot across offloads.  Working on the checkpoint's graph rather
        than a re-modelled one is what keeps every module path identical to the
        checkpoint layout, so visit, quant and the calibration forward all act on
        the same instances.

        The artifact-specific wiring follows, one helper each: the Engram tables
        (:meth:`_mount_source_engram` / :meth:`_mount_export_engram`), the rotation
        basis a rotated export was written with (:meth:`_mount_rotation`), the
        class rename the AscendV1 format matches on, and the hash state
        ``forward`` needs on real memory.  ``generate`` and ``forward`` are the
        graph's own.
        """
        from msmodelslim.model.deepseek_v4_1_flash.inference import model as G

        dsv41_reference.install_calibration_runtime(G)
        G.default_dtype = torch.bfloat16
        # Geometry comes from the artifact's own config.json (read_model_config
        # resolves text_config / vision_config / rope_scaling / top-level fields;
        # no embedded config copy).
        # _build_model_args also bounds the shell to batch 1 / 2k context and
        # decoded BF16, with the experts unpacked to their logical [out, in].
        args = dsv41_reference._build_model_args(G, dsv41_reference.read_model_config(self.model_path), max_length=2048)
        prev_default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            from accelerate import init_empty_weights

            with init_empty_weights(include_buffers=False):
                # The tokenizer feeds NgramHashState (compressed token map + hash
                # multipliers); without it engram lookups cannot be computed.
                model = G.Transformer(args, tokenizer=_load_tokenizer(self.model_path))
        finally:
            torch.set_default_dtype(prev_default)

        if (Path(self.model_path) / 'model.safetensors.index.json').is_file():
            self._mount_source_engram(model, args)
        else:
            self._mount_export_engram(model, args)
            self._mount_rotation(model)
        # AscendV1's format loader recognises decoder layers by class-name suffix
        # (``decoder_prefixes``); the graph's own Block / DSparkBlock do not match.
        for block in list(model.layers) + list(model.mtp):
            block.__class__ = type('DeepSeekV41FlashDecoderLayer', (type(block),), {})
        # Buffers are not parameters, so ``include_buffers=False`` left them real
        # -- but the engine's contract is that they are on CPU (it snapshots them
        # across offloads), so put them there whatever the graph's default device.
        for module in model.modules():
            for key, buf in list(module._buffers.items()):
                if buf is not None and not buf.is_meta and buf.device.type != 'cpu':
                    module._buffers[key] = buf.cpu()
        self._rebuild_engram_hash(model, args)
        # The draft switch belongs to the adapter, not the checkpoint; it is
        # controlled by spec.model_adapter.dspark (default True).
        model.dspark = self._dspark
        model.eval()
        return model

    def _mount_source_engram(self, model: nn.Module, args) -> None:
        """Engram for a source checkpoint: memmap table + real projections.

        The table is never materialised -- 384M rows x 256 is ~98 GB in BF16 --
        so it becomes a read-only ``MappedEngram`` that decodes only the rows the
        current forward's hash ids hit.  The projections ARE real: ``forward``
        reads them before the decoder block's pre-hook fires, so they have to be
        in memory already or the offload-restore cache is never populated.
        """
        mounted = []
        for layer in model.layers:
            eng = getattr(layer, 'engram', None)
            if eng is None or layer.layer_id not in set(args.engram_layer_ids):
                continue
            base = f'layers.{layer.layer_id}.engram'
            eng.embed = dsv41_reference.MappedEngram(self.model_path, f'{base}.embed.weight', 'cpu')
            installed = []
            for pname, param in list(eng.named_parameters()):
                if param.is_meta and pname != 'embed.weight':
                    _install_param(eng, pname, decode_tensor(self.model_path, f'{base}.{pname}').to(param.dtype))
                    installed.append(pname)
            mounted.append(f'{base}[{",".join(installed) or "table only"}]')
        # The export path logs this; without the same line here, a quantization log
        # cannot be told apart from one where the engram was never mounted at all.
        get_logger().info(
            'DeepSeek-V4.1-Flash engram (source) mounted and INJECTED by the calibration forward for %d layer(s): %s',
            len(mounted),
            '; '.join(mounted) if mounted else 'none',
        )

    def _mount_export_engram(self, model: nn.Module, args) -> None:
        """Engram for an AscendV1 export: I8+scale table + the FLOAT projections.

        An export has no ``model.safetensors.index.json`` -- the AscendV1 loader
        hydrates its quantized tensors through the quant description instead, and
        its per-file size cap would reject a wholesale table hydrate anyway.
        """
        if not args.engram_layer_ids:
            return
        # The export's own text_config carries its engram layout: a reduced
        # export (or the mini debug checkpoint) has its own layer ids.
        export_cfg = Path(self.model_path) / 'config.json'
        if export_cfg.is_file():
            try:
                ids = json.loads(export_cfg.read_text()).get('text_config', {}).get('engram_layer_ids')
            except ValueError:
                ids = None
            if ids is not None:
                args.engram_layer_ids = tuple(ids)
        for layer in model.layers:
            eng = getattr(layer, 'engram', None)
            if eng is None or layer.layer_id not in set(args.engram_layer_ids):
                continue
            eng.embed = dsv41_reference.AscendV1EngramTable(Path(self.model_path), layer.layer_id)
            base = f'layers.{layer.layer_id}.engram.'
            tensors = self._engram_extra_tensors(layer.layer_id)
            for pname in ('wkv.weight', 'q_weight', 'k_weight'):
                if base + pname in tensors:
                    _install_param(eng, pname, tensors[base + pname].to('cpu'))
            get_logger().info(
                'DeepSeek-V4.1-Flash engram (export) mounted for layer %d (table + params)', layer.layer_id
            )

    def _engram_extra_tensors(self, layer_id: int) -> Dict[str, torch.Tensor]:
        """The FLOAT engram projections an export wrote for one layer.

        The saver puts them in a dedicated ``engram-extra`` file; an export made
        before that file existed left them in whichever shard carries them, so
        fall back to scanning the index for that layer's keys.
        """
        import safetensors.torch as st

        extra = Path(self.model_path) / f'quant_model_weights-engram-extra-{layer_id}.safetensors'
        if extra.is_file():
            return dict(st.load_file(str(extra)))
        index_file = Path(self.model_path) / 'quant_model_weights.safetensors.index.json'
        if not index_file.is_file():
            return {}
        base = f'layers.{layer_id}.engram.'
        shards = {
            shard
            for key, shard in json.loads(index_file.read_text()).get('weight_map', {}).items()
            if key.startswith(base)
        }
        tensors: Dict[str, torch.Tensor] = {}
        for shard in shards:
            for key, value in st.load_file(str(Path(self.model_path) / shard)).items():
                if key.startswith(base):
                    tensors.setdefault(key, value)
        return tensors

    def _mount_rotation(self, model: nn.Module) -> None:
        """Re-attach the rotation basis a rotated export was written with.

        ``optional/quarot.safetensors`` carries the global rotation, and its
        presence IS the scheme (see ``RotationScheme``).  Two consequences at the
        backbone/draft boundary: the Engram gate is defined in the original basis
        and so inverse-rotates the stream, and the draft needs its own
        pre-rotation embed/head because the shared ones are rotated and
        norm-fused.
        """
        rotation = Path(self.model_path) / 'optional' / 'quarot.safetensors'
        if not rotation.is_file():
            return
        import safetensors.torch as st

        model.rotation_plan = RotationPlan(RotationScheme.BACKBONE_ONLY, st.load_file(str(rotation))['global_rotation'])
        leaves = _load_dedicated_draft_leaves(self.model_path)
        if leaves == 'stale' or leaves is None:
            # Falling back to the shared leaves here would silently feed the
            # draft a rotated (and norm-fused) embed/head that it has no way to
            # undo -- a wrong draft, not a slow one.
            raise SchemaValidateError(
                'Rotated export carries no pre-rotation draft embed/head'
                + (
                    ' (its mtp.<i>.embed.weight is the rotated shared copy)'
                    if leaves == 'stale'
                    else ' (expected mtp.<i>.embed.weight and mtp.<i>.head.weight)'
                ),
                action='Re-export with the current save path, which writes the pre-rotation copies for the draft.',
            )
        _separate_draft_leaves(model, leaves[0], leaves[1])

    def _rebuild_engram_hash(self, model: nn.Module, args) -> None:
        """Put the Engram hash state on real memory.

        ``NgramHashState`` was constructed inside the ``init_empty_weights``
        context, so its buffers (token map, multipliers, primes, offsets) are
        meta placeholders -- while the forward hashes on CPU.
        """
        if getattr(model, 'engram_hash', None) is None:
            return
        from msmodelslim.model.deepseek_v4_1_flash.inference.engram import EngramLayout, NgramHashState

        model.engram_hash = NgramHashState(args, EngramLayout.from_args(args), _load_tokenizer(self.model_path))
        still_meta = [n for n, b in model.engram_hash.named_buffers() if b.is_meta]
        if still_meta:
            # A meta buffer hashes to nothing but zeros, which the forward cannot
            # tell apart from a real hash -- so say it here rather than there.
            get_logger().warning('DeepSeek-V4.1-Flash engram hash buffers still meta: %s', still_meta)

    # ==================== ModelInfoInterface ====================
    def get_model_pedigree(self) -> str:
        return 'deepseek_v4_1_flash'

    def get_model_type(self) -> str:
        return self.model_type

    # ==================== QuaRotInterface ====================
    def get_ln_fuse_map(self) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Norm -> linear folding map, mirroring the recipe's rotation rules."""

        fused: Dict[str, List[str]] = {'norm': ['head']}
        n_backbone, _ = self._layer_counts()
        n_experts = self._expert_counts()[0]
        owned = self._layer_optional_modules()
        # Draft blocks are not rotated, so their norms are not folded either.
        for i in range(n_backbone):
            pre = f'layers.{i}'
            # Same source of truth as the subgraph map: the optional attention
            # targets come from the config, not from listing everything and
            # filtering by what the checkpoint happens to carry.
            fused[f'{pre}.attn_norm'] = [f'{pre}.attn.wq_a', f'{pre}.attn.wkv'] + [
                f'{pre}.{s}' for s in self._OPTIONAL_ATTN_MODULES if s in owned[pre]
            ]
            fused[f'{pre}.attn.q_norm'] = [f'{pre}.attn.wq_b'] + (
                [f'{pre}.attn.indexer.wq_b'] if 'attn.indexer.wq_b' in owned[pre] else []
            )
            fused[f'{pre}.ffn_norm'] = (
                [f'{pre}.ffn.gate']
                + [f'{pre}.ffn.experts.{j}.{name}' for j in self._local_experts(n_experts) for name in ('w1', 'w3')]
                + [f'{pre}.ffn.shared_experts.{name}' for name in ('w1', 'w3')]
            )
        # Still filtered by the layout: a reduced checkpoint (the mini debug
        # build) legitimately lacks some of these, and this map is consumed by a
        # producer that tolerates absence.  The subgraph map is the strict one,
        # and it fails fast on a declaration the model cannot satisfy.
        fused = {k: [t for t in v if self._in_model(t)] for k, v in fused.items()}
        return {}, fused

    def get_bake_names(self) -> Tuple[List[str], List[str]]:
        # RMSNorm (no mean term) — nothing to bake.
        return [], []

    # ==================== subgraph interface (flex_smooth / flex_awq_ssz) ====
    def _text_config(self) -> dict:
        """The model's ``text_config``, or ``{}`` when it cannot be read."""
        cfg = Path(self.model_path) / 'config.json'
        if not cfg.is_file():
            return {}
        try:
            text = json.loads(cfg.read_text())
        except ValueError:
            return {}
        return text.get('text_config', {}) or {}

    def _source_layer_ids(self, key: str) -> set:
        """Layer ids a role list names, read from the model's own config.

        Training decides this per layer and records it: ``kv_source_layer_ids``
        are the layers that build a compressed KV, ``index_source_layer_ids``
        the ones that build an indexer.  A layer outside these lists has no
        compressor/indexer *by construction*, so the subgraph maps below can
        simply not name them -- no need to probe a checkpoint for their
        absence.

        Note ``compress_ratios`` does not answer this: 38 of 43 layers carry a
        non-zero ratio, but only the kv-source layers produce their own KV (the
        rest read it from the layer that does).  It answers a *different*
        question -- see :meth:`_compress_ratios`.
        """
        return set(self._text_config().get(key, ()) or ())

    #: Optional per-layer submodules, as (suffix, predicate over the config facts).
    #: Order is the declaration order the subgraph maps use.
    _OPTIONAL_ATTN_MODULES = ('attn.compressor.wkv', 'attn.compressor.wgate', 'attn.indexer.weights_proj')

    def _layer_optional_modules(self) -> Dict[str, set]:
        """Per layer, the optional submodules the CONFIG says that layer owns.

        Every one of these is decided by training and recorded in ``config.json``
        -- none is inferred from the checkpoint, and none should be re-derived
        from a class's ``__init__`` at the point of use.  Three maps (fuse,
        subgraph, rotate) need this; keeping one answer means they cannot drift,
        which is what put ``layers.20.attn.compressor.wgate`` into the subgraph
        map for a module the layer does not build.

        The conditions are a boolean combination, not a hierarchy, and the two
        gated facts read *different* fields:

          ``kv_source_layer_ids``    -> attn.compressor, attn.compressor.wkv
          ``compress_ratios[i] > 1`` -> that compressor also builds wgate
          ``index_source_layer_ids`` -> attn.indexer, .wq_b, .weights_proj
          ``kv_source_layer_ids``    -> that indexer also builds wk and k_norm
                                        (``Indexer.owns_k`` is the same list)
          ``engram_layer_ids``       -> engram

        So layer 20 -- a kv-source layer pooling 1 token -- has wkv, wk and
        k_norm but no wgate.

        Being listed here does NOT mean a map has to use it.  Three of these are
        deliberately absent from every map, and the data flow is why -- traced
        from ``Attention.forward`` / ``_window_kv`` / ``_compress_kv`` /
        ``Indexer.forward``, not inferred::

            x  <- residual stream (5120), already QuaRot-rotated
            |
            +- attn_norm(x) -+-> wq_a                  reads residual -> folded
            |                +-> wkv                   reads residual -> folded
            |                +-> compressor.wkv        reads residual -> folded
            |                +-> compressor.wgate      reads residual -> folded
            |                +-> indexer.weights_proj  reads residual -> folded
            |
            +- compressor(x):                        (_compress_kv)
            |    kv, score = wkv(x), wgate(x)
            |    latent = norm(pool(kv, score))      <- compressor.norm, RMSNorm(512)
            |       +-> indexer.wk(latent)           a Linear -- but see below
            |       +-> RoPE -> fp4 -> kv cache      NOT a linear
            |
            +- qr = q_norm(wq_a(x)) -+-> wq_b        reads q_lora   -> folded
            |                       +-> indexer.wq_b reads q_lora   -> folded
            |
            +- kv = kv_norm(wkv(x)) -> RoPE -> act_quant -> window kv cache
                                                     NOT a linear

        ``indexer.wk`` is the one that looks like a gap, and is not one:

        * not rotated.  Its input is ``latent``, the head_dim-wide space
          ``compressor.wkv`` produces -- not the residual stream.  ``rot_global``
          is 5120x5120 and cannot act on it.
        * not folded.  It does read a norm, but ``compressor.norm``'s output also
          feeds the compressed KV cache through RoPE and fp4 quantization.  A
          norm-linear fold divides that norm's weight by ``s``, and this second,
          non-linear consumer would then see a rescaled activation -- so the fold
          is unavailable here, not overlooked.
        * not quantized.  The recipe already excludes ``*indexer.wk``.

        ``indexer.k_norm``'s output goes straight into ``k_cache`` and then an
        einsum, and ``attn.kv_norm``'s into RoPE, quantization and attention:
        neither has a Linear downstream, so there is nothing to fold into them.
        """
        cfg = self._text_config()
        kv = set(cfg.get('kv_source_layer_ids', ()) or ())
        index = set(cfg.get('index_source_layer_ids', ()) or ())
        engram = set(cfg.get('engram_layer_ids', ()) or ())
        ratios = list(cfg.get('compress_ratios', ()) or ())
        n_backbone, n_mtp = self._layer_counts()

        owned: Dict[str, set] = {}
        for i in range(n_backbone):
            here: set = set()
            if i in engram:
                here.add('engram')
            if i in kv:
                here.update(('attn.compressor', 'attn.compressor.wkv'))
                if i < len(ratios) and ratios[i] > 1:
                    here.add('attn.compressor.wgate')
            if i in index:
                here.update(('attn.indexer', 'attn.indexer.wq_b', 'attn.indexer.weights_proj'))
                if i in kv:
                    # Indexer.owns_k: the indexer reuses the kv-source key.
                    here.update(('attn.indexer.wk', 'attn.indexer.k_norm'))
            owned[f'layers.{i}'] = here
        for i in range(n_mtp):
            owned[f'mtp.{i}'] = set()
        return owned

    def get_adapter_config_for_subgraph(self) -> List[Any]:
        """Declare the DeepSeek-V4.1-Flash subgraphs for FlexAWQSSZ / FlexSmoothQuant.

        Up-down smoothing follows the legacy PR#757 recipe: for every routed and
        shared expert the *up* projection is ``w3`` and the *down* projection
        ``w2`` (the ``w1`` gate is not part of the smoothed pair — it stays
        integer, its clamp is handled by the model forward).  Norm-linear
        configs mirror ``get_ln_fuse_map`` and are consumed only when a practice
        enables ``norm-linear``.

        A norm-linear target list is per layer, not per model: ``attn_norm``
        feeds the compressor and the indexer only on the layers that own them,
        so those targets are added by consulting the layer-type lists the config
        already carries.  Naming a module a layer does not have would only be
        discovered later, by a ``get_submodule`` failure or by silently dropping
        the whole subgraph.
        """
        from msmodelslim.core.graph import AdapterConfig, MappingConfig

        configs = []
        n_backbone, n_mtp = self._layer_counts()
        kv_sources = self._source_layer_ids('kv_source_layer_ids')
        index_sources = self._source_layer_ids('index_source_layer_ids')
        optional = self._layer_optional_modules()
        for prefix, n_layers, n_experts in (
            ('layers', n_backbone, self._expert_counts()[0]),
            ('mtp', n_mtp, self._expert_counts()[1]),
        ):
            for i in range(n_layers):
                pre = f'{prefix}.{i}'
                for j in self._local_experts(n_experts):
                    configs.append(
                        AdapterConfig(
                            subgraph_type='up-down',
                            mapping=MappingConfig(
                                source=f'{pre}.ffn.experts.{j}.w3', targets=[f'{pre}.ffn.experts.{j}.w2']
                            ),
                        )
                    )
                configs.append(
                    AdapterConfig(
                        subgraph_type='up-down',
                        mapping=MappingConfig(
                            source=f'{pre}.ffn.shared_experts.w3', targets=[f'{pre}.ffn.shared_experts.w2']
                        ),
                    )
                )
                # norm-linear (only active when the practice enables it)
                owns = optional[pre]
                attn_targets = [f'{pre}.attn.wq_a', f'{pre}.attn.wkv']
                attn_targets += [f'{pre}.{s}' for s in self._OPTIONAL_ATTN_MODULES if s in owns]
                configs.append(
                    AdapterConfig(
                        subgraph_type='norm-linear',
                        mapping=MappingConfig(source=f'{pre}.attn_norm', targets=attn_targets),
                    )
                )
                q_targets = [f'{pre}.attn.wq_b']
                if 'attn.indexer.wq_b' in owns:
                    q_targets.append(f'{pre}.attn.indexer.wq_b')
                configs.append(
                    AdapterConfig(
                        subgraph_type='norm-linear',
                        mapping=MappingConfig(source=f'{pre}.attn.q_norm', targets=q_targets),
                    )
                )
                configs.append(
                    AdapterConfig(
                        subgraph_type='norm-linear',
                        mapping=MappingConfig(
                            source=f'{pre}.ffn_norm',
                            targets=(
                                [f'{pre}.ffn.gate']
                                + [f'{pre}.ffn.experts.{j}.w3' for j in self._local_experts(n_experts)]
                                + [f'{pre}.ffn.shared_experts.w3']
                            ),
                        ),
                    )
                )

        # Fail fast instead of at layer N.  The DTS checks a dependency path
        # only when it submits that layer's task, so a target named for a module
        # the layer does not own is not discovered until the run reaches that
        # layer -- twenty layers and half an hour in, on this checkpoint, which
        # is what ``layers.20.attn.compressor.wgate`` cost.  The source layout
        # answers the same question now, so ask it once for the whole map.
        #
        # This is a check on the declaration, not the declaration itself: what
        # each layer should own still comes from the config's layer-type and
        # ratio lists, not from probing.
        if self._source_keys is not None:
            missing = sorted(
                {
                    path
                    for config in configs
                    for path in (
                        ([config.mapping.source] if config.mapping.source else []) + list(config.mapping.targets)
                    )
                    if not self._in_model(path)
                }
            )
            if missing:
                raise SchemaValidateError(
                    f'subgraph map names {len(missing)} path(s) the model does not have: {missing[:8]}',
                    action='Every declared target must exist on every layer it is '
                    'declared for.  Check the config layer-type and '
                    'compress_ratio lists against the model structure.',
                )

        by_type = collections.Counter(c.subgraph_type for c in configs)

        # 一层一个 dump，便于直接对着日志核对：前两层（是否带 compressor/indexer 由
        # 层型决定）与一层 MTP（不带）。
        def _fmt(c):
            return f'{c.subgraph_type}({c.mapping.source} -> {len(c.mapping.targets)} targets)'

        samples = []
        for want in ('layers.2.attn_norm', 'layers.0.attn_norm', 'mtp.0.attn_norm'):
            hit = next((c for c in configs if c.mapping.source == want), None)
            if hit is not None:
                samples.append(_fmt(hit))
        get_logger().info(
            'Subgraph map: %d entries %s | kv-source layers %s, index-source layers %s | %s',
            len(configs),
            dict(by_type),
            sorted(kv_sources),
            sorted(index_sources),
            '; '.join(samples),
        )
        return configs

    def get_rotate_map(self, block_size: int) -> Tuple[List[Any], List[Any]]:
        """``([pre_run_embed], [rot_global, rot_query])``.

        ``rot_global`` is the global HADAMARD rotation on hidden_size (5120,
        32-blocks) and ``rot_query`` the BLOCK_HADAMARD_SHIFTED rotation on
        q_lora_rank (1280).  Same ``create_rot`` the legacy converter used, so
        the exported matrices match the serving contract.
        """
        cfg = dsv41_reference.read_model_config(self.model_path)
        rot_global = QuaRotInterface.get_rotate_command(
            mode=QuaRotInterface.QuaRotMode.HADAMARD, size=cfg['dim'], block_size=block_size
        )
        rot_query = QuaRotInterface.get_rotate_command(
            mode=QuaRotInterface.QuaRotMode.BLOCK_HADAMARD_SHIFTED, size=cfg['q_lora_rank'], block_size=block_size
        )

        # Pre-run: rotate the embedding (first read).
        pre_run_pairs = [QuaRotInterface.RotatePair(left_rot={}, right_rot={'embed': rot_global})]

        right_global: Dict[str, torch.Tensor] = {'head': rot_global}
        left_global: Dict[str, torch.Tensor] = {'aligner.w2': rot_global}
        right_query: Dict[str, torch.Tensor] = {}
        left_query: Dict[str, torch.Tensor] = {}

        n_backbone, _ = self._layer_counts()
        # The draft (MTP) blocks stay in the original basis: only the backbone is
        # rotated, and mtp.0.main_proj is the single bridge that brings the
        # backbone hidden back for them (the legacy recipe's convention).
        owned = self._layer_optional_modules()
        for prefix, n_layers, n_experts in (('layers', n_backbone, self._expert_counts()[0]),):
            for i in range(n_layers):
                pre = f'{prefix}.{i}'
                here = owned[pre]
                right_global[pre + '.attn.wq_a'] = rot_global
                right_global[pre + '.attn.wkv'] = rot_global
                # Only the layers that own these gain them, same source of truth
                # as the subgraph and fuse maps.  Naming a path a layer does not
                # have was tolerated here (``_rotate`` falls back and then skips)
                # but not by the DTS, which is how the same slip cost a run.
                for suffix in self._OPTIONAL_ATTN_MODULES:
                    if suffix in here:
                        right_global[f'{pre}.{suffix}'] = rot_global
                right_global[pre + '.ffn.gate'] = rot_global
                # The hyper-connection mixes read the residual stream, so they
                # live in whatever basis it is in: leaving them unrotated while
                # every producer/consumer around them is rotated makes
                # hc_mixes() compute its coefficients off a basis mismatch.
                # They are raw (mix_hc, hc_mult * dim) parameters, not Linears,
                # so the rotate command reaches them through _rotate's
                # attribute fallback and rotate_linear expands Q over hc_mult.
                right_global[pre + '.hc_attn_fn'] = rot_global
                right_global[pre + '.hc_ffn_fn'] = rot_global
                right_global[pre + '.ffn.shared_experts.w1'] = rot_global
                right_global[pre + '.ffn.shared_experts.w3'] = rot_global
                left_global[pre + '.attn.wo_b'] = rot_global
                left_global[pre + '.ffn.shared_experts.w2'] = rot_global
                right_query[pre + '.attn.wq_b'] = rot_query
                if 'attn.indexer.wq_b' in here:
                    right_query[pre + '.attn.indexer.wq_b'] = rot_query
                left_query[pre + '.attn.wq_a'] = rot_query
                for j in self._local_experts(n_experts):
                    right_global[f'{pre}.ffn.experts.{j}.w1'] = rot_global
                    right_global[f'{pre}.ffn.experts.{j}.w3'] = rot_global
                    left_global[f'{pre}.ffn.experts.{j}.w2'] = rot_global
        right_global['mtp.0.main_proj'] = rot_global
        # Image token embeddings are added straight into the residual stream, so
        # they carry the same basis as everything else feeding it.
        for image_key in ('image_start', 'image_end', 'image_newline'):
            left_global[image_key] = rot_global
        # Engram WKV: row-block = wc (K/gating, first 20480 out rows) + wv (V,
        # last 5120).  Only the V block must land in the rotated residual basis,
        # so left-rotate the whole wkv with a block-diagonal matrix that leaves
        # the wc rows untouched: blkdiag(I_20480, rot_global).  No wc/wv split or
        # rebuild needed.
        # The engram layers come from the shell the model was actually built from:
        # the same ``engram_layout`` the engram mount reads, so the rotation and the
        # table can never disagree about which layers exist.
        layout = getattr(self._native_shell, 'engram_layout', None)
        engram_layer_ids = list(getattr(layout, 'layer_ids', ()) or ())
        engram_left: Dict[str, torch.Tensor] = {}
        for layer in engram_layer_ids:
            wkv_name = f'layers.{layer}.engram.wkv'
            if self._in_model(wkv_name):
                # wkv out rows = wc (hc_mult copies of hidden) + wv (1 hidden);
                # only the trailing wv block lands in the rotated basis.  Derive
                # the split shape-based (wv_out == rot_global.size), not hardcoded.
                wkv_out = tensor_spec(self.model_path, f'{wkv_name}.weight')[0][0]
                wv_rows = rot_global.size(0)
                wc_rows = wkv_out - wv_rows
                wkv_rot = torch.block_diag(
                    torch.eye(wc_rows, dtype=rot_global.dtype, device=rot_global.device),
                    rot_global,
                )
                engram_left[wkv_name] = wkv_rot
        if engram_left:
            # pre_run, not rotate_pairs: the Engram table is read from
            # Transformer.forward before any decoder-block hook, and the embedding
            # whose basis it must match is rotated in pre_run too, so both have to
            # land in the same phase.  Its own pair, appended after the embed one,
            # keeps pre_run_commands[0] -- the matrix the saver exports as
            # optional.quarot.global_rotation -- the hidden-size rotation.
            pre_run_pairs.append(QuaRotInterface.RotatePair(left_rot=engram_left, right_rot={}))
            # The gate dot stays in the original basis while the residual stream is
            # now rotated, so Engram.forward undoes the rotation for it.  Setting
            # it here ties the quantization forward and the AscendV1 eval to the
            # same condition (QuaRot ran) and so to one implementation.
            if self._model is not None:
                for layer in engram_layer_ids:
                    block = self._model.layers[layer] if 0 <= layer < len(self._model.layers) else None
                    eng = getattr(block, 'engram', None)
                    if eng is not None:
                        pass  # the engram gate reads the model-level switch
            # The draft blocks work in the original basis but borrow the rotated
            # embed/head; they need the same rotation to undo at their boundary.
            if self._model is not None:
                # Rotation is on from here: the draft blocks must stop borrowing
                # the (rotated, norm-fused) shared embed/head.
                _separate_draft_leaves(
                    self._model,
                    decode_tensor(self.model_path, 'embed.weight')
                    if self._source_keys is not None
                    else self._model.embed.weight.detach(),
                    decode_tensor(self.model_path, 'head.weight')
                    if self._source_keys is not None
                    else self._model.head.weight.detach(),
                )
                self._model.rotation_plan = RotationPlan(RotationScheme.BACKBONE_ONLY, rot_global)
                get_logger().info(
                    'DeepSeek-V4.1-Flash rotation plan set for the quantization forward: '
                    'scheme=%s global_rotation=%s engram gate basis=%s',
                    RotationScheme.BACKBONE_ONLY,
                    tuple(rot_global.shape),
                    (
                        'global_rotation (the gate inverse-rotates to the original basis)'
                        if self._model.rotation_plan.engram_gate_basis is not None
                        else 'ABSENT -- the gate would read the rotated stream'
                    ),
                )
        # hc_attn_fn / hc_ffn_fn / image_* are 1-D leaves the off-line quarot
        # rotate_linear cannot address (it requires a 2-D Linear.weight).  The
        # legacy converter rotated them into the QuaRot basis; in this standard
        # flow they stay in the original basis and the exporter writes them
        # verbatim.  Their serving-side basis must be verified on node.
        # aligner.w2.bias is similarly not rotated (rotate_linear touches weight
        # only) — document, verify against the loader on node.

        right_global = {k: v for k, v in right_global.items() if self._in_model(k)}
        left_global = {k: v for k, v in left_global.items() if self._in_model(k)}
        right_query = {k: v for k, v in right_query.items() if self._in_model(k)}
        left_query = {k: v for k, v in left_query.items() if self._in_model(k)}
        return pre_run_pairs, [
            QuaRotInterface.RotatePair(left_rot=left_global, right_rot=right_global),
            QuaRotInterface.RotatePair(left_rot=left_query, right_rot=right_query),
        ]

    # ==================== ModelSlimPipelineInterfaceV1 ====================
    def handle_dataset(self, dataset: Any, device: DeviceType = DeviceType.NPU) -> List[Any]:
        # Data-free best practices (W8A8 / W4A8 per-token activation) never carry
        # calibration data; the runner calls this unconditionally but only passes
        # a real dataset for non-data-free processors (flex_awq_ssz).
        if not dataset:
            return []
        if isinstance(dataset, str):
            dataset = [dataset]
        # jsonl rows: {"inputs_pretokenized": "..."} (lab_calib convention)
        dataset = [row.get('inputs_pretokenized') if isinstance(row, dict) else row for row in dataset]
        # Optional: render each sample as a V4.1 chat turn.  Calibration data is raw
        # text and must NOT go through this -- only eval prompts are conversations.
        if os.environ.get('DSV41_CHAT_TEMPLATE'):
            from msmodelslim.model.deepseek_v4_1_flash.prompt import render_user_prompt

            mode = os.environ.get('DSV41_THINKING_MODE', 'chat')
            dataset = [render_user_prompt(text, mode) for text in dataset]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)  # nosec B615
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        samples = []
        for text in dataset:
            encoded = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            # Plain dict (BatchEncoding is not accepted by the engine's
            # _to_generate_kwargs); drop None entries.
            samples.append({k: v for k, v in dict(encoded).items() if v is not None})
        return samples

    def init_model(self, device: DeviceType = DeviceType.NPU) -> nn.Module:
        model_path = Path(self.model_path)
        if device.value.startswith('npu'):
            import torch_npu  # noqa: F401

            model_device = torch.device(device.value)
        else:
            model_device = torch.device('cpu')
            torch.set_default_dtype(torch.bfloat16)

        self._source_keys = set(checkpoint_index(model_path))

        # Unified model construction through the inference-derived Transformer
        # (build_meta_model).  The native model is returned as-is — its module
        # paths already match the checkpoint layout — and shared leaves are
        # hydrated for the source path (AscendV1 exports get weights from the
        # format loader's hydrate instead).
        native = self.build_meta_model()
        self._model = native
        self._model_param_names = None  # rebuilt model, stale names
        self._native_shell = native
        self._model_device = model_device
        self._blocks: Dict[str, nn.Module] = {}
        if (model_path / 'inference' / 'config.json').exists():
            self._hydrate_native_shared(native)
        get_logger().info('DeepSeek-V4.1-Flash model ready (%d layers)', len(native.layers))
        return native

    def generate_decoder_layer(self, model: nn.Module) -> Generator[tuple, Any, None]:
        """Stream one native block at a time (layer-wise).

        The blocks ARE the transformer's own ``layers[i]`` / ``mtp[i]`` -- the
        same instances the calibration forward drives -- so visit/quant and
        forward share one tree.  Engram layers get their wkv split (wc + wv)
        once so the quarot processor can left-rotate only the V block.
        """
        assert self._source_keys is not None  # nosec
        native = getattr(self, '_native_shell', None)
        if native is None:
            raise UnsupportedError(
                "DeepSeek-V4.1-Flash layer-wise flow requires the checkpoint to "
                "ship inference/config.json (the structural decode-only fallback "
                "has been removed).",
                action="Use a DeepSeek-V4.1-Flash checkpoint that includes inference/config.json.",
            )
        names = [f'layers.{i}' for i in range(len(native.layers))]
        names += [f'mtp.{i}' for i in range(len(native.mtp))]
        for name in names:
            block = self._blocks.get(name)
            if block is None:
                block = (
                    native.layers[int(name.split('.')[1])]
                    if name.startswith('layers.')
                    else native.mtp[int(name.split('.')[1])]
                )
                # Decode this block's weights onto CPU (experts/attention/
                # norms/hc). The layer-wise LoadProcessor then only moves
                # cpu->npu and offloads to meta after the block is done.
                self._hydrate_native_block(block, name)
                if getattr(block, 'engram', None) is not None:
                    self._hydrate_native_engram(block)
                self._blocks[name] = block
            yield name, block

    def _hydrate_native_shared(self, native: nn.Module) -> None:
        """Hydrate forward-critical shared params: embed/head/norm, aligner,
        image tokens, per-block hc_* / norms / gate (all small), and engram
        params. Decoder expert/attention linears stay meta (layer-wise flow).
        """
        block_prefixes = tuple(f'layers.{i}.' for i in range(len(native.layers))) + tuple(
            f'mtp.{i}.' for i in range(len(native.mtp))
        )
        for name, module in native.named_modules():
            for pname, param in list(module.named_parameters(recurse=False)):
                key = f"{name}.{pname}" if name else pname
                if not param.is_meta:
                    continue
                if key not in self._source_keys:
                    continue
                # keep big per-block linears (experts/attention) on meta: the
                # LoadProcessor hydrates them per layer.
                if name.startswith(block_prefixes) and _is_layerwise_param(pname, name):
                    continue
                value = decode_tensor(self.model_path, key)
                if tuple(value.shape) != tuple(param.shape):
                    continue
                if pname.startswith('hc_'):
                    value = value.float()
                elif value.dtype != torch.bfloat16 and pname in ('weight', 'bias') and '.engram.' not in key:
                    value = value.bfloat16()
                module._parameters[pname] = torch.nn.Parameter(value.to('cpu'), requires_grad=False)
        left = [k for k, v in native.named_parameters() if v.is_meta]
        get_logger().info(
            'DeepSeek-V4.1-Flash native shared hydrate done; %d meta params left (layer-wise linears)', len(left)
        )

    def _hydrate_native_block(self, block: nn.Module, name: str) -> None:
        """Decode one native block's parameters from the source onto CPU
        (idempotent; skips non-meta and engram tables).
        """
        if self._source_keys is None:
            return
        prefix = name
        loaded = skipped = 0
        for sub_name, module in block.named_modules():
            for pname, param in list(module.named_parameters(recurse=False)):
                if not param.is_meta:
                    continue
                key = f"{prefix}.{sub_name}.{pname}" if sub_name else f"{prefix}.{pname}"
                if key not in self._source_keys:
                    skipped += 1
                    continue
                value = decode_tensor(self.model_path, key)
                if tuple(value.shape) != tuple(param.shape):
                    skipped += 1
                    continue
                if pname.startswith('hc_'):
                    value = value.float()
                else:
                    # the meta shell defaults some params to fp32; the graph
                    # computes in bf16 (except hc_* which stay fp32 by design)
                    value = value.to(torch.bfloat16)
                module._parameters[pname] = torch.nn.Parameter(value.to('cpu'), requires_grad=False)
                loaded += 1
        left = [k for k, v in block.named_parameters() if v.is_meta]
        get_logger().info(
            'DeepSeek-V4.1-Flash native block %s hydrated (%d tensors, %d skipped, %d left meta)',
            name,
            loaded,
            skipped,
            len(left),
        )

    def _hydrate_native_engram(self, block: nn.Module) -> None:
        """Pre-hydrate a native block's engram params from the source (idempotent).

        build_meta_model already did this for its own shell; this covers shells
        rebuilt by other paths.  Only the FLOAT params need it: the table stays a
        memmap, and these params are hydrated eagerly on build, so they survive
        the layer cycle on their own.
        """
        eng = getattr(block, 'engram', None)
        if eng is None or self._source_keys is None:
            return
        layer_id = getattr(block, 'layer_id', None)
        for pname, param in list(eng.named_parameters()):
            if param.is_meta and pname != 'embed.weight':
                key = f'layers.{layer_id}.engram.{pname}'
                value = decode_tensor(self.model_path, key).to(torch.bfloat16)
                parent_name, _, leaf = pname.rpartition('.')
                parent = eng.get_submodule(parent_name) if parent_name else eng
                parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)

    def generate_model_visit(self, model: nn.Module) -> Generator[ProcessRequest, Any, None]:
        for name, block in self.generate_decoder_layer(model):
            yield ProcessRequest(name, block, args=(), kwargs={})

    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        """Calibration forward on the NATIVE transformer via ``pausable_forward``.

        ``model`` is the native ``Transformer`` returned by init_model.  The
        forward is the real
        ``Transformer.forward`` — engram hashes (CPU buffers, aligned to the
        hidden stream), hyper-connections, window + compressed attention — the
        same code the eval engine verified.  ``pausable_forward`` pauses at
        every ``layers.*`` / ``mtp.*`` block so the calibration processors
        observe/replace each block's output on the very instances visit/quant
        processed; the final hidden comes back via StopIteration.value.
        """
        native = getattr(self, '_native_shell', None)
        if native is None:
            raise RuntimeError(
                'DeepSeek-V4.1-Flash forward requires the checkpoint to ship inference/config.json '
                '(native shell). Data-free W8A8/W4A8 run through generate_model_visit '
                'only.'
            )
        inputs = dict(inputs)
        ids = inputs['input_ids']
        token_types = inputs.get('token_type_ids')

        def forward_func():
            # no_grad, not inference_mode: the runner drives each block's forward
            # itself, and a calibration processor's hook may edit that block's
            # input in place -- which torch refuses on a tensor born inside
            # inference mode ("Inference tensors do not track version counter").
            # The draft walk below runs eagerly in this context, so its tensors
            # are the ones that would reach a block un-versioned.
            with torch.no_grad():
                # One entry for both halves: the backbone pauses at every layer
                # and ``dspark`` walks the draft prefill afterwards, so the mtp.*
                # blocks reach the same pause points, get the same processors and
                # are materialised by the same callback -- this function hydrates
                # nothing itself.  A schedule built from the backbone alone would
                # leave every mtp.* linear unquantized in the export.
                _log_calibration_basis_once(native)
                extra = {} if token_types is None else {'token_types': token_types}
                return native(ids, dspark=True, **extra)

        from msmodelslim.model.common.pausable_forward import ClassSpec

        def _materialise(point):
            # Decode this block's weights onto CPU just before it runs: the
            # LoadProcessor that receives the yielded request then only moves
            # cpu->npu (and offloads to meta afterwards).
            self._hydrate_native_block(point.module, point.name)
            # Remember the instance.  The visit generator is what normally fills
            # this memo, and a calibration run never drives it -- the runner
            # builds forward generators instead -- so without this the save path
            # finds no block and cannot reach the engram tensor the quarot
            # processor left rotated in it.
            self._blocks[point.name] = point.module
            eng = getattr(point.module, 'engram', None)
            if eng is not None:
                self._hydrate_native_engram(point.module)

        bindings = {
            native: {
                # Block / DSparkBlock carry the DeepSeekV41FlashDecoderLayer name after the
                # meta-shell rename; matching by class keeps submodules such as
                # layers.1.engram out of the pause points. It picks up the mtp.*
                # draft blocks too, which the same forward reaches via ``dspark``.
                ClassSpec('*DecoderLayer'): PointConfig(handler=_materialise, should_yield=lambda pt: True),
            },
        }

        drive = pausable_forward(forward_func, bindings)
        # EP collectives: every rank has to be inside the forward before any of
        # them starts one.  The barrier goes HERE, on the thread that drives the
        # generator, the way the framework's own
        # generated_decoder_layer_visit_func does it -- run inside the pausable
        # worker instead, torch_npu opens a second HCCL communicator for the same
        # device and the ranks collide on it.  ``drive`` is a generator, so
        # nothing has started yet at this point.
        if dist.is_initialized():
            dist.barrier()
        return drive

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        # KV cache in the layer-wise flow lives in RuntimeBufferStore, not a
        # module flag; nothing to toggle here.
        pass

    # ==================== Leaves & serving contract ====================
    def ascendv1_save_postprocess(self, model: nn.Module, save_directory: str) -> None:
        """Restore the DeepSeek-V4.1-Flash serving contract on top of the standard export:
        QuaRot matrices, Engram BF16 tables, and the model description JSON.
        """
        if self._source_keys is None:
            return
        save_dir = Path(save_directory)

        # optional/quarot.safetensors belongs to the QuaRot processor's
        # QuaRotExtraInfoHookIR -> AscendV1 saver path, so it shows up exactly when
        # the weights really were rotated; writing one here would hand a no-QuaRot
        # recipe a global_rotation that its weights were never produced with.
        # Serving contract fields on config.json.
        config_path = save_dir / 'config.json'
        if config_path.exists():
            model_config = json.loads(config_path.read_text())
            model_config.setdefault('quantization_config', {}).setdefault('quant_method', 'ascend')
            model_config['engram_rotation_config'] = ENGRAM_ROTATION_CONFIG
            write_json(config_path, model_config)

        # Engram: stream the memory tables, write the projections the layer walk
        # left in the rotated basis, and describe the storage so the serving
        # loader knows what it is reading (see ENGRAM_FORMAT).
        weight_map: Dict[str, str] = {}
        engram_tensors: Dict[str, Dict[str, Any]] = {}
        # actual engram layers: from the source layout (source has the tables),
        # not the pinned recipe (mini/debug ckpts carry their own ids).
        engram_layers = sorted(int(k.split('.')[1]) for k in self._source_keys if k.endswith('.engram.embed.weight'))
        for layer in engram_layers:
            base = f'layers.{layer}.engram'
            # (1) memory table — table rows are never rotated.
            embed_key = f'{base}.embed.weight'
            embed_file = f'quant_model_weights-engram-{layer}.safetensors'
            writer = write_engram_int8 if self._engram_quant_type == 'int8_sym' else write_engram_bf16
            table = writer(save_dir / embed_file, self.model_path, embed_key)
            # Register from what the writer actually produced, so the storage
            # description cannot drift from the bytes written beside it.
            for name, spec in table['tensors'].items():
                weight_map[name] = embed_file
                engram_tensors[name] = {'dtype': spec['dtype'], 'shape': list(spec['shape'])}
            # (2) wkv: the quarot processor left-rotated the whole wkv with a
            # block-diagonal matrix (wc rows kept, V block into the rotated
            # basis); write the rotated wkv as-is.
            tensors: Dict[str, torch.Tensor] = {}
            wkv = None
            block = self._blocks.get(f'layers.{layer}')
            eng = None if block is None else getattr(block, 'engram', None)
            linear = None if eng is None else getattr(eng, 'wkv', None)
            if linear is not None and linear.weight is not None:
                if linear.weight.is_meta:
                    # The engram FLOAT params are re-hydrated eagerly per layer, so
                    # a meta one means the layer walk never saw them and the
                    # pre_run rotation (if any) is not in
                    # this tensor; writing either basis silently would produce a
                    # checkpoint that disagrees with engram_rotation_config.
                    raise RuntimeError(
                        f'{base}.wkv.weight is meta at save time; the engram params did not survive the layer walk'
                    )
                wkv = linear.weight.detach().to(torch.bfloat16).contiguous()
            if wkv is None:
                # The quarot processor rotates layers.N.engram.wkv in the model
                # tree, so that tensor is the only place the served basis exists;
                # folding it out of the source by hand would be a second, silently
                # different answer.  Name what was missing instead.
                raise RuntimeError(
                    f'{base}.wkv.weight unavailable at save time: '
                    f'block={"present" if block is not None else "MISSING"}, '
                    f'engram={"present" if eng is not None else "MISSING"}, '
                    f'wkv={"present" if linear is not None else "MISSING"}'
                )
            if wkv is not None:
                tensors[f'{base}.wkv.weight'] = wkv.contiguous()
            # (3) remaining engram tensors (projection / gate): decoded BF16.
            for key in sorted(
                k for k in self._source_keys if k.startswith(base + '.') and k != embed_key and not k.endswith('.scale')
            ):
                if key == f'{base}.wkv.weight':
                    continue
                tensors[key] = decode_tensor(self.model_path, key).detach().to(torch.bfloat16).contiguous()
            if tensors:
                extra_file = f'quant_model_weights-engram-extra-{layer}.safetensors'
                write_shard(save_dir / extra_file, tensors)
                for key in tensors:
                    weight_map[key] = extra_file
        # (3) per-layer Engram quant type description.
        self._write_engram_description(save_dir, engram_tensors)
        indices = sorted(save_dir.glob('*.index.json'))
        if indices:
            self._merge_backbone_index(save_dir, indices[0], weight_map)
        else:
            get_logger().warning(
                'No *.index.json found after save; Engram tensors are not referenced '
                'by an index. Verify the saver index name on node.'
            )
        self._write_shared_leaves(save_dir, model)

    def _write_shared_leaves(self, save_dir: Path, model: nn.Module) -> None:
        """Persist top-level embed/head when the saver walk skipped them.

        embed/head are shared with the MTP draft blocks: the block pass saves
        them as ``mtp.N.embed`` / ``mtp.N.head`` and memoises the instances, so
        the post-run walk never writes the top-level ``embed.weight`` /
        ``head.weight`` the serving contract needs.
        """
        from safetensors.torch import save_file

        indices = sorted(save_dir.glob('*.index.json'))
        descp = save_dir / 'quant_model_description.json'
        if not indices or not descp.is_file():
            return
        index_path = indices[0]
        idx = json.loads(index_path.read_text())
        desc = json.loads(descp.read_text())
        weight_map = idx.setdefault('weight_map', {})
        tensors = {}
        for key in ('embed.weight', 'head.weight'):
            if key in weight_map:
                continue
            mod_name, _, pname = key.rpartition('.')
            leaf = getattr(model, mod_name, None)
            value = getattr(leaf, pname, None) if leaf is not None else None
            if value is None and self._source_keys is not None and key in self._source_keys:
                value = decode_tensor(self.model_path, key)
            if value is None:
                continue
            tensors[key] = value.detach().to(torch.bfloat16).cpu().contiguous()
        if not tensors:
            return
        shard = 'quant_model_weights-shared.safetensors'
        save_file(tensors, str(save_dir / shard))
        for key in tensors:
            weight_map[key] = shard
            desc[key] = 'FLOAT'
        write_json(index_path, idx)
        write_json(descp, desc)

    def _write_engram_description(self, save_dir: Path, engram_tensors: Dict[str, Dict[str, Any]]) -> None:
        """Describe each Engram table's quant type in quant_model_description.json.

        One ``layers.N.engram.embed.quant_type`` entry per table only when the
        saver quantizes it (``spec.model_adapter.engram_quant_type: int8_sym``);
        a plain bf16 table carries no such key.  When ``export_embedding_storage``
        is set, the legacy ``optional.embedding_storage`` block is emitted for
        old serving contracts.
        """
        path = save_dir / 'quant_model_description.json'
        description = json.loads(path.read_text()) if path.exists() else {}
        if self._engram_quant_type == 'int8_sym':
            for key in engram_tensors:
                if key.endswith('.engram.embed.weight'):
                    layer = key.split('.')[1]
                    description[f'layers.{layer}.engram.embed.quant_type'] = 'INT8_SYM'
        if self._export_embedding_storage:
            description.setdefault('optional', {})['embedding_storage'] = engram_source_description(engram_tensors)
        write_json(path, description)

    @staticmethod
    def _merge_backbone_index(save_dir: Path, index_path: Path, weight_map: Dict[str, str]) -> None:
        if not index_path.exists():
            return
        content = json.loads(index_path.read_text())
        content.setdefault('weight_map', {}).update(weight_map)
        write_json(index_path, content)


def _is_layerwise_param(pname: str, module_name: str) -> bool:
    """True for per-block weights the layer-wise LoadProcessor owns (attention
    / expert linears and their scales); everything else hydrates eagerly.
    """
    if pname.startswith('hc_'):
        return False
    if (
        module_name.endswith(('.attn', '.indexer', '.compressor'))
        or '.attn.' in module_name
        or '.indexer.' in module_name
        or '.compressor.' in module_name
    ):
        return True
    if '.ffn.experts.' in module_name or '.ffn.shared_experts' in module_name:
        return True
    if module_name.endswith('.ffn.gate') or module_name.endswith('.ffn'):
        return pname == 'weight'  # gate weight is a big linear; bias is small
    if module_name.endswith('.engram'):
        return False  # engram params are pre-hydrated separately
    if pname in ('norm.weight',) or module_name.endswith(('_norm', '.norm')):
        return False
    return False


def _load_tokenizer(model_path):
    """DeepSeek-V4.1-Flash checkpoint tokenizer (local files only)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)  # nosec B615
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
