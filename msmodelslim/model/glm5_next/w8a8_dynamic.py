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

# Calibration-free BF16 -> W8A8_DYNAMIC conversion for GLM-5.3-Flash (``model_type=glm5_next``).
#
# GLM-5.3-Flash is a 45-layer hybrid model (34 KDA linear-attention layers + 11 DeepSeek sparse-MLA
# layers) with multi-hyper-connection (MHC) residual streams, a 288-expert MoE and one MTP draft
# layer. Only the FFN projections are quantized here:
#
#   * ``mlp.experts.{gate,up,down}_proj``   routed experts of the MoE layers
#   * ``mlp.shared_experts.{gate,up,down}_proj``
#   * ``mlp.{gate,up,down}_proj``           dense FFN of ``first_k_dense_replace`` layers
#
# Everything else stays BF16 (``FLOAT``):
#
#   * KDA attention projections (``q/k/v_proj``, ``b_proj``, ``f_a/g_a_proj``, ``f_b/g_b_proj``,
#     ``o_proj``) and the depthwise ``q/k/v_conv1d`` weights: the KDA projections are fused into a
#     single ``in_proj_qkvbfg_a`` projection by the inference engines, so their activation ranges
#     are not observable per projection without a calibration forward, and the native HF FP8
#     checkpoints keep them in BF16 as well.
#   * Sparse-MLA projections (``q_a/q_b/kv_a/kv_b_proj``, ``o_proj``) and the indexer
#     (``wq_b``, ``wk``, ``weights_proj``): the ``wk`` + ``weights_proj`` pair consumes the
#     ``input_layernorm`` output, i.e. the norm-linear fusion partner set differs from every
#     already supported pedigree.
#   * MHC hyper-connection modules (``hc_attn_*``, ``hc_ffn_*``), the MoE router, all norms,
#     embeddings / ``lm_head``, the vision tower and the MTP draft layer.
#
# The weights are quantized per output channel with a symmetric INT8 grid directly from the BF16
# checkpoint (min/max, no calibration data), and the activations are quantized per token at
# runtime (``W8A8_DYNAMIC``). Output layout:
#
#   * ``quant_model_description.json``
#   * ``quant_model_weights-<n>.safetensors`` (+ ``quant_model_weights.safetensors.index.json``)
#   * tokenizer / config files copied from the source model
#
# NOTE: the description file intentionally keeps the minimal key set (``group_size`` and
# ``metadata``) that was used for the on-device verification of this conversion; official
# ModelSlim products additionally carry ``model_quant_type`` / ``version`` / ``is_rot_used`` /
# ``optional`` entries which are not needed to load the weights back.

import os
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import get_logger
from msmodelslim.utils.security import get_valid_read_path, json_safe_dump, json_safe_load

DESCRIPTION_FILE = "quant_model_description.json"
WEIGHTS_INDEX_FILE = "quant_model_weights.safetensors.index.json"
WEIGHTS_FILE_TEMPLATE = "quant_model_weights-{:05d}.safetensors"

# Default size of one output shard, in GiB, same default as the ascendv1_saver, and hard cap of
# one shard kept in memory while converting.
DEFAULT_PART_FILE_SIZE = 4
MAX_IN_MEMORY_SHARD_SIZE = 5 * 1024**3

# ``model.language_model.layers.*`` for the Glm5NextForConditionalGeneration checkpoints,
# ``model.layers.*`` for a text-only Glm5NextForCausalLM checkpoint.
QUANTIZED_TENSOR_PATTERNS = (
    re.compile(
        r"^(?:model\.language_model|model)\.layers\.(?P<layer_idx>\d+)\.mlp\.experts\.\d+"
        r"\.(?:gate|up|down)_proj\.weight$"
    ),
    re.compile(
        r"^(?:model\.language_model|model)\.layers\.(?P<layer_idx>\d+)\.mlp\.shared_experts"
        r"\.(?:gate|up|down)_proj\.weight$"
    ),
    re.compile(r"^(?:model\.language_model|model)\.layers\.(?P<layer_idx>\d+)\.mlp\.(?:gate|up|down)_proj\.weight$"),
)

# Auxiliary files copied to the quantized model so that it is directly loadable by the
# inference engines.
_AUXILIARY_FILES = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def get_quantized_tensor_patterns() -> tuple:
    """Return the patterns of the tensors that are converted to W8A8_DYNAMIC."""
    return QUANTIZED_TENSOR_PATTERNS


def get_quant_layer_count(text_config: dict) -> int:
    """Return the number of decoder layers that are quantized, MTP draft layers excluded."""
    if not isinstance(text_config, dict):
        raise InvalidModelError(
            "The text_config section of config.json is expected to be a dict.",
            action="Please check the config.json of the GLM-5.3-Flash model",
        )

    num_hidden_layers = text_config.get("num_hidden_layers")
    num_nextn_predict_layers = text_config.get("num_nextn_predict_layers", 0)
    if not isinstance(num_hidden_layers, int) or num_hidden_layers <= 0:
        raise InvalidModelError(
            f"Invalid num_hidden_layers: {num_hidden_layers}.",
            action="Please check the config.json of the GLM-5.3-Flash model",
        )
    if not isinstance(num_nextn_predict_layers, int) or num_nextn_predict_layers < 0:
        raise InvalidModelError(
            f"Invalid num_nextn_predict_layers: {num_nextn_predict_layers}.",
            action="Please check the config.json of the GLM-5.3-Flash model",
        )
    return num_hidden_layers - num_nextn_predict_layers


def should_quantize(tensor_name: str, num_quant_layers: int) -> bool:
    """Tell whether ``tensor_name`` belongs to the W8A8_DYNAMIC scope of GLM-5.3-Flash."""
    for pattern in QUANTIZED_TENSOR_PATTERNS:
        matched = pattern.match(tensor_name)
        if matched:
            # The MTP draft layer (index >= num_quant_layers) keeps its BF16 weights, which is
            # also how the officially released GLM MoE checkpoints treat the draft layer.
            return int(matched.group("layer_idx")) < num_quant_layers
    return False


def quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``weight`` per output channel with a symmetric INT8 grid.

    Args:
        weight: BF16 / FP16 weight of shape ``[out_features, in_features]``.

    Returns:
        A tuple ``(quantized_weight, weight_scale, weight_offset)``. The offset is all zeros
        because the grid is symmetric; it is still exported so that the layout matches the
        ModelSlim products.
    """
    if weight.dim() != 2:
        raise InvalidModelError(
            f"Only 2D weights can be quantized, but got {weight.dim()}D tensor.",
            action="Please check the GLM-5.3-Flash checkpoint",
        )
    if not torch.isfinite(weight).all():
        raise InvalidModelError(
            "The weight to quantize contains NaN or Inf values.",
            action="Please check the GLM-5.3-Flash checkpoint",
        )

    amax = weight.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-8)
    scale = amax / 127.0
    quantized = torch.clamp(torch.round(weight.float() / scale), -127, 127).to(torch.int8)
    return quantized, scale, torch.zeros_like(scale)


def _get_weight_files(model_path: Path) -> list[str]:
    """Return the safetensors shards of the source model in ascending order."""
    index_file = model_path.joinpath("model.safetensors.index.json")
    if index_file.is_file():
        weight_map = json_safe_load(str(index_file)).get("weight_map", {})
        if not weight_map:
            raise InvalidModelError(
                f"No weight_map found in {index_file.name}.",
                action="Please check the GLM-5.3-Flash checkpoint",
            )
        return sorted(set(weight_map.values()), key=_shard_index)

    return sorted(path.name for path in model_path.glob("*.safetensors") if path.is_file())


def _shard_index(file_name: str) -> int:
    matched = re.search(r"-(\d+)-of-", file_name)
    if matched is None:
        matched = re.search(r"-(\d+)\.safetensors$", file_name)
    return int(matched.group(1)) if matched else 0


def _copy_auxiliary_files(model_path: Path, save_path: Path) -> None:
    for file_name in _AUXILIARY_FILES:
        source = model_path.joinpath(file_name)
        if source.is_file():
            shutil.copy(source, save_path.joinpath(file_name))
    json_safe_dump(
        {"framework": "pytorch", "task": "others", "allow_remote": True},
        str(save_path.joinpath("configuration.json")),
        indent=1,
    )


def convert_to_w8a8_dynamic(
    model_path: str,
    save_path: str,
    part_file_size: int = DEFAULT_PART_FILE_SIZE,
    copy_auxiliary_files: bool = True,
) -> None:
    """Convert a BF16 GLM-5.3-Flash checkpoint into the ModelSlim W8A8_DYNAMIC format.

    No calibration data and no forward pass are required: the weights are quantized with a
    per-channel symmetric min/max grid and the activations are quantized per token at runtime.

    Args:
        model_path: path of the BF16 safetensors checkpoint.
        save_path: output directory, created when it does not exist.
        part_file_size: soft limit of one output shard, in GiB.
        copy_auxiliary_files: whether tokenizer / config files are copied to ``save_path``.
    """
    if not isinstance(part_file_size, int) or part_file_size <= 0:
        raise InvalidModelError(
            f"part_file_size is expected to be a positive int, but got {part_file_size}.",
            action="Please pass the shard size in GiB",
        )

    source = Path(model_path)
    destination = Path(save_path)
    if not source.is_dir():
        raise InvalidModelError(
            f"Model path {source!s} does not exist.",
            action="Please check the GLM-5.3-Flash checkpoint path",
        )
    os.makedirs(destination, exist_ok=True)
    os.chmod(destination, 0o750)

    text_config = json_safe_load(str(source.joinpath("config.json"))).get("text_config", {})
    num_quant_layers = get_quant_layer_count(text_config)
    weight_files = _get_weight_files(source)
    get_logger().info(
        "Quantize the FFN projections of the first %s layers from %s safetensors shards.",
        num_quant_layers,
        len(weight_files),
    )

    description: dict[str, object] = {"group_size": 0, "metadata": {}}
    weight_index: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_size = 0
    shard_count = 0
    num_quantized = 0

    def flush_shard() -> None:
        nonlocal shard, shard_size, shard_count
        if not shard:
            return
        shard_count += 1
        file_name = WEIGHTS_FILE_TEMPLATE.format(shard_count)
        save_file(shard, str(destination.joinpath(file_name)), metadata={"format": "pt"})
        os.chmod(destination.joinpath(file_name), 0o600)
        weight_index.update({name: file_name for name in shard})
        get_logger().info("Exported %s (%.2f GiB, %s tensors).", file_name, shard_size / 1024**3, len(shard))
        shard = {}
        shard_size = 0

    for file_name in weight_files:
        file_path = get_valid_read_path(
            str(source.joinpath(file_name)), extensions="safetensors", size_max=64 * 1024**3
        )
        with safe_open(file_path, framework="pt", device="cpu") as safetensors_file:
            for tensor_name in sorted(safetensors_file.keys()):
                weight = safetensors_file.get_tensor(tensor_name)
                if should_quantize(tensor_name, num_quant_layers) and weight.dim() == 2:
                    # ModelSlim hangs the scale / offset on the layer name, without the
                    # trailing ".weight" of the quantized tensor.
                    base_name = tensor_name.removesuffix(".weight")
                    quantized, scale, offset = quantize_per_channel(weight)
                    shard[tensor_name] = quantized
                    shard[base_name + ".weight_scale"] = scale
                    shard[base_name + ".weight_offset"] = offset
                    description[tensor_name] = "W8A8_DYNAMIC"
                    description[base_name + ".weight_scale"] = "W8A8_DYNAMIC"
                    description[base_name + ".weight_offset"] = "W8A8_DYNAMIC"
                    num_quantized += 1
                    shard_size += quantized.numel() + scale.numel() * 4 * 2
                else:
                    shard[tensor_name] = weight
                    description[tensor_name] = "FLOAT"
                    shard_size += weight.numel() * weight.element_size()
                if shard_size >= min(part_file_size * 1024**3, MAX_IN_MEMORY_SHARD_SIZE):
                    flush_shard()
    flush_shard()

    if not weight_index:
        raise InvalidModelError(
            "No tensor was loaded from the GLM-5.3-Flash checkpoint.",
            action="Please check that the model path contains safetensors weights",
        )

    json_safe_dump(description, str(destination.joinpath(DESCRIPTION_FILE)), indent=1)
    os.chmod(destination.joinpath(DESCRIPTION_FILE), 0o600)
    json_safe_dump(
        {"metadata": {"total_size": 0}, "weight_map": weight_index},
        str(destination.joinpath(WEIGHTS_INDEX_FILE)),
        indent=1,
    )
    os.chmod(destination.joinpath(WEIGHTS_INDEX_FILE), 0o600)

    if copy_auxiliary_files:
        _copy_auxiliary_files(source, destination)

    get_logger().info(
        "GLM-5.3-Flash W8A8_DYNAMIC conversion done: %s quantized projections in %s shards.",
        num_quantized,
        shard_count,
    )
