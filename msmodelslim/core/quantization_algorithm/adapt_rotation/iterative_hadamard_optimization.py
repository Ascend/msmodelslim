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

from typing import Dict, Literal, Optional

import torch

from msmodelslim.utils.logging import get_logger

from msmodelslim.core.quantization_schema.api import calculate_qparam, fake_quantize
from msmodelslim.core.quantization_schema.qal import QDType, QScope, QStorage


def orthogonal_transform(A: torch.Tensor, B: torch.Tensor, iters: int = 100) -> torch.Tensor:
    """Newton-Schulz iteration for orthogonal polar factor of A.T @ B. Returns (D, D)."""
    C = A.T @ B
    D = C.shape[0]
    eye = torch.eye(D, device=C.device, dtype=C.dtype)
    norm = torch.linalg.norm(C)
    X = C / (norm + 1e-12)
    for _ in range(iters):
        XtX = X.T @ X
        X = 0.5 * X @ (3 * eye - XtX)
    return X


def reconstruction_loss(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean element-wise reconstruction error between A and B."""
    diff = A - B
    return torch.norm(diff).item() / diff.numel()


def quant_tensor_sym(
    tensor: torch.Tensor,
    quant_dtype: Literal["int4", "int8"],
    q_scale_thresh: float = 1e-5,
) -> torch.Tensor:
    """
    Symmetric per-token quantization (QDQ).
    Reuses msmodelslim.core.quantization_schema.api (calculate_qparam, fake_quantize). Supports quant_dtype in ("int4", "int8").
    """
    q_dtype = QDType.INT4 if quant_dtype == "int4" else QDType.INT8
    min_val = tensor.amin(dim=-1, keepdim=True)  # (N, 1) for (N, D) broadcast
    max_val = tensor.amax(dim=-1, keepdim=True)
    q_param = calculate_qparam(
        min_val=min_val,
        max_val=max_val,
        q_dtype=q_dtype,
        q_scope=QScope.PER_TOKEN,
        symmetric=True,
    )
    if q_scale_thresh > 0:
        scale = q_param.ext["scale"]
        q_param.ext["scale"] = torch.clamp(scale, min=q_scale_thresh)
    out = fake_quantize(QStorage(QDType.FLOAT, tensor), q_param)
    return out.value.to(tensor.dtype)


def quant_tensor_sym_batched(
    tensor: torch.Tensor,
    batch_size: int,
    quant_dtype: Literal["int4", "int8"],
    q_scale_thresh: float = 1e-5,
) -> torch.Tensor:
    """Batch-wise symmetric quantization to control memory."""
    results = []
    for i in range(0, tensor.shape[0], batch_size):
        batch = tensor[i : i + batch_size]
        results.append(quant_tensor_sym(batch, quant_dtype, q_scale_thresh))
    return torch.cat(results, dim=0)


class HadamardOptimizer:
    """
    Iterative Hadamard rotation optimization for quantization-friendly activation transform.
    """

    def __init__(
        self,
        quant_dtype: Literal["int4", "int8"] = "int4",
        batch_size: int = 128,
        steps: int = 20,
        patience: int = 5,
        min_steps: int = 6,
        max_samples: int = 2048,
    ):
        self.quant_dtype = quant_dtype
        self.batch_size = batch_size
        self.steps = steps
        self.patience = patience
        self.min_steps = min_steps
        self.max_samples = max_samples

    @torch.no_grad()
    def optimize(
        self,
        activations_dict: Dict[str, torch.Tensor],
        hadamard_matrix: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Optimize Hadamard rotation matrix for quantization-friendly projection.

        Args:
            activations_dict: Dict mapping layer name to activation tensor (N, D).
            hadamard_matrix: Initial Hadamard matrix (D, D).
            device: Device for computation. Defaults to hadamard_matrix device.

        Returns:
            Optimized rotation matrix H_adapted (D, D).
        """
        max_samples = self.max_samples
        if device is None:
            device = hadamard_matrix.device

        D = hadamard_matrix.shape[1]
        R_acc = torch.eye(D, device=device, dtype=torch.float32)
        best_R_acc = R_acc.clone()
        best_loss = float("inf")
        hadamard_matrix = hadamard_matrix.to(device=device, dtype=torch.float32)
        no_improve = 0
        fixed_layers = list(activations_dict.keys())

        for step in range(self.steps):
            get_logger().info(
                "Hadamard optimization step %d/%d, layers=%d, quant_dtype=%s",
                step + 1,
                self.steps,
                len(fixed_layers),
                self.quant_dtype,
            )
            A_all, B_all = [], []
            for layer_name in fixed_layers:
                acts_full = activations_dict[layer_name].to(torch.float32).to(device)
                acts = acts_full[:max_samples, :]
                projected = acts @ hadamard_matrix @ R_acc
                projected = projected.to(device)
                quantized = quant_tensor_sym_batched(projected, self.batch_size, self.quant_dtype)
                A_all.append(projected)
                B_all.append(quantized)

            A_cat = torch.cat(A_all, dim=0)
            B_cat = torch.cat(B_all, dim=0)
            del A_all, B_all
            R_step = orthogonal_transform(A_cat, B_cat)
            R_acc = R_acc @ R_step
            projected_final = A_cat @ R_step
            quantized_final = quant_tensor_sym_batched(projected_final, self.batch_size, self.quant_dtype)
            loss = reconstruction_loss(projected_final, quantized_final)
            if loss < best_loss:
                best_loss = loss
                best_R_acc = R_acc.clone()
                no_improve = 0
                get_logger().debug("  -> loss=%.6f (new best)", loss)
            else:
                no_improve += 1
                get_logger().debug(
                    "  -> loss=%.6f, no_improve=%d/%d",
                    loss,
                    no_improve,
                    self.patience,
                )
                if step >= self.min_steps and no_improve >= self.patience:
                    get_logger().debug(
                        "Early stop at step %d: no improvement for %d steps",
                        step + 1,
                        self.patience,
                    )
                    break

        get_logger().info(
            "Hadamard optimization done: %d steps, best_loss=%.6f",
            step + 1,
            best_loss,
        )
        H_adapted = hadamard_matrix @ best_R_acc
        return H_adapted
