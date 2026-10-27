"""Gaussian 1-bit QJL with persistent packed signs and row-wise norms."""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(maxsize=8)
def gaussian_projection(input_dim: int, dimension: int, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(input_dim, dimension, generator=generator, dtype=torch.float32).to(device)


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    if signs.dtype != torch.bool:
        raise TypeError("signs must be boolean")
    padded = F.pad(signs.to(torch.uint8), (0, -signs.shape[-1] % 8))
    bits = padded.reshape(*signs.shape[:-1], -1, 8)
    return (bits * (2 ** torch.arange(8, device=signs.device, dtype=torch.uint8))).sum(-1).to(torch.uint8)


def unpack_signs(packed: torch.Tensor, dimension: int) -> torch.Tensor:
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    signs = ((packed.unsqueeze(-1) >> shifts) & 1).reshape(*packed.shape[:-1], -1)
    return signs[..., :dimension].to(torch.float32).mul_(2).sub_(1)


def encode(weight: torch.Tensor, dimension: int, seed: int, *,
           scale_dtype: torch.dtype = torch.float16, row_chunk: int = 64,
           device: str = "cpu") -> dict:
    """Rows are encoded in tiles; the projection is shared by input dim and seed."""
    if weight.ndim != 2 or dimension < 1 or row_chunk < 1:
        raise ValueError("invalid QJL matrix, dimension, or row chunk")
    projection = gaussian_projection(weight.shape[1], dimension, seed, device)
    norm = torch.linalg.vector_norm(weight.float(), dim=1)
    packed = []
    for start in range(0, weight.shape[0], row_chunk):
        rows = weight[start:start + row_chunk].to(device=device, dtype=torch.float32)
        packed.append(pack_signs(rows @ projection >= 0).cpu())
    return {"packed_signs": torch.cat(packed), "row_norm": norm.to(scale_dtype).cpu(),
            "input_dim": weight.shape[1], "dimension": dimension, "seed": seed,
            "scale_dtype": str(scale_dtype)}


def estimate(x: torch.Tensor, code: dict) -> torch.Tensor:
    dimension = int(code["dimension"])
    if x.shape[-1] != code["input_dim"]:
        raise ValueError("QJL input width mismatch")
    original_shape = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1]).float()
    projection = gaussian_projection(code["input_dim"], dimension,
                                     int(code["seed"]), str(flat.device))
    projected = flat @ projection
    signs = unpack_signs(code["packed_signs"].to(flat.device), dimension)
    scales = code["row_norm"].to(device=flat.device, dtype=torch.float32)
    result = (projected @ signs.T) * (math.sqrt(math.pi / 2) / dimension) * scales
    return result.reshape(*original_shape, signs.shape[0])


def persistent_bytes(code: dict) -> int:
    return sum(t.numel() * t.element_size() for t in code.values() if isinstance(t, torch.Tensor))


def expected_mse(weight: torch.Tensor, x: torch.Tensor, dimension: int) -> float:
    """Gaussian-projection expectation, excluding scale rounding and bit errors."""
    x = x.reshape(-1, weight.shape[1]).float()
    weight = weight.float()
    exact = x @ weight.T
    value = ((math.pi / 2) * x.square().sum(-1) * weight.square().sum()
             - exact.square().sum(-1)) / dimension
    return float(value.clamp_min(0).mean())
