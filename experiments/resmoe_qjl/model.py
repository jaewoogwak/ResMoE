"""Mixtral expert replacement preserving Gate -> SiLU -> Up -> Down order."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .alignment import split_expert
from .qjl import estimate
from .residual import alignment_dir, load_payload


class SharedExpert(nn.Module):
    def __init__(self, weight: torch.Tensor, device: torch.device):
        super().__init__()
        gate, up, down = split_expert(weight)
        for name, matrix in (("gate", gate), ("up", up), ("down", down)):
            self.register_buffer(name, matrix.to(device=device, dtype=torch.bfloat16))


class ResidualExpert(nn.Module):
    def __init__(self, shared: SharedExpert, payload: dict, device: torch.device):
        super().__init__()
        self.shared = shared
        self.kinds = {}
        self.codes = {}
        for name, row in payload.items():
            self.kinds[name] = row["kind"]
            if row["kind"] == "qjl":
                self.codes[name] = {key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                                    for key, value in row["code"].items()}
            else:
                self.register_buffer(f"residual_{name}", row["weight"].to(device))

    def linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        shared = F.linear(x, getattr(self.shared, name))
        if self.kinds[name] == "qjl":
            residual = estimate(x, self.codes[name])
        else:
            residual = F.linear(x, getattr(self, f"residual_{name}"))
        return (shared.float() + residual.float()).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.linear("gate", x)
        up = self.linear("up", x)
        hidden = F.silu(gate) * up
        return self.linear("down", hidden)


def install_layer(model, root: Path, method: str, dimension: int,
                  seed: int, layer: int) -> None:
    block = model.model.layers[layer].block_sparse_moe
    shared_path = alignment_dir(root, method, layer) / "shared_bf16.pt"
    shared_weight = torch.load(shared_path, map_location="cpu", weights_only=True)
    if shared_weight.shape[0] != model.config.intermediate_size:
        raise ValueError("partial-neuron checkpoints cannot be installed into a full Mixtral model")
    device = block.gate.weight.device
    shared = SharedExpert(shared_weight, device)
    for expert in range(len(block.experts)):
        payload = load_payload(root, method, dimension, seed, layer, expert)
        block.experts[expert] = ResidualExpert(shared, payload, device)
