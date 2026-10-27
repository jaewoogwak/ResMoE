"""Stream one Mixtral MoE layer from Hugging Face safetensors shards."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open


def model_index(model: str) -> tuple[Path, dict]:
    path = Path(hf_hub_download(model, "model.safetensors.index.json"))
    return path.parent, json.loads(path.read_text())["weight_map"]


def load_layer(model: str, layer: int, neurons: int | None = None) -> torch.Tensor:
    """Return [expert, neuron, gate_input|up_input|down_output] in FP32.

    This preserves ResMoE's concat(w1, w3, w2.T, dim=1) convention.
    Only one shard is open at a time and only one layer remains resident.
    """
    root, index = model_index(model)
    prefix = f"model.layers.{layer}.block_sparse_moe.experts."
    names = [f"{prefix}{expert}.{projection}.weight"
             for expert in range(8) for projection in ("w1", "w3", "w2")]
    missing = [name for name in names if name not in index]
    if missing:
        raise ValueError(f"missing Mixtral expert tensors: {missing[:3]}")
    shards: dict[str, list[str]] = {}
    for name in names:
        shards.setdefault(index[name], []).append(name)
    tensors = {}
    for shard, shard_names in shards.items():
        path = root / shard
        if not path.is_file():
            path = Path(hf_hub_download(model, shard))
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in shard_names:
                if neurons is None:
                    value = handle.get_tensor(name)
                elif name.endswith(".w2.weight"):
                    value = handle.get_slice(name)[:, :neurons]
                else:
                    value = handle.get_slice(name)[:neurons]
                tensors[name] = value.float()
    experts = []
    for expert in range(8):
        w1, w3, w2 = (tensors[f"{prefix}{expert}.{projection}.weight"]
                      for projection in ("w1", "w3", "w2"))
        if w1.shape != w3.shape or w2.shape != (w1.shape[1], w1.shape[0]):
            raise ValueError(f"invalid Mixtral expert tensor shapes in layer {layer}")
        experts.append(torch.cat((w1, w3, w2.T), dim=1))
    return torch.stack(experts)
