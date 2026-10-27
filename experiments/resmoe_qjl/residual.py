"""Prepare and load shared weights and residual checkpoint assets."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .alignment import aligned_weights, anchor_mean, original_expert, ot_barycenter, split_expert
from .progress import progress_step
from .qjl import encode

METHODS = ("resmoe-bf16", "resmoe-qjl", "aligned-mean-qjl")
PROJECTIONS = ("gate", "up", "down")


def alignment_dir(root: Path, method: str, layer: int) -> Path:
    family = "ot" if method != "aligned-mean-qjl" else "anchor_mean"
    return root / "shared" / family / f"layer_{layer}"


def checkpoint_dir(root: Path, method: str, dimension: int, seed: int, layer: int) -> Path:
    if method == "resmoe-bf16":
        dimension, seed = 0, 0
    return root / "checkpoints" / method / f"d{dimension}" / f"seed{seed}" / f"layer_{layer}"


def prepare_alignment(root: Path, weights: torch.Tensor, layer: int,
                      method: str, anchor: int, max_iter: int,
                      ot_solver: str = "pot", ot_cost_device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    target = alignment_dir(root, method, layer)
    shared_path, permutation_path = target / "shared_fp32.pt", target / "permutations.pt"
    if shared_path.exists() and permutation_path.exists():
        metadata = json.loads((target / "metadata.json").read_text())
        if metadata["neurons"] != weights.shape[1] or metadata["anchor_expert"] != anchor:
            raise ValueError(f"cached alignment disagrees with experiment: {target}")
        return (torch.load(shared_path, map_location="cpu", weights_only=True),
                torch.load(permutation_path, map_location="cpu", weights_only=True))
    if method == "aligned-mean-qjl":
        shared, permutations = anchor_mean(weights, anchor,
                                           progress_label=f"[prepare] layer {layer} aligned mean",
                                           solver=ot_solver, cost_device=ot_cost_device)
    else:
        shared, permutations = ot_barycenter(weights, anchor, max_iter,
                                             progress_label=f"[prepare] layer {layer} OT",
                                             solver=ot_solver, cost_device=ot_cost_device)
    target.mkdir(parents=True, exist_ok=True)
    torch.save(shared.float(), shared_path)
    torch.save(shared.to(torch.bfloat16), target / "shared_bf16.pt")
    torch.save(permutations, permutation_path)
    (target / "metadata.json").write_text(json.dumps({
        "method": "anchor_mean" if method == "aligned-mean-qjl" else "ot_barycenter",
        "layer": layer, "neurons": weights.shape[1], "anchor_expert": anchor,
        "shape": list(shared.shape), "ot_max_iter": max_iter if method != "aligned-mean-qjl" else None,
        "ot_solver": ot_solver, "ot_cost_device_requested": ot_cost_device,
        "sparsity": 0}, indent=2) + "\n")
    return shared, permutations


def projection_matrices(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    gate, up, down = split_expert(weight)
    return {"gate": gate, "up": up, "down": down}


def norm_rows(original: torch.Tensor, shared: torch.Tensor, residual: torch.Tensor,
              method: str, layer: int, expert: int) -> list[dict]:
    rows = []
    for name in (*PROJECTIONS, "combined"):
        if name == "combined":
            weights = (original, shared, residual)
        else:
            weights = (projection_matrices(original)[name],
                       projection_matrices(shared)[name],
                       projection_matrices(residual)[name])
        norms = [float(torch.linalg.vector_norm(w.float())) for w in weights]
        rows.append({"method": method, "layer": layer, "expert": expert,
                     "projection": name, "original_norm": norms[0],
                     "shared_norm": norms[1], "residual_norm": norms[2],
                     "relative_residual_norm": norms[2] / max(norms[0], 1e-12)})
    return rows


def save_method(root: Path, weights: torch.Tensor, layer: int, method: str,
                dimension: int, seed: int, anchor: int, max_iter: int,
                qjl_projections: set[str], device: str, row_chunk: int,
                ot_solver: str = "pot", ot_cost_device: str = "cpu") -> tuple[list[dict], dict]:
    target = checkpoint_dir(root, method, dimension, seed, layer)
    info_path = target / "metadata.json"
    stats_path, norms_path = target / "stats.json", target / "norms.json"
    if info_path.is_file():
        info = json.loads(info_path.read_text())
        expected = sorted(qjl_projections) if method != "resmoe-bf16" else []
        if (info["neurons"] != weights.shape[1] or info["experts"] != weights.shape[0] or
                info["qjl_projections"] != expected or info["method"] != method):
            raise ValueError(f"existing checkpoint has a different configuration: {target}")
        if (stats_path.is_file() and norms_path.is_file() and
                all((target / f"expert_{expert}.pt").is_file()
                    for expert in range(weights.shape[0]))):
            print(f"[prepare] layer {layer} {method} D={dimension}: cached", flush=True)
            return json.loads(norms_path.read_text()), json.loads(stats_path.read_text())
    shared, permutations = prepare_alignment(root, weights, layer, method, anchor, max_iter,
                                              ot_solver, ot_cost_device)
    aligned = aligned_weights(weights, permutations)
    target.mkdir(parents=True, exist_ok=True)
    norms = []
    residual_bytes = 0
    packed_bytes = 0
    scale_bytes = 0
    max_reconstruction_error = 0.0
    max_permutation_output_error = 0.0
    max_oracle_output_error = 0.0
    for expert in range(weights.shape[0]):
        print(f"[prepare] layer {layer} {method} D={dimension} "
              f"expert {expert + 1}/{weights.shape[0]}", flush=True)
        residual = aligned[expert] - shared
        reconstructed = shared + residual
        max_reconstruction_error = max(max_reconstruction_error,
                                       float((reconstructed - aligned[expert]).abs().max()))
        probe = torch.randn(2, weights.shape[2] // 3,
                            generator=torch.Generator().manual_seed(1000 + layer * 8 + expert))
        reference = original_expert(probe, weights[expert])
        max_permutation_output_error = max(max_permutation_output_error,
            float((reference - original_expert(probe, aligned[expert])).abs().max()))
        max_oracle_output_error = max(max_oracle_output_error,
            float((reference - original_expert(probe, reconstructed)).abs().max()))
        norms.extend(norm_rows(weights[expert], shared, residual, method, layer, expert))
        payload = {}
        for projection, matrix in projection_matrices(residual).items():
            if method != "resmoe-bf16" and projection in qjl_projections:
                with progress_step(f"[prepare] layer {layer} {method} D={dimension} "
                                   f"expert {expert + 1}/{weights.shape[0]} {projection} QJL"):
                    code = encode(matrix, dimension, seed + (1 if projection == "down" else 0),
                                  device=device, row_chunk=row_chunk)
                payload[projection] = {"kind": "qjl", "code": code}
                packed_bytes += code["packed_signs"].numel()
                scale_bytes += code["row_norm"].numel() * code["row_norm"].element_size()
            else:
                dense = matrix.to(torch.bfloat16)
                payload[projection] = {"kind": "bf16", "weight": dense}
                residual_bytes += dense.numel() * dense.element_size()
        torch.save(payload, target / f"expert_{expert}.pt")
    shared_bytes = shared.numel() * 2
    original_bytes = weights.numel() * 2
    stored_bytes = shared_bytes + residual_bytes + packed_bytes + scale_bytes
    stats = {"method": method, "layer": layer,
             "dimension": 0 if method == "resmoe-bf16" else dimension,
             "seed": 0 if method == "resmoe-bf16" else seed,
             "original_bf16_bytes": original_bytes, "shared_bf16_bytes": shared_bytes,
             "residual_bf16_bytes": residual_bytes, "qjl_sign_bytes": packed_bytes,
             "theoretical_packed_sign_bytes": (
                 0 if method == "resmoe-bf16" else
                 sum(projection_matrices(shared)[name].shape[0]
                     for name in qjl_projections) * weights.shape[0] * ((dimension + 7) // 8)),
             "qjl_scale_bytes": scale_bytes, "persistent_expert_bytes": stored_bytes,
             "saving_percent": 100 * (1 - stored_bytes / original_bytes),
             "projection_seed_only_persistent": True,
             "shared_projection_runtime_bytes": (
                 (weights.shape[2] // 3) * dimension * 4 +
                 ((weights.shape[1] * dimension * 4) if "down" in qjl_projections else 0))
                 if method != "resmoe-bf16" else 0,
             "fp32_aligned_reconstruction_max_abs": max_reconstruction_error}
    stats["fp32_original_vs_permuted_expert_max_abs"] = max_permutation_output_error
    stats["fp32_reconstructed_expert_max_abs"] = max_oracle_output_error
    if max_permutation_output_error > 1e-3 or max_oracle_output_error > 1e-3:
        raise AssertionError("neuron permutation or FP32 restoration changed the expert function")
    (target / "metadata.json").write_text(json.dumps({
        "method": method, "layer": layer, "dimension": dimension, "seed": seed,
        "neurons": weights.shape[1], "experts": weights.shape[0],
        "qjl_projections": sorted(qjl_projections) if method != "resmoe-bf16" else [],
        "down_mode": "qjl" if "down" in qjl_projections and method != "resmoe-bf16" else "bf16",
        "alignment_dir": str(alignment_dir(root, method, layer)), "sparsity": 0}, indent=2) + "\n")
    stats["checkpoint_disk_bytes"] = sum(path.stat().st_size for path in target.glob("expert_*.pt"))
    stats["shared_bf16_disk_bytes"] = (alignment_dir(root, method, layer) /
                                       "shared_bf16.pt").stat().st_size
    stats["total_checkpoint_disk_bytes"] = (stats["checkpoint_disk_bytes"] +
                                             stats["shared_bf16_disk_bytes"])
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    norms_path.write_text(json.dumps(norms, indent=2) + "\n")
    return norms, stats


def load_payload(root: Path, method: str, dimension: int, seed: int,
                 layer: int, expert: int) -> dict:
    path = checkpoint_dir(root, method, dimension, seed, layer) / f"expert_{expert}.pt"
    return torch.load(path, map_location="cpu", weights_only=True)
