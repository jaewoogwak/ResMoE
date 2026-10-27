"""Prepare, analyze, and evaluate ResMoE residual restoration with QJL."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
import time
from pathlib import Path

if __package__ in (None, ""):
    # Support `python experiments/resmoe_qjl/run.py` from a repository checkout.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.resmoe_qjl"

import torch

from .alignment import aligned_expert, original_expert, split_expert
from .evaluate import collect_activations, evaluate_all
from .metrics import (merge_csv, output_mse, plot_layers, relative_l2,
                      summarize_norms, write_csv)
from .model import ResidualExpert, SharedExpert
from .progress import progress_step
from .qjl import estimate, expected_mse
from .residual import (METHODS, alignment_dir, checkpoint_dir, load_payload,
                       prepare_alignment, projection_matrices, save_method)
from .weights import load_layer, model_index


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "analyze", "eval", "all"), default="all")
    parser.add_argument("--method", choices=(*METHODS, "all"), default="all")
    parser.add_argument("--model", default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--qjl-d", default="8192", help="One or more comma-separated dimensions")
    parser.add_argument("--qjl-projections", default="gate,up")
    parser.add_argument("--projection-type", choices=("gaussian",), default="gaussian")
    parser.add_argument("--down-mode", choices=("bf16", "qjl"), default="bf16")
    parser.add_argument("--anchor-expert", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--calibration-seq-len", type=int, default=512)
    parser.add_argument("--calibration-source", choices=("wikitext", "synthetic"), default="wikitext")
    parser.add_argument("--eval-tasks", default="winogrande,piqa,wikitext")
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("results/resmoe_qjl"))
    parser.add_argument("--neurons", type=int, help="Smoke test: truncate each expert's intermediate neurons")
    parser.add_argument("--ot-max-iter", type=int, default=20)
    parser.add_argument("--ot-max-neurons", type=int, default=512)
    parser.add_argument("--ot-solver", choices=("pot", "lapjv"), default="lapjv",
                        help="Exact assignment solver; lapjv solves the same uniform-mass OT problem")
    parser.add_argument("--ot-cost-device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Where to calculate FP64 OT distance matrices")
    parser.add_argument("--allow-expensive-ot", action="store_true")
    parser.add_argument("--row-chunk", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--sparsity", type=float, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.layers = list(range(32)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    args.dimensions = [int(x) for x in args.qjl_d.split(",")]
    args.methods = list(METHODS) if args.method == "all" else [args.method]
    args.projections = set(args.qjl_projections.split(","))
    args.tasks = set(args.eval_tasks.split(","))
    if (not args.layers or len(set(args.layers)) != len(args.layers) or
            any(layer < 0 or layer >= 32 for layer in args.layers)):
        parser.error("--layers must contain unique Mixtral layer indices 0..31")
    if not args.dimensions or any(d < 1 for d in args.dimensions) or len(set(args.dimensions)) != len(args.dimensions):
        parser.error("--qjl-d needs unique positive dimensions")
    if args.projections not in ({"gate", "up"}, {"gate", "up", "down"}):
        parser.error("--qjl-projections must be gate,up or gate,up,down")
    if (args.down_mode == "qjl") != ("down" in args.projections):
        parser.error("--down-mode=qjl requires --qjl-projections=gate,up,down")
    if args.tasks - {"winogrande", "piqa", "wikitext"}:
        parser.error("--eval-tasks supports winogrande,piqa,wikitext")
    if not args.tasks:
        parser.error("--eval-tasks must contain at least one task")
    if not 0 <= args.anchor_expert < 8 or args.sparsity != 0:
        parser.error("anchor must be 0..7 and this experiment requires --sparsity=0")
    if any(value is not None and value < 1 for value in (args.neurons, args.eval_limit)):
        parser.error("--neurons and --eval-limit must be positive")
    if min(args.calibration_samples, args.calibration_seq_len, args.eval_batch_size,
           args.ot_max_iter, args.ot_max_neurons, args.row_chunk) < 1 or args.epsilon <= 0:
        parser.error("sample, sequence, OT, chunk, batch, and epsilon values must be positive")
    if args.stage in ("eval", "all") and args.neurons is not None and not args.dry_run:
        parser.error("end-to-end eval requires all 14,336 neurons; omit --neurons")
    return args


def estimate_resources(args) -> dict:
    n = args.neurons or 14336
    d = 4096
    features = 3 * d
    return {"layers": args.layers, "neurons_per_expert": n,
            "layer_expert_fp32_gib": 8 * n * features * 4 / 2**30,
            "ot_source_gpu_fp64_gib": 8 * n * features * 8 / 2**30,
            "one_ot_cost_matrix_fp64_gib": n * n * 8 / 2**30,
            "eight_ot_cost_matrices_fp64_gib": 8 * n * n * 8 / 2**30,
            "one_projection_fp32_gib": {str(D): d * D * 4 / 2**30 for D in args.dimensions},
            "ot_solver_work": "superlinear in neurons; full 14336-neuron OT can take many hours per layer"}


def config_record(args) -> dict:
    info = {key: value for key, value in vars(args).items()
            if key not in {"output_dir", "projections", "tasks", "methods", "dimensions"}}
    info.update(output_dir=str(args.output_dir), projections=sorted(args.projections),
                tasks=sorted(args.tasks), methods=args.methods, dimensions=args.dimensions,
                dtype="bfloat16", projection="N(0,1)", sparsity=0,
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    try:
        root, _ = model_index(args.model)
        info["model_snapshot"] = root.name
    except Exception as error:
        info["model_snapshot_error"] = str(error)
    return info


def _read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.is_file() else []


def prepare(args) -> None:
    if args.neurons is None and not args.allow_expensive_ot and 14336 > args.ot_max_neurons:
        raise RuntimeError("full 14,336-neuron OT exceeds --ot-max-neurons; use --allow-expensive-ot "
                           "after reviewing --dry-run resources")
    norms, compression = [], []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for layer_number, layer in enumerate(args.layers, start=1):
        layer_started = time.monotonic()
        print(f"[prepare] layer {layer_number}/{len(args.layers)}: loading layer {layer}", flush=True)
        with progress_step(f"[prepare] layer {layer} weight loading"):
            weights = load_layer(args.model, layer, args.neurons)
        if weights.shape[1] > args.ot_max_neurons and not args.allow_expensive_ot:
            raise RuntimeError(f"layer {layer} has {weights.shape[1]} neurons; "
                               "increase --ot-max-neurons or use --allow-expensive-ot")
        for method in args.methods:
            dimensions = args.dimensions[:1] if method == "resmoe-bf16" else args.dimensions
            for dimension in dimensions:
                print(f"[prepare] layer {layer} {method} D={dimension}", flush=True)
                rows, stats = save_method(args.output_dir, weights, layer, method,
                                          dimension, args.seed, args.anchor_expert,
                                          args.ot_max_iter, args.projections, device, args.row_chunk,
                                          args.ot_solver, args.ot_cost_device)
                if not any(row["method"] == method and row["layer"] == layer for row in norms):
                    norms.extend(rows)
                compression.append(stats)
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        merge_csv(args.output_dir / "norms.csv", norms,
                  ("method", "layer", "expert", "projection"))
        merge_csv(args.output_dir / "compression.csv", compression,
                  ("method", "dimension", "seed", "layer"))
        write_csv(args.output_dir / "norm_summary.csv",
                  summarize_norms(_read_rows(args.output_dir / "norms.csv")))
        print(f"[prepare] layer {layer_number}/{len(args.layers)} complete "
              f"in {time.monotonic() - layer_started:.1f}s", flush=True)
    plot_layers(args.output_dir / "plots" / "residual_norm_by_layer.png",
                _read_rows(args.output_dir / "norms.csv"),
                "relative_residual_norm", "Residual Frobenius norm ratio")


def calibration_inputs(args) -> dict[int, dict[int, torch.Tensor]]:
    if args.calibration_source == "wikitext":
        missing = []
        for layer in args.layers:
            base = args.output_dir / "calibration" / f"layer_{layer}"
            if not base.with_suffix(".pt").is_file() or not base.with_suffix(".json").is_file():
                missing.append(layer)
                continue
            metadata = json.loads(base.with_suffix(".json").read_text())
            if (metadata.get("model") != args.model or
                    metadata.get("calibration_samples") != args.calibration_samples or
                    metadata.get("calibration_seq_len") != args.calibration_seq_len):
                missing.append(layer)
        if missing:
            collect_activations(args.model, missing, args.output_dir,
                                args.calibration_samples, args.calibration_seq_len)
        return {layer: torch.load(args.output_dir / "calibration" / f"layer_{layer}.pt",
                                  map_location="cpu", weights_only=True)
                for layer in args.layers}
    generator = torch.Generator().manual_seed(args.seed + 1000)
    return {layer: {expert: torch.randn(args.calibration_samples, 4096, generator=generator)
                    for expert in range(8)} for layer in args.layers}


def analyze(args) -> None:
    missing = [(method, dimension, layer)
               for method in args.methods
               for dimension in (args.dimensions[:1] if method == "resmoe-bf16" else args.dimensions)
               for layer in args.layers
               if not (checkpoint_dir(args.output_dir, method, dimension, args.seed, layer) /
                       "stats.json").is_file()]
    if missing:
        print(f"[analyze] preparing {len(missing)} missing checkpoint configurations", flush=True)
        prepare(args)
    samples = calibration_inputs(args)
    errors, expert_errors = [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for layer_number, layer in enumerate(args.layers, start=1):
        layer_started = time.monotonic()
        print(f"[analyze] layer {layer_number}/{len(args.layers)}: layer {layer}", flush=True)
        with progress_step(f"[analyze] layer {layer} weight loading"):
            weights = load_layer(args.model, layer, args.neurons)
        for method in args.methods:
            shared = torch.load(alignment_dir(args.output_dir, method, layer) / "shared_bf16.pt",
                                map_location="cpu", weights_only=True)
            dimensions = args.dimensions[:1] if method == "resmoe-bf16" else args.dimensions
            for dimension in dimensions:
                print(f"[analyze] layer {layer} {method} D={dimension}", flush=True)
                shared_module = SharedExpert(shared, device)
                for expert in range(8):
                    print(f"[analyze] layer {layer} {method} D={dimension} "
                          f"expert {expert + 1}/8", flush=True)
                    x = samples[layer][expert]
                    if x.numel() == 0:
                        continue
                    x = x.to(device=device, dtype=torch.bfloat16)
                    original = weights[expert].to(device)
                    payload = load_payload(args.output_dir, method, dimension, args.seed, layer, expert)
                    module = ResidualExpert(shared_module, payload, device)
                    with torch.inference_mode():
                        original_bf16 = original_expert(x, original.to(torch.bfloat16))
                        restored = module(x)
                    expert_errors.append({"method": method,
                                          "dimension": 0 if method == "resmoe-bf16" else dimension,
                                          "seed": 0 if method == "resmoe-bf16" else args.seed,
                                          "layer": layer, "expert": expert,
                                          "calibration_source": args.calibration_source,
                                          "samples": x.shape[0],
                                          "relative_l2": relative_l2(original_bf16, restored, args.epsilon),
                                          "output_mse": output_mse(original_bf16, restored)})
                    if method == "resmoe-bf16":
                        continue
                    shared_fp32 = torch.load(
                        alignment_dir(args.output_dir, method, layer) / "shared_fp32.pt",
                        map_location="cpu", weights_only=True).to(device)
                    shared_parts = projection_matrices(shared_fp32)
                    permutation = torch.load(
                        alignment_dir(args.output_dir, method, layer) / "permutations.pt",
                        map_location="cpu", weights_only=True)[expert].to(device)
                    aligned = original.index_select(0, permutation)
                    residual_parts = {key: value - shared_parts[key]
                                      for key, value in projection_matrices(aligned).items()}
                    gate, up, _ = split_expert(original.to(torch.bfloat16))
                    down_input = (torch.nn.functional.silu(x @ gate.T) * (x @ up.T)).float()
                    down_input = down_input.index_select(1, permutation)
                    for projection in sorted(args.projections):
                        inp = down_input if projection == "down" else x.float()
                        residual = residual_parts[projection].float()
                        exact = inp @ residual.T
                        code = payload[projection]["code"]
                        approx = estimate(inp, code)
                        original_matrix = projection_matrices(aligned)[projection].float()
                        reference_linear = inp @ original_matrix.T
                        bf16_shared_matrix = projection_matrices(shared.float().to(device))[projection]
                        full_approx = inp @ bf16_shared_matrix.T + approx
                        errors.append({"method": method, "dimension": dimension,
                                       "seed": args.seed, "layer": layer, "expert": expert,
                                       "projection": projection,
                                       "calibration_source": args.calibration_source,
                                       "samples": x.shape[0],
                                       "absolute_mse": output_mse(exact, approx),
                                       "relative_l2": relative_l2(exact, approx, args.epsilon),
                                       "original_linear_relative_l2": relative_l2(
                                           reference_linear, full_approx, args.epsilon),
                                       "theoretical_mse_fp32": expected_mse(residual, inp, dimension)})
        merge_csv(args.output_dir / "qjl_errors.csv", errors,
                  ("method", "dimension", "seed", "layer", "expert", "projection", "calibration_source"))
        merge_csv(args.output_dir / "expert_output_errors.csv", expert_errors,
                  ("method", "dimension", "seed", "layer", "expert", "calibration_source"))
        del weights
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[analyze] layer {layer_number}/{len(args.layers)} complete "
              f"in {time.monotonic() - layer_started:.1f}s", flush=True)
    plot_layers(args.output_dir / "plots" / "qjl_error_by_layer.png",
                _read_rows(args.output_dir / "qjl_errors.csv"),
                "relative_l2", "QJL residual output relative L2 error")


def eval_stage(args) -> None:
    if args.neurons is not None:
        raise ValueError("full-model evaluation cannot use truncated neurons")
    missing = [(method, dimension, layer)
               for method in args.methods
               for dimension in (args.dimensions[:1] if method == "resmoe-bf16" else args.dimensions)
               for layer in args.layers
               if not all((checkpoint_dir(args.output_dir, method, dimension, args.seed, layer) /
                           f"expert_{expert}.pt").is_file() for expert in range(8))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} full-layer checkpoints are missing; "
                                f"run --stage prepare first. First missing: {missing[0]}")
    rows = []
    for index, dimension in enumerate(args.dimensions, start=1):
        print(f"[eval] dimension {index}/{len(args.dimensions)}: D={dimension}", flush=True)
        rows.extend(evaluate_all(args.model, args.output_dir, args.methods, args.layers,
                                 dimension, args.seed, args.tasks, args.eval_batch_size,
                                 args.eval_limit))
    merge_csv(args.output_dir / "eval_results.csv", rows,
              ("method", "dimension", "seed", "layers"))


def summary(root: Path) -> None:
    norms, errors, experts = (_read_rows(root / name) for name in
                             ("norms.csv", "qjl_errors.csv", "expert_output_errors.csv"))
    evaluation, compression = (_read_rows(root / name) for name in
                               ("eval_results.csv", "compression.csv"))
    groups = {}
    for row in compression:
        groups.setdefault((row["method"], int(row["dimension"]), int(row["seed"])), {})
    for row in norms:
        if row["projection"] == "combined":
            groups.setdefault((row["method"], 0, 0), {}).setdefault("norm", []).append(
                float(row["relative_residual_norm"]))
    preferred = {(row["method"], int(row["dimension"]), int(row["seed"]),
                  row["layer"], row["expert"], row["projection"])
                 for row in errors if row["calibration_source"] == "wikitext"}
    for row in errors:
        key = (row["method"], int(row["dimension"]), int(row["seed"]))
        detail_key = (*key, row["layer"], row["expert"], row["projection"])
        if detail_key in preferred and row["calibration_source"] != "wikitext":
            continue
        groups.setdefault(key, {}).setdefault("mse", []).append(
            float(row["absolute_mse"]))
    preferred_expert = {(row["method"], int(row["dimension"]), int(row["seed"]),
                         row["layer"], row["expert"])
                        for row in experts if row["calibration_source"] == "wikitext"}
    for row in experts:
        key = (row["method"], int(row["dimension"]), int(row["seed"]))
        detail_key = (*key, row["layer"], row["expert"])
        if detail_key in preferred_expert and row["calibration_source"] != "wikitext":
            continue
        groups.setdefault(key, {}).setdefault("expert", []).append(
            float(row["relative_l2"]))
    lines = ["# ResMoE residual restoration with QJL", "",
             "Metrics are measured only for completed stages; blank cells are unmeasured.", "",
             "| Method | D | Seed | Mean Residual Ratio | Residual MSE | Expert RelL2 | Wino Acc | PIQA Acc | Wiki PPL | Storage Saving |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in evaluation:
        if row["method"] == "original-bf16":
            lines.append("| " + " | ".join(("original-bf16", "0", row["seed"], "", "", "",
                                            row.get("winogrande", ""), row.get("piqa", ""),
                                            row.get("wikitext_ppl", ""), "0")) + " |")
    visible = {key for key in groups if key[1] != 0 or key[0] == "resmoe-bf16"}
    for method, dimension, seed in sorted(visible):
        values = groups[(method, dimension, seed)]
        norm = groups.get((method, 0, 0), {}).get("norm", [])
        ev = next((r for r in evaluation if r["method"] == method and
                   int(r["dimension"]) == dimension and int(r["seed"]) == seed), {})
        storage = [float(r["saving_percent"]) for r in compression
                   if r["method"] == method and int(r["dimension"]) == dimension and
                   int(r["seed"]) == seed]
        avg = lambda seq: f"{sum(seq)/len(seq):.5g}" if seq else ""
        lines.append("| " + " | ".join((method, str(dimension), str(seed), avg(norm), avg(values.get("mse", [])),
                                        avg(values.get("expert", [])), ev.get("winogrande", ""),
                                        ev.get("piqa", ""), ev.get("wikitext_ppl", ""), avg(storage))) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    args = arguments(argv)
    if args.dry_run:
        print(json.dumps(estimate_resources(args), indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    config = config_record(args)
    if config_path.is_file():
        prior = json.loads(config_path.read_text())
        immutable = ("model", "model_snapshot", "neurons", "anchor_expert",
                     "projection_type", "qjl_projections", "down_mode", "ot_max_iter")
        mismatched = [key for key in immutable if prior.get(key) != config.get(key)]
        if mismatched:
            raise ValueError(f"output directory contains a different experiment ({mismatched}); "
                             "choose a new --output-dir")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("analyze", "all"):
        analyze(args)
    if args.stage in ("eval", "all"):
        eval_stage(args)
    summary(args.output_dir)


if __name__ == "__main__":
    main()
