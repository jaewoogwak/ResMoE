"""Numerical error definitions and lightweight result writers."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import torch


def relative_l2(reference: torch.Tensor, approximate: torch.Tensor,
                epsilon: float = 1e-12) -> float:
    """Zero reference and exact zero error -> 0; otherwise divide by epsilon."""
    numerator = float(torch.linalg.vector_norm((reference.float() - approximate.float()).reshape(-1)))
    denominator = float(torch.linalg.vector_norm(reference.float().reshape(-1)))
    return numerator / max(denominator, epsilon)


def output_mse(reference: torch.Tensor, approximate: torch.Tensor) -> float:
    return float((reference.float() - approximate.float()).square().sum(-1).mean())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def merge_csv(path: Path, rows: list[dict], keys: tuple[str, ...]) -> None:
    if not rows:
        return
    existing = list(csv.DictReader(path.open())) if path.is_file() else []
    merged = {tuple(str(row[key]) for key in keys): row for row in existing}
    merged.update({tuple(str(row[key]) for key in keys): row for row in rows})
    write_csv(path, list(merged.values()))


def summarize_norms(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row["method"], int(row["layer"]), row["projection"])
        grouped.setdefault(key, []).append(float(row["relative_residual_norm"]))
    return [{"method": method, "layer": layer, "projection": projection,
             "mean": statistics.mean(values), "median": statistics.median(values),
             "min": min(values), "max": max(values)}
            for (method, layer, projection), values in sorted(grouped.items())]


def plot_layers(path: Path, rows: list[dict], value_key: str, title: str) -> None:
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    for row in rows:
        key = (row.get("method", ""), row.get("projection", ""))
        groups.setdefault(key, {}).setdefault(int(row["layer"]), []).append(float(row[value_key]))
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, by_layer in groups.items():
        layers = sorted(by_layer)
        ax.plot(layers, [sum(by_layer[layer]) / len(by_layer[layer]) for layer in layers],
                marker="o", label="/".join(filter(None, key)))
    ax.set(xlabel="Layer", ylabel=value_key, title=title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
