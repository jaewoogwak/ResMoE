"""ResMoE joint OT barycenter and independent anchor alignment."""

from __future__ import annotations

import numpy as np
import ot
import torch
from scipy.optimize import linear_sum_assignment

from .progress import progress_step


def validate_permutations(permutations: torch.Tensor, neurons: int) -> None:
    expected = torch.arange(neurons)
    if permutations.dtype != torch.long or permutations.ndim != 2:
        raise ValueError("permutations must be [expert, neuron] int64")
    for row in permutations:
        if not torch.equal(torch.sort(row.cpu()).values, expected):
            raise ValueError("OT assignment is not a one-to-one neuron permutation")


def aligned_weights(weights: torch.Tensor, permutations: torch.Tensor) -> torch.Tensor:
    """Reorder the concatenated rows; w2.T then follows the same permutation."""
    validate_permutations(permutations, weights.shape[1])
    return torch.stack([weights[e].index_select(0, permutations[e])
                        for e in range(weights.shape[0])])


class _CostComputer:
    """Squared Euclidean distances in FP64, optionally using GPU GEMM."""

    def __init__(self, arrays: np.ndarray, requested: str, label: str | None):
        self.arrays = arrays
        self.active_device = "cpu"
        self.sources = self.source_norms = None
        n = arrays.shape[1]
        source_bytes = arrays.nbytes
        cost_bytes = n * n * 8
        enough_gpu_memory = (torch.cuda.is_available() and
                             torch.cuda.mem_get_info()[0] >
                             source_bytes + 3 * cost_bytes + 4 * 2**30)
        if requested == "cuda" and not enough_gpu_memory:
            raise RuntimeError("FP64 CUDA OT cost requested but GPU memory is insufficient")
        if requested == "cuda" or (requested == "auto" and enough_gpu_memory and n >= 512):
            self.active_device = "cuda"
            self.sources = [torch.from_numpy(array).to("cuda") for array in arrays]
            self.source_norms = [source.square().sum(dim=1) for source in self.sources]
        if label is not None:
            print(f"{label}: FP64 OT cost device={self.active_device}", flush=True)

    def matrix(self, target: np.ndarray, expert: int) -> np.ndarray:
        if self.active_device == "cpu":
            return ot.utils.dist(target, self.arrays[expert])
        with torch.no_grad():
            x = torch.from_numpy(target).to("cuda")
            start = x.square().sum(dim=1)[:, None] + self.source_norms[expert][None, :]
            cost = torch.addmm(start, x, self.sources[expert].T, beta=1, alpha=-2)
            return cost.clamp_min_(0).cpu().numpy()


def _assignment(cost: np.ndarray, solver: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Equal-mass EMD is the exact minimum-cost one-to-one assignment."""
    n = cost.shape[0]
    if solver == "lapjv":
        rows, cols = linear_sum_assignment(cost)
        if not np.array_equal(rows, np.arange(n)):
            raise AssertionError("assignment rows are not ordered")
        return cols.astype(np.int64, copy=False), None
    if solver == "pot":
        masses = np.full(n, 1.0 / n, dtype=np.float64)
        transport = ot.lp.emd(masses, masses, cost, numThreads="max")
        return transport.argmax(axis=1).astype(np.int64), transport
    raise ValueError(f"unsupported exact OT solver: {solver}")


def _emd_permutation(target: np.ndarray, computer: _CostComputer, expert: int,
                     solver: str) -> torch.Tensor:
    """Same target-to-expert orientation as mixtral/wb.py."""
    perm, _ = _assignment(computer.matrix(target, expert), solver)
    result = torch.from_numpy(perm)
    validate_permutations(result[None], target.shape[0])
    return result


def ot_barycenter(weights: torch.Tensor, anchor: int, max_iter: int,
                  *, progress_label: str | None = None, solver: str = "pot",
                  cost_device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Use the free-support OT barycenter of mixtral/mixtral-ot.py.

    Unlike the source script, inputs are FP32 CPU arrays and EMD assignments
    are checked for a complete bijection before any residual is calculated.
    """
    e, n, _ = weights.shape
    if not 0 <= anchor < e:
        raise ValueError("invalid anchor expert")
    arrays = weights.detach().cpu().numpy().astype(np.float64, copy=False)
    computer = _CostComputer(arrays, cost_device, progress_label)
    if progress_label is not None:
        print(f"{progress_label}: exact assignment solver={solver}", flush=True)
    # The fixed-point updates below follow POT's free_support_barycenter loop.
    # Exposing each EMD call lets a long full-neuron run report real progress.
    bary = arrays[anchor].copy()
    bary_mass = np.full(n, 1.0 / n, dtype=np.float64)
    for iteration in range(max_iter):
        updated = np.zeros_like(bary)
        for expert in range(e):
            label = (f"{progress_label} barycenter iteration {iteration + 1}/{max_iter} "
                     f"expert {expert + 1}/{e}")
            with progress_step(label, enabled=progress_label is not None):
                permutation, transport = _assignment(computer.matrix(bary, expert), solver)
                if transport is None:
                    updated += arrays[expert][permutation] / e
                else:
                    updated += (1.0 / e) / bary_mass[:, None] * (transport @ arrays[expert])
        displacement = np.sum((updated - bary) ** 2)
        bary = updated
        if progress_label is not None:
            print(f"{progress_label} barycenter iteration {iteration + 1}/{max_iter}: "
                  f"displacement={displacement:.6g}", flush=True)
        if displacement <= 1e-7:
            break
    permutations = []
    for expert in range(e):
        with progress_step(f"{progress_label} final alignment expert {expert + 1}/{e}",
                           enabled=progress_label is not None):
            permutations.append(_emd_permutation(bary, computer, expert, solver))
    permutations = torch.stack(permutations)
    return torch.from_numpy(np.asarray(bary, dtype=np.float32)), permutations


def anchor_mean(weights: torch.Tensor, anchor: int,
                *, progress_label: str | None = None, solver: str = "pot",
                cost_device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Independently match every expert to the anchor, then take their mean."""
    e, _, _ = weights.shape
    if not 0 <= anchor < e:
        raise ValueError("invalid anchor expert")
    arrays = weights.detach().cpu().numpy().astype(np.float64, copy=False)
    computer = _CostComputer(arrays, cost_device, progress_label)
    if progress_label is not None:
        print(f"{progress_label}: exact assignment solver={solver}", flush=True)
    permutations = []
    for expert in range(e):
        with progress_step(f"{progress_label} anchor alignment expert {expert + 1}/{e}",
                           enabled=progress_label is not None):
            permutations.append(_emd_permutation(arrays[anchor], computer, expert, solver))
    permutations = torch.stack(permutations)
    aligned = aligned_weights(weights, permutations)
    return aligned.mean(dim=0), permutations


def split_expert(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, width = weight.shape
    if width % 3:
        raise ValueError("concatenated expert width must be 3 * hidden_size")
    hidden = width // 3
    # Mixtral representation has gate/up input width d and down output width d.
    if width != 3 * hidden:
        raise ValueError("invalid concatenated expert width")
    return weight[:, :hidden], weight[:, hidden:2 * hidden], weight[:, 2 * hidden:].T


def original_expert(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gate, up, down = split_expert(weight)
    intermediate = torch.nn.functional.silu(x @ gate.T) * (x @ up.T)
    return intermediate @ down.T


def aligned_expert(x: torch.Tensor, weight: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    # Both Gate/Up output and Down input are permuted by the same row index.
    aligned = weight.index_select(0, permutation)
    return original_expert(x, aligned)
