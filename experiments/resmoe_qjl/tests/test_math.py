import math
from types import SimpleNamespace

import numpy as np
import ot
import torch

from experiments.resmoe_qjl.alignment import (aligned_expert, aligned_weights, anchor_mean,
                                             original_expert, ot_barycenter, split_expert)
from experiments.resmoe_qjl.metrics import relative_l2
from experiments.resmoe_qjl.model import ResidualExpert, SharedExpert, install_layer
from experiments.resmoe_qjl.qjl import encode, estimate, expected_mse, pack_signs, unpack_signs


def test_joint_permutation_preserves_expert_function():
    generator = torch.Generator().manual_seed(7)
    weights = torch.randn(3, 8, 12, generator=generator)
    perm = torch.stack([torch.randperm(8, generator=generator) for _ in range(3)])
    x = torch.randn(5, 4, generator=generator)
    for expert in range(3):
        assert torch.allclose(original_expert(x, weights[expert]),
                              aligned_expert(x, weights[expert], perm[expert]), atol=1e-5)
    aligned = aligned_weights(weights, perm)
    shared = aligned.mean(0)
    residual = aligned - shared
    assert torch.allclose(shared + residual, aligned, atol=1e-6)


def test_ot_and_anchor_alignment_have_distinct_valid_assets():
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(6, 12, generator=generator)
    weights = torch.stack([base] + [base[torch.randperm(6, generator=generator)] + 0.01 *
                                  torch.randn(6, 12, generator=generator) for _ in range(2)])
    ot_shared, ot_perm = ot_barycenter(weights, anchor=0, max_iter=3)
    mean_shared, mean_perm = anchor_mean(weights, anchor=0)
    assert ot_shared.shape == mean_shared.shape == (6, 12)
    for perm in (ot_perm, mean_perm):
        assert all(torch.equal(torch.sort(row).values, torch.arange(6)) for row in perm)
    assert torch.allclose(mean_shared, aligned_weights(weights, mean_perm).mean(0))


def test_progressive_ot_matches_pot_reference():
    weights = torch.randn(3, 5, 12, generator=torch.Generator().manual_seed(101))
    arrays = weights.numpy().astype(np.float64)
    mass = np.full(5, 1 / 5)
    reference = ot.lp.free_support_barycenter(
        measures_locations=list(arrays), measures_weights=[mass] * 3,
        X_init=arrays[0].copy(), numItermax=4, numThreads="max")
    actual, _ = ot_barycenter(weights, anchor=0, max_iter=4)
    assert torch.allclose(actual, torch.from_numpy(reference).float(), atol=1e-6)


def test_exact_lapjv_ot_matches_pot_alignment():
    weights = torch.randn(3, 9, 24, generator=torch.Generator().manual_seed(109))
    pot_shared, pot_permutations = ot_barycenter(weights, anchor=0, max_iter=4)
    fast_shared, fast_permutations = ot_barycenter(
        weights, anchor=0, max_iter=4, solver="lapjv", cost_device="cpu")
    assert torch.equal(fast_permutations, pot_permutations)
    assert torch.allclose(fast_shared, pot_shared, atol=1e-6)
    pot_mean, pot_mean_permutations = anchor_mean(weights, anchor=0)
    fast_mean, fast_mean_permutations = anchor_mean(
        weights, anchor=0, solver="lapjv", cost_device="cpu")
    assert torch.equal(fast_mean_permutations, pot_mean_permutations)
    assert torch.allclose(fast_mean, pot_mean, atol=1e-6)
    if torch.cuda.is_available():
        gpu_shared, gpu_permutations = ot_barycenter(
            weights, anchor=0, max_iter=4, solver="lapjv", cost_device="cuda")
        assert torch.equal(gpu_permutations, pot_permutations)
        assert torch.allclose(gpu_shared, pot_shared, atol=1e-6)


def test_qjl_bit_packing_scaling_and_unbiased_seed_average():
    signs = torch.tensor([[True, False, True, True, False, False, True, False, True]])
    packed = pack_signs(signs)
    assert packed.dtype == torch.uint8 and packed.shape == (1, 2)
    assert torch.equal(unpack_signs(packed, 9) > 0, signs)
    weight = torch.tensor([[1.0, -0.7, 0.2], [-0.5, 0.4, 0.9]])
    x = torch.tensor([[0.3, -0.2, 1.0]])
    exact = x @ weight.T
    estimates = []
    for seed in range(32):
        code = encode(weight, 1024, seed, scale_dtype=torch.float32)
        assert code["packed_signs"].numel() == 2 * 128
        estimates.append(estimate(x, code))
    mean = torch.stack(estimates).mean(0)
    assert torch.allclose(mean, exact, atol=0.04)
    empirical = float((torch.stack(estimates) - exact).square().sum(-1).mean())
    theory = expected_mse(weight, x, 1024)
    assert theory > 0 and 0.3 < empirical / theory < 2.5


def test_zero_reference_error_rule():
    assert relative_l2(torch.zeros(3), torch.zeros(3)) == 0
    assert math.isfinite(relative_l2(torch.zeros(3), torch.ones(3)))


def test_residual_expert_restores_linears_before_nonlinearity():
    generator = torch.Generator().manual_seed(23)
    original = torch.randn(6, 12, generator=generator) * 0.2
    shared = torch.randn(6, 12, generator=generator) * 0.1
    residual = original - shared
    from experiments.resmoe_qjl.residual import projection_matrices
    payload = {name: {"kind": "bf16", "weight": matrix.to(torch.bfloat16)}
               for name, matrix in projection_matrices(residual).items()}
    expert = ResidualExpert(SharedExpert(shared, torch.device("cpu")), payload,
                            torch.device("cpu"))
    x = torch.randn(3, 4, generator=generator).to(torch.bfloat16)
    reference = original_expert(x, original.to(torch.bfloat16))
    assert relative_l2(reference, expert(x)) < 0.05


def test_toy_mixtral_expert_installation_for_all_methods(tmp_path):
    from experiments.resmoe_qjl.residual import save_method

    generator = torch.Generator().manual_seed(31)
    weights = torch.randn(3, 4, 12, generator=generator) * 0.2
    for method in ("resmoe-bf16", "resmoe-qjl", "aligned-mean-qjl"):
        save_method(tmp_path, weights, 0, method, 128, 7, 0, 2,
                    {"gate", "up"}, "cpu", 4)
    block = SimpleNamespace(
        gate=torch.nn.Linear(4, 3, bias=False),
        experts=torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)]))
    model = SimpleNamespace(
        config=SimpleNamespace(intermediate_size=4),
        model=SimpleNamespace(layers=[SimpleNamespace(block_sparse_moe=block)]))
    x = torch.randn(2, 4, generator=generator).to(torch.bfloat16)
    for method in ("resmoe-bf16", "resmoe-qjl", "aligned-mean-qjl"):
        install_layer(model, tmp_path, method, 128, 7, 0)
        for expert in range(3):
            result = block.experts[expert](x)
            assert result.shape == (2, 4) and torch.isfinite(result).all()
            if method == "resmoe-bf16":
                reference = original_expert(x, weights[expert].to(torch.bfloat16))
                assert relative_l2(reference, result) < 0.06
