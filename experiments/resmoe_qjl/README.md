# Mixtral residual restoration with QJL

Run from the repository root with the `mixtral` Python environment. Install
`POT` (Python Optimal Transport), `safetensors`, `datasets`, `lm-eval`,
`matplotlib`, `transformers`, and `pytest` in that environment. The code reads
the official Mixtral safetensors shards one layer at a time. It never modifies
the original model or the older ResMoE scripts.
Both `python -m experiments.resmoe_qjl.run` and
`python experiments/resmoe_qjl/run.py` are supported.
Progress is printed for each layer, OT iteration and expert, QJL encoding,
calibration, and evaluation. Long individual operations print elapsed time
every 60 seconds. These messages apply to processes started after this update;
an already running Python process keeps its previously loaded code.

The source scripts `mixtral/mixtral-ot.py` and `mixtral/wb.py` use a free-support
Wasserstein barycenter initialized from an anchor expert, then uniform-mass EMD
from barycenter neurons to each expert. This experiment preserves those OT
operations and the `concat(w1, w3, w2.T, dim=1)` row layout. It validates that
each EMD assignment is a bijection. The aligned-mean method independently
matches each expert to the anchor before taking an arithmetic mean; it does
not reuse ResMoE's final permutations.

## Fast validation

```bash
cd /home/jaewoo/ResMoE
PATH=/venv/mixtral/bin:$PATH python -m pytest -q experiments/resmoe_qjl/tests
PATH=/venv/mixtral/bin:$PATH python -m experiments.resmoe_qjl.run \
  --stage prepare --method all --layers 0 --neurons 16 --qjl-d 256 \
  --output-dir results/resmoe_qjl_smoke
PATH=/venv/mixtral/bin:$PATH python -m experiments.resmoe_qjl.run \
  --stage analyze --method all --layers 0 --neurons 16 --qjl-d 256 \
  --calibration-source synthetic --calibration-samples 4 \
  --output-dir results/resmoe_qjl_smoke
```

The smoke uses `--stage prepare` and `--stage analyze` separately because
truncated-neuron assets are deliberately rejected by end-to-end evaluation.

## Real experiments

```bash
python -m experiments.resmoe_qjl.run --stage prepare --method all \
  --layers 0,6 --qjl-d 8192,16384 --output-dir results/resmoe_qjl \
  --allow-expensive-ot
python -m experiments.resmoe_qjl.run --stage analyze --method all \
  --layers 0,6 --qjl-d 8192,16384 --calibration-samples 128 \
  --output-dir results/resmoe_qjl
python -m experiments.resmoe_qjl.run --stage eval --method all \
  --layers 0,6 --qjl-d 16384 --eval-tasks winogrande,piqa,wikitext \
  --output-dir results/resmoe_qjl
```

`--stage eval` requires checkpoints already prepared for every requested layer,
method, dimension, and seed. To evaluate all 32 layers, prepare all 32 first.
`--stage analyze` prepares any missing dimension checkpoints before measuring
errors, so the preparation example can also start with only `--qjl-d 8192`.
`--dry-run` prints memory estimates without reading or writing model weights.
Full 14,336-neuron exact EMD is computationally expensive. The default guard
requires `--allow-expensive-ot` for it; partial-neuron runs are analysis smoke
tests and cannot claim whole-model accuracy.

For equal neuron masses, EMD is exactly a minimum-cost one-to-one assignment.
The default `--ot-solver lapjv` uses SciPy's exact assignment solver; `pot`
selects the original POT network-simplex path. `--ot-cost-device auto` computes
squared Euclidean distances in FP64 on CUDA when enough GPU memory is free,
otherwise on CPU. `--ot-cost-device cpu` forces the original CPU distances.
These settings retain the same OT objective and do not truncate neurons,
iterations, or QJL dimensions. Floating-point evaluation order can differ,
so exact bitwise equality is not guaranteed. The toy tests compare both
solvers and devices against the POT result. A running process must be restarted
to use these options; completed layer checkpoints are reused.

The default QJL projections are Gate and Up. Down stays BF16. To encode Down too,
set both `--qjl-projections gate,up,down` and `--down-mode qjl`. Gaussian
projection seed and dimension are shared across methods B and C. QJL signs are
packed into 8 bits per byte. The projection matrix is regenerated from its seed
and counted separately as resident runtime memory, while checkpoint storage
includes the packed signs, FP16 row norms, BF16 shared matrix, and BF16 residuals.
`compression.csv` separates theoretical packed-sign bytes, actual packed tensor
bytes, and checkpoint file bytes (including serialization overhead).

`norms.csv` lists every expert and projection; `norm_summary.csv` gives each
layer's mean, median, minimum, and maximum residual ratio. `qjl_errors.csv`
contains empirical and theoretical output MSE, residual relative L2, and full
linear-output relative L2. `expert_output_errors.csv` measures the entire
SiLU/Gate/Up/Down function. `compression.csv` reports persistent tensor and
checkpoint bytes. `eval_results.csv` records end-to-end metrics only if an
evaluation actually ran. `summary.md` and plots summarize completed stages.

Relative L2 uses `max(reference_norm, 1e-12)` as denominator. A zero reference
and zero error is exactly zero; a zero reference with nonzero error is finite
but large. The theoretical QJL MSE is the Gaussian FP32 expectation; observed
MSE additionally includes FP16 scale and BF16 shared-weight rounding.
