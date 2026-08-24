# PIVOT_EXP_A3R Development Runtime Smoke Report

The smoke ran every registered secondary cell on all eight frozen development scenes from seed
`20260830`: 160 trials, each generated twice with deterministic decoding. The final 16
original-image CV1 cells were completed without regenerating the first 144 rows. The smoke did not read held-out
validation predictions and did not compute accuracy, candidate superiority, semantic sufficiency,
or candidate performance comparisons.

Runtime checks passed: Python 3.12.6, PyTorch 2.7.1+cu126, CUDA 12.6, HF library 4.52.4,
bitsandbytes 0.48.2, RTX 3060 Laptop GPU, NF4 with FP16 compute, all model components on `cuda:0`,
no CPU fallback, no OOM, and deterministic repeat rate 1.0.

The contract gate failed. Raw-continuation parse rate and reconstructed-output parse rate were both
`127/160 = 0.79375`; 33 outputs were retained as invalid. Their raw values were 25 instances of
`E2`, six of `E4`, and two of `E1`. Mechanical reconstruction was correct for every row, so the
failure is model emission outside the preregistered option-ID grammar, not a reconstruction bug.
Entity aliases remain invalid and were not converted.

Outcome: `SECONDARY_CONTRACT_RUNTIME_FAILURE`.
