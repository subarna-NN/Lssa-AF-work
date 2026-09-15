# LSSA: A Learnable Spectral-Smooth Activation Function for Physics-Informed Neural Networks

Reference implementation and experiment scripts for the paper
*"LSSA: A Learnable Spectral-Smooth Activation Function for Physics-Informed
Neural Networks"*.

## Overview

LSSA is a three-parameter per-neuron activation for PINNs:

**φ(z; α, β, γ) = α·tanh(βz) + (1−α)·sin(γz)·exp(−z²/2)**

combining a learnable-stiffness smooth monotone term (β) with a
Gaussian-windowed spectral term (learnable frequency γ), blended by a
learnable coefficient α. All three parameters are optimized jointly with the
network weights. Initialization: (α₀, β₀, γ₀) = (0.5, 1.0, π).

All experiments use an identical 5×128 MLP, a two-phase Adam + L-BFGS
optimizer, Latin Hypercube collocation sampling, and seed 42, benchmarked on
the 1D Burgers', 2D Helmholtz, and 1D Allen–Cahn equations against tanh, GELU,
Swish, and SELU. Finite-difference (FDM) reference solutions are computed
inside each script.

## Requirements
torch>=2.0
numpy
scipy
matplotlib
## Scripts

| Script | Produces |
|---|---|
| `Burger_lssa_main.py` | LSSA on 1D Burgers': solution, error, loss, and learned-parameter figures |
| `Helmholtz_lssa_main.py` | LSSA on 2D Helmholtz: solution, error, loss, and learned-parameter figures |
| `Allencahn_lssa_main.py` | LSSA on 1D Allen–Cahn: solution, error, loss, and learned-parameter figures |
| `Allencahn_baselines_multiseed.py` | Multi-seed (42, 123, 2024, 7) mean±std for tanh/GELU/Swish/SELU on Allen–Cahn |
| `Helmholtz_component_ablation.py` | Ablation of the (α, β, γ) components on Helmholtz (V1–V4) |
| `Helmholtz_architecture_ablation.py` | LSSA vs tanh across depths/widths (3×64, 3×128, 5×128, 7×128) on Helmholtz |
| `Helmholtz_gamma0_sensitivity.py` | γ₀ initialization sweep on Helmholtz (robust band and failure region) |
| `Helmholtz_spectral_baselines.py` | LSSA vs SIREN, FF-PINN, and GaborPINN on Helmholtz |
| `FFT_walltime_errormap.py` | Output-spectrum FFT, loss-vs-wall-clock, and squared-error maps |

## Citation
If you use this code, please cite the associated paper.
