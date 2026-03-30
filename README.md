# THE EMERALD APEX: Unified Spectral Dominance

[![DOI: 10.5281/zenodo.19252002](https://zenodo.org/badge/DOI/10.5281/zenodo.19252002.svg)](https://doi.org/10.5281/zenodo.19252002)
[![DOI: 10.5281/zenodo.19205731](https://zenodo.org/badge/DOI/10.5281/zenodo.19205731.svg)](https://doi.org/10.5281/zenodo.19205731)
[![Website](https://img.shields.io/badge/Live_Vault-Active-emerald)](https://andrewkkim58-afk.github.io/emerald-apex/)

**Emerald Research Group** | Lead Researcher: Andrew Kim

This repository contains the foundational code, theoretical manuscripts, and analytical telemetry for the **Emerald Apex** framework—a closed operator-theoretic synthesis for achieving deterministic stability in noisy dynamical systems.

## 📚 Formal Publications

The framework is bifurcated into two foundational layers:

* **Layer 1 (The Microscopic):** * *Spectral Phase-Less Transition Travel: Operator-Theoretic and Gauge-Geometric Foundations*
    * **DOI:** [10.5281/zenodo.19252002](https://doi.org/10.5281/zenodo.19252002)
* **Layer 2 (The Macroscopic):**
    * *The Emerald Apex: Solipsistic Phase Transitions and the Kim-Einstein Field Law*
    * **DOI:** [10.5281/zenodo.19205731](https://doi.org/10.5281/zenodo.19205731)

## ⚙️ Core Artifacts

1.  **`kalki_telemetry.py` (JAX Simulation Engine)**
    The bare-metal execution script. It computes high-dimensional matrix exponentials to empirically validate spectral lifting, semigroup suppression, and von Neumann entropy collapse ($S \to 0$).
2.  **`index.html`**
    The terminal interface for the Vault, rendering canonical equations and the "Kalki" spectral regime.

## 🚀 Reproduction Protocol
Requires JAX (64-bit precision).
```bash
pip install jax jaxlib numpy
python kalki_telemetry.py

## 🟢 Phase 4: The WXY-8 Manifold Hypervisor (Manuscript V)

**Axiomatic Bounding of Thermodynamic Drift in Heterogeneous LLM Training**

Phase 4 elevates Emerald Apex from a hardware offloading experiment into a formalized operator-theoretic framework. We treat the 192GB DDR5 system memory as a **pristine anchor manifold** and the RTX 5090 VRAM as an **active computational arena**. 

During optimization, momentum-based solvers (AdamW) cause the network to fracture away from its out-of-core anchor—a phenomenon defined as **Thermodynamic Drift ($\epsilon_t$)**. The WXY-8 Hypervisor computes the orthogonal leakage of the active weights across the PCIe bus and applies a dynamic propensity penalty ($\mu$) to enforce a strict spectral gap.

### Key Breakthroughs
* **The Spectral-Empirical Trade-off:** Empirical proof on a 1.5B parameter causal transformer demonstrating that gradient-level projection minimizes loss but suffers linear momentum leakage, whereas absolute weight-level projection achieves a strict plateau ($\epsilon_t \approx 0.66$).
* **The Manifold Lock:** Bypassing the optimizer to physically squash weights back into the permitted geometry post-update, trapping the LLM in a mathematical cage without catastrophic learning failure.
* **Multi-Anchor Topological Retention:** Theoretical proofs for extending the operator to support multi-anchor subspaces, laying the mathematical groundwork for continuous learning without catastrophic interference.

### 🚀 Reproducing the Bare-Metal Telemetry
```bash
# 1. Setup sterile Equinox/JAX environment
conda create -n emerald_env python=3.11 -y
conda activate emerald_env
pip install -U "jax[cuda12]" equinox optax numpy
conda install -c conda-forge matplotlib -y

# 2. Fire the reactor 
python run_regime_a.py         # Regime A: Gradient Projection (Soft Bounding)
python emerald_apex_phase4.py  # Regime B: Absolute Weight Projection (Hard Lock)
python plot_telemetry.py       # Render dual-axis telemetry graphs