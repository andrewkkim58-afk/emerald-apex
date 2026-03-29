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