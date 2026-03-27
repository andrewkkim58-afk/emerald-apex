# THE EMERALD APEX: Unified Spectral Dominance

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19248447.svg)](https://doi.org/10.5281/zenodo.19248447)
[![Website](https://img.shields.io/badge/Live_Vault-Active-blue)](https://andrewkkim58-afk.github.io/emerald-apex/)

**Emerald Research Group** | Principal IC: Andrew Kim

This repository contains the foundational code, theoretical manuscripts, and front-end architecture for the **Emerald Apex** framework. The framework provides a closed operator-theoretic synthesis connecting microscopic spectral gap amplification to macroscopic thermodynamic collapse and coherent gravitational transport.

## The Axiom
> *No heuristic assumptions. All claims derivational. Structure is enforced, not assumed.*

## Core Artifacts

1. **`Spectral_Phase-Less_Transition_Travel.pdf`**
   The formal manuscript detailing the derivation of the Kim-Einstein Field Law and the operator-theoretic bounds of the Solipsistic Phase. 
   
2. **`kalki_telemetry.py` (JAX Simulation Engine)**
   The bare-metal execution script. It natively computes the high-dimensional matrix exponentials to empirically validate the spectral lifting, semigroup suppression, and von Neumann entropy collapse ($S(\rho_\mu) \to 0$) defined in the manuscript. 
   * Outputs raw telemetry to CSV for reproducible phase-space plotting.

3. **`index.html`**
   The frontend UI for the Terminal Vault, rendering the canonical equations via CHtml MathJax.

## Execution Protocol

To reproduce the terminal regime telemetry locally, you require a compute environment capable of handling high-precision tensor operations (JAX 64-bit enabled).

### Dependencies
```bash
pip install jax jaxlib numpy