import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.linalg import expm
import numpy as np
import os

# Set JAX to use 64-bit precision for strict numerical validation
jax.config.update("jax_enable_x64", True)

print("INITIATING EMERALD APEX: KALKI TERMINAL REGIME VALIDATION")
print("=========================================================")

# --- 1. Base Operator & Projections ---
A_0 = jnp.diag(jnp.array([1.0, 3.0, 5.0]))
phi_0 = jnp.array([1.0, 0.0, 0.0])

# Construct orthogonal projector P_perp
P_0 = jnp.outer(phi_0, phi_0)
P_perp = jnp.eye(3) - P_0

# --- 2. Penalized Operator Function ---
@jit
def get_A_mu(mu):
    return A_0 + mu * P_perp

# --- 3. Telemetry: Spectral Splitting (Figure 1 Data) ---
print("\n[+] Computing Spectral Splitting...")
mu_vals = jnp.linspace(0.0, 6.0, 100)

@jit
def compute_spectra(mu):
    A_mu = get_A_mu(mu)
    evals = jnp.linalg.eigvalsh(A_mu)
    return evals

spectra_trajectory = vmap(compute_spectra)(mu_vals)

# Export for PGFPlots
np.savetxt("spectral_splitting.csv", 
           np.column_stack((mu_vals, spectra_trajectory)), 
           delimiter=",", header="mu,l0,l1,l2", comments="")
print("    -> Saved 'spectral_splitting.csv'")

# --- 4. Telemetry: Semigroup Suppression (Figure 2 Data) ---
print("[+] Computing Semigroup Suppression (mu = 4.0)...")
mu_target = 4.0
A_target = get_A_mu(mu_target)
t_vals = jnp.linspace(0.0, 1.5, 100)
psi_init = jnp.array([1.0, 1.0, 1.0])

@jit
def evolve_relative_amplitude(t):
    # Compute exp(-t * A_mu)
    propagator = expm(-t * A_target)
    state = jnp.dot(propagator, psi_init)
    
    # Factor out the protected decay e^{-t * lambda_0}
    protected_decay = jnp.exp(-t * A_0[0,0])
    relative_state = state / protected_decay
    return relative_state

relative_trajectories = vmap(evolve_relative_amplitude)(t_vals)

# Export for PGFPlots
np.savetxt("semigroup_suppression.csv", 
           np.column_stack((t_vals, relative_trajectories)), 
           delimiter=",", header="t,mode0,mode1,mode2", comments="")
print("    -> Saved 'semigroup_suppression.csv'")

# --- 5. Telemetry: Thermal Collapse & Entropy (Section 9.4 - 9.6) ---
print("\n[+] Validating Thermal Dominance & Entropy Collapse...")
beta = 1.0

def compute_thermodynamics(mu):
    A_mu = get_A_mu(mu)
    # Gibbs state components
    boltzmann_weights = jnp.exp(-beta * jnp.diag(A_mu))
    Z = jnp.sum(boltzmann_weights)
    probs = boltzmann_weights / Z
    
    # von Neumann Entropy: -sum(p * log(p))
    # Add tiny epsilon to prevent log(0)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-15))
    return Z, probs[0], entropy

Z_val, p0_val, entropy_val = compute_thermodynamics(mu_target)

print(f"    Target Penalty (mu) : {mu_target}")
print(f"    Inverse Temp (beta) : {beta}")
print(f"    Partition Func (Z)  : {Z_val:.6f}")
print(f"    Protected Weight    : {p0_val * 100:.3f}%")
print(f"    von Neumann Entropy : {entropy_val:.6f}")

if p0_val > 0.99:
    print("\n[!] STATUS: Solipsistic Phase Transition Confirmed.")
else:
    print("\n[!] STATUS: Thermal Coherence Failed.")