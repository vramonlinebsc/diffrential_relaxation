# DiffRelax: Differentiable NMR Relaxation Engine

## Project Overview

**Goal:** Build the first JAX-based differentiable physics engine for NMR relaxation that enables gradient-based structure refinement using R1, R2, NOE, and **cross-correlated relaxation** (CCR) data.

**Timeline:** 6-8 weeks  
**Target:** Structure validation for AlphaFold/cryo-EM, publishable in JACS/JCP/JCIM

---

## Architecture

### Phase 1: Core Physics Engine (Weeks 1-2)

```
Structure Coordinates → Geometry → Spectral Densities → Relaxation Rates
                ↓          ↓              ↓                    ↓
            (x,y,z)    vectors,       J(ω)              R1, R2, NOE, CCR
                       angles
```

**Key components:**

1. **Geometry Calculator** (JAX)
   - Input: Atomic coordinates from PDB
   - Output: NH bond vectors, internuclear distances, CSA tensor orientations
   - Must be differentiable: `grad(vectors, coordinates)`

2. **Spectral Density Functions** (JAX)
   - Isotropic: Lorentzian `J(ω) = τc / (1 + ω²τc²)`
   - Anisotropic: Includes diffusion tensor D
   - Model-free: Lipari-Szabo formalism
   - Must handle multiple timescales

3. **Relaxation Rate Calculator** (JAX)
   - Standard: R1, R2, NOE (Solomon equations)
   - **CCR: DD-CSA cross-correlation** (YOUR DIFFERENTIATOR)
   - All rates as functions of J(ω)
   - Fully differentiable

### Phase 2: Forward Model (Week 3)

**Complete pipeline:**

```python
def predict_relaxation(structure_params, dynamics_params):
    """
    structure_params: (x, y, z) for all atoms
    dynamics_params: (τc, S², exchange terms)
    
    Returns: Dict of predicted relaxation rates
    """
    # 1. Geometry
    nh_vectors = compute_nh_vectors(structure_params)
    dd_distances = compute_distances(structure_params)
    csa_orientations = compute_csa_angles(structure_params, nh_vectors)
    
    # 2. Spectral densities
    J = spectral_density_model_free(
        nh_vectors, 
        dynamics_params['tau_c'],
        dynamics_params['S2']
    )
    
    # 3. Relaxation rates
    R1 = calculate_R1(J, dd_distances)
    R2 = calculate_R2(J, dd_distances)
    NOE = calculate_NOE(J, dd_distances)
    CCR = calculate_CCR(J, dd_distances, csa_orientations)  # KEY
    
    return {'R1': R1, 'R2': R2, 'NOE': NOE, 'CCR': CCR}
```

**Critical:** Every operation uses JAX primitives for automatic differentiation.

### Phase 3: Inverse Problem - Structure Refinement (Week 4)

**Loss function:**

```python
def loss(structure_params, dynamics_params, experimental_data):
    """
    Compare predicted vs experimental relaxation
    """
    predicted = predict_relaxation(structure_params, dynamics_params)
    
    loss_R1 = jnp.sum((predicted['R1'] - experimental_data['R1'])**2 / errors['R1']**2)
    loss_R2 = jnp.sum((predicted['R2'] - experimental_data['R2'])**2 / errors['R2']**2)
    loss_NOE = jnp.sum((predicted['NOE'] - experimental_data['NOE'])**2 / errors['NOE']**2)
    loss_CCR = jnp.sum((predicted['CCR'] - experimental_data['CCR'])**2 / errors['CCR']**2)
    
    # Optional: structural priors (bond lengths, angles)
    regularization = structure_regularization(structure_params)
    
    return loss_R1 + loss_R2 + loss_NOE + loss_CCR + regularization
```

**Optimization:**

```python
import optax

# Gradient descent with Adam optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(structure_params)

for iteration in range(n_iterations):
    # Compute loss and gradients
    loss_val, grads = jax.value_and_grad(loss)(
        structure_params, 
        dynamics_params, 
        experimental_data
    )
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    structure_params = optax.apply_updates(structure_params, updates)
    
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss_val:.4f}")
```

---

## Implementation Details

### Week 1: Geometry Module

**File:** `diffrelax/geometry.py`

```python
import jax.numpy as jnp
from jax import vmap

def compute_nh_vectors(coords, N_indices, H_indices):
    """
    Compute NH bond vectors
    
    Args:
        coords: (N_atoms, 3) array of coordinates
        N_indices: indices of nitrogen atoms
        H_indices: indices of bonded hydrogens
    
    Returns:
        nh_vectors: (N_residues, 3) normalized vectors
    """
    N_pos = coords[N_indices]
    H_pos = coords[H_indices]
    vectors = H_pos - N_pos
    norms = jnp.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def compute_csa_angles(nh_vectors, csa_tensor_axes):
    """
    Compute angles between NH vectors and CSA principal axes
    Critical for DD-CSA cross-correlation
    """
    # theta: angle between NH and unique CSA axis
    cos_theta = jnp.dot(nh_vectors, csa_tensor_axes)
    theta = jnp.arccos(cos_theta)
    return theta
```

### Week 2: Relaxation Physics

**File:** `diffrelax/relaxation.py`

```python
# Physical constants
GAMMA_H = 2.6752e8  # rad/(s·T)
GAMMA_N = -2.713e7  # rad/(s·T)
H_BAR = 1.054571817e-34  # J·s
MU_0 = 4*jnp.pi*1e-7  # T·m/A

def spectral_density_lorentzian(omega, tau_c):
    """Simple Lorentzian spectral density"""
    return tau_c / (1 + (omega * tau_c)**2)

def R1_dipolar(r_NH, tau_c, omega_H, omega_N):
    """
    Dipolar R1 relaxation rate
    
    R1 = (d²/4) [J(ωH - ωN) + 3J(ωN) + 6J(ωH + ωN)]
    where d = (μ0/4π) γH γN ħ / r³
    """
    d_squared = (MU_0 / (4*jnp.pi))**2 * (GAMMA_H * GAMMA_N * H_BAR)**2 / r_NH**6
    
    J_diff = spectral_density_lorentzian(omega_H - omega_N, tau_c)
    J_N = spectral_density_lorentzian(omega_N, tau_c)
    J_sum = spectral_density_lorentzian(omega_H + omega_N, tau_c)
    
    R1 = (d_squared / 4) * (J_diff + 3*J_N + 6*J_sum)
    return R1

def R2_dipolar(r_NH, tau_c, omega_H, omega_N):
    """
    Dipolar R2 relaxation rate
    
    R2 = (d²/8) [4J(0) + J(ωH - ωN) + 3J(ωN) + 6J(ωH) + 6J(ωH + ωN)]
    """
    d_squared = (MU_0 / (4*jnp.pi))**2 * (GAMMA_H * GAMMA_N * H_BAR)**2 / r_NH**6
    
    J_0 = tau_c  # J(0) = τc for Lorentzian
    J_diff = spectral_density_lorentzian(omega_H - omega_N, tau_c)
    J_N = spectral_density_lorentzian(omega_N, tau_c)
    J_H = spectral_density_lorentzian(omega_H, tau_c)
    J_sum = spectral_density_lorentzian(omega_H + omega_N, tau_c)
    
    R2 = (d_squared / 8) * (4*J_0 + J_diff + 3*J_N + 6*J_H + 6*J_sum)
    return R2

def CCR_DD_CSA(r_NH, tau_c, omega_N, theta, delta_sigma):
    """
    Cross-correlated relaxation between dipolar and CSA
    
    THIS IS YOUR UNIQUE CONTRIBUTION
    
    η = (c·d/4) [4J(0) + 3J(ωN)]
    
    where:
    - d = dipolar coupling constant
    - c = CSA coupling = (ωN·Δσ·P2(cos θ)) / √6
    - P2(x) = (3x² - 1)/2 (second Legendre polynomial)
    - θ = angle between NH and CSA principal axis
    """
    # Dipolar term
    d = (MU_0 / (4*jnp.pi)) * (GAMMA_H * GAMMA_N * H_BAR) / r_NH**3
    
    # CSA term
    P2_cos_theta = (3*jnp.cos(theta)**2 - 1) / 2
    c = (omega_N * delta_sigma * P2_cos_theta) / jnp.sqrt(6)
    
    # Spectral densities
    J_0 = tau_c
    J_N = spectral_density_lorentzian(omega_N, tau_c)
    
    # CCR rate
    eta = (c * d / 4) * (4*J_0 + 3*J_N)
    return eta
```

### Week 3: Integration & Testing

**File:** `diffrelax/model.py`

Combine all components, test gradients:

```python
def test_gradients():
    """Verify automatic differentiation works"""
    
    # Test structure
    coords = jnp.array([[0., 0., 0.], [0., 0., 1.02]])  # N-H bond
    
    # Test gradient flow
    def loss_fn(coords):
        r_NH = jnp.linalg.norm(coords[1] - coords[0])
        R1 = R1_dipolar(r_NH, tau_c=5e-9, omega_H=600e6, omega_N=60e6)
        return R1
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(coords)
    
    print("Gradients:", grads)
    assert jnp.all(jnp.isfinite(grads)), "Gradients must be finite!"
```

---

## Key Innovations

1. **First JAX implementation** of full NMR relaxation theory
2. **CCR is differentiable** - enables gradient-based optimization
3. **End-to-end pipeline** - structure → relaxation → refinement
4. **Validates AlphaFold/cryo-EM** using experimental dynamics data

## Validation Strategy

**Benchmark proteins:**
- GB3 (BMRB 15477, PDB 2OED)
- Ubiquitin (BMRB 6457, PDB 1UBQ)
- TDP-43 CTD (BMRB 26823)

**Tests:**
1. Forward model accuracy: Compare predicted vs experimental R1/R2/NOE
2. Gradient correctness: Finite difference check
3. Refinement: Start with perturbed structure, recover original
4. CCR necessity: Show refinement fails without CCR for certain cases

## Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Geometry + spectral densities | Working forward model (R1, R2, NOE) |
| 2 | CCR implementation + testing | Full relaxation predictor with CCR |
| 3 | Integration + validation | Gradients verified, benchmarks pass |
| 4 | Structure refinement | Working optimizer on test cases |
| 5 | Benchmarking | Results on 5-10 proteins |
| 6 | Analysis | Identify when CCR is essential |
| 7-8 | Writing + code release | Manuscript draft + GitHub repo |

---

## Success Criteria

**Minimum viable:**
- Predicts R1, R2, NOE with <10% error on GB3
- Gradients pass finite difference tests
- Refines perturbed structure back to original

**Publication quality:**
- Works on 5+ different proteins
- Shows CCR improves refinement for flexible/anisotropic regions
- Computational tools released on GitHub
- Clear demonstration of utility for structure validation

## Computational Requirements

**Minimum:** 
- Laptop with 16GB RAM
- CPU-only JAX (slower but works)
- ~1-2 hours per protein for optimization

**Optimal:**
- GPU (NVIDIA with CUDA)
- JAX GPU acceleration
- ~5-10 minutes per protein

---

## Next Steps

1. **TODAY:** Set up environment, fetch BMRB data
2. **Tomorrow:** Implement geometry module + unit tests
3. **Day 3:** Implement R1/R2/NOE forward model
4. **Day 4:** Add CCR physics
5. **Day 5-7:** Test on GB3, verify accuracy

Ready to start coding?