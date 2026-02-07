"""
# ============================================================
# MOLFUN FUSED DFT KERNEL
# ============================================================

This kernel accelerates one of the most expensive steps inside a KS-DFT SCF cycle:
building the real-space XC contribution and projecting it back to the AO matrix.

Where this sits in SCF
----------------------
In each SCF iteration (simplified):
  1) Start from density matrix P (AO basis).
  2) Build electron density on grid points:
         rho(r_g) = sum_{i,j} phi_i(r_g) * P_ij * phi_j(r_g)
  3) Evaluate XC potential on each grid point:
         v_xc(r_g) = f_xc(rho(r_g))
  4) Build AO XC matrix:
         V_xc,ij = sum_g w_g * v_xc(r_g) * phi_i(r_g) * phi_j(r_g)
  5) Build Fock/Kohn-Sham matrix, diagonalize, update P, repeat.

This kernel fuses steps (2) + (3) + (4) into one GPU launch.

What AO and P mean
------------------
AO = Atomic Orbitals.
- In practice, AOs are basis functions centered on atoms (e.g., Gaussian basis).
- If the basis has N functions, indices i,j run from 1..N.
- phi_i(r_g) means: value of AO i evaluated at grid point r_g.
- The matrix phi has shape [n_grid, n_basis], where:
    - n_grid = number of numerical integration points
    - n_basis = number of AO basis functions

P = Density Matrix in the AO basis.
- P_ij encodes electron density / occupancy coupling between AO i and AO j.
- In closed-shell KS-DFT, P is built from occupied molecular orbitals:
      P_ij = 2 * sum_occ C_{i,a} C_{j,a}
  (factor 2 for spin-restricted case).
- P is typically symmetric and has shape [n_basis, n_basis].
- Physically, P + AO basis fully determine rho(r) for a given SCF step.

Algorithmic structure
---------------------
The kernel is grid-tiled by BG (grid points per program):
- Each program handles a chunk of grid points g.
- For each block of basis functions (i, j, size BB):
  - Load AO values phi(g,i), phi(g,j)
  - Load P(i,j)
  - Accumulate rho(g) in FP32 using tiled dot products.
- Apply XC function in-register:
      rho_safe = max(rho, eps)
      v_xc_weighted(g) = w(g) * rho_safe^(1/3)   (LDA-like toy model)
- Re-loop over (i, j) blocks:
  - Compute tile contribution to V_xc:
      V_tile(i,j) += dot( phi_i^T, phi_j * v_xc_weighted )
  - Accumulate into global V using atomic_add.

Why this is faster
------------------
1) Kernel fusion:
   Avoids writing/reading large intermediates (rho, v_xc) from global memory.

2) Tiling:
   Reuses AO/P tiles in SRAM/registers and converts work into dense dot-style math.

3) FP32 accumulation:
   Keeps better numerical stability than pure low-precision accumulation while
   still exploiting GPU throughput.

4) Autotuning:
   Multiple (BG, BB, warps, stages) configs let Triton pick better launch geometry
   for each (n_grid, n_basis) regime.

What was improved in this version
---------------------------------
- Stride-safe density matrix access:
    P is loaded using stride_p_i / stride_p_j instead of assuming contiguous
    row-major layout. This is critical for correctness with views/non-contiguous tensors.
- Wider autotune search:
    Better portability and performance across molecule sizes and GPU architectures.
- Same fused design maintained:
    rho + XC + V assembly remain in one flow to minimize memory traffic.

Numerical behavior
------------------
Small differences vs CPU references are expected due to:
- parallel reduction order,
- FMA behavior on GPU,
- mixed-precision pathways.

Trade-offs / limitations
------------------------
- atomic_add on V is simple and robust, but can become a scalability bottleneck
  at very large n_grid / n_basis (contention).
- A more scalable next step is a 2D tiled V kernel without global atomics
  (each program owns a unique V tile and reduces over grid internally).

In short:
This fused kernel targets the SCF XC-matrix build bottleneck, reducing memory
traffic and improving throughput while preserving good numerical fidelity.
"""


import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BG": 64,  "BB": 16}, num_warps=2, num_stages=2),
        triton.Config({"BG": 128, "BB": 16}, num_warps=4, num_stages=2),
        triton.Config({"BG": 128, "BB": 32}, num_warps=4, num_stages=2),
        triton.Config({"BG": 256, "BB": 32}, num_warps=8, num_stages=2),
    ],
    key=["n_grid", "n_basis"],
)
@triton.jit
def molfun_fused_dft_kernel(
    phi_ptr, p_ptr, v_ptr, w_ptr,
    n_grid, n_basis,
    # phi strides
    stride_phi_g, stride_phi_b,
    # p/dm strides (NEW)
    stride_p_i, stride_p_j,
    # v strides
    stride_v_i, stride_v_j,
    eps,
    BG: tl.constexpr, BB: tl.constexpr,
):
    pid_g = tl.program_id(0)
    offs_g = pid_g * BG + tl.arange(0, BG)
    mask_g = offs_g < n_grid

    rho = tl.zeros((BG,), dtype=tl.float32)

    # --- PHASE 1: Density rho(r) ---
    for i_start in range(0, n_basis, BB):
        offs_i = i_start + tl.arange(0, BB)
        mask_i = offs_i < n_basis

        phi_i = tl.load(
            phi_ptr + offs_g[:, None] * stride_phi_g + offs_i[None, :] * stride_phi_b,
            mask=mask_g[:, None] & mask_i[None, :],
            other=0.0
        ).to(tl.float32)

        tmp_p_phi = tl.zeros((BG, BB), dtype=tl.float32)

        for j_start in range(0, n_basis, BB):
            offs_j = j_start + tl.arange(0, BB)
            mask_j = offs_j < n_basis

            # NEW: use true P strides instead of assuming contiguous n_basis layout
            p_ij = tl.load(
                p_ptr + offs_i[:, None] * stride_p_i + offs_j[None, :] * stride_p_j,
                mask=mask_i[:, None] & mask_j[None, :],
                other=0.0
            ).to(tl.float32)

            phi_j = tl.load(
                phi_ptr + offs_g[:, None] * stride_phi_g + offs_j[None, :] * stride_phi_b,
                mask=mask_g[:, None] & mask_j[None, :],
                other=0.0
            ).to(tl.float32)

            tmp_p_phi += tl.dot(phi_j, tl.trans(p_ij))

        rho += tl.sum(phi_i * tmp_p_phi, axis=1)

    # --- PHASE 2: XC functional (in-register) ---
    weights = tl.load(w_ptr + offs_g, mask=mask_g, other=0.0).to(tl.float32)
    rho_safe = tl.maximum(rho, eps)

    # Stable cubic root via exp(log(x)/3)
    # (Can be swapped for tl.pow(rho_safe, 1/3) if desired)
    v_xc_weighted = tl.exp(tl.log(rho_safe) * (1.0 / 3.0)) * weights

    # --- PHASE 3: Assemble V (atomic add baseline) ---
    for i_start in range(0, n_basis, BB):
        offs_i = i_start + tl.arange(0, BB)
        mask_i = offs_i < n_basis

        phi_i = tl.load(
            phi_ptr + offs_g[:, None] * stride_phi_g + offs_i[None, :] * stride_phi_b,
            mask=mask_g[:, None] & mask_i[None, :],
            other=0.0
        ).to(tl.float32)

        for j_start in range(0, n_basis, BB):
            offs_j = j_start + tl.arange(0, BB)
            mask_j = offs_j < n_basis

            phi_j = tl.load(
                phi_ptr + offs_g[:, None] * stride_phi_g + offs_j[None, :] * stride_phi_b,
                mask=mask_g[:, None] & mask_j[None, :],
                other=0.0
            ).to(tl.float32)

            v_tile = tl.dot(tl.trans(phi_i), (phi_j * v_xc_weighted[:, None]))

            v_ptr_curr = v_ptr + offs_i[:, None] * stride_v_i + offs_j[None, :] * stride_v_j
            tl.atomic_add(v_ptr_curr, v_tile, mask=mask_i[:, None] & mask_j[None, :])


def molfun_dft_predict(phi, dm, weights, eps=1e-12):
    """
    Unified API for benchmarking and inference.
    Inputs:
      - phi: [n_grid, n_basis]
      - dm : [n_basis, n_basis]
      - weights: [n_grid]
    Output:
      - V matrix [n_basis, n_basis], symmetrized
    """
    n_grid, n_basis = phi.shape
    v_out = torch.zeros((n_basis, n_basis), device=phi.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(n_grid, META["BG"]),)

    molfun_fused_dft_kernel[grid](
        phi, dm, v_out, weights,
        n_grid, n_basis,
        # phi strides
        phi.stride(0), phi.stride(1),
        # dm strides (NEW)
        dm.stride(0), dm.stride(1),
        # v strides
        v_out.stride(0), v_out.stride(1),
        eps,
    )

    # Enforce symmetry
    return 0.5 * (v_out + v_out.T)
