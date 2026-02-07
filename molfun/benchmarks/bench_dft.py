import time
import numpy as np
import torch
from pyscf import gto, dft

from molfun.kernels.models.fused_dft import molfun_dft_predict


def ms(fn, n_repeat=10, sync_cuda=False):
    """
    Timing helper that returns the median runtime in milliseconds.
    Optionally synchronizes CUDA before/after each call for accurate GPU timing.
    """
    times = []
    for _ in range(n_repeat):
        if sync_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def metrics(ref: np.ndarray, pred: np.ndarray):
    """
    Compute error metrics between reference and prediction:
      - max absolute error
      - mean absolute error (MAE)
      - relative Frobenius norm error
    """
    diff = pred - ref
    max_abs = float(np.max(np.abs(diff)))
    mae = float(np.mean(np.abs(diff)))
    fro_rel = float(np.linalg.norm(diff) / (np.linalg.norm(ref) + 1e-12))
    return max_abs, mae, fro_rel


def print_metrics_block(title, ref, pred):
    """
    Print a formatted error block and return the metrics tuple.
    """
    max_abs, mae, fro_rel = metrics(ref, pred)
    print(f"\n[{title}]")
    print(f"  max_abs : {max_abs:.3e}")
    print(f"  mae     : {mae:.3e}")
    print(f"  fro_rel : {fro_rel:.3e}")
    return max_abs, mae, fro_rel


def run_dft_comparison():
    assert torch.cuda.is_available(), "CUDA required"

    # ---------------------------
    # 1) Build test system
    # ---------------------------
    mol = gto.M(
        atom="C 0 0 0; C 1.39 0 0; C 2.09 1.2 0; C 1.39 2.4 0; C 0 2.4 0; C -0.7 1.2 0",
        basis="6-31g",
        verbose=0,
    )
    mol = gto.M(
        atom="""
        C   0.0000   1.4027   0.0000
        C   1.2148   0.7014   0.0000
        C   1.2148  -0.7014   0.0000
        C   0.0000  -1.4027   0.0000
        C  -1.2148  -0.7014   0.0000
        C  -1.2148   0.7014   0.0000
        C   2.4296   1.4027   0.0000
        C   3.6444   0.7014   0.0000
        C   3.6444  -0.7014   0.0000
        C   2.4296  -1.4027   0.0000
        H   0.0000   2.4900   0.0000
        H   2.4296   2.4900   0.0000
        H   4.5860   1.2450   0.0000
        H   4.5860  -1.2450   0.0000
        H   2.4296  -2.4900   0.0000
        H   0.0000  -2.4900   0.0000
        H  -2.1560  -1.2450   0.0000
        H  -2.1560   1.2450   0.0000
        """,
        basis="6-31g",
        unit="Angstrom",
        verbose=0
    )
    grids = dft.gen_grid.Grids(mol)
    grids.level = 4
    grids.build()

    ni = dft.numint.NumInt()
    phi = ni.eval_ao(mol, grids.coords).astype(np.float32)  # shape: [G, B]
    n_grid, n_basis = phi.shape
    weights = grids.weights.astype(np.float32)

    # Deterministic density matrix for reproducibility
    np.random.seed(42)
    a = np.random.randn(n_basis, n_basis).astype(np.float32)
    dm = (a @ a.T).astype(np.float32)  # symmetric positive-ish

    print(f"\n🧪 Molecule: Benzene | Grid: {n_grid} | Basis: {n_basis}")
    print("-" * 72)

    # ---------------------------
    # 2) PySCF CPU reference
    # ---------------------------
    def pyscf_run():
        rho_p = ni.eval_rho(mol, phi, dm)
        rho_p = np.maximum(rho_p, 1e-12)
        vxc_p = rho_p ** (1 / 3)
        v_native = dft.numint.eval_mat(mol, phi, weights, rho_p, vxc_p)
        return v_native

    # One run for the output matrix + robust CPU timing
    v_native = pyscf_run()
    t_pyscf = ms(lambda: pyscf_run(), n_repeat=7, sync_cuda=False)

    # ---------------------------
    # 3) Torch GPU equivalent reference
    #    (same algebra as the Triton kernel)
    # ---------------------------
    phi_t = torch.from_numpy(phi).cuda().float()
    dm_t = torch.from_numpy(dm).cuda().float()
    w_t = torch.from_numpy(weights).cuda().float()

    @torch.no_grad()
    def torch_ref_run():
        # rho[g] = phi[g,:] @ dm @ phi[g,:]^T
        rho = torch.einsum("gi,ij,gj->g", phi_t, dm_t, phi_t)
        rho = torch.clamp(rho, min=1e-12)
        vxc = rho.pow(1.0 / 3.0)
        v = torch.einsum("g,gi,gj->ij", w_t * vxc, phi_t, phi_t)
        v = 0.5 * (v + v.T)  # enforce symmetry
        return v

    # GPU warmup
    for _ in range(3):
        _ = torch_ref_run()
    torch.cuda.synchronize()

    v_torch_ref_t = torch_ref_run()
    v_torch_ref = v_torch_ref_t.cpu().numpy()
    t_torch_ref = ms(lambda: torch_ref_run(), n_repeat=20, sync_cuda=True)

    # ---------------------------
    # 4) Molfun Triton GPU
    # ---------------------------
    @torch.no_grad()
    def molfun_run():
        return molfun_dft_predict(phi_t, dm_t, w_t)

    # Warmup for compile + autotune
    for _ in range(5):
        _ = molfun_run()
    torch.cuda.synchronize()

    v_molfun_t = molfun_run()
    v_molfun = v_molfun_t.cpu().numpy()
    t_molfun = ms(lambda: molfun_run(), n_repeat=20, sync_cuda=True)

    # ---------------------------
    # 5) Value inspection
    # ---------------------------
    print("\n🔍 5x5 corner of V:")
    print("\n--- PySCF CPU ---")
    print(np.round(v_native[:5, :5], 4))
    print("\n--- Torch GPU ref ---")
    print(np.round(v_torch_ref[:5, :5], 4))
    print("\n--- Molfun Triton GPU ---")
    print(np.round(v_molfun[:5, :5], 4))

    # ---------------------------
    # 6) Timings + error metrics
    # ---------------------------
    print("\n" + "=" * 72)
    print("⏱️  TIMINGS (median):")
    print(f"  PySCF CPU        : {t_pyscf:10.3f} ms")
    print(f"  Torch GPU ref    : {t_torch_ref:10.3f} ms")
    print(f"  Molfun Triton GPU: {t_molfun:10.3f} ms")
    print("-" * 72)
    print(f"🚀 Speedup vs PySCF CPU : {t_pyscf / t_molfun:8.2f}x")
    print(f"🚀 Speedup vs Torch GPU : {t_torch_ref / t_molfun:8.2f}x")

    # Triton vs Torch (true algebraic equivalence target)
    mx1, mae1, fro1 = print_metrics_block("Triton vs Torch-GPU-ref", v_torch_ref, v_molfun)
    # Triton vs PySCF (may differ due to internal implementation details)
    mx2, mae2, fro2 = print_metrics_block("Triton vs PySCF-CPU", v_native, v_molfun)

    # Symmetry check
    sym = float(np.max(np.abs(v_molfun - v_molfun.T)))
    print(f"\n[Triton symmetry]")
    print(f"  max |V - V^T|: {sym:.3e}")
    print("=" * 72)

    # ---------------------------
    # 7) Validation criteria
    # ---------------------------
    # Main validation (against equivalent Torch algebra)
    ok_torch = fro1 < 2e-3 and mx1 < 1e-2 and sym < 1e-6

    # Relaxed consistency check against PySCF
    ok_pyscf = fro2 < 5e-3 and mx2 < 2e-2

    if ok_torch:
        print("✅ MAIN VALIDATION OK (Triton ≈ equivalent Torch reference).")
    else:
        print("❌ MAIN VALIDATION FAILED (Triton vs Torch). Check kernel/strides/precision.")

    if ok_pyscf:
        print("✅ CONSISTENCY WITH PySCF is reasonable.")
    else:
        print("⚠️ Difference vs PySCF is larger than expected (possibly convention/precision).")


if __name__ == "__main__":
    run_dft_comparison()
