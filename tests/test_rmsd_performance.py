"""
test_rmsd_performance.py

Performance test for RMSD calculation comparing:
- MolfunAnalysis (Triton GPU - optimized fused kernels)
- PyTorch GPU (vectorized baseline - fair GPU vs GPU comparison)
- MDTraj (CPU - optimized C/Cython)
- MDAnalysis (CPU - Python/C)

Uses proper GPU timing with:
- time.perf_counter() for high precision
- torch.cuda.synchronize() before/after GPU operations
- Multiple runs for stable measurements
"""

import time
import pytest
import torch
import numpy as np
import mdtraj

from molfun.analysis.md import MolfunAnalysis, TrajAnalysis, AnalysisMD


# Paths to MD files
#TOPOLOGY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.prmtop"
#TRAJECTORY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.dcd"
TOPOLOGY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_INH_Reduced.prmtop"
TRAJECTORY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_INH_Reduced.dcd"
# Number of timing runs for stable measurements
N_RUNS = 5
N_WARMUP = 20


def rmsd_pytorch_gpu_batch(ref: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
    """
    PyTorch GPU baseline: Batch RMSD with Kabsch alignment using only PyTorch ops.
    
    This is a vectorized GPU implementation WITHOUT Triton kernels.
    Used as a fair GPU-vs-GPU baseline to show Triton optimization gains.
    
    Args:
        ref: [N, 3] reference coordinates (CUDA)
        traj: [M, N, 3] trajectory coordinates (CUDA)
    
    Returns:
        rmsds: [M] RMSD values
    """
    M, N, _ = traj.shape
    
    # Step 1: Compute centroids
    ref_centroid = ref.mean(dim=0, keepdim=True)  # [1, 3]
    traj_centroids = traj.mean(dim=1, keepdim=True)  # [M, 1, 3]
    
    # Step 2: Center coordinates
    ref_centered = ref - ref_centroid  # [N, 3]
    traj_centered = traj - traj_centroids  # [M, N, 3]
    
    # Step 3: Compute covariance matrices H = ref^T @ traj for each frame
    # ref_centered: [N, 3] -> expand to [M, N, 3]
    ref_expanded = ref_centered.unsqueeze(0).expand(M, -1, -1)  # [M, N, 3]
    # H = sum over atoms of outer product: [M, 3, 3]
    H = torch.bmm(ref_expanded.transpose(1, 2), traj_centered)  # [M, 3, 3]
    
    # Step 4: SVD
    U, S, Vt = torch.linalg.svd(H)  # U: [M,3,3], S: [M,3], Vt: [M,3,3]
    
    # Step 5: Handle reflections
    det = torch.det(torch.bmm(U, Vt))  # [M]
    
    # Step 6: Compute RMSD using closed form
    # RMSD² = (||ref_c||² + ||traj_c||² - 2*trace) / N
    ref_norm_sq = (ref_centered ** 2).sum()  # scalar
    traj_norm_sq = (traj_centered ** 2).sum(dim=(1, 2))  # [M]
    
    # trace = sum of singular values (with sign correction for reflections)
    trace = S[:, 0] + S[:, 1] + torch.where(det >= 0, S[:, 2], -S[:, 2])
    
    rmsd_sq = (ref_norm_sq + traj_norm_sq - 2.0 * trace) / float(N)
    rmsd_sq = torch.clamp(rmsd_sq, min=0.0)
    
    return torch.sqrt(rmsd_sq)


def test_rmsd_performance():
    """Test RMSD performance for MolfunAnalysis, MDTraj, and MDAnalysis."""
    
    print("\n" + "=" * 80)
    print("RMSD Performance Test")
    print("=" * 80)
    
    # MolfunAnalysis
    print("\n--- MolfunAnalysis (Triton GPU) ---")
    start = time.perf_counter()
    molfun = MolfunAnalysis(TOPOLOGY_PATH, TRAJECTORY_PATH, selection="protein")
    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {molfun.n_atoms}, Frames: {molfun.n_frames}")
    
    n_frames = molfun.n_frames
    print(f"Testing with all {n_frames} frames")
    
    # Warmup: compile Triton kernels and initialize GPU (multiple iterations)
    print(f"Warming up ({N_WARMUP} iterations)...")
    for _ in range(N_WARMUP):
        _ = molfun.rmsd(0, superposition=True)
    torch.cuda.synchronize()
    
    # Measure with multiple runs for stable timing
    print(f"Measuring ({N_RUNS} runs)...")
    molfun_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rmsds_molfun = molfun.rmsd(0, superposition=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        molfun_times.append((t1 - t0) * 1000)  # ms
    
    molfun_time = sum(molfun_times) / len(molfun_times)  # mean in ms
    print(f"Run times: {[f'{t:.2f}ms' for t in molfun_times]}")
    print(f"Mean: {molfun_time:.2f}ms ({molfun_time/n_frames*1000:.2f}µs/frame)")
    
    # PyTorch GPU baseline (fair GPU vs GPU comparison)
    print("\n--- PyTorch GPU (vectorized baseline) ---")
    print("Using same GPU-cached coordinates as Molfun")
    
    # Get coordinates from MolfunAnalysis (already on GPU)
    ref_coords = molfun.get_coords(0)  # [N, 3]
    traj_coords = molfun.get_coords()  # [M, N, 3]
    
    # Warmup
    print(f"Warming up ({N_WARMUP} iterations)...")
    for _ in range(N_WARMUP):
        _ = rmsd_pytorch_gpu_batch(ref_coords, traj_coords)
    torch.cuda.synchronize()
    
    # Measure
    print(f"Measuring ({N_RUNS} runs)...")
    pytorch_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rmsds_pytorch = rmsd_pytorch_gpu_batch(ref_coords, traj_coords)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pytorch_times.append((t1 - t0) * 1000)  # ms
    
    pytorch_time = sum(pytorch_times) / len(pytorch_times)  # mean in ms
    print(f"Run times: {[f'{t:.2f}ms' for t in pytorch_times]}")
    print(f"Mean: {pytorch_time:.2f}ms ({pytorch_time/n_frames*1000:.2f}µs/frame)")
    
    # TrajAnalysis (MDTraj)
    print("\n--- TrajAnalysis (MDTraj) ---")
    start = time.perf_counter()
    traj = TrajAnalysis(TOPOLOGY_PATH, TRAJECTORY_PATH, selection="protein")
    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {traj.n_atoms}, Frames: {traj.n_frames}")
    
    # Warmup
    print(f"Warming up ({N_WARMUP} iterations)...")
    for _ in range(N_WARMUP):
        _ = mdtraj.rmsd(traj.traj, traj.traj[0]) * 10.0
    
    # Measure with multiple runs
    print(f"Measuring ({N_RUNS} runs)...")
    traj_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        rmsds_traj = mdtraj.rmsd(traj.traj, traj.traj[0]) * 10.0
        t1 = time.perf_counter()
        traj_times.append((t1 - t0) * 1000)  # ms
    
    traj_time = sum(traj_times) / len(traj_times)  # mean in ms
    print(f"Run times: {[f'{t:.2f}ms' for t in traj_times]}")
    print(f"Mean: {traj_time:.2f}ms ({traj_time/n_frames*1000:.2f}µs/frame)")
    
    # AnalysisMD (MDAnalysis) - using optimized RMSD.run() API
    print("\n--- AnalysisMD (MDAnalysis) ---")
    from MDAnalysis.analysis.rms import RMSD as MDA_RMSD
    import MDAnalysis as mda
    
    start = time.perf_counter()
    universe = mda.Universe(TOPOLOGY_PATH, TRAJECTORY_PATH)
    atoms = universe.select_atoms("protein")
    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {len(atoms)}, Frames: {len(universe.trajectory)}")
    
    # Warmup with optimized RMSD.run()
    print("Warming up (RMSD.run() batch API)...")
    rmsd_analysis = MDA_RMSD(universe, universe, select='protein', ref_frame=0)
    rmsd_analysis.run()
    
    # Measure with multiple runs using optimized batch API
    print(f"Measuring ({N_RUNS} runs with RMSD.run())...")
    mda_times = []
    for _ in range(N_RUNS):
        rmsd_analysis = MDA_RMSD(universe, universe, select='protein', ref_frame=0)
        t0 = time.perf_counter()
        rmsd_analysis.run()
        t1 = time.perf_counter()
        mda_times.append((t1 - t0) * 1000)  # ms
    
    mda_time = sum(mda_times) / len(mda_times)  # mean in ms
    rmsds_mda = rmsd_analysis.results.rmsd[:, 2]  # Column 2 = RMSD values in Angstrom
    print(f"Run times: {[f'{t:.2f}ms' for t in mda_times]}")
    print(f"Mean: {mda_time:.2f}ms ({mda_time/n_frames*1000:.2f}µs/frame)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(f"Total frames: {n_frames}, Atoms: {molfun.n_atoms}")
    print()
    print("GPU Methods:")
    print(f"  Molfun (Triton):   {molfun_time:.2f}ms ({molfun_time/n_frames*1000:.2f}µs/frame)")
    print(f"  PyTorch GPU:       {pytorch_time:.2f}ms ({pytorch_time/n_frames*1000:.2f}µs/frame) | vs Molfun: {pytorch_time/molfun_time:.1f}x slower")
    print()
    print("CPU Methods:")
    print(f"  MDTraj (C/Cython): {traj_time:.2f}ms ({traj_time/n_frames*1000:.2f}µs/frame) | vs Molfun: {traj_time/molfun_time:.1f}x slower")
    print(f"  MDAnalysis:        {mda_time:.2f}ms ({mda_time/n_frames*1000:.2f}µs/frame) | vs Molfun: {mda_time/molfun_time:.1f}x slower")
    
    # Verify correctness - compare all frames
    rmsds_molfun_np = rmsds_molfun.cpu().numpy()
    rmsds_pytorch_np = rmsds_pytorch.cpu().numpy()
    rmsds_traj_np = np.array(rmsds_traj)
    rmsds_mda_np = np.array(rmsds_mda)
    
    max_diff_pytorch = np.abs(rmsds_molfun_np - rmsds_pytorch_np).max()
    max_diff_traj = np.abs(rmsds_molfun_np - rmsds_traj_np).max()
    max_diff_mda = np.abs(rmsds_molfun_np - rmsds_mda_np).max()
    
    print(f"\nCorrectness check (all {n_frames} frames):")
    print(f"  Max diff vs PyTorch GPU: {max_diff_pytorch:.2e} Å")
    print(f"  Max diff vs MDTraj:      {max_diff_traj:.2e} Å")
    print(f"  Max diff vs MDAnalysis:  {max_diff_mda:.2e} Å")
    
    # Assert that values match (tolerance allows for numerical differences between implementations)
    assert max_diff_pytorch < 0.02, f"Molfun vs PyTorch GPU mismatch: {max_diff_pytorch:.2e} Å"
    assert max_diff_traj < 0.02, f"Molfun vs MDTraj mismatch: {max_diff_traj:.2e} Å"
    assert max_diff_mda < 0.02, f"Molfun vs MDAnalysis mismatch: {max_diff_mda:.2e} Å"
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_rmsd_performance()
