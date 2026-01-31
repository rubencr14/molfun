"""
test_rmsd_performance.py

Performance test for RMSD calculation comparing MolfunAnalysis, MDTraj, and MDAnalysis.

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
TOPOLOGY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.prmtop"
TRAJECTORY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.dcd"

# Number of timing runs for stable measurements
N_RUNS = 5
N_WARMUP = 20


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
    
    # AnalysisMD (MDAnalysis)
    print("\n--- AnalysisMD (MDAnalysis) ---")
    start = time.perf_counter()
    mda_obj = AnalysisMD(TOPOLOGY_PATH, TRAJECTORY_PATH, selection="protein")
    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {mda_obj.n_atoms}, Frames: {mda_obj.n_frames}")
    
    # Warmup
    print("Warming up...")
    _ = [mda_obj.rmsd(0, f, superposition=True) for f in range(10)]
    
    # Measure (single run - MDAnalysis is slow)
    print(f"Measuring (1 run - frame-by-frame is slow)...")
    t0 = time.perf_counter()
    rmsds_mda = [mda_obj.rmsd(0, f, superposition=True) for f in range(n_frames)]
    t1 = time.perf_counter()
    mda_time = (t1 - t0) * 1000  # ms
    print(f"Time: {mda_time:.2f}ms ({mda_time/n_frames*1000:.2f}µs/frame)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(f"Total frames: {n_frames}, Atoms: {molfun.n_atoms}")
    print(f"MolfunAnalysis: {molfun_time:.2f}ms total ({molfun_time/n_frames*1000:.2f}µs/frame)")
    print(f"TrajAnalysis:   {traj_time:.2f}ms total ({traj_time/n_frames*1000:.2f}µs/frame) | Speedup: {traj_time/molfun_time:.1f}x")
    print(f"AnalysisMD:     {mda_time:.2f}ms total ({mda_time/n_frames*1000:.2f}µs/frame) | Speedup: {mda_time/molfun_time:.1f}x")
    
    # Verify correctness - compare all frames
    rmsds_molfun_np = rmsds_molfun.cpu().numpy()
    rmsds_traj_np = np.array(rmsds_traj)
    rmsds_mda_np = np.array(rmsds_mda)
    
    max_diff_traj = np.abs(rmsds_molfun_np - rmsds_traj_np).max()
    max_diff_mda = np.abs(rmsds_molfun_np - rmsds_mda_np).max()
    mean_diff_traj = np.abs(rmsds_molfun_np - rmsds_traj_np).mean()
    mean_diff_mda = np.abs(rmsds_molfun_np - rmsds_mda_np).mean()
    
    print(f"\nCorrectness check (all {n_frames} frames):")
    print(f"  Max diff vs MDTraj:    {max_diff_traj:.2e} Å")
    print(f"  Mean diff vs MDTraj:  {mean_diff_traj:.2e} Å")
    print(f"  Max diff vs MDAnalysis: {max_diff_mda:.2e} Å")
    print(f"  Mean diff vs MDAnalysis: {mean_diff_mda:.2e} Å")
    
    # Assert that values match (tolerance allows for numerical differences between implementations)
    assert max_diff_traj < 0.02, f"MolfunAnalysis vs MDTraj mismatch: {max_diff_traj:.2e} Å"
    assert max_diff_mda < 0.02, f"MolfunAnalysis vs MDAnalysis mismatch: {max_diff_mda:.2e} Å"
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_rmsd_performance()
