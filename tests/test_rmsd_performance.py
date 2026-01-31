"""
test_rmsd_performance.py

Performance test for RMSD calculation comparing MolfunAnalysis, MDTraj, and MDAnalysis.
"""

import time
import pytest
import torch
import numpy as np

from molfun.analysis.md import MolfunAnalysis, TrajAnalysis, AnalysisMD


# Paths to MD files
TOPOLOGY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.prmtop"
TRAJECTORY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.dcd"


def test_rmsd_performance():
    """Test RMSD performance for MolfunAnalysis, MDTraj, and MDAnalysis."""
    
    print("\n" + "=" * 80)
    print("RMSD Performance Test")
    print("=" * 80)
    
    # MolfunAnalysis
    print("\n--- MolfunAnalysis (Triton GPU) ---")
    start = time.time()
    molfun = MolfunAnalysis(TOPOLOGY_PATH, TRAJECTORY_PATH, selection="protein")
    load_time = time.time() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {molfun.n_atoms}, Frames: {molfun.n_frames}")
    
    # Use all frames for performance test
    n_frames = molfun.n_frames
    print(f"Testing with all {n_frames} frames")
    
    # Warmup: compile Triton kernels and initialize GPU
    print("Warming up (compiling kernels)...")
    _ = molfun.rmsd(0, superposition=True)[:10]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Now measure actual performance - MolfunAnalysis calculates ALL frames in one batch
    print(f"Calculating RMSD for all {n_frames} frames (single GPU batch operation)...")
    start = time.time()
    rmsds_molfun = molfun.rmsd(0, superposition=True)  # Returns RMSD for all frames
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    molfun_time = time.time() - start
    print(f"RMSD calculation time (all {n_frames} frames): {molfun_time:.3f}s ({molfun_time*1000/n_frames:.3f}ms/frame)")
    
    # TrajAnalysis (MDTraj)
    print("\n--- TrajAnalysis (MDTraj) ---")
    start = time.time()
    traj = TrajAnalysis(TOPOLOGY_PATH, TRAJECTORY_PATH, selection="protein")
    load_time = time.time() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {traj.n_atoms}, Frames: {traj.n_frames}")
    
    # Warmup
    print("Warming up...")
    _ = [traj.rmsd(0, f) for f in range(10)]
    
    # Now measure actual performance - process all frames
    print(f"Calculating RMSD for all {n_frames} frames (frame-by-frame)...")
    start = time.time()
    rmsds_traj = [traj.rmsd(0, f) for f in range(n_frames)]
    traj_time = time.time() - start
    print(f"RMSD calculation time (all {n_frames} frames): {traj_time:.3f}s ({traj_time*1000/n_frames:.3f}ms/frame)")
    
    # AnalysisMD (MDAnalysis)
    print("\n--- AnalysisMD (MDAnalysis) ---")
    start = time.time()
    mda_obj = AnalysisMD(TOPOLOGY_PATH, TRAJECTORY_PATH, selection="protein")
    load_time = time.time() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {mda_obj.n_atoms}, Frames: {mda_obj.n_frames}")
    
    # Warmup
    print("Warming up...")
    _ = [mda_obj.rmsd(0, f, superposition=True) for f in range(10)]
    
    # Now measure actual performance - process all frames
    print(f"Calculating RMSD for all {n_frames} frames (frame-by-frame)...")
    start = time.time()
    rmsds_mda = [mda_obj.rmsd(0, f, superposition=True) for f in range(n_frames)]
    mda_time = time.time() - start
    print(f"RMSD calculation time (all {n_frames} frames): {mda_time:.3f}s ({mda_time*1000/n_frames:.3f}ms/frame)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(f"Total frames processed: {n_frames}")
    print(f"MolfunAnalysis: {molfun_time:.3f}s ({molfun_time*1000/n_frames:.3f}ms/frame)")
    print(f"TrajAnalysis:   {traj_time:.3f}s ({traj_time*1000/n_frames:.3f}ms/frame) | Speedup: {traj_time/molfun_time:.1f}x")
    print(f"AnalysisMD:     {mda_time:.3f}s ({mda_time*1000/n_frames:.3f}ms/frame) | Speedup: {mda_time/molfun_time:.1f}x")
    
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
    
    # Assert that values match (within numerical precision)
    assert max_diff_traj < 0.01, f"MolfunAnalysis vs MDTraj mismatch: {max_diff_traj:.2e} Å"
    assert max_diff_mda < 0.01, f"MolfunAnalysis vs MDAnalysis mismatch: {max_diff_mda:.2e} Å"
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_rmsd_performance()
