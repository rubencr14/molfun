"""
test_contact_map_performance.py

Performance test for contact map calculation comparing:
- MolfunAnalysis (Triton GPU - optimized bit-packed kernels)
- PyTorch GPU (vectorized baseline - fair GPU vs GPU comparison)
- MDTraj (CPU)
- MDAnalysis (CPU)

Uses proper GPU timing with:
- time.perf_counter() for high precision
- torch.cuda.synchronize() before/after GPU operations
- Multiple runs for stable measurements
"""

import time
import pytest
import torch
import numpy as np

from molfun.analysis.md import MolfunAnalysis, TrajAnalysis, AnalysisMD


# Paths to MD files
TOPOLOGY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.prmtop"
TRAJECTORY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.dcd"

# Timing parameters
N_RUNS = 5
N_WARMUP = 10


def contact_map_pytorch_gpu_batch(coords: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    PyTorch GPU baseline: Batch contact map using only PyTorch ops.
    
    This is a vectorized GPU implementation WITHOUT Triton kernels.
    Used as a fair GPU-vs-GPU baseline to show Triton optimization gains.
    
    Args:
        coords: [M, N, 3] coordinates (CUDA) or [N, 3] for single frame
        cutoff: Distance cutoff in Angstrom
    
    Returns:
        contact_maps: [M, N, N] boolean contact maps (or [N, N] for single)
    """
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)  # [1, N, 3]
    
    M, N, _ = coords.shape
    
    # Compute pairwise distances: [M, N, N]
    # diff[m, i, j] = coords[m, i] - coords[m, j]
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [M, N, 1, 3] - [M, 1, N, 3] = [M, N, N, 3]
    dist_sq = (diff ** 2).sum(dim=-1)  # [M, N, N]
    
    # Contact map: distance < cutoff (using squared to avoid sqrt)
    cutoff_sq = cutoff ** 2
    contact_maps = dist_sq < cutoff_sq  # [M, N, N]
    
    # Zero diagonal (self-contacts)
    diag_mask = torch.eye(N, dtype=torch.bool, device=coords.device)
    contact_maps = contact_maps & ~diag_mask.unsqueeze(0)
    
    return contact_maps.squeeze(0) if M == 1 else contact_maps


def test_contact_map_performance():
    """Test contact map performance for MolfunAnalysis, MDTraj, and MDAnalysis."""
    
    print("\n" + "=" * 80)
    print("Contact Map Performance Test")
    print("=" * 80)
    
    # Parameters
    cutoff = 8.0  # Angstrom
    selection = "ca"  # Alpha carbons
    
    # MolfunAnalysis
    print("\n--- MolfunAnalysis (Triton GPU) ---")
    start = time.perf_counter()
    molfun = MolfunAnalysis(TOPOLOGY_PATH, TRAJECTORY_PATH, selection=selection)
    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {molfun.n_atoms}, Frames: {molfun.n_frames}")
    
    # Use ALL frames
    n_frames = molfun.n_frames
    print(f"Testing with all {n_frames} frames")
    
    # Warmup: compile Triton kernels and initialize GPU
    print("Warming up (compiling kernels)...")
    _ = molfun.contact_map(cutoff, frame=0)
    torch.cuda.synchronize()
    
    # Batch test with multiple runs (ALL frames)
    print(f"\nBatch contact map (all {n_frames} frames, cutoff={cutoff}Å):")
    molfun_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cm_molfun_batch = molfun.contact_map(cutoff)  # ALL frames
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        molfun_times.append((t1 - t0) * 1000)  # ms
    
    molfun_time_batch = sum(molfun_times) / len(molfun_times)
    print(f"  Run times: {[f'{t:.1f}ms' for t in molfun_times]}")
    print(f"  Mean: {molfun_time_batch:.2f}ms ({molfun_time_batch/n_frames:.3f}ms/frame)")
    
    # PyTorch GPU baseline
    print("\n--- PyTorch GPU (vectorized baseline) ---")
    print("Using same GPU-cached coordinates as Molfun")
    
    # Get coordinates (ALL frames)
    coords_batch = molfun.get_coords()  # [n_frames, N, 3]
    
    # Warmup
    print(f"Warming up ({N_WARMUP} iterations)...")
    for _ in range(N_WARMUP):
        for f in range(min(10, n_frames)):
            _ = contact_map_pytorch_gpu_batch(coords_batch[f], cutoff)
    torch.cuda.synchronize()
    
    # Measure - frame by frame (fair comparison with Molfun's loop)
    print(f"Measuring ({N_RUNS} runs, all {n_frames} frames each)...")
    pytorch_times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for f in range(n_frames):
            cm_pytorch = contact_map_pytorch_gpu_batch(coords_batch[f], cutoff)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        pytorch_times.append((t1 - t0) * 1000)  # ms
    
    pytorch_time_batch = sum(pytorch_times) / len(pytorch_times)
    print(f"  Run times: {[f'{t:.1f}ms' for t in pytorch_times]}")
    print(f"  Mean: {pytorch_time_batch:.2f}ms ({pytorch_time_batch/n_frames:.3f}ms/frame)")
    
    # TrajAnalysis (MDTraj)
    print("\n--- TrajAnalysis (MDTraj) ---")
    start = time.perf_counter()
    traj = TrajAnalysis(TOPOLOGY_PATH, TRAJECTORY_PATH, selection=selection)
    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {traj.n_atoms}, Frames: {traj.n_frames}")
    
    # Warmup
    print("Warming up...")
    _ = traj.contact_map(cutoff, frame=0)
    
    # Batch test (ALL frames)
    print(f"\nBatch contact map (all {n_frames} frames, cutoff={cutoff}Å):")
    t0 = time.perf_counter()
    cm_traj_batch = traj.contact_map(cutoff)  # ALL frames
    t1 = time.perf_counter()
    traj_time_batch = (t1 - t0) * 1000  # ms
    print(f"  Time: {traj_time_batch:.2f}ms ({traj_time_batch/n_frames:.3f}ms/frame)")
    
    # AnalysisMD (MDAnalysis)
    print("\n--- AnalysisMD (MDAnalysis) ---")
    start = time.perf_counter()
    mda_obj = AnalysisMD(TOPOLOGY_PATH, TRAJECTORY_PATH, selection=selection)
    load_time = time.perf_counter() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {mda_obj.n_atoms}, Frames: {mda_obj.n_frames}")
    
    # Warmup
    print("Warming up...")
    _ = mda_obj.contact_map(cutoff, frame=0)
    
    # Batch test (ALL frames)
    print(f"\nBatch contact map (all {n_frames} frames, cutoff={cutoff}Å):")
    t0 = time.perf_counter()
    cm_mda_batch = mda_obj.contact_map(cutoff)  # ALL frames
    t1 = time.perf_counter()
    mda_time_batch = (t1 - t0) * 1000  # ms
    print(f"  Time: {mda_time_batch:.2f}ms ({mda_time_batch/n_frames:.3f}ms/frame)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(f"All {n_frames} frames, {molfun.n_atoms} atoms (CA), cutoff={cutoff}Å")
    print()
    print("GPU Methods:")
    print(f"  Molfun (Triton):   {molfun_time_batch:.2f}ms ({molfun_time_batch/n_frames:.4f}ms/frame)")
    print(f"  PyTorch GPU:       {pytorch_time_batch:.2f}ms ({pytorch_time_batch/n_frames:.4f}ms/frame) | vs Molfun: {pytorch_time_batch/molfun_time_batch:.1f}x slower")
    print()
    print("CPU Methods:")
    print(f"  MDTraj:            {traj_time_batch:.2f}ms ({traj_time_batch/n_frames:.4f}ms/frame) | vs Molfun: {traj_time_batch/molfun_time_batch:.1f}x slower")
    print(f"  MDAnalysis:        {mda_time_batch:.2f}ms ({mda_time_batch/n_frames:.4f}ms/frame) | vs Molfun: {mda_time_batch/molfun_time_batch:.1f}x slower")
    
    # Verify correctness - count contacts in frame 10
    from molfun.kernels.analysis.contact_map_atoms import unpack_contact_map
    
    frame_idx = 10
    
    # Unpack MolfunAnalysis bitpacked output for frame 10
    cm_molfun_unpacked = unpack_contact_map(cm_molfun_batch[frame_idx], molfun.n_atoms).cpu().numpy()
    
    # Count contacts
    contacts_molfun = np.sum(cm_molfun_unpacked)
    contacts_traj = np.sum(cm_traj_batch[frame_idx])
    contacts_mda = np.sum(cm_mda_batch[frame_idx])
    
    print(f"\nCorrectness check (frame {frame_idx} contact counts):")
    print(f"  MolfunAnalysis: {contacts_molfun} contacts")
    print(f"  TrajAnalysis:   {contacts_traj} contacts | Diff: {abs(contacts_molfun - contacts_traj)}")
    print(f"  AnalysisMD:     {contacts_mda} contacts | Diff: {abs(contacts_molfun - contacts_mda)}")
    
    # Assert that contact counts match (should be identical)
    assert contacts_molfun == contacts_traj, f"MolfunAnalysis vs MDTraj contact count mismatch: {contacts_molfun} vs {contacts_traj}"
    assert contacts_molfun == contacts_mda, f"MolfunAnalysis vs MDAnalysis contact count mismatch: {contacts_molfun} vs {contacts_mda}"
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_contact_map_performance()
