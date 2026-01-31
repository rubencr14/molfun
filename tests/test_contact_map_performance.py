"""
test_contact_map_performance.py

Performance test for contact map calculation comparing MolfunAnalysis, MDTraj, and MDAnalysis.
"""

import time
import pytest
import torch
import numpy as np

from molfun.analysis.md import MolfunAnalysis, TrajAnalysis, AnalysisMD


# Paths to MD files
TOPOLOGY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.prmtop"
TRAJECTORY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.dcd"


def test_contact_map_performance():
    """Test contact map performance for MolfunAnalysis, MDTraj, and MDAnalysis."""
    
    print("\n" + "=" * 80)
    print("Contact Map Performance Test")
    print("=" * 80)
    
    # Parameters
    cutoff = 8.0  # Angstrom
    n_frames = 100
    
    # MolfunAnalysis
    print("\n--- MolfunAnalysis (Triton GPU) ---")
    start = time.time()
    molfun = MolfunAnalysis(TOPOLOGY_PATH, TRAJECTORY_PATH, selection="ca")
    load_time = time.time() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {molfun.n_atoms} (CA only), Frames: {molfun.n_frames}")
    
    # Warmup: compile Triton kernels and initialize GPU
    print("Warming up (compiling kernels)...")
    _ = molfun.contact_map(cutoff, frame=0)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Single frame test
    print(f"\nSingle frame contact map (cutoff={cutoff}Å):")
    start = time.time()
    cm_molfun_single = molfun.contact_map(cutoff, frame=10)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    molfun_time_single = time.time() - start
    print(f"  Time: {molfun_time_single*1000:.3f}ms")
    
    # Batch test
    print(f"\nBatch contact map ({n_frames} frames, cutoff={cutoff}Å):")
    start = time.time()
    cm_molfun_batch = molfun.contact_map(cutoff)[:n_frames]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    molfun_time_batch = time.time() - start
    print(f"  Time: {molfun_time_batch:.3f}s ({molfun_time_batch*1000/n_frames:.3f}ms/frame)")
    
    # TrajAnalysis (MDTraj)
    print("\n--- TrajAnalysis (MDTraj) ---")
    start = time.time()
    traj = TrajAnalysis(TOPOLOGY_PATH, TRAJECTORY_PATH, selection="ca")
    load_time = time.time() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {traj.n_atoms} (CA only), Frames: {traj.n_frames}")
    
    # Warmup
    print("Warming up...")
    _ = traj.contact_map(cutoff, frame=0)
    
    # Single frame test
    print(f"\nSingle frame contact map (cutoff={cutoff}Å):")
    start = time.time()
    cm_traj_single = traj.contact_map(cutoff, frame=10)
    traj_time_single = time.time() - start
    print(f"  Time: {traj_time_single*1000:.3f}ms")
    
    # Batch test
    print(f"\nBatch contact map ({n_frames} frames, cutoff={cutoff}Å):")
    start = time.time()
    cm_traj_batch = traj.contact_map(cutoff)[:n_frames]
    traj_time_batch = time.time() - start
    print(f"  Time: {traj_time_batch:.3f}s ({traj_time_batch*1000/n_frames:.3f}ms/frame)")
    
    # AnalysisMD (MDAnalysis)
    print("\n--- AnalysisMD (MDAnalysis) ---")
    start = time.time()
    mda_obj = AnalysisMD(TOPOLOGY_PATH, TRAJECTORY_PATH, selection="ca")
    load_time = time.time() - start
    print(f"Load time: {load_time:.3f}s")
    print(f"Atoms: {mda_obj.n_atoms} (CA only), Frames: {mda_obj.n_frames}")
    
    # Warmup
    print("Warming up...")
    _ = mda_obj.contact_map(cutoff, frame=0)
    
    # Single frame test
    print(f"\nSingle frame contact map (cutoff={cutoff}Å):")
    start = time.time()
    cm_mda_single = mda_obj.contact_map(cutoff, frame=10)
    mda_time_single = time.time() - start
    print(f"  Time: {mda_time_single*1000:.3f}ms")
    
    # Batch test
    print(f"\nBatch contact map ({n_frames} frames, cutoff={cutoff}Å):")
    start = time.time()
    cm_mda_batch = mda_obj.contact_map(cutoff)[:n_frames]
    mda_time_batch = time.time() - start
    print(f"  Time: {mda_time_batch:.3f}s ({mda_time_batch*1000/n_frames:.3f}ms/frame)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print("\nSingle frame:")
    print(f"  MolfunAnalysis: {molfun_time_single*1000:.3f}ms")
    print(f"  TrajAnalysis:   {traj_time_single*1000:.3f}ms | Speedup: {traj_time_single/molfun_time_single:.1f}x")
    print(f"  AnalysisMD:     {mda_time_single*1000:.3f}ms | Speedup: {mda_time_single/molfun_time_single:.1f}x")
    
    print(f"\nBatch ({n_frames} frames):")
    print(f"  MolfunAnalysis: {molfun_time_batch:.3f}s ({molfun_time_batch*1000/n_frames:.3f}ms/frame)")
    print(f"  TrajAnalysis:   {traj_time_batch:.3f}s ({traj_time_batch*1000/n_frames:.3f}ms/frame) | Speedup: {traj_time_batch/molfun_time_batch:.1f}x")
    print(f"  AnalysisMD:     {mda_time_batch:.3f}s ({mda_time_batch*1000/n_frames:.3f}ms/frame) | Speedup: {mda_time_batch/molfun_time_batch:.1f}x")
    
    # Verify correctness - count contacts
    from molfun.kernels.analysis.contact_map_atoms import unpack_contact_map
    
    # Unpack MolfunAnalysis bitpacked output
    cm_molfun_unpacked = unpack_contact_map(cm_molfun_single, molfun.n_atoms).cpu().numpy()
    
    # Count contacts (upper triangle)
    contacts_molfun = np.sum(cm_molfun_unpacked)
    contacts_traj = np.sum(cm_traj_single)
    contacts_mda = np.sum(cm_mda_single)
    
    print(f"\nCorrectness check (contact counts):")
    print(f"  MolfunAnalysis: {contacts_molfun} contacts")
    print(f"  TrajAnalysis:   {contacts_traj} contacts | Diff: {abs(contacts_molfun - contacts_traj)}")
    print(f"  AnalysisMD:     {contacts_mda} contacts | Diff: {abs(contacts_molfun - contacts_mda)}")
    
    # Assert that contact counts match (should be identical)
    assert contacts_molfun == contacts_traj, f"MolfunAnalysis vs MDTraj contact count mismatch: {contacts_molfun} vs {contacts_traj}"
    assert contacts_molfun == contacts_mda, f"MolfunAnalysis vs MDAnalysis contact count mismatch: {contacts_molfun} vs {contacts_mda}"
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_contact_map_performance()
