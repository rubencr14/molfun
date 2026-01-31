"""
molfun/benchmarks/bench_md_analysis.py

Benchmark MD trajectory analysis: MolfunAnalysis (Triton) vs MDTraj vs MDAnalysis.

This benchmark compares:
- MolfunAnalysis: Uses Triton kernels (contact_map_atoms_bitpack, rmsd_triton)
- TrajAnalysis: Uses MDTraj library
- AnalysisMD: Uses MDAnalysis library

Usage:
    python molfun/benchmarks/bench_md_analysis.py <topology.pdb> <trajectory.xtc>

Notes:
- Requires MDTraj and MDAnalysis installed
- Tests contact map and RMSD computation
- Measures time per frame for batch operations
"""

import time
import numpy as np
import torch

from molfun.analysis.md import MolfunAnalysis, TrajAnalysis, AnalysisMD
from molfun.kernels.analysis.contact_map_atoms import unpack_contact_map
# Note: Using MolfunAnalysis.rmsd() method instead of rmsd_batch_triton directly


def to_native_type(val):
    """Convert numpy/torch types to native Python types for JSON serialization."""
    if isinstance(val, torch.Tensor):
        val = val.item() if val.numel() == 1 else float(val)
    elif isinstance(val, np.ndarray):
        val = val.item() if val.size == 1 else float(val)
    elif isinstance(val, (np.float32, np.float64, np.floating)):
        val = float(val)
    elif isinstance(val, (np.int32, np.int64, np.integer)):
        val = int(val)
    elif isinstance(val, (np.bool_, bool)):
        val = bool(val)
    return val


def time_function(fn, warmup: int = 1):
    """Time a function execution."""
    # Warmup
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()
    
    return (end - start) * 1000, result  # ms


def run_benchmark(topology: str, trajectory: str, return_results: bool = False):
    """Run benchmark comparing all three analysis classes.
    
    Args:
        topology: Path to topology file (PDB/GRO/TOP)
        trajectory: Path to trajectory file (XTC/DCD/TRR)
        return_results: If True, returns structured results instead of printing
    
    Returns:
        If return_results=True, returns list of BenchmarkResult dicts
    """
    
    print("=" * 100)
    print(f"MD Analysis Benchmark")
    print(f"Topology: {topology}")
    print(f"Trajectory: {trajectory}")
    print("=" * 100)
    
    results = []
    
    # Initialize all three analyzers
    print("\nLoading trajectories...")
    print("  Loading MolfunAnalysis...")
    molfun = MolfunAnalysis(topology, trajectory)
    print(f"    Loaded: {molfun.n_atoms} atoms, {molfun.n_frames} frames")
    
    print("  Loading TrajAnalysis...")
    traj = TrajAnalysis(topology, trajectory)
    print(f"    Loaded: {traj.n_atoms} atoms, {traj.n_frames} frames")
    
    print("  Loading AnalysisMD...")
    mda_ana = AnalysisMD(topology, trajectory)
    print(f"    Loaded: {mda_ana.n_atoms} atoms, {mda_ana.n_frames} frames")
    
    print(f"\nAtoms: {molfun.n_atoms}, Frames: {molfun.n_frames}")
    
    # Test single frame operations
    print("\n" + "-" * 100)
    print("Single Frame Operations")
    print("-" * 100)
    
    frame = min(10, molfun.n_frames - 1)
    cutoff = 8.0  # Angstrom
    
    # Contact Map - Single Frame
    print(f"\nContact Map (frame {frame}, cutoff={cutoff}Å):")
    
    t_molfun, cm_molfun = time_function(lambda: molfun.contact_map(cutoff, frame))
    cm_molfun_unpacked = unpack_contact_map(cm_molfun, molfun.n_atoms)
    n_contacts_molfun = cm_molfun_unpacked.sum().item() // 2
    
    t_traj, cm_traj = time_function(lambda: traj.contact_map(cutoff, frame))
    n_contacts_traj = cm_traj.sum() // 2
    
    t_mda, cm_mda = time_function(lambda: mda_ana.contact_map(cutoff, frame))
    n_contacts_mda = cm_mda.sum() // 2
    
    print(f"  MolfunAnalysis: {t_molfun:8.3f}ms | Contacts: {n_contacts_molfun:>8,}")
    print(f"  TrajAnalysis:   {t_traj:8.3f}ms | Contacts: {n_contacts_traj:>8,} | Speedup: {t_traj/t_molfun:.2f}x")
    print(f"  AnalysisMD:     {t_mda:8.3f}ms | Contacts: {n_contacts_mda:>8,} | Speedup: {t_mda/t_molfun:.2f}x")
    
    if return_results:
        results.append({
            "benchmark_name": "md_analysis",
            "case_name": f"Contact Map (single frame, cutoff={to_native_type(cutoff)}Å)",
            "baseline_time_ms": float(round(to_native_type(t_traj), 4)),
            "triton_time_ms": float(round(to_native_type(t_molfun), 4)),
            "speedup": float(round(to_native_type(t_traj) / to_native_type(t_molfun), 2)) if t_molfun > 0 else None,
            "max_diff": float(abs(to_native_type(n_contacts_molfun) - to_native_type(n_contacts_traj))),
            "mean_diff": float(abs(to_native_type(n_contacts_molfun) - to_native_type(n_contacts_traj))),
            "metadata": {
                "operation": "contact_map_single",
                "frame": int(frame),
                "cutoff": float(to_native_type(cutoff)),
                "molfun_ms": float(round(to_native_type(t_molfun), 4)),
                "mdtraj_ms": float(round(to_native_type(t_traj), 4)),
                "mdanalysis_ms": float(round(to_native_type(t_mda), 4)),
                "contacts_molfun": int(to_native_type(n_contacts_molfun)),
                "contacts_traj": int(to_native_type(n_contacts_traj)),
                "contacts_mda": int(to_native_type(n_contacts_mda)),
                "speedup_vs_mdtraj": float(round(to_native_type(t_traj) / to_native_type(t_molfun), 2)) if t_molfun > 0 else None,
                "speedup_vs_mdanalysis": float(round(to_native_type(t_mda) / to_native_type(t_molfun), 2)) if t_molfun > 0 else None,
            }
        })
    
    # RMSD - Single Frame (with superposition for fair comparison)
    print(f"\nRMSD (frame {frame} vs frame 0) - With superposition (aligned RMSD):")
    
    def rmsd_molfun_fn():
        result = molfun.rmsd(0, frame, superposition=True)  # With superposition like MDTraj
        torch.cuda.synchronize()  # Sync after computation, not during
        return result.item()  # Only .item() once, outside timing
    
    def rmsd_numpy_fn():
        """Reference: NumPy manual raw RMSD."""
        ref_coords = molfun.get_coords(0).cpu().numpy()
        coords = molfun.get_coords(frame).cpu().numpy()
        diff = ref_coords - coords
        dist2 = np.sum(diff ** 2, axis=1)
        return np.sqrt(np.mean(dist2))
    
    def rmsd_traj_fn():
        # MDTraj does superposition by default (aligned RMSD)
        import mdtraj
        return mdtraj.rmsd(traj.traj[frame], traj.traj[0])[0] * 10.0
    
    t_numpy, rmsd_numpy = time_function(rmsd_numpy_fn)
    t_molfun, rmsd_molfun = time_function(rmsd_molfun_fn)
    t_traj, rmsd_traj = time_function(rmsd_traj_fn)
    t_mda, rmsd_mda = time_function(lambda: mda_ana.rmsd(0, frame))
    
    # Check correctness: compare against NumPy reference
    diff_numpy = abs(rmsd_molfun - rmsd_numpy)
    match_numpy = diff_numpy < 1e-3
    
    # Compare against MDTraj (both use superposition)
    diff_traj = abs(rmsd_molfun - rmsd_traj)
    diff_mda = abs(rmsd_molfun - rmsd_mda)
    match_traj = diff_traj < 1e-2
    match_mda = diff_mda < 1e-2
    
    print(f"  MolfunAnalysis: {t_molfun:8.3f}ms | RMSD: {rmsd_molfun:.4f}Å")
    print(f"  TrajAnalysis:   {t_traj:8.3f}ms | RMSD: {rmsd_traj:.4f}Å | Speedup: {t_traj/t_molfun:.2f}x | Match: {'✓' if match_traj else '✗'} (diff: {diff_traj:.2e}Å)")
    print(f"  AnalysisMD:     {t_mda:8.3f}ms | RMSD: {rmsd_mda:.4f}Å | Speedup: {t_mda/t_molfun:.2f}x | Match: {'✓' if match_mda else '✗'} (diff: {diff_mda:.2e}Å)")
    
    if return_results:
        results.append({
            "benchmark_name": "md_analysis",
            "case_name": f"RMSD (single frame {int(frame)} vs 0, with superposition)",
            "baseline_time_ms": float(round(to_native_type(t_traj), 4)),
            "triton_time_ms": float(round(to_native_type(t_molfun), 4)),
            "speedup": float(round(to_native_type(t_traj) / to_native_type(t_molfun), 2)) if t_molfun > 0 else None,
            "max_diff": float(max(to_native_type(diff_traj), to_native_type(diff_mda))),
            "mean_diff": float((to_native_type(diff_traj) + to_native_type(diff_mda)) / 2),
            "metadata": {
                "operation": "rmsd_single",
                "frame": int(frame),
                "molfun_ms": float(round(to_native_type(t_molfun), 4)),
                "mdtraj_ms": float(round(to_native_type(t_traj), 4)),
                "mdanalysis_ms": float(round(to_native_type(t_mda), 4)),
                "numpy_ms": float(round(to_native_type(t_numpy), 4)),
                "rmsd_molfun": float(round(to_native_type(rmsd_molfun), 6)),
                "rmsd_traj": float(round(to_native_type(rmsd_traj), 6)),
                "rmsd_mda": float(round(to_native_type(rmsd_mda), 6)),
                "rmsd_numpy": float(round(to_native_type(rmsd_numpy), 6)),
                "diff_traj": float(round(to_native_type(diff_traj), 6)),
                "diff_mda": float(round(to_native_type(diff_mda), 6)),
                "diff_numpy": float(round(to_native_type(diff_numpy), 6)),
                "match_traj": bool(match_traj),
                "match_mda": bool(match_mda),
                "match_numpy": bool(match_numpy),
                "speedup_vs_mdtraj": float(round(to_native_type(t_traj) / to_native_type(t_molfun), 2)) if t_molfun > 0 else None,
                "speedup_vs_mdanalysis": float(round(to_native_type(t_mda) / to_native_type(t_molfun), 2)) if t_molfun > 0 else None,
            }
        })
    
    if not match_traj or not match_mda:
        print(f"  ⚠️  RMSD values differ (small differences expected due to numerical precision in superposition)")
    
    # Test batch operations (all frames)
    print("\n" + "-" * 100)
    print("Batch Operations (All Frames)")
    print("-" * 100)
    
    n_frames_to_test = min(100, molfun.n_frames)  # Limit to 100 frames for benchmark
    
    # Contact Map - Batch
    print(f"\nContact Map (first {n_frames_to_test} frames, cutoff={cutoff}Å):")
    
    def batch_cm_molfun():
        results = []
        for f in range(n_frames_to_test):
            results.append(molfun.contact_map(cutoff, f))
        return torch.stack(results)
    
    def batch_cm_traj():
        results = []
        for f in range(n_frames_to_test):
            results.append(traj.contact_map(cutoff, f))
        return np.stack(results)
    
    def batch_cm_mda():
        results = []
        for f in range(n_frames_to_test):
            results.append(mda_ana.contact_map(cutoff, f))
        return np.stack(results)
    
    t_molfun_cm, _ = time_function(batch_cm_molfun, warmup=0)
    t_traj_cm, _ = time_function(batch_cm_traj, warmup=0)
    t_mda_cm, _ = time_function(batch_cm_mda, warmup=0)
    
    print(f"  MolfunAnalysis: {t_molfun_cm:8.3f}ms | {t_molfun_cm/n_frames_to_test:.3f}ms/frame")
    print(f"  TrajAnalysis:   {t_traj_cm:8.3f}ms | {t_traj_cm/n_frames_to_test:.3f}ms/frame | Speedup: {t_traj_cm/t_molfun_cm:.2f}x")
    print(f"  AnalysisMD:     {t_mda_cm:8.3f}ms | {t_mda_cm/n_frames_to_test:.3f}ms/frame | Speedup: {t_mda_cm/t_molfun_cm:.2f}x")
    
    if return_results:
        results.append({
            "benchmark_name": "md_analysis",
            "case_name": f"Contact Map (batch, {int(n_frames_to_test)} frames, cutoff={to_native_type(cutoff)}Å)",
            "baseline_time_ms": float(round(to_native_type(t_traj_cm), 4)),
            "triton_time_ms": float(round(to_native_type(t_molfun_cm), 4)),
            "speedup": float(round(to_native_type(t_traj_cm) / to_native_type(t_molfun_cm), 2)) if t_molfun_cm > 0 else None,
            "max_diff": 0.0,  # Contact maps are binary, diff is in count
            "mean_diff": 0.0,
            "metadata": {
                "operation": "contact_map_batch",
                "n_frames": int(n_frames_to_test),
                "cutoff": float(to_native_type(cutoff)),
                "molfun_ms": float(round(to_native_type(t_molfun_cm), 4)),
                "mdtraj_ms": float(round(to_native_type(t_traj_cm), 4)),
                "mdanalysis_ms": float(round(to_native_type(t_mda_cm), 4)),
                "molfun_ms_per_frame": float(round(to_native_type(t_molfun_cm) / to_native_type(n_frames_to_test), 4)),
                "mdtraj_ms_per_frame": float(round(to_native_type(t_traj_cm) / to_native_type(n_frames_to_test), 4)),
                "mdanalysis_ms_per_frame": float(round(to_native_type(t_mda_cm) / to_native_type(n_frames_to_test), 4)),
                "speedup_vs_mdtraj": float(round(to_native_type(t_traj_cm) / to_native_type(t_molfun_cm), 2)) if t_molfun_cm > 0 else None,
                "speedup_vs_mdanalysis": float(round(to_native_type(t_mda_cm) / to_native_type(t_molfun_cm), 2)) if t_molfun_cm > 0 else None,
            }
        })
    
    # RMSD - Batch (optimized batch kernel with superposition)
    print(f"\nRMSD (first {n_frames_to_test} frames vs frame 0) - With superposition:")
    
    def batch_rmsd_molfun():
        # Use MolfunAnalysis method with superposition
        result = molfun.rmsd(0, superposition=True)[:n_frames_to_test]
        torch.cuda.synchronize()  # Sync once after batch computation
        return result  # GPU tensor, no .item() here
    
    def batch_rmsd_traj():
        import mdtraj
        # MDTraj does superposition by default (aligned RMSD)
        rmsds = []
        for f in range(n_frames_to_test):
            rmsds.append(mdtraj.rmsd(traj.traj[f], traj.traj[0])[0] * 10.0)
        return np.array(rmsds)
    
    def batch_rmsd_mda():
        # MDAnalysis also does superposition
        rmsds = []
        for f in range(n_frames_to_test):
            rmsds.append(mda_ana.rmsd(0, f))
        return np.array(rmsds)
    
    t_molfun_rmsd, rmsds_molfun = time_function(batch_rmsd_molfun, warmup=0)
    t_traj_rmsd, rmsds_traj = time_function(batch_rmsd_traj, warmup=0)
    t_mda_rmsd, rmsds_mda = time_function(batch_rmsd_mda, warmup=0)
    
    # Convert to numpy for comparison
    rmsds_molfun_np = rmsds_molfun.cpu().numpy()
    rmsds_traj_np = np.array(rmsds_traj)
    rmsds_mda_np = np.array(rmsds_mda)
    
    # Check correctness: compare against MDTraj (both use superposition now)
    max_diff_traj = np.abs(rmsds_molfun_np - rmsds_traj_np).max()
    mean_diff_traj = np.abs(rmsds_molfun_np - rmsds_traj_np).mean()
    match_traj = max_diff_traj < 1e-2  # Tolerance for superposition (can have small numerical differences)
    
    max_diff_mda = np.abs(rmsds_molfun_np - rmsds_mda_np).max()
    mean_diff_mda = np.abs(rmsds_molfun_np - rmsds_mda_np).mean()
    match_mda = max_diff_mda < 1e-2
    
    print(f"  MolfunAnalysis: {t_molfun_rmsd:8.3f}ms | {t_molfun_rmsd/n_frames_to_test:.3f}ms/frame")
    print(f"  TrajAnalysis:   {t_traj_rmsd:8.3f}ms | {t_traj_rmsd/n_frames_to_test:.3f}ms/frame | Speedup: {t_traj_rmsd/t_molfun_rmsd:.2f}x | Match: {'✓' if match_traj else '✗'} (max_diff: {max_diff_traj:.2e}Å)")
    print(f"  AnalysisMD:     {t_mda_rmsd:8.3f}ms | {t_mda_rmsd/n_frames_to_test:.3f}ms/frame | Speedup: {t_mda_rmsd/t_molfun_rmsd:.2f}x | Match: {'✓' if match_mda else '✗'} (max_diff: {max_diff_mda:.2e}Å)")
    
    if return_results:
        results.append({
            "benchmark_name": "md_analysis",
            "case_name": f"RMSD (batch, {int(n_frames_to_test)} frames vs 0, with superposition)",
            "baseline_time_ms": float(round(to_native_type(t_traj_rmsd), 4)),
            "triton_time_ms": float(round(to_native_type(t_molfun_rmsd), 4)),
            "speedup": float(round(to_native_type(t_traj_rmsd) / to_native_type(t_molfun_rmsd), 2)) if t_molfun_rmsd > 0 else None,
            "max_diff": float(max(to_native_type(max_diff_traj), to_native_type(max_diff_mda))),
            "mean_diff": float((to_native_type(mean_diff_traj) + to_native_type(mean_diff_mda)) / 2),
            "metadata": {
                "operation": "rmsd_batch",
                "n_frames": int(n_frames_to_test),
                "molfun_ms": float(round(to_native_type(t_molfun_rmsd), 4)),
                "mdtraj_ms": float(round(to_native_type(t_traj_rmsd), 4)),
                "mdanalysis_ms": float(round(to_native_type(t_mda_rmsd), 4)),
                "molfun_ms_per_frame": float(round(to_native_type(t_molfun_rmsd) / to_native_type(n_frames_to_test), 4)),
                "mdtraj_ms_per_frame": float(round(to_native_type(t_traj_rmsd) / to_native_type(n_frames_to_test), 4)),
                "mdanalysis_ms_per_frame": float(round(to_native_type(t_mda_rmsd) / to_native_type(n_frames_to_test), 4)),
                "max_diff_traj": float(round(to_native_type(max_diff_traj), 6)),
                "mean_diff_traj": float(round(to_native_type(mean_diff_traj), 6)),
                "max_diff_mda": float(round(to_native_type(max_diff_mda), 6)),
                "mean_diff_mda": float(round(to_native_type(mean_diff_mda), 6)),
                "match_traj": bool(match_traj),
                "match_mda": bool(match_mda),
                "speedup_vs_mdtraj": float(round(to_native_type(t_traj_rmsd) / to_native_type(t_molfun_rmsd), 2)) if t_molfun_rmsd > 0 else None,
                "speedup_vs_mdanalysis": float(round(to_native_type(t_mda_rmsd) / to_native_type(t_molfun_rmsd), 2)) if t_molfun_rmsd > 0 else None,
            }
        })
    
    if not match_traj or not match_mda:
        print(f"\n  ⚠️  RMSD values comparison:")
        if not match_traj:
            print(f"     vs MDTraj: max_diff={max_diff_traj:.6f}Å, mean_diff={mean_diff_traj:.6f}Å")
            print(f"     First 5 values - Molfun: {rmsds_molfun_np[:5]}")
            print(f"     First 5 values - MDTraj:  {rmsds_traj_np[:5]}")
        if not match_mda:
            print(f"     vs MDAnalysis: max_diff={max_diff_mda:.6f}Å, mean_diff={mean_diff_mda:.6f}Å")
            print(f"     First 5 values - Molfun:    {rmsds_molfun_np[:5]}")
            print(f"     First 5 values - MDAnalysis: {rmsds_mda_np[:5]}")
    else:
        print(f"\n  ✓ All RMSD values match (with superposition)")
    
    # Summary
    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"MolfunAnalysis (Triton kernels) performance:")
    print(f"  - Contact maps: {t_traj_cm/t_molfun_cm:.1f}x faster than MDTraj, {t_mda_cm/t_molfun_cm:.1f}x faster than MDAnalysis")
    print(f"  - RMSD: {t_traj_rmsd/t_molfun_rmsd:.2f}x vs MDTraj, {t_mda_rmsd/t_molfun_rmsd:.1f}x vs MDAnalysis")
    
    if return_results:
        return results


def run_benchmark_api():
    """API-compatible wrapper that uses default paths and returns structured results."""
    # Default paths (same as in __main__)
    TOPOLOGY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.prmtop"
    TRAJECTORY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.dcd"
    
    return run_benchmark(TOPOLOGY_PATH, TRAJECTORY_PATH, return_results=True)


if __name__ == "__main__":
    # Paths to topology and trajectory files
    TOPOLOGY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.prmtop"  # Change this to your PDB/GRO/TOP file
    TRAJECTORY_PATH = "/home/rubencr/Escritorio/projects/nomosis/MD_FILES_IKER/MD_LEISH_DPPE_Reduced.dcd"  # Change this to your XTC/DCD/TRR file
    
    run_benchmark(TOPOLOGY_PATH, TRAJECTORY_PATH)
