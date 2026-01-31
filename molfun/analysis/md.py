"""
molfun/analysis/md.py

Optimized MD trajectory analysis classes using Triton kernels, MDTraj, and MDAnalysis.
"""

import numpy as np
import torch
from typing import Optional, Union, List
from scipy.spatial.transform import Rotation

from molfun.kernels.analysis.contact_map_atoms import contact_map_atoms_bitpack, unpack_contact_map
from molfun.kernels.analysis.rmsd import rmsd_triton, rmsd_batch_triton


def kabsch_superposition(P: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Kabsch algorithm for optimal rotation between two point sets using scipy.
    
    Args:
        P: [N, 3] reference coordinates
        Q: [N, 3] or [M, N, 3] coordinates to align
    
    Returns:
        P_centered: [N, 3] centered reference
        Q_aligned: [N, 3] or [M, N, 3] aligned coordinates (centered and rotated)
    """
    device = P.device
    P_np = P.cpu().numpy()
    P_centered_np = P_np - P_np.mean(axis=0)
    
    if Q.dim() == 2:
        # Single frame
        Q_np = Q.cpu().numpy()
        Q_centered_np = Q_np - Q_np.mean(axis=0)
        
        # scipy.align_vectors(a, b) finds R such that a ≈ R @ b
        rot, _ = Rotation.align_vectors(P_centered_np, Q_centered_np)
        Q_aligned_np = rot.apply(Q_centered_np)
        
        P_centered = torch.from_numpy(P_centered_np).float().to(device)
        Q_aligned = torch.from_numpy(Q_aligned_np).float().to(device)
        return P_centered, Q_aligned
    else:
        # Batch: [M, N, 3]
        M = Q.shape[0]
        Q_np = Q.cpu().numpy()
        Q_aligned_np = np.zeros_like(Q_np)
        
        for i in range(M):
            Q_i = Q_np[i]
            Q_i_centered = Q_i - Q_i.mean(axis=0)
            rot, _ = Rotation.align_vectors(P_centered_np, Q_i_centered)
            Q_aligned_np[i] = rot.apply(Q_i_centered)
        
        P_centered = torch.from_numpy(P_centered_np).float().to(device)
        Q_aligned = torch.from_numpy(Q_aligned_np).float().to(device)
        return P_centered, Q_aligned

# Optional dependencies
try:
    import mdtraj
    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms
    HAS_MDANALYSIS = True
except ImportError:
    HAS_MDANALYSIS = False


class MolfunAnalysis:
    """MD analysis using Triton kernels (fastest)."""
    
    def __init__(self, topology: str, trajectory: str):
        """
        Args:
            topology: Path to topology file (PDB, GRO, etc.)
            trajectory: Path to trajectory file (XTC, DCD, TRR, etc.)
        """
        if not HAS_MDTRAJ:
            raise ImportError("MDTraj required for MolfunAnalysis")
        
        traj_full = mdtraj.load(trajectory, top=topology)
        # Filter only CA atoms
        ca_indices = traj_full.topology.select("name CA")
        self.traj = traj_full.atom_slice(ca_indices)
        self.n_atoms = self.traj.n_atoms
        self.n_frames = self.traj.n_frames
        
    def get_coords(self, frame: Optional[int] = None) -> torch.Tensor:
        """Get coordinates as CUDA tensor."""
        if frame is not None:
            coords = self.traj.xyz[frame]  # [N, 3] nm
        else:
            coords = self.traj.xyz  # [F, N, 3] nm
        
        # Convert nm to Angstrom and move to GPU
        coords_angstrom = coords * 10.0  # nm -> Angstrom
        return torch.from_numpy(coords_angstrom).float().cuda()
    
    def contact_map(self, cutoff: float, frame: Optional[int] = None) -> torch.Tensor:
        """Compute contact map using Triton kernel."""
        coords = self.get_coords(frame)
        if coords.dim() == 2:
            return contact_map_atoms_bitpack(coords, cutoff)
        else:
            # Batch: compute for each frame
            results = []
            for f in range(coords.shape[0]):
                results.append(contact_map_atoms_bitpack(coords[f], cutoff))
            return torch.stack(results)
    
    def rmsd(self, ref_frame: int = 0, frame: Optional[int] = None, superposition: bool = True) -> Union[float, torch.Tensor]:
        """
        Compute RMSD using Triton kernel with optional superposition (Kabsch alignment).
        
        Args:
            ref_frame: Reference frame index
            frame: Frame to compare (None for batch)
            superposition: If True, align coordinates before RMSD (like MDTraj/MDAnalysis)
        
        Returns:
            RMSD value(s) as GPU tensor
        """
        ref_coords = self.get_coords(ref_frame)
        
        if frame is not None:
            coords = self.get_coords(frame)
            if superposition:
                # Kabsch alignment returns (centered_ref, aligned_coords)
                ref_centered, coords_aligned = kabsch_superposition(ref_coords, coords)
                return rmsd_triton(ref_centered, coords_aligned)
            else:
                return rmsd_triton(ref_coords, coords)
        else:
            # Batch: all frames vs reference using optimized batch kernel
            coords_batch = self.get_coords()  # [F, N, 3]
            
            if superposition:
                # Kabsch alignment for batch
                ref_centered, coords_batch_aligned = kabsch_superposition(ref_coords, coords_batch)
                return rmsd_batch_triton(ref_centered, coords_batch_aligned)
            else:
                return rmsd_batch_triton(ref_coords, coords_batch)


class TrajAnalysis:
    """MD analysis using MDTraj."""
    
    def __init__(self, topology: str, trajectory: str):
        if not HAS_MDTRAJ:
            raise ImportError("MDTraj required for TrajAnalysis")
        
        traj_full = mdtraj.load(trajectory, top=topology)
        # Filter only CA atoms
        ca_indices = traj_full.topology.select("name CA")
        self.traj = traj_full.atom_slice(ca_indices)
        self.n_atoms = self.traj.n_atoms
        self.n_frames = self.traj.n_frames
    
    def contact_map(self, cutoff: float, frame: Optional[int] = None) -> np.ndarray:
        """Compute contact map using MDTraj."""
        if frame is not None:
            # Use pairwise distances
            coords = self.traj.xyz[frame]  # [N, 3] nm
            diff = coords[:, None, :] - coords[None, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2)) * 10.0  # nm -> Angstrom
            cm = dists < cutoff
            np.fill_diagonal(cm, False)
            return cm
        else:
            # Batch computation
            results = []
            for f in range(self.n_frames):
                results.append(self.contact_map(cutoff, f))
            return np.stack(results)
    
    def rmsd(self, ref_frame: int = 0, frame: Optional[int] = None) -> Union[float, np.ndarray]:
        """Compute RMSD using MDTraj."""
        if frame is not None:
            return mdtraj.rmsd(self.traj[frame], self.traj[ref_frame])[0] * 10.0  # nm -> Angstrom
        else:
            return mdtraj.rmsd(self.traj, self.traj[ref_frame]) * 10.0  # nm -> Angstrom


class AnalysisMD:
    """MD analysis using MDAnalysis."""
    
    def __init__(self, topology: str, trajectory: str):
        if not HAS_MDANALYSIS:
            raise ImportError("MDAnalysis required for AnalysisMD")
        
        self.universe = mda.Universe(topology, trajectory)
        # Select only CA atoms
        self.ca_atoms = self.universe.select_atoms("name CA")
        self.n_atoms = len(self.ca_atoms)
        self.n_frames = len(self.universe.trajectory)
    
    def get_coords(self, frame: Optional[int] = None) -> np.ndarray:
        """Get coordinates as numpy array."""
        if frame is not None:
            self.universe.trajectory[frame]
            return self.ca_atoms.positions  # [N, 3] Angstrom
        else:
            coords = []
            for ts in self.universe.trajectory:
                coords.append(self.ca_atoms.positions)
            return np.array(coords)  # [F, N, 3] Angstrom
    
    def contact_map(self, cutoff: float, frame: Optional[int] = None) -> np.ndarray:
        """Compute contact map using MDAnalysis."""
        coords = self.get_coords(frame)
        
        if coords.ndim == 2:
            # Single frame
            diff = coords[:, None, :] - coords[None, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            cm = dists < cutoff
            np.fill_diagonal(cm, False)
            return cm
        else:
            # Batch
            results = []
            for f in range(coords.shape[0]):
                results.append(self.contact_map(cutoff, f))
            return np.stack(results)
    
    def rmsd(self, ref_frame: int = 0, frame: Optional[int] = None, superposition: bool = True) -> Union[float, np.ndarray]:
        """Compute RMSD using MDAnalysis."""
        ref_coords = self.get_coords(ref_frame)
        
        if frame is not None:
            coords = self.get_coords(frame)
            return rms.rmsd(coords, ref_coords, superposition=superposition)
        else:
            # Batch
            coords_batch = self.get_coords()
            rmsds = []
            for f in range(coords_batch.shape[0]):
                rmsds.append(rms.rmsd(coords_batch[f], ref_coords, superposition=superposition))
            return np.array(rmsds)
