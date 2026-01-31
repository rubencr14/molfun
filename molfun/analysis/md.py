"""
molfun/analysis/md.py

Optimized MD trajectory analysis classes using Triton kernels, MDTraj, and MDAnalysis.
"""

import numpy as np
import torch
from typing import Optional, Union, List

from molfun.kernels.analysis.contact_map_atoms import contact_map_atoms_bitpack, unpack_contact_map
from molfun.kernels.analysis.rmsd import rmsd_triton, rmsd_batch_triton


def kabsch_gpu(P: torch.Tensor, Q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Kabsch algorithm for optimal rotation - fully on GPU using PyTorch batched SVD.
    
    Finds rotation R that minimizes ||P - Q @ R||
    
    Args:
        P: [N, 3] reference coordinates (GPU)
        Q: [N, 3] or [M, N, 3] coordinates to align (GPU)
    
    Returns:
        P_centered: [N, 3] centered reference
        Q_aligned: [N, 3] or [M, N, 3] aligned coordinates (centered and rotated)
    """
    # Center reference
    P_centered = P - P.mean(dim=0, keepdim=True)  # [N, 3]
    
    if Q.dim() == 2:
        # Single frame: [N, 3]
        Q_centered = Q - Q.mean(dim=0, keepdim=True)  # [N, 3]
        
        # Covariance matrix H = P^T @ Q -> [3, 3]
        H = P_centered.T @ Q_centered
        
        # SVD: H = U @ S @ Vt
        U, S, Vt = torch.linalg.svd(H)
        V = Vt.T
        
        # Rotation R = V @ U^T (standard Kabsch)
        R = V @ U.T
        
        # Handle reflection (ensure det(R) = 1)
        if torch.det(R) < 0:
            V[:, -1] *= -1
            R = V @ U.T
        
        # Apply rotation: Q_aligned = Q_centered @ R
        Q_aligned = Q_centered @ R
        
        return P_centered, Q_aligned
    
    else:
        # Batch: [M, N, 3]
        M = Q.shape[0]
        
        # Center each frame: [M, N, 3]
        Q_centered = Q - Q.mean(dim=1, keepdim=True)
        
        # Covariance matrices H[m] = P^T @ Q[m] -> [M, 3, 3]
        # P_centered: [N, 3], Q_centered: [M, N, 3]
        # H[m,i,j] = sum_n P_centered[n,i] * Q_centered[m,n,j]
        H = torch.einsum('ni,mnj->mij', P_centered, Q_centered)
        
        # Batched SVD: U[M,3,3], S[M,3], Vt[M,3,3]
        U, S, Vt = torch.linalg.svd(H)
        V = Vt.transpose(-2, -1)  # [M, 3, 3]
        
        # Rotation R = V @ U^T -> [M, 3, 3]
        R = torch.bmm(V, U.transpose(-2, -1))
        
        # Handle reflections (det < 0)
        det = torch.det(R)  # [M]
        mask = det < 0
        if mask.any():
            # Fix reflections by flipping last column of V
            V_fixed = V.clone()
            V_fixed[mask, :, -1] *= -1
            R[mask] = torch.bmm(V_fixed[mask], U[mask].transpose(-2, -1))
        
        # Apply rotation: Q_aligned[m] = Q_centered[m] @ R[m]
        # [M, N, 3] @ [M, 3, 3] -> [M, N, 3]
        Q_aligned = torch.bmm(Q_centered, R)
        
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
    
    # Selection presets
    SELECTIONS = {
        "ca": "name CA",
        "backbone": "backbone",
        "protein": "protein",
    }
    
    def __init__(self, topology: str, trajectory: str, selection: str = "ca"):
        """
        Args:
            topology: Path to topology file (PDB, GRO, etc.)
            trajectory: Path to trajectory file (XTC, DCD, TRR, etc.)
            selection: Atom selection - "ca" (alpha carbons), "backbone", or "protein" (all protein atoms)
        """
        if not HAS_MDTRAJ:
            raise ImportError("MDTraj required for MolfunAnalysis")
        
        traj_full = mdtraj.load(trajectory, top=topology)
        
        # Get selection string
        if selection in self.SELECTIONS:
            sel_str = self.SELECTIONS[selection]
        else:
            sel_str = selection  # Allow custom MDTraj selection strings
        
        # Filter atoms
        indices = traj_full.topology.select(sel_str)
        self.traj = traj_full.atom_slice(indices)
        self.n_atoms = self.traj.n_atoms
        self.n_frames = self.traj.n_frames
        self.selection = selection
        
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
                # Kabsch alignment on GPU
                ref_centered, coords_aligned = kabsch_gpu(ref_coords, coords)
                return rmsd_triton(ref_centered, coords_aligned)
            else:
                return rmsd_triton(ref_coords, coords)
        else:
            # Batch: all frames vs reference using optimized batch kernel
            coords_batch = self.get_coords()  # [F, N, 3]
            
            if superposition:
                # Kabsch alignment on GPU (batched SVD)
                ref_centered, coords_batch_aligned = kabsch_gpu(ref_coords, coords_batch)
                return rmsd_batch_triton(ref_centered, coords_batch_aligned)
            else:
                return rmsd_batch_triton(ref_coords, coords_batch)


class TrajAnalysis:
    """MD analysis using MDTraj."""
    
    # Selection presets
    SELECTIONS = {
        "ca": "name CA",
        "backbone": "backbone",
        "protein": "protein",
    }
    
    def __init__(self, topology: str, trajectory: str, selection: str = "ca"):
        if not HAS_MDTRAJ:
            raise ImportError("MDTraj required for TrajAnalysis")
        
        traj_full = mdtraj.load(trajectory, top=topology)
        
        # Get selection string
        if selection in self.SELECTIONS:
            sel_str = self.SELECTIONS[selection]
        else:
            sel_str = selection
        
        indices = traj_full.topology.select(sel_str)
        self.traj = traj_full.atom_slice(indices)
        self.n_atoms = self.traj.n_atoms
        self.n_frames = self.traj.n_frames
        self.selection = selection
    
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
    
    # Selection presets (MDAnalysis syntax)
    SELECTIONS = {
        "ca": "name CA",
        "backbone": "backbone",
        "protein": "protein",
    }
    
    def __init__(self, topology: str, trajectory: str, selection: str = "ca"):
        if not HAS_MDANALYSIS:
            raise ImportError("MDAnalysis required for AnalysisMD")
        
        self.universe = mda.Universe(topology, trajectory)
        
        # Get selection string
        if selection in self.SELECTIONS:
            sel_str = self.SELECTIONS[selection]
        else:
            sel_str = selection
        
        self.atoms = self.universe.select_atoms(sel_str)
        self.n_atoms = len(self.atoms)
        self.n_frames = len(self.universe.trajectory)
        self.selection = selection
    
    def get_coords(self, frame: Optional[int] = None) -> np.ndarray:
        """Get coordinates as numpy array."""
        if frame is not None:
            self.universe.trajectory[frame]
            return self.atoms.positions.copy()  # [N, 3] Angstrom
        else:
            coords = []
            for ts in self.universe.trajectory:
                coords.append(self.atoms.positions.copy())
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
