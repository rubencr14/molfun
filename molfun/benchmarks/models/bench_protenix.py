import torch
import torch.nn as nn
import triton.testing as tt
import importlib
import gc
import time
# Import your custom Triton kernel
from molfun.kernels.models.triangle_update import FastTriangleUpdate

TARGET_MODULE = "protenix.model.modules.triangle_multiplication"

def get_protenix_module():
    try:
        return importlib.import_module(TARGET_MODULE)
    except ImportError:
        # Minimal mock for testing
        class Mock:
            class TriangleMultiplicationOutgoing(nn.Module):
                def __init__(self, c_z):
                    super().__init__()
                    self.layer_norm = nn.LayerNorm(c_z)
                def forward(self, z, mask=None):
                    z = self.layer_norm(z)
                    return torch.sigmoid(z) * torch.einsum("bikc, bjkc -> bijc", z, z)
        return Mock()

pm_tri = get_protenix_module()

class BenchmarkProtenix:
    @staticmethod
    def run(n_res=512, channels=128):
        device = "cuda"
        dtype = torch.bfloat16
        z = torch.randn(1, n_res, n_res, channels, device=device, dtype=dtype)
        
        # --- NATIVE REFERENCE (Einsum like Protenix) ---
        torch.cuda.reset_peak_memory_stats()
        # Native Outgoing: sum over k of Z[i,k] * Z[j,k]
        # This is where PyTorch usually creates huge temporary tensors
        t0 = time.time()
        # We use a lambda to avoid measuring the setup time
        native_op = lambda: torch.sigmoid(torch.einsum('bikc,bjkc->bijc', z, z)) * torch.einsum('bikc,bjkc->bijc', z, z)
        
        native_ms = tt.do_bench(native_op)
        native_mem = torch.cuda.max_memory_allocated() / 1e9
        with torch.no_grad():
            native_out = native_op()

        # --- MOLFUN (TRITON) ---
        torch.cuda.reset_peak_memory_stats()
        # Ensure apply_unsloth_patch() was called before this
        opt_ms = tt.do_bench(lambda: FastTriangleUpdate.apply(z, 0))
        opt_mem = torch.cuda.max_memory_allocated() / 1e9
        with torch.no_grad():
            opt_out = FastTriangleUpdate.apply(z, 0)

        # Accuracy Check
        diff = torch.max(torch.abs(native_out - opt_out)).item()

        print(f"\n[N={n_res} | C={channels}]")
        print(f"Latency: {native_ms:.2f}ms (Native) vs {opt_ms:.2f}ms (Triton) -> {native_ms/opt_ms:.2f}x")
        print(f"Memory:  {native_mem:.2f}GB (Native) vs {opt_mem:.2f}GB (Triton) -> {native_mem - opt_mem:.2f}GB Saved")
        print(f"Max Diff: {diff:.2e}")

if __name__ == "__main__":
    for n in [512, 1024, 2048, 4000]:
        BenchmarkProtenix.run(n_res=n)