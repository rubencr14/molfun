import os
import sys
import time
import importlib
import pkgutil
import inspect
from typing import Optional, Tuple

import torch
import torch.nn as nn

# IMPORTANT:
# Si uses "pip install protenix", normalment NO cal afegir cap path manual.
# Només posa-ho si tens un checkout local de protenix que vols prioritzar.
PROTENIX_PATH = "/home/rubencr/Escritorio/projects/nomosis"
if os.path.isdir(PROTENIX_PATH) and PROTENIX_PATH not in sys.path:
    sys.path.insert(0, PROTENIX_PATH)

from molfun.kernels.models.triangle_update import FastTriangleUpdate


def find_triangle_classes() -> Tuple[Optional[type], Optional[type], Optional[str]]:
    """
    Busca dins del paquet protenix les classes TriangleMultiplicationOutgoing/Incoming.
    Retorna: (OutgoingClass, IncomingClass, module_name)
    """
    import protenix  # ha d'estar instal·lat o disponible al path

    out_cls, in_cls, found_module = None, None, None

    for mod in pkgutil.walk_packages(protenix.__path__, prefix=protenix.__name__ + "."):
        modname = mod.name

        # Evita mòduls pesats que poden compilar extensions al import
        if ".layer_norm" in modname:
            continue

        try:
            m = importlib.import_module(modname)
        except Exception:
            continue

        if hasattr(m, "TriangleMultiplicationOutgoing"):
            c = getattr(m, "TriangleMultiplicationOutgoing")
            if inspect.isclass(c):
                out_cls = c
                found_module = modname

        if hasattr(m, "TriangleMultiplicationIncoming"):
            c = getattr(m, "TriangleMultiplicationIncoming")
            if inspect.isclass(c):
                in_cls = c
                found_module = modname if found_module is None else found_module

        if out_cls and in_cls:
            break

    return out_cls, in_cls, found_module


class MolfunProtenixPredictor:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.patched = False

        self._inject_triton_kernels()

        try:
            from protenix.model.protenix import Protenix  # pot variar segons versió
            self.model = None
            print("📦 Protenix found (model class import OK).")
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"✅ Weights path exists: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Could not import Protenix model class: {e}")
            self.model = None

    def _inject_triton_kernels(self):
        try:
            out_cls, in_cls, module_name = find_triangle_classes()

            if out_cls is None and in_cls is None:
                print("❌ No TriangleMultiplication classes found in protenix.")
                return

            # Guardem originals (per debug / rollback)
            if out_cls is not None and not hasattr(out_cls, "_molfun_orig_forward"):
                out_cls._molfun_orig_forward = out_cls.forward
            if in_cls is not None and not hasattr(in_cls, "_molfun_orig_forward"):
                in_cls._molfun_orig_forward = in_cls.forward

            def fast_out_fwd(self, z, mask=None):
                z_ln = self.layer_norm(z) if hasattr(self, "layer_norm") else z
                return FastTriangleUpdate.apply(z_ln, 0)

            def fast_in_fwd(self, z, mask=None):
                z_ln = self.layer_norm(z) if hasattr(self, "layer_norm") else z
                return FastTriangleUpdate.apply(z_ln, 1)

            if out_cls is not None:
                out_cls.forward = fast_out_fwd
                print(f"🚀 Patched Outgoing: {out_cls.__module__}.{out_cls.__name__}")

            if in_cls is not None:
                in_cls.forward = fast_in_fwd
                print(f"🚀 Patched Incoming: {in_cls.__module__}.{in_cls.__name__}")

            print(f"📍 Triangle module detected at: {module_name}")
            self.patched = True

        except Exception as e:
            print(f"❌ Patching failed: {e}")
            self.patched = False


def microbench_fast_triangle(n_res=76, channels=128, iters=200, warmup=50):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    x = torch.randn(1, n_res, n_res, channels, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(warmup):
        _ = FastTriangleUpdate.apply(x, 0)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = FastTriangleUpdate.apply(x, 0)
    torch.cuda.synchronize()
    t_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return t_ms


if __name__ == "__main__":
    N_RES = 76
    CH = 128

    print(f"🧬 Initializing Molfun-Protenix for Ubiquitin (N={N_RES}, C={CH})...")
    predictor = MolfunProtenixPredictor()

    if predictor.patched:
        print("🔥 Patch active. Running Triton microbenchmark...")
    else:
        print("⚠️ Patch not active. Running kernel-only microbenchmark anyway...")

    lat = microbench_fast_triangle(n_res=N_RES, channels=CH)
    print(f"✨ FastTriangleUpdate latency (avg): {lat:.4f} ms")
