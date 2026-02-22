"""
molfun info — system and registry summary.
"""

from pathlib import Path

_DEFAULT_WEIGHTS = Path.home() / ".molfun" / "weights" / "finetuning_ptm_2.pt"


def info():
    """Show system info, available strategies, losses, and heads."""
    import sys
    import platform

    # ── Versions ──────────────────────────────────────────────────────
    print(f"Python    : {platform.python_version()}")

    try:
        import torch
        cuda_str = f" | CUDA {torch.version.cuda}" if torch.cuda.is_available() else " | CPU only"
        print(f"PyTorch   : {torch.__version__}{cuda_str}")
    except ImportError:
        print("PyTorch   : not installed")

    # ── GPU ───────────────────────────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram  = props.total_memory / 1024 ** 3
            print(f"GPU       : {props.name} | {vram:.0f} GB VRAM")
        else:
            print("GPU       : not available")
    except Exception:
        pass

    print()

    # ── Weights ───────────────────────────────────────────────────────
    found = "✓" if _DEFAULT_WEIGHTS.exists() else "✗ not found"
    print(f"Weights   : {_DEFAULT_WEIGHTS}  [{found}]")

    # ── OpenFold ──────────────────────────────────────────────────────
    try:
        import openfold
        print(f"OpenFold  : installed ({Path(openfold.__file__).parent})")
    except ImportError:
        print("OpenFold  : NOT installed")

    print()

    # ── Registries ────────────────────────────────────────────────────
    try:
        from molfun.losses import LOSS_REGISTRY
        print(f"Losses    : {', '.join(sorted(LOSS_REGISTRY))}")
    except Exception as e:
        print(f"Losses    : error loading registry — {e}")

    try:
        from molfun.models.structure import HEAD_REGISTRY
        print(f"Heads     : {', '.join(sorted(HEAD_REGISTRY))}")
    except Exception as e:
        print(f"Heads     : error loading registry — {e}")

    strategies = ["lora", "partial", "full", "head_only"]
    print(f"Strategies: {', '.join(strategies)}")
