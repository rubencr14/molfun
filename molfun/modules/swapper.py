"""
Runtime module swapping for pre-trained models.

ModuleSwapper allows replacing internal components of a model (attention,
structure module, blocks, etc.) while preserving the rest of the weights.
This is essential for research: take a pre-trained OpenFold, swap its
attention for FlashAttention, and fine-tune.

Usage
-----
    from molfun.modules.swapper import ModuleSwapper

    # Swap a single module by dotted path
    ModuleSwapper.swap(
        model.adapter.model,
        "structure_module",
        MyCustomSM(d_single=384, d_pair=128),
    )

    # Swap all modules matching a pattern
    n = ModuleSwapper.swap_all(
        model.adapter.model,
        pattern="msa_att",
        factory=lambda name, old: FlashAttention.from_standard(old),
    )
    print(f"Swapped {n} attention modules")

    # List swappable modules
    for name, mod in ModuleSwapper.discover(model):
        print(f"{name}: {type(mod).__name__}")
"""

from __future__ import annotations
from typing import Callable, Optional
import re

import torch.nn as nn


class ModuleSwapper:
    """
    Replace submodules inside an nn.Module at runtime.

    All methods are static — no instance state needed.
    """

    @staticmethod
    def swap(
        model: nn.Module,
        target_path: str,
        new_module: nn.Module,
        transfer_weights: bool = False,
    ) -> nn.Module:
        """
        Replace a single submodule identified by dotted path.

        Args:
            model: The parent model.
            target_path: Dotted path (e.g. "evoformer.blocks.0.msa_att").
            new_module: The replacement module.
            transfer_weights: If True, copy matching weights from old → new.

        Returns:
            The old (replaced) module.

        Raises:
            KeyError: If target_path does not exist in model.
        """
        parts = target_path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = _get_child(parent, part)

        attr_name = parts[-1]
        old_module = _get_child(parent, attr_name)

        if transfer_weights:
            _transfer_state(old_module, new_module)

        _set_child(parent, attr_name, new_module)
        return old_module

    @staticmethod
    def swap_all(
        model: nn.Module,
        pattern: str,
        factory: Callable[[str, nn.Module], nn.Module],
        transfer_weights: bool = False,
    ) -> int:
        """
        Swap all submodules whose full name matches a regex pattern.

        Args:
            model: The parent model.
            pattern: Regex pattern matched against the full dotted name
                     (e.g. "msa_att" matches "evoformer.blocks.3.msa_att").
            factory: ``factory(name, old_module) → new_module``.
            transfer_weights: If True, copy matching weights.

        Returns:
            Number of modules swapped.
        """
        compiled = re.compile(pattern)
        targets = []

        for name, mod in model.named_modules():
            if compiled.search(name):
                targets.append((name, mod))

        count = 0
        for name, old_mod in targets:
            new_mod = factory(name, old_mod)
            ModuleSwapper.swap(model, name, new_mod, transfer_weights=transfer_weights)
            count += 1

        return count

    @staticmethod
    def swap_by_type(
        model: nn.Module,
        old_type: type,
        factory: Callable[[str, nn.Module], nn.Module],
        transfer_weights: bool = False,
    ) -> int:
        """
        Swap all submodules of a given type.

        Args:
            model: The parent model.
            old_type: Type to match (e.g. ``nn.MultiheadAttention``).
            factory: ``factory(name, old_module) → new_module``.
            transfer_weights: If True, copy matching weights.

        Returns:
            Number of modules swapped.
        """
        targets = [
            (name, mod) for name, mod in model.named_modules()
            if isinstance(mod, old_type)
        ]

        count = 0
        for name, old_mod in targets:
            new_mod = factory(name, old_mod)
            ModuleSwapper.swap(model, name, new_mod, transfer_weights=transfer_weights)
            count += 1

        return count

    @staticmethod
    def discover(
        model: nn.Module,
        pattern: Optional[str] = None,
        module_type: Optional[type] = None,
    ) -> list[tuple[str, nn.Module]]:
        """
        List modules in the model, optionally filtered by name pattern or type.

        Useful for inspecting a model's structure before deciding what to swap.

        Returns:
            List of (dotted_name, module) tuples.
        """
        compiled = re.compile(pattern) if pattern else None
        results = []
        for name, mod in model.named_modules():
            if not name:
                continue
            if compiled and not compiled.search(name):
                continue
            if module_type and not isinstance(mod, module_type):
                continue
            results.append((name, mod))
        return results

    @staticmethod
    def summary(model: nn.Module, pattern: Optional[str] = None) -> str:
        """
        Human-readable summary of swappable modules.

        Returns a formatted string listing module paths, types, and param counts.
        """
        lines = []
        for name, mod in ModuleSwapper.discover(model, pattern=pattern):
            n_params = sum(p.numel() for p in mod.parameters(recurse=False))
            trainable = sum(p.numel() for p in mod.parameters(recurse=False) if p.requires_grad)
            cls_name = type(mod).__name__
            lines.append(f"  {name:60s} {cls_name:30s} params={n_params:>8,} trainable={trainable:>8,}")
        header = f"Modules in {type(model).__name__}"
        if pattern:
            header += f" matching '{pattern}'"
        return f"{header} ({len(lines)} found):\n" + "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────

def _get_child(module: nn.Module, name: str) -> nn.Module:
    """Get a child by name, supporting both attributes and ModuleList indices."""
    if name.isdigit():
        return module[int(name)]
    if hasattr(module, name):
        return getattr(module, name)
    raise KeyError(
        f"Module {type(module).__name__} has no child '{name}'. "
        f"Available: {[n for n, _ in module.named_children()]}"
    )


def _set_child(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    """Set a child by name, supporting both attributes and ModuleList indices."""
    if name.isdigit():
        parent[int(name)] = new_module
    else:
        setattr(parent, name, new_module)


def _transfer_state(old: nn.Module, new: nn.Module) -> int:
    """
    Transfer matching state_dict entries from old to new module.

    Returns number of transferred parameters.
    """
    old_state = old.state_dict()
    new_state = new.state_dict()

    transferred = 0
    for key in new_state:
        if key in old_state and old_state[key].shape == new_state[key].shape:
            new_state[key] = old_state[key]
            transferred += 1

    if transferred > 0:
        new.load_state_dict(new_state, strict=False)

    return transferred
