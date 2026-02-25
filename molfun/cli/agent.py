"""
CLI command for launching the autonomous research agent.
"""

from __future__ import annotations
from pathlib import Path
from enum import Enum
from typing import Annotated, Optional

import typer


class LLMBackend(str, Enum):
    openai = "openai"
    lmstudio = "lmstudio"
    ollama = "ollama"
    anthropic = "anthropic"
    litellm = "litellm"


def agent(
    objective: Annotated[str, typer.Argument(help="Research objective in natural language.")],
    llm: Annotated[LLMBackend, typer.Option(help="LLM backend.")] = LLMBackend.openai,
    model: Annotated[str, typer.Option(help="Model name (e.g. gpt-4o-mini, qwen2.5-7b).")] = "gpt-4o-mini",
    api_key: Annotated[Optional[str], typer.Option(envvar="OPENAI_API_KEY", help="API key (or set env var).")] = None,
    base_url: Annotated[Optional[str], typer.Option(help="API base URL (for LM Studio/vLLM).")] = None,
    port: Annotated[int, typer.Option(help="Port for local servers (LM Studio/Ollama).")] = 1234,
    pdbs: Annotated[Path, typer.Option(help="Directory with PDB/CIF structure files.")] = Path("data/structures"),
    data: Annotated[Optional[Path], typer.Option(help="CSV with pdb_id and affinity columns.")] = None,
    device: Annotated[str, typer.Option(help="Device: cuda, cpu, mps.")] = "cpu",
    max_steps: Annotated[int, typer.Option(help="Max agent steps.")] = 100,
    max_seq_len: Annotated[int, typer.Option(help="Max sequence length.")] = 128,
    memory_dir: Annotated[Path, typer.Option(help="Directory to persist experiment memory.")] = Path("runs/agent/memory"),
    tracker: Annotated[Optional[str], typer.Option(help="Tracker: console, wandb, comet, mlflow.")] = None,
    verbose: Annotated[bool, typer.Option(help="Verbose agent output.")] = True,
):
    """Launch an autonomous research agent for architecture search."""
    import torch
    from torch.utils.data import DataLoader, random_split

    llm_instance = _build_llm(llm, model, api_key, base_url, port)

    train_loader, val_loader, test_loader = _build_data(pdbs, data, max_seq_len, device)

    from molfun.agents.tools import MolfunTools
    from molfun.agents.researcher import ResearchAgent
    from molfun.agents.base import AgentConfig
    from molfun.agents.memory import ExperimentMemory

    tools = MolfunTools(train_loader, val_loader, test_loader, device=device)
    memory = ExperimentMemory(persist_dir=str(memory_dir))
    config = AgentConfig(max_steps=max_steps, verbose=verbose)

    tracker_instance = _build_tracker(tracker) if tracker else None

    research_agent = ResearchAgent(
        llm=llm_instance,
        tools=tools,
        memory=memory,
        config=config,
        tracker=tracker_instance,
    )

    typer.echo(f"Agent starting with {llm.value}/{model}")
    typer.echo(f"Objective: {objective}")
    typer.echo(f"Max steps: {max_steps} | Device: {device}")
    typer.echo(f"Memory: {memory_dir}")
    typer.echo("")

    result = research_agent.run(objective)

    best = result.best()
    if best:
        typer.echo(f"\nBest experiment: {best.summary_line()}")
    typer.echo(f"Total experiments: {result.count}")


def _build_llm(backend: LLMBackend, model: str, api_key, base_url, port):
    if backend == LLMBackend.openai:
        from molfun.agents.llm import OpenAIBackend
        kwargs = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIBackend(**kwargs)

    elif backend == LLMBackend.lmstudio:
        from molfun.agents.llm import lm_studio
        url = base_url or f"http://localhost:{port}"
        return lm_studio(model=model, base_url=url)

    elif backend == LLMBackend.ollama:
        from molfun.agents.llm import OllamaBackend
        url = base_url or f"http://localhost:11434"
        return OllamaBackend(model=model, host=url)

    elif backend == LLMBackend.anthropic:
        from molfun.agents.llm import AnthropicBackend
        kwargs = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        return AnthropicBackend(**kwargs)

    elif backend == LLMBackend.litellm:
        from molfun.agents.llm import LiteLLMBackend
        return LiteLLMBackend(model=model)

    raise typer.BadParameter(f"Unknown LLM backend: {backend}")


def _build_data(pdbs: Path, data_csv: Optional[Path], max_seq_len: int, device: str):
    import torch
    from torch.utils.data import DataLoader, random_split

    if data_csv and data_csv.exists():
        from molfun.data import AffinityDataset, DataSplitter
        from molfun.data.datasets.structure import collate_structure_batch
        dataset = AffinityDataset.from_csv(str(data_csv), str(pdbs))
        train_ds, val_ds, test_ds = DataSplitter.random(dataset, val_frac=0.15, test_frac=0.15, seed=42)
        collate = AffinityDataset.collate_fn
    elif pdbs.exists() and pdbs.is_dir():
        from molfun.data.datasets.structure import StructureDataset, collate_structure_batch
        pdb_paths = sorted(pdbs.glob("*.pdb")) + sorted(pdbs.glob("*.cif"))
        if not pdb_paths:
            raise typer.BadParameter(f"No PDB/CIF files found in {pdbs}")
        dataset = StructureDataset(pdb_paths=pdb_paths, max_seq_len=max_seq_len)
        n = len(dataset)
        n_test = max(1, int(n * 0.15))
        n_val = max(1, int(n * 0.15))
        n_train = n - n_val - n_test
        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )
        collate = collate_structure_batch
    else:
        from molfun.data.datasets.structure import StructureDataset, collate_structure_batch
        typer.echo("No data found — using synthetic dataset for demo.")
        _SyntheticDataset = _make_synthetic_dataset(max_seq_len)
        train_ds = _SyntheticDataset(6)
        val_ds = _SyntheticDataset(2)
        test_ds = _SyntheticDataset(2)
        collate = collate_structure_batch

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate)
    typer.echo(f"Data: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")
    return train_loader, val_loader, test_loader


def _make_synthetic_dataset(seq_len: int):
    """Create a minimal synthetic dataset class for demo/testing."""
    import torch
    from torch.utils.data import Dataset

    class SyntheticDataset(Dataset):
        def __init__(self, n: int):
            self.n = n
            self.seq_len = seq_len

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            L = self.seq_len
            features = {
                "sequence": "A" * L,
                "residue_index": torch.arange(L),
                "all_atom_positions": torch.randn(L, 3),
                "all_atom_mask": torch.ones(L),
                "seq_length": torch.tensor([L]),
            }
            label = torch.tensor([torch.randn(1).item()])
            return features, label

    return SyntheticDataset


def _build_tracker(name: str):
    if name == "console":
        from molfun.tracking import ConsoleTracker
        return ConsoleTracker(verbose=True)
    elif name == "wandb":
        from molfun.tracking import WandbTracker
        return WandbTracker(project="molfun-agent")
    elif name == "comet":
        from molfun.tracking import CometTracker
        return CometTracker(project_name="molfun-agent")
    elif name == "mlflow":
        from molfun.tracking import MLflowTracker
        return MLflowTracker(experiment_name="molfun-agent")
    return None
