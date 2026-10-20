# Kinase Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In molfun, fine-tune SimpleFold-100M on kinase structures and measure, on held-out kinase clusters, whether it beats the base model on accuracy and DFG/αC state recovery.

**Architecture:** A new `simplefold` backend plugs into molfun's existing facade (`MolfunStructureModel`) and training loop: the adapter computes SimpleFold's own flow-matching + smooth-lDDT losses and returns them through `TrunkOutput.extra`; a parameter-free `generative` head hands them to the loop. Kinase-specific logic (KLIFS table, chain extraction, state labeler) lives in `molfun/families/kinase/`. Leakage-safe splitting is a generic module in `molfun/data/leakage.py`. Evaluation metrics live in `molfun/eval/`.

**Tech Stack:** Python 3.10, PyTorch 2.11 cu128 (RTX 5090), SimpleFold fork `rubencr14/simplefold-optimization`, gemmi, MMseqs2, tmtools, scikit-learn, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-10-20-kinase-experts-design.md` (milestone **P Pilot**).

## Global Constraints

- Repo: `casa:~/Escritorio/projects/molfun`, branch `feat/simplefold-kinase-experts`. Commit after every task; push the branch; never commit to `main`.
- Conda env: `molfun-sf`, cloned from `nomosis` (keeps its working torch cu128 build). Never modify the `nomosis` env.
- No hard-coded paths: data locations come from `.env` keys `MOLFUN_KINASE_DATA` (= `~/Escritorio/projects/nomosis/nomosis-kinases/data`), `MOLFUN_WORK_DIR` (pilot outputs), `MOLFUN_SIMPLEFOLD_CCD_RDB` (Boltz `ccd.rdb`).
- `nomosis-kinases` data is read in place, never copied into git.
- Leakage rule (spec §5.3): a held-out target is dropped if any training chain has ≥ 40% identity over ≥ 80 aligned residues (MMseqs2 search). Held-out clusters chosen by cluster count; no held-out cluster may hold > 2% of all structures.
- Pilot state = DFG ∈ {in, out} × αC ∈ {in, out}. KLIFS `out-like` counts as `out`; `na` → `unassigned` (excluded from state metrics). The 3-way KinCoRe DFG comes in M0.
- Pilot held-out clusters become **validation** clusters in M0, never test (we will have looked at them).
- Report "100M trainable trunk + frozen ESM2-3B" — never "100M model".
- Existing tests and `ruff check` / `ruff format --check` must stay green.
- Docstrings on public functions; one responsibility per file.

## Review Focus

1. **Name collision:** SimpleFold uses top-level packages `datasets`, `model`, `utils`; the env has HuggingFace `datasets` 5.0.0. Importing HF `datasets` first must not break the SimpleFold backend (Task 3 test).
2. **Sequence-only sampling:** `sample()` must build inputs from the sequence alone (FASTA path), never from the experimental structure (Task 5 test asserts no coordinates enter).
3. **Residue correspondence:** predicted residue *i* ↔ experimental `label_seq_id = i+1`; unresolved experimental residues are skipped in metrics, never zero-filled (Task 7 test).
4. **Save/load of a full fine-tune:** `load()` must restore trunk weights, not only PEFT/head; loading into a fresh model must reproduce the same parameters (Task 4 test).
5. **Over-represented kinases:** training samples by UniProt, so CDK2's hundreds of entries count once per epoch (Task 6 test).

---

### Task 1: Environment and pilot configuration

**Files:**
- Create: `env/molfun-sf.md` (how the env was built: clone command, extra pip installs with versions, `pip freeze` excerpt)
- Modify: `setup.py` (new extra `simplefold`: `gemmi>=0.7`, `tmtools>=0.2`, `hydra-core>=1.3`, `lightning==2.5.2`, `einops`, `mashumaro`, `fair-esm`)
- Modify: `.env.example` (add the three `MOLFUN_*` keys above, with placeholder values)
- Create: `molfun/config/paths.py`
- Test: `tests/config/test_paths.py`

**Interfaces:**
- Produces: `pilot_paths() -> PilotPaths` with fields `kinase_data: Path`, `work_dir: Path`, `ccd_rdb: Path`; raises `RuntimeError` naming the missing key when a key is unset.

- [ ] **Step 1:** Create env: `conda create -n molfun-sf --clone nomosis`; `pip install -e ".[simplefold]"`; `pip install -e <fork checkout> --no-deps` (fork pinned to the pushed commit of `feat/padding-aware-batching` merged with `feat/static-feature-cache`, or `main` if it already contains both — record the commit hash in `env/molfun-sf.md`). Verify: `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
- [ ] **Step 2:** Download `ccd.rdb` (80 MB) and `symmetry.pkl` (220 MB) from `https://boltz1.s3.us-east-2.amazonaws.com/` into `$MOLFUN_WORK_DIR/boltz/`; install `redis-server` from conda-forge into `molfun-sf`.
- [ ] **Step 3: Write failing test** `test_pilot_paths_reads_env(monkeypatch, tmp_path)` (sets the three keys → fields equal those paths) and `test_pilot_paths_missing_key_names_it` (unset `MOLFUN_KINASE_DATA` → `RuntimeError` whose message contains `MOLFUN_KINASE_DATA`).
- [ ] **Step 4:** Run `pytest tests/config/test_paths.py -v` → FAIL (module missing).
- [ ] **Step 5:** Implement `pilot_paths()` reading `os.environ` after `dotenv.load_dotenv()`.
- [ ] **Step 6:** Run the test → PASS. Run `pytest -q -m "not gpu and not slow"` → no new failures.
- [ ] **Step 7:** Commit `feat: add simplefold extra, pilot paths and env notes`.

### Task 2: Generative contract in the core (types, head, bf16, registry)

**Files:**
- Modify: `molfun/core/types.py` (`TrunkOutput.single_repr` becomes `torch.Tensor | None = None`; add `SampledStructure`)
- Create: `molfun/heads/generative.py`
- Modify: `molfun/models/structure.py:54-67` (register head `"generative"`; `_register_adapters` registers each backend independently, skipping one whose import fails, so a custom or missing backend never blocks the others)
- Modify: `molfun/training/base.py` (new `amp_dtype: str = "float16"` param; `"bfloat16"` → autocast bf16 and GradScaler disabled; device type from `model.device`, not hard-coded `"cuda"`)
- Test: `tests/models/test_generative_contract.py`, `tests/training/test_amp_dtype.py`

**Interfaces:**
- Produces:
  - `SampledStructure(sequence: str, mmcif: str, plddt: np.ndarray | None = None)` with method `ca_coords() -> np.ndarray` (shape `(L, 3)`, `NaN` rows for residues without CA).
  - `GenerativeLossHead(nn.Module)`: `forward(trunk_output, mask=None, batch=None) -> dict[str, Tensor]` returns `trunk_output.extra["losses"]`; `loss(preds, targets=None, loss_fn=None) -> dict[str, Tensor]` returns `preds` with key `"total"` first. Raises `KeyError` with a clear message if `"losses"` is missing.
  - `FinetuneStrategy(..., amp_dtype="float16" | "bfloat16")`.

- [ ] **Step 1: Write failing tests:**
  - `test_generative_head_returns_total_first`: extra `{"flow_mse": a, "total": b}` → `list(head.loss(head(out)))[0] == "total"`.
  - `test_generative_head_missing_losses_raises`.
  - `test_trunk_output_without_single_repr`: `TrunkOutput(extra={})` constructs.
  - `test_register_adapters_survives_failing_backend` (monkeypatch one backend import to raise `ImportError` → the other is still registered).
  - `test_bf16_disables_grad_scaler`: a toy adapter + generative head trained 2 steps with `amp_dtype="bfloat16"` on CPU-or-CUDA → loss finite, `scaler.is_enabled() is False`.
- [ ] **Step 2:** Run them → FAIL.
- [ ] **Step 3:** Implement the changes listed in Files.
- [ ] **Step 4:** Run the new tests → PASS; run `pytest -q tests/training tests/models -m "not gpu and not slow"` → no regressions.
- [ ] **Step 5:** Commit `feat: add generative head, optional single_repr, bf16 training`.

### Task 3: SimpleFold backend — load, losses, freezing

**Files:**
- Create: `molfun/backends/simplefold/__init__.py`, `config.py`, `imports.py`, `adapter.py`
- Modify: `molfun/hub/registry.py` (spec `simplefold_100M`: backend `simplefold`, URL `https://ml-site.cdn-apple.com/models/simplefold/simplefold_100M.ckpt`, filename `simplefold_100M.ckpt`, preset `foldingdit_100M`, size 387 MB; `get_config` returns `SimpleFoldConfig(architecture=preset)` for this backend)
- Test: `tests/backends/test_simplefold_adapter.py` (unit, CPU, tiny architecture) and `tests/backends/test_simplefold_real.py` (`@pytest.mark.gpu`)

**Interfaces:**
- Produces:
  - `SimpleFoldConfig(architecture: str = "foldingdit_100M", esm_model: str = "esm2_3B", multiplicity: int = 16, num_steps: int = 500, tau: float = 0.1)`.
  - `import_simplefold() -> ModuleType` in `imports.py`: puts the installed fork's `src/simplefold` first on `sys.path`, imports its `model`/`boltz_data_pipeline`/`utils` modules, and restores any pre-existing `sys.modules["datasets"]` entry afterwards.
  - `SimpleFoldAdapter(BaseAdapter)` with `__init__(model=None, config=None, weights_path=None, device="cuda")`; in training mode `forward(batch) -> TrunkOutput(extra={"losses": {"total", "flow_mse", "smooth_lddt"}})` by calling upstream `SimpleFold.flow_matching_train_step` (its `self.log` silenced); `freeze_trunk()`/`unfreeze_trunk()` toggle the architecture only (ESM always frozen); `get_trunk_blocks()` = the residue trunk's `blocks` ModuleList (not the atom encoder/decoder); `peft_target_module` = the architecture; `default_peft_targets = ["qkv", "proj", "w1", "w2"]`.

- [ ] **Step 1: Write failing tests:**
  - `test_imports_coexist_with_hf_datasets`: `import datasets` (HF) then `import_simplefold()` → succeeds and `sys.modules["datasets"]` is still the HF package afterwards.
  - `test_adapter_freeze_keeps_esm_frozen`: after `unfreeze_trunk()`, no ESM parameter has `requires_grad`.
  - `test_forward_returns_three_losses` (tiny random architecture, synthetic batch from one 30-residue fixture `tests/fixtures/simplefold/tiny_record/`): keys `{"total","flow_mse","smooth_lddt"}`, `total ≈ flow_mse + smooth_lddt`, all finite.
  - GPU: `test_load_100M_weights_strict`: `from_pretrained("simplefold_100M", head="generative")` loads with `strict=True`; trunk parameter count within 5% of 100M.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement; build the upstream `SimpleFold` module from its Hydra configs (`configs/model/simplefold.yaml` + `architecture/<preset>.yaml`) with `hydra.compose`, load the released state dict into `.model` with `strict=True`.
- [ ] **Step 4:** Run unit tests → PASS; run the GPU test on casa → PASS.
- [ ] **Step 5:** Commit `feat: add simplefold backend with flow-matching losses`.

### Task 4: Save/load of full fine-tunes

**Files:**
- Modify: `molfun/models/structure.py` (`save()`/`load()`)
- Test: `tests/models/test_save_load_trunk.py`

**Interfaces:**
- Produces: `save(path)` also writes `trunk.pt` (the state dict of every adapter parameter with `requires_grad=True`) when no PEFT is attached; `load(path)` restores it and raises `RuntimeError` listing missing or unexpected keys.

- [ ] **Step 1: Write failing test** `test_full_finetune_roundtrip` (toy adapter: perturb trunk weights, save, load into a fresh model → all trunk tensors equal) and `test_load_reports_key_mismatch`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS; existing save/load tests still pass.
- [ ] **Step 5:** Commit `fix: save and load trunk weights for full fine-tunes`.

### Task 5: Sequence-only sampling

**Files:**
- Modify: `molfun/backends/simplefold/adapter.py` (add `sample`)
- Modify: `molfun/models/structure.py` (add `MolfunStructureModel.sample`)
- Test: `tests/backends/test_simplefold_sample.py`

**Interfaces:**
- Produces:
  - `SimpleFoldAdapter.sample(sequence: str, num_samples: int = 5, num_steps: int | None = None, tau: float | None = None, seed: int = 0) -> list[SampledStructure]` — builds the input through upstream's FASTA path (`process_fastas` → `process_one_inference_structure`), samples with the Euler–Maruyama sampler, writes each sample to mmCIF text with upstream's writer. Uses EMA/eval weights; no gradients.
  - `MolfunStructureModel.sample(sequence, **kwargs) -> list[SampledStructure]`; raises `NotImplementedError` for adapters without `sample`.

- [ ] **Step 1: Write failing tests:**
  - `test_sample_uses_sequence_only`: monkeypatch upstream `process_one_inference_structure` to capture its `structure_path`; assert it lies under the adapter's temporary FASTA cache, never under any experimental data directory.
  - `test_sample_count_and_length` (tiny model, 20-residue sequence, `num_samples=3`, `num_steps=5`) → 3 results, each `ca_coords().shape == (20, 3)`.
  - `test_sample_seed_reproducible`: same seed → identical CA coordinates.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat: add sequence-only sampling to the simplefold backend`.

### Task 6: Kinase pilot dataset and leakage-safe split

**Files:**
- Create: `molfun/families/__init__.py`, `molfun/families/kinase/__init__.py`, `klifs.py`, `chains.py`, `pilot_split.py`
- Create: `molfun/data/leakage.py`
- Create: `molfun/backends/simplefold/records.py` (preprocessing + training dataset)
- Test: `tests/families/test_klifs.py`, `tests/families/test_chains.py`, `tests/data/test_leakage.py`, `tests/families/test_pilot_split.py`, `tests/backends/test_simplefold_records.py`

**Interfaces:**
- Produces:
  - `load_klifs_table(master_csv: Path) -> pd.DataFrame` — curated rows only; columns `structure_id:str, pdb, chain, alt, uniprot, kinase, family, group, species, pocket, resolution, dfg, ac_helix` with `dfg`/`ac_helix` ∈ {`"in"`, `"out"`, `None`} (`out-like`→`out`, `na`→`None`).
  - `extract_chain(cif_gz: Path, chain: str, out_path: Path) -> ChainInfo(seqres: str, n_observed: int)` — gemmi; first model, one chain, protein residues only, highest-occupancy altloc, writes mmCIF named `<structure_id>.cif`.
  - `mmseqs_clusters(seqs: dict[str, str], min_seq_id: float = 0.4, coverage: float = 0.5) -> dict[str, str]` — raises `FileNotFoundError` if `mmseqs` is missing (no silent fallback).
  - `select_holdout(cluster_of: dict[str,str], fraction: float, max_cluster_share: float = 0.02, seed: int = 0) -> set[str]` (cluster ids).
  - `leakage_hits(test: dict[str,str], train: dict[str,str], max_identity: float = 0.4, min_aln_len: int = 80) -> pd.DataFrame` (columns `query, target, identity, aln_len`).
  - `build_pilot_split(table, structures_dir, out_dir, seed=0) -> PilotSplit(train_ids: list[str], eval_targets: pd.DataFrame, report: dict)` — writes `train_ids.json`, `eval_targets.csv`, `leakage_report.json`, `uniprot_groups.json` into `out_dir`. Eval targets: per held-out UniProt up to two structures, one per distinct pilot state when available. Held-out fraction 0.15 of clusters.
  - `prepare_simplefold_records(cif_dir: Path, out_dir: Path, ccd_rdb: Path, port: int = 7777) -> Path` — starts and stops `redis-server` around upstream `process_mmcif` (no `--use-assembly`) and `process_structure`; returns the tokenized dir.
  - `SimpleFoldRecordDataset(tokenized_dir, target_dir, groups: dict[str, list[str]], max_tokens=256, max_atoms=2304)` — one item per group (UniProt), random member per draw; reuses upstream cropper/featurizer/collate; the `datasets` package of the fork is not imported.

- [ ] **Step 1: Write failing tests:**
  - `test_klifs_normalizes_labels` (fixture CSV with `out-like`, `na`, non-curated rows → `out`, `None`, dropped).
  - `test_extract_chain_single_chain_no_hetero` (fixture 2-chain mmCIF with ligand + waters → one chain, no HETATM water/ligand; `seqres` equals the entity sequence).
  - `test_mmseqs_missing_raises` (PATH without mmseqs → `FileNotFoundError`).
  - `test_leakage_hits_flags_close_homolog` (a test sequence 60% identical over 120 residues is flagged; a 30% one is not).
  - `test_select_holdout_respects_share_cap`.
  - `test_pilot_split_no_leak` (fixture of ~12 sequences: report shows zero hits; no UniProt on both sides).
  - `test_dataset_samples_by_group` (groups `{A: 50 ids, B: 1 id}` → `len(ds) == 2`).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Run the real build on casa: `python -m molfun.families.kinase.pilot_split` (reads `pilot_paths()`), then `prepare_simplefold_records`. Expected: `leakage_report.json` shows 0 hits, ≥ 30 held-out UniProts, ≥ 50 eval targets, and states per eval target counted. If fewer than 30 UniProts, raise the fraction to 0.20 and record that in the report.
- [ ] **Step 6:** Commit `feat: add kinase pilot dataset with leakage-safe hold-out`.

### Task 7: Evaluation metrics and the state labeler

**Files:**
- Create: `molfun/eval/__init__.py`, `molfun/eval/structure.py`
- Create: `molfun/families/kinase/states.py`
- Test: `tests/eval/test_structure_metrics.py`, `tests/families/test_states.py`

**Interfaces:**
- Produces:
  - `paired_ca(pred: SampledStructure, ref_cif: Path, chain: str) -> tuple[np.ndarray, np.ndarray]` — CA pairs where predicted residue *i* ↔ reference `label_seq_id == i+1`; unresolved reference residues are dropped.
  - `ca_rmsd(p, r) -> float` (after Kabsch), `lddt_ca(p, r, cutoff=15.0) -> float` (thresholds 0.5/1/2/4 Å), `tm_score(p, r) -> float` (tmtools, normalised by reference length).
  - `locate_motif(pocket: str, seq: str) -> tuple[int, int, int] | None` (β3-Lys, αC-Glu, DFG-Phe at KLIFS positions 17/24/82; ported from `nomosis-kinases/src/model/experts/dfg_expert.py`, with a docstring citing it).
  - `motif_features(structure: gemmi.Structure, idx) -> dict[str, float]` (CA distances Phe–Glu, Phe–Lys, Lys–Glu; Lys NZ–Glu CD when atoms exist).
  - `StateLabeler.fit(features: pd.DataFrame, dfg: pd.Series, ac: pd.Series) -> StateLabeler`, `.predict(features) -> pd.DataFrame` (`dfg`, `ac_helix`), `.save(path)` / `.load(path)` (JSON thresholds; depth-2 decision trees).

- [ ] **Step 1: Write failing tests:**
  - `test_rmsd_zero_for_identical` and `test_rmsd_invariant_to_rotation`.
  - `test_lddt_one_for_identical`, `test_lddt_drops_with_noise`.
  - `test_paired_ca_skips_unresolved` (reference missing residues 5–9 → pairs = L − 5, no zeros).
  - `test_locate_motif_on_known_pocket` (one real KLIFS pocket + sequence fixture → expected indices).
  - `test_labeler_roundtrip_json`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** On casa, fit the labeler on **training** structures against KLIFS labels and score it on held-out experimental structures. Gate: DFG balanced accuracy ≥ 0.90; αC ≥ 0.80. If αC misses, the pilot reports DFG-only states (recorded in the report).
- [ ] **Step 6:** Commit `feat: add structure metrics and kinase state labeler`.

### Task 8: Pilot recipe — train, sample, compare

**Files:**
- Create: `recipes/kinase_pilot.yaml`
- Create: `molfun/families/kinase/pilot.py` (entry points `train_pilot(cfg) -> Path`, `evaluate_pilot(cfg, checkpoint: Path | None) -> pd.DataFrame`, `pilot_report(base: pd.DataFrame, tuned: pd.DataFrame) -> dict`)
- Test: `tests/families/test_pilot_report.py`

**Interfaces:**
- Consumes: Tasks 2–7.
- Produces: `pilot_report` returns `{"n_targets", "n_uniprots", "tm": {"base","tuned","delta","ci95"}, "lddt": {...}, "rmsd": {...}, "state_top1": {...}, "state_coverage@5": {...}}`, with 95% CIs from 2,000 bootstrap resamples **over held-out UniProts**, paired (same targets).

Recipe values: `FullFinetune(lr=2e-5, layer_lr_decay=1.0, warmup_steps=200, scheduler="cosine", ema_decay=0.999, grad_clip=2.0, amp_dtype="bfloat16")`, `epochs=20` (one sample per training UniProt per epoch), batch size 1 (SimpleFold multiplicity 16), crop 256 tokens. Sampling: `num_samples=5`, upstream default `num_steps=500`, `tau=0.1`, seed fixed per target. Metrics per target: mean over samples and best-of-5 (labelled oracle). States: top-1 = the first sample's state; coverage@5 = fraction of a UniProt's distinct experimental states matched by ≥ 1 sample.

- [ ] **Step 1: Write failing test** `test_pilot_report_bootstrap_by_uniprot` (synthetic: 10 UniProts, tuned = base + 0.05 TM → delta 0.05, CI excludes 0; one UniProt with 20 targets does not shift the CI more than one with 1) and `test_pilot_report_keys`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `pilot_report`, then `train_pilot` (applies EMA before `save`) and `evaluate_pilot`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** On casa, run training in the background (`setsid nohup … > $MOLFUN_WORK_DIR/logs/train.log`). Expected: finite losses, `total` trending down, checkpoint saved; wall time logged.
- [ ] **Step 6:** Run `evaluate_pilot` for base and tuned; write `pilot_report.json` and `pilot_report.md` (table + the go/no-go line from spec §5.8) into `$MOLFUN_WORK_DIR`.
- [ ] **Step 7:** Commit `feat: add kinase pilot recipe and report`; push; open a PR to `main` once all tests pass.
