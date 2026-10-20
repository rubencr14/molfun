# Kinase Experts — small generative specialists vs. generalist folding models

- **Date:** 2026-10-20
- **Status:** draft, awaiting owner approval
- **Repos:** `molfun`, `moldata` (both `rubencr14/*`), SimpleFold fork `rubencr14/simplefold-optimization`
- **Machine:** `casa` (Ubuntu, RTX 5090 32 GB, 24 cores, 125 GB RAM)

## 1. Intent

Build a protein structure predictor for **protein kinase domains** that is small
(SimpleFold-100M trunk + family expert) and beats large generalist models on
kinases, with a test pipeline strict enough that the result survives a hostile
reviewer. The long-term idea is a mixture of family experts: a router decides
the family of a sequence and dispatches it to a specialised predictor, falling
back to the generalist when unsure. Kinases are family #1; the design must make
a second family a matter of data and config, not new architecture — but nothing
is built for families that don't exist yet.

Why kinases: largest family in the PDB (7,892 entries with PF00069/PF07714;
1,195 released since 2024-01-01), domain length (~250–300 aa) fits SimpleFold-100M
training on one 5090, top drug-target class (relevant to Nomosis), and a known
generalist weakness — **conformational states** (DFG-in/out, αC-in/out,
activation loop). AlphaFold-style models tend to predict one dominant state;
SimpleFold is generative, so a family expert can learn to sample the family's
state landscape.

The owner's framing: "aim at the stars to reach the moon". The headline we're
after is "small expert beats big generalist on kinases", and the most
interesting version of it is recovering experimental states the generalists miss.

## 2. The claim, in three levels

| Level | Claim | Compared against |
|---|---|---|
| Floor | Kinase expert beats the model it was fine-tuned from | SimpleFold-100M (generalist) |
| Moon (headline) | 100M + expert beats the largest SimpleFold on kinases, at a fraction of trunk size and time | SimpleFold-3B |
| Stars | Beats AlphaFold2 on kinases on a leak-free test | AF2 (AFDB predictions) |

"Beats" means: paired comparison on the same targets, cluster-bootstrap 95% CI of
the difference excluding zero, on the primary metric of that test tier (§5.6).
Agreed with the owner: **the main victory may be state recovery even if mean
accuracy only ties the large models.** Both are always reported.

Honesty note that must appear in every report: SimpleFold-100M conditions on a
frozen ESM2-3B (≈96% of its inference VRAM). "Small" means **100M trainable
folding trunk**, not 100M total parameters.

## 3. Goals, non-goals, anti-goals

### Goals
1. A leakage-safe data and split pipeline for kinase domains, with an automatic
   leakage report and a locked test set.
2. A SimpleFold backend in molfun that can fine-tune (LoRA and full) and sample.
3. A kinase expert, evaluated on three test tiers against SimpleFold-100M,
   SimpleFold-3B, AF2 and a same-budget generalist control.
4. A router (kinase vs. not, with domain boundaries) and an end-to-end
   `predict` command: sequence in → routed → N samples ranked by confidence →
   mmCIF + per-sample state labels + JSON summary.
5. moldata able to mirror whole datasets reliably (PDB + metadata first), with
   provenance, and without its current data-loss behaviours.

### Non-goals (for now)
- Fixing molfun's OpenFold featurizer/loss bugs (separate follow-up track).
- A second family; multi-family routing.
- Full-length multi-domain chains, complexes, ligands. v1 predicts the kinase
  domain region the router cuts out.
- OpenProteinSet MSAs, full AFESM mirror, MinIO sharded mirroring.
- Replacing ESM2-3B with a smaller language model (needs retraining from scratch).
- Integration with the Nomosis viewer.

### Anti-goals (failure even if the numbers look good)
- Any headline number from a split where leakage rules (§5.3) are violated.
- Choosing a model, checkpoint, hyperparameter or threshold by looking at test.
- Presenting oracle best-of-N (picked with the true structure) as achievable
  accuracy. Headline uses confidence-selected samples.
- Statistics dominated by over-represented kinases (CDK2 alone has hundreds of
  entries): aggregate and bootstrap over clusters, not chains.
- Comparing at unequal compute without saying so.
- Breaking existing molfun/moldata behaviour or tests; deleting user data.

## 4. Constraints

- **Hardware:** single RTX 5090 (sm_120, needs CUDA ≥ 12.8 builds). Everything
  must train/evaluate on it. SimpleFold-3B inference must fit in 32 GB (bf16).
- **Data root:** `/data/moldata` → symlink to `/mnt/Datos SSD/moldata` (3.4 TB
  NVMe). Configured via `MOLDATA_LOCAL_ROOT` in `.env`; no hard-coded paths in
  code. HDDs (`/mnt/Datos HDD 1|2`) are overflow for raw archives.
- **Already on disk / in flight:** SimpleFold-100M checkpoint; OpenFold weights;
  phase-1 mirror script (PDB divided mmCIF via rsync from PDBe; holdings;
  `pdb_seqres`; CCD; RCSB clusters 30–100; SIFTS chain→UniProt/Pfam/CATH/SCOP2/
  taxonomy/InterPro/EC/GO; Pfam-A HMMs + clans; CATH; ECOD; AFDB SwissProt v6
  mmCIF tar; Apple's SimpleFold training lists and benchmark predictions), each
  file recorded with URL, date and sha256 in `/data/moldata/PROVENANCE.tsv`.
- **Training cutoffs of the compared models:** SimpleFold PDB 2020-05-01
  (verified in its data configs and paper); AF2 PDB 2018-04-30. SimpleFold also
  saw distilled AFDB SwissProt (~270K) and AFESM (~1.9M) predictions — so most
  natural kinase *sequences* were seen by it as predicted structures. Our test
  design accounts for this (§5.3).
- **Environment:** Python in conda (`nomosis` env has torch 2.11 cu128, openfold,
  peft, mmseqs, gemmi). The builder may create a dedicated env if SimpleFold's
  deps conflict; document it.
- **Code standards (owner's rules):** simple functions over frameworks; docstrings;
  tests for every feature; no hard-coded IPs/ports/keys (`.env`); English for all
  spec/plan/docs artifacts; performance first (cache what doesn't change, batch).
- **Git (owner's decision):** never commit to `main` directly. Work on feature
  branches (molfun, moldata, the SimpleFold fork); commit and push them freely;
  land work on `main` through PRs (branch → main) once its tests pass. The fork
  commit molfun pins must be pushed.
- molfun's existing 642 passing tests and ruff must stay green.

## 5. The how — decided parts

### 5.0 Prior work we build on (`~/Escritorio/projects/nomosis/nomosis-kinases`)

The owner's July 2026 project "Nomosis Kinase Modular Folding" already holds:
- **KLIFS + KinCoRe** downloads: 13,445 kinase structures (6,331 PDB entries,
  all mmCIF on disk), 329 UniProts, **human and mouse only**; pocket alignment
  (85 KLIFS residues), ligands, quality flags, in `data/processed/master.csv`.
- **State labels from two sources** (96% DFG agreement): KLIFS `DFG`/`aC_helix`
  and KinCoRe `dfg_spatial`/`dihedral_label`. Counts: DFG in 11,504 / out 1,271 /
  out-like 505; αC in 10,723 / out 2,497.
- A **geometric DFG labeler** (`experts/dfg_expert.py`, balanced acc 0.94,
  KLIFS numbering β3-Lys=17, αC-Glu=24, DFG-Phe=82) usable on predicted structures.
- Lessons: its from-scratch generator plateaued (TM 0.48 de novo, 0.82 with
  templates); ESMFold already folds kinases at TM 0.92 but predicts one state
  (p38α: "in" vs experimental "out"). Both support starting from a pretrained
  folding model and making **states** the target.
- Its splits are by UniProt / subfamily / scaffold / DFG. Split-by-UniProt leaks
  orthologs (mouse vs human, ~95% identity, different UniProt); this design's
  40% identity clusters replace them.

Reuse rules: read its data in place (path configured in `.env`, never copied
into git); port the DFG labeler into molfun's kinase module with a reference to
its origin; do not depend on its Python package at runtime.

**Still missing, to download (a few GB, not the whole PDB):** release dates
(wwPDB holdings), the ~1,500 PDB kinase entries outside KLIFS (non-human /
non-mouse — the natural material for Tier A), RCSB clusters, SIFTS Pfam/UniProt,
`pdb_seqres`, CCD + Boltz `ccd.rdb`/`symmetry.pkl` for SimpleFold processing,
AFDB entries **for kinase UniProts only** (per-entry API, not the 40 GB tar),
Apple's SimpleFold training lists, and a non-kinase PDB sample for the control
and router negatives. The whole-PDB mirror is deferred until a second family.

### 5.1 Data layer (moldata)

- **Mirror:** turn the phase-1 script into a `moldata mirror <source>` command:
  resumable (rsync, or HTTP Range into `*.part` + atomic rename), sha256/size
  recorded, non-zero exit on any failure, provenance file updated.
- **Parsing:** replace the hand-written mmCIF parser with gemmi. Its current
  output (resolution/dates always None, wrong entity counts) is wrong on real files.
- **Chain manifest** (parquet, versioned by mirror date): one row per polymer
  chain with PDB id, chain/entity id, SEQRES sequence, length, molecule type,
  initial release date (from holdings, not deposition date), method, resolution,
  UniProt accession + ranges (SIFTS), Pfam/CATH/ECOD annotations (SIFTS/ECOD),
  RCSB entity cluster ids at 30/40/50/70/90/95/100%.
- **P0 safety fixes:** `keep_local` defaults to keeping user data; downloads
  stream to disk instead of RAM; tar extraction filtered against path traversal;
  failures exit non-zero.
- Generic split builder + leakage report live in moldata (they are dataset
  logic); kinase is their first user.

### 5.2 Kinase dataset

- **Unit:** one kinase domain on one PDB chain. Domain boundaries from an HMM
  search (pyhmmer, PF00069 + PF07714, Pfam gathering thresholds) over the SEQRES
  sequence, mapped to observed residues. One model (first), highest-occupancy
  altloc.
- **Quality filters:** X-ray/cryo-EM ≤ 3.5 Å and NMR allowed but flagged;
  minimum observed fraction of the domain (builder sets, reports it).
- **Keep conformational diversity:** do **not** deduplicate multiple structures
  of the same protein away — they carry the states. Redundancy is handled by
  sampling weights (inverse cluster size), not deletion.
- **State labels:** a deterministic labeler from coordinates, following the
  Modi & Dunbrack (KinCoRe) nomenclature. The **same code** labels experimental
  and predicted structures. Key residues (β3-Lys, αC-Glu, DFG) located via the
  HMM alignment. *(Designer's call, fixed before the test lock at M0:)*
  - The **state** used everywhere (top-1 accuracy, coverage@N, Tier B dedup,
    Tier B-novel-state membership) is the coarse pair
    **spatial DFG label (in / out / inter) × αC label (in / out)** — at most six
    states. Coarse on purpose: dihedral flips at cluster boundaries are not
    "new states".
  - The KinCoRe **dihedral label** (BLAminus, ABAminus, …) is computed and
    reported as a secondary, finer metric only.
  - **Sources:** for structures in KLIFS/KinCoRe, the published labels
    (KinCoRe `dfg_spatial`, KLIFS `aC_helix`) are the experimental truth. The
    geometric labeler (ported from `nomosis-kinases`) labels predicted
    structures and kinases outside KLIFS, and is validated against the published
    labels before use (its agreement is reported).
  - Structures where the DFG motif, β3-Lys or αC-Glu are unresolved get state
    `unassigned`: they count for accuracy metrics but never for state metrics or
    novel-state membership.
  - A state counts as "seen in training" if it appears in **any** training
    example of that 95% cluster, including AFDB distillation entries
    (conservative).
- **Distillation data:** AFDB SwissProt v6 kinase domains with domain pLDDT ≥ 70,
  mixed at a lower ratio than experimental data (AF2 is single-state biased; it
  must not wash out the experimental state diversity). *(Designer's call:)*
  AFDB models may have used PDB templates newer than our cutoffs, so the
  distillation set **drops every AFDB entry whose UniProt has any validation or
  test structure** (any tier). This shrinks the lever, accepted for safety.

### 5.3 Leakage-safe splits (the core of this project)

1. **Clustering:** MMseqs2 on **kinase domain sequences** (not full chains) over
   the union of all PDB kinase domains (all dates) and AFDB SwissProt kinase
   domains. 40% identity, coverage ≥ 0.8. Also 95% clusters (≈ "same protein").
2. **Held-out clusters:** seeded selection of ~10% of 40%-clusters for **test**
   and ~5% for **validation**, chosen among clusters that have experimental
   structures released after 2020-05-01. Every member of a held-out cluster —
   PDB structure or AFDB entry, any date — is removed from **all** our training
   data (expert, control, router, distillation). *(Designer's call:)*
   - Clustering is greedy (MMseqs2 default set-cover mode), not connected
     components: kinases form a near-continuum at 30–50% identity, so connected
     components would likely collapse the family into one component.
   - Percentages are **by cluster count** among eligible clusters (those with
     post-2020-05-01 experimental structures). No single held-out cluster may
     hold more than 2% of all training structures (keeps CDKs etc. in training).
   - **Target size for Tier A:** ≥ 30 clusters and ≥ 100 structures after the
     §5.3.5 drop rule. If short, raise the test fraction step by step up to 20%
     of eligible clusters. If still short, Tier A is declared **underpowered**
     in every report, and the headline leads with Tier B / B-novel-state —
     consistent with the owner's choice that state recovery is the main victory.
   - Validation power for the M2 gate comes from both the held-out validation
     clusters **and** the 2023 structures of training clusters; the gate may be
     passed on either, and the report says which.
3. **Temporal cutoff for our training:** PDB structures released before
   2023-01-01 only. 2023 releases → validation. ≥ 2024-01-01 → test.
4. **Three test tiers** (one training set serves all):
   - **Tier A — new families of kinases (generalisation, primary for accuracy):**
     structures in held-out test clusters released ≥ 2020-05-01 (so not in any
     generalist's PDB training). Our model never saw anything within 40% identity.
   - **Tier B — new structures of known kinases (real-world use):** structures
     released ≥ 2024-01-01 in non-held-out clusters; at most one structure per
     (95%-cluster, state label) to stop CDK2 dominating.
   - **Tier B-novel-state — the breakthrough test:** Tier B structures whose
     state label never appears in our training data for that protein (95%
     cluster). This measures predicting a conformation, not remembering one.
5. **Enforcement:** after building, MMseqs2 search every test domain against
   every training domain; Tier A must have max identity < 40%, else the target
   is dropped from test (never moved into training). The build fails if the
   report can't be produced.
6. **Leakage report** (auto-generated per split version): counts per tier,
   identity histograms test→train, nearest training UniProt per target, and
   **generalist exposure** (whether each test sequence/UniProt appears in
   SimpleFold's SwissProt/AFESM lists or AF2's PDB cutoff). Generalist exposure
   is reported, not removed — it can only favour the generalists, which makes
   our comparison conservative.
7. **Locked test:** test lists written once with sha256 in a `LOCK` file. The
   evaluator refuses changed lists. Every test evaluation is appended to a
   `test_access.log` (git commit, model hash, date). Model selection uses
   validation only; test is opened at milestone M4 (and nowhere else).
8. **Router splits** follow the same held-out clusters; non-kinase negatives are
   held out by their own 40% clusters.

### 5.4 Models

- **Expert:** SimpleFold-100M official checkpoint, ESM2-3B frozen. Compare on
  validation: LoRA on trunk attention/MLP projections (a few ranks) vs. full
  fine-tune of the 100M trunk. Pick by validation. Reuse SimpleFold's own losses
  (flow matching + its lDDT-style term), augmentation and sampler. The pLDDT head
  is fine-tuned too, since confidence picks the sample.
- **Baselines (same targets, same metric code):**
  - SimpleFold-100M and SimpleFold-3B, official checkpoints and sampler settings.
  - AF2: AFDB v6 predictions for test chains whose domain maps exactly to a
    UniProt entry via SIFTS (AF2's PDB cutoff 2018 makes these legit). Coverage
    reported. AFDB may have used templates newer than 2018; reports label this
    baseline "generous to AF2".
  - *(Designer's call — input fairness:)* the expert sees the domain crop. Every
    generalist is run on **both** the same domain crop and the full PDB chain
    (SEQRES), when the full chain fits in 32 GB; both are scored on the domain
    region and the generalist is credited with **the better of the two**. AFDB
    is full-length by nature and scored on the domain region. Compute numbers
    are reported for the input that was credited.
- **Control (same-budget generalist):** SimpleFold-100M fine-tuned with the same
  steps and compute on kinase data plus an equal amount of non-kinase PDB data
  under the same cutoffs. Separates "specialisation helps" from "more/newer
  training data helps".
- **Test-time sampling:** N samples per target, ranked by predicted pLDDT. Report
  top-1-by-confidence (headline) and oracle best-of-N (labelled as upper bound).
  Compute-matched view: wall-clock on the 5090 at equal budget (e.g. N samples of
  100M vs. 1 sample of 3B), using the fork's inference optimisations.

### 5.5 Router

- **Task (v1):** kinase domain present? where? Designed so K families = K labels.
- **Labels:** Pfam HMM hits (pyhmmer, gathering thresholds).
- **Two routers, both evaluated:** (1) pyhmmer search itself (deterministic
  baseline); (2) a small classifier on the ESM2-3B embedding SimpleFold already
  computes (near-zero extra cost).
- **Decision:** route to the expert if p(kinase) ≥ τ, else SimpleFold-100M.
  τ chosen on validation for high precision.
- **End-to-end evaluation:** on a mixed set (Tier A kinases + held-out non-kinase
  chains): the routed system must be no worse than the generalist on non-kinases
  and better on kinases; routing errors counted.

### 5.6 Evaluation and scoreboard

- **Accuracy (kinase domain region):** lDDT-Cα (primary), TM-score, GDT-TS,
  Cα-RMSD after superposition. One tool for all models (OpenStructure, as Apple
  used, or US-align — builder's call, fixed before M1).
- **States:** top-1 state accuracy; **state coverage@N** for proteins with ≥ 2
  experimental states in a tier: fraction of states matched by ≥ 1 of N samples
  (same label and lDDT to that state's structure ≥ a threshold fixed before M1).
- **Confidence:** pLDDT–lDDT correlation (calibration).
- **Compute:** seconds per target and peak VRAM on the 5090.
- **Statistics:** paired differences, bootstrap over 40% clusters, n clusters and
  n structures always shown.
- **Primary metric per tier:** Tier A → lDDT-Cα; Tier B → lDDT-Cα; Tier
  B-novel-state → state coverage@N. The scoreboard shows all metrics for all tiers.

### 5.7 Code ownership and layout

- **moldata** (branch `feat/training-data`): mirror, gemmi parser, chain
  manifest, annotations, split builder, leakage report, P0 safety fixes.
- **molfun** (branch `feat/simplefold-kinase-experts`): SimpleFold backend with a
  generative contract (training returns a dict of loss terms, plus a `sample()`
  API), dataset reading moldata manifests, ESM feature cache, LoRA/full
  fine-tune, named-adapter save/load (fixes today's bug where `load()` skips the
  adapter), evaluation harness, state labeler, router, `predict` command,
  baseline runners, a kinase recipe.
- **SimpleFold fork:** pinned dependency on a commit of
  `rubencr14/simplefold-optimization` that includes the static-feature cache and
  padding-aware batching. Changes to SimpleFold internals go on a branch there.
- Kinase-specific code (domain HMMs, key residues, state labeler) sits in one
  self-contained place in molfun, so a second family adds its own module/config.

### 5.8 Milestones and gates

**Owner's rule:** everything is built **in molfun** (and moldata for data), as
reusable framework features that serve molfun's purpose — not one-off scripts.
The pilot below is the first vertical slice of that code, not throwaway work.

| Milestone | Done when | Gate |
|---|---|---|
| **P Pilot** (~1 week, ~10–15 h GPU) | In molfun: SimpleFold backend (fine-tune + sample through molfun's API), kinase dataset reading the `nomosis-kinases` data, a quick 40%-cluster hold-out with MMseqs2 leakage check, one fine-tune of SimpleFold-100M (~3–6 h), base vs fine-tuned on ~50 held-out proteins: TM-score, lDDT-Cα, DFG×αC top-1 and coverage@5 | **Go** if fine-tuned ≥ base on TM/lDDT and better on states. Otherwise stop and rethink with the owner before M0 |
| M0 Data | Mirror complete; chain manifest; kinase dataset; splits; leakage report; test locked | Leakage report passes (§5.3.5) |
| M1 Baselines | SimpleFold-100M/3B and AF2 scored on **validation**; metric code frozen | Metrics reproduce Apple's published CAMEO22 numbers for SimpleFold-100M within tolerance (sanity check of the harness) |
| M2 Expert v1 | LoRA vs full FT on validation | **Floor:** expert beats SimpleFold-100M on validation lDDT or state coverage (CI excludes 0). If not after both variants + a data-mix ablation → stop and report to the owner before continuing |
| M3 Levers | Best-of-N + confidence selection, distillation mix, router, control | Each lever kept only if it improves validation |
| M4 Final | Single locked test evaluation, scoreboard, model card, leakage report, figures | — |

## 6. Left to the builder

LoRA ranks/targets and learning rates; crop sizes; sampling weights; distillation
mixing ratio; quality thresholds (observed fraction, resolution) as long as they
are reported; N for test-time sampling; lDDT threshold for state matching (fixed
before M1); MMseqs2 parameters beyond identity/coverage; evaluation tool (OpenStructure vs
US-align); whether a separate conda env is needed; experiment tracking backend;
internal module names.

## 7. Budget (lean version, chosen with the owner)

Calendar **3–4 weeks after the pilot**; the 5090 is dedicated during training
and evaluation windows (mostly overnight), free for the owner while code is
written. Total GPU estimate **~120–150 h** (≈ 6 GPU-days), on top of the pilot:

| Block | Estimate | Cut vs. the full version |
|---|---|---|
| M0 data | 5–10 h | — |
| M1 baselines (validation) | ~25 h | 200 targets instead of 300; 3 samples for SimpleFold-3B instead of 5 |
| M2 expert v1 | ~15–20 h | 2 runs (one LoRA, one full fine-tune) instead of 4 |
| M3 levers | ~40 h | Best-of-N (no training), same-budget control, one distillation mix |
| M4 final | ~30–40 h | One pass |

All numbers are estimates: SimpleFold-100M takes ~6 s per target on this GPU
(owner's own benchmark); **SimpleFold-3B speed is unmeasured** and is the first
thing M1 measures, re-planning before anything expensive runs. ESM features:
computed on the fly during the pilot (as upstream SimpleFold does); from M0 on,
cache the combined representation with `esm_s_combine` frozen (≈1.5 MB per
chain) unless an ablation needs all 37 layers.

Environment: a dedicated conda env (e.g. cloned from `nomosis`, which already
has a working torch cu128 build for the 5090), so other projects using
`nomosis` are not disturbed.
