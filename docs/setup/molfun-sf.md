# `molfun-sf` environment (SimpleFold backend)

Built on `casa` (RTX 5090, CUDA 12.8) for the kinase pilot. It is a clone of
the `nomosis` env, which already has a working PyTorch build for the 5090
(sm_120); the clone keeps `nomosis` untouched for other projects.

## Build

```bash
conda create -y -n molfun-sf --clone nomosis
git clone https://github.com/rubencr14/simplefold-optimization.git ~/Escritorio/projects/simplefold-molfun
git -C ~/Escritorio/projects/simplefold-molfun checkout -B molfun/pin 62f1a225ec8b8fcac9f800832793e868c4ea4d08
conda activate molfun-sf
pip install "lightning==2.5.2" hydra-core hydra-colorlog mashumaro ihm modelcif \
    dm-tree p-tqdm timm biotite tmtools python-dotenv
pip install --no-deps fairscale   # imported by upstream model/simplefold.py
pip install --no-deps -e ~/Escritorio/projects/simplefold-molfun
pip install --no-deps -e .            # molfun itself
conda install -y -c conda-forge redis-server
```

The SimpleFold fork is pinned to `62f1a22` of `rubencr14/simplefold-optimization`
(`main`, which contains the static-feature cache and padding-aware batching).
The fork is installed with `--no-deps` so its pinned versions cannot replace
the env's PyTorch build; the script that builds the env checks that the torch
version is unchanged.

## Data files

Boltz processing files go to `$MOLFUN_WORK_DIR/boltz/`:

- `ccd.rdb` — https://boltz1.s3.us-east-2.amazonaws.com/ccd.rdb (served by
  `redis-server` during `process_mmcif`)
- `symmetry.pkl` — https://boltz1.s3.us-east-2.amazonaws.com/symmetry.pkl

## Environment keys (`.env`)

See `.env.example`: `MOLFUN_KINASE_DATA`, `MOLFUN_WORK_DIR`,
`MOLFUN_SIMPLEFOLD_CCD_RDB`.

## Installed versions (2026-10-20)

| Package | Version |
|---|---|
| torch | 2.11.0.dev20260206+cu128 (unchanged from `nomosis`) |
| lightning | 2.5.2 |
| hydra-core | 1.3.7 |
| gemmi | 0.7.5 |
| tmtools | 0.3.0 |
| fair-esm | 2.0.0 |
| einops | 0.8.2 |
| timm | 1.0.30 |
| biotite | 1.0.1 |
| mashumaro | 3.23 |
| fairscale | 0.4.13 (installed --no-deps) |
| redis-server | 8.10.2 (conda-forge) |

Note: the env also contains HuggingFace `datasets` (inherited from `nomosis`),
whose top-level name collides with SimpleFold's own `datasets` package; the
SimpleFold backend handles this (see `molfun/backends/simplefold/imports.py`).
