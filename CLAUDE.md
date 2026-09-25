# CLAUDE.md

## Training run cache (`src/thai_char_cnn/train.py`)

A run lives at `runs/<run_id>/`, where `run_id = hash(resolved config, split_id, train_version)`.
`train_version` is `TRAIN_VERSION` plus `data.PREPROCESS_VERSION`. The source code is **not** hashed:
`code_hash` is recorded in `config.json` as provenance only. Retraining costs hours, so keep run ids
stable unless results really change.

When you change training-side code (`augment`, `data`, `metrics`, `model`, `preprocess`, `train`):

- **Changes what an existing config produces** (training loop, loss, optimiser, metrics, an existing
  model's layers, augmentation semantics, preprocessing) → change `TRAIN_VERSION` to a **new name**
  (e.g. `"2-cosine-fix"`), not +1, so two branches that both bump don't collide in the shared cache.
  Preprocessing changes bump `PREPROCESS_VERSION` instead (it also invalidates the tensor cache).
- **Adds something existing configs don't run** (a new model, a new sampler, a new option) → no bump.
- **Adds a config key to `DEFAULT_CONFIG`** → its default must reproduce the old behaviour, and the
  key goes in `IDENTITY_NEUTRAL` with that default (e.g. `{"pretrained": None}`), or every existing
  run id changes. A key that is optional rather than defaulted goes in `OPTIONAL_KEYS`.
- **Refactors, comments, logging** → no bump.
- Unsure → run `VERIFY=1` on `notebooks/04_train.ipynb`; it retrains the first run and fails if the
  result no longer matches the cache, which is the signal a bump was needed.

Never add `code_hash` or any other source hash back into the run id.

`fit(..., data=...)` with injected data (e.g. extra synthetic images) is not part of the run id: pass
a separate `runs_dir` for such arms, and leave `runs_dir` at its default for plain real-data arms so
they reuse cached runs.

## Frozen models (`models/`)

`models/<experiment>/` holds the checked-in submission weights, written only by `04_train` with
`FREEZE=1`. `05_submit` loads them without `runs/`, so they must not drift: don't refreeze unless the
selected model should change, and never hand-edit `model.pt` (its sha256 is checked on load).

## Worktrees

`runs/` in each git worktree is a symlink to the main checkout's `runs/`
(`ln -s /mnt/srv/home/whats/cnn/repo/runs <worktree>/runs`); create it for new worktrees. `paths.RUNS`
stays `ROOT / "runs"` — don't resolve the main checkout in code.
