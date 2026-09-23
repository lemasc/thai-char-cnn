"""`fit(cfg)` trains one run, or returns it from `runs/` if it already exists.

A run is identified by a hash of its resolved config (which includes the seed and, when augmenting,
the augmentation parameters), the split it was trained on, and `code_hash` -- a hash of the modules
that decide what training produces. Editing any of them invalidates every cached run, so a cached
result always belongs to the code on disk. A finished run -- one with `metrics.json` -- is otherwise
never retrained, the same idea as the audit's scan cache: adding an experiment to the notebook
trains only that experiment. `fit(..., force=True)` retrains anyway.

Each run writes `runs/<run_id>/`:
`config.json`, `history.csv`, `metrics.json`, `per_class.csv`, `confusion.npy`,
`val_predictions.csv`, `model.pt`.
"""

import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .augment import augment_params, build_train_transform, load_augment_config
from .data import SplitData, make_sampler
from .metrics import confusion, macro_f1, per_class
from .model import build_model
from .paths import CONFIGS, RUNS, SPLIT_DIR
from .split import stable_hash

DEFAULT_CONFIG = dict(
    model="small_cnn", width=32, dropout=0.3, use_geometry=False, img_size=32,
    augment=False, sampler="none",
    epochs=40, batch_size=256, lr=3e-3, weight_decay=5e-4, warmup_epochs=1, label_smoothing=0.0,
    patience=8, seed=42, amp=True,
)
# not part of the run identity: they change speed, not results
RUNTIME_KEYS = {"num_workers"}
# modules whose source decides what a run produces; split logic is covered by split_id instead
CODE_MODULES = ["augment", "data", "metrics", "model", "preprocess", "train"]


def code_hash() -> str:
    h = hashlib.sha256()
    for m in CODE_MODULES:
        h.update(m.encode() + b"\0" + (Path(__file__).parent / f"{m}.py").read_bytes())
    return h.hexdigest()[:12]


def current_split_id(split_dir: Path = SPLIT_DIR) -> str:
    return json.loads((split_dir / "run.json").read_text())["split_id"]


def resolve_config(cfg: dict) -> dict:
    """Defaults + overrides, with the augmentation actually used spelled out so it is hashed."""
    unknown = set(cfg) - set(DEFAULT_CONFIG) - RUNTIME_KEYS - {"name"}
    assert not unknown, f"unknown config keys {unknown}"
    out = DEFAULT_CONFIG | cfg
    out["augment_params"] = augment_params(load_augment_config(CONFIGS / "augment.json")) if out["augment"] else None
    out["code_hash"] = code_hash()
    return out


def run_id(cfg: dict, split_id: str) -> str:
    ident = {k: v for k, v in cfg.items() if k not in RUNTIME_KEYS | {"name"}}
    return stable_hash(dict(config=ident, split_id=split_id))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


@torch.no_grad()
def predict(model: nn.Module, data: SplitData, idx: np.ndarray, device: str, batch: int = 2048) -> torch.Tensor:
    """Softmax probabilities for rows `idx`, no augmentation."""
    model.eval()
    out = []
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        x, g = data.normalise(data.pixels[j], data.geo[j])
        with torch.autocast(device, enabled=device == "cuda"):
            out.append(model(x.to(device), g.to(device)).float().softmax(1).cpu())
    return torch.cat(out)


def fit(cfg: dict, split_id: str | None = None, runs_dir: Path = RUNS, split_dir: Path = SPLIT_DIR,
        data: SplitData | None = None, verbose: bool = True, force: bool = False) -> Path:
    cfg = resolve_config(cfg)
    split_id = split_id or current_split_id(split_dir)
    rid = run_id(cfg, split_id)
    run_dir = runs_dir / rid
    if (run_dir / "metrics.json").exists():
        if not force:
            if verbose:
                print(f"[{cfg.get('name', rid)}] cached -> {run_dir.relative_to(runs_dir.parent)}")
            return run_dir
        shutil.rmtree(run_dir)            # never leave old files next to a half-written new run
    assert split_id == current_split_id(split_dir), "can only train on the current split"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _seed_everything(cfg["seed"])
    data = data or SplitData(cfg["img_size"], split_dir)
    classes = data.classes
    n_classes = len(classes)
    validated = (classes.status == "validated").to_numpy()

    tr_idx, va_idx = data.idx["train"], data.idx["val"]
    transform = build_train_transform(load_augment_config(CONFIGS / "augment.json")) if cfg["augment"] else None
    sampler = make_sampler(data.y[tr_idx].numpy(), cfg["sampler"], cfg["seed"])
    workers = cfg.get("num_workers", min(8, os.cpu_count() or 1))
    loader = DataLoader(data.dataset("train", transform), batch_size=cfg["batch_size"],
                        shuffle=sampler is None, sampler=sampler, num_workers=workers,
                        persistent_workers=workers > 0, worker_init_fn=_worker_init, drop_last=True,
                        generator=torch.Generator().manual_seed(cfg["seed"]), pin_memory=device == "cuda")

    model = build_model(cfg, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * len(loader)
    warm = cfg["warmup_epochs"] * len(loader)

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = cfg["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    y_val = data.y[va_idx].numpy()

    history, best_f1, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    t_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        tot_loss = tot_ok = tot_n = 0
        for x, g, y in loader:
            x, g, y = x.to(device, non_blocking=True), g.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device, enabled=use_amp):
                logits = model(x, g)
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += float(loss.detach()) * len(y)
            tot_ok += int((logits.argmax(1) == y).sum())
            tot_n += len(y)
        prob = predict(model, data, va_idx, device)
        val_loss = float(nn.functional.nll_loss(prob.clamp_min(1e-12).log(), torch.from_numpy(y_val)))
        cm = confusion(y_val, prob.argmax(1).numpy(), n_classes)
        f1 = macro_f1(cm, validated)
        history.append(dict(epoch=epoch, train_loss=tot_loss / tot_n, train_acc=tot_ok / tot_n,
                            val_loss=val_loss, val_acc=float(np.trace(cm) / cm.sum()), val_macro_f1=f1,
                            lr=opt.param_groups[0]["lr"], seconds=round(time.time() - t0, 1)))
        if verbose:
            h = history[-1]
            print(f"[{cfg.get('name', rid)}] ep {epoch:3d}  loss {h['train_loss']:.3f}  "
                  f"val loss {h['val_loss']:.3f}  acc {h['val_acc']:.4f}  macro-F1 {f1:.4f}  ({h['seconds']}s)")
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    train_time = time.time() - t_start

    model.load_state_dict(best_state)
    prob = predict(model, data, va_idx, device)
    pred = prob.argmax(1).numpy()
    cm = confusion(y_val, pred, n_classes)
    pc = per_class(cm).merge(classes[["class_idx", "folder", "character", "status", "train_n", "val_n"]],
                             on="class_idx")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg | dict(split_id=split_id, run_id=rid), indent=2))
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pc.to_csv(run_dir / "per_class.csv", index=False)
    np.save(run_dir / "confusion.npy", cm)
    vi = data.images.iloc[va_idx]
    pd.DataFrame(dict(path=vi.path.to_numpy(), writer_id=vi.writer_id.to_numpy(), true_idx=y_val, pred_idx=pred,
                      p_pred=prob.max(1).values.numpy().round(4),
                      p_true=prob[torch.arange(len(y_val)), torch.from_numpy(y_val)].numpy().round(4))
                 ).to_csv(run_dir / "val_predictions.csv", index=False)
    torch.save(dict(state_dict=best_state, config=cfg, n_classes=n_classes,
                    pixel_mean=data.pixel_mean, pixel_std=data.pixel_std,
                    geo_mean=data.geo_mean, geo_std=data.geo_std), run_dir / "model.pt")
    metrics = dict(run_id=rid, name=cfg.get("name", ""), split_id=split_id, seed=cfg["seed"],
                   val_macro_f1=macro_f1(cm, validated), val_macro_f1_all=macro_f1(cm),
                   val_acc=float(np.trace(cm) / cm.sum()), best_epoch=best_epoch, epochs_run=len(history),
                   n_validated_classes=int(validated.sum()), n_train=len(tr_idx), n_val=len(va_idx),
                   n_params=sum(p.numel() for p in model.parameters()), device=device,
                   train_time_s=round(train_time, 1))
    # metrics.json last: its presence is what marks the run finished
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if verbose:
        print(f"[{cfg.get('name', rid)}] best epoch {best_epoch}: val macro-F1 {metrics['val_macro_f1']:.4f} "
              f"-> {run_dir.relative_to(runs_dir.parent)}")
    return run_dir


def load_runs(runs_dir: Path = RUNS) -> pd.DataFrame:
    """One row per finished run: its metrics plus its config (prefixed `cfg.`)."""
    rows = []
    for m in sorted(runs_dir.glob("*/metrics.json")):
        cfg = json.loads((m.parent / "config.json").read_text())
        rows.append(json.loads(m.read_text()) | {f"cfg.{k}": v for k, v in cfg.items()
                                                 if k not in ("split_id", "run_id", "augment_params")}
                    | {"code_hash": cfg.get("code_hash", "")})
    return pd.DataFrame(rows)
