"""Writer-safe, cached transfer-learning experiments for notebook 04."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm.auto import tqdm

from .metrics import confusion, macro_f1, per_class
from .quality import preprocess_candidate


def secure_split(images: pd.DataFrame, manifest: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Remove training rows from validation groups and exact-hash twins.

    Existing split puts tiny-class images of validation filename groups in train.
    Dropping those train rows preserves the validation set and group purity.
    """
    selected = images[images.included & images.is_split_canonical & images.split.isin(["train", "val"])].copy()
    if manifest is not None:
        selected = selected.merge(manifest[["path", "near_dup_group_id"]], on="path", validate="one_to_one")
    val = selected[selected.split == "val"]
    train = selected[selected.split == "train"]
    writer_conflict = train.writer_id.isin(set(val.writer_id))
    train = train[~writer_conflict]
    hash_conflict = train.sha_group_id.isin(set(val.sha_group_id))
    train = train[~hash_conflict]
    near_removed = 0
    if manifest is not None:
        near_conflict = train.near_dup_group_id.isin(set(val.near_dup_group_id))
        near_removed = int(near_conflict.sum())
        train = train[~near_conflict]
    out = pd.concat([train, val], ignore_index=True)
    assert not set(train.writer_id) & set(val.writer_id)
    assert not set(train.sha_group_id) & set(val.sha_group_id)
    if manifest is not None:
        assert not set(train.near_dup_group_id) & set(val.near_dup_group_id)
    return out, dict(writer_conflict_removed=int(writer_conflict.sum()),
                     exact_twin_removed=int(hash_conflict.sum()), near_twin_removed=near_removed,
                     train=len(train), val=len(val),
                     train_writers=train.writer_id.nunique(), val_writers=val.writer_id.nunique())


@dataclass(frozen=True)
class Experiment:
    name: str
    phase: str
    preprocess: str = "RAW"
    augmentation: str = "NONE"
    model: str = "efficientnet_v2_s"
    sampler: str = "standard"
    loss: str = "ce"
    epochs: int = 5
    lr: float = 3e-4
    batch_size: int = 32
    image_size: int = 224
    seed: int = 42
    two_stage: bool = False
    mixup: bool = False


class GlyphDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, raw: Path, spec: Experiment, training: bool):
        self.rows = frame.reset_index(drop=True)
        self.raw = raw
        self.spec = spec
        self.training = training
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.raw / row.path) as image:
            arr = preprocess_candidate(image, self.spec.image_size, self.spec.preprocess)
        image = Image.fromarray(arr).convert("RGB")
        if self.training and self.spec.augmentation != "NONE":
            levels = {"LIGHT": (4, .02), "MODERATE": (8, .04), "STRONG": (12, .06)}
            level = self.spec.augmentation
            if level == "RARE_CLASS_ADAPTIVE":
                level = "MODERATE" if row.rare_class else "LIGHT"
            angle, shift = levels[level]
            image = transforms.RandomAffine(degrees=angle, translate=(shift, shift),
                                            scale=(.94, 1.06), fill=255)(image)
        x = self.to_tensor(image)
        x = transforms.Normalize((.485, .456, .406), (.229, .224, .225))(x)
        return x, int(row.class_idx), row.path


def build_backbone(name: str, n_classes: int) -> nn.Module:
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    else:
        raise ValueError(f"unknown model {name!r}")
    return model


def _set_trainable(model: nn.Module, name: str, head_only: bool):
    for p in model.parameters():
        p.requires_grad = not head_only
    if head_only:
        head = model.fc if name == "resnet50" else model.classifier
        for p in head.parameters():
            p.requires_grad = True


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(y: np.ndarray, pred: np.ndarray, n_classes: int, rare: set[int]) -> dict:
    cm = confusion(y, pred, n_classes)
    pc = per_class(cm)
    rare_pc = pc[pc.class_idx.isin(rare) & (pc.support > 0)]
    return dict(val_accuracy=float((y == pred).mean()), macro_f1=macro_f1(cm),
                weighted_f1=float(np.nansum(pc.f1 * pc.support) / pc.support.sum()),
                rare_recall=float(rare_pc.recall.mean()) if len(rare_pc) else float("nan"))


def _id(spec: Experiment, frame: pd.DataFrame) -> str:
    # Hash exact membership and implementation, not just the phase name.
    source = Path(__file__).read_bytes() + (Path(__file__).parent / "quality.py").read_bytes()
    members = frame[["path", "class_idx", "split"]].sort_values("path").to_csv(index=False).encode()
    return hashlib.sha256(json.dumps(asdict(spec), sort_keys=True).encode() + members + source).hexdigest()[:16]


def run_experiment(spec: Experiment, frame: pd.DataFrame, raw: Path, out_root: Path,
                   n_classes: int, rare: set[int], workers: int = 0) -> tuple[Path, dict]:
    """Train one controlled experiment, or load a completed run by content hash."""
    run_dir = out_root / _id(spec, frame)
    result_file = run_dir / "metrics.json"
    if result_file.exists():
        result = json.loads(result_file.read_text())
        print(f"{spec.name}: cached, accuracy {result['val_accuracy']:.4f}, macro F1 {result['macro_f1']:.4f}")
        return run_dir, result
    _seed(spec.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    train = frame[frame.split == "train"].copy()
    val = frame[frame.split == "val"].copy()
    train["rare_class"] = train.class_idx.isin(rare)
    val["rare_class"] = val.class_idx.isin(rare)
    train_ds = GlyphDataset(train, raw, spec, training=True)
    val_ds = GlyphDataset(val, raw, spec, training=False)
    counts = train.class_idx.value_counts()
    sampler = None
    if spec.sampler == "weighted":
        weights = train.class_idx.map(lambda k: 1.0 / counts[k]).to_numpy(float)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(train), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=workers, pin_memory=device == "cuda")
    val_loader = DataLoader(val_ds, batch_size=spec.batch_size, shuffle=False,
                            num_workers=workers, pin_memory=device == "cuda")
    model = build_backbone(spec.model, n_classes).to(device)
    _set_trainable(model, spec.model, head_only=spec.two_stage)
    weights = None
    if spec.loss == "weighted_ce":
        weights = torch.tensor([1 / np.sqrt(counts.get(i, 1)) for i in range(n_classes)],
                               dtype=torch.float32, device=device)
        weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=.1 if spec.loss == "label_smoothing" else 0.)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=spec.lr)
    history = []
    best = -1.
    best_state = None
    best_pred = None
    best_prob = None
    t0 = time.monotonic()
    for epoch in range(1, spec.epochs + 1):
        if spec.two_stage and epoch == max(2, spec.epochs // 3 + 1):
            _set_trainable(model, spec.model, head_only=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=spec.lr * .1)
        model.train()
        seen = correct = 0
        loss_sum = 0.
        for x, y, _ in tqdm(train_loader, desc=f"{spec.name} {epoch}/{spec.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            if spec.mixup:
                mix = float(np.random.beta(.2, .2))
                perm = torch.randperm(len(y), device=device)
                logits = model(mix * x + (1 - mix) * x[perm])
                loss = mix * criterion(logits, y) + (1 - mix) * criterion(logits, y[perm])
            else:
                logits = model(x)
                loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            seen += len(y)
        model.eval()
        val_loss = 0.
        y_true, y_pred, probability_batches = [], [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                logits = model(x.to(device))
                val_loss += float(criterion(logits, y.to(device))) * len(y)
                y_true.extend(y.numpy().tolist())
                y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
                probability_batches.append(logits.softmax(1).cpu().numpy())
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        metrics = _metrics(y_true, y_pred, n_classes, rare)
        row = dict(epoch=epoch, train_loss=loss_sum / seen, train_accuracy=correct / seen,
                   val_loss=val_loss / len(val), lr=optimizer.param_groups[0]["lr"], **metrics)
        history.append(row)
        saved = metrics["macro_f1"] > best
        if saved:
            best = metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_pred = y_pred.copy()
            best_prob = np.concatenate(probability_batches)
        print(f"{spec.name} epoch {epoch:02d}/{spec.epochs} | train loss {row['train_loss']:.4f}, acc {row['train_accuracy']:.2%} | "
              f"val loss {row['val_loss']:.4f}, acc {row['val_accuracy']:.2%}, macro F1 {row['macro_f1']:.4f} | "
              f"lr {row['lr']:.6f} | checkpoint {'SAVED' if saved else '-'}")
    best_row = max(history, key=lambda r: r["macro_f1"])
    result = dict(asdict(spec), **{k: v for k, v in best_row.items() if k != "epoch"},
                  best_epoch=best_row["epoch"], training_seconds=round(time.monotonic() - t0, 1),
                  train_images=len(train), val_images=len(val), run_id=run_dir.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best_state, spec=asdict(spec), n_classes=n_classes), run_dir / "model.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    pd.DataFrame(dict(path=val.path.to_numpy(), true_idx=y_true, pred_idx=best_pred,
                      writer_id=val.writer_id.to_numpy(), p_pred=best_prob.max(1),
                      p_true=best_prob[np.arange(len(y_true)), y_true])).to_csv(
                          run_dir / "val_predictions.csv", index=False)
    result_file.write_text(json.dumps(result, indent=2))
    print(f"EXPERIMENT {spec.name} COMPLETE | accuracy {result['val_accuracy']:.4f} | macro F1 {result['macro_f1']:.4f} | "
          f"rare recall {result['rare_recall']:.4f} | {result['training_seconds']:.1f}s")
    return run_dir, result
