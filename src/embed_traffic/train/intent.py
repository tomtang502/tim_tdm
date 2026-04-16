"""Crossing Intent Classification training.

Trains a bi-LSTM over pedestrian trajectory features from PIE+JAAD, saves the
final state dict to `checkpoints/{run_name}/intent_lstm.pt` and logs metrics to
`checkpoints/{run_name}/training_log.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset

from embed_traffic.data.loader import UnifiedDataLoader
from embed_traffic.models.intent_model import (
    CrossingIntentLSTM,
    FEATURE_DIM,
    SEQ_LEN,
)
from embed_traffic.paths import checkpoint_dir

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IntentDataset(Dataset):
    """Dataset of pedestrian trajectory sequences with crossing intent labels."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = sequences  # (N, SEQ_LEN, FEATURE_DIM)
        self.labels = labels        # (N,) 0=not-crossing, 1=crossing

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.labels[idx]]),
        )


def extract_features_from_samples(
    samples, seq_len: int = SEQ_LEN, img_w: int = 1920, img_h: int = 1080
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sliding-window trajectory sequences with crossing-intent labels."""
    ped_tracks = defaultdict(list)
    for s in samples:
        if s.crossing_intent == -1:
            continue
        ped_tracks[s.ped_id].append(s)

    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for _, track in ped_tracks.items():
        track.sort(key=lambda s: s.frame_id)
        if len(track) < seq_len:
            continue

        features = []
        for s in track:
            cx = ((s.bbox[0] + s.bbox[2]) / 2) / img_w
            cy = ((s.bbox[1] + s.bbox[3]) / 2) / img_h
            w = abs(s.bbox[2] - s.bbox[0]) / img_w
            h = abs(s.bbox[3] - s.bbox[1]) / img_h
            area = w * h
            features.append([cx, cy, w, h, 0.0, 0.0, 0.0, area])

        for i in range(1, len(features)):
            dx = features[i][0] - features[i - 1][0]
            dy = features[i][1] - features[i - 1][1]
            speed = float(np.sqrt(dx ** 2 + dy ** 2))
            features[i][4] = dx
            features[i][5] = dy
            features[i][6] = speed

        arr = np.array(features)
        for i in range(0, len(track) - seq_len + 1, seq_len // 2):
            window = arr[i : i + seq_len]
            window_labels = [track[j].crossing_intent for j in range(i, i + seq_len)]
            label = 1 if sum(window_labels) > len(window_labels) / 2 else 0
            sequences.append(window)
            labels.append(label)

    return np.array(sequences), np.array(labels)


def train_intent_classifier(
    train_seqs, train_labels, val_seqs, val_labels,
    epochs: int = 30, batch_size: int = 64, lr: float = 0.001,
):
    train_ds = IntentDataset(train_seqs, train_labels)
    val_ds = IntentDataset(val_seqs, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = CrossingIntentLSTM().to(DEVICE)

    n_pos = (train_labels == 1).sum()
    n_neg = (train_labels == 0).sum()
    weight = torch.FloatTensor([1.0, n_neg / max(n_pos, 1)]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state = None
    logs: list[dict] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for seqs, lbls in train_loader:
            seqs = seqs.to(DEVICE)
            lbls = lbls.squeeze().to(DEVICE)
            optimizer.zero_grad()
            out = model(seqs)
            loss = criterion(out, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * seqs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_preds: list[int] = []
        val_true: list[int] = []
        with torch.no_grad():
            for seqs, lbls in val_loader:
                seqs = seqs.to(DEVICE)
                out = model(seqs)
                val_preds.extend(out.argmax(dim=1).cpu().numpy())
                val_true.extend(lbls.squeeze().numpy())
        val_acc = accuracy_score(val_true, val_preds)

        logs.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1}/{epochs}: loss={train_loss:.4f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    if best_state:
        model.load_state_dict(best_state)

    return model, logs, best_val_acc


def evaluate_classifier(model, test_seqs, test_labels):
    test_ds = IntentDataset(test_seqs, test_labels)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    model.eval()
    all_preds: list[int] = []
    all_true: list[int] = []
    with torch.no_grad():
        for seqs, lbls in test_loader:
            seqs = seqs.to(DEVICE)
            out = model(seqs)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_true.extend(lbls.squeeze().numpy())
    print("\n  Classification Report:")
    print(
        classification_report(
            all_true, all_preds, target_names=["not-crossing", "crossing"], digits=3
        )
    )
    acc = accuracy_score(all_true, all_preds)
    return acc, all_preds, all_true


def main(run_name: str, epochs: int = 30, batch_size: int = 64, lr: float = 0.001) -> None:
    print(f"=== Crossing Intent Classification: run '{run_name}' ===\n")

    loader = UnifiedDataLoader()

    print("--- Extracting features ---")
    t0 = time.time()
    pie_train = loader.get_pie_samples(split="train")
    pie_val = loader.get_pie_samples(split="val")
    pie_test = loader.get_pie_samples(split="test")
    jaad_train = loader.get_jaad_samples(split="train")
    jaad_val = loader.get_jaad_samples(split="val")
    jaad_test = loader.get_jaad_samples(split="test")

    train_seqs, train_labels = extract_features_from_samples(pie_train + jaad_train)
    val_seqs, val_labels = extract_features_from_samples(pie_val + jaad_val)
    test_seqs, test_labels = extract_features_from_samples(pie_test + jaad_test)

    print(
        f"  Train: {len(train_labels)} sequences "
        f"(crossing={int(sum(train_labels))}, "
        f"not={len(train_labels) - int(sum(train_labels))})"
    )
    print(f"  Val:   {len(val_labels)} sequences")
    print(f"  Test:  {len(test_labels)} sequences")
    print(f"  Feature extraction: {time.time()-t0:.1f}s")

    print("\n--- Training LSTM classifier ---")
    model, logs, best_val_acc = train_intent_classifier(
        train_seqs, train_labels, val_seqs, val_labels,
        epochs=epochs, batch_size=batch_size, lr=lr,
    )
    print(f"\n  Best val accuracy: {best_val_acc:.3f}")

    save_dir = checkpoint_dir(run_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_path = save_dir / "intent_lstm.pt"
    torch.save(model.state_dict(), weights_path)
    with open(save_dir / "training_log.json", "w") as f:
        json.dump(logs, f, indent=2)
    print(f"  Model saved to {weights_path}")

    print("\n--- Evaluating on test set ---")
    test_acc, _, _ = evaluate_classifier(model, test_seqs, test_labels)
    print(f"  Test accuracy: {test_acc:.3f}")


def cli() -> None:
    p = argparse.ArgumentParser(description="Train crossing intent classifier.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    args = p.parse_args()
    main(args.run_name, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    cli()
