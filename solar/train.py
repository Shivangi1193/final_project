import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from preprocess import SEVERITY_NAMES, augment, build_cache

SEED = 42
EPOCHS = 80
BATCH = 32

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
torch.set_num_threads(os.cpu_count())

class CellDataset(Dataset):
    def __init__(self, images, labels, train):
        self.images, self.labels, self.train = images, labels, train
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.train:
            img = augment(img, self.rng)
        x = torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0)
        return (x - 0.5) / 0.25, self.labels[i]


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class DefectCNN(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, classes))

    def forward(self, x):
        return self.head(self.features(x))


def predict(model, loader, loss_fn):
    model.eval()
    probs, total, n = [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            total += loss_fn(logits, y).item() * len(y)
            n += len(y)
            probs.append(torch.softmax(logits, 1).numpy())
    return total / n, np.concatenate(probs)


def main():
    torch.manual_seed(SEED)
    if os.path.exists("data/preprocessed.npz"):
        d = np.load("data/preprocessed.npz")
        images, severity, types = d["images"], d["severity"], d["types"]
    else:
        images, severity, types = build_cache()

    exploration = {
        "samples": len(severity),
        "severity_counts": dict(zip(SEVERITY_NAMES, np.bincount(severity).tolist())),
        "functional_vs_defective": {"functional": int((severity == 0).sum()), "defective": int((severity > 0).sum())},
        "type_counts": {t: int((types == t).sum()) for t in np.unique(types)},
    }
    print(json.dumps(exploration, indent=2))

    strata = np.char.add(severity.astype(str), types)
    idx = np.arange(len(severity))
    train_idx, rest = train_test_split(idx, test_size=0.3, stratify=strata, random_state=SEED)
    val_idx, test_idx = train_test_split(rest, test_size=0.5, stratify=strata[rest], random_state=SEED)
    np.savez("outputs/splits.npz", train=train_idx, val=val_idx, test=test_idx)

    def loader(ids, train):
        return DataLoader(CellDataset(images[ids], severity[ids], train), batch_size=BATCH, shuffle=train)

    train_dl, val_dl, test_dl = loader(train_idx, True), loader(val_idx, False), loader(test_idx, False)
    counts = np.bincount(severity[train_idx], minlength=4)
    weights = torch.tensor(np.sqrt(len(train_idx) / (4 * counts)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    model = DefectCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1, best_state = -1, None
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        val_loss, val_probs = predict(model, val_dl, loss_fn)
        val_f1 = f1_score(severity[val_idx], val_probs.argmax(1), average="macro")
        history["train_loss"].append(total / len(train_idx))
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        print(f"epoch {epoch + 1} train {history['train_loss'][-1]:.4f} val {val_loss:.4f} f1 {val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    seconds = time.time() - start
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), "models/defect_cnn.pt")

    _, val_probs = predict(model, val_dl, loss_fn)
    grid = np.arange(0.05, 0.96, 0.01)
    val_bin_f1 = [f1_score(severity[val_idx] > 0, 1 - val_probs[:, 0] >= th) for th in grid]
    threshold = float(grid[int(np.argmax(val_bin_f1))])

    _, probs = predict(model, test_dl, loss_fn)
    y_true, y_pred = severity[test_idx], probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    bin_true, bin_score = y_true > 0, 1 - probs[:, 0]
    bin_pred = bin_score >= threshold
    bp, br, bf, _ = precision_recall_fscore_support(bin_true, bin_pred, average="binary", zero_division=0)
    results = {
        "framing": "multi-class (4 defect severity levels: 0.00, 0.33, 0.66, 1.00)",
        "exploration": exploration,
        "parameters": sum(t.numel() for t in model.parameters()),
        "train_seconds": seconds,
        "best_val_macro_f1": best_f1,
        "test_severity": {
            "accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f,
            "roc_auc": roc_auc_score(y_true, probs, multi_class="ovr", average="macro"),
            "per_class_recall": dict(zip(SEVERITY_NAMES, (confusion_matrix(y_true, y_pred).diagonal()
                                                          / np.bincount(y_true, minlength=4)).tolist())),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        },
        "defect_threshold": threshold,
        "test_functional_vs_defective": {
            "accuracy": accuracy_score(bin_true, bin_pred), "precision": bp, "recall": br, "f1": bf,
            "roc_auc": roc_auc_score(bin_true, bin_score),
            "confusion_matrix": confusion_matrix(bin_true, bin_pred).tolist(),
        },
        "history": history,
    }
    with open("outputs/results.json", "w") as fh:
        json.dump(results, fh, indent=2)

    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/loss.png")
    plt.close()

    print(json.dumps({k: v for k, v in results.items() if k.startswith("test") or k in ("parameters", "train_seconds")}, indent=2))


if __name__ == "__main__":
    main()
