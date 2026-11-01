import csv
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from data import (LABELS, Tokenizer, code_mixing_index, language_tags,
                  load_dataset, make_splits, normalize)

SEED = 42
MAX_LEN = {"word": 60, "char": 300, "subword": 80}
EMB_DIM = 100
MIXED_CMI = 20
GRID = [(64, 0.3), (64, 0.5), (128, 0.3), (128, 0.5)]

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
torch.set_num_threads(os.cpu_count())


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SentimentRNN(nn.Module):
    def __init__(self, cell, embeddings, hidden, dropout):
        super().__init__()
        self.cell = cell
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(embeddings.shape[1], hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, len(LABELS))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.drop(self.emb(x)), lengths, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        if self.cell == "lstm":
            h = h[0]
        return self.out(self.drop(torch.cat([h[-2], h[-1]], dim=1)))


def to_batches(seqs, labels, batch_size, shuffle):
    order = np.random.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        lengths = torch.tensor([len(seqs[j]) for j in idx])
        x = torch.zeros(len(idx), lengths.max(), dtype=torch.long)
        for row, j in enumerate(idx):
            x[row, :len(seqs[j])] = torch.tensor(seqs[j])
        yield x, lengths, torch.tensor(labels[idx])


def train_embeddings(tokenizer, token_lists):
    w2v = Word2Vec(token_lists, vector_size=EMB_DIM, window=5, min_count=1, sg=1,
                   epochs=20, seed=SEED, workers=1)
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(0, 0.1, (len(tokenizer.itos), EMB_DIM)).astype("float32")
    matrix[0] = 0
    for i, tok in enumerate(tokenizer.itos):
        if tok in w2v.wv:
            matrix[i] = w2v.wv[tok]
    return torch.tensor(matrix)


def evaluate(model, seqs, labels, loss_fn):
    model.eval()
    preds, total = [], 0.0
    with torch.no_grad():
        for x, lengths, y in to_batches(seqs, labels, 128, False):
            logits = model(x, lengths)
            total += loss_fn(logits, y).item() * len(y)
            preds.append(logits.argmax(1).numpy())
    return total / len(labels), np.concatenate(preds)


def fit(cell, embeddings, hidden, dropout, data, class_weights, max_epochs=25, patience=5):
    seed_all()
    model = SentimentRNN(cell, embeddings.clone(), hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    (xtr, ytr), (xva, yva) = data
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1, best_state, wait = -1, None, 0
    start = time.time()
    for _ in range(max_epochs):
        model.train()
        total = 0.0
        for x, lengths, y in to_batches(xtr, ytr, 32, True):
            opt.zero_grad()
            loss = loss_fn(model(x, lengths), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)
        val_loss, preds = evaluate(model, xva, yva, loss_fn)
        val_f1 = f1_score(yva, preds, average="macro")
        history["train_loss"].append(total / len(ytr))
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    seconds = time.time() - start
    model.load_state_dict(best_state)
    return model, history, best_f1, seconds


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f}


def main():
    seed_all()
    raw, labels = load_dataset("IIITH_Codemixed.txt")
    texts = [normalize(t) for t in raw]
    cmi = np.array([code_mixing_index(language_tags(t)) for t in texts])

    exploration = {
        "samples": len(texts),
        "class_counts": {LABELS[c]: int((labels == c).sum()) for c in range(3)},
        "cmi_mean": float(cmi.mean()),
        "cmi_median": float(np.median(cmi)),
        "monolingual_share": float((cmi == 0).mean()),
        "heavily_mixed_share": float((cmi >= MIXED_CMI).mean()),
    }
    print(json.dumps(exploration, indent=2))
    plt.hist(cmi, bins=30)
    plt.xlabel("Code-Mixing Index")
    plt.ylabel("Samples")
    plt.savefig("outputs/cmi_hist.png")
    plt.close()

    train_idx, val_idx, test_idx = make_splits(labels)
    train_texts = [texts[i] for i in train_idx]
    counts = np.bincount(labels[train_idx], minlength=3)
    class_weights = torch.tensor(len(train_idx) / (3 * counts), dtype=torch.float32)

    prepared, comparison = {}, {}
    for kind in ["word", "char", "subword"]:
        tok = Tokenizer(kind, train_texts, sp_prefix="models/subword")
        seqs = [tok.encode(t, MAX_LEN[kind]) for t in texts]
        emb = train_embeddings(tok, [tok.split(t) for t in train_texts])
        val_tokens = [t for i in val_idx for t in tok.split(texts[i])]
        unk_rate = sum(t not in tok.stoi for t in val_tokens) / len(val_tokens)
        data = (([seqs[i] for i in train_idx], labels[train_idx]),
                ([seqs[i] for i in val_idx], labels[val_idx]))
        _, _, val_f1, seconds = fit("lstm", emb, 64, 0.3, data, class_weights, max_epochs=10, patience=3)
        comparison[kind] = {
            "vocab_size": len(tok.itos),
            "avg_tokens": float(np.mean([len(tok.split(t)) for t in train_texts])),
            "val_unk_rate": unk_rate,
            "val_macro_f1": val_f1,
            "seconds": seconds,
        }
        prepared[kind] = (tok, seqs, emb)
        print(kind, comparison[kind])

    best_kind = max(comparison, key=lambda k: comparison[k]["val_macro_f1"])
    tok, seqs, emb = prepared[best_kind]
    print("selected tokenization:", best_kind)
    data = (([seqs[i] for i in train_idx], labels[train_idx]),
            ([seqs[i] for i in val_idx], labels[val_idx]))
    test_seqs, test_y = [seqs[i] for i in test_idx], labels[test_idx]

    results, preds = {}, {}
    for cell in ["lstm", "gru"]:
        tuning, best = [], None
        for hidden, dropout in GRID:
            model, history, val_f1, seconds = fit(cell, emb, hidden, dropout, data, class_weights)
            tuning.append({"hidden": hidden, "dropout": dropout, "val_macro_f1": val_f1})
            print(cell, hidden, dropout, round(val_f1, 4))
            if best is None or val_f1 > best[2]:
                best = (model, history, val_f1, seconds, hidden, dropout)
        model, history, val_f1, seconds, hidden, dropout = best
        _, pred = evaluate(model, test_seqs, test_y, nn.CrossEntropyLoss())
        preds[cell] = pred
        results[cell] = {
            "hidden": hidden,
            "dropout": dropout,
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_seconds": seconds,
            "epochs_run": len(history["train_loss"]),
            "val_macro_f1": val_f1,
            "test": metrics(test_y, pred),
            "confusion_matrix": confusion_matrix(test_y, pred).tolist(),
            "tuning": tuning,
            "history": history,
        }
        torch.save({"cell": cell, "hidden": hidden, "dropout": dropout, "tokenizer": best_kind,
                    "max_len": MAX_LEN[best_kind], "itos": tok.itos, "state_dict": model.state_dict()},
                   f"models/{cell}.pt")

        plt.plot(history["train_loss"], label="train")
        plt.plot(history["val_loss"], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(cell.upper())
        plt.legend()
        plt.savefig(f"outputs/{cell}_loss.png")
        plt.close()

    test_cmi = cmi[test_idx]
    buckets = {"mostly_one_language": test_cmi < MIXED_CMI, "heavily_mixed": test_cmi >= MIXED_CMI}
    error_analysis = {
        name: {"samples": int(mask.sum()),
               **{cell: metrics(test_y[mask], preds[cell][mask]) for cell in preds}}
        for name, mask in buckets.items()
    }

    with open("outputs/misclassified.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "cmi", "bucket", "true", "lstm", "gru"])
        for k, i in enumerate(test_idx):
            if preds["lstm"][k] != test_y[k] or preds["gru"][k] != test_y[k]:
                bucket = "heavily_mixed" if cmi[i] >= MIXED_CMI else "mostly_one_language"
                writer.writerow([raw[i], round(cmi[i], 1), bucket, LABELS[test_y[k]],
                                 LABELS[preds["lstm"][k]], LABELS[preds["gru"][k]]])

    report = {"exploration": exploration, "tokenization_comparison": comparison,
              "selected_tokenization": best_kind, "models": results, "error_analysis": error_analysis}
    with open("outputs/results.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nmodel  params   seconds  acc     prec    rec     f1")
    for cell, r in results.items():
        t = r["test"]
        print(f"{cell:6} {r['parameters']:<8} {r['train_seconds']:<8.1f} "
              f"{t['accuracy']:.4f}  {t['precision']:.4f}  {t['recall']:.4f}  {t['f1']:.4f}")
        print(np.array(r["confusion_matrix"]))
    print(json.dumps(error_analysis, indent=2))


if __name__ == "__main__":
    main()
