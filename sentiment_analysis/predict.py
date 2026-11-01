import sys
import time

import sentencepiece as spm
import torch

from data import LABELS, code_mixing_index, language_tags, normalize
from train import SentimentRNN

sp = spm.SentencePieceProcessor(model_file="models/subword.model")
models = {}
for cell in ["lstm", "gru"]:
    ck = torch.load(f"models/{cell}.pt")
    model = SentimentRNN(cell, torch.zeros(len(ck["itos"]), 100), ck["hidden"], ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        model(torch.tensor([[1, 1, 1]]), torch.tensor([3]))
    models[cell] = (model, {t: i for i, t in enumerate(ck["itos"])}, ck["max_len"])


def predict(text):
    clean = normalize(text)
    tags = language_tags(clean)
    print(f"  normalized : {clean}")
    print(f"  languages  : {' '.join(f'{w}/{t}' for w, t in zip(clean.split(), tags))}")
    print(f"  CMI        : {code_mixing_index(tags):.1f} ({'heavily mixed' if code_mixing_index(tags) >= 20 else 'mostly one language'})")
    pieces = sp.encode(clean, out_type=str)
    for cell, (model, stoi, max_len) in models.items():
        ids = [stoi.get(p, 1) for p in pieces][:max_len] or [1]
        start = time.perf_counter()
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor([ids]), torch.tensor([len(ids)])), 1)[0]
        ms = (time.perf_counter() - start) * 1000
        scores = "  ".join(f"{LABELS[i]}={probs[i]:.2f}" for i in range(3))
        print(f"  {cell.upper():5}      : {LABELS[probs.argmax()].upper():9} [{scores}]  {ms:.1f} ms")


if len(sys.argv) > 1:
    predict(" ".join(sys.argv[1:]))
else:
    print("Type a Hinglish sentence (empty line to quit)")
    while True:
        text = input("\n> ").strip()
        if not text:
            break
        predict(text)
