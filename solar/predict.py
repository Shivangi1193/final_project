import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from preprocess import SEVERITY_NAMES, load_labels, preprocess
from train import DefectCNN

model = DefectCNN()
model.load_state_dict(torch.load("models/defect_cnn.pt"))
model.eval()
with torch.no_grad():
    model(torch.zeros(1, 1, 128, 128))
threshold = json.load(open("outputs/results.json"))["defect_threshold"]
paths, severity, _ = load_labels()
truth = {os.path.basename(p): SEVERITY_NAMES[s] for p, s in zip(paths, severity)}

show = "--show" in sys.argv
files = [a for a in sys.argv[1:] if a != "--show"]
if not files:
    test_idx = np.random.default_rng().choice(np.load("outputs/splits.npz")["test"], 5, replace=False)
    files = [f"data/{paths[i]}" for i in test_idx]

for path in files:
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"{path}: cannot read image")
        continue
    img = preprocess(raw)
    x = (torch.from_numpy(img.astype("float32") / 255.0)[None, None] - 0.5) / 0.25
    start = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(x), 1)[0].numpy()
    ms = (time.perf_counter() - start) * 1000
    defect = 1 - probs[0]
    verdict = "DEFECTIVE" if defect >= threshold else "FUNCTIONAL"
    name = os.path.basename(path)
    print(f"\n{name}")
    print(f"  severity   : {SEVERITY_NAMES[probs.argmax()]}  [{'  '.join(f'{n}={p:.2f}' for n, p in zip(SEVERITY_NAMES, probs))}]")
    print(f"  verdict    : {verdict} (P(defective) {defect:.2f}, threshold {threshold:.2f})")
    if name in truth:
        print(f"  true label : {truth[name]}")
    print(f"  time       : {ms:.1f} ms")
    if show:
        title = f"{name}  pred {SEVERITY_NAMES[probs.argmax()]} {verdict}  true {truth.get(name, '?')}"
        cv2.imshow(title, np.hstack([cv2.resize(raw, (300, 300)), cv2.resize(img, (300, 300))]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
