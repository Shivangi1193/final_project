import cv2
import numpy as np

SIZE = 128
SEVERITIES = [0.0, 1 / 3, 2 / 3, 1.0]
SEVERITY_NAMES = ["0.00", "0.33", "0.66", "1.00"]


def load_labels(path="data/labels.csv"):
    data = np.genfromtxt(path, dtype=["U19", "f8", "U4"], names=["path", "prob", "type"])
    severity = np.array([int(np.argmin([abs(p - s) for s in SEVERITIES])) for p in data["prob"]])
    return list(data["path"]), severity, data["type"]


def crop_cell(img):
    _, mask = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w * h < 0.5 * img.size:
        return img
    return img[y:y + h, x:x + w]


def preprocess(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = crop_cell(img)
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def build_cache(out="data/preprocessed.npz"):
    paths, severity, types = load_labels()
    images = np.stack([preprocess(cv2.imread(f"data/{p}", cv2.IMREAD_GRAYSCALE)) for p in paths])
    np.savez_compressed(out, images=images, severity=severity, types=types)
    return images, severity, types


def elastic(img, rng, alpha=8.0, sigma=4.0):
    h, w = img.shape
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype("float32"), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype("float32"), (0, 0), sigma) * alpha
    xs, ys = np.meshgrid(np.arange(w, dtype="float32"), np.arange(h, dtype="float32"))
    return cv2.remap(img, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def augment(img, rng):
    if rng.random() < 0.5:
        img = np.fliplr(img)
    if rng.random() < 0.5:
        img = np.flipud(img)
    img = np.ascontiguousarray(img)
    if rng.random() < 0.5:
        img = elastic(img, rng)
    angle = rng.uniform(-5, 5)
    m = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)
    gain, bias = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
    return np.clip(img.astype("float32") * gain + bias, 0, 255).astype("uint8")


if __name__ == "__main__":
    images, severity, types = build_cache()
    print(images.shape, np.bincount(severity), dict(zip(*np.unique(types, return_counts=True))))
