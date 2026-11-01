# Solar Panel Defect Detection using a CNN on Electroluminescence Images

Custom Convolutional Neural Network in PyTorch that classifies solar cell EL images by defect severity.

## Dataset

ELPV dataset (Buerhop-Lutz et al., 2018), https://github.com/zae-bayern/elpv-dataset. 2,624 grayscale 300x300 EL images of solar cells from 44 modules. `data/labels.csv` is included. Fetch the images once:

```
git clone --depth 1 https://github.com/zae-bayern/elpv-dataset.git
cp -r elpv-dataset/src/elpv_dataset/data/images data/
```

| Defect probability | Count |
| ------------------ | ----- |
| 0.00               | 1508  |
| 0.33               | 295   |
| 0.66               | 106   |
| 1.00               | 715   |

Functional (0.00): 1508, defective (> 0): 1116. Monocrystalline: 1074, polycrystalline: 1550.

## Run

```
pip install -r requirements.txt
python preprocess.py
python train.py
```

## Live demo

Uses the trained model in `models/`, no retraining needed (images must be in `data/images/`).

```
python predict.py
python predict.py data/images/cell0001.png data/images/cell0004.png --show
```

With no arguments it picks 5 random test-set images. For each image it shows the predicted severity with probabilities, the functional/defective verdict, the true label and inference time. `--show` also opens a window with the raw and preprocessed image (press any key for the next one). Any EL image path works.

`preprocess.py` builds `data/preprocessed.npz` (train.py also builds it if missing). Outputs go to `models/` and `outputs/`.

## Files

| File                     | Purpose                                                                                 |
| ------------------------ | --------------------------------------------------------------------------------------- |
| `preprocess.py`        | Label loading, contrast normalization, denoising, cell cropping, resizing, augmentation |
| `train.py`             | Exploration, splits, CNN definition, training, evaluation                               |
| `models/defect_cnn.pt` | Trained CNN weights                                                                     |
| `outputs/results.json` | Metrics, confusion matrices, threshold, loss history                                    |
| `outputs/loss.png`     | Training/validation loss curves                                                         |
| `outputs/splits.npz`   | Train/validation/test indices                                                           |

## Task framing

Multi-class classification with 4 classes, one per defect severity (0.00, 0.33, 0.66, 1.00). The same model is also reported as binary functional vs defective by using P(defective) = 1 - P(0.00).

## Preprocessing decisions

1. Min-max intensity normalization to 0 to 255 to correct inconsistent illumination between images.
2. CLAHE (clip 2.0, 8x8 tiles) for local contrast normalization, which makes cracks and dark regions stand out.
3. Gaussian 3x3 denoising. Non-local means (h = 7) was tried first but smoothed away thin cracks and gave worse validation results.
4. Cell-boundary cropping: Otsu threshold, bounding box of the largest bright region, kept only if it covers at least half the image (otherwise the image is left uncropped).
5. Resize to 128x128 (INTER_AREA). 192x192 was tested and gave no gain on CPU for 2.25x the cost.
6. Model input scaled to 0 to 1, then (x - 0.5) / 0.25.

## Augmentation decisions (training only)

Chosen for crack-like defects instead of generic augmentation:

1. Horizontal and vertical flips (each p = 0.5). Cracks have no preferred direction, and flips keep busbars horizontal.
2. Elastic distortion (p = 0.5, alpha 8, sigma 4) so cracks bend slightly differently each time, as real micro-cracks do.
3. Small rotation, uniform -5 to 5 degrees, for slight cell misalignment.
4. Brightness/contrast jitter: gain 0.8 to 1.2, bias -20 to 20, for illumination differences.

90 degree rotations were tried first and dropped, since they turn busbars vertical, which never happens in the data, and slowed learning.

## Split

Stratified 70 / 15 / 15 train / validation / test (seed 42), stratified on severity and cell type together. The same split is used for training, tuning, threshold selection and testing.

## Model

4 convolution blocks (16, 32, 64, 128 channels). Each block: Conv 3x3, BatchNorm, ReLU, Conv 3x3, BatchNorm, ReLU, MaxPool 2x2. Then global average pooling, dropout 0.4, linear to 4 classes. 294,228 parameters.

Training: AdamW (lr 0.001, weight decay 0.0001), cosine learning rate schedule, batch 32, 80 epochs. The checkpoint with the best validation macro F1 is kept (epoch 54). Training time 1768 s on CPU.

## Class imbalance handling

1. Class-weighted cross entropy with weights sqrt(N / (4 x class count)). Full inverse-frequency weights were tried first and made training unstable, since the 0.66 class has only 74 training images and got a weight of about 6.
2. Stratified splits so every class is present in validation and test.
3. Model selection by macro F1, which counts the minority classes equally.
4. For functional vs defective, the decision threshold on P(defective) is chosen on the validation set to maximize F1 (selected threshold 0.40 instead of the default argmax). A recall-weighted F2 threshold was also tried and gave 0.08, which reached 0.99 recall but flagged almost every cell, so F1 was kept.

## Results (test set, 394 images)

Severity (4 classes, macro averaged):

| Accuracy | Precision | Recall | F1     | ROC-AUC (one vs rest) |
| -------- | --------- | ------ | ------ | --------------------- |
| 0.7234   | 0.6153    | 0.4826 | 0.5152 | 0.7981                |

Per-class recall: 0.00: 0.942, 0.33: 0.273, 0.66: 0.188, 1.00: 0.528.

```
Confusion matrix (rows true, columns predicted: 0.00, 0.33, 0.66, 1.00)
[213   5   4   4]
[ 27  12   5   0]
[ 13   0   3   0]
[ 46   3   2  57]
```

Functional vs defective (threshold 0.40):

| Accuracy | Precision | Recall | F1     | ROC-AUC |
| -------- | --------- | ------ | ------ | ------- |
| 0.7208   | 0.6629    | 0.7024 | 0.6821 | 0.8069  |

```
Confusion matrix (rows true, columns predicted: functional, defective)
[166  60]
[ 50 118]
```

## Observations

Validation loss follows training loss down until about epoch 60 and then levels off around 0.93 while training loss keeps slowly dropping, so overfitting is mild. The early spikes in validation loss (epochs 18 to 30) come from the high learning rate early in the cosine schedule. The 0.33 and 0.66 classes are the hardest: they are small and their labels are uncertain by design (the annotators were unsure), so most of their errors are predicted as functional. Recall for defective cells overall is 0.70.
