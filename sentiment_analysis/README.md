# Hinglish Code-Mixed Sentiment Classification (LSTM vs GRU)

Sentiment classification (negative / neutral / positive) of Hindi-English code-mixed text using LSTM and GRU networks in PyTorch.

## Dataset

IIITH Hindi-English Code-Mixed Sentiment Dataset (Prabhu, Joshi, Shrivastava and Varma, COLING 2016), file `IIITH_Codemixed.txt` from https://github.com/drimpossible/Sub-word-LSTM.
3,879 Facebook comments. Label column: 0 = negative, 1 = neutral, 2 = positive.

| Class | Count |
| --- | --- |
| negative | 570 |
| neutral | 1957 |
| positive | 1352 |

## Run

```
pip install -r requirements.txt
python train.py
```

Outputs go to `models/` (trained models) and `outputs/` (metrics, plots, error analysis).

## Live demo

Uses the trained models in `models/`, no retraining needed.

```
python predict.py
```

Type any Hinglish sentence and press Enter. It shows the normalized text, word-level language tags, the Code-Mixing Index, and the LSTM and GRU predictions with probabilities and inference time. An empty line quits. For a single sentence: `python predict.py "bahut accha video hai sir"`.

## Files

| File | Purpose |
| --- | --- |
| `data.py` | Loading, text normalization, word-level language tagging, Code-Mixing Index, splits, word/char/subword tokenizers |
| `train.py` | Exploration, tokenization comparison, embedding training, LSTM and GRU training, tuning, evaluation, error analysis |
| `models/lstm.pt`, `models/gru.pt` | Trained models (state dict, vocabulary, hyperparameters) |
| `models/subword.model` | SentencePiece model used by both networks |
| `outputs/results.json` | All metrics, tuning runs, loss histories |
| `outputs/*_loss.png` | Training/validation loss curves |
| `outputs/cmi_hist.png` | Code-mixing distribution |
| `outputs/misclassified.csv` | Test samples misclassified by either model |

## Preprocessing decisions

1. Lowercase everything.
2. URLs become `url`, @mentions become `user`, hashtags keep the word without `#`, emojis become `emoji`, numbers become `num`.
3. Characters repeated 3 or more times are collapsed to 2 (`nahiiii` to `nahii`), then a spelling map normalizes common transliteration variants (`nhi`, `nai`, `nahin` to `nahi`; `bhut`, `bohot` to `bahut`; `acha`, `achha` to `accha`; chat shorthand like `u`, `plz`, `gud`).
4. Only letters, digits and `!` `?` are kept as tokens.
5. Word-level language tags: a token is `univ` for placeholders and punctuation, `hi` if in a hand-built list of common romanized Hindi words, `en` if in the top 20,000 English words (wordfreq), otherwise `hi` (unknown romanized words are almost always Hindi).
6. Code-Mixing Index (Das and Gamback): CMI = 100 x (1 - dominant language tokens / language tokens). A sample is "heavily mixed" when CMI >= 20, otherwise "mostly one language".

Exploration: mean CMI 23.1, median 25.0, 15.7% of samples monolingual, 56.1% heavily mixed.

## Split

Stratified 70 / 15 / 15 train / validation / test with seed 42. Both models use the same split, the same vocabulary, the same token sequences and the same initial embeddings.

## Tokenization comparison

Each tokenizer is fitted on the training split only. A reference LSTM (hidden 64, dropout 0.3, up to 10 epochs) is trained for each and scored on validation macro F1.

| Tokenization | Vocab | Avg tokens | Val unknown rate | Val macro F1 |
| --- | --- | --- | --- | --- |
| word (min freq 2) | 2129 | 11.7 | 17.5% | 0.6546 |
| character | 31 | 56.4 | 0.0% | 0.5097 |
| subword (SentencePiece unigram, 2000) | 1995 | 14.7 | 0.0% | 0.6549 |

Subword was selected. It matches word-level accuracy while having no unknown tokens, which matters for inconsistent transliteration where unseen spellings share pieces with known ones.

## Embeddings

Skip-gram Word2Vec (100 dimensions, window 5, 20 epochs) trained from scratch on the tokenized training split. No pretrained English embeddings are used. The vectors initialize the embedding layer and are fine-tuned during training.

## Model design

Both networks: embedding (100) then a 1-layer bidirectional recurrent layer, final forward and backward hidden states concatenated, dropout, linear layer to 3 classes.

Training: Adam (lr 0.002), batch 32, class-weighted cross entropy (inverse class frequency, since negative is the minority), gradient clipping 1.0, up to 25 epochs, early stopping with patience 5 on validation macro F1. The checkpoint with the best validation macro F1 is kept.

Tuning grid for each network: hidden size {64, 128} x dropout {0.3, 0.5}, selected by validation macro F1.

| Hidden | Dropout | LSTM val F1 | GRU val F1 |
| --- | --- | --- | --- |
| 64 | 0.3 | 0.6549 | 0.6567 |
| 64 | 0.5 | **0.6635** | 0.6430 |
| 128 | 0.3 | 0.6349 | 0.6394 |
| 128 | 0.5 | 0.6494 | **0.6609** |

## Results (test set, macro averaged)

| Model | Hidden | Dropout | Parameters | Train time (s, CPU) | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM | 64 | 0.5 | 284,879 | 36.9 | 0.6787 | 0.6522 | 0.6395 | 0.6405 |
| GRU | 128 | 0.5 | 376,911 | 40.2 | 0.6959 | 0.6606 | 0.6601 | 0.6603 |

Confusion matrices (rows true, columns predicted: negative, neutral, positive):

```
LSTM            GRU
[ 49  27  10]   [ 47  23  16]
[ 25 235  33]   [ 25 218  50]
[ 17  75 111]   [ 13  50 140]
```

Parameter efficiency: at equal hidden size a GRU has 3 gates against the LSTM's 4, so it uses about 25% fewer recurrent parameters. Here the tuned GRU ended up with a larger hidden size, so it has more parameters overall and took slightly longer, for about 2 points more F1.

## Loss curves

Validation loss bottoms out around epoch 2 to 3 while training loss keeps falling, so both models start to overfit early on this small dataset. Validation macro F1 keeps fluctuating around 0.64 to 0.66 after that, which is why the checkpoint is chosen by validation F1 and early stopping ends training a few epochs later.

## Error analysis (test set)

| Bucket | Samples | LSTM acc | LSTM F1 | GRU acc | GRU F1 |
| --- | --- | --- | --- | --- | --- |
| Mostly one language (CMI < 20) | 262 | 0.6336 | 0.5675 | 0.6450 | 0.6008 |
| Heavily mixed (CMI >= 20) | 320 | 0.7156 | 0.6806 | 0.7375 | 0.6919 |

Both models do better on heavily mixed text than on mostly one-language text. 251 of the 262 "mostly one language" test samples are Hindi-dominant, so this bucket is mostly romanized Hindi with inconsistent spellings, while heavily mixed samples contain more English sentiment words (love, great, good) that the models pick up easily. Average length is the same in both buckets (about 11.6 tokens), so length does not explain the gap. The most common errors are neutral versus positive confusions. Misclassified test samples are listed in `outputs/misclassified.csv`.
