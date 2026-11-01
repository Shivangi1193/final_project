import re
from collections import Counter

import numpy as np
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from wordfreq import top_n_list

LABELS = ["negative", "neutral", "positive"]

SPELLING = {
    "nhi": "nahi", "nai": "nahi", "nahin": "nahi", "nhin": "nahi", "nahii": "nahi", "nahee": "nahi",
    "h": "hai", "hy": "hai", "haii": "hai", "hain": "hai", "hn": "hai",
    "mai": "main", "mein": "me", "mei": "me", "mae": "me",
    "kyaa": "kya", "kia": "kya",
    "kyu": "kyun", "kyon": "kyun", "kyo": "kyun",
    "ap": "aap", "apko": "aapko", "apke": "aapke", "apki": "aapki",
    "bhut": "bahut", "bahot": "bahut", "bohot": "bahut", "bhot": "bahut", "boht": "bahut", "bhaut": "bahut",
    "acha": "accha", "achha": "accha", "acchha": "accha", "achaa": "accha",
    "jee": "ji", "jii": "ji",
    "yr": "yaar", "yar": "yaar",
    "thaa": "tha", "sb": "sab",
    "hm": "hum", "ham": "hum", "humm": "hum", "tm": "tum",
    "bi": "bhi", "aor": "aur", "pr": "par",
    "u": "you", "ur": "your", "urs": "yours", "plz": "please", "pls": "please",
    "thx": "thanks", "thnx": "thanks", "tnx": "thanks", "gud": "good", "gd": "good",
    "vedio": "video", "vdo": "video", "luv": "love", "soo": "so",
}

HINDI_WORDS = set(
    "hai hain ho tha thi the hoga hogi ka ki ke ko me main se par pe ne na nahi mat bhi to hi aur ya "
    "ye yeh wo woh vo is us iss uss inka unka inki unki inke unke inko unko isko usko yahan wahan "
    "kya kyun kaise kab kahan kaun koi kuch sab sabhi bahut bohut zyada kam accha achha bura "
    "aap aapka aapki aapke aapko tum tumhara tera teri mera meri mere hum humara hamara hamare "
    "ji bhai bhaiya didi beta beti sahab saheb yaar dost jai jay hind bharat desh log logo logon "
    "karo karna kar kare karte karta karti kiya kiye raha rahi rahe gaya gayi gaye diya dena de do "
    "hona hota hoti hote ho gaya jana jao jata jati aana aaya aayi aaye dekho dekh dekha suna sun "
    "bolo bol bola kaha kehna lagta laga lagi achchha sundar pyar pyaar dil zindagi duniya "
    "abhi kabhi phir fir sirf bas lekin magar kyunki agar jab tab sath saath bina liye liya "
    "ek do teen char panch pehle baad upar niche andar bahar aaj kal raat din saal".split()
)
ENGLISH_WORDS = {w for w in top_n_list("en", 20000) if w.isalpha()} - HINDI_WORDS
UNIVERSAL = {"url", "user", "emoji", "num", "!", "?"}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️]+")
REPEAT_RE = re.compile(r"(.)\1{2,}")
TOKEN_RE = re.compile(r"[a-z]+|\d+|[!?]")


def normalize(text):
    text = text.lower()
    text = URL_RE.sub(" url ", text)
    text = MENTION_RE.sub(" user ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = EMOJI_RE.sub(" emoji ", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    words = []
    for tok in TOKEN_RE.findall(text):
        if tok.isdigit():
            tok = "num"
        words.append(SPELLING.get(tok, tok))
    return " ".join(words)


def language_tags(text):
    tags = []
    for w in text.split():
        if w in UNIVERSAL:
            tags.append("univ")
        elif w in HINDI_WORDS:
            tags.append("hi")
        elif w in ENGLISH_WORDS:
            tags.append("en")
        else:
            tags.append("hi")
    return tags


def code_mixing_index(tags):
    lang = [t for t in tags if t != "univ"]
    if not lang:
        return 0.0
    top = Counter(lang).most_common(1)[0][1]
    return 100 * (1 - top / len(lang))


def load_dataset(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[2] not in ("0", "1", "2"):
                continue
            texts.append(parts[1])
            labels.append(int(parts[2]))
    return texts, np.array(labels)


def make_splits(labels, seed=42):
    idx = np.arange(len(labels))
    train, rest = train_test_split(idx, test_size=0.3, stratify=labels, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=labels[rest], random_state=seed)
    return train, val, test


class Tokenizer:
    def __init__(self, kind, train_texts, sp_prefix="subword", sp_vocab=2000, min_freq=2):
        self.kind = kind
        if kind == "subword":
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(train_texts), model_prefix=sp_prefix, vocab_size=sp_vocab,
                model_type="unigram", character_coverage=1.0, minloglevel=2,
            )
            self.sp = spm.SentencePieceProcessor(model_file=sp_prefix + ".model")
        counts = Counter(t for text in train_texts for t in self.split(text))
        self.itos = ["<pad>", "<unk>"] + sorted(t for t, c in counts.items() if c >= min_freq)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def split(self, text):
        if self.kind == "word":
            return text.split()
        if self.kind == "char":
            return list(text)
        return self.sp.encode(text, out_type=str)

    def encode(self, text, max_len):
        ids = [self.stoi.get(t, 1) for t in self.split(text)][:max_len]
        return ids or [1]
