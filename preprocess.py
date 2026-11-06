"""
preprocess.py  -  Step 1: Data preparation
Loads the CFPB Consumer Complaints dataset, analyses it, cleans the text,
tokenizes, pads, splits into train/val/test and saves everything to disk.
Run:  python preprocess.py
"""
import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # draw graphs without opening a window
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- Settings (all in one place) ----------------
DATASET_PATH = "dataset/complaints.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
NUM_CLASSES = 6          # keep the 6 most frequent complaint categories
MAX_PER_CLASS = 4000     # balance classes and keep training fast on a laptop
SAMPLE_FRACTION = 1    # the full file is huge, so we read 20% of the complaints
VOCAB_SIZE = 10000       # only the 10,000 most frequent words are kept
MAX_LEN = 200            # every complaint is cut/padded to 200 tokens
SEED = 42

TEXT_CANDIDATES = ["Consumer complaint narrative", "consumer_complaint_narrative"]
LABEL_CANDIDATES = ["Product", "product"]


def find_column(columns, candidates):
    """Return the first candidate name that exists in the dataset."""
    for name in candidates:
        if name in columns:
            return name
    return None


def clean_text(text):
    """Lowercase, remove redacted 'XXXX', remove numbers/symbols, fix spaces."""
    text = str(text).lower()
    text = re.sub(r"x{2,}", " ", text)          # CFPB hides private data as XXXX
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters
    text = re.sub(r"\s+", " ", text).strip()    # remove extra spaces
    return text


def load_dataset():
    """Read the big CSV in chunks so it does not fill the computer's memory."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            "Dataset not found. Download 'complaints.csv' from Kaggle "
            "(CFPB Consumer Complaints) and place it in the dataset/ folder.")

    header = pd.read_csv(DATASET_PATH, nrows=0)
    all_columns = list(header.columns)
    text_col = find_column(all_columns, TEXT_CANDIDATES)
    label_col = find_column(all_columns, LABEL_CANDIDATES)
    if text_col is None or label_col is None:
        raise ValueError("Could not find text/label columns. Columns are: " + str(all_columns))

    total_records = 0
    records_with_text = 0
    kept_chunks = []
    for chunk in pd.read_csv(DATASET_PATH, usecols=[text_col, label_col], chunksize=100000):
        total_records += len(chunk)
        chunk = chunk.dropna()                    # missing narrative or product -> remove
        records_with_text += len(chunk)
        kept_chunks.append(chunk.sample(frac=SAMPLE_FRACTION, random_state=SEED))

    df = pd.concat(kept_chunks, ignore_index=True)
    df = df.rename(columns={text_col: "text", label_col: "category"})
    info = {"all_columns": all_columns, "text_column": text_col, "label_column": label_col,
            "total_records": total_records, "records_with_text": records_with_text}
    return df, info


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---- 1. Load and analyse ----
    df, info = load_dataset()
    print("Total records in file      :", info["total_records"])
    print("Records with a narrative   :", info["records_with_text"])
    print("Columns in file            :", info["all_columns"])
    print("Text column                :", info["text_column"])
    print("Target column              :", info["label_column"])
    print("Rows read into memory      :", len(df))

    full_distribution = df["category"].value_counts()
    print("\nClass distribution (all categories):\n", full_distribution)

    # ---- 2. Keep top categories, balance them ----
    top_classes = list(full_distribution.index[:NUM_CLASSES])
    parts = []
    for name in top_classes:
        rows = df[df["category"] == name]
        if len(rows) > MAX_PER_CLASS:
            rows = rows.sample(MAX_PER_CLASS, random_state=SEED)
        parts.append(rows)
    df = pd.concat(parts, ignore_index=True)

    # ---- 3. Clean text, remove very short / duplicate complaints ----
    df["clean_text"] = df["text"].apply(clean_text)
    df["word_count"] = df["clean_text"].str.split().str.len()
    df = df[df["word_count"] >= 5]
    df = df.drop_duplicates(subset="clean_text")
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)   # shuffle
    print("\nFinal dataset size:", len(df))
    print(df["category"].value_counts())

    # ---- 4. Class distribution graph ----
    counts = df["category"].value_counts()
    plt.figure(figsize=(9, 5))
    plt.barh(counts.index, counts.values, color="steelblue")
    plt.xlabel("Number of complaints")
    plt.title("Class distribution used for training")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    plt.close()

    # ---- 5. Encode labels (text -> number) ----
    class_names = sorted(df["category"].unique())
    label_to_index = {}
    for index, name in enumerate(class_names):
        label_to_index[name] = index
    y = df["category"].map(label_to_index).to_numpy()

    # ---- 6. Split: 70% train, 10% validation, 20% test (stratified) ----
    texts_train, texts_temp, y_train, y_temp = train_test_split(
        df["clean_text"].tolist(), y, test_size=0.30, random_state=SEED, stratify=y)
    texts_val, texts_test, y_val, y_test = train_test_split(
        texts_temp, y_temp, test_size=2 / 3, random_state=SEED, stratify=y_temp)

    # ---- 7. Tokenize (fit ONLY on training text) and pad ----
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts_train)

    def to_padded(texts):
        sequences = tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=MAX_LEN, padding="pre", truncating="post")

    X_train, X_val, X_test = to_padded(texts_train), to_padded(texts_val), to_padded(texts_test)

    # ---- 8. Save everything the other files need ----
    np.savez(os.path.join(OUTPUT_DIR, "data.npz"),
             X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    with open(os.path.join(MODEL_DIR, "tokenizer.json"), "w") as f:
        f.write(tokenizer.to_json())
    with open(os.path.join(MODEL_DIR, "classes.json"), "w") as f:
        json.dump(class_names, f)

    dataset_info = {
        "total_records": info["total_records"],
        "records_with_text": info["records_with_text"],
        "records_used_after_cleaning": int(len(df)),
        "all_columns": info["all_columns"],
        "text_column": info["text_column"],
        "label_column": info["label_column"],
        "num_categories": len(class_names),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "vocabulary_size_total": int(len(tokenizer.word_index)),
        "vocabulary_size_used": VOCAB_SIZE,
        "max_sequence_length": MAX_LEN,
        "average_words_per_complaint": round(float(df["word_count"].mean()), 1),
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    print("\nPreprocessing finished. Train/Val/Test:", len(X_train), len(X_val), len(X_test))


if __name__ == "__main__":
    main()
