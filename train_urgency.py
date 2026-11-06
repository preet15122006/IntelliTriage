import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocess import clean_text, MAX_LEN, VOCAB_SIZE, MODEL_DIR

SEED = 42
OUTPUT_DIR = "outputs"

DATASET_CANDIDATES = [
    "dataset/cfpb_labeled_3000(1).csv",
    "dataset/cfpb_labeled_3000.csv",
    "cfpb_labeled_3000(1).csv",
    "cfpb_labeled_3000.csv",
    "dataset/complaints.csv"
]

MODEL_PATH = os.path.join(MODEL_DIR, "urgency_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "urgency_tokenizer.json")
CLASSES_PATH = os.path.join(MODEL_DIR, "urgency_classes.json")
METRICS_PATH = os.path.join(OUTPUT_DIR, "urgency_metrics.json")

np.random.seed(SEED)
tf.random.set_seed(SEED)


def find_column(columns, candidates):
    lookup = {str(column).strip().lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset_path = next(
        (path for path in DATASET_CANDIDATES if os.path.isfile(path)),
        None
    )

    if dataset_path is None:
        raise FileNotFoundError(
            "Urgency CSV not found. Place cfpb_labeled_3000(1).csv "
            "inside the dataset folder or the project root."
        )

    df = pd.read_csv(dataset_path)
    df.columns = [str(column).strip() for column in df.columns]

    text_column = find_column(
        df.columns,
        [
            "narrative",
            "complaint",
            "complaint_text",
            "consumer complaint narrative",
            "consumer_complaint_narrative",
            "text"
        ]
    )

    label_column = find_column(
        df.columns,
        ["urgency_label", "urgency", "priority", "priority_label"]
    )

    if text_column is None or label_column is None:
        raise ValueError(
            "Required complaint-text or urgency-label column not found. "
            "CSV columns: " + str(list(df.columns))
        )

    df = df[[text_column, label_column]].copy()
    df.columns = ["narrative", "urgency_label"]

    df["narrative"] = df["narrative"].fillna("").astype(str)
    df["urgency_label"] = (
        df["urgency_label"].fillna("").astype(str).str.strip().str.lower()
    )

    label_mapping = {
        "high": "High",
        "urgent": "High",
        "critical": "High",
        "medium": "Medium",
        "moderate": "Medium",
        "normal": "Medium",
        "low": "Low",
        "minor": "Low"
    }

    df["urgency_label"] = df["urgency_label"].map(label_mapping)
    df = df.dropna(subset=["urgency_label"])

    df["clean_text"] = df["narrative"].apply(clean_text)
    df["word_count"] = df["clean_text"].str.split().str.len()
    df = df[df["word_count"] >= 3]
    df = df[df["clean_text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable complaints remain after text cleaning.")

    required_classes = {"High", "Medium", "Low"}
    if set(df["urgency_label"].unique()) != required_classes:
        raise ValueError(
            "The dataset must contain High, Medium and Low urgency examples. "
            "Current labels: " + str(df["urgency_label"].value_counts().to_dict())
        )

    counts = df["urgency_label"].value_counts()

    if counts.min() < 5:
        raise ValueError(
            "At least five examples per urgency class are required. "
            "Current counts: " + str(counts.to_dict())
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["urgency_label"])
    texts = df["clean_text"].tolist()

    texts_train, texts_temp, y_train, y_temp = train_test_split(
        texts, y, test_size=0.30, random_state=SEED, stratify=y
    )

    texts_val, texts_test, y_val, y_test = train_test_split(
        texts_temp, y_temp, test_size=2 / 3, random_state=SEED, stratify=y_temp
    )

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts_train)

    def encode(items):
        return pad_sequences(
            tokenizer.texts_to_sequences(items),
            maxlen=MAX_LEN,
            padding="pre",
            truncating="post"
        )

    X_train = encode(texts_train)
    X_val = encode(texts_val)
    X_test = encode(texts_test)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = {
        int(label): float(weight)
        for label, weight in zip(np.unique(y_train), weights)
    }

    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(VOCAB_SIZE, 64, mask_zero=True),
        Bidirectional(LSTM(64)),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(len(encoder.classes_), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    print("Dataset:", dataset_path)
    print("Rows used:", len(df))
    print("Class distribution:", counts.to_dict())
    print("Training urgency model...")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=15,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)

    accuracy = float(accuracy_score(y_test, predictions))

    print("\nTest accuracy:", round(accuracy, 4))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_))
    ))

    with open(TOKENIZER_PATH, "w", encoding="utf-8") as file:
        file.write(tokenizer.to_json())

    with open(CLASSES_PATH, "w", encoding="utf-8") as file:
        json.dump(encoder.classes_.tolist(), file, indent=2)

    metrics = {
        "dataset": dataset_path,
        "total_samples": int(len(df)),
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "testing_samples": int(len(X_test)),
        "classes": encoder.classes_.tolist(),
        "test_accuracy": accuracy,
        "class_distribution": {
            str(label): int(count)
            for label, count in counts.items()
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nTraining complete.")
    print("Saved model:", MODEL_PATH)
    print("Saved tokenizer:", TOKENIZER_PATH)
    print("Saved classes:", CLASSES_PATH)
    print("Saved metrics:", METRICS_PATH)


if __name__ == "__main__":
    main()