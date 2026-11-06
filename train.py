"""
train.py  -  Step 2: Train the LSTM model and the GRU model
Both models use the SAME data, SAME architecture and SAME settings.
Only the recurrent layer is different.
Run:  python train.py        (optional: python train.py 5   -> 5 epochs)
"""
import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import VOCAB_SIZE, MAX_LEN, SEED, OUTPUT_DIR, MODEL_DIR

EMBEDDING_DIM = 64
RNN_UNITS = 64
DROPOUT_RATE = 0.3
BATCH_SIZE = 64
EPOCHS = 10.
LEARNING_RATE = 0.001

if len(sys.argv) > 1:
    EPOCHS = int(sys.argv[1])


def build_model(rnn_type, num_classes):
    """Embedding -> LSTM or GRU -> Dropout -> Dense -> Dropout -> Softmax"""
    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM))
    if rnn_type == "LSTM":
        model.add(LSTM(RNN_UNITS))
    else:
        model.add(GRU(RNN_UNITS))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(num_classes, activation="softmax"))
    model.build(input_shape=(None, MAX_LEN))
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def train_one_model(rnn_type, data, num_classes):
    tf.keras.utils.set_random_seed(SEED)          # same starting conditions for fairness
    model = build_model(rnn_type, num_classes)
    model.summary()

    stop_early = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    start = time.time()
    history = model.fit(data["X_train"], data["y_train"],
                        validation_data=(data["X_val"], data["y_val"]),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[stop_early], verbose=2)
    train_seconds = time.time() - start

    model.save(os.path.join(MODEL_DIR, rnn_type.lower() + "_model.keras"))

    result = {"history": {}, "train_time_seconds": round(train_seconds, 1),
              "parameters": int(model.count_params()),
              "epochs_run": len(history.history["loss"])}
    for key, values in history.history.items():
        result["history"][key] = [float(v) for v in values]
    with open(os.path.join(OUTPUT_DIR, rnn_type.lower() + "_history.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(rnn_type, "finished in", round(train_seconds, 1), "seconds")


def main():
    if not os.path.exists(os.path.join(OUTPUT_DIR, "data.npz")):
        raise FileNotFoundError("Run preprocess.py first.")
    data = np.load(os.path.join(OUTPUT_DIR, "data.npz"))
    num_classes = int(len(np.unique(data["y_train"])))
    for rnn_type in ["LSTM", "GRU"]:
        print("\n========== Training", rnn_type, "==========")
        train_one_model(rnn_type, data, num_classes)


if __name__ == "__main__":
    main()
