"""
compare_models.py  -  Step 3: Evaluate both models on the TEST set and draw graphs
Run:  python compare_models.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from tensorflow.keras.models import load_model
from preprocess import OUTPUT_DIR, MODEL_DIR


def load_history(name):
    with open(os.path.join(OUTPUT_DIR, name.lower() + "_history.json")) as f:
        return json.load(f)


def plot_loss_and_accuracy(name, history):
    h = history["history"]
    epochs = range(1, len(h["loss"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, h["loss"], marker="o", label="Training loss")
    plt.plot(epochs, h["val_loss"], marker="o", label="Validation loss")
    plt.title(name + ": Training vs Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name.lower() + "_loss.png")); plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, h["accuracy"], marker="o", label="Training accuracy")
    plt.plot(epochs, h["val_accuracy"], marker="o", label="Validation accuracy")
    plt.title(name + ": Training vs Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name.lower() + "_accuracy.png")); plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 8))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    short_names = [n[:25] for n in class_names]
    plt.xticks(range(len(class_names)), short_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), short_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            colour = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            plt.text(j, i, matrix[i, j], ha="center", va="center", color=colour)
    plt.xlabel("Predicted category"); plt.ylabel("Actual category")
    plt.title("Confusion Matrix - " + model_name + " (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png")); plt.close()


def main():
    data = np.load(os.path.join(OUTPUT_DIR, "data.npz"))
    X_test, y_test = data["X_test"], data["y_test"]
    with open(os.path.join(MODEL_DIR, "classes.json")) as f:
        class_names = json.load(f)

    results = {}
    predictions = {}
    for name in ["LSTM", "GRU"]:
        model = load_model(os.path.join(MODEL_DIR, name.lower() + "_model.keras"))
        history = load_history(name)
        probabilities = model.predict(X_test, verbose=0)
        y_pred = np.argmax(probabilities, axis=1)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        h = history["history"]
        results[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "test_loss": float(test_loss),
            "training_time_seconds": history["train_time_seconds"],
            "parameters": history["parameters"],
            "epochs_run": history["epochs_run"],
            "final_train_loss": h["loss"][-1], "final_val_loss": h["val_loss"][-1],
            "final_train_accuracy": h["accuracy"][-1], "final_val_accuracy": h["val_accuracy"][-1],
        }
        predictions[name] = y_pred
        plot_loss_and_accuracy(name, history)
        print("\n", name, "classification report")
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        print(report)
        with open(os.path.join(OUTPUT_DIR, name.lower() + "_report.txt"), "w") as f:
            f.write(report)

    best = "LSTM"
    if results["GRU"]["f1_score"] > results["LSTM"]["f1_score"]:
        best = "GRU"
    results["best_model"] = best
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # LSTM vs GRU bar charts
    names = ["Accuracy", "Precision", "Recall", "F1-score"]
    keys = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(names))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, [results["LSTM"][k] for k in keys], 0.4, label="LSTM")
    plt.bar(x + 0.2, [results["GRU"][k] for k in keys], 0.4, label="GRU")
    plt.xticks(x, names); plt.ylim(0, 1); plt.ylabel("Score")
    plt.title("LSTM vs GRU (test set, weighted average)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lstm_vs_gru.png")); plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(["LSTM", "GRU"], [results["LSTM"]["training_time_seconds"],
                              results["GRU"]["training_time_seconds"]],
            color=["tab:blue", "tab:orange"])
    plt.ylabel("Seconds"); plt.title("Training time comparison"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_time.png")); plt.close()

    plot_confusion_matrix(y_test, predictions[best], class_names, best)
    print("\nBest model (by weighted F1):", best)


if __name__ == "__main__":
    main()
