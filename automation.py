import os
import csv
import json
from datetime import datetime

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import clean_text, MAX_LEN, MODEL_DIR

OUTPUT_DIR = "outputs"
TICKET_FILE = os.path.join(OUTPUT_DIR, "tickets.csv")
CONFIDENCE_LIMIT = 0.50

URGENCY_MODEL_PATH = os.path.join(MODEL_DIR, "urgency_model.keras")
URGENCY_TOKENIZER_PATH = os.path.join(MODEL_DIR, "urgency_tokenizer.json")
URGENCY_CLASSES_PATH = os.path.join(MODEL_DIR, "urgency_classes.json")

TICKET_FIELDS = [
    "Ticket ID",
    "Complaint",
    "Category",
    "Priority",
    "Department",
    "Model",
    "Confidence",
    "Urgency Model",
    "Urgency Confidence",
    "Status",
    "Date/Time"
]

DEPARTMENT_RULES = [
    ("credit report", "Credit Reporting Team"),
    ("debt collection", "Collections Compliance Team"),
    ("credit card", "Cards & Payments Team"),
    ("prepaid", "Cards & Payments Team"),
    ("checking", "Retail Banking Team"),
    ("savings", "Retail Banking Team"),
    ("bank account", "Retail Banking Team"),
    ("mortgage", "Mortgage Team"),
    ("student loan", "Loans Team"),
    ("vehicle", "Loans Team"),
    ("consumer loan", "Loans Team"),
    ("payday", "Loans Team"),
    ("money transfer", "Payments Team"),
    ("virtual currency", "Payments Team"),
    ("wire transfer", "Payments Team"),
    ("fraud", "Fraud Investigation Team"),
    ("identity theft", "Fraud Investigation Team"),
    ("unauthorized", "Fraud Investigation Team")
]

DEFAULT_DEPARTMENT = "General Customer Support"


def get_department(category):
    name = str(category).lower()

    for word, department in DEPARTMENT_RULES:
        if word in name:
            return department

    return DEFAULT_DEPARTMENT


def predict_urgency(complaint):
    required_files = [
        URGENCY_MODEL_PATH,
        URGENCY_TOKENIZER_PATH,
        URGENCY_CLASSES_PATH
    ]

    missing = [path for path in required_files if not os.path.isfile(path)]

    if missing:
        raise FileNotFoundError(
            "Urgency model files are missing. Run python train_urgency.py first. "
            "Missing files: " + ", ".join(missing)
        )

    cleaned = clean_text(complaint)

    if not cleaned:
        raise ValueError("The complaint has no readable words.")

    with open(URGENCY_TOKENIZER_PATH, "r", encoding="utf-8") as file:
        tokenizer = tokenizer_from_json(file.read())

    with open(URGENCY_CLASSES_PATH, "r", encoding="utf-8") as file:
        classes = json.load(file)

    model = load_model(URGENCY_MODEL_PATH)

    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="pre",
        truncating="post"
    )

    probabilities = model.predict(padded, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return classes[predicted_index], float(probabilities[predicted_index])


def get_priority(complaint, category=None):
    urgency, confidence = predict_urgency(complaint)

    reason = (
        "Deep-learning urgency prediction: "
        + urgency
        + " (confidence "
        + str(round(confidence * 100, 1))
        + "%)"
    )

    if confidence < CONFIDENCE_LIMIT:
        reason += "; human review recommended"

    return urgency, reason


def read_existing_tickets():
    if not os.path.isfile(TICKET_FILE):
        return []

    recovered = []

    with open(TICKET_FILE, "r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)

        try:
            header = next(reader)
        except StopIteration:
            return []

        header = [value.strip() for value in header]

        for row in reader:
            if not row:
                continue

            if len(row) == len(header):
                old_ticket = dict(zip(header, row))

            elif len(row) > len(header):
                old_ticket = dict(zip(header, row[:len(header)]))

                extra_values = row[len(header):]
                complaint_key = "Complaint"

                if complaint_key in old_ticket:
                    old_ticket[complaint_key] = (
                        str(old_ticket.get(complaint_key, ""))
                        + ", "
                        + ", ".join(extra_values)
                    )

            else:
                padded_row = row + [""] * (len(header) - len(row))
                old_ticket = dict(zip(header, padded_row))

            normalized = {field: "" for field in TICKET_FIELDS}

            for field in TICKET_FIELDS:
                if field in old_ticket:
                    normalized[field] = old_ticket[field]

            if not normalized["Ticket ID"]:
                continue

            if not normalized["Priority"] and normalized["Urgency Confidence"]:
                normalized["Priority"] = "Unknown"

            if not normalized["Urgency Model"]:
                normalized["Urgency Model"] = "Not recorded"

            recovered.append(normalized)

    return recovered


def rewrite_ticket_file(tickets):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    temporary_file = TICKET_FILE + ".tmp"

    with open(
        temporary_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=TICKET_FIELDS,
            extrasaction="ignore"
        )

        writer.writeheader()

        for ticket in tickets:
            writer.writerow({
                field: ticket.get(field, "")
                for field in TICKET_FIELDS
            })

    os.replace(temporary_file, TICKET_FILE)


def next_ticket_number():
    tickets = read_existing_tickets()
    numbers = []

    for ticket in tickets:
        ticket_id = str(ticket.get("Ticket ID", ""))

        if ticket_id.startswith("TKT-"):
            try:
                numbers.append(int(ticket_id[4:]))
            except ValueError:
                pass

    return max(numbers, default=10000) + 1


def create_ticket(
    complaint,
    category,
    confidence,
    priority,
    department,
    model_name,
    urgency=None,
    urgency_confidence=None
):
    if urgency is None or urgency_confidence is None:
        urgency, urgency_confidence = predict_urgency(complaint)

    manual_review = (
        float(confidence) < CONFIDENCE_LIMIT
        or float(urgency_confidence) < CONFIDENCE_LIMIT
    )

    return {
        "Ticket ID": "TKT-" + str(next_ticket_number()),
        "Complaint": complaint,
        "Category": category,
        "Priority": priority,
        "Department": department,
        "Model": model_name,
        "Confidence": str(round(float(confidence) * 100, 1)) + "%",
        "Urgency Model": "Bidirectional LSTM",
        "Urgency Confidence": (
            str(round(float(urgency_confidence) * 100, 1)) + "%"
        ),
        "Status": (
            "Manual Review Required"
            if manual_review
            else "Automatically Routed"
        ),
        "Date/Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def save_ticket(ticket):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing_tickets = read_existing_tickets()
    normalized_ticket = {
        field: ticket.get(field, "")
        for field in TICKET_FIELDS
    }

    existing_tickets.append(normalized_ticket)
    rewrite_ticket_file(existing_tickets)