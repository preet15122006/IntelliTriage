# Run once:  pip install datasets   then   python get_dataset.py
# Streams the CFPB complaints from Hugging Face and saves only rows that HAVE complaint text.
import os
from datasets import load_dataset
import pandas as pd

os.makedirs("dataset", exist_ok=True)

stream = load_dataset("Mouwiya/cfpb-consumer-complaints", split="train", streaming=True)
stream = stream.shuffle(seed=42, buffer_size=20000)

rows = []
for row in stream:
    text = row["consumer_complaint_narrative"]
    if text:
        rows.append({"product": row["product"], "consumer_complaint_narrative": text})
    if len(rows) >= 150000:
        break

df = pd.DataFrame(rows)
df.to_csv("dataset/complaints.csv", index=False)
print("Saved", len(df), "complaints")
print(df["product"].value_counts())
