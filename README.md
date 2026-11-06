# IntelliTriage
Deep Learning-Based Automated Customer Complaint Triage, Priority Prediction and Department Routing using LSTM and GRU.

## Dataset
CFPB Consumer Complaints (Kaggle: "Consumer Complaints Dataset" / `complaints.csv`).
Download it and place it at `dataset/complaints.csv`.
- Text column: `Consumer complaint narrative`
- Target column: `Product` (the 6 most frequent products are used)

## Setup and run (in this order)
```
pip install -r requirements.txt
python preprocess.py        # analyse, clean, tokenize, pad, split
python train.py             # trains LSTM and GRU  (python train.py 3 for a quick 3-epoch run)
python compare_models.py    # metrics, graphs, confusion matrix
streamlit run app.py        # dashboard
```

## Files
| File | Purpose |
|---|---|
| preprocess.py | Loads CSV in chunks, handles missing values, cleans text, balances classes, splits 70/10/20, tokenizes, pads, saves data + tokenizer |
| train.py | Builds and trains LSTM and GRU with identical settings; saves models, histories, training time |
| compare_models.py | Evaluates both on the test set (accuracy, precision, recall, F1), draws all graphs and the confusion matrix |
| prediction.py | Loads a model and predicts category + confidence; word-importance (explainability) |
| automation.py | RULE-BASED part: priority, department mapping, ticket generation |
| app.py | Streamlit dashboard (5 pages) |

## Deep learning vs rules
- **Deep learning (LSTM/GRU):** complaint category + confidence.
- **Rule-based:** priority (keywords), department (category mapping), ticket, manual-review flag (confidence < 50%).
The dataset has no priority or department labels, so none are claimed to come from the model.

## Notes
- All numbers in the dashboard are read from files produced by real runs (`outputs/metrics.json`, etc.).
- Settings (vocabulary 10,000, length 200, embedding 64, units 64, dropout 0.3, batch 64, max 10 epochs with early stopping) are at the top of `preprocess.py` / `train.py`.
