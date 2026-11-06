# IntelliTriage: Deep Learning-Based Automated Customer Complaint Triage, Priority Prediction and Department Routing using LSTM and GRU

> Note: Sections marked **[fill from your run]** must be completed with numbers from `outputs/metrics.json` after you run the project. Nothing below invents results.

## 1. Title
IntelliTriage: Deep Learning-Based Automated Customer Complaint Triage, Priority Prediction and Department Routing using LSTM and GRU

## 2. Abstract
Organisations receive large numbers of written customer complaints that must be classified and sent to the right team. This project presents IntelliTriage, an automation pipeline that reads a complaint, classifies it using LSTM and GRU recurrent neural networks trained on the CFPB Consumer Complaints dataset, assigns a priority and department using transparent rules, and generates a support ticket displayed in a Streamlit dashboard. Both models are trained on identical data and settings and compared using accuracy, precision, recall, F1-score, training time, parameters and learning curves. **[fill from your run: best model and its F1-score]**

## 3. Introduction
Customer support is a major business function. Complaints arrive as free text, whose meaning depends on word order and context. Recurrent neural networks process text as a sequence and are therefore suitable for this task.

## 4. Problem Statement
To design a system that automatically understands a text complaint, predicts its category using deep learning, assigns a priority, routes it to a department and creates a ticket, with minimal human effort.

## 5. Motivation
Manual triage is slow, inconsistent and costly; urgent complaints (e.g. fraud) may wait in the same queue as simple queries. Automation reduces response time and workload.

## 6. Objectives
1. Build LSTM and GRU text classifiers for complaint categories.
2. Compare them fairly on the same data.
3. Automatically assign priority and department.
4. Generate tickets automatically.
5. Provide an interactive dashboard with simple explainability.

## 7. Existing System
Manual reading and forwarding of complaints, or keyword-only filters that fail when customers use different wording.

## 8. Proposed System
A pipeline combining deep learning (category prediction) with rule-based automation (priority, department, ticket) in one dashboard. Rules are used only where the dataset has no labels.

## 9. System Architecture
Complaint -> Cleaning -> Tokenizer -> Padding -> Embedding -> LSTM/GRU -> Dense -> Softmax -> Category + Confidence -> Priority rules -> Department mapping -> Ticket -> Dashboard.
(Draw this as a block diagram in your report.)

## 10. Dataset
CFPB Consumer Complaints (Kaggle). Text column: *Consumer complaint narrative*; target: *Product*. Records with missing narrative or product are removed; the six most frequent products are kept and capped per class for balance. **[fill from your run: total records, records used, class counts from `outputs/dataset_info.json`, class distribution graph]**

## 11. Data Preprocessing
Lowercasing; removal of redacted "XXXX", digits and symbols; removal of very short and duplicate complaints; label encoding; stratified 70/10/20 train/validation/test split; Keras Tokenizer (vocabulary 10,000, OOV token) fitted on training data only; padding/truncation to 200 tokens.

## 12. LSTM Architecture
Embedding(10000, 64) -> LSTM(64) -> Dropout(0.3) -> Dense(64, ReLU) -> Dropout(0.3) -> Dense(classes, Softmax). The LSTM uses forget, input and output gates and a cell state to keep long-range information.

## 13. GRU Architecture
Identical to the LSTM model but with GRU(64). The GRU uses update and reset gates and no separate cell state, giving fewer parameters.

## 14. Training Methodology
Adam optimiser (learning rate 0.001), sparse categorical cross-entropy, batch size 64, up to 10 epochs with early stopping on validation loss (patience 2, best weights restored), same random seed and data for both models.

## 15. Model Comparison
Metrics on the test set: accuracy, weighted precision, recall and F1, training time, parameter count, final train/validation loss and accuracy. **[fill from your run: table from `metrics.json`, loss/accuracy graphs, LSTM vs GRU chart, training-time chart, confusion matrix, and your honest discussion of which is better and why]**

## 16. Complaint Triage
The selected model outputs a probability for each category; the highest is the predicted category and its probability is the confidence. If confidence is below 50%, the ticket is flagged for manual review.

## 17. Priority Prediction
Priority is **not** predicted by the neural network because the dataset has no priority labels. It is assigned by keyword rules: High (e.g. fraud, stolen, foreclosure), Medium (e.g. dispute, error, delay; mortgage by policy), Low otherwise. The dashboard shows the reason for each decision.

## 18. Department Routing
A fixed mapping from category to department (e.g. Debt collection -> Collections Compliance Team, Credit reporting -> Credit Reporting Team, Mortgage -> Mortgage Team), with a general support default.

## 19. Automated Ticket Generation
Each analysis creates a ticket with ID, complaint, category, priority, department, model, confidence, status and timestamp, saved to `outputs/tickets.csv`.

## 20. Dashboard
Streamlit app with Home, Complaint Analysis (with word-importance chart), LSTM vs GRU, Ticket and Dataset/Model Information pages. Word importance uses leave-one-word-out probability change and is only an approximate indicator.

## 21. Results
**[fill from your run: screenshots of each dashboard page and the metrics table]**

## 22. Limitations
Six categories only; sampled US financial complaints; simple rule-based priority; approximate explainability; embeddings trained from scratch; limited handling of very short or off-topic text.

## 23. Future Scope
Pretrained embeddings, bidirectional LSTM, attention/Transformer models, learned priority from labelled data, multilingual support, email/chat integration, agent feedback loop.

## 24. Conclusion
IntelliTriage shows that LSTM/GRU networks can classify complaint text and, combined with simple transparent automation, form a complete triage pipeline. **[fill from your run: final conclusion about LSTM vs GRU]**

## 25. References
1. Consumer Financial Protection Bureau, Consumer Complaint Database.
2. Hochreiter, S. and Schmidhuber, J., "Long Short-Term Memory", Neural Computation, 1997.
3. Cho, K. et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", 2014.
4. Zhang, A. et al., Dive into Deep Learning, Cambridge University Press.
5. Goodfellow, I., Bengio, Y., Courville, A., Deep Learning, MIT Press.
6. TensorFlow/Keras and Streamlit documentation.
