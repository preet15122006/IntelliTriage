# IntelliTriage - Viva Answers (simple language)

**1. Why this problem?** Companies get thousands of complaints. Reading and sorting them by hand is slow and inconsistent. Text is also sequential data, which fits the LSTM/GRU part of our syllabus.

**2. Why is it automation?** A human normally reads the complaint, decides the type, urgency and team, and writes a ticket. Our system does all of this without a human, and only asks for human review when it is not confident.

**3. Why Deep Learning?** Rules cannot cover every way people describe a problem. A neural network learns patterns from thousands of examples, including word order and context.

**4. Why LSTM?** A normal RNN forgets early words in long text (vanishing gradient). LSTM has a memory cell and gates, so it can remember important information over long complaints.

**5. Why GRU?** GRU does the same job with a simpler design (fewer gates, fewer parameters), so it is often faster. We train it to see if the simpler model is just as accurate.

**6. LSTM vs GRU?** LSTM has 3 gates (forget, input, output) and a separate cell state. GRU has 2 gates (update, reset) and no separate cell state. GRU has fewer parameters; LSTM is slightly more expressive.

**7. What is an RNN?** A network that reads a sequence one step at a time and passes a hidden state to the next step, so it remembers what came before.

**8. Tokenization?** Splitting text into words and giving each word a number (e.g. "refund" = 57).

**9. Why padding?** Complaints have different lengths but the network needs fixed-size input in a batch. We cut long ones and add zeros to short ones so all have length 200.

**10. Word embedding?** A layer that turns each word number into a vector of 64 values. Words used in similar contexts get similar vectors, and the vectors are learned during training.

**11. Why not give raw text?** Neural networks only do maths on numbers. Text must become numbers (tokens, then vectors).

**12. Why Softmax?** It converts the 6 outputs into probabilities that add up to 1, so we can pick the highest and also report a confidence.

**13. Overfitting?** The model memorises training data and does badly on new data. Sign: training accuracy high, validation accuracy low (the graphs show this).

**14. Why Dropout?** It randomly switches off 30% of neurons during training so the model cannot depend on a few neurons, which reduces overfitting.

**15. Why split data?** Training data teaches the model, validation data helps us tune/stop training, test data (never seen) gives an honest final score. We used 70/10/20.

**16. Epoch?** One full pass of the whole training set through the network.

**17. Batch size?** Number of complaints processed before weights are updated (we used 64).

**18. Learning rate?** The step size of weight updates. Too big = unstable, too small = very slow. We used 0.001 with Adam.

**19. Accuracy?** Correct predictions divided by total predictions.

**20. Precision, recall, F1?** Precision: of complaints predicted as a category, how many were right. Recall: of the real complaints of that category, how many we found. F1: the balance of both. We use the weighted average across classes.

**21. Confusion matrix?** A table of actual vs predicted categories. The diagonal is correct predictions; off-diagonal cells show which categories get confused.

**22. Why GRU may train faster?** Fewer gates means fewer calculations per step and fewer parameters. (We report the real measured times; if LSTM was faster in your run, say so honestly - it can depend on hardware.)

**23. How is priority decided?** By rules: if the complaint has words like fraud, stolen, foreclosure -> High; words like dispute, error, delay -> Medium; otherwise Low (mortgage is at least Medium). It is NOT predicted by the neural network.

**24. How is the department chosen?** A dictionary-style mapping from the predicted category (e.g. Debt collection -> Collections Compliance Team).

**25. Deep learning vs rule-based?** Deep learning: category and confidence. Rule-based: priority, department, ticket fields, manual-review flag.

**26. How is the ticket generated?** After prediction, code builds a record with ticket ID (counter starting at TKT-10001), complaint, category, priority, department, model, confidence, status and time, and appends it to `tickets.csv`.

**27. Unseen complaint?** Cleaned the same way, words are converted using the saved tokenizer; unknown words become `<OOV>`. The model still outputs probabilities. If the top probability is below 50%, status becomes "Manual Review Required".

**28. Limitations?** Only 6 categories; trained on a sample of US financial complaints; priority rules are simple and may be wrong; word importance is approximate; no pretrained embeddings; model can be unsure on very short or off-topic text.

**29. Future improvements?** Pretrained embeddings (GloVe), attention layer or Transformer, bidirectional LSTM, learning priority from labelled data, multilingual support, email/chat integration, feedback loop where agents correct predictions.

## Code questions you may get
- **Why `fit_on_texts` only on training data?** To avoid data leakage from the test set.
- **Why `padding="pre"`?** The final hidden state is used for classification, so real words should be near the end.
- **Why `sparse_categorical_crossentropy`?** Labels are integers (0-5), not one-hot vectors.
- **Why EarlyStopping?** Stops when validation loss stops improving and restores the best weights, avoiding overfitting.
- **Why same seed for both models?** For a fair comparison.
- **How does word importance work?** Remove one word at a time and measure the drop in the predicted category's probability. Approximate, not causal.
