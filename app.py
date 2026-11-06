import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from prediction import load_resources, predict_category, word_importance
from automation import (
    get_department,
    create_ticket,
    save_ticket,
    CONFIDENCE_LIMIT,
    predict_urgency
)

OUTPUT_DIR = "outputs"

st.set_page_config(
    page_title="IntelliTriage",
    layout="wide"
)


def read_json(name):
    path = os.path.join(OUTPUT_DIR, name)

    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def show_image(name, caption):
    path = os.path.join(OUTPUT_DIR, name)

    if os.path.exists(path):
        st.image(path, caption=caption)


metrics = read_json("metrics.json")
dataset_info = read_json("dataset_info.json")

st.sidebar.title("IntelliTriage")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Complaint Analysis",
        "LSTM vs GRU",
        "Ticket",
        "Dataset / Model Information"
    ]
)

if page == "Home":
    st.title("IntelliTriage")

    st.subheader(
        "Automated Customer Complaint Analysis, "
        "Urgency Prediction and Department Routing"
    )

    st.write(
        "IntelliTriage analyzes customer complaints using deep-learning "
        "models. It predicts the complaint category, predicts urgency, "
        "routes the complaint to a department and creates a support ticket."
    )

    st.markdown("### System workflow")

    st.code(
        "Customer complaint\n"
        "   -> Text cleaning and preprocessing\n"
        "   -> LSTM / GRU category model\n"
        "   -> Predicted complaint category and confidence\n"
        "   -> Separate Bidirectional LSTM urgency model\n"
        "   -> Predicted urgency: High / Medium / Low\n"
        "   -> Department routing\n"
        "   -> Support ticket creation"
    )

    st.info(
        "Complaint category and urgency are predicted using separate "
        "deep-learning models. Department routing uses predefined category "
        "mappings. Low-confidence predictions are marked for manual review."
    )

elif page == "Complaint Analysis":
    st.title("Complaint Analysis")

    choices = ["LSTM", "GRU"]
    default_index = 0

    if metrics is not None and metrics.get("best_model") == "GRU":
        default_index = 1

    model_name = st.selectbox(
        "Category prediction model",
        choices,
        index=default_index
    )

    complaint = st.text_area(
        "Enter the customer complaint",
        height=150,
        placeholder=(
            "Example: Someone accessed my bank account without permission "
            "and transferred my money."
        )
    )

    show_explanation = st.checkbox(
        "Show important words (category explainability)",
        value=True
    )

    if st.button("Analyze"):
        if not complaint.strip():
            st.warning("Please type a complaint first.")

        else:
            try:
                with st.spinner("Analyzing complaint..."):
                    model, tokenizer, class_names = load_resources(model_name)

                    category, confidence, cleaned, probabilities = predict_category(
                        complaint,
                        model,
                        tokenizer,
                        class_names
                    )

                    urgency, urgency_confidence = predict_urgency(complaint)

                    department = get_department(category)

                    review_reasons = []

                    if confidence < CONFIDENCE_LIMIT:
                        review_reasons.append("low category confidence")

                    if urgency_confidence < CONFIDENCE_LIMIT:
                        review_reasons.append("low urgency confidence")

                    if review_reasons:
                        department = "Triage Desk (manual review)"

                    priority = urgency

                    reason = (
                        "Predicted by the separate deep-learning urgency model. "
                        "Urgency confidence: "
                        + str(round(urgency_confidence * 100, 1))
                        + "%."
                    )

                    ticket = create_ticket(
                        complaint,
                        category,
                        confidence,
                        priority,
                        department,
                        model_name,
                        urgency=urgency,
                        urgency_confidence=urgency_confidence
                    )

                    save_ticket(ticket)
                    st.session_state["ticket"] = ticket

                st.markdown("### Analysis Results")

                row1 = st.columns(3)

                row1[0].metric(
                    "Predicted Category",
                    category
                )

                row1[1].metric(
                    "Predicted Urgency",
                    urgency
                )

                row1[2].metric(
                    "Urgency Confidence",
                    f"{urgency_confidence * 100:.1f}%"
                )

                row2 = st.columns(3)

                row2[0].metric(
                    "Category Confidence",
                    f"{confidence * 100:.1f}%"
                )

                row2[1].metric(
                    "Category Model",
                    model_name
                )

                row2[2].metric(
                    "Urgency Model",
                    "Bidirectional LSTM"
                )

                st.markdown("#### Full Prediction Details")

                st.write(f"**Predicted category:** {category}")
                st.write(f"**Predicted urgency:** {urgency}")
                st.write(f"**Assigned department:** {department}")
                st.write(f"**Priority reason:** {reason}")

                if urgency == "High":
                    st.error(
                        "The model classified this complaint as High urgency. "
                        "Review the complaint promptly."
                    )

                if urgency_confidence < CONFIDENCE_LIMIT:
                    st.warning(
                        "The urgency model has low confidence. "
                        "Human review is recommended."
                    )

                if confidence < CONFIDENCE_LIMIT:
                    st.warning(
                        "The category model has low confidence. "
                        "Human review is recommended."
                    )

                if ticket["Status"] != "Automatically Routed":
                    st.warning(
                        "This ticket requires manual review before routing."
                    )

                st.success(
                    "Ticket "
                    + ticket["Ticket ID"]
                    + " created. Open the Ticket page to view it."
                )

                st.markdown("#### Category prediction probabilities")

                prob_table = pd.DataFrame({
                    "Category": class_names,
                    "Probability (%)": [
                        round(float(probability) * 100, 1)
                        for probability in probabilities
                    ]
                })

                st.dataframe(
                    prob_table,
                    hide_index=True,
                    use_container_width=True
                )

                if show_explanation:
                    st.markdown("#### Important words for category prediction")

                    class_index = class_names.index(category)

                    scores = word_importance(
                        cleaned,
                        model,
                        tokenizer,
                        class_index
                    )

                    words = [word for word, score in scores]
                    values = [score for word, score in scores]

                    if words:
                        fig, ax = plt.subplots(figsize=(10, 3))

                        colours = [
                            "tab:green" if value > 0 else "tab:red"
                            for value in values
                        ]

                        ax.bar(
                            range(len(words)),
                            values,
                            color=colours
                        )

                        ax.set_xticks(range(len(words)))
                        ax.set_xticklabels(words, rotation=90)
                        ax.set_ylabel("Change in predicted probability")
                        ax.set_title("Approximate category word importance")

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        st.caption(
                            "This explanation estimates how the predicted "
                            "category probability changes when words are "
                            "removed. It is an approximation, not a guarantee "
                            "of how the model reasons."
                        )

            except FileNotFoundError as error:
                st.error(str(error))

            except ValueError as error:
                st.error(str(error))

            except Exception as error:
                st.error("Could not analyze: " + str(error))

elif page == "LSTM vs GRU":
    st.title("LSTM vs GRU")

    if metrics is None:
        st.warning("Run compare_models.py first.")

    else:
        rows = []

        for name in ["LSTM", "GRU"]:
            if name not in metrics:
                continue

            model_metrics = metrics[name]

            rows.append({
                "Model": name,
                "Accuracy": round(model_metrics["accuracy"], 4),
                "Precision": round(model_metrics["precision"], 4),
                "Recall": round(model_metrics["recall"], 4),
                "F1-score": round(model_metrics["f1_score"], 4),
                "Training time (s)": model_metrics["training_time_seconds"],
                "Parameters": model_metrics["parameters"],
                "Epochs run": model_metrics["epochs_run"],
                "Train loss": round(model_metrics["final_train_loss"], 4),
                "Validation loss": round(model_metrics["final_val_loss"], 4),
                "Train accuracy": round(
                    model_metrics["final_train_accuracy"], 4
                ),
                "Validation accuracy": round(
                    model_metrics["final_val_accuracy"], 4
                )
            })

        if rows:
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                use_container_width=True
            )

        if metrics.get("best_model"):
            st.success(
                "Best category model by weighted F1-score: "
                + metrics["best_model"]
            )

        col1, col2 = st.columns(2)

        with col1:
            show_image("lstm_loss.png", "LSTM loss")
            show_image("lstm_accuracy.png", "LSTM accuracy")
            show_image("lstm_vs_gru.png", "LSTM vs GRU metrics")

        with col2:
            show_image("gru_loss.png", "GRU loss")
            show_image("gru_accuracy.png", "GRU accuracy")
            show_image("training_time.png", "Training time")

        show_image(
            "confusion_matrix.png",
            "Confusion matrix of the best category model"
        )

        urgency_metrics = read_json("urgency_metrics.json")

        st.markdown("### Urgency model")

        if urgency_metrics is None:
            st.info(
                "Train the urgency model to display its evaluation metrics."
            )

        else:
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Urgency Test Accuracy",
                f"{urgency_metrics['test_accuracy'] * 100:.1f}%"
            )

            col2.metric(
                "Urgency Training Samples",
                urgency_metrics["training_samples"]
            )

            col3.metric(
                "Urgency Testing Samples",
                urgency_metrics["testing_samples"]
            )

            st.write("Urgency classes:", ", ".join(urgency_metrics["classes"]))

elif page == "Ticket":
    st.title("Automatically Generated Ticket")

    if "ticket" not in st.session_state:
        st.info("Analyze a complaint first.")

    else:
        for key, value in st.session_state["ticket"].items():
            st.write(f"**{key}:** {value}")

    ticket_file = os.path.join(OUTPUT_DIR, "tickets.csv")

    if os.path.exists(ticket_file):
        st.markdown("#### All tickets so far")

        try:
            ticket_data = pd.read_csv(ticket_file)

            st.dataframe(
                ticket_data,
                hide_index=True,
                use_container_width=True
            )

        except Exception as error:
            st.warning("Could not read the ticket file: " + str(error))

elif page == "Dataset / Model Information":
    st.title("Dataset / Model Information")

    if dataset_info is None:
        st.warning("Run preprocess.py first.")

    else:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Records in file",
            dataset_info["total_records"]
        )

        col2.metric(
            "Records used",
            dataset_info["records_used_after_cleaning"]
        )

        col3.metric(
            "Categories",
            dataset_info["num_categories"]
        )

        col1.metric(
            "Training samples",
            dataset_info["train_samples"]
        )

        col2.metric(
            "Validation samples",
            dataset_info["validation_samples"]
        )

        col3.metric(
            "Testing samples",
            dataset_info["test_samples"]
        )

        col1.metric(
            "Total vocabulary",
            dataset_info["vocabulary_size_total"]
        )

        col2.metric(
            "Vocabulary used",
            dataset_info["vocabulary_size_used"]
        )

        col3.metric(
            "Maximum sequence length",
            dataset_info["max_sequence_length"]
        )

        st.write(
            "Text column: **"
            + str(dataset_info["text_column"])
            + "**, target column: **"
            + str(dataset_info["label_column"])
            + "**"
        )

        st.dataframe(
            pd.DataFrame({
                "Category": list(
                    dataset_info["class_counts"].keys()
                ),
                "Complaints": list(
                    dataset_info["class_counts"].values()
                )
            }),
            hide_index=True,
            use_container_width=True
        )

        show_image(
            "class_distribution.png",
            "Category class distribution"
        )

        st.markdown("#### Category model architecture")

        choice = st.selectbox(
            "Show architecture of",
            ["LSTM", "GRU"]
        )

        try:
            model, tokenizer, class_names = load_resources(choice)

            lines = []
            model.summary(print_fn=lines.append)

            st.code("\n".join(lines))

        except Exception as error:
            st.warning(str(error))

    st.markdown("#### Urgency model information")

    urgency_metrics = read_json("urgency_metrics.json")

    if urgency_metrics is None:
        st.info(
            "The urgency model has not been trained yet. "
            "Run python train_urgency.py."
        )

    else:
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Urgency Test Accuracy",
            f"{urgency_metrics['test_accuracy'] * 100:.1f}%"
        )

        col2.metric(
            "Urgency Training Samples",
            urgency_metrics["training_samples"]
        )

        col3.metric(
            "Urgency Testing Samples",
            urgency_metrics["testing_samples"]
        )

        st.write(
            "Urgency classes: "
            + ", ".join(urgency_metrics["classes"])
        )