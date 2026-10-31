import io
import hashlib
from pathlib import Path
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.predict import predict_ticket

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="HCLTech | IT Support Intelligence",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)


DATA_PATH = ROOT_DIR / "data" / "processed" / "final_data.csv"

CATEGORIES = [
    "Access",
    "HR Support",
    "Hardware",
    "Purchase",
    "Storage"
]

# Final test results
MODEL_ACCURACY = 87.83
MACRO_F1 = 0.8871
WEIGHTED_F1 = 0.8800

# Validation comparison results
MODEL_COMPARISON = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Logistic Regression + Class Weights",
        "Logistic Regression + Random Oversampling",
        "Linear SVM + Random Oversampling"
    ],
    "Accuracy": [79.58, 84.79, 85.21, 84.58],
    "Macro F1": [0.7805, 0.8614, 0.8655, 0.8635]
})

# Final test classification report
CLASSIFICATION_REPORT = pd.DataFrame({
    "Category": [
        "Access",
        "HR Support",
        "Hardware",
        "Purchase",
        "Storage"
    ],
    "Precision": [0.90, 0.87, 0.84, 1.00, 0.95],
    "Recall": [0.82, 0.87, 0.92, 0.90, 0.81],
    "F1 Score": [0.86, 0.87, 0.88, 0.95, 0.88],
    "Support": [113, 174, 230, 40, 43]
})

# Final test confusion matrix
CONFUSION_MATRIX = np.array([
    [93, 7, 13, 0, 0],
    [4, 151, 18, 0, 1],
    [5, 12, 212, 0, 1],
    [0, 1, 3, 36, 0],
    [1, 2, 5, 0, 35]
])

CONFUSION_LABELS = [
    "Access",
    "HR Support",
    "Hardware",
    "Purchase",
    "Storage"
]


# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    [data-testid="stMetric"] {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
    }

    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    [data-testid="stMetricDelta"] {
        color: #94a3b8 !important;
    }

    .section-description {
        color: #aab4c3;
        font-size: 15px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #334155;
        border-radius: 8px;
    }

    .stButton button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


df = load_dataset()

if df is None:
    st.error(
        "Dataset not found. Please check that "
        "data/processed/final_data.csv exists."
    )
    st.stop()

required_columns = {"Document", "Topic_group"}

if not required_columns.issubset(df.columns):
    st.error(
        "Dataset must contain 'Document' and 'Topic_group' columns."
    )
    st.stop()


# --------------------------------------------------
# 4. HELPER FUNCTIONS
# --------------------------------------------------

def get_category_distribution(data):
    return (
        data["Topic_group"]
        .value_counts()
        .rename_axis("Category")
        .reset_index(name="Tickets")
    )


def display_confidence(confidence):
    percentage = confidence * 100

    if confidence < 0.60:
        st.error(
            f"Low confidence: {percentage:.2f}%. "
            "Please review this prediction."
        )
    elif confidence < 0.75:
        st.warning(
            f"Moderate confidence: {percentage:.2f}%. "
            "Manual verification is recommended."
        )
    else:
        st.success(
            f"Confidence: {percentage:.2f}%"
        )


def classify_dataframe(data):
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    total = len(data)

    for index, ticket in enumerate(data["Document"]):

        if pd.isna(ticket) or not str(ticket).strip():
            results.append({
                "Predicted Category": "Invalid / Empty",
                "Confidence (%)": None,
                "Status": "Empty ticket"
            })

        else:
            try:
                prediction = predict_ticket(str(ticket))

                result = {
                    "Predicted Category":
                        prediction["predicted_category"],

                    "Confidence (%)":
                        round(prediction["confidence"] * 100, 2),

                    "Status": "Success"
                }

                for category, probability in prediction[
                    "probabilities"
                ].items():

                    result[f"{category} Probability (%)"] = round(
                        probability * 100, 2
                    )

                results.append(result)

            except Exception as error:
                results.append({
                    "Predicted Category": "Error",
                    "Confidence (%)": None,
                    "Status": str(error)
                })

        progress_bar.progress((index + 1) / total)

        status_text.text(
            f"Processing ticket {index + 1} of {total}"
        )

    progress_bar.empty()
    status_text.success("Classification completed!")

    result_df = pd.DataFrame(results)

    output_df = pd.concat(
        [
            data.reset_index(drop=True),
            result_df.reset_index(drop=True)
        ],
        axis=1
    )

    return output_df


def get_text_frequency(data):
    text_column = (
        "processed_document"
        if "processed_document" in data.columns
        else "Document"
    )

    words = []

    for text in data[text_column].dropna():
        words.extend(str(text).split())

    return Counter(words).most_common(20)


def show_category_chart(data):
    fig = px.bar(
        data,
        x="Category",
        y="Tickets",
        color="Category",
        text="Tickets",
        title="Ticket Category Distribution"
    )

    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        xaxis_title="Category",
        yaxis_title="Number of Tickets"
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🎫 IT Support Intelligence")

st.sidebar.caption("HCLTech | Data Science Hackathon")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Ticket Classification",
        "Analytics & Insights",
        "Model Evaluation",
        "Worked Examples",
        "Documentation"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "AI-powered IT support ticket classification "
    "and operational analytics."
)


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.title("IT Support Intelligence")

    st.markdown(
        '<p class="section-description">'
        'Intelligent ticket classification and operational analytics'
        '</p>',
        unsafe_allow_html=True
    )

    st.divider()

    category_counts = get_category_distribution(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Tickets", f"{len(df):,}")
    col2.metric("Categories", "5")
    col3.metric("Model Accuracy", f"{MODEL_ACCURACY}%")
    col4.metric("Macro F1 Score", f"{MACRO_F1:.4f}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Ticket Category Distribution")

        fig = px.bar(
            category_counts,
            x="Category",
            y="Tickets",
            color="Category",
            text="Tickets"
        )

        fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            xaxis_title="Category",
            yaxis_title="Tickets"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Category Share")

        fig = px.pie(
            category_counts,
            names="Category",
            values="Tickets",
            hole=0.45
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Dataset Snapshot")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Largest Category",
        category_counts.iloc[0]["Category"]
    )

    col2.metric(
        "Largest Category Tickets",
        f"{category_counts.iloc[0]['Tickets']:,}"
    )

    col3.metric(
        "Smallest Category",
        category_counts.iloc[-1]["Category"]
    )

    st.subheader("Sample Tickets")

    st.dataframe(
        df[["Document", "Topic_group"]].head(10),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Operational Insights")

    st.markdown("""
    - Hardware and HR Support account for a large share of the tickets.
    - Purchase and Storage have comparatively fewer training examples.
    - The model uses TF-IDF features and Logistic Regression.
    - Batch classification can process multiple tickets from a CSV file.
    - Predictions with lower confidence should be reviewed manually.
    """)


# ==================================================
# PAGE 2: TICKET CLASSIFICATION
# ==================================================

elif page == "Ticket Classification":

    st.title("Ticket Classification")

    st.markdown(
        "Classify individual tickets or process multiple tickets "
        "using a CSV upload."
    )

    tab_single, tab_batch = st.tabs([
        "Single Ticket",
        "Batch Prediction"
    ])

    # ----------------------------------------------
    # SINGLE TICKET CLASSIFICATION
    # ----------------------------------------------

    with tab_single:

        st.subheader("Enter Ticket Description")

        ticket_text = st.text_area(
            "Ticket Description",
            placeholder=(
                "Example: Unable to access my company account..."
            ),
            height=150
        )

        if st.button(
            "Classify Ticket",
            type="primary",
            key="single_predict"
        ):

            if not ticket_text.strip():
                st.warning("Please enter a ticket description.")

            else:
                try:
                    prediction = predict_ticket(ticket_text)

                    st.session_state["single_result"] = prediction
                    st.session_state["single_ticket"] = ticket_text

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

        if "single_result" in st.session_state:

            prediction = st.session_state["single_result"]

            st.divider()
            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Category",
                prediction["predicted_category"]
            )

            col2.metric(
                "Confidence",
                f"{prediction['confidence'] * 100:.2f}%"
            )

            display_confidence(prediction["confidence"])

            st.subheader("Category Probabilities")

            probability_df = pd.DataFrame({
                "Category": list(
                    prediction["probabilities"].keys()
                ),
                "Probability": [
                    value * 100
                    for value in prediction["probabilities"].values()
                ]
            }).sort_values("Probability", ascending=True)

            fig = px.bar(
                probability_df,
                x="Probability",
                y="Category",
                orientation="h",
                text=probability_df["Probability"].map(
                    lambda x: f"{x:.2f}%"
                ),
                title="Prediction Probability by Category"
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Probability (%)",
                yaxis_title="Category"
            )

            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------
    # BATCH CLASSIFICATION
    # ----------------------------------------------

    with tab_batch:

        st.subheader("Batch Ticket Classification")

        st.markdown("""
        Upload a CSV file containing a column named **Document**.
        The system will classify every ticket and generate a downloadable
        results file.
        """)

        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            key="batch_upload"
        )

        if uploaded_file is not None:

            file_bytes = uploaded_file.getvalue()

            file_id = hashlib.md5(file_bytes).hexdigest()

            try:
                batch_df = pd.read_csv(io.BytesIO(file_bytes))

                if "Document" not in batch_df.columns:

                    st.error(
                        "The uploaded CSV must contain a "
                        "'Document' column."
                    )

                elif batch_df.empty:

                    st.warning("The uploaded CSV contains no tickets.")

                else:

                    st.success(
                        f"File uploaded successfully: "
                        f"{len(batch_df):,} tickets found."
                    )

                    st.write("Uploaded data preview")

                    st.dataframe(
                        batch_df.head(10),
                        use_container_width=True,
                        hide_index=True
                    )

                    if st.button(
                        "Classify All Tickets",
                        type="primary",
                        key="run_batch"
                    ):

                        output_df = classify_dataframe(batch_df)

                        st.session_state["batch_results"] = output_df
                        st.session_state["batch_file_id"] = file_id

                    # Show results only for the currently uploaded file
                    if (
                        "batch_results" in st.session_state
                        and st.session_state.get("batch_file_id") == file_id
                    ):

                        output_df = st.session_state["batch_results"]

                        st.divider()
                        st.subheader("Classification Results")

                        successful = (
                            output_df["Status"] == "Success"
                        ).sum()

                        failed = len(output_df) - successful

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Total Tickets",
                            f"{len(output_df):,}"
                        )

                        col2.metric(
                            "Successfully Classified",
                            f"{successful:,}"
                        )

                        col3.metric(
                            "Failed / Empty",
                            f"{failed:,}"
                        )

                        st.dataframe(
                            output_df,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Category distribution
                        successful_rows = output_df[
                            output_df["Status"] == "Success"
                        ]

                        if not successful_rows.empty:

                            st.subheader(
                                "Predicted Category Distribution"
                            )

                            prediction_counts = (
                                successful_rows[
                                    "Predicted Category"
                                ]
                                .value_counts()
                                .rename_axis("Category")
                                .reset_index(name="Tickets")
                            )

                            fig = px.bar(
                                prediction_counts,
                                x="Category",
                                y="Tickets",
                                color="Category",
                                text="Tickets",
                                title="Batch Classification Summary"
                            )

                            fig.update_layout(
                                template="plotly_dark",
                                showlegend=False
                            )

                            st.plotly_chart(
                                fig,
                                use_container_width=True
                            )

                        # Download output CSV
                        csv_data = output_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Download Classification Results",
                            data=csv_data,
                            file_name="batch_classification_results.csv",
                            mime="text/csv",
                            type="primary",
                            key="download_batch"
                        )

            except Exception as error:

                st.error(
                    f"Unable to read the uploaded CSV: {error}"
                )


# ==================================================
# PAGE 3: ANALYTICS & INSIGHTS
# ==================================================

elif page == "Analytics & Insights":

    st.title("Analytics & Insights")

    st.markdown(
        "Explore ticket volumes, category imbalance, "
        "and recurring terms in the dataset."
    )

    category_counts = get_category_distribution(df)

    total_tickets = len(df)

    category_counts["Percentage"] = (
        category_counts["Tickets"] / total_tickets * 100
    ).round(2)

    st.subheader("Category-wise Ticket Analysis")

    st.dataframe(
        category_counts,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Ticket Distribution")

        fig = px.bar(
            category_counts,
            x="Category",
            y="Tickets",
            color="Category",
            text="Tickets"
        )

        fig.update_layout(
            template="plotly_dark",
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Category Percentage")

        fig = px.pie(
            category_counts,
            names="Category",
            values="Percentage",
            hole=0.4
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Ticket Description Length")

    lengths = df["Document"].fillna("").astype(str).str.split().str.len()

    length_df = pd.DataFrame({
        "Word Count": lengths
    })

    fig = px.histogram(
        length_df,
        x="Word Count",
        nbins=30,
        title="Distribution of Ticket Lengths"
    )

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Frequently Occurring Terms")

    common_words = get_text_frequency(df)

    words_df = pd.DataFrame(
        common_words,
        columns=["Word", "Frequency"]
    )

    if not words_df.empty:

        fig = px.bar(
            words_df.sort_values("Frequency"),
            x="Frequency",
            y="Word",
            orientation="h",
            title="Top 20 Frequently Occurring Terms"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Operational Insights")

    largest = category_counts.iloc[0]
    smallest = category_counts.iloc[-1]

    st.markdown(f"""
    - **Highest ticket volume:** {largest['Category']} 
      ({largest['Tickets']} tickets, {largest['Percentage']}%).
    - **Lowest ticket volume:** {smallest['Category']} 
      ({smallest['Tickets']} tickets, {smallest['Percentage']}%).
    - The category distribution is imbalanced.
    - Lower-volume categories may require additional training examples.
    """)

    st.info(
        "The dataset does not contain a verified timestamp column. "
        "Therefore, historical or monthly ticket trends are not displayed."
    )


# ==================================================
# PAGE 4: MODEL EVALUATION
# ==================================================

elif page == "Model Evaluation":

    st.title("Model Evaluation")

    st.markdown(
        "Evaluation of the selected classification model "
        "on the untouched test set."
    )

    # Overall metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Test Accuracy", f"{MODEL_ACCURACY}%")
    col2.metric("Macro F1 Score", f"{MACRO_F1:.4f}")
    col3.metric("Weighted F1 Score", f"{WEIGHTED_F1:.4f}")

    st.divider()

    # Model comparison
    st.subheader("Model Comparison")

    st.caption(
        "Validation-set results used to select the final model."
    )

    st.dataframe(
        MODEL_COMPARISON,
        use_container_width=True,
        hide_index=True
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=MODEL_COMPARISON["Model"],
        y=MODEL_COMPARISON["Accuracy"],
        name="Accuracy (%)"
    ))

    fig.add_trace(go.Bar(
        x=MODEL_COMPARISON["Model"],
        y=MODEL_COMPARISON["Macro F1"] * 100,
        name="Macro F1 (%)"
    ))

    fig.update_layout(
        title="Validation Model Comparison",
        template="plotly_dark",
        barmode="group",
        yaxis_title="Score (%)",
        xaxis_title="Model"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "Selected model: Logistic Regression with Random Oversampling."
    )

    st.divider()

    # Per-class report
    st.subheader("Per-Class Test Performance")

    st.dataframe(
        CLASSIFICATION_REPORT.style.format({
            "Precision": "{:.2f}",
            "Recall": "{:.2f}",
            "F1 Score": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    fig = go.Figure()

    for metric in ["Precision", "Recall", "F1 Score"]:

        fig.add_trace(go.Bar(
            name=metric,
            x=CLASSIFICATION_REPORT["Category"],
            y=CLASSIFICATION_REPORT[metric]
        ))

    fig.update_layout(
        title="Per-Class Performance",
        template="plotly_dark",
        barmode="group",
        yaxis_title="Score",
        xaxis_title="Category"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Confusion matrix
    st.subheader("Confusion Matrix")

    fig = go.Figure(
        data=go.Heatmap(
            z=CONFUSION_MATRIX,
            x=CONFUSION_LABELS,
            y=CONFUSION_LABELS,
            colorscale="Blues",
            text=CONFUSION_MATRIX,
            texttemplate="%{text}",
            hovertemplate=(
                "Actual: %{y}<br>"
                "Predicted: %{x}<br>"
                "Tickets: %{z}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title="Final Test Confusion Matrix",
        template="plotly_dark",
        xaxis_title="Predicted Category",
        yaxis_title="Actual Category"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Rows represent actual categories; columns represent predicted categories."
    )

    st.divider()

    # Error analysis
    st.subheader("Error Analysis")

    st.markdown("""
    - **Access:** Recall of 0.82. Some Access tickets were predicted as Hardware.
    - **Storage:** Recall of 0.81. Some Storage tickets were classified as Hardware.
    - **Hardware:** Recall of 0.92, the highest among the five categories.
    - **Purchase:** F1 score of 0.95, the highest among the five categories.
    """)

    st.subheader("Model Limitations")

    st.markdown("""
    - The dataset has an imbalanced category distribution.
    - TF-IDF primarily captures word and phrase patterns.
    - Similar ticket descriptions across categories can cause confusion.
    - Prediction probabilities are not guaranteed to be calibrated confidence estimates.
    - Low-confidence predictions should be reviewed by a human.
    """)


# ==================================================
# PAGE 5: WORKED EXAMPLES
# ==================================================

elif page == "Worked Examples":

    st.title("Worked Examples")

    st.markdown(
        "Explore real tickets from the dataset and see how "
        "the trained model classifies them."
    )

    if st.button(
        "Generate Worked Examples",
        type="primary"
    ):

        with st.spinner("Generating examples..."):

            examples = {}

            # Common category
            common_category = (
                df["Topic_group"].value_counts().idxmax()
            )

            common_row = df[
                df["Topic_group"] == common_category
            ].iloc[0]

            examples["Common Category"] = {
                "ticket": common_row["Document"],
                "actual": common_row["Topic_group"]
            }

            # Minority category
            minority_category = (
                df["Topic_group"].value_counts().idxmin()
            )

            minority_row = df[
                df["Topic_group"] == minority_category
            ].iloc[0]

            examples["Minority Category"] = {
                "ticket": minority_row["Document"],
                "actual": minority_row["Topic_group"]
            }

            # Use one sample for ambiguous and low-confidence examples
            sample_size = min(150, len(df))

            sample_df = df.sample(
                n=sample_size,
                random_state=42
            )

            prediction_records = []

            for _, row in sample_df.iterrows():

                try:
                    result = predict_ticket(str(row["Document"]))

                    probabilities = result["probabilities"]

                    sorted_probabilities = sorted(
                        probabilities.values(),
                        reverse=True
                    )

                    margin = (
                        sorted_probabilities[0]
                        - sorted_probabilities[1]
                    )

                    prediction_records.append({
                        "ticket": row["Document"],
                        "actual": row["Topic_group"],
                        "predicted": result["predicted_category"],
                        "confidence": result["confidence"],
                        "margin": margin,
                        "probabilities": probabilities
                    })

                except Exception:
                    continue

            if prediction_records:

                # Lowest top-two probability margin
                ambiguous = min(
                    prediction_records,
                    key=lambda x: x["margin"]
                )

                # Lowest maximum predicted probability
                low_confidence = min(
                    prediction_records,
                    key=lambda x: x["confidence"]
                )

                examples["Ambiguous Ticket"] = ambiguous
                examples["Low Confidence Ticket"] = low_confidence

            st.session_state["worked_examples"] = examples

    if "worked_examples" in st.session_state:

        examples = st.session_state["worked_examples"]

        for title, example in examples.items():

            st.divider()
            st.subheader(title)

            st.write("**Ticket Description:**")
            st.write(example["ticket"])

            st.write(
                f"**Actual Category:** {example['actual']}"
            )

            if "predicted" not in example:

                try:
                    result = predict_ticket(
                        str(example["ticket"])
                    )

                    predicted = result["predicted_category"]
                    confidence = result["confidence"]

                except Exception as error:
                    st.error(f"Prediction failed: {error}")
                    continue

            else:

                predicted = example["predicted"]
                confidence = example["confidence"]

            col1, col2 = st.columns(2)

            col1.metric("Predicted Category", predicted)

            col2.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            if predicted == example["actual"]:
                st.success("Correct classification")
            else:
                st.warning("Misclassified ticket")


# ==================================================
# PAGE 6: DOCUMENTATION
# ==================================================

elif page == "Documentation":

    st.title("Project Documentation")

    st.subheader("Project Overview")

    st.write("""
    IT Support Intelligence is an AI-powered ticket classification
    system designed to automatically categorize IT support requests.
    It also provides ticket analytics and model evaluation.
    """)

    st.divider()

    st.subheader("Input Schema")

    st.write("""
    The application accepts ticket descriptions as text.

    For batch classification, upload a CSV file containing:

    - `Document`: Ticket description.
    """)

    st.divider()

    st.subheader("Ticket Categories")

    taxonomy = pd.DataFrame({
        "Category": [
            "Hardware",
            "HR Support",
            "Access",
            "Storage",
            "Purchase"
        ],
        "Description": [
            "Hardware-related support requests",
            "Human resources-related requests",
            "Account and access-related requests",
            "Storage-related requests",
            "Purchase-related requests"
        ]
    })

    st.dataframe(
        taxonomy,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Text Preprocessing")

    st.markdown("""
    1. Convert text to lowercase.
    2. Remove HTML tags.
    3. Remove URLs and email addresses.
    4. Remove special characters and punctuation.
    5. Normalize whitespace.
    6. Remove English stopwords while retaining negation words.
    """)

    st.divider()

    st.subheader("Feature Extraction")

    st.markdown("""
    **TF-IDF Vectorization**

    - Maximum features: 5,000
    - N-gram range: (1, 2)
    - Minimum document frequency: 2
    - Maximum document frequency: 0.95
    """)

    st.divider()

    st.subheader("Model Architecture")

    st.markdown("""
    **Selected Algorithm:** Logistic Regression

    **Class Imbalance Handling:** Random Oversampling

    **Training Strategy:**

    - Stratified 80:20 train-test split.
    - Validation data used for model comparison.
    - Random oversampling applied only to training data.
    - Final evaluation performed on the untouched test set.
    """)

    st.divider()

    st.subheader("Final Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{MODEL_ACCURACY}%")
    col2.metric("Macro F1", f"{MACRO_F1:.4f}")
    col3.metric("Weighted F1", f"{WEIGHTED_F1:.4f}")

    st.divider()

    st.subheader("Technology Stack")

    st.markdown("""
    - **Programming Language:** Python
    - **Data Processing:** Pandas, NumPy
    - **Machine Learning:** Scikit-learn
    - **Class Balancing:** Imbalanced-learn
    - **Visualization:** Plotly
    - **Application:** Streamlit
    - **Model Storage:** Joblib
    """)

    st.divider()

    st.subheader("Prediction Workflow")

    st.code("""
Ticket Description
        |
        v
Text Preprocessing
        |
        v
TF-IDF Vectorization
        |
        v
Trained Logistic Regression Model
        |
        v
Predicted Category + Probabilities
        |
        v
Display Results
    """, language="text")

    st.info(
        "Prediction uses the saved preprocessing logic, "
        "TF-IDF vectorizer, and trained model. "
        "The vectorizer is not refitted during prediction."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.caption("HCLTech Data Science Hackathon")
st.sidebar.caption("IT Support Intelligence")