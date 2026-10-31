# AI-Powered IT Support Ticket Classification System

An NLP-based machine learning application that automatically classifies IT support tickets into predefined categories. The system uses TF-IDF feature extraction and Logistic Regression to predict ticket categories, provide confidence scores, process tickets in batches, and generate useful analytics through an interactive Streamlit dashboard.

## Project Overview

IT support teams receive tickets related to hardware, account access, HR support, storage, and purchase requests. Manually categorizing these tickets can be time-consuming and inconsistent.

This project automates ticket classification using machine learning, helping streamline ticket categorization and providing insights into ticket distribution.

## Key Features

- **Single Ticket Classification:** Predicts the category of an individual support ticket.
- **Batch Ticket Classification:** Upload a CSV file and classify multiple tickets at once.
- **Confidence Scores:** Displays prediction probabilities for each category.
- **Ticket Analytics:** Visualizes category distribution, ticket volumes, and text characteristics.
- **Model Evaluation:** Displays model performance metrics and confusion matrix.
- **Worked Examples:** Demonstrates predictions using common, minority, ambiguous, and low-confidence tickets.
- **Downloadable Results:** Exports batch classification results as a CSV file.

## Ticket Categories

The model classifies tickets into five categories:

| Category | Training Dataset Count |
|---|---:|
| Hardware | 1,150 |
| HR Support | 871 |
| Access | 565 |
| Storage | 213 |
| Purchase | 201 |
| **Total** | **3,000** |

## Machine Learning Workflow

1. Dataset loading and exploration
2. Exploratory Data Analysis (EDA)
3. Text preprocessing and cleaning
4. TF-IDF feature extraction
5. Stratified train-test splitting
6. Random oversampling of training data
7. Model training and comparison
8. Validation-based model selection
9. Final model evaluation
10. Saving the trained model and vectorizer

## Model Development

The following models were evaluated:

- Logistic Regression
- Logistic Regression with class weighting
- Logistic Regression with Random Oversampling
- Linear Support Vector Machine with Random Oversampling

**Selected Model:** Logistic Regression with Random Oversampling.

### Text Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) converts cleaned ticket descriptions into numerical feature vectors.

Configuration:
- Maximum features: 5,000
- N-gram range: (1, 2)
- Minimum document frequency: 2
- Maximum document frequency: 0.95

### Final Model Performance

Evaluated on the untouched test dataset of 600 tickets.

| Metric | Score |
|---|---:|
| Accuracy | 87.83% |
| Macro F1-Score | 0.8871 |
| Weighted F1-Score | 0.8800 |

### Classification Report

| Category | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Access | 0.90 | 0.82 | 0.86 |
| HR Support | 0.87 | 0.87 | 0.87 |
| Hardware | 0.84 | 0.92 | 0.88 |
| Purchase | 1.00 | 0.90 | 0.95 |
| Storage | 0.95 | 0.81 | 0.88 |

## System Architecture

```text
              IT Support Ticket Dataset
                         |
                         v
                Data Exploration (EDA)
                         |
                         v
                 Text Preprocessing
                         |
                         v
                  TF-IDF Vectorization
                         |
                         v
                  Train-Test Split
                         |
                         v
                Random Oversampling
                  (Training Only)
                         |
                         v
                  Model Training
                         |
                         v
                 Model Validation
                         |
                         v
                Final Model Selection
                         |
                         v
              Save Model + Vectorizer
                         |
                         v
                Streamlit Application
                         |
                         v
                   Input Handling
                    /         \
                   v           v
             Single Ticket   Batch CSV
                    \         /
                     v       v
                  Preprocessing
                         |
                         v
                Saved TF-IDF Vectorizer
                         |
                         v
                Logistic Regression
                         |
                         v
              Category + Probabilities
                         |
                         v
                 Results & Dashboard
```

**Note:** The application uses the saved model and vectorizer for predictions. No external API calls or separate backend API are required.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| NLP & Text Processing | NLTK |
| Feature Extraction | TF-IDF |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Web Application | Streamlit |
| Model Serialization | Joblib |
| Imbalanced Data Handling | Imbalanced-learn |

## Project Structure

```text
Hackathon HCL DS/
│
├── app/
│   └── app.py
│
├── data/
│   └── processed/
│       ├── final_data.csv
│       └── support_tickets.csv
│
├── models/
│   ├── tfidf_vectorizer.joblib
│   └── ticket_classifier.joblib
│
├── notebooks/
│   └── hcl_it_ticket_final_model.ipynb
│
├── outputs/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "Hackathon HCL DS"
```

### 2. Create a Virtual Environment

```bash
python -m venv hackathon_env
```

Activate it:

**Windows:**
```bash
hackathon_env\Scripts\activate
```

**Linux/macOS:**
```bash
source hackathon_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

From the project root directory:

```bash
streamlit run app/app.py
```

The application will open in your browser.

## Batch Prediction Input

To use batch classification, upload a CSV file containing a column named `Document`.

Example:

| Document |
|---|
| My laptop is not turning on. |
| Unable to access my company account. |
| I need help with my salary slip. |

The application generates predicted categories, confidence scores, and category-wise probabilities. The results can be downloaded as a CSV file.

## Limitations

- Performance depends on the quality and diversity of the training dataset.
- The dataset contains class imbalance, particularly in Purchase and Storage categories.
- Confidence scores represent model-estimated probabilities and are not guaranteed certainty.
- The model is limited to the five categories used during training.
- The application does not currently integrate with external IT service management platforms.

## Future Enhancements

- Expand the dataset, especially for minority categories.
- Explore advanced NLP models such as BERT.
- Add ticket priority prediction.
- Integrate with IT service management platforms.
- Introduce feedback-based model improvement.
- Incorporate historical ticket trends when timestamp data is available.

## Conclusion

The AI-Powered IT Support Ticket Classification System demonstrates how NLP and machine learning can automate IT support ticket categorization. With an 87.83% test accuracy and an interactive Streamlit dashboard, the application provides a practical approach to ticket classification, batch processing, and operational analytics.

---

**Developed as part of the HCLTech Data Science Hackathon.**