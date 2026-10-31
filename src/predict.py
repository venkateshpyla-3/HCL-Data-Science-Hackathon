import joblib
from pathlib import Path

from src.preprocessing import clean_text, remove_stopwords


ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "ticket_classifier.joblib"
VECTORIZER_PATH = ROOT_DIR / "models" / "tfidf_vectorizer.joblib"


model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)


def predict_ticket(ticket):

    if not isinstance(ticket, str) or not ticket.strip():
        raise ValueError("Please enter a valid ticket description.")

    # Preprocessing
    cleaned = clean_text(ticket)
    processed = remove_stopwords(cleaned)

    # TF-IDF transformation
    vectorized = tfidf.transform([processed])

    # Prediction
    prediction = model.predict(vectorized)[0]

    probabilities = model.predict_proba(vectorized)[0]

    probability_dict = {
        category: float(probability)
        for category, probability in zip(
            model.classes_,
            probabilities
        )
    }

    confidence = float(probabilities.max())

    return {
        "predicted_category": prediction,
        "confidence": confidence,
        "probabilities": probability_dict
    }