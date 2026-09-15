"""
Text Classification with Naive Bayes
-------------------------------------
A bag-of-words text classifier using CountVectorizer + Multinomial Naive
Bayes, trained on a labeled text dataset. Covers data loading, vectorization,
training, evaluation, and a confusion matrix visualization.
"""

from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    csv_path: str = "synthetic_text_data.csv"
    text_column: str = "text"
    label_column: str = "label"
    test_fraction: float = 0.2
    random_seed: int = 42


def load_dataset(cfg: PipelineConfig) -> pd.DataFrame:
    """Load the labeled dataset from disk."""
    df = pd.read_csv(cfg.csv_path)
    print(f"Loaded {len(df)} rows from {cfg.csv_path}")
    return df


def vectorize_text(train_texts: pd.Series, test_texts: pd.Series):
    """Fit a bag-of-words vectorizer on training text only, then encode both splits."""
    vectorizer = CountVectorizer()
    train_vec = vectorizer.fit_transform(train_texts)
    test_vec = vectorizer.transform(test_texts)
    return train_vec, test_vec, vectorizer


def train_model(X_train, y_train) -> MultinomialNB:
    """Train a Multinomial Naive Bayes classifier."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: MultinomialNB, X_test, y_test) -> dict:
    """Generate predictions and compute evaluation metrics."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "y_pred": y_pred,
    }


def plot_confusion_matrix(model: MultinomialNB, y_test, y_pred) -> None:
    """Render a confusion matrix heatmap for predicted vs. actual labels."""
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=model.classes_, yticklabels=model.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    """Execute the full training and evaluation pipeline."""
    df = load_dataset(cfg)
    X, y = df[cfg.text_column], df[cfg.label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_fraction, random_state=cfg.random_seed
    )

    X_train_vec, X_test_vec, _ = vectorize_text(X_train, X_test)

    model = train_model(X_train_vec, y_train)

    metrics = evaluate_model(model, X_test_vec, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    plot_confusion_matrix(model, y_test, metrics["y_pred"])


if __name__ == "__main__":
    run_pipeline()