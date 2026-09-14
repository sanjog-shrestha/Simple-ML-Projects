"""
Email Spam Classifier
----------------------
An LSTM-based text classifier that separates spam from legitimate ("ham")
emails. The pipeline covers data loading, class balancing, text cleanup,
tokenization/padding, model training, and evaluation.
"""

import string
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")


@dataclass
class PipelineConfig:
    """Central place for the knobs used throughout the pipeline."""

    csv_path: str = "spam_ham_dataset.csv"
    text_column: str = "text"
    label_column: str = "label"
    max_sequence_len: int = 100
    embedding_dim: int = 32
    lstm_units: int = 16
    dense_units: int = 32
    test_fraction: float = 0.2
    random_seed: int = 42
    batch_size: int = 32
    max_epochs: int = 20
    show_plots: bool = True


def ensure_nltk_resources() -> None:
    """Download the stopword corpus if it isn't already available locally."""
    nltk.download("stopwords", quiet=True)


def load_dataset(cfg: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)
    print(f"Loaded {len(df)} rows from {cfg.csv_path}")
    return df


def plot_class_counts(df: pd.DataFrame, cfg: PipelineConfig, title: str) -> None:
    if not cfg.show_plots:
        return
    sns.countplot(x=cfg.label_column, data=df)
    plt.title(title)
    plt.show()


def balance_classes(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Undersample the majority class so ham/spam counts match."""
    spam_rows = df[df[cfg.label_column] == "spam"]
    ham_rows = df[df[cfg.label_column] == "ham"].sample(
        n=len(spam_rows), random_state=cfg.random_seed
    )
    balanced = pd.concat([ham_rows, spam_rows]).sample(
        frac=1, random_state=cfg.random_seed
    ).reset_index(drop=True)
    return balanced


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def strip_punctuation(text: str) -> str:
    return text.translate(_PUNCT_TABLE)


def clean_text_column(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Remove the leading 'Subject' marker, punctuation, and stopwords."""
    stop_set = set(stopwords.words("english"))

    def _clean(raw: str) -> str:
        no_subject = raw.replace("Subject", "", 1)
        no_punct = strip_punctuation(no_subject)
        tokens = [w.lower() for w in no_punct.split() if w.lower() not in stop_set]
        return " ".join(tokens)

    df = df.copy()
    df[cfg.text_column] = df[cfg.text_column].apply(_clean)
    return df


def render_word_cloud(df: pd.DataFrame, cfg: PipelineConfig, category: str) -> None:
    if not cfg.show_plots:
        return
    corpus = " ".join(df[df[cfg.label_column] == ("spam" if category == "Spam" else "ham")][cfg.text_column])
    cloud = WordCloud(background_color="black", max_words=100, width=800, height=400).generate(corpus)
    plt.figure(figsize=(7, 7))
    plt.imshow(cloud, interpolation="bilinear")
    plt.title(f"Most Frequent Words — {category} Emails", fontsize=15)
    plt.axis("off")
    plt.show()


def vectorize_text(train_texts, test_texts, cfg: PipelineConfig):
    """Fit a tokenizer on training text only, then encode + pad both splits."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts)

    train_ids = pad_sequences(
        tokenizer.texts_to_sequences(train_texts),
        maxlen=cfg.max_sequence_len, padding="post", truncating="post",
    )
    test_ids = pad_sequences(
        tokenizer.texts_to_sequences(test_texts),
        maxlen=cfg.max_sequence_len, padding="post", truncating="post",
    )
    return train_ids, test_ids, tokenizer


def build_model(vocab_size: int, cfg: PipelineConfig) -> tf.keras.Model:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=cfg.embedding_dim, input_length=cfg.max_sequence_len
        ),
        tf.keras.layers.LSTM(cfg.lstm_units),
        tf.keras.layers.Dense(cfg.dense_units, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def train_model(model: tf.keras.Model, train_X, train_Y, test_X, test_Y, cfg: PipelineConfig):
    callbacks = [
        EarlyStopping(patience=3, monitor="val_accuracy", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, verbose=0),
    ]
    return model.fit(
        train_X, train_Y,
        validation_data=(test_X, test_Y),
        epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
    )


def plot_training_curves(history, cfg: PipelineConfig) -> None:
    if not cfg.show_plots:
        return
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Over Training")
    plt.legend()
    plt.show()


def run_pipeline(cfg: PipelineConfig = PipelineConfig()) -> None:
    ensure_nltk_resources()

    raw_df = load_dataset(cfg)
    plot_class_counts(raw_df, cfg, "Original Class Distribution")

    balanced_df = balance_classes(raw_df, cfg)
    plot_class_counts(balanced_df, cfg, "Balanced Class Distribution (Ham vs Spam)")

    cleaned_df = clean_text_column(balanced_df, cfg)

    render_word_cloud(cleaned_df, cfg, "Non-Spam")
    render_word_cloud(cleaned_df, cfg, "Spam")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        cleaned_df[cfg.text_column],
        cleaned_df[cfg.label_column],
        test_size=cfg.test_fraction,
        random_state=cfg.random_seed,
    )

    train_X, test_X, tokenizer = vectorize_text(train_texts, test_texts, cfg)
    train_Y = (train_labels == "spam").astype(int).to_numpy()
    test_Y = (test_labels == "spam").astype(int).to_numpy()

    model = build_model(vocab_size=len(tokenizer.word_index) + 1, cfg=cfg)
    model.summary()

    history = train_model(model, train_X, train_Y, test_X, test_Y, cfg)

    test_loss, test_accuracy = model.evaluate(test_X, test_Y)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

    plot_training_curves(history, cfg)


if __name__ == "__main__":
    run_pipeline()