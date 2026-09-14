import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Config
DATA_PATH = "spam.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 5
EMBED_DIM = 128
USE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

# Step 1: Load & clean data
df = pd.read_csv(DATA_PATH, encoding="latin-1").iloc[:, :2]
df.columns = ["label", "text"]
df["label_enc"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label_enc"], test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

# Step 2: Dataset stats
avg_words_len = round(sum(len(t.split()) for t in df["text"]) / len(df))
approx_vocab_size = len(set(" ".join(df["text"]).split()))
print(f"Train: {len(X_train)} | Test: {len(X_test)} | Avg words: {avg_words_len} | Vocab: {approx_vocab_size}")

# Step 3: Text vectorization (fit on train only)
text_vec = TextVectorization(
    max_tokens=approx_vocab_size,
    standardize="lower_and_strip_punctuation",
    output_mode="int",
    output_sequence_length=avg_words_len,
)
text_vec.adapt(X_train)
VOCAB_SIZE = text_vec.vocabulary_size()

# Step 4: Model architectures
def build_dense_model():
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    x = text_vec(inputs)
    x = layers.Embedding(VOCAB_SIZE, EMBED_DIM)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, out, name="Dense_Embedding")


def build_bilstm_model():
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    x = text_vec(inputs)
    x = layers.Embedding(VOCAB_SIZE, EMBED_DIM)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, out, name="Bi_LSTM")


def build_use_model():
    use_layer = hub.KerasLayer(USE_URL, trainable=False, input_shape=[], dtype=tf.string, name="USE")
    inputs = layers.Input(shape=[], dtype=tf.string)
    x = layers.Lambda(lambda t: use_layer(t), output_shape=(512,))(inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, out, name="USE_Transfer")

# Step 5: Train / evaluate helpers
def compile_and_fit(model, epochs=EPOCHS):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)


def evaluate(model, X, y):
    preds = np.round(model.predict(X, verbose=0)).astype(int)
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1-score": f1_score(y, preds),
    }

# Step 6: Train all models
model_builders = {
    "Dense Embedding": build_dense_model,
    "Bi-LSTM": build_bilstm_model,
    "Transfer Learning (USE)": build_use_model,
}

results = {}
for name, build_fn in model_builders.items():
    print(f"\n=== Training: {name} ===")
    model = build_fn()
    compile_and_fit(model)
    results[name] = evaluate(model, X_test, y_test)

# Step 7: Results table
results_df = pd.DataFrame(results).T
print("\nPerformance Table:")
print(results_df)

# Step 8: Visualize comparisons
results_df.plot(kind="bar", figsize=(10, 6), ylim=(0.8, 1.0), title="Model Performance Metrics")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

plt.figure(figsize=(10, 6))
for name in results_df.index:
    plt.plot(results_df.columns, results_df.loc[name], marker="o", linewidth=2, label=name)
plt.title("Model Performance Trends")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.ylim(0.8, 1.0)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()