import os
import pandas as pd
import joblib

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/quora_questions.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "question_similarity_model.pkl")

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["question1", "question2", "is_duplicate"]
)

df = df[df["is_duplicate"].isin([0, 1])].copy()

df["question1"] = df["question1"].astype(str)
df["question2"] = df["question2"].astype(str)
df["is_duplicate"] = df["is_duplicate"].astype(int)

# Remove empty question pairs
df = df[
    (df["question1"].str.strip() != "") &
    (df["question2"].str.strip() != "")
]

X1 = df["question1"].reset_index(drop=True)
X2 = df["question2"].reset_index(drop=True)
y = df["is_duplicate"].reset_index(drop=True)

print("Usable question pairs:", len(df))

# Split the data before fitting the vectorizer
indices = range(len(df))

train_idx, test_idx = train_test_split(
    list(indices),
    test_size=0.2,
    random_state=42,
    stratify=y
)

q1_train = X1.iloc[train_idx]
q2_train = X2.iloc[train_idx]
q1_test = X1.iloc[test_idx]
q2_test = X2.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("Converting questions into TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    dtype="float32"
)

# Learn vocabulary from training questions only
train_texts = pd.concat(
    [q1_train, q2_train],
    ignore_index=True
)

vectorizer.fit(train_texts)

del train_texts

def make_pair_features(question1, question2, vectorizer):
    v1 = vectorizer.transform(question1)
    v2 = vectorizer.transform(question2)

    # Difference between question vectors
    difference = abs(v1 - v2)

    # Words/features shared by both questions
    shared = v1.multiply(v2)

    return hstack(
        [difference, shared],
        format="csr",
        dtype="float32"
    )

print("Creating training features...")

X_train = make_pair_features(
    q1_train, q2_train, vectorizer
)

print("Training Logistic Regression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear"
)

model.fit(X_train, y_train)

del X_train

print("Evaluating model...")

X_test = make_pair_features(
    q1_test, q2_test, vectorizer
)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    {
        "vectorizer": vectorizer,
        "model": model
    },
    MODEL_PATH
)

print("\nImproved model saved successfully!")
print("Path:", MODEL_PATH)