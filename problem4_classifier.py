"""
SPORTS VS POLITICS CLASSIFIER
This script trains 3 ML models and compares performance.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# ---------- READ DATA ----------
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


sports = read_file("sports.txt")
politics = read_file("politics.txt")

texts = sports + politics
labels = ["sports"] * len(sports) + ["politics"] * len(politics)


# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ---------- TF-IDF ----------
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ---------- MODELS ----------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC()
}

print("\nMODEL COMPARISON RESULTS:\n")

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    print(f"{name} Accuracy: {acc:.2f}")
