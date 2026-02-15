"""
SPORTS VS POLITICS CLASSIFIER
This script trains 3 ML models and compares performance.
"""

# importing TF-IDF vectorizer to convert text into numerical features
from sklearn.feature_extraction.text import TfidfVectorizer

# used to split dataset into training and testing sets
from sklearn.model_selection import train_test_split

# used to calculate accuracy of models
from sklearn.metrics import accuracy_score

# importing the three machine learning models we want to compare
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# ---------- READ DATA ----------
def read_file(path):
    # this function reads a text file and returns each line as a sample
    # stripping removes extra spaces and blank lines are ignored
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# reading sports and politics datasets from text files
sports = read_file("sports.txt")
politics = read_file("politics.txt")

# combining both datasets into one list of texts
texts = sports + politics

# creating labels: sports samples labeled as "sports" and politics as "politics"
labels = ["sports"] * len(sports) + ["politics"] * len(politics)


# ---------- SPLIT ----------
# splitting the dataset into training and testing parts
# 80% training and 20% testing, random_state ensures same split every run
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ---------- TF-IDF ----------
# creating TF-IDF vectorizer to convert text into numerical vectors
vectorizer = TfidfVectorizer()

# fitting on training data and transforming it into feature vectors
X_train_vec = vectorizer.fit_transform(X_train)

# transforming test data using the same fitted vectorizer
X_test_vec = vectorizer.transform(X_test)


# ---------- MODELS ----------
# defining a dictionary of models to compare
# using three different algorithms to see how they perform
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC()
}

# printing header for results
print("\nMODEL COMPARISON RESULTS:\n")

# looping through each model to train and evaluate
for name, model in models.items():
    # training the model using training data
    model.fit(X_train_vec, y_train)

    # predicting labels for test data
    preds = model.predict(X_test_vec)

    # calculating accuracy by comparing predictions with actual labels
    acc = accuracy_score(y_test, preds)

    # printing model name and accuracy rounded to 2 decimal places
    print(f"{name} Accuracy: {acc:.2f}")
