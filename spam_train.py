import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import joblib
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load tab-separated dataset
df = pd.read_csv(r"./data/SMSSpamCollection",
                 sep="\t",
                 names=["label", "text"],
                 encoding="latin-1")

df["text"] = df["text"].apply(clean_text)

# Map labels to numbers
df["y"] = df["label"].map({"ham": 0, "spam": 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"]
)

# Create pipeline
model = Pipeline([
    ("vect", TfidfVectorizer(stop_words="english",
                             ngram_range=(1,2),
                             min_df=2,
                             sublinear_tf=True)),
    ("clf", MultinomialNB(alpha=0.5))
])

# Train
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, pred))
print("Report:\n", metrics.classification_report(y_test, pred, target_names=["ham", "spam"]))

# Save model
joblib.dump(model, "spam_model.joblib")
print("\nModel saved as spam_model.joblib")
