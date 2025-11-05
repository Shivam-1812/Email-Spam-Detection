import joblib, csv, pandas as pd
from sklearn import metrics

model = joblib.load("spam_model.joblib")

# Load original dataset used for testing
df = pd.read_csv(r"./data/SMSSpamCollection", sep="\t", names=["label", "text"], encoding="latin-1")
df["y"] = df["label"].map({"ham":0, "spam":1})

# Split same way as training script to get the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"])

# Predictions
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:,1]

# Metrics and confusion
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", metrics.classification_report(y_test, y_pred, target_names=["ham","spam"]))

# Save misclassified examples
rows = []
for text, true, pred, prob in zip(X_test, y_test, y_pred, probs):
    if true != pred:
        rows.append([text, "spam" if true==1 else "ham", "spam" if pred==1 else "ham", round(float(prob),4)])

mis_df = pd.DataFrame(rows, columns=["text","true_label","pred_label","prob_spam"])
mis_df.to_csv("misclassified.csv", index=False, encoding="utf-8")
print(f"\nSaved {len(mis_df)} misclassified examples to misclassified.csv")
