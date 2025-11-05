import joblib, csv

# Load trained model
model = joblib.load("spam_model.joblib")

# Read messages from file and write results
with open("test_messages.txt", "r", encoding="utf-8") as fin, \
     open("test_results.csv", "w", newline="", encoding="utf-8") as fout:
    writer = csv.writer(fout)
    writer.writerow(["message", "pred_label", "prob_spam"])

    for line in fin:
        msg = line.strip()
        if not msg:
            continue
        p = model.predict([msg])[0]
        prob = model.predict_proba([msg])[0][1]
        writer.writerow([msg, "spam" if p == 1 else "ham", round(float(prob), 4)])

print("Created test_results.csv in current folder.")
