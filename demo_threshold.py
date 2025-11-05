# demo_threshold.py
import joblib

# Load trained model (ensure spam_model.joblib is in the same folder)
model = joblib.load("spam_model.joblib")

# Thresholds (tune for demo)
THRESHOLD = 0.30     # >= this => SPAM
SUSPECT_LOWER = 0.20 # >= this and < THRESHOLD => SUSPECT

def label_from_prob(p):
    if p >= THRESHOLD:
        return "SPAM"
    if p >= SUSPECT_LOWER:
        return "SUSPECT"
    return "HAM"

def main():
    print("=== Spam Detection Demo (threshold-aware) ===")
    print(f"Using THRESHOLD={THRESHOLD}, SUSPECT_LOWER={SUSPECT_LOWER}")
    print("Type a message and press Enter. Empty line to quit.\n")

    try:
        while True:
            msg = input("Message: ").strip()
            if not msg:
                break
            prob = float(model.predict_proba([msg])[0][1])
            label = label_from_prob(prob)
            print(f"Prob(spam) = {prob:.3f}  →  {label}\n")
    except KeyboardInterrupt:
        print("\nExiting demo.")

if __name__ == "__main__":
    main()
