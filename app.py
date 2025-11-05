# app.py (threshold-aware)
from flask import Flask, render_template_string, request
import joblib

app = Flask(__name__)
model = joblib.load("spam_model.joblib")

THRESHOLD = 0.30   # messages with prob_spam >= THRESHOLD are labeled SPAM
SUSPECT_LOWER = 0.20  # optional: show 'suspect' range for teaching

html = """
<h2>Email/Message Spam Detector</h2>
<form method="post">
  <textarea name="msg" rows="6" cols="80" placeholder="Type or paste a message"></textarea><br><br>
  <input type="submit" value="Check">
</form>

{% if msg %}
  <h3>Message:</h3>
  <pre style="white-space:pre-wrap">{{ msg }}</pre>
  <h3>Result</h3>
  <p>Prob(spam): <strong>{{ prob }}</strong></p>
  <p>Label: <strong style="color:{{ color }}">{{ label }}</strong></p>
  <p style="font-size:0.9em; color:#444">{{ note }}</p>
{% endif %}
"""

def decide(prob):
    if prob >= THRESHOLD:
        return "SPAM", "red", "Above threshold — treat as spam."
    if prob >= SUSPECT_LOWER:
        return "SUSPECT", "orange", "Low-confidence. Manual review recommended."
    return "HAM", "green", "Below suspect threshold."

@app.route("/", methods=["GET","POST"])
def index():
    msg = None
    prob = None
    label = None
    color = "black"
    note = None
    if request.method == "POST":
        msg = request.form.get("msg","").strip()
        if msg:
            prob_val = float(model.predict_proba([msg])[0][1])
            prob = f"{prob_val:.3f}"
            label, color, note = decide(prob_val)
    return render_template_string(html, msg=msg, prob=prob, label=label, color=color, note=note)

if __name__ == "__main__":
    app.run(debug=True)
