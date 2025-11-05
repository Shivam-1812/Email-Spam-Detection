# 📧 Email / SMS Spam Detection using Naive Bayes and NLP

A Machine Learning project that classifies text messages or emails as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and **Multinomial Naive Bayes**.  
Built with **Python**, **scikit-learn**, and a simple **Flask web interface** for demonstration.

---

## 🚀 Project Overview

This project detects whether a given message (email or SMS) is spam or ham based on its textual content.  
It uses the **UCI SMS Spam Collection Dataset** and applies NLP preprocessing, TF-IDF vectorization, and Naive Bayes classification to achieve around 98% accuracy.

### 🔍 Example
| Message | Prediction | Probability (Spam) |
|----------|-------------|--------------------|
| `Congratulations! You won a brand new phone!` | **Spam** | 0.82 |
| `Hey, what time is the class tomorrow?` | **Ham** | 0.02 |

---

## 🧠 Features
- Cleaned and preprocessed text messages  
- TF-IDF vectorization (unigrams + bigrams)  
- Multinomial Naive Bayes model  
- Adjustable threshold for better spam recall  
- Interactive **Flask web app** interface  
- Supports both console and browser testing  
- Model saved as `spam_model.joblib` for reuse  

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python 3 |
| **Machine Learning** | scikit-learn |
| **Data Handling** | pandas, numpy |
| **Web Framework** | Flask |
| **Environment** | Virtualenv |
| **Dataset** | [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) |

---

## 🗂️ Project Structure

Spam-Detector/
│
├── data/
│ ├── SMSSpamCollection
│ └── readme
│
├── spam_train.py # trains and saves the model
├── test_model.py # predicts from a file
├── analyze_errors.py # evaluates and finds misclassified samples
├── demo_threshold.py # console demo with custom threshold
├── app.py # Flask web app
├── spam_model.joblib # trained model
├── test_messages.txt
├── test_results.csv
└── misclassified.csv


---

## ⚙️ Setup & Installation

### 1️⃣ Clone this repository
git clone https://github.com/Shivam-1812/Email-Spam-Detection.git
cd Email-Spam-Detection


### 2️⃣ Create a virtual environment
python -m venv venv


### 3️⃣ Activate the environment
**Windows PowerShell:**
.\venv\Scripts\Activate.ps1


**macOS/Linux:**
source venv/bin/activate

### 4️⃣ Install dependencies
python -m pip install --upgrade pip setuptools
pip install pandas scikit-learn flask joblib numpy

### 5️⃣ Verify dataset
Ensure the file `data/SMSSpamCollection` exists.  
If not, download it from the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) and place it inside the `data/` folder.

---

## 🧮 How It Works
1. **Preprocessing:** Cleans text (lowercase, remove numbers, symbols, and URLs).  
2. **Vectorization:** TF-IDF converts text to numerical vectors.  
3. **Training:** Multinomial Naive Bayes learns spam/ham patterns.  
4. **Prediction:** Outputs probability (`prob_spam`) for each message.  
5. **Decision:** Labels messages using a threshold (default = 0.30).

---

## ▶️ Run the Project

### 🔹 Train the Model
python spam_train.py


### 🔹 Analyze Errors
python analyze_errors.py


### 🔹 Run Console Demo
python demo_threshold.py


### 🔹 Run Flask Web App
python app.py


Then open your browser and go to:  
👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🧾 Example Output (Flask App)

Message: Congratulations! You won a free iPhone!
Prob(spam): 0.82
Label: SPAM

---

## 📊 Model Performance

| Metric | Score |
|--------|--------|
| Accuracy | 98.0% |
| Precision (Spam) | 0.99 |
| Recall (Spam) | 0.86 |
| F1-Score (Spam) | 0.92 |

---

## 🧩 Future Improvements
- Use Logistic Regression or Transformer Models for better recall  
- Add character n-grams for obfuscated spam  
- Integrate with Gmail API for real email testing  
- Deploy on Render / Vercel / Heroku for online use  

---

## 👨‍💻 Author

**Shivam Bande**   
🔗 [GitHub Profile](https://github.com/Shivam-1812)

---

## 📜 License

This project is open source and available under the **MIT License**.

> “Spam filtering isn’t just classification — it’s digital hygiene for the modern world.”  
> — Project by Shivam Bande
