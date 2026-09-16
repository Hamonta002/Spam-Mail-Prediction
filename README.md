# Spam-Mail-Prediction

# Spam Mail Prediction Using Machine Learning 📧

A machine learning project that classifies email messages as **Spam** or **Ham (Non-Spam)** using Natural Language Processing (NLP) and supervised machine learning techniques.

The project starts with a **Logistic Regression** baseline and is later improved using a **Linear Support Vector Machine (LinearSVC)** with TF-IDF n-gram features and hyperparameter tuning using `GridSearchCV`.

The improved model achieves **98.92% accuracy** on the test dataset.

---

## 🚀 Live Demo

[Try the Spam Mail Detector](https://spam-mail-prediction-o4z2fdkpev5nh6fel3ky7t.streamlit.app/)

## 📌 Project Overview

Spam emails are unwanted messages that may contain advertisements, scams, fraudulent links, or other potentially harmful content.

This project uses machine learning to automatically classify an email message as either:

- **Spam (0)** – unwanted or suspicious email
- **Ham (1)** – legitimate email

The project was initially developed by following a machine learning tutorial and was later extended with a more advanced classification pipeline.

The improved implementation includes:

- TF-IDF feature extraction
- Word n-grams
- Linear Support Vector Machine
- Scikit-Learn Pipeline
- GridSearchCV hyperparameter tuning
- Classification report
- Confusion matrix
- Model serialization using Joblib

---

## 📊 Dataset

The project uses a labeled email dataset containing two main columns:

| Column     | Description                  |
| ---------- | ---------------------------- |
| `Category` | Email class: `spam` or `ham` |
| `Message`  | Email/message text           |

The dataset contains **5,572 email messages**.

The notebook uses an **80/20 train-test split** with stratification:

```python
train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=3,
    stratify=y
)
```

The resulting test set contains **1,115 messages**.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas** – data loading and manipulation
- **NumPy** – numerical operations
- **Scikit-Learn** – machine learning and evaluation
- **TF-IDF Vectorizer** – text feature extraction
- **Logistic Regression** – baseline model
- **LinearSVC** – improved classification model
- **Pipeline** – combining preprocessing and classification
- **GridSearchCV** – hyperparameter tuning
- **Matplotlib** – visualization
- **Seaborn** – confusion matrix visualization
- **Joblib** – model saving
- **Google Colab** – development environment

---

## 🔄 Machine Learning Workflow

The overall workflow is:

```text
                    Email Dataset
                         │
                         ▼
                   Data Cleaning
                         │
                         ▼
                   Label Encoding
                         │
                         ▼
                  Train/Test Split
                         │
                         ▼
                  TF-IDF Features
                         │
                         ▼
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
     Logistic Regression       Linear SVM Pipeline
        (Baseline)             + TF-IDF N-Grams
             │                       │
             ▼                       ▼
       Model Evaluation        GridSearchCV
                                     │
                                     ▼
                              Best Performing Model
                                     │
                                     ▼
                              Spam / Ham Prediction
```

---

# 1️⃣ Baseline Model — Logistic Regression

The initial model uses:

```text
Email Text
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Spam / Ham
```

The notebook uses:

```python
TfidfVectorizer(
    min_df=1,
    stop_words='english',
    lowercase=True
)
```

The resulting TF-IDF training matrix contains **4,457 training messages and 7,493 features**.

The Logistic Regression model achieved:

**Test Accuracy: 97.13%**

---

# 2️⃣ Improved Model — Linear SVM

The baseline approach was improved using a Scikit-Learn Pipeline containing:

```text
TF-IDF Vectorizer
       ↓
LinearSVC
```

The pipeline is defined using:

```python
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        lowercase=True
    )),

    ('clf', LinearSVC(
        dual=False,
        random_state=3
    ))
])
```

This allows the text transformation and classification steps to be treated as one complete machine learning workflow.

---

## 🔤 TF-IDF N-Gram Features

The improved model tests two different n-gram configurations:

```text
(1,1) → Unigrams
(1,2) → Unigrams + Bigrams
```

The parameter grid used in the notebook is:

```python
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1.0, 10.0]
}
```

This allows the model to compare individual words against combinations of one- and two-word sequences.

---

## ⚙️ Hyperparameter Tuning

`GridSearchCV` is used with:

```python
cv=5
```

and:

```python
scoring='f1_macro'
```

The model tests different values of the SVM regularization parameter:

```text
C = 0.1
C = 1.0
C = 10.0
```

The best configuration found by the search was:

```text
C = 10.0
N-Gram Range = (1, 2)
```

In other words, the best model uses **unigrams + bigrams** with `C=10.0`.

---

# 📈 Model Performance

The improved model achieved:

```text
Accuracy: 98.92%
```

on the test dataset.

### Model Comparison

| Model                     | Test Accuracy |
| ------------------------- | ------------: |
| Logistic Regression       |    **97.13%** |
| Linear SVM + GridSearchCV |    **98.92%** |

### Improvement

The improved Linear SVM model increased test accuracy by:

**1.79 percentage points**

compared with the original Logistic Regression model.

---

# 📋 Classification Report

The final model produced the following results on the test set:

| Class        | Precision | Recall | F1-Score |  Support |
| ------------ | --------: | -----: | -------: | -------: |
| Spam (0)     |      0.99 |   0.93 |     0.96 |      149 |
| Ham (1)      |      0.99 |   1.00 |     0.99 |      966 |
| **Accuracy** |           |        | **0.99** | **1115** |
| Macro Avg    |      0.99 |   0.97 |     0.98 |     1115 |
| Weighted Avg |      0.99 |   0.99 |     0.99 |     1115 |

The Spam class has a recall of **0.93**, meaning the model correctly identifies most of the spam messages in the test set.

---

# 📊 Confusion Matrix

The final model produced the following confusion matrix:

```text
[[139  10]
 [  2 964]]
```

Using the class order:

```text
Spam = 0
Ham  = 1
```

this corresponds to:

|                 | Predicted Spam | Predicted Ham |
| --------------- | -------------: | ------------: |
| **Actual Spam** |            139 |            10 |
| **Actual Ham**  |              2 |           964 |

So, out of 149 Spam messages:

- **139** were correctly classified as Spam
- **10** were classified as Ham

And out of 966 Ham messages:

- **964** were correctly classified as Ham
- **2** were classified as Spam

The notebook also generates a Seaborn heatmap to visualize this confusion matrix.

---

# 🧪 Example Prediction

The final pipeline can directly classify new email messages.

Example:

```python
custom_emails = [
    "URGENT: Your bank account has been locked. Click here to verify your identity.",
    "Hey, are we still on for lunch tomorrow at 12?"
]

predictions = best_model.predict(custom_emails)

for text, pred in zip(custom_emails, predictions):
    label = "Ham" if pred == 1 else "Spam"
    print(f"[{label}] - {text}")
```

The notebook demonstrates prediction on custom messages using the trained `best_model`.

> **Note:** Machine learning predictions depend on patterns learned from the training dataset, so unusual or previously unseen messages may sometimes be misclassified.

---

# 💾 Saving the Trained Model

After finding the best configuration, the trained pipeline is saved using Joblib:

```python
model_filename = 'advanced_spam_classifier.joblib'

joblib.dump(best_model, model_filename)
```

The saved file is:

```text
advanced_spam_classifier.joblib
```

This makes it possible to reuse the trained pipeline without retraining the model from scratch.

---

# 📂 Project Structure

```text
Spam-Mail-Prediction/
│
├── Spam_Mail_Prediction_using_Machine_Learning.ipynb
│   └── Complete machine learning workflow
│
├── mail_data.csv
│   └── Email dataset
│
├── advanced_spam_classifier.joblib
│   └── Saved trained Linear SVM pipeline
│
├── app.py
│   └── Application for using the trained model
│
├── requirements.txt
│   └── Required Python packages
│
├── README.md
│
└── .gitignore
```

---

# ▶️ How to Run the Project

## Using Google Colab

1. Open the `.ipynb` file in Google Colab.
2. Upload `mail_data.csv`.
3. Run the notebook cells from top to bottom.
4. Train the Logistic Regression baseline model.
5. Run the upgraded Linear SVM section.
6. Allow `GridSearchCV` to find the best parameters.
7. Review the classification report and confusion matrix.
8. Test the model with custom email messages.

---

## Using Jupyter Notebook

Clone the repository:

```bash
git clone https://github.com/Hamonta002/Spam-Mail-Prediction.git
```

Move into the project directory:

```bash
cd Spam-Mail-Prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Spam_Mail_Prediction_using_Machine_Learning.ipynb
```

Run the cells in order.

---

# 🌐 Streamlit Application

The repository also contains an `app.py` application for using the saved trained model.

After installing the dependencies, run:

```bash
streamlit run app.py
```

The application can be used to enter email text and obtain a Spam/Ham prediction using the saved model.

---

# 🚀 Future Improvements

Possible improvements for this project include:

- Testing additional machine learning algorithms
- Character-level TF-IDF features
- More advanced text preprocessing
- Feature engineering
- Handling class imbalance
- Experimenting with different classifiers
- Improving the Streamlit interface
- Adding prediction confidence
- Deploying the application online
- Building a REST API
- Adding automated model retraining
- Adding unit tests
- Adding model monitoring

---

# 🙏 Acknowledgements

The initial version of this project was developed by following a machine learning tutorial by **Siddhardhan**.

Original tutorial:

[Spam Mail Prediction Using Machine Learning — Siddhardhan](https://www.youtube.com/watch?v=rxkGItX5gGE&utm_source=chatgpt.com)

The project was subsequently extended with:

- Linear SVM
- TF-IDF n-gram features
- Scikit-Learn Pipeline
- GridSearchCV
- F1-based hyperparameter selection
- Classification report
- Confusion matrix
- Model serialization using Joblib

---

# 👨‍💻 Author

**Hamonta**

This project is part of my machine learning learning journey and demonstrates the process of building, evaluating, and improving an **NLP-based text classification model**.
