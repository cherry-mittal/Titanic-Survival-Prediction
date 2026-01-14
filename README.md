
# 🚢 Titanic Survival Prediction using Machine Learning

## 📌 Project Overview

The project builds a **machine learning model to predict passenger survival on the Titanic** using demographic and travel-related features. It is based on the classic Kaggle Titanic dataset and demonstrates an **end‑to‑end ML workflow**, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and prediction.

---

## 🎯 Problem Statement

Given passenger information such as age, gender, ticket class, fare, and family size, predict whether a passenger **survived (1)** or **did not survive (0)** the Titanic disaster.

This is a **binary classification problem**.

---

## 📂 Dataset

* **Source:** Kaggle – Titanic: Machine Learning from Disaster
* **Training set:** 891 rows × 12 columns
* **Test set:** 418 rows × 11 columns

### Key Features

| Feature  | Description                       |
| -------- | --------------------------------- |
| Pclass   | Passenger class (1st, 2nd, 3rd)   |
| Sex      | Gender of passenger               |
| Age      | Age in years                      |
| SibSp    | Number of siblings/spouses aboard |
| Parch    | Number of parents/children aboard |
| Fare     | Ticket fare                       |
| Embarked | Port of embarkation (C, Q, S)     |

Target Variable:

* **Survived** → `1 = Survived`, `0 = Did not survive`

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed using:

* Distribution plots
* Survival rate comparisons
* Missing value analysis

---

## 🛠️ Data Preprocessing

The following preprocessing steps were applied:

* Missing value imputation (Age, Embarked)
* Dropping irrelevant features (e.g., PassengerId, Name, Ticket)
* Encoding categorical variables (Sex, Embarked)
* Feature scaling (where required)
* Creating derived features such as **FamilySize**

---

## 🤖 Models Used

Multiple machine learning models were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

The final model was selected based on **accuracy, precision, recall, and cross‑validation performance**.

---

## 📊 Model Evaluation

Evaluation metrics used:

* Accuracy
* Confusion Matrix
* Precision, Recall, F1‑Score
* Cross‑Validation Score

Example Results:

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~78%     |
| Random Forest       | ~82–85%  |

*(Exact results may vary depending on hyperparameters and random seed.)*

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Training Script

```bash
python train.py
```

### 5️⃣ Generate Predictions

```bash
python predict.py
```

---

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧪 Future Improvements

* Hyperparameter tuning using GridSearchCV / Optuna
* Model explainability using SHAP or LIME
* Pipeline integration with `sklearn.pipeline`
* Deployment using Flask / FastAPI
* Experiment tracking with MLflow

---

## 📌 Key Learnings

* Importance of feature engineering in classical ML
* Handling missing and categorical data effectively
* Comparing multiple models instead of relying on one
* Evaluating models beyond accuracy

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

* Kaggle Titanic Dataset
* Scikit‑learn documentation
* Open‑source ML community

---

## 👤 Author

**Cherry Mittal**
Machine Learning Enthusiast | AI & Data Science

If you find this project helpful, feel free to ⭐ the repository!


