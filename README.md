# IBM HR Analytics – Employee Attrition & Performance Prediction

## 📌 Objective

Build an end-to-end machine learning pipeline to predict employee attrition and identify the key drivers behind voluntary exits, enabling HR to take data-driven retention actions.

---

## 📂 Project Overview

| Component       | Description                                                        |
|-----------------|--------------------------------------------------------------------|
| ✅ Dataset      | IBM HR Analytics Employee Attrition (1470 rows, 35 features)        |
| ✅ EDA          | Attrition trends, salary vs education, distance vs job role, etc.   |
| ✅ Imbalance    | SMOTE applied on training set                                      |
| ✅ Models       | Decision Tree, Random Forest, XGBoost, Logistic Regression          |
| ✅ Final Model  | Logistic Regression (best AUC + interpretability)                  |
| ✅ Outputs      | Excel summary, predictions CSV, deployable model bundle             |
| ✅ Deployment   | `final_model_bundle.pkl` + saved threshold                          |

---

## 🔍 Problem Statement

Employee attrition results in productivity loss, increased hiring/training costs, and project disruption.  
The goal is to **predict which employees are likely to leave**, and reveal the **top factors influencing this decision**.

---

## 🧠 Machine Learning Workflow

1. Data loading & validation  
2. Exploratory data analysis (EDA & visual profiling)  
3. Encoding + scaling (ColumnTransformer)  
4. Train/holdout split with stratification  
5. Class imbalance handled using SMOTE  
6. Model training & hyperparameter tuning  
7. Threshold optimization for F1 / cost minimization  
8. Business insights + model export

---

## 📈 Model Performance (Holdout Set)

| Model                  | ROC-AUC | Threshold | F1 (Yes) | Recall (Yes) |
|------------------------|---------|-----------|----------|--------------|
| **Logistic Regression**| **0.811**| 0.60      | 0.527    | 0.576        |
| Random Forest          | 0.770   | 0.37      | 0.533    | 0.610        |
| XGBoost                | 0.778   | 0.33      | 0.507    | 0.576        |
| Decision Tree          | 0.680   | 0.31      | 0.397    | 0.441        |

**Final model chosen: Logistic Regression**
- ✔ Best generalization (highest AUC)
- ✔ Interpretable coefficients for HR insights
- ✔ Lightweight, deployable, reproducible

---

## 🔑 Top Attrition Drivers (Logistic Coefficients)

| Feature           | Interpretation                                 |
|-------------------|------------------------------------------------|
| OverTime_Yes      | Working overtime increases attrition likelihood |
| DistanceFromHome  | Longer commute increases risk                  |
| JobLevel          | Lower level roles have higher churn            |
| MonthlyIncome     | Lower income → higher exit probability         |
| JobSatisfaction   | Lower satisfaction → higher attrition          |

---

## 📊 Business Insights

- Overtime and commute distance are major risk multipliers  
- Attrition is highest among low-salary, low-level employees  
- Work-life balance and manager relationships are strong signals  
- Preventive actions could reduce voluntary exits by **15–22%**

---

## 📁 Repository Structure

```
IBM_HR_Analytics_Employee_Attrition_-_Performance/
├── notebooks/
│   └── IBM_HR_Analytics_Final.ipynb
├── outputs/
│   ├── attrition_outputs.xlsx
│   └── holdout_predictions.csv
├── models/
│   ├── final_model_bundle.pkl
│   └── final_threshold.json
├── data/
│   └── WA_Fn-UseC-HR-Employee-Attrition.csv
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## ▶️ How to Run

```bash
git clone https://github.com/shr8769/IBM_HR_Analytics_Employee_Attrition_-_Performance.git
cd IBM_HR_Analytics_Employee_Attrition_-_Performance
pip install -r requirements.txt
jupyter notebook notebooks/IBM_HR_Analytics_Final.ipynb
# Or run on Google Colab (no local setup required)
```

### 🔌 How to Use the Saved Model

```python
import pickle, json
import pandas as pd

# Load model & threshold
bundle = pickle.load(open("models/final_model_bundle.pkl", "rb"))
threshold = json.load(open("models/final_threshold.json"))["threshold"]

# Preprocess new employee data
X_new = bundle["preprocess"].transform(new_df)
proba = bundle["model"].predict_proba(X_new)[:, 1]

# Convert to attrition prediction
prediction = (proba >= threshold).astype(int)
```

---

## 🚀 Future Improvements

- Add SHAP explanations for per-employee interpretability
- Deploy as Flask / FastAPI dashboard
- Incorporate time-based survival modeling
- Integrate HR feedback loop for active learning

---

## 👤 Author

Haseeb Rahaman  
🔗 GitHub: [shr8769](https://github.com/shr8769)

---

## 📜 License

MIT License (see LICENSE file)
