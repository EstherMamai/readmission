# 🏥 Hospital Readmission Prediction

## 📘 Project Overview

Hospital readmission is a key quality metric in healthcare systems. Predicting whether a patient will be readmitted after discharge can help hospitals improve care quality, allocate resources efficiently, and reduce unnecessary costs.

This project uses **machine learning** to predict the likelihood of a patient being readmitted based on demographic, medical, and treatment-related features. The goal is to identify **high-risk patients** early and support proactive intervention.

---

## 🎯 Objectives

* Preprocess and encode patient data for machine learning.
* Handle class imbalance in the readmission variable.
* Build and evaluate different models for readmission prediction.
* Optimize the best-performing model for better precision and recall.
* Interpret model results to derive meaningful clinical insights.

---

## 📂 Dataset

The dataset used contains information about **diabetic patients** and includes demographic, medical, and encounter details.

### Key Columns:

* `race`, `gender`, `age` — patient demographics
* `admission_type_id`, `discharge_disposition_id`, `admission_source_id` — hospital encounter details
* `time_in_hospital`, `num_lab_procedures`, `num_medications` — healthcare utilization features
* `change`, `diabetesMed` — treatment-related binary indicators
* `readmitted_flag` — **target variable (1 = readmitted, 0 = not readmitted)**

### Target Distribution:

```
0 (Not Readmitted): 90409
1 (Readmitted):     11357
```

This shows a strong **class imbalance**, which required resampling strategies.

---

## 🧹 Data Preprocessing

### 1. **Encoding**

* **Label Encoding** for binary categorical columns (`change`, `diabetesMed`)
* **One-Hot Encoding** for multi-category columns

```python
binary_cols = ['change', 'diabetesMed']
le = LabelEncoder()
for col in binary_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
```

---

### 2. **Feature Scaling**

Features were standardized using **StandardScaler** to normalize numeric distributions.

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### 3. **Class Imbalance Handling**

Since the dataset was highly imbalanced, two techniques were explored:

* **Class weights** in Logistic Regression and Random Forest.
* **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic samples for the minority class.

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
```

---

## 🤖 Models Tested

| Model                   | Description                      | Notes                                    |
| ----------------------- | -------------------------------- | ---------------------------------------- |
| **Logistic Regression** | Baseline model                   | Good interpretability but limited recall |
| **Random Forest**       | Ensemble model of decision trees | Balanced accuracy and recall             |
| **XGBoost**             | Gradient boosting algorithm      | Slight overfitting observed              |
| **Random Classifier**   | Benchmark                        | Used as a control model                  |

---

## 📊 Model Performance Summary

| Model                            | Accuracy | Precision | Recall   | F1-score | ROC-AUC  |
| -------------------------------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression (SMOTE)      | 0.63     | 0.16      | 0.55     | 0.25     | 0.63     |
| Random Forest (Balanced Weights) | **0.64** | **0.16**  | **0.55** | **0.25** | **0.64** |
| Random Classifier                | 0.89     | 0.64      | 0.00     | 0.01     | 0.50     |

> 🧠 *Random Forest was selected for further optimization due to its strong recall on minority class and balanced overall performance.*

---

## 🔧 Model Optimization

A **GridSearchCV** was applied to tune key Random Forest parameters such as:

* Number of trees (`n_estimators`)
* Tree depth (`max_depth`)
* Minimum samples per split/leaf
* Class weights
* Feature selection strategy (`max_features`)

The goal was to **maximize F1-macro** score to handle class imbalance effectively.

---

## 📈 Next Steps

* [ ] Finalize hyperparameter optimization results
* [ ] Analyze feature importance
* [ ] Visualize confusion matrix and ROC curve
* [ ] Experiment with ensemble or stacking models
* [ ] Deploy model as an API (FastAPI or Flask)

---

## 🧠 Key Insights (So Far)

* Severe **class imbalance** affected baseline models.
* **Balancing techniques (class weights & SMOTE)** significantly improved recall for the minority class.
* **Random Forest** demonstrated the best trade-off between interpretability and predictive power.
* Further tuning is expected to improve recall and AUC for the minority class.

---

## 🧰 Tools & Libraries

* Python (Pandas, NumPy)
* scikit-learn
* imbalanced-learn (SMOTE)
* Matplotlib, Seaborn (Visualization)
* Jupyter Notebook / VS Code
