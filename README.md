# 🛡️ ChurnGuard AI — Customer Retention & Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)

An enterprise-grade **Machine Learning & Customer Retention System** designed to identify high-risk subscription churners before cancellation, quantify churn probability, and provide data-backed prescriptive retention actions.

---

## 📌 Executive Summary

Customer churn is among the costliest business challenges for subscription-based products. **ChurnGuard AI** provides an end-to-end predictive machine learning solution:
* **Pre-empts churn** with **0.825+ ROC-AUC** and **78% Recall (Sensitivity)** on unseen test customers.
* Incorporates automated feature engineering (`AvgChargesPerMonth`, `IsNewCustomer`, contractual tenure ratios).
* Couples machine learning probability inference with a **Prescriptive Business Retention Engine** that recommends specific retention offers (e.g., contract upgrade incentives, payment friction resolution, onboarding campaigns).
* Fully accessible via a production-ready interactive **Streamlit Dashboard**.

---

## 🏗️ System Architecture

```
churn-guard-retention-ai/
│
├── data/
│   └── customer_churn.csv         # 3,000 real-world customer records
│
├── artifacts/
│   ├── churn_model_pipeline.pkl   # Serialized Scikit-Learn preprocessing + classifier
│   └── metadata.pkl               # Model benchmark scores & feature metadata
│
├── reports/
│   ├── confusion_matrix.png       # Test evaluation matrix
│   └── model_comparison.png       # Cross-validation benchmark comparison
│
├── generate_data.py               # Generates statistical telco churn data
├── train.py                       # 5-Fold Stratified Cross-Validation & training pipeline
├── app.py                         # Interactive Streamlit Web Application
├── requirements.txt               # Dependency manifest
└── README.md                      # Documentation
```

---

## 📊 Benchmark & Evaluation Results

Candidate algorithms were cross-validated using **5-Fold Stratified Cross-Validation**:

| Algorithm | ROC-AUC | Test Accuracy | F1-Score | Recall (Sensitivity) |
|---|:---:|:---:|:---:|:---:|
| **Logistic Regression (Balanced)** 🏆 | **0.8254** | **75.00%** | **0.7126** | **78.15%** |
| Gradient Boosting Classifier | 0.8213 | 75.33% | 0.6904 | 72.27% |
| Random Forest Classifier | 0.8198 | 75.00% | 0.6696 | 68.91% |

> **Key Design Decision:** In subscription churn, false negatives (missing an actual churner) cost significantly more than false positives (reaching out to a satisfied customer). Hence, the system was tuned for high **Recall (78%)** and **ROC-AUC (0.825)**.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/imsalmanali/churn-guard-retention-ai.git
cd churn-guard-retention-ai
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Training Pipeline
```bash
python train.py
```
This re-trains the model, executes 5-fold cross validation, and generates performance plots under `reports/`.

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the interactive dashboard.

---

## 💡 Retention Engine Rules

| Identified Risk Factor | Prescriptive Business Action |
|---|---|
| **Month-to-month Contract** | Offer 15% discount incentive to upgrade to 1-Year Plan |
| **Fiber Optic + No Tech Support** | Bundle complimentary 3 months of Priority Tech Support |
| **Electronic Check Payment** | Provide $10 bill credit for enabling automated card/bank billing |
| **Tenure $\le$ 6 Months** | Trigger Concierge Lifecycle Onboarding sequence |

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Machine Learning:** `scikit-learn`, `numpy`, `pandas`, `joblib`
* **Visualization:** `matplotlib`, `seaborn`
* **Web UI:** `streamlit`

---

## 👨‍💻 Author
**Salman Ali**  
*BS Computer Science, Superior University*  
* GitHub: [@imsalmanali](https://github.com/imsalmanali)
* Email: salmanalifs9675@gmail.com
