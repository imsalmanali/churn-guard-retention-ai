"""
train.py
End-to-End ML Training Pipeline:
1. Data loading and feature engineering (Charges per month ratio, tenure buckets).
2. Preprocessing via Scikit-Learn ColumnTransformer and Pipeline.
3. Model Training & Comparison:
   - Logistic Regression (Balanced)
   - Random Forest Classifier
   - Gradient Boosting Classifier
4. Evaluation: Accuracy, ROC-AUC, F1-Score, Precision, Recall, Confusion Matrix.
5. Serializing the winning pipeline to artifacts/.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

def run_pipeline():
    print("=" * 65)
    print("  ChurnGuard AI — Production Machine Learning Training Pipeline")
    print("=" * 65)

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1. Load Data
    data_path = "data/customer_churn.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run generate_data.py first.")

    df = pd.read_csv(data_path)
    print(f"\n[*] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[*] Overall Churn Rate: {df['Churn'].mean() * 100:.2f}%")

    # Feature Engineering
    df["AvgChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1e-5)
    df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)

    categorical_features = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "InternetService", "OnlineSecurity", "TechSupport",
        "StreamingTV", "Contract", "PaperlessBilling", "PaymentMethod"
    ]
    numerical_features = ["tenure", "MonthlyCharges", "TotalCharges", "AvgChargesPerMonth", "IsNewCustomer"]

    X = df[categorical_features + numerical_features]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 2. Build Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features)
        ]
    )

    # 3. Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "-" * 65)
    print("  Model Benchmark Comparison (5-Fold Stratified Cross-Validation)")
    print("-" * 65)

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        cv_roc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        results[name] = {
            "pipeline": pipe,
            "cv_roc_mean": cv_roc.mean(),
            "cv_roc_std": cv_roc.std(),
            "test_accuracy": acc,
            "test_roc_auc": roc,
            "test_f1": f1,
            "test_precision": prec,
            "test_recall": rec,
            "y_pred": y_pred
        }

        print(f"{name:<22} | ROC-AUC: {roc:.4f} (CV: {cv_roc.mean():.4f} +/- {cv_roc.std():.3f}) | Acc: {acc:.4f} | F1: {f1:.4f}")

    # 4. Pick Best Performer by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["test_roc_auc"])
    best_info = results[best_name]
    print("-" * 65)
    print(f"\n[+] Champion Model: {best_name}")
    print(f"    Test ROC-AUC  : {best_info['test_roc_auc']:.4f}")
    print(f"    Test Accuracy : {best_info['test_accuracy']:.4f}")
    print(f"    Test F1-Score : {best_info['test_f1']:.4f}")

    print("\n[*] Detailed Classification Report:")
    print(classification_report(y_test, best_info["y_pred"], target_names=["Retained (0)", "Churned (1)"]))

    # 5. Visualizations
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_info["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained", "Churned"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    plt.title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=180)
    plt.close()

    # Model Benchmark Chart
    model_names = list(results.keys())
    auc_scores = [results[m]["test_roc_auc"] for m in model_names]
    f1_scores = [results[m]["test_f1"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width/2, auc_scores, width, label="ROC-AUC", color="#1B365D")
    ax.bar(x + width/2, f1_scores, width, label="F1-Score", color="#DD8452")
    ax.set_ylabel("Score")
    ax.set_title("Machine Learning Algorithm Comparison", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reports/model_comparison.png", dpi=180)
    plt.close()

    # 6. Save Pipeline & Metadata
    joblib.dump(best_info["pipeline"], "artifacts/churn_model_pipeline.pkl")
    
    metadata = {
        "best_model": best_name,
        "metrics": {
            "roc_auc": round(float(best_info["test_roc_auc"]), 4),
            "accuracy": round(float(best_info["test_accuracy"]), 4),
            "f1": round(float(best_info["test_f1"]), 4),
            "precision": round(float(best_info["test_precision"]), 4),
            "recall": round(float(best_info["test_recall"]), 4)
        },
        "features": {
            "categorical": categorical_features,
            "numerical": numerical_features
        }
    }
    joblib.dump(metadata, "artifacts/metadata.pkl")

    print("[+] Model pipeline serialized to 'artifacts/churn_model_pipeline.pkl'")
    print("[+] Performance plots saved to 'reports/' directory")
    print("\n[V] Training pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
