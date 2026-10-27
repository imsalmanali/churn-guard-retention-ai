"""
generate_data.py
Synthesizes a realistic Telco Customer Churn dataset (3,000 customers)
with nuanced statistical relationships and realistic business features.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000

customer_ids = [f"CUST-{10000 + i}" for i in range(N)]
gender = np.random.choice(["Male", "Female"], size=N)
senior_citizen = np.random.choice([0, 1], size=N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], size=N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], size=N, p=[0.30, 0.70])

# Tenure (months): 1 to 72
tenure = np.random.randint(1, 73, size=N)

# Contract type
contract_choices = ["Month-to-month", "One year", "Two year"]
contract = np.random.choice(contract_choices, size=N, p=[0.55, 0.24, 0.21])

# Internet Service
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=N, p=[0.35, 0.44, 0.21])

# Additional Services
online_security = []
tech_support = []
streaming_tv = []

for s in internet_service:
    if s == "No":
        online_security.append("No internet service")
        tech_support.append("No internet service")
        streaming_tv.append("No internet service")
    else:
        online_security.append(np.random.choice(["Yes", "No"], p=[0.38, 0.62]))
        tech_support.append(np.random.choice(["Yes", "No"], p=[0.39, 0.61]))
        streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))

paperless_billing = np.random.choice(["Yes", "No"], size=N, p=[0.60, 0.40])
payment_methods = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
payment_method = np.random.choice(payment_methods, size=N, p=[0.34, 0.22, 0.22, 0.22])

# Monthly Charges base
monthly_charges = []
for i in range(N):
    base = 20.0
    if internet_service[i] == "DSL":
        base += 25.0
    elif internet_service[i] == "Fiber optic":
        base += 50.0
    
    if online_security[i] == "Yes":
        base += 10.0
    if tech_support[i] == "Yes":
        base += 10.0
    if streaming_tv[i] == "Yes":
        base += 15.0
        
    # Add small variance
    base += np.random.normal(0, 3.5)
    monthly_charges.append(round(max(18.5, base), 2))

monthly_charges = np.array(monthly_charges)
total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 15, size=N), 2)
total_charges = np.maximum(monthly_charges, total_charges)

# Realistic Churn Probability Model
# High risk factors: Month-to-month, Fiber optic, No tech support, Short tenure, Electronic check
z = (
    - 0.05 * tenure
    + 0.02 * monthly_charges
    + (contract == "Month-to-month") * 1.5
    - (contract == "Two year") * 1.8
    + (internet_service == "Fiber optic") * 0.9
    - (tech_support == "Yes") * 0.8
    - (online_security == "Yes") * 0.6
    + (payment_method == "Electronic check") * 0.7
    + senior_citizen * 0.4
    - 1.4
)

prob_churn = 1 / (1 + np.exp(-z))
churn = (np.random.rand(N) < prob_churn).astype(int)

df = pd.DataFrame({
    "customerID": customer_ids,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/customer_churn.csv", index=False)
print(f"Dataset generated successfully: {df.shape[0]} records, Churn rate: {df['Churn'].mean()*100:.1f}%")
