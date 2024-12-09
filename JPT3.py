import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

data = {
    'Age': [25, 45, 35, 50, 30, 40, 60, 28, 42, 36],
    'Income': [4000, 8000, 6000, 10000, 4500, 7500, 11000, 4200, 7000, 6400],
    'LoanAmount': [15000, 20000, 12000, 30000, 10000, 25000, 40000, 13000, 24000, 20000],
    'LoanTerm': [36, 60, 48, 72, 36, 60, 84, 48, 60, 36],
    'CreditScore': [600, 750, 700, 800, 650, 720, 820, 680, 710, 690],
    'Default': [1, 0, 0, 0, 1, 0, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

X = df[['Age', 'Income', 'LoanAmount', 'LoanTerm', 'CreditScore']]
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Model Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

def predict_default_probability(age, income, loan_amount, loan_term, credit_score):
    input_data = np.array([[age, income, loan_amount, loan_term, credit_score]])
    probability = model.predict_proba(input_data)[0, 1]
    return round(probability, 2)

new_borrower = {'Age': 32, 'Income': 5500, 'LoanAmount': 15000, 'LoanTerm': 36, 'CreditScore': 680}
default_probability = predict_default_probability(**new_borrower)
print(f"Probability of Default for new borrower: {default_probability}")
