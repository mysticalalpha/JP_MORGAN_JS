import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

data = {
    'FICO Score': [550, 620, 680, 720, 750, 810, 590, 640, 700, 770],
    'Default': [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

def bucket_fico(score):
    if score < 580:
        return "Poor"
    elif 580 <= score <= 669:
        return "Fair"
    elif 670 <= score <= 739:
        return "Good"
    elif 740 <= score <= 799:
        return "Very Good"
    else:
        return "Exceptional"

df['FICO Bucket'] = df['FICO Score'].apply(bucket_fico)

bucket_summary = df.groupby('FICO Bucket')['Default'].mean().reset_index()
bucket_summary.rename(columns={'Default': 'Default Rate'}, inplace=True)
print("Default Rates by FICO Bucket:")
print(bucket_summary)

bucket_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Exceptional': 4}
df['Bucket Code'] = df['FICO Bucket'].map(bucket_mapping)

X = df[['Bucket Code']]
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

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

def predict_default_probability(fico_score):
    bucket = bucket_fico(fico_score)
    bucket_code = bucket_mapping[bucket]
    input_data = np.array([[bucket_code]])
    probability = model.predict_proba(input_data)[0, 1]
    return bucket, round(probability, 2)

new_fico_score = 690
bucket, default_probability = predict_default_probability(new_fico_score)
print(f"\nFor FICO Score {new_fico_score}:")
print(f"Bucket: {bucket}")
print(f"Probability of Default: {default_probability}")
