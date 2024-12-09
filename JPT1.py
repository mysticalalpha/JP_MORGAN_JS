from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np


data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%y')
data = data.sort_values(by='Dates')  
data['days_from_start'] = (data['Dates'] - data['Dates'].min()).dt.days


X = data[['days_from_start']]
y = data['Prices']

model = LinearRegression()
model.fit(X, y)


def predict_price(input_date: str):
    try:
        
        input_date = pd.to_datetime(input_date)
        days_from_start = (input_date - data['Dates'].min()).days

        
        estimated_price = model.predict([[days_from_start]])[0]
        return round(estimated_price, 2)
    except Exception as e:
        return str(e)


example_date = "01/31/23"  
predicted_price = predict_price(example_date)
predicted_price
