# minor-project
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def stock_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data.set_index('Date')

def prepdata(data):
    data['Prediction'] = data['Close'].shift(-30)
    return data[:-30]

def train_model(data):
    X = data[['Close']]  
    y = data['Prediction']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def plot_predictions(data, model):
    last_30_days = data[-30:][['Close']]
    predicted_prices = model.predict(last_30_days)

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
    predicted_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Price'])

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Historical Prices')
    plt.plot(predicted_df.index, predicted_df['Predicted Price'], label='Predicted Prices', color='orange')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file_path = r"C:\Users\DELL\Downloads\NVDA.csv"
    data = stock_data(file_path)
    prepared_data = prepdata(data)

    model, X_test, y_test = train_model(prepared_data)

    plot_predictions(prepared_data, model)

