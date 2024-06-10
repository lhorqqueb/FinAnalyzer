import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import seaborn as sns

# Load dataset
data = pd.read_csv('/Users/louishorqque/Documents/DataScience/FinAnalyzer/Data/S&PHistoricalData.csv')

# Data preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data = data.dropna(subset=['Price', 'Open', 'High', 'Low', 'Change %'])
numeric_cols = ['Price', 'Open', 'High', 'Low']
for col in numeric_cols:
    data[col] = data[col].str.replace(',', '').astype(float)

# Scale the 'Price' prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))

# Create a function to prepare the dataset
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Parameters
time_step = 50
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Load model
model = load_model('/Users/louishorqque/Documents/DataScience/FinAnalyzer/model.h5')

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

# Visualization
st.title('S&P 500 Stock Price Analysis and Prediction')

# Plot Price Over Time
fig, ax = plt.subplots()
ax.plot(data['Date'], data['Price'], label='Actual Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('S&P 500 Stock Price Over Time')
ax.legend()
st.pyplot(fig)

# Plot Predictions
fig, ax = plt.subplots()
ax.plot(data['Date'], scaler.inverse_transform(scaled_data), label='Actual Price')
ax.plot(data['Date'][time_step:len(train_predict) + time_step], train_predict, label='Train Prediction')
ax.plot(data['Date'][len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1], test_predict, label='Test Prediction')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Stock Price Prediction')
ax.legend()
st.pyplot(fig)

# Show RMSE
st.write(f'Train RMSE: {train_rmse}')
st.write(f'Test RMSE: {test_rmse}')

# Price Distribution
fig, ax = plt.subplots()
ax.hist(data['Price'], bins=50)
ax.set_xlabel('Price')
ax.set_ylabel('Frequency')
ax.set_title('Price Distribution')
st.pyplot(fig)

# Correlation Matrix
corr_matrix = data[numeric_cols].corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix')
st.pyplot(fig)

# Moving Averages
data['MA50'] = data['Price'].rolling(window=50).mean()
data['MA200'] = data['Price'].rolling(window=200).mean()

fig, ax = plt.subplots()
ax.plot(data['Date'], data['Price'], label='Price')
ax.plot(data['Date'], data['MA50'], label='50-Day MA')
ax.plot(data['Date'], data['MA200'], label='200-Day MA')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Moving Averages')
ax.legend()
st.pyplot(fig)

# Daily Returns
data['Daily Return'] = data['Price'].pct_change()
fig, ax = plt.subplots()
ax.plot(data['Date'], data['Daily Return'])
ax.set_xlabel('Date')
ax.set_ylabel('Daily Return')
ax.set_title('Daily Returns Over Time')
st.pyplot(fig)

# Close Streamlit app gracefully
st.write("Close this window to stop the Streamlit app.")
