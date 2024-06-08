import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load dataset
data = pd.read_csv('Data/S&PHistoricalData.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Handle missing values
data = data.dropna(subset=['Price', 'Open', 'High', 'Low', 'Change %'])

# Remove commas and convert numeric columns to float
numeric_cols = ['Price', 'Open', 'High', 'Low']
for col in numeric_cols:
    data[col] = data[col].str.replace(',', '').astype(float)

# Scale the 'Price' prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))

# Dynamically set time_step based on the dataset size
time_step = min(50, len(scaled_data) // 10)

# Create a function to prepare the dataset
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size, :], scaled_data[train_size:, :]

# Create the training and testing datasets
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=50, validation_split=0.2)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform predictions back to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

# Shift train predictions for plotting
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

# Shift test predictions for plotting
test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("S&P 500 Stock Price Prediction"),
    dcc.Graph(id='price-graph'),
    html.Div([
        html.Label('Train RMSE: {:.2f}'.format(train_rmse)),
        html.Label('Test RMSE: {:.2f}'.format(test_rmse)),
    ])
])

@app.callback(
    Output('price-graph', 'figure'),
    [Input('price-graph', 'id')]
)
def update_graph(_):
    actual_price = go.Scatter(x=data['Date'], y=scaler.inverse_transform(scaled_data).flatten(), mode='lines', name='Actual Price')
    train_prediction = go.Scatter(x=data['Date'][time_step:len(train_predict) + time_step], y=train_predict.flatten(), mode='lines', name='Train Prediction')
    test_prediction = go.Scatter(x=data['Date'][len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1], y=test_predict.flatten(), mode='lines', name='Test Prediction')

    return {
        'data': [actual_price, train_prediction, test_prediction],
        'layout': go.Layout(
            title='S&P 500 Stock Price Prediction',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Price'},
            hovermode='closest'
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)