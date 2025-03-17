import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import r2_score

# Helper function to compute MAPE
def MAPE(y_true, y_pred):
    epsilon = 1e-10  # Avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# Helper function to inverse transform only the target feature (Close price)
def inverse_transform_feature(scaled_values, scaler, feature_index=0):
    dummy = np.zeros((scaled_values.shape[0], scaler.scale_.shape[0]))
    dummy[:, feature_index] = scaled_values[:, 0]
    inv = scaler.inverse_transform(dummy)
    return inv[:, feature_index].reshape(-1, 1)

# Data preparation function for multiple predictors
def prepare_data(data, look_back=5):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, :])
        y.append(data[i, 0])  # Using 'Close' as the target variable
    return np.array(X), np.array(y)

# Streamlit app layout
st.title("📊 Stock Analysis & Prediction Dashboard")

# Sidebar options (Fundamental & Technical Analysis first, then LSTM)
ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")  # Default: Apple Inc.
analysis_option = st.sidebar.radio(
    "Select Analysis Type:", 
    ["Fundamental & Technical Analysis", "LSTM Prediction"]
)

# Fetch stock data
df = yf.download(ticker, period="5y")  # Using yf.download() instead of stock.history()

if df.empty:
    st.error("No stock price data found for the ticker. Please try another one.")
else:
    st.success(f"Fetched data for {ticker} ✅")

    ### 📌 Fundamental & Technical Analysis (Comes First)
    if analysis_option == "Fundamental & Technical Analysis":
        st.header("📊 Fundamental & Technical Analysis")

        ## 1️⃣ P/E Ratio & Estimated Stock Price
        st.subheader("📈 P/E Ratio & Estimated Stock Price")
        stock = yf.Ticker(ticker)
        pe_ratio = stock.info.get('forwardPE', stock.info.get('trailingPE', None))
        eps = stock.info.get('trailingEps', stock.info.get('forwardEps', None))

        if pe_ratio is not None and eps is not None:
            estimated_price = eps * pe_ratio
            st.write(f"**P/E Ratio:** {pe_ratio}")
            st.write(f"**EPS (Earnings Per Share):** {eps}")
            st.write(f"**Estimated Stock Price (EPS × P/E Ratio):** {estimated_price:.2f}")
        else:
            st.warning("⚠️ P/E ratio or EPS data is unavailable.")

        ## 2️⃣ Stock Price with Moving Averages
        st.subheader("📊 Stock Price with Moving Averages")
        df['SMA 20'] = df['Close'].rolling(window=20).mean()
        df['SMA 50'] = df['Close'].rolling(window=50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name="Stock Price"))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA 20'], mode='lines', name="20-Day SMA", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA 50'], mode='lines', name="50-Day SMA", line=dict(color='blue')))
        st.plotly_chart(fig)

    ### 📌 LSTM Prediction (Comes Second)
    elif analysis_option == "LSTM Prediction":
        st.header("📈 LSTM-Based Stock Price Prediction (Including EPS & P/E Ratio)")

        # Sidebar for model parameters
        st.sidebar.header("Model Parameters")
        look_back = st.sidebar.slider("Look-back Period (Days)", min_value=1, max_value=30, value=5, step=1)
        epochs = st.sidebar.slider("Epochs", min_value=5, max_value=100, value=20, step=5)
        batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=64, value=16, step=8)
        lstm_units = st.sidebar.slider("LSTM Units per Layer", min_value=5, max_value=100, value=10, step=5)

        if st.button("Predict Stock Price"):
            df = df[['Close', 'Open', 'Volume', 'High', 'Low']]

            # Fetch EPS and P/E Ratio
            eps = stock.info.get('trailingEps', stock.info.get('forwardEps', None))
            pe_ratio = stock.info.get('forwardPE', stock.info.get('trailingPE', None))

            if eps is None or pe_ratio is None:
                st.warning("⚠️ EPS or P/E ratio is missing. Using default values.")
                eps = df['Close'].pct_change().mean()
                pe_ratio = df['Close'].mean() / eps if eps != 0 else 15

            df['EPS'] = eps
            df['P/E Ratio'] = pe_ratio
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)

            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_scaled = scaler.fit_transform(df)

            # Prepare data
            X, y = prepare_data(df_scaled, look_back)
            train_size = int(len(X) * 0.5)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Build LSTM Model
            model = Sequential([
                LSTM(units=lstm_units, return_sequences=True, input_shape=(look_back, df_scaled.shape[1])),
                LSTM(units=lstm_units, return_sequences=False),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            st.write("Training the model...")
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

            # Predictions
            y_pred = inverse_transform_feature(model.predict(X_test), scaler, feature_index=0)
            y_test = inverse_transform_feature(y_test.reshape(-1, 1), scaler, feature_index=0)

            # Performance Metrics
            st.write(f"**R² Score:** {r2_score(y_test, y_pred):.3f}")
            st.write(f"**MAPE:** {MAPE(y_test, y_pred):.2f}%")

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_test.flatten(), mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(y=y_pred.flatten(), mode='lines', name='Predicted', line=dict(dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
