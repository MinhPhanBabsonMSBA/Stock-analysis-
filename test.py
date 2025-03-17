import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

# Sidebar options
ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")  # Default: Apple Inc.
analysis_option = st.sidebar.radio(
    "Select Analysis Type:", 
    ["Fundamental & Technical Analysis", "Portfolio Simulation", "LSTM Prediction"]
)

# Fetch stock data
stock = yf.Ticker(ticker)
price_data = stock.history(period="5y")

if price_data.empty:
    st.error("No stock price data found for the ticker. Please try another one.")
else:
    st.success(f"Fetched data for {ticker} ✅")

    ### 📌 Fundamental & Technical Analysis
    if analysis_option == "Fundamental & Technical Analysis":
        st.header("📊 Fundamental & Technical Analysis")

        ## 1️⃣ P/E Ratio & Estimated Stock Price
        st.subheader("📈 Key Financial Indicators & Estimated Stock Price")
        pe_ratio = stock.info.get('forwardPE', stock.info.get('trailingPE', None))
        eps = stock.info.get('trailingEps', stock.info.get('forwardEps', None))
        info = stock.info  # Fetch the info dictionary once and reuse it

        if pe_ratio is not None and eps is not None:
            estimated_price = eps * pe_ratio
            st.write(f"**P/E Ratio:** {pe_ratio}")
            st.write(f"**EPS (Earnings Per Share):** {eps}")
            st.write(f"**P/B Ratio:** {info.get('priceToBook', 'N/A')}")
            st.write(f"**Debt-to-Equity Ratio:** {info.get('debtToEquity', 'N/A')}")
            st.write(f"**Return on Equity (ROE):** {info.get('returnOnEquity', 'N/A')}")
            st.write(f"**Estimated Stock Price (EPS × P/E Ratio):** {estimated_price:.2f}")
        else:
            st.warning("⚠️ P/E ratio or EPS data is unavailable.")

        ## 2️⃣ Stock Price with Moving Averages
        st.subheader("📊 Stock Price with Moving Averages")
        price_data['SMA 20'] = price_data['Close'].rolling(window=20).mean()
        price_data['SMA 50'] = price_data['Close'].rolling(window=50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'], mode='lines', name="Stock Price"))
        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['SMA 20'], mode='lines', name="20-Day SMA", line=dict(color='red')))
        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['SMA 50'], mode='lines', name="50-Day SMA", line=dict(color='blue')))
        st.plotly_chart(fig)

        ## 3️⃣ RSI Indicator (Momentum)
        st.subheader("⚡ RSI Indicator")
        price_data['RSI'] = 100 - (100 / (1 + (price_data['Close'].diff().where(price_data['Close'].diff() > 0, 0).rolling(14).mean() / price_data['Close'].diff().where(price_data['Close'].diff() < 0, 0).rolling(14).mean())))
        fig = go.Figure(go.Scatter(x=price_data.index, y=price_data['RSI'], mode='lines', name="RSI"))
        st.plotly_chart(fig)

        ## 4️⃣ Dividends
        st.subheader("📅 Dividends Over Time")
        dividends = stock.dividends
        if not dividends.empty:
            # Convert dividends to a DataFrame for easier plotting
            dividends_df = dividends.reset_index()
            dividends_df.columns = ['Date', 'Dividend Amount']
            
            # Extract the year from the Date column
            dividends_df['Year'] = dividends_df['Date'].dt.year
            
            # Create a color map for each year
            color_map = px.colors.qualitative.Plotly  # Use a predefined color palette
            
            # Plot dividends using Plotly with color-coded bars by year
            fig = go.Figure()
            
            # Add a bar trace for each year
            for year, color in zip(dividends_df['Year'].unique(), color_map):
                year_data = dividends_df[dividends_df['Year'] == year]
                fig.add_trace(go.Bar(
                    x=year_data['Date'],
                    y=year_data['Dividend Amount'],
                    name=str(year),  # Legend label for the year
                    marker_color=color,  # Assign a unique color to each year
                    text=[f"${amount:.2f}" for amount in year_data['Dividend Amount']],  # Format labels as currency
                    textposition='auto'  # Automatically position labels
                ))
            
            # Update layout
            fig.update_layout(
                title=f"Dividend History for {ticker}",
                xaxis_title="Date",
                yaxis_title="Dividend Amount",
                hovermode="x unified",
                barmode='group',  # Group bars by year
                showlegend=True  # Show legend for each year
            )
            st.plotly_chart(fig)
        else:
            st.write("No dividend data available.")

    ### 📌 Portfolio Simulation
          ### 📌 Portfolio Simulation
    elif analysis_option == "Portfolio Simulation":
        st.header("📂 Portfolio Simulation")
        
        # Sidebar for portfolio simulation
        tickers = st.sidebar.text_input("Enter Tickers (comma-separated):", "AAPL,MSFT,GOOGL")
        tickers = [ticker.strip() for ticker in tickers.split(",")]

        if tickers:
            # Fetch portfolio data
            portfolio_data = yf.download(tickers, period="1y", group_by='ticker')
            portfolio_returns = pd.DataFrame()

            for ticker in tickers:
                portfolio_returns[ticker] = portfolio_data[ticker]['Close'].pct_change().dropna()

            # Fetch S&P 500 data for benchmarking
            sp500_data = yf.download("^GSPC", period="1y")['Close'].pct_change().dropna()

            # Align dates between portfolio_returns and sp500_data
            aligned_data = pd.concat([portfolio_returns, sp500_data], axis=1, join='inner')
            aligned_data.columns = tickers + ['S&P 500']

            # Reassign aligned data
            portfolio_returns = aligned_data[tickers]
            sp500_data = aligned_data['S&P 500']

            # Calculate cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod()

            # Plot portfolio performance
            st.subheader("📈 Portfolio Performance")
            st.line_chart(cumulative_returns)

            # Correlation heatmap with red-blue color coding
            st.subheader("📊 Portfolio Correlation Heatmap")
            corr_matrix = portfolio_returns.corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale='RdBu',  # Red-blue diverging color scale
                zmin=-1,  # Minimum correlation value
                zmax=1    # Maximum correlation value
            )
            fig.update_layout(title="Correlation Heatmap")
            st.plotly_chart(fig)

            # Calculate technical metrics
            st.subheader("📊 Portfolio Metrics")

            # Risk-free rate (approximated as 0 for simplicity)
            risk_free_rate = 0

            # Average annualized return
            avg_annual_return = portfolio_returns.mean() * 252  # 252 trading days in a year

            # Portfolio volatility (annualized)
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)

            # Sharpe Ratio
            sharpe_ratio = (avg_annual_return - risk_free_rate) / portfolio_volatility

            # Treynor Ratio (Beta is required)
            beta = portfolio_returns.apply(lambda x: x.cov(sp500_data) / sp500_data.var())
            treynor_ratio = (avg_annual_return - risk_free_rate) / beta

            # Information Ratio (Benchmark is S&P 500)
            excess_returns = portfolio_returns.subtract(sp500_data, axis=0)
            information_ratio = excess_returns.mean() / excess_returns.std()

            # Comparison to S&P 500 return
            sp500_cumulative_return = (1 + sp500_data).cumprod().iloc[-1] - 1

            # Display metrics
            metrics_df = pd.DataFrame({
                "Ticker": tickers,
                "Avg Annual Return": avg_annual_return,
                "Sharpe Ratio": sharpe_ratio,
                "Treynor Ratio": treynor_ratio,
                "Information Ratio": information_ratio,
                "Beta": beta
            })
            st.write(metrics_df)

            # Comparison to S&P 500
            st.subheader("📊 Comparison to S&P 500")
            st.write(f"**S&P 500 Cumulative Return (1 Year):** {sp500_cumulative_return:.2%}")
            for ticker in tickers:
                ticker_cumulative_return = cumulative_returns[ticker].iloc[-1] - 1
                st.write(f"**{ticker} Cumulative Return (1 Year):** {ticker_cumulative_return:.2%}")

    ### 📌 LSTM Prediction (with EPS & P/E Ratio)
    elif analysis_option == "LSTM Prediction":
        st.header("📈 LSTM-Based Stock Price Prediction (Including EPS & P/E Ratio)")
        # (Your existing LSTM code here)

        # Sidebar for model parameters
        st.sidebar.header("Model Parameters")
        look_back = st.sidebar.slider("Look-back Period (Days)", min_value=1, max_value=30, value=5, step=1)
        epochs = st.sidebar.slider("Epochs", min_value=5, max_value=100, value=20, step=5)
        batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=64, value=16, step=8)
        num_layers = st.sidebar.selectbox("Number of LSTM Layers", options=[1, 2, 3], index=0)
        lstm_units = st.sidebar.slider("LSTM Units per Layer", min_value=5, max_value=100, value=10, step=5)

        if st.button("Predict Stock Price"):
            df = stock.history(period="1y")  # 1 year of historical data
            
            if df.empty:
                st.error("No data found for the ticker.")
            else:
                df = df[['Close', 'Open', 'Volume', 'High', 'Low']]

                # Fetch EPS and P/E Ratio
                eps = stock.info.get('trailingEps', stock.info.get('forwardEps', None))
                pe_ratio = stock.info.get('forwardPE', stock.info.get('trailingPE', None))

                if eps is None or pe_ratio is None:
                    st.warning("⚠️ EPS or P/E ratio is missing. Using default values.")
                    eps = df['Close'].pct_change().mean()  # Approximate from stock price change
                    pe_ratio = df['Close'].mean() / eps if eps != 0 else 15  # Approximate P/E ratio

                # Add EPS and P/E Ratio to the dataset
                df['EPS'] = eps
                df['P/E Ratio'] = pe_ratio

                # Fill missing values if any
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

                num_features = df_scaled.shape[1]

                # Build LSTM Model
                model = Sequential()
                if num_layers == 1:
                    model.add(LSTM(units=lstm_units, return_sequences=False, input_shape=(look_back, num_features)))
                else:
                    model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=(look_back, num_features)))
                    for _ in range(num_layers - 2):
                        model.add(LSTM(units=lstm_units, return_sequences=True))
                    model.add(LSTM(units=lstm_units, return_sequences=False))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')

                st.write("Training the model...")
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

                # Make predictions
                y_pred = model.predict(X_test)
                y_pred = inverse_transform_feature(y_pred, scaler, feature_index=0)
                y_test = inverse_transform_feature(y_test.reshape(-1, 1), scaler, feature_index=0)

                # Performance Metrics
                r2 = r2_score(y_test, y_pred)
                mape = MAPE(y_test, y_pred)
                accuracy_rate = 100 - mape

                st.write(f"**R² Score:** {r2:.3f}")
                st.write(f"**MAPE:** {mape:.2f}%")
                st.write(f"**Accuracy Rate:** {accuracy_rate:.2f}%")

                # Plot Actual vs Predicted
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=y_test.flatten(), mode='lines', name='Actual Prices'))
                fig.add_trace(go.Scatter(y=y_pred.flatten(), mode='lines', name='Predicted Prices', line=dict(dash='dash')))
                fig.update_layout(title=f"LSTM Prediction for {ticker} (Including EPS & P/E Ratio)", xaxis_title="Time Steps", yaxis_title="Stock Price")
                st.plotly_chart(fig, use_container_width=True)