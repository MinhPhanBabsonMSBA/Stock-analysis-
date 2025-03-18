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
from scipy.optimize import minimize

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

# Function to fetch portfolio data
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0'})

def fetch_portfolio_data(tickers, period="1y"):
    """Fetches stock data using yfinance with session headers."""
    try:
        portfolio_data = yf.download(tickers, period=period, progress=False, session=session)
        if portfolio_data.empty:
            raise ValueError("No data retrieved. Check ticker symbols.")
        portfolio_returns = portfolio_data['Close'].pct_change().dropna()
        return portfolio_returns
    except Exception as e:
        print(f"Error fetching portfolio data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

def get_current_prices(tickers):
    """ Fetches the latest closing prices for tickers."""
    current_prices = {}
    try:
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1d", session=session)
            if df.empty:
                current_prices[ticker] = None
            else:
                current_prices[ticker] = df['Close'].iloc[-1]
    except Exception as e:
        print(f"Error fetching stock price for {ticker}: {e}")
        current_prices[ticker] = None
    return current_prices

# Functions for portfolio optimization using Scipy.optimize
def portfolio_annualized_performance(weights, mean_returns, cov_matrix):
    """
    Calculate annualized return and volatility for a portfolio
    """
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def portfolio_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Calculate Sharpe Ratio for a portfolio
    """
    p_returns, p_std = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
    return (p_returns - risk_free_rate) / p_std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Return negative Sharpe ratio (for minimization)
    """
    return -portfolio_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)

def check_sum(weights):
    """
    Check if weights sum to 1
    """
    return np.sum(weights) - 1

def optimize_portfolio_sharpe(returns, risk_free_rate=0.02):
    """
    Find the portfolio weights that maximize the Sharpe ratio
    """
    try:
        # Prepare mean returns and covariance matrix
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Add a small regularization term to the covariance matrix
        cov_matrix = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6

        # Number of assets
        num_assets = len(mean_returns)

        # Initial guess: equal allocation to each asset
        initial_guess = np.array([1.0/num_assets for _ in range(num_assets)])

        # Constraints - make sure the weights sum to 1
        constraints = ({'type': 'eq', 'fun': check_sum})

        # Bounds for weights (0 <= weight <= 1)
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Optimization
        result = minimize(
            negative_sharpe,
            initial_guess,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )

        # Check if optimization was successful
        if not result['success']:
            # If optimization fails, try a simpler approach: equal weights
            st.warning(f"Advanced optimization failed: {result['message']}. Using equal weights as fallback.")
            weights = np.array([1.0/num_assets for _ in range(num_assets)])
        else:
            weights = result['x']

        # Clean small weights (less than 1%)
        weights[weights < 0.01] = 0

        # Re-normalize to ensure sum is 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

        # Convert weights to a dictionary
        cleaned_weights = {returns.columns[i]: weights[i] for i in range(len(weights))}

        # Calculate portfolio performance
        p_returns, p_std = portfolio_annualized_performance(weights, mean_returns, cov_matrix)
        sharpe = portfolio_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate)

        performance = (p_returns, p_std, sharpe)

        return cleaned_weights, performance

    except Exception as e:
        st.error(f"Error in optimization: {str(e)}")
        # Create equal weight portfolio as fallback
        num_assets = len(returns.columns)
        equal_weights = {col: 1.0/num_assets for col in returns.columns}

        # Calculate performance for equal weights
        weights_array = np.array([1.0/num_assets for _ in range(num_assets)])
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        p_returns, p_std = portfolio_annualized_performance(weights_array, mean_returns, cov_matrix)
        sharpe = portfolio_sharpe_ratio(weights_array, mean_returns, cov_matrix, risk_free_rate)

        performance = (p_returns, p_std, sharpe)

        return equal_weights, performance

# Streamlit app layout
st.title("Stock Analytics Dashboard")

# Sidebar options
analysis_option = st.sidebar.radio(
    "Select Analysis Type:", 
    ["Fundamental & Technical Analysis", "Portfolio Simulation", "LSTM Prediction(demo)", "Portfolio Optimization"]
)

# Show "Enter Stock Ticker" only for Fundamental Analysis and LSTM Prediction
if analysis_option in ["Fundamental & Technical Analysis", "LSTM Prediction(demo)"]:
    ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")  # Default: Apple Inc.
else:
    ticker = None  # Hide the ticker input for Portfolio Simulation and Portfolio Optimization

# Fetch stock data (only if ticker is provided)
if ticker:
    stock = yf.Ticker(ticker)
    price_data = stock.history(period="5y")

    if price_data.empty:
        st.error("No stock price data found for the ticker. Please try another one.")
    else:
        ### 📌 Fundamental & Technical Analysis
        if analysis_option == "Fundamental & Technical Analysis":
            st.header("📈 Fundamental & Technical Analysis")

            ## 1️⃣ P/E Ratio & Estimated Stock Price
            st.subheader(" Key Financial Indicators & Estimated Stock Price")
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

        ### 📌 LSTM Prediction (with EPS & P/E Ratio)
        elif analysis_option == "LSTM Prediction(demo)":
            st.header("📈 LSTM-Based Stock Price Prediction (Including EPS & P/E Ratio)")
            
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

### 📌 Portfolio Simulation
elif analysis_option == "Portfolio Simulation":
    st.header("📊 Portfolio Simulation")
    
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
        fig = go.Figure()
        for ticker in tickers:
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[ticker],
                mode='lines',
                name=ticker
            ))
        fig.update_layout(
            title="Portfolio Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode="x unified"
        )
        st.plotly_chart(fig)

        # Correlation heatmap with red-blue color coding
        st.subheader("Portfolio Correlation Heatmap")
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
        st.subheader("Portfolio Metrics")

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

        # Calculate portfolio return (equal-weighted)
        portfolio_return = portfolio_returns.mean(axis=1)  # Equal-weighted portfolio
        portfolio_cumulative_return = (1 + portfolio_return).cumprod().iloc[-1] - 1

        # S&P 500 cumulative return
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
        st.subheader("Comparison to S&P 500")
        st.write(f"**Portfolio Cumulative Return (1 Year):** {portfolio_cumulative_return:.2%}")
        st.write(f"**S&P 500 Cumulative Return (1 Year):** {sp500_cumulative_return:.2%}")

### 📌 Portfolio Optimization
elif analysis_option == "Portfolio Optimization":
    st.header("🔧 Portfolio Optimization")
    st.write("This tool finds the optimal allocation of assets to maximize the Sharpe ratio (risk-adjusted return).")

    # Sidebar for portfolio simulation
    default_tickers = "AAPL,MSFT,GOOGL,AMZN,META"
    tickers_input = st.sidebar.text_area("Enter Tickers (comma-separated):", default_tickers)
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

    # Time period selection
    period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    selected_period = st.sidebar.selectbox("Select Time Period:", list(period_options.keys()), index=3)
    period = period_options[selected_period]

    # Limit the number of assets to prevent performance issues
    if len(tickers) > 10:
        st.warning("Too many assets in the portfolio. Limiting to the first 10 assets.")
        tickers = tickers[:10]

    if len(tickers) == 0:
        st.warning("Please enter at least one valid ticker symbol.")
    elif len(tickers) == 1:
        st.warning("Portfolio optimization requires at least 2 assets. Please add more tickers.")
    else:
        st.write(f"Analyzing the following tickers: {', '.join(tickers)}")

        # Fetch portfolio data with progress indicator
        with st.spinner("Fetching market data... This may take a moment."):
            portfolio_returns = fetch_portfolio_data(tickers, period=period)

        # Check if data is fetched successfully
        if portfolio_returns.empty or portfolio_returns.shape[1] == 0:
            st.error("No data found for the tickers. Please verify the ticker symbols and try again.")
            st.info("Example of valid tickers: AAPL (Apple), MSFT (Microsoft), GOOGL (Google), AMZN (Amazon), META (Meta/Facebook)")
        else:
            # Show which tickers were found
            found_tickers = list(portfolio_returns.columns)
            st.success(f"Successfully retrieved data for: {', '.join(found_tickers)}")

            # Check for missing tickers
            missing_tickers = [ticker for ticker in tickers if ticker not in found_tickers]
            if missing_tickers:
                st.warning(f"Could not find data for: {', '.join(missing_tickers)}")

            # Clean the data
            portfolio_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
            portfolio_returns = portfolio_returns.dropna()

            # Only show raw data if checkbox is selected
            with st.expander("View Raw Data and Statistics"):
                st.write("Portfolio Returns Data:")
                st.dataframe(portfolio_returns)

                # Display statistics
                st.write("Data Statistics:")
                st.dataframe(portfolio_returns.describe())

                # Show correlation matrix
                st.write("Correlation Matrix:")
                corr_matrix = portfolio_returns.corr()
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))

            if portfolio_returns.shape[1] < 2:
                st.error("Need at least 2 assets with valid data for optimization.")
            else:
                ### Portfolio Optimization
                st.subheader("🔧 Portfolio Optimization")

                # Risk-free rate input
                risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.1) / 100

                # Get current prices of assets
                current_prices = get_current_prices(found_tickers)

                # Input for number of shares and total budget
                st.subheader("Portfolio Setup")
                total_budget = st.number_input("Total Budget ($)", min_value=1.0, value=10000.0, step=100.0)

                shares = {}
                for ticker in found_tickers:
                    shares[ticker] = st.number_input(f"Number of Shares for {ticker}", min_value=0, value=0, step=1)

                # Calculate total value of the portfolio based on shares and current prices
                total_value = sum(shares[ticker] * current_prices[ticker] for ticker in found_tickers)

                # Auto-adjust shares if total value exceeds the budget
                if total_value > total_budget:
                    st.error(f"Total portfolio value (${total_value:.2f}) exceeds your budget (${total_budget:.2f}). Auto-adjusting shares...")

                    # Calculate the scaling factor to fit within the budget
                    scaling_factor = total_budget / total_value

                    # Adjust shares
                    adjusted_shares = {ticker: int(shares[ticker] * scaling_factor) for ticker in found_tickers}

                    # Ensure at least 1 share for each asset (if possible)
                    for ticker in adjusted_shares:
                        if adjusted_shares[ticker] < 1 and current_prices[ticker] <= total_budget:
                            adjusted_shares[ticker] = 1

                    # Recalculate total value with adjusted shares
                    adjusted_total_value = sum(adjusted_shares[ticker] * current_prices[ticker] for ticker in found_tickers)

                    # Update shares and notify the user
                    shares = adjusted_shares
                    st.success(f"Adjusted shares to fit within your budget. New total portfolio value: ${adjusted_total_value:.2f}")

                # Display adjusted shares
                st.write("Adjusted Shares:")
                st.write(pd.Series(shares))

                # Calculate initial weights based on adjusted shares and current prices
                initial_values = {ticker: shares[ticker] * current_prices[ticker] for ticker in found_tickers}
                total_value = sum(initial_values.values())

                if total_value == 0:
                    st.error("Total portfolio value is $0. Please enter valid shares.")
                else:
                    initial_weights = {ticker: value / total_value for ticker, value in initial_values.items()}

                    st.write("Initial Portfolio Weights:")
                    st.write(pd.Series(initial_weights).round(4))

                    # Optimize portfolio
                    if st.button("Optimize Portfolio"):
                        with st.spinner("Optimizing portfolio..."):
                            weights, performance = optimize_portfolio_sharpe(portfolio_returns, risk_free_rate)

                        # Display optimized portfolio weights
                        st.subheader("📊 Optimization Results")

                        # Convert weights to DataFrame for better display
                        weights_df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
                        weights_df['Weight'] = weights_df['Weight'] * 100  # Convert to percentage
                        weights_df = weights_df.sort_values('Weight', ascending=False)  # Sort by weight
                        weights_df = weights_df[weights_df['Weight'] > 0]  # Only show non-zero weights

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Portfolio Weights:**")
                            st.dataframe(weights_df.style.format({'Weight': '{:.2f}%'}))

                        with col2:
                            # Display portfolio performance metrics
                            st.write("**Portfolio Performance:**")
                            metrics_df = pd.DataFrame({
                                'Metric': ['Expected Annual Return', 'Annual Volatility', 'Sharpe Ratio'],
                                'Value': [f"{performance[0]:.2%}", f"{performance[1]:.2%}", f"{performance[2]:.2f}"]
                            })
                            st.dataframe(metrics_df.set_index('Metric'))

                        # Plot portfolio weights
                        st.subheader("Portfolio Allocation")

                        # Only include assets with weights > 0
                        significant_weights = {k: v for k, v in weights.items() if v > 0}

                        # Sort for better visualization
                        sorted_weights = dict(sorted(significant_weights.items(), key=lambda x: x[1], reverse=True))

                        # Bar chart
                        fig = go.Figure(go.Bar(
                            x=list(sorted_weights.keys()),
                            y=[v*100 for v in sorted_weights.values()],  # Convert to percentages
                            text=[f"{w*100:.2f}%" for w in sorted_weights.values()],
                            textposition='auto',
                            marker_color='royalblue'
                        ))
                        fig.update_layout(
                            title="Portfolio Weights",
                            xaxis_title="Assets",
                            yaxis_title="Allocation (%)",
                            showlegend=False,
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Add a pie chart visualization for the allocation
                        pie_fig = go.Figure(go.Pie(
                            labels=list(sorted_weights.keys()),
                            values=[v*100 for v in sorted_weights.values()],
                            textinfo='label+percent',
                            hole=.3
                        ))
                        pie_fig.update_layout(
                            title="Portfolio Allocation (Pie Chart)",
                            height=500
                        )
                        st.plotly_chart(pie_fig, use_container_width=True)
