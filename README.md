# Stock-Analysis

## Overview
This project consists of two parts, focusing on analyzing financial data and using machine learning to predict stock prices for asset management companies. The goal is to derive insights from financial statements, implement predictive models, and compare machine learning predictions with traditional Discounted Cash Flow (DCF) valuations.

---

## Project Breakdown

### Part 1: Correlation Analysis of Income Statement Factors
- **Objective**: Analyze the relationships between various factors in the income statements of asset management companies to understand their impact on financial performance and stock prices.
- **Key Tasks**:
  - Collect and preprocess financial data for selected asset management companies.
  - Perform statistical correlation analysis to identify significant financial indicators.
  - Visualize relationships between income statement factors and stock prices.
- **Deliverables**:
  - Insights into the key drivers of financial performance for asset management companies.
  - Correlation matrices and visualizations of significant relationships.

### Part 2: Predicting Stock Prices and Comparing Methods
- **Objective**: Build machine learning models (LSTM) to predict stock prices for asset management companies and compare the results with DCF valuations.
- **Key Tasks**:
  - Use DCF to calculate the intrinsic value of stocks.
  - Train and evaluate LSTM models using historical stock price data.
  - Compare the accuracy and insights of DCF vs. LSTM predictions over a 5-year horizon.
- **Deliverables**:
  - DCF valuation results.
  - Trained LSTM model and predicted stock prices.
  - Comparative analysis of DCF and LSTM performance.

---

## Project Life Cycle

### 1. Problem Definition
- **Part 1**: Identify key financial factors that influence stock prices for asset management companies.
- **Part 2**: Compare DCF and ML (LSTM) approaches for stock price prediction.
- **Deliverables**:
  - Evaluation metrics for both parts (e.g., correlation coefficients for Part 1, RMSE/MAPE for Part 2).
  - Clearly defined scope, including selected companies and timeframes.

---

### 2. Data Collection
- **Sources**:
  - **Financial Data**: `yfinance`, financial databases, or SEC filings for income statements.
  - **Stock Price Data**: Historical prices from `yfinance`.
- **Deliverables**:
  - Collected and cleaned datasets for income statements and stock prices.

---

### 3. Data Preparation
- **Part 1**:
  - Process income statement data to extract relevant financial metrics (e.g., revenue, net income, operating expenses).
  - Normalize data for comparison.
- **Part 2**:
  - For DCF: Calculate Free Cash Flow (FCF) and prepare assumptions for growth rates, WACC, and terminal value.
  - For LSTM: Normalize stock price data and create time-series sequences for model input.
- **Deliverables**:
  - Cleaned and prepared datasets for correlation analysis and predictive modeling.

---

### 4. Exploratory Data Analysis (EDA)
- **Part 1**:
  - Perform correlation analysis on income statement factors.
  - Visualize relationships using correlation heatmaps, scatterplots, and pairplots.
- **Part 2**:
  - Analyze trends in stock prices and financial metrics.
  - Identify patterns and anomalies in the data.
- **Tools**: `Pandas`, `Seaborn`, `Matplotlib`.
- **Deliverables**:
  - Visualizations and statistical summaries for both parts.

---

### 5. Model Development
- **Part 1**: 
  - Use statistical analysis tools to measure correlations.
- **Part 2**:
  - **DCF**:
    - Build a model to forecast FCF and calculate intrinsic stock value.
    - Use terminal value and discount rate for valuation.
  - **LSTM**:
    - Train an LSTM model using historical stock prices.
    - Tune hyperparameters for optimal performance.
- **Deliverables**:
  - Statistical insights for Part 1.
  - DCF and LSTM models for Part 2.

---

### 6. Model Evaluation
- **Part 1**:
  - Assess the strength and significance of correlations.
- **Part 2**:
  - Compare DCF predictions with actual market prices.
  - Evaluate LSTM performance using metrics like RMSE and MAPE.
- **Deliverables**:
  - Evaluation reports for both parts.
  - Comparative analysis of DCF and LSTM predictions.

---

### 7. Deployment
- Package the results into a reproducible pipeline:
  - Automate correlation analysis for Part 1.
  - Create reusable scripts for DCF and LSTM modeling in Part 2.
- **Deliverables**:
  - Python scripts or notebooks for both parts.
  - Dashboard or visualizations summarizing key insights.

---

### 8. Communication of Results
- **Tasks**:
  - Summarize insights:
    - Highlight significant income statement factors (Part 1).
    - Compare the performance of DCF and LSTM models (Part 2).
  - Use clear visualizations to present findings.
- **Deliverables**:
  - Final report or presentation with visualizations.

---

### 9. Post-Project Reflection
- **Tasks**:
  - Document lessons learned and challenges faced in each part.
  - Identify areas for improvement, such as incorporating external factors (e.g., sentiment analysis for LSTM).
- **Deliverables**:
  - Reflection document summarizing key takeaways.

---

## Tools and Technologies
- **Data Collection**: `yfinance`, SEC filings.
- **Data Processing and Visualization**: `Pandas`, `Matplotlib`, `Seaborn`.
- **Statistical Analysis**: Correlation analysis using `scipy` or `numpy`.
- **Model Development**:
  - **DCF**: Custom Python scripts.
  - **LSTM**: `TensorFlow`/`Keras`.
- **Evaluation**: RMSE, MAPE, and correlation coefficients.
- **Deployment**: Python scripts, Jupyter notebooks.

---

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name
