import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Download stock data
stock = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=False
)

# Download market data (S&P 500)
market = yf.download(
    "^GSPC",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=False
)

# Calculate daily returns
stock["return"] = stock["Adj Close"].pct_change()
market["return"] = market["Adj Close"].pct_change()

# Create a combined DataFrame for returns
data = pd.DataFrame({
    "stock_return": stock["return"],
    "market_return": market["return"]
})

# Drop any rows with NaN values resulting from pct_change()
data = data.dropna()

# Display basic information about the data
print("--- Data Head ---")
print(data.head())
print("\n--- Data Description ---")
print(data.describe())

# Plot AAPL Daily Returns
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["stock_return"], label='AAPL Daily Return', alpha=0.8)
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.title("AAPL Daily Returns (2020-2025)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Prepare data for Linear Regression
X = data[["market_return"]]
y = data["stock_return"]

# Initialize and fit the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Print Beta and Alpha (coefficients and intercept)
print(f"\nBeta (Market Sensitivity): {model.coef_[0]:.4f}")
print(f"Alpha (Intercept): {model.intercept_:.4f}")

# Make predictions and calculate R2 Score
prediction = model.predict(X)
score = r2_score(y, prediction)
print(f"R2 Score: {score:.4f}")

# Plot CAPM: AAPL vs S&P500 (Scatter Plot)
plt.figure(figsize=(10, 7))
plt.scatter(
    data["market_return"],
    data["stock_return"],
    alpha=0.6, s=20, label='Daily Returns'
)
plt.xlabel("Market Return")
plt.ylabel("Stock Return")
plt.title("CAPM: AAPL vs S&P500 (Scatter Plot)")
plt.grid(True, linestyle='--', alpha=0.6)

# Add the regression line to the scatter plot
x_line = np.linspace(
    data["market_return"].min(),
    data["market_return"].max(),
    100
)
y_line = model.intercept_ + model.coef_[0] * x_line
plt.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, label=f'Regression Line (y = {model.coef_[0]:.2f}x + {model.intercept_:.2f})')

plt.legend()
plt.tight_layout()
plt.show()
