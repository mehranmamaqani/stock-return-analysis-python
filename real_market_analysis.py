import yfinance as yf
import pandas as pd
import numpy as np # Added this line


stock = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=False
)
market = yf.download(
    "^GSPC",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=False
)
stock["return"] = stock["Adj Close"].pct_change()
market["return"] = market["Adj Close"].pct_change()
data = pd.DataFrame({
    "stock_return": stock["return"],
    "market_return": market["return"]
})

data = data.dropna()

print(data.head())
print(data.describe())

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.plot(data.index, data["stock_return"])

plt.xlabel("Date")
plt.ylabel("Daily Return")

plt.title("AAPL Daily Returns (2020-2025)")

plt.show()

# regression of x and y

from sklearn.linear_model import LinearRegression

X = data[["market_return"]]
y = data["stock_return"]

model = LinearRegression()

model.fit(X, y)

print("Beta:", model.coef_[0])
print("Alpha:", model.intercept_)
from sklearn.metrics import r2_score

prediction = model.predict(X)

score = r2_score(y, prediction)

print("R2 Score:", score)

plt.figure(figsize=(8,5))

plt.scatter(
    data["market_return"],
    data["stock_return"]
)

plt.xlabel("Market Return")
plt.ylabel("Stock Return")

plt.title("CAPM: AAPL vs S&P500")

plt.show()

plt.figure(figsize=(8,5))

plt.scatter(
    data["market_return"],
    data["stock_return"]
)

# regression line
x_line = np.linspace(
    data["market_return"].min(),
    data["market_return"].max(),
    100
)

y_line = model.intercept_ + model.coef_[0] * x_line

plt.plot(x_line, y_line)

plt.xlabel("Market Return")
plt.ylabel("Stock Return")
plt.title("CAPM Regression: AAPL vs S&P500")

plt.show()
