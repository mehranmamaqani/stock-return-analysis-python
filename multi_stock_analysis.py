import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Download market data (S&P500)
market = yf.download(
    "^GSPC",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=False
)

market_return = market["Adj Close"].squeeze().pct_change()


def analyze_stock(stock_symbol):

    # Download stock data
    stock = yf.download(
        stock_symbol,
        start="2020-01-01",
        end="2025-01-01",
        auto_adjust=False
    )

    stock_return = stock["Adj Close"].squeeze().pct_change()

    # Combine stock and market returns
    data = pd.DataFrame({
        "stock_return": stock_return,
        "market_return": market_return
    })

    data = data.dropna()

    # CAPM Regression
    X = data[["market_return"]]
    y = data["stock_return"]

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict(X)

    return {
        "Stock": stock_symbol,
        "Beta": round(model.coef_[0], 3),
        "Alpha": round(model.intercept_, 6),
        "R2": round(r2_score(y, prediction), 3),
        "Volatility": round(data["stock_return"].std(), 4)
    }


# Stocks to analyze
stocks = ["AAPL", "MSFT", "TSLA"]

results = []

for stock in stocks:
    results.append(analyze_stock(stock))


# Create results table
results_df = pd.DataFrame(results)
import matplotlib.pyplot as plt


# Beta comparison

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Stock"],
    results_df["Beta"]
)

plt.xlabel("Stock")
plt.ylabel("Beta")
plt.title("CAPM Beta Comparison")

plt.savefig("images/beta_comparison.png")
plt.show()


# Volatility comparison

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Stock"],
    results_df["Volatility"]
)

plt.xlabel("Stock")
plt.ylabel("Volatility")
plt.title("Stock Return Volatility Comparison")

plt.savefig("images/volatility_comparison.png")
plt.show()
print(results_df)


# Save results
results_df.to_csv(
    "stock_capm_results.csv",
    index=False
)
