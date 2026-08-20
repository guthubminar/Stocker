"""
Rank a list of stock tickers by current RSI (Relative Strength Index).

Requirements:
    pip install yfinance pandas openpyxl

Usage:
    python rank_stocks_by_rsi.py --input my_stocks.xlsx --column Ticker --output rsi_ranked.xlsx

Input Excel file:
    A single column of ticker symbols. By default the script looks for a
    column named "Ticker" (case-insensitive); if not found, it just uses
    the first column.

Output:
    - Prints the tickers sorted by RSI (ascending: most "oversold" first).
    - Saves the full table (Ticker, RSI, Close) to an Excel file if --output
      is given.
"""

import argparse
import sys
import time

import pandas as pd
import yfinance as yf


def compute_rsi(close_prices: pd.Series, period: int = 14) -> float:
    """
    Compute the most recent RSI value for a series of closing prices
    using Wilder's smoothing method (the standard/original RSI formula).
    """
    delta = close_prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing = an EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1]


def load_tickers(path: str, column: str | None) -> list[str]:
    df = pd.read_excel(path)

    if column and column in df.columns:
        col = column
    else:
        # try case-insensitive match for "ticker"
        matches = [c for c in df.columns if str(c).strip().lower() == "ticker"]
        col = matches[0] if matches else df.columns[0]

    tickers = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )
    return tickers


def get_rsi_table(tickers: list[str], period: int = 14, lookback: str = "6mo") -> pd.DataFrame:
    rows = []

    for i, ticker in enumerate(tickers, 1):
        try:
            hist = yf.Ticker(ticker).history(period=lookback, interval="1d")
            if hist.empty or len(hist) < period + 1:
                print(f"[{i}/{len(tickers)}] {ticker}: not enough data, skipping")
                continue

            rsi_value = compute_rsi(hist["Close"], period=period)
            last_close = hist["Close"].iloc[-1]

            rows.append({"Ticker": ticker, "RSI": round(rsi_value, 2), "Close": round(last_close, 2)})
            print(f"[{i}/{len(tickers)}] {ticker}: RSI={rsi_value:.2f}")

        except Exception as e:
            print(f"[{i}/{len(tickers)}] {ticker}: ERROR - {e}")

        # small delay to be polite to the data provider / avoid rate limiting
        time.sleep(0.3)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("RSI", ascending=True).reset_index(drop=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="Rank stock tickers by current RSI")
    parser.add_argument("--input", "-i", required=True, help="Path to input Excel file with tickers")
    parser.add_argument("--column", "-c", default="Ticker", help="Column name containing tickers (default: Ticker)")
    parser.add_argument("--period", "-p", type=int, default=14, help="RSI period (default: 14)")
    parser.add_argument("--lookback", "-l", default="6mo", help="History lookback window for yfinance (default: 6mo)")
    parser.add_argument("--output", "-o", default=None, help="Optional path to save results as Excel")
    args = parser.parse_args()

    tickers = load_tickers(args.input, args.column)
    print(f"Loaded {len(tickers)} tickers from {args.input}\n")

    table = get_rsi_table(tickers, period=args.period, lookback=args.lookback)

    if table.empty:
        print("\nNo RSI values could be computed.")
        sys.exit(1)

    print("\n=== Ranked by RSI (ascending: most oversold first) ===")
    print(table.to_string(index=False))

    if args.output:
        table.to_excel(args.output, index=False)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
