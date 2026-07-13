"""
NIFTY Linear-Regression Mean-Reversion Strategy — Streamlit Dashboard
-----------------------------------------------------------------------
Live dashboard version of the original cron script. Recomputes the
strategy/backtest with a cache TTL (so it doesn't hammer Yahoo Finance),
and auto-refreshes the browser tab periodically so an open tab picks
up fresh data through the trading day and after NSE close (~3:30pm IST).

IMPORTANT — read this before you assume it "just refreshes at 5pm":
Streamlit Community Cloud only runs your code when a browser loads the
page. There is no server-side cron. Two things make 5pm IST refreshes
work in practice:
  1. st_autorefresh() below reruns the page every 30 min IF a tab is
     open in someone's browser.
  2. To get a refresh even when nobody has it open, set up a free
     external ping (e.g. https://cron-job.org, free tier) to GET your
     deployed app's URL once a day around 5pm IST. That request wakes
     the app and forces a recompute (see README.md).

Optional: sends a Telegram summary + chart if TELEGRAM_BOT_TOKEN /
TELEGRAM_CHAT_ID are set as Streamlit secrets (st.secrets), triggered
by the "Send to Telegram" button — not automatic, since Streamlit
can't fire actions with nobody there to click them.
"""

import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
import requests
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------- config
TICKER = "^NSEI"          # NSE Nifty 50 Index
START_DATE = "2020-01-01"
WINDOW = 50
THRESHOLD = 2
INITIAL_CAPITAL = 10000
PLOT_LAST_N_DAYS = 90
AUTOREFRESH_MS = 30 * 60 * 1000   # 30 minutes — only matters if a tab is open
CACHE_TTL_SECONDS = 30 * 60       # 30 minutes — bounds how often yfinance is hit

st.set_page_config(page_title="NIFTY Mean Reversion", layout="wide", page_icon="📈")

# Reruns the whole script every AUTOREFRESH_MS while a tab is open.
st_autorefresh(interval=AUTOREFRESH_MS, key="auto_refresh_tick")


# ---------------------------------------------------------------- data
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching NIFTY data...")
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker} — market may be closed or ticker invalid.")
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"][ticker]
    return data["Close"]


# ---------------------------------------------------------------- strategy
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Running regression strategy...")
def linear_regression_mean_reversion_strategy(prices, window=50, threshold=2):
    signals = pd.DataFrame(index=prices.index)
    signals["price"] = prices
    signals["regression_line"] = np.nan
    signals["standard_error"] = np.nan
    signals["deviation"] = np.nan
    signals["upper_2se"] = np.nan
    signals["lower_2se"] = np.nan

    for i in range(window, len(prices)):
        y = prices.iloc[i - window:i].values.reshape(-1, 1)
        X = np.arange(window).reshape(-1, 1)
        model = LinearRegression().fit(X, y)

        regression_line = model.predict(np.array([[window - 1]]))[0][0]

        residuals = y - model.predict(X)
        residual_sum_of_squares = np.sum(residuals ** 2)
        standard_error = np.sqrt(residual_sum_of_squares / (window - 2))

        current_date = signals.index[i]
        signals.loc[current_date, "regression_line"] = regression_line
        signals.loc[current_date, "standard_error"] = standard_error
        signals.loc[current_date, "deviation"] = signals["price"].iloc[i] - regression_line
        signals.loc[current_date, "upper_2se"] = regression_line + 2 * standard_error
        signals.loc[current_date, "lower_2se"] = regression_line - 2 * standard_error

    signals["buy_signal"] = signals["deviation"] < -threshold * signals["standard_error"]
    signals["sell_signal"] = signals["deviation"] > threshold * signals["standard_error"]

    return signals.dropna()


# ---------------------------------------------------------------- backtest
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def backtest_strategy(signals, initial_capital=10000):
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    portfolio = pd.DataFrame(index=signals.index).fillna(0.0)

    positions["stock"] = 0.0
    current_position = 0.0

    for i in range(1, len(signals)):
        if signals["buy_signal"].iloc[i]:
            current_position = initial_capital // signals["price"].iloc[i]
        elif signals["sell_signal"].iloc[i]:
            current_position = 0.0
        positions.loc[positions.index[i], "stock"] = current_position

    portfolio["positions"] = positions["stock"] * signals["price"]
    trade_flows = positions["stock"].diff().fillna(0.0) * signals["price"]
    portfolio["cash"] = initial_capital - trade_flows.cumsum()
    portfolio["total"] = portfolio["positions"] + portfolio["cash"]

    return portfolio


# ---------------------------------------------------------------- chart
def make_chart_figure(signals, portfolio, last_n_days=90):
    view_signals = signals.iloc[-last_n_days:]
    view_portfolio = portfolio.iloc[-last_n_days:]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))

    ax1.plot(view_signals.index, view_signals["price"], label="Price")
    ax1.plot(view_signals.index, view_signals["regression_line"], label="Regression Line", color="orange")
    ax1.fill_between(
        view_signals.index,
        view_signals["lower_2se"],
        view_signals["upper_2se"],
        color="lightgrey", label="2 SE Band",
    )

    buy_dates = view_signals[view_signals["buy_signal"]].index
    ax1.scatter(buy_dates, view_signals.loc[buy_dates, "price"],
                label="Buy Signal", marker="^", color="green", s=100, zorder=5)

    sell_dates = view_signals[view_signals["sell_signal"]].index
    ax1.scatter(sell_dates, view_signals.loc[sell_dates, "price"],
                label="Sell Signal", marker="v", color="red", s=100, zorder=5)

    ax1.legend()
    ax1.set_title(f"{TICKER} Linear Regression Mean Reversion (Last {last_n_days} Days)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(view_portfolio.index, view_portfolio["total"], label="Portfolio Value", color="purple")
    ax2.set_title(f"Portfolio Value (Last {last_n_days} Days)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------- telegram (optional, manual trigger)
def send_telegram_message(text):
    token = st.secrets.get("TELEGRAM_BOT_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise EnvironmentError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in Streamlit secrets.")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(url, data={"chat_id": chat_id, "text": text[:4000]}, timeout=30)
    resp.raise_for_status()


def send_telegram_photo(fig, caption=""):
    token = st.secrets.get("TELEGRAM_BOT_TOKEN")
    chat_id = st.secrets.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise EnvironmentError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set in Streamlit secrets.")
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    resp = requests.post(
        url,
        data={"chat_id": chat_id, "caption": caption[:1000]},
        files={"photo": buf},
        timeout=60,
    )
    resp.raise_for_status()


# ---------------------------------------------------------------- main app
st.title("📈 NIFTY Linear-Regression Mean-Reversion Dashboard")

ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
st.caption(
    f"Page loaded at {ist_now.strftime('%Y-%m-%d %H:%M')} IST · "
    f"data cached for {CACHE_TTL_SECONDS // 60} min · "
    f"tab auto-refreshes every {AUTOREFRESH_MS // 60000} min while open"
)

try:
    end_date = datetime.today().strftime("%Y-%m-%d")
    prices = fetch_stock_data(TICKER, START_DATE, end_date)
    signals = linear_regression_mean_reversion_strategy(prices, WINDOW, THRESHOLD)
    portfolio = backtest_strategy(signals, INITIAL_CAPITAL)

    latest = signals.iloc[-1]
    latest_date = signals.index[-1].strftime("%Y-%m-%d")
    portfolio_value = portfolio["total"].iloc[-1]
    portfolio_start = portfolio["total"].iloc[0]
    pnl_pct = (portfolio_value / portfolio_start - 1) * 100

    if latest["buy_signal"]:
        signal_text = "🟢 BUY (price below regression band)"
    elif latest["sell_signal"]:
        signal_text = "🔴 SELL (price above regression band)"
    else:
        signal_text = "⚪ No signal (price within band)"

    st.subheader(f"Latest close: {latest_date}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"{latest['price']:.2f}")
    col2.metric("Regression Price", f"{latest['regression_line']:.2f}")
    col3.metric("Standard Error", f"{latest['standard_error']:.2f}")
    col4.metric("Signal", signal_text)

    fig = make_chart_figure(signals, portfolio, PLOT_LAST_N_DAYS)
    st.pyplot(fig)

    st.subheader("Backtest performance")
    colA, colB = st.columns(2)
    colA.metric("Portfolio Value", f"{portfolio_value:,.2f}", f"{pnl_pct:+.2f}%")
    colB.metric("Started at", f"{portfolio_start:,.2f}")

    st.subheader("Recent signals (last 10 trading days)")
    recent = signals.iloc[-10:][["price", "regression_line", "buy_signal", "sell_signal"]].copy()
    recent["signal"] = recent.apply(
        lambda r: "BUY" if r["buy_signal"] else ("SELL" if r["sell_signal"] else "-"), axis=1
    )
    st.dataframe(recent[["price", "regression_line", "signal"]], use_container_width=True)

    st.divider()
    if st.button("📨 Send current summary + chart to Telegram"):
        try:
            summary_text = (
                f"NIFTY Mean Reversion — {latest_date}\n"
                f"Price: {latest['price']:.2f}  Regression: {latest['regression_line']:.2f}\n"
                f"Signal: {signal_text}\n"
                f"Portfolio: {portfolio_value:,.2f} ({pnl_pct:+.2f}%)"
            )
            send_telegram_message(summary_text)
            send_telegram_photo(fig, caption=f"{TICKER} chart — {latest_date}")
            st.success("Sent to Telegram.")
        except Exception as e:
            st.error(f"Telegram send failed: {e}")

except Exception as e:
    st.error(f"Data fetch or computation failed: {e}")
    st.info("This can happen on market holidays or if Yahoo Finance is briefly rate-limiting. Try again shortly.")
