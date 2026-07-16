import io
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings
import streamlit as st

# Suppress minor pandas fragmentation warnings for clean output
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

st.set_page_config(page_title="Mean Reversion Strategy Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# 0. Index constituent lists (for the index + ticker dropdowns)
# ---------------------------------------------------------------------------

# Display name -> niftyindices.com constituent CSV filename.
# Base URLs are tried in order in _index_csv_urls().
INDEX_CSV_FILENAMES = {
    # Broad market
    "NIFTY 50": "ind_nifty50list.csv",
    "NIFTY Next 50": "ind_niftynext50list.csv",
    "NIFTY 100": "ind_nifty100list.csv",
    "NIFTY 200": "ind_nifty200list.csv",
    "NIFTY 500": "ind_nifty500list.csv",
    "NIFTY Midcap 50": "ind_niftymidcap50list.csv",
    "NIFTY Midcap 150": "ind_niftymidcap150list.csv",
    "NIFTY Smallcap 50": "ind_niftysmallcap50list.csv",
    "NIFTY Smallcap 250": "ind_niftysmallcap250list.csv",
    "NIFTY Midsmallcap 400": "ind_niftymidsmallcap400list.csv",
    # Sectoral
    "NIFTY Auto": "ind_niftyautolist.csv",
    "NIFTY Bank": "ind_niftybanklist.csv",
    "NIFTY Financial Services": "ind_niftyfinancelist.csv",
    "NIFTY FMCG": "ind_niftyfmcglist.csv",
    "NIFTY IT": "ind_niftyitlist.csv",
    "NIFTY Media": "ind_niftymedialist.csv",
    "NIFTY Metal": "ind_niftymetallist.csv",
    "NIFTY Pharma": "ind_niftypharmalist.csv",
    "NIFTY Private Bank": "ind_nifty_privatebanklist.csv",
    "NIFTY PSU Bank": "ind_niftypsubanklist.csv",
    "NIFTY Realty": "ind_niftyrealtylist.csv",
    # Thematic
    "NIFTY Commodities": "ind_niftycommoditieslist.csv",
    "NIFTY CPSE": "ind_niftycpselist.csv",
    "NIFTY Energy": "ind_niftyenergylist.csv",
    "NIFTY India Consumption": "ind_niftyconsumptionlist.csv",
    "NIFTY Infrastructure": "ind_niftyinfralist.csv",
    "NIFTY PSE": "ind_niftypselist.csv",
}

# Small, hand-maintained fallback (NIFTY 50) used only if the live fetch below
# fails for every index/base-URL combination (e.g. NSE/niftyindices blocking
# the request, or no internet access) — keeps the app usable regardless.
_FALLBACK_CONSTITUENTS = [
    ("RELIANCE", "Reliance Industries Ltd."), ("TCS", "Tata Consultancy Services Ltd."),
    ("HDFCBANK", "HDFC Bank Ltd."), ("ICICIBANK", "ICICI Bank Ltd."),
    ("INFY", "Infosys Ltd."), ("BHARTIARTL", "Bharti Airtel Ltd."),
    ("ITC", "ITC Ltd."), ("LT", "Larsen & Toubro Ltd."),
    ("KOTAKBANK", "Kotak Mahindra Bank Ltd."), ("AXISBANK", "Axis Bank Ltd."),
    ("SBIN", "State Bank of India"), ("HINDUNILVR", "Hindustan Unilever Ltd."),
    ("BAJFINANCE", "Bajaj Finance Ltd."), ("ASIANPAINT", "Asian Paints Ltd."),
    ("MARUTI", "Maruti Suzuki India Ltd."), ("TITAN", "Titan Company Ltd."),
    ("SUNPHARMA", "Sun Pharmaceutical Industries Ltd."), ("TATAMOTORS", "Tata Motors Ltd."),
    ("TATASTEEL", "Tata Steel Ltd."), ("WIPRO", "Wipro Ltd."),
    ("ULTRACEMCO", "UltraTech Cement Ltd."), ("NESTLEIND", "Nestle India Ltd."),
    ("NTPC", "NTPC Ltd."), ("POWERGRID", "Power Grid Corporation of India Ltd."),
    ("HCLTECH", "HCL Technologies Ltd."), ("ADANIENT", "Adani Enterprises Ltd."),
    ("TATAELXSI", "Tata Elxsi Ltd."), ("M&M", "Mahindra & Mahindra Ltd."),
    ("JSWSTEEL", "JSW Steel Ltd."), ("ONGC", "Oil & Natural Gas Corporation Ltd."),
]


def _index_csv_urls(filename):
    return [
        f"https://niftyindices.com/IndexConstituent/{filename}",
        f"https://archives.nseindia.com/content/indices/{filename}",
    ]


@st.cache_data(ttl=86400, show_spinner=False)
def load_index_constituents(index_name):
    """
    Fetch the live constituent list (symbol + company name) for the given index.
    Falls back to a small hardcoded NIFTY 50 list if every source fails, so the
    app never breaks even if NSE/niftyindices blocks or renames the CSV.
    Returns (pairs, is_live) where pairs is a list of (symbol, company_name)
    tuples sorted by symbol, and is_live indicates whether the live fetch worked.
    """
    filename = INDEX_CSV_FILENAMES.get(index_name)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/csv,*/*",
    }
    if filename:
        for url in _index_csv_urls(filename):
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                df.columns = [c.strip() for c in df.columns]
                symbol_col = next(c for c in df.columns if c.lower() == "symbol")
                name_col = next(
                    (c for c in df.columns if "company" in c.lower() or "name" in c.lower()),
                    symbol_col,
                )
                pairs = list(zip(
                    df[symbol_col].astype(str).str.strip(),
                    df[name_col].astype(str).str.strip(),
                ))
                if len(pairs) >= 5:  # sanity check we actually got real data
                    return sorted(pairs, key=lambda x: x[0]), True
            except Exception:
                continue

    # Live fetch failed everywhere — use the fallback list
    return sorted(_FALLBACK_CONSTITUENTS, key=lambda x: x[0]), False


# ---------------------------------------------------------------------------
# 1. Fetch historical stock data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Handle new yfinance (v0.2.40+) MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        return data['Close'][ticker]
    else:
        return data['Close']

# ---------------------------------------------------------------------------
# 2. Linear regression-based mean reversion strategy
# ---------------------------------------------------------------------------
def linear_regression_mean_reversion_strategy(prices, window=50, threshold=2):
    signals = pd.DataFrame(index=prices.index)
    signals['price'] = prices

    # Initialize columns to avoid Pandas fragmentation warnings
    signals['regression_line'] = np.nan
    signals['standard_error'] = np.nan
    signals['deviation'] = np.nan
    signals['upper_2se'] = np.nan
    signals['lower_2se'] = np.nan

    for i in range(window, len(prices)):
        # Use .iloc for numerical positional indexing
        y = prices.iloc[i-window:i].values.reshape(-1, 1)
        X = np.arange(window).reshape(-1, 1)
        model = LinearRegression().fit(X, y)

        # [0][0] extracts the scalar value from the 2D array
        regression_line = model.predict(np.array([[window-1]]))[0][0]

        # Calculate residuals and standard error
        residuals = y - model.predict(X)
        residual_sum_of_squares = np.sum(residuals**2)
        standard_error = np.sqrt(residual_sum_of_squares / (window - 2))

        # Using .loc for assignment by date
        current_date = signals.index[i]
        signals.loc[current_date, 'regression_line'] = regression_line
        signals.loc[current_date, 'standard_error'] = standard_error
        signals.loc[current_date, 'deviation'] = signals['price'].iloc[i] - regression_line
        signals.loc[current_date, 'upper_2se'] = regression_line + 2 * standard_error
        signals.loc[current_date, 'lower_2se'] = regression_line - 2 * standard_error

    # Calculate signals
    signals['buy_signal'] = signals['deviation'] < -threshold * signals['standard_error']
    signals['sell_signal'] = signals['deviation'] > threshold * signals['standard_error']

    signals = signals.dropna()

    return signals

# ---------------------------------------------------------------------------
# 3. Backtesting the strategy
# ---------------------------------------------------------------------------
def backtest_strategy(signals, initial_capital=10000):
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    portfolio = pd.DataFrame(index=signals.index).fillna(0.0)

    positions['stock'] = 0.0
    current_position = 0.0  # Track holding state day-to-day

    for i in range(1, len(signals)):
        # Check signals using .iloc
        if signals['buy_signal'].iloc[i]:
            current_position = initial_capital // signals['price'].iloc[i]
        elif signals['sell_signal'].iloc[i]:
            current_position = 0.0

        # Carry the position forward to the current day
        positions.loc[positions.index[i], 'stock'] = current_position

    portfolio['positions'] = positions['stock'] * signals['price']

    # Calculate cash flows
    trade_flows = positions['stock'].diff().fillna(0.0) * signals['price']
    portfolio['cash'] = initial_capital - trade_flows.cumsum()
    portfolio['total'] = portfolio['positions'] + portfolio['cash']

    return portfolio

# ---------------------------------------------------------------------------
# 4. Plotting the results (returns a Figure instead of calling plt.show())
# ---------------------------------------------------------------------------
def plot_results(signals, portfolio, lookback_days=90):
    last_week_signals = signals.iloc[-lookback_days:]
    last_week_portfolio = portfolio.iloc[-lookback_days:]

    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 8))

    ax1.plot(last_week_signals.index, last_week_signals['price'], label='Price')
    ax1.plot(last_week_signals.index, last_week_signals['regression_line'], label='Regression Line', color='orange')

    # Fill between the regression line ± 2 * standard error
    ax1.fill_between(last_week_signals.index,
                      last_week_signals['lower_2se'],
                      last_week_signals['upper_2se'],
                      color='lightgrey', label='2 SE Band')

    # Plot signals
    buy_dates = last_week_signals[last_week_signals['buy_signal']].index
    buy_prices = last_week_signals.loc[buy_dates, 'price']
    ax1.scatter(buy_dates, buy_prices, label='Buy Signal', marker='^', color='green', s=100, zorder=5)

    sell_dates = last_week_signals[last_week_signals['sell_signal']].index
    sell_prices = last_week_signals.loc[sell_dates, 'price']
    ax1.scatter(sell_dates, sell_prices, label='Sell Signal', marker='v', color='red', s=100, zorder=5)

    ax1.legend()
    ax1.set_title(f'Linear Regression-Based Mean Reversion Strategy (Last {lookback_days} Days)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(last_week_portfolio.index, last_week_portfolio['total'], label='Portfolio Value', color='purple')
    ax2.set_title(f'Portfolio Value Over Time (Last {lookback_days} Days)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# 5. Displaying current data
# ---------------------------------------------------------------------------
def display_current_and_regression_prices(signals):
    current_price = signals['price'].iloc[-1]
    regression_price = signals['regression_line'].iloc[-1]
    standard_error = signals['standard_error'].iloc[-1]
    upper_2se = signals['upper_2se'].iloc[-1]
    lower_2se = signals['lower_2se'].iloc[-1]

    data = {
        'Current Price': [current_price],
        'Regression Price': [regression_price],
        'Standard Error': [standard_error],
        'Upper 2SE': [upper_2se],
        'Lower 2SE': [lower_2se]
    }
    df = pd.DataFrame(data)
    return df

# 5b. Displaying the last N days of signals (most recent first)
def display_recent_signals(signals, days=60):
    # Grab the last N days of data
    recent_df = signals.tail(days).copy()

    # Sort descending so the most recent date is at the top
    recent_df = recent_df.sort_index(ascending=False)

    # Create a unified 'Signal' column
    recent_df['Signal'] = '-'
    recent_df.loc[recent_df['buy_signal'], 'Signal'] = 'BUY'
    recent_df.loc[recent_df['sell_signal'], 'Signal'] = 'SELL'

    # Select and rename columns to match your screenshot
    display_df = recent_df[['price', 'regression_line', 'standard_error', 'upper_2se', 'lower_2se', 'Signal']]
    display_df = display_df.rename(columns={
        'price': 'Price',
        'regression_line': 'Regression',
        'standard_error': 'Std Error',
        'upper_2se': 'Upper 2SE',
        'lower_2se': 'Lower 2SE'
    })

    # Format the index to display as 'YYYY-MM-DD 00:00:00'
    display_df.index = display_df.index.strftime('%Y-%m-%d 00:00:00')
    display_df.index.name = 'Date'

    # Optional: Round numbers for cleaner console output
    display_df = display_df.round(4)

    return display_df

# ---------------------------------------------------------------------------
# 6. Streamlit app (replaces the original main() / CLI entry point)
# ---------------------------------------------------------------------------
def main():
    st.title("📈 Linear Regression Mean-Reversion Strategy Dashboard")
    st.caption("Same logic as the original script — now wrapped in a Streamlit UI.")

    with st.sidebar:
        st.header("Settings")

        index_name = st.selectbox(
            "Index",
            list(INDEX_CSV_FILENAMES.keys()),
            index=0,
        )
        constituents, is_live_list = load_index_constituents(index_name)
        options = [f"{sym} — {name}" for sym, name in constituents]

        if not is_live_list:
            st.caption(f"⚠️ Couldn't reach the live {index_name} list — showing a NIFTY 50 fallback.")

        default_idx = next(
            (i for i, (sym, _) in enumerate(constituents) if sym == "TATAELXSI"), 0
        )
        use_custom = st.checkbox("Enter a custom ticker instead", value=False)

        if use_custom:
            ticker = st.text_input("Ticker (yfinance format, e.g. RELIANCE.NS)", value="TATAELXSI.NS")
        else:
            selected = st.selectbox(
                f"Ticker ({index_name}{'' if is_live_list else ' — fallback'})",
                options,
                index=default_idx,
            )
            symbol = selected.split(" — ")[0]
            ticker = f"{symbol}.NS"

        start_date = st.date_input("Start date", value=datetime(2020, 1, 1))
        window = st.number_input("Regression window (days)", min_value=10, max_value=250, value=50, step=5)
        threshold = st.number_input("Signal threshold (× SE)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
        initial_capital = st.number_input("Initial capital", min_value=1000, value=10000, step=1000)
        lookback_days = st.slider("Chart lookback (days)", min_value=30, max_value=250, value=90, step=10)
        recent_days = st.slider("Signals table (days)", min_value=10, max_value=120, value=60, step=10)
        run = st.button("Run analysis", type="primary")

    if not run and "signals" not in st.session_state:
        st.info("Set your parameters in the sidebar and click **Run analysis** to get started.")
        return

    if run:
        # Add 1 day because the yfinance 'end' parameter is exclusive
        end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        start_date_str = start_date.strftime('%Y-%m-%d')

        with st.spinner(f"Fetching data for {ticker}..."):
            prices = fetch_stock_data(ticker, start_date_str, end_date)

        if prices.empty:
            st.error("No data returned. Check the ticker symbol and try again.")
            return

        with st.spinner("Calculating regression lines and trading signals..."):
            signals = linear_regression_mean_reversion_strategy(prices, window=window, threshold=threshold)

        with st.spinner("Running backtest..."):
            portfolio = backtest_strategy(signals, initial_capital=initial_capital)

        st.session_state["signals"] = signals
        st.session_state["portfolio"] = portfolio
        st.session_state["ticker"] = ticker

    signals = st.session_state["signals"]
    portfolio = st.session_state["portfolio"]
    ticker = st.session_state["ticker"]

    # --- Current snapshot ---
    st.subheader(f"Current snapshot — {ticker}")
    current_df = display_current_and_regression_prices(signals)
    st.dataframe(current_df, use_container_width=True, hide_index=True)

    # --- Charts ---
    st.subheader("Charts")
    fig = plot_results(signals, portfolio, lookback_days=lookback_days)
    st.pyplot(fig)

    # --- Recent signals table ---
    st.subheader(f"Recent signals (last {recent_days} trading days, most recent first)")
    recent_table = display_recent_signals(signals, days=recent_days)
    st.dataframe(recent_table, use_container_width=True)

    # --- Portfolio value ---
    final_value = portfolio['total'].iloc[-1]
    st.metric("Final portfolio value", f"{final_value:,.2f}", delta=f"{final_value - initial_capital:,.2f}")

if __name__ == "__main__":
    main()
