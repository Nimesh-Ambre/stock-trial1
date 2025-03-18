import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM # type: ignore


# Set up Streamlit app
st.set_page_config(page_title="Stock Market Data analysis", layout="wide")
st.title("📈 Stock Market Data analysis")
st.markdown("### A user-friendly dashboard to analyze sector performance, risk, and stock trends in India")

# Fetch live stock data from Yahoo Finance
@st.cache_data
def fetch_stock_list():
    try:
        stock_symbols = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'].tolist()
        return pd.DataFrame({"Ticker": stock_symbols})
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

stock_list = fetch_stock_list()
if stock_list.empty:
    st.error("⚠️ No stock data available.")
    st.stop()

# Sidebar for user selection
time_range = st.sidebar.selectbox("Select Time Range", ["1d", "3mo", "6mo", "1y", "5y", "10y", "max"])
selected_stock = st.sidebar.selectbox("Choose a Stock", stock_list["Ticker"].unique())

@st.cache_data
def get_stock_data(ticker, period):
    try:
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            raise ValueError(f"No historical data found for {ticker}")
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

stock_data = get_stock_data(selected_stock, time_range)
if stock_data.empty:
    st.warning("⚠️ No data available for the selected stock.")
    st.stop()

# Display stock chart
st.subheader(f"📊 {selected_stock} Stock Performance")
fig = px.line(stock_data, x=stock_data.index, y='Close', title=f"{selected_stock} Closing Prices")
st.plotly_chart(fig)

# Compute Technical Indicators
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
stock_data['Volatility'] = stock_data['Close'].pct_change().rolling(20).std()
stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
stock_data['Signal Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

# Relative Strength Index (RSI)
avg_gain = stock_data['Close'].diff().where(stock_data['Close'].diff() > 0, 0).rolling(14).mean().fillna(0)
avg_loss = -stock_data['Close'].diff().where(stock_data['Close'].diff() < 0, 0).rolling(14).mean().fillna(0)
rs = avg_gain / avg_loss.replace(0, 1e-6)
stock_data['RSI'] = 100 - (100 / (1 + rs))

# Determine trend direction
trend_note = ""
if stock_data['SMA_50'].iloc[-1] > stock_data['SMA_200'].iloc[-1]:
    trend_note = "🔼 Bullish Trend: Short-term moving average is above long-term moving average."
elif stock_data['SMA_50'].iloc[-1] < stock_data['SMA_200'].iloc[-1]:
    trend_note = "🔽 Bearish Trend: Short-term moving average is below long-term moving average."
st.markdown(f"**📌 Trend Analysis:** {trend_note}")

# AI-Based Stock Recommendations
st.subheader("📊 AI-Based Stock Recommendation")

def stock_recommendation(stock_data):
    try:
        if stock_data.empty or len(stock_data) < 200:  # Ensure enough data
            return "⚠️ Not enough data for analysis"

        latest_rsi = stock_data['RSI'].iloc[-1]
        latest_macd = stock_data['MACD'].iloc[-1]
        latest_signal = stock_data['Signal Line'].iloc[-1]
        latest_volatility = stock_data['Volatility'].iloc[-1]
        latest_sma_50 = stock_data['SMA_50'].iloc[-1]
        latest_sma_200 = stock_data['SMA_200'].iloc[-1]

        # Ensure enough previous data exists
        if len(stock_data) < 2:
            return "⚠️ Insufficient data for trend analysis"

        previous_sma_50 = stock_data['SMA_50'].iloc[-2]
        previous_sma_200 = stock_data['SMA_200'].iloc[-2]

        # Check Golden Cross and Death Cross conditions
        golden_cross = previous_sma_50 < previous_sma_200 and latest_sma_50 > latest_sma_200
        death_cross = previous_sma_50 > previous_sma_200 and latest_sma_50 < latest_sma_200
        macd_histogram = latest_macd - latest_signal

        if latest_rsi > 70 and latest_macd < latest_signal and death_cross:
            return "📉 Strong Sell - Overbought, Bearish MACD & Death Cross"
        elif latest_rsi < 30 and latest_macd > latest_signal and golden_cross:
            return "📈 Strong Buy - Oversold, Bullish MACD & Golden Cross"
        elif macd_histogram > 0 and latest_sma_50 > latest_sma_200:
            return "📈 Buy - Bullish MACD & Uptrend"
        elif macd_histogram < 0 and latest_sma_50 < latest_sma_200:
            return "📉 Sell - Bearish MACD & Downtrend"
        elif latest_volatility > stock_data['Volatility'].quantile(0.75):
            return "⚠️ Hold - High Volatility Detected"
        else:
            return "🔄 Hold - Weak or No Strong Signal"
    except Exception as e:
        return f"Error in recommendation: {e}"

# LSTM Stock Price Prediction
@st.cache_resource
def build_lstm_model():
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
        LSTM(units=50, return_sequences=False),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_lstm_model()

# Prepare Data for Prediction
def prepare_data(stock_data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close']].values)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

X, y, scaler = prepare_data(stock_data)
model.fit(X, y, epochs=10, batch_size=32)

# Predict for Next 30 Days
future_prices = []
input_data = X[-1].reshape(1, 60, 1)
for _ in range(30):
    predicted_price = model.predict(input_data)
    future_prices.append(predicted_price[0][0])
    input_data = np.append(input_data[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)
future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1)).flatten()

# Display Predictions
st.subheader("🔮 Predicted Stock Prices for Next 30 Days")
st.line_chart(pd.DataFrame(future_prices, columns=['Predicted Price']))
