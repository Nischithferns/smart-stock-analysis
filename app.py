import requests
from textblob import TextBlob
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go

st.set_page_config(page_title="Smart Stock Analysis", layout="wide")

# ================= 🎨 GROWW STYLE UI =================
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.metric-card {
    padding: 15px;
    border-radius: 12px;
    background-color: #111827;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Smart Stock Analysis Dashboard")

# ================= 🔐 API KEY =================
API_KEY = "9a54cb178a9b41dc9ea4a9298da78f05"

# ================= LOAD NSE DATA =================
@st.cache_data
def load_stock_list():
    df = pd.read_csv("EQUITY_L.csv")
    df.columns = df.columns.str.strip()

    df['name'] = df['NAME OF COMPANY']
    df['symbol'] = df['SYMBOL'] + ".NS"
    df['keywords'] = df['name'].str.lower()

    return df[['name', 'symbol', 'keywords']]

stocks_df = load_stock_list()

# ================= 🔍 SEARCH =================
search_query = st.text_input("🔍 Search stocks like Groww (Airtel, Adani, Tata...)")

stock_symbol = None

if search_query:
    query = search_query.lower()

    results = stocks_df[
        stocks_df['name'].str.lower().str.contains(query, na=False) |
        stocks_df['symbol'].str.lower().str.contains(query, na=False) |
        stocks_df['keywords'].str.contains(query, na=False)
    ].copy()

    results['score'] = (
        results['name'].str.lower().str.startswith(query).astype(int) * 3 +
        results['symbol'].str.lower().str.startswith(query).astype(int) * 2
    )

    results = results.sort_values(by='score', ascending=False)

    if not results.empty:
        options = results['name'] + " (" + results['symbol'] + ")"
        selected = st.selectbox("Select Stock", options)
        stock_symbol = selected.split("(")[-1].replace(")", "")
    else:
        st.warning("No matching stocks found")

# ================= LOAD STOCK DATA =================
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="1y")

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    return df

# ================= NEWS =================
def get_news(query):
    try:
        if "PASTE" in API_KEY:
            return ["⚠️ Add your NewsAPI key"], 0

        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}"
        res = requests.get(url).json()

        headlines = []
        sentiments = []

        for article in res.get("articles", [])[:5]:
            title = article.get("title")
            if title:
                headlines.append(title)
                sentiments.append(TextBlob(title).sentiment.polarity)

        return headlines, np.mean(sentiments) if sentiments else 0

    except:
        return ["Error fetching news"], 0

# ================= MAIN =================
if stock_symbol:

    data = load_data(stock_symbol)

    if data.empty:
        st.error("❌ No data found")
        st.stop()

    current_price = float(data['Close'].iloc[-1])
    open_price = float(data['Open'].iloc[-1])

    # ================= METRICS =================
    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Current Price", f"₹ {current_price:.2f}")
    col2.metric("📊 Opening Price", f"₹ {open_price:.2f}")
    col3.metric("📅 Data Points", len(data))

    # ================= INDICATORS =================
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()

    data['Signal'] = 0
    data.loc[data['MA50'] > data['MA200'], 'Signal'] = 1
    data.loc[data['MA50'] < data['MA200'], 'Signal'] = -1

    # ================= LSTM =================
    close_data = data['Close'].values.reshape(-1,1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data)

    if len(scaled_data) < 60:
        st.warning("Not enough data")
        st.stop()

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    @st.cache_resource
    def train_model(X, y):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60,1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=3, batch_size=32, verbose=0)
        return model

    with st.spinner("🤖 AI analyzing..."):
        model = train_model(X, y)

    last_60 = scaled_data[-60:]
    X_test = np.reshape(last_60, (1,60,1))

    predicted = model.predict(X_test, verbose=0)
    predicted_price = float(scaler.inverse_transform(predicted)[0][0])

    # ================= PREDICTION UI =================
    st.subheader("📊 Next Day Prediction")

    confidence = abs(predicted_price - current_price) / current_price * 100

    col1, col2 = st.columns(2)

    col1.metric("🔮 Predicted Price", f"₹ {predicted_price:.2f}")

    if predicted_price > current_price:
        col2.success(f"📈 BUY Signal\nConfidence: {confidence:.2f}%")
    else:
        col2.error(f"📉 SELL Signal\nConfidence: {confidence:.2f}%")

    # ================= PRICE DIFFERENCE =================
    diff = predicted_price - current_price

    if diff > 0:
        st.success(f"Expected Increase: ₹ {diff:.2f}")
    else:
        st.error(f"Expected Decrease: ₹ {abs(diff):.2f}")

    # ================= CHART =================
    st.subheader("📈 Price Chart")

    df_chart = data.reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_chart['Date'],
        y=df_chart['Close'],
        mode='lines',
        line=dict(color='#00c853', width=2)
    ))

    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================= NEWS =================
    news, sentiment = get_news(search_query)

    st.subheader("📰 Market Sentiment")

    if sentiment > 0:
        st.success("🟢 Positive Market")
    elif sentiment < 0:
        st.error("🔴 Negative Market")
    else:
        st.warning("🟡 Neutral Market")

    for n in news:
        st.markdown(f"- {n}")

# ================= DISCLAIMER =================
st.markdown("---")
st.warning(
    "⚠️ Disclaimer: This application is for educational purposes only. "
    "Predictions are generated using AI/ML models and should NOT be considered financial advice."
)