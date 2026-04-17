import requests
from textblob import TextBlob
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Smart Stock Analysis", layout="wide")

# ================= UI STYLE =================
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Smart Stock Analysis Dashboard")

# ================= API KEY =================
API_KEY = os.getenv("NEWS_API_KEY")

# ================= LOAD STOCK LIST =================
@st.cache_data
def load_stock_list():
    df = pd.read_csv("EQUITY_L.csv")
    df.columns = df.columns.str.strip()

    df['name'] = df['NAME OF COMPANY']
    df['symbol'] = df['SYMBOL'] + ".NS"
    df['keywords'] = df['name'].str.lower()

    return df[['name', 'symbol', 'keywords']]

stocks_df = load_stock_list()

# ================= SEARCH =================
search_query = st.text_input("🔍 Search stocks (Airtel, Adani, Tata...)")

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
        if not API_KEY:
            return ["⚠️ Add your News API key in Streamlit secrets"], 0

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
        st.error("❌ No data found for this stock")
        st.stop()

    current_price = float(data['Close'].iloc[-1])
    open_price = float(data['Open'].iloc[-1])

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Current Price", f"₹ {current_price:.2f}")
    col2.metric("📊 Opening Price", f"₹ {open_price:.2f}")
    col3.metric("📅 Data Points", len(data))

    # ================= SIMPLE ML MODEL =================
    data['Days'] = np.arange(len(data))

    X = data[['Days']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    next_day = np.array([[len(data)]])
    predicted_price = float(model.predict(next_day)[0])

    # ================= PREDICTION =================
    st.subheader("📊 Next Day AI Prediction")

    confidence = abs(predicted_price - current_price) / current_price * 100

    col1, col2 = st.columns(2)

    col1.metric("🔮 Predicted Price", f"₹ {predicted_price:.2f}")

    if predicted_price > current_price:
        col2.success(f"📈 BUY Signal\nConfidence: {confidence:.2f}%")
    else:
        col2.error(f"📉 SELL Signal\nConfidence: {confidence:.2f}%")

    # ================= CHANGE =================
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
        line=dict(color='#00c853', width=2),
        name='Price'
    ))

    # Prediction point
    future_date = df_chart['Date'].iloc[-1] + pd.Timedelta(days=1)

    fig.add_trace(go.Scatter(
        x=[future_date],
        y=[predicted_price],
        mode='markers',
        marker=dict(color='yellow', size=10),
        name='Prediction'
    ))

    fig.update_layout(
        template="plotly_dark",
        height=600
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
    "Predictions are AI-based and should not be considered financial advice."
)
