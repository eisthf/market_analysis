
import importlib
import streamlit as st
import market
importlib.reload(market)
from market import update_stock_price_db_iter, strong_stocks_iter, get_ohlc
import pandas as pd

from mplchart.chart import Chart
from mplchart.primitives import Candlesticks, Volume
from mplchart.indicators import ROC, SMA, EMA, RSI, MACD
import matplotlib.pyplot as plt

if 'inited' not in st.session_state:
    st.session_state.inited = False
    st.session_state.market = "KOSPI"
    st.session_state.rate = dict(kospi=0.0, kosdaq=0.0)
    st.session_state.strong_stock = dict(kospi=None, kosdaq=None)
    st.session_state.index = dict(kospi=0, kosdaq=0)


@st.cache_data
def _get_ohlc(code: str):
    df = get_ohlc(code=code)
    return df


@st.fragment
def _update_stock_price_db():
    def update():
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        for (i, n) in update_stock_price_db_iter():
            my_bar.progress(i/n, text= progress_text + f"({i} / {n})")
        my_bar.empty()
    st.button("Update DB", on_click=update)


@st.cache_data(show_spinner=False)
def get_strong_stock_list(market: str):
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    it = strong_stocks_iter(market)
    for (i, n) in it:
        if isinstance(i, int):
            my_bar.progress(i/n, text= progress_text + f"({i+1} / {n})")
    my_bar.empty()
    rate, df = i, n
    return rate, df


def update_strong_stock():
    for market in ["kospi", "kosdaq"]:
        rate, df = get_strong_stock_list(market)
        st.session_state.rate[market] = rate
        st.session_state.strong_stock[market] = df 
    st.session_state.inited = True


@st.fragment
def _strong_stocks():
    st.button("Load strong stocks", on_click=update_strong_stock)



@st.cache_resource
def get_chart(code: str):
    df = _get_ohlc(code)
    indicators = [
        Candlesticks(), SMA(5), SMA(10), SMA(20), SMA(60), SMA(120), SMA(240), Volume(),
        RSI(14),
        MACD(5,20,5),
    ]

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    
    fig, ax = plt.subplots()
    chart = Chart(title=code, max_bars=300, figure=fig)
    chart.plot(df, indicators)
    return fig


def stock_nav_btn(direction: int):
    if st.session_state.inited == False:
        st.warning("Load strong stock first!", icon="⚠️")
        return
    
    market = st.session_state.market.lower()
    n = st.session_state.strong_stock[market].shape[0]
    if st.session_state.inited:
        st.session_state.index[market] += direction
        st.session_state.index[market] %= n
    else:
        st.session_state.inited = True
    
    index = st.session_state.index[market]
    code = st.session_state.strong_stock[market].iloc[index, :]["code"]
    fig = get_chart(code)
    st.pyplot(fig)


with st.sidebar:
    st.radio("Market", ["KOSPI", "KOSDAQ"], key="market")

    cols = st.columns(2)
    cols[0].button("Prev", on_click=stock_nav_btn, args=(-1, ))
    cols[1].button("Next", on_click=stock_nav_btn, args=(1, ))
    
    _strong_stocks()
    _update_stock_price_db()
