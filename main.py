
import importlib
import streamlit as st
import market
importlib.reload(market)
from market import update_stock_price_db_iter, strong_stocks_iter, get_ohlc
import pandas as pd
from pykrx import stock as krx
import numpy as np

from mplchart.chart import Chart
from mplchart.primitives import Candlesticks, Volume
from mplchart.indicators import ROC, SMA, EMA, RSI, MACD, BBANDS
import matplotlib.pyplot as plt
from utils import ENVELOPE, svg_write
from enum import Enum


class Mode(Enum):
    UNKNOWN = 0
    STRONG_STOCK = 1
    WON_VOLUME = 2
    TRADING_VOLUME_STOCK = 3


if 'inited' not in st.session_state:
    st.session_state.inited = False
    st.session_state.market = "KOSPI"
    st.session_state.rate = dict(kospi=0.0, kosdaq=0.0)
    st.session_state.strong_stock = dict(kospi=None, kosdaq=None)
    st.session_state.index = dict(kospi=0, kosdaq=0)
    st.session_state.index_begin = dict(kospi=0, kosdaq=0)
    st.session_state.index_end = dict(kospi=0, kosdaq=0)
    st.session_state.mode = Mode.UNKNOWN.value


SMALL_SIZE = 8*2
MEDIUM_SIZE = 10*2
BIGGER_SIZE = 12*2

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# from matplotlib_inline.backend_inline import set_matplotlib_formats
# set_matplotlib_formats('svg')

st.set_page_config(layout="wide")

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
    df['name'] = df['code'].map(lambda x: krx.get_market_ticker_name(x))
    return rate, df


def update_strong_stock():
    for market in ["kospi", "kosdaq"]:
        rate, df = get_strong_stock_list(market)
        df = df.sort_values('rate').reset_index(drop=True)
        st.session_state.rate[market] = rate
        st.session_state.strong_stock[market] = df
        st.session_state.index_begin[market] = 0
        st.session_state.index_end[market] = df.index[-1]
        print(st.session_state.strong_stock[market])
    st.session_state.inited = True
    st.session_state.mode = Mode.STRONG_STOCK.value
    market = st.session_state.market
    send_chart()

# @st.fragment
# def _strong_stocks():
#     if st.button("Load strong stocks", on_click=update_strong_stock):
#         st.session_state.mode = Mode.STRONG_STOCK
#         st.rerun()



@st.cache_resource
def get_chart(code: str, name: str, rate: float, market: str, max_bars: int=120):
    df = _get_ohlc(code)
    indicators = [
        Candlesticks(colorup='r', colordn='b', use_bars=False), 
        SMA(5), SMA(10), SMA(20), SMA(60), SMA(120), SMA(240),
        ENVELOPE(20, 0.1),
        Volume(colorup='r', colordn='b'),
        RSI(14),
        MACD(5,20,5),
 
    ]

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    
    fig, ax = plt.subplots(figsize=(24,12), dpi=100)

    index_rate = np.round(st.session_state.rate[market], 2)
    rate = np.round(rate, 2)
    chart = Chart(title=f"{market.upper()}[{index_rate}] / {code} / {name}[{rate}]", max_bars=max_bars, figure=fig)
    chart.plot(df, indicators)
    return fig


def send_chart():
    market = st.session_state.market.lower()
    index = st.session_state.index[market]
    item = st.session_state.strong_stock[market].iloc[index, :][["code", "name", "rate"]]
    fig_120 = get_chart(item["code"], item["name"], item["rate"], market, 120)
    fig_360 = get_chart(item["code"], item["name"], item["rate"], market, 360)
    svg_write(fig_120)
    svg_write(fig_360)


def stock_nav_btn(direction: int):
    if st.session_state.inited == False:
        st.warning("Load strong stock first!", icon="⚠️")
        return
    
    print(f"direction: {direction}")
    print(f"session_state.inited: {st.session_state.inited}")
    market = st.session_state.market.lower()
    if st.session_state.inited:
        st.session_state.index[market] += direction
        print(st.session_state.index[market])
        if st.session_state.index[market] < st.session_state.index_begin[market]:
            st.session_state.index[market] = st.session_state.index_end[market]
        elif st.session_state.index[market] > st.session_state.index_end[market]:
            st.session_state.index[market] = st.session_state.index_begin[market]
    else:
        st.session_state.inited = True

    send_chart()



def strong_stock_slider_change():
    min_, max_ = st.session_state.strong_rate
    market = st.session_state.market.lower()
    df = st.session_state.strong_stock[market]
    idx = df[(min_ <= df['rate']) & (df['rate'] <= max_)].index
    st.session_state.index_begin[market] = idx[0]
    st.session_state.index_end[market] = idx[-1]
    st.session_state.index[market] = idx[0]
    send_chart()


def strong_stock_range_slider():
    if st.session_state.mode == Mode.STRONG_STOCK.value:
        market = st.session_state.market.lower()
        df = st.session_state.strong_stock[market]
        rate = df["rate"]
        min_, max_ = rate.min(), rate.max()
        st.slider("Select a range of values", min_-1, max_+1, 
                   (min_, max_), on_change=strong_stock_slider_change, key="strong_rate")



with st.sidebar:
    st.radio("Market", ["KOSPI", "KOSDAQ"], key="market")
    strong_stock_range_slider()

    cols = st.columns(2)
    cols[0].button("Prev", on_click=stock_nav_btn, args=(-1, ))
    cols[1].button("Next", on_click=stock_nav_btn, args=(1, ))
    
    st.button("Load strong stocks", on_click=update_strong_stock)
    # _strong_stocks()
    _update_stock_price_db()

