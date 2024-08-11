
import importlib
import streamlit as st
import market
importlib.reload(market)
from market import update_stock_price_db_iter, strong_stocks_iter, get_ohlc

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

        for (i, n) in update_stock_price_db_iter(True):
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


def stock_nav_btn(direction: int):
    if st.session_state.inited == False:
        st.write("Load strong stock first!")
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
    df = _get_ohlc(code)
    st.write(df.iloc[0, :][["Code", "Market"]])


with st.sidebar:
    st.radio("Market", ["KOSPI", "KOSDAQ"], key="market")

    cols = st.columns(2)
    cols[0].button("Prev", on_click=stock_nav_btn, args=(-1, ))
    cols[1].button("Next", on_click=stock_nav_btn, args=(1, ))
    
    _strong_stocks()
    _update_stock_price_db()
