
import importlib
import streamlit as st
import market
importlib.reload(market)
from market import (update_stock_price_db_iter, 
                    strong_stocks_iter,
                    update_krx_stock_price_db_iter)
import pandas as pd
from pykrx import stock as krx
import numpy as np
from datetime import datetime, date

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
    st.session_state.strong_stock_rate = dict(kospi=0.0, kosdaq=0.0)
    st.session_state.strong_stock = dict(kospi=None, kosdaq=None)
    st.session_state.won_vol_stock = pd.DataFrame|None
    st.session_state.strong_week = 52
    st.session_state.ref_date = date.today()
    st.session_state.show_today = False
    st.session_state.index = dict(kospi=0, kosdaq=0)
    st.session_state.index_begin = dict(kospi=0, kosdaq=0)
    st.session_state.index_end = dict(kospi=0, kosdaq=0)
    st.session_state.mode = Mode.UNKNOWN.value

    

g_mode = {
    Mode.UNKNOWN.value: "Unknown", 
    Mode.STRONG_STOCK.value: "지수보다 강한 종목", 
    Mode.WON_VOLUME.value: "거래 대금 상위", 
    Mode.TRADING_VOLUME_STOCK.value: "1000만 주", 
}


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
def get_ohlc(code: str):
    df = market.get_ohlc(code=code)
    return df



@st.cache_data
def get_market_cap():
    return (market.get_market_cap()['시가총액'].rename('market_cap') / 10**8).astype(int)



@st.fragment
def update_stock_price_db():
    def update():
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        for (i, n) in update_krx_stock_price_db_iter(): 
            my_bar.progress(i/n, text= progress_text + f"({i} / {n})")
        my_bar.empty()
    st.button("Update DB", on_click=update)


@st.cache_data(show_spinner=False)
def get_strong_stock_list(market: str, week:int, ref_date: str):
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    it = strong_stocks_iter(market, week, ref_date)
    for (i, n) in it:
        if isinstance(i, int):
            my_bar.progress(i/n, text= progress_text + f"({i+1} / {n})")
    my_bar.empty()
    rate, df = i, n
    df['name'] = df['code'].map(lambda x: krx.get_market_ticker_name(x))
    return rate, df


def update_strong_stock(week:int=52):
    ref_date = st.session_state.ref_date.strftime("%Y-%m-%d")
    for market in ["kospi", "kosdaq"]:
        rate, df = get_strong_stock_list(market, week, ref_date)
        df = df.sort_values('rate').reset_index(drop=True)
        st.session_state.strong_stock_rate[market] = rate
        st.session_state.strong_stock[market] = df
        st.session_state.index[market] = 0
        st.session_state.index_begin[market] = 0
        st.session_state.index_end[market] = df.index[-1]
    st.session_state.inited = True
    st.session_state.mode = Mode.STRONG_STOCK.value
    market = st.session_state.market
    st.session_state.strong_week = week



@st.cache_resource
def get_chart(code: str, name: str,  market: str, refdate: str, last_date: str,
              rate: float|None=None, max_bars: int=120):
    df = get_ohlc(code)
    indicators = [
        Candlesticks(colorup='r', colordn='b', use_bars=False), 
        SMA(5), SMA(10), SMA(20), SMA(60), SMA(120), SMA(240),
        ENVELOPE(20, 10),
        Volume(colorup='r', colordn='b'),
        RSI(14),
        MACD(5,20,5),
 
    ]

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df[df["Date"] <= last_date]
    refdate = df[df["Date"] <= refdate]["Date"].max()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    volume = int(df.iloc[-1]['Volume']/10000)
    volrate = np.round(df.iloc[-1]['Volume'] *100 / df.iloc[-2]['Volume'],2)
    fig, ax = plt.subplots(figsize=(24,12), dpi=100)
    market_cap = get_market_cap()[code]

    if st.session_state.mode == Mode.STRONG_STOCK.value:
        index_rate = np.round(st.session_state.strong_stock_rate[market], 2)
        rate = np.round(rate, 2)        
        title = f"{market.upper()}[{index_rate}%] \
/ {code} / {name}[{rate}%] / 시총: {market_cap}억 /현재가: {int(df.iloc[-1]['Close'])}\
/ 거래량: {int(df.iloc[-1]['Volume'])}/ 거래량: {volume}만 /거래량 증가: {volrate}%"
        
    elif st.session_state.mode == Mode.WON_VOLUME.value:
        df_won_vol = st.session_state.won_vol_stock
        won_vol = int(df_won_vol[df_won_vol["Code"]==code].iloc[0, :]["won_vol"]/1000000)
        title=f"{market.upper()} / {code} / {name} / 시총: {market_cap}억 /상승률={np.round(rate,2)}% / 거래대금:{won_vol}백만 / 거래량: {volume}만 /거래량 증가: {volrate}%"
    
    chart = Chart(title=title, max_bars=max_bars, figure=fig)
    chart.plot(df, indicators, refdate=refdate)
    return fig



def send_chart():
    refdate = st.session_state.ref_date.strftime("%Y-%m-%d")
    if not st.session_state.show_today:
        last_date = refdate
    else:
        last_date = date.today().strftime("%Y-%m-%d")
    if st.session_state.mode == Mode.STRONG_STOCK.value:
        market = st.session_state.market.lower()
        index = st.session_state.index[market]
        item = st.session_state.strong_stock[market].iloc[index, :][["code", "name", "rate"]]
        fig_120 = get_chart(item["code"], item["name"], market, refdate, last_date, item["rate"], 120)
        fig_360 = get_chart(item["code"], item["name"], market, refdate, last_date, item["rate"], 360)
    elif st.session_state.mode == Mode.WON_VOLUME.value:
        name = st.session_state.selected_won_vol_stock
        df = st.session_state.won_vol_stock  # Date, won_vol, Market, rate, name, Volume
        row = df[df["name"]==name].iloc[0, :]
        fig_120 = get_chart(row["Code"], row["name"], row["Market"], refdate, last_date, row["rate"], 120)
        fig_360 = get_chart(row["Code"], row["name"], row["Market"], refdate, last_date, row["rate"], 360)        
    if st.session_state.mode in [Mode.STRONG_STOCK.value, Mode.WON_VOLUME.value]:
        svg_write(fig_120)
        svg_write(fig_360)



def stock_nav_btn(direction: int):
    if st.session_state.inited == False:
        st.warning("Load strong stock first!", icon="⚠️")
        return
    
    market = st.session_state.market.lower()
    if st.session_state.inited:
        st.session_state.index[market] += direction
        if st.session_state.index[market] < st.session_state.index_begin[market]:
            st.session_state.index[market] = st.session_state.index_end[market]
        elif st.session_state.index[market] > st.session_state.index_end[market]:
            st.session_state.index[market] = st.session_state.index_begin[market]
            
        code = st.session_state.selected_strong_stock.split(' ')[0]   

        idx_begin = st.session_state.strong_stock_sel_list.index[0]
        idx_end = st.session_state.strong_stock_sel_list.index[-1]    
        l = st.session_state.strong_stock_sel_list
        idx = l[l.str.contains(code)].index[0] + direction
        if idx > idx_end:
            idx = idx_begin
        elif idx < idx_begin:
            idx = idx_end
        st.session_state.selected_strong_stock = st.session_state.strong_stock_sel_list[idx]
        
        # l = [x.split(' ')[0] for x in st.session_state.strong_stock_sel_list]
        # idx = l.index(code)
        # idx = (idx + direction) % len(l)
        # st.session_state.selected_strong_stock = st.session_state.strong_stock_sel_list.iloc[idx]
    else:
        st.session_state.inited = True



def strong_stock_slider_change():
    min_, max_ = st.session_state.strong_rate
    market = st.session_state.market.lower()
    df = st.session_state.strong_stock[market]
    idx = df[(min_ <= df['rate']) & (df['rate'] <= max_)].index
    st.session_state.index_begin[market] = idx[0]
    st.session_state.index_end[market] = idx[-1]
    st.session_state.index[market] = idx[0]



def on_sel_strong_stock():
    code = st.session_state.selected_strong_stock.split(' ')[0]
    market = st.session_state.market.lower()
    df = st.session_state.strong_stock[market]
    st.session_state.index[market] = df[df['code'] == code].index[0]



def on_change_strong_week():
    state = st.session_state
    update_strong_stock(state.strong_week)



def strong_stock_menu():
    if st.session_state.mode == Mode.STRONG_STOCK.value:
        market = st.radio("Market", ["KOSPI", "KOSDAQ"], horizontal=True, key="market").lower()
        df = st.session_state.strong_stock[market]
        rate = df["rate"]
        min_, max_ = rate.min(), rate.max()

        idx_begin = st.session_state.index_begin[market]
        idx_end = st.session_state.index_end[market]
        df = df.iloc[idx_begin:idx_end+1]

        st.slider("Select a range of values", min_-1, max_+1, 
                   (df['rate'].min(), df['rate'].max()), on_change=strong_stock_slider_change, key="strong_rate")
        
        l = st.session_state.strong_stock_sel_list = \
                df.apply(lambda x: f"{x['code']} {x['name']} {round(x['rate'])}", axis=1)
        st.selectbox("Select a stock", l, on_change=on_sel_strong_stock, key='selected_strong_stock')        
        cols = st.columns(2)
        cols[0].button("Prev", on_click=stock_nav_btn, args=(-1, ))
        cols[1].button("Next", on_click=stock_nav_btn, args=(1, ))
        st.number_input("Week periods", 
                         min_value=1, max_value=52,                         
                         on_change=on_change_strong_week,
                         key="strong_week")



def on_refdate_change():
    date_: str = st.session_state.ref_date.strftime("%Y-%m-%d")
    if st.session_state.mode == Mode.WON_VOLUME.value:
        df = get_won_volume_stock(date_)
        st.session_state.selected_won_vol_stock = df.loc[df.index[0], "name"]
        st.session_state.won_vol_stock = df
    elif st.session_state.mode == Mode.STRONG_STOCK.value:
        update_strong_stock(st.session_state.strong_week)



def won_volume_stock_menu():
    if st.session_state.mode == Mode.WON_VOLUME.value:
        df = st.session_state.won_vol_stock                
        st.selectbox("Select a stock", df['name'], key='selected_won_vol_stock')


@st.cache_data
def get_won_volume_stock(date_: str|None=None):
    df = pd.concat([market.get_2days_prices('kospi', date_), market.get_2days_prices('kosdaq', date_)], axis=0)
    df = df.groupby('Code').filter(lambda x: len(x)>1)
    df = df.sort_values(["Code", "Date"])
    d1, d0 = df['Date'].unique()
    st.session_state.ref_date = datetime.strptime(d0, "%Y-%m-%d")
    df_today = df[df['Date']==d0].pipe(lambda x: x.assign(won_vol=lambda x: 0.5*(x['High'] + x['Low'])*x['Volume']))\
                                .sort_values('won_vol', ascending=False).iloc[:30, :].set_index("Code")

    df_rate = df.groupby('Code')['Close'].apply(lambda x: 100*(x.values[1]-x.values[0])/x.values[0])\
                                        .rename('rate').to_frame()
    df = df_today[['won_vol', 'Market', 'Volume']].join(df_rate['rate'])\
                        .sort_values('won_vol', ascending=False)[:30]\
                        .pipe(lambda x: x[x['rate'] >= 5]).reset_index(drop=False)    
    df['name'] = df['Code'].map(lambda x: krx.get_market_ticker_name(x))
    return df # Date, won_vol, Market, rate, name, Volume



def on_won_volume_stock():
    st.session_state.mode = Mode.WON_VOLUME.value
    refdate = st.session_state.ref_date
    df = get_won_volume_stock(refdate) # Date, won_vol, Market, rate, name, Volume
    st.session_state.won_vol_stock = df
    st.session_state.selected_won_vol_stock = df["name"][0]



def on_1000_stock():
    st.session_state.mode = Mode.TRADING_VOLUME_STOCK.value



with st.sidebar:
    st.subheader("{mode}".format(mode=g_mode[st.session_state.mode]))
    strong_stock_menu()
    won_volume_stock_menu()
    
    st.divider()
    st.date_input("날짜", date.today(), on_change=on_refdate_change, key="ref_date")
    st.checkbox("today", key="show_today")
    st.button("지수보다 강한 종목", on_click=update_strong_stock)
    st.button("거래대금 상위 종목", on_click=on_won_volume_stock)
    st.button("1000만주 이상 종목", on_click=on_1000_stock)
    st.divider()
    update_stock_price_db()


send_chart()