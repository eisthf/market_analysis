import pandas as pd
from pykrx import stock as krx
import FinanceDataReader as fdr
from datetime import datetime, date, timedelta

from collections import namedtuple
from tqdm import tqdm
import sqlite3
from sqlite3 import Connection, Cursor, IntegrityError
from typing import NamedTuple
from pydantic import BaseModel


DBFILE = 'stock_prices.db'

def resample_df(df, period='W-FRI'):
    wm_df = pd.DataFrame()
    wm_df = pd.concat([wm_df, df[['Open']].resample(period).first()], axis=1)
    wm_df = pd.concat([wm_df, df[['High']].resample(period).max()], axis=1)
    wm_df = pd.concat([wm_df, df[['Low']].resample(period).min()], axis=1)
    wm_df = pd.concat([wm_df, df[['Close']].resample(period).last()], axis=1)
    wm_df = pd.concat([wm_df, df[['Volume']].resample(period).mean()], axis=1)
    return wm_df


class StockPrice(BaseModel):
    code: str
    market: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def db_insert(curs: Cursor, info: StockPrice):
    qry = "insert into StockPrice values" \
          "(:code, :market, :date, :open, :high, :low, :close, :volume)"
    params = info.model_dump()
    curs.execute(qry, params)


def get_ohlc(*, curs: Cursor|None=None, code: str, start_date: str|None=None, end_date: str|None=None):
    conn: Connection|None = None
    if not curs:
        conn = sqlite3.connect(DBFILE)
        curs = conn.cursor()
        
    qry = "select * from StockPrice where code=:code "
    if start_date and end_date:
        qry += f"AND date BETWEEN '{start_date}' and '{end_date}'"
    params = {"code": code}
    curs.execute(qry, params)
    res = curs.fetchall()
    df = pd.DataFrame(res, columns=["Code", "Market", "Date", "Open", "High", "Low", "Close", "Volume"])
    if conn:
        conn.close()
    return df


def get_stock_list(*, curs: Cursor, market: str):
    conn: Connection|None = None
    if not curs:
        conn = sqlite3.connect(DBFILE)
        curs = conn.cursor()
    qry = f"select DISTINCT code from StockPrice where market='{market}'"
    curs.execute(qry)
    res = curs.fetchall() 
    if conn:
        conn.close()
    return res


def build_stock_price_db(start_date: str, end_date: str):
    conn = sqlite3.connect(DBFILE)
    curs = conn.cursor()
    curs.execute("create table if not exists \
    StockPrice(code text, market text, \
    date text, open, high, low, close, volume, constraint pkey primary key(code, date))")
    def build(stock_list: list, market: str):
        for code in tqdm(stock_list, desc=market):
            df = fdr.DataReader(code, start_date, end_date).reset_index()
            for _, row in df.iterrows():
                sp = StockPrice(code=code, 
                                market=market, 
                                date=row['Date'].strftime('%Y-%m-%d'), 
                                open=row['Open'],
                                high=row['High'],
                                low=row['Low'],
                                close=row['Close'],
                                volume=row['Volume'])
                try:
                    db_insert(curs, sp)                    
                except IntegrityError:
                    pass
    build(fdr.StockListing('KOSPI')['Code'], 'kospi')
    build(['KS11'], 'KS11')
    build(fdr.StockListing('KOSDAQ')['Code'], 'kosdaq')
    build(['KQ11'], 'KQ11')
    conn.commit()
    conn.close()


def update_stock_price_db():
    conn = sqlite3.connect(DBFILE)
    curs = conn.cursor()
    qry = "select DISTINCT code, market from StockPrice"
    curs.execute(qry)
    res = curs.fetchall() 
    for code, market in tqdm(res):
        qry = "select MAX(date) from StockPrice where code=:code"
        params = {"code": code}
        curs.execute(qry, params)
        res = curs.fetchall()
        start_date = res[0][0]
        today = date.today()
        if today == start_date:
            continue
        df = fdr.DataReader(code, start_date, today.strftime("%Y-%m-%d")).reset_index()
        for _, row in df.iterrows():
            sp = StockPrice(code=code, 
                            market=market, 
                            date=row['Date'].strftime('%Y-%m-%d'), 
                            open=row['Open'],
                            high=row['High'],
                            low=row['Low'],
                            close=row['Close'],
                            volume=row['Volume'])
            try:
                db_insert(curs, sp)                    
            except IntegrityError:
                pass
    conn.commit()
    conn.close()


def oneday_update_stock_price_db(date_: str|None = None):
    conn = sqlite3.connect(DBFILE)
    curs = conn.cursor()
    if not date_:
        date_ = date.today().strftime("%Y-%m-%d")
        
    def update(df, market):
        for _, row in df.iterrows():
            sp = StockPrice(code=row['Code'], 
                            market=market, 
                            date=date_, 
                            open=row['Open'],
                            high=row['High'],
                            low=row['Low'],
                            close=row['Close'],
                            volume=row['Volume'])
            try:
                db_insert(curs, sp)
            except IntegrityError:
                pass          
        
    columns = {'티커':'Code','시가':'Open', '고가':'High', '저가':'Low', '종가':'Close', '거래량':'Volume'}
    df = krx.get_market_ohlcv(date_, market='KOSPI').reset_index()
    df = df.rename(columns=columns)
    if df.values.any():
        update(df, 'kospi')
        
    df = krx.get_market_ohlcv(date_, market='KOSDAQ').reset_index()
    df = df.rename(columns=columns)
    if df.values.any():
        update(df, 'kosdaq')
     
    for code in ['KS11', 'KQ11']:
        df = fdr.DataReader(code, date_, date_).reset_index()
        df['Code'] = code
        update(df, code)
        
    conn.commit()
    conn.close()


def strong_stocks(market: str):
    conn = sqlite3.connect(DBFILE)
    curs = conn.cursor()
    if market.lower() == 'kospi':
        market_ = 'KS11'
    elif market.lower() == 'kosdaq':
        market_ = 'KQ11'
    else:
        assert False, "wrong market"
    
    df = get_ohlc(curs=curs, code=market_)
    df['Date'] = pd.to_datetime(df['Date'])
    df_market_index = df.set_index('Date')    
    df_market_index_week = resample_df(df_market_index, period='W-FRI')[::-1]
    date = df_market_index_week[:52]["Low"].idxmin()
    lowest = df_market_index_week.loc[date, "Low"]
    currval = df_market_index_week.iloc[0]["Close"] 
    ks_rate = (currval - lowest)/lowest*100
    res = []
    codes = [x[0] for x in get_stock_list(curs=curs, market=market.lower())]
    last_date = df_market_index.index.max()
    start_date = last_date - timedelta(days=400)
    df_marcap = krx.get_market_cap(last_date.strftime('%Y-%m-%d'))
    for code in tqdm(codes):
        df = get_ohlc(curs=curs, code=code, start_date=start_date.strftime('%Y-%m-%d'), end_date=last_date.strftime('%Y-%m-%d'))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df_week = resample_df(df, period='W-FRI')[::-1]
        if len(df_week) < 52:
            continue
        date = df_week[:52]["Low"].idxmin()        
        lowest = df_week.loc[date, "Low"]
        if lowest == 0:
            continue
        currval = df["Close"].iloc[-1]
        rate = (currval - lowest)/lowest*100
        if (rate > ks_rate):            
            res.append([code, rate, df_marcap.loc[code, "시가총액"]])
    
    return ks_rate, pd.DataFrame(res, columns=["code", "rate", "marcap"])








