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



def get_stock_list(*, curs: Cursor|None=None, market: str):
    conn: Connection|None = None
    if not curs:
        conn = sqlite3.connect(DBFILE)
        curs = conn.cursor()
    qry = f"select DISTINCT code from StockPrice where market='{market}'"
    curs.execute(qry)
    res = curs.fetchall() 
    res = [x[0] for x in res]
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
    for i, n in update_stock_price_db_iter():
        ...



def update_stock_price_db_iter():
    conn = sqlite3.connect(DBFILE)
    curs = conn.cursor()
    qry = "select DISTINCT code, market from StockPrice"
    curs.execute(qry)
    res = curs.fetchall() 

    def num_missing_days():
        res = curs.execute("select MAX(date) from StockPrice where code='KS11'")

        start_day = res.fetchone()[0]
        start_day = datetime.strptime(start_day, "%Y-%m-%d") + timedelta(days=1)
        start_day = start_day.strftime("%Y-%m-%d")
        end_day = datetime.today().date().strftime("%Y-%m-%d")
        
        print(f"{start_day}, {end_day}")
        df = fdr.DataReader('KS11', start_day, end_day)
        return start_day, len(df)
            
    start_day, missing_days = num_missing_days()
    if missing_days > 1:
        n_items = len(res)
        for i, (code, market) in tqdm(enumerate(res)):
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
            yield (i, n_items)
        conn.commit()
        conn.close()
    else:
        oneday_update_stock_price_db(start_day)
        yield (1, 1)



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
            except IntegrityError as exc:
                print(exc)

    columns = {'티커':'Code','시가':'Open', '고가':'High', '저가':'Low', '종가':'Close', '거래량':'Volume'}
    for market in ['KOSPI', 'KOSDAQ']:
        df = krx.get_market_ohlcv(date_, market=market).reset_index()
        df = df.rename(columns=columns)
        if df[['Open', 'High', 'Low', 'Close']].values.any():
            update(df, market.lower())       
     
    for code in ['KS11', 'KQ11']:
        df = fdr.DataReader(code, date_, date_).reset_index()
        df['Code'] = code
        update(df, code)
        
    conn.commit()
    conn.close()


    
def strong_stocks(market: str):
    for ks_rate, df in strong_stocks_iter(market):
        ...
    return ks_rate, df
    

    
def strong_stocks_iter(market: str, week: int=52):
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
    date = df_market_index_week[:week]["Low"].idxmin()
    lowest = df_market_index_week.loc[date, "Low"]
    currval = df_market_index_week.iloc[0]["Close"] 
    ks_rate = (currval - lowest)/lowest*100
    res = []
    codes = get_stock_list(curs=curs, market=market.lower())
    last_date = df_market_index.index.max()
    start_date = last_date - timedelta(days=400)
    df_marcap = krx.get_market_cap(last_date.strftime('%Y-%m-%d'))
    n = len(codes)
    for idx, code in enumerate(tqdm(codes)):
        df = get_ohlc(curs=curs, code=code, start_date=start_date.strftime('%Y-%m-%d'), end_date=last_date.strftime('%Y-%m-%d'))
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        df_week = resample_df(df, period='W-FRI')[::-1]
        if len(df_week) < week:
            continue
        date = df_week[:week]["Low"].idxmin()        
        lowest = df_week.loc[date, "Low"]
        if lowest == 0:
            continue
        currval = df["Close"].iloc[-1]
        rate = (currval - lowest)/lowest*100
        if (rate > ks_rate):            
            res.append([code, rate, df_marcap.loc[code, "시가총액"]])        
        yield (idx, n)
    
    df = pd.DataFrame(res, columns=["code", "rate", "marcap"])
    df = df.sort_values(by="rate", ascending=False)
    yield (ks_rate, df)
    

    
def get_2days_prices(market: str, date_: str|None=None):
    conn = sqlite3.connect(DBFILE)
    curs = conn.cursor()

    if not date_:
        qry = f"""
SELECT * 
FROM StockPrice 
WHERE market='{'KS11' if market.lower() == 'kospi' else 'KQ11'}'
ORDER BY date desc
limit 2
    """
    else:
        qry = f"""
SELECT * 
FROM StockPrice 
WHERE market='{'KS11' if market.lower() == 'kospi' else 'KQ11'}'
    AND date <= '{date_}'
ORDER BY date desc
limit 2
"""
        
    curs.execute(qry)
    res = curs.fetchall()

    qry = f"""
SELECT * 
FROM StockPrice 
WHERE market='{market.lower()}'
    AND (date BETWEEN '{res[1][2]}' AND '{res[0][2]}')
"""    
    curs.execute(qry)
    res = curs.fetchall()       

    df = pd.DataFrame(res, columns=["Code", "Market", "Date", "Open", "High", "Low", "Close", "Volume"])
    conn.close()
    return df




