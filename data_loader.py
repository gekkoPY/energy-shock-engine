# data_loader.py
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from config import FRED_API_KEY, START_DATE, END_DATE

def fetch_macro_data():
    fred = Fred(api_key=FRED_API_KEY)
    
    print("[1/3] Download serie storiche da FRED...")
    
    fred_series = {
        "Breakeven_5Y": "T5YIE",
        "Treasury_2Y": "DGS2",
        "Treasury_10Y": "DGS10",
        "TIPS_10Y_Real": "DFII10",
        "Fed_Funds_Target": "DFEDTARU",
        "Fed_Funds_Effective": "DFF"
    }
    
    fred_dict = {}
    for col_name, ticker in fred_series.items():
        s = fred.get_series(ticker, observation_start=START_DATE)
        fred_dict[col_name] = s
    
    df_fred = pd.DataFrame(fred_dict)
    
    print("[2/3] Download serie di mercato da Yahoo Finance...")
    
    yf_tickers = {
        "Brent": "BZ=F",
        "OVX": "^OVX",
        "SP500_EW": "RSP",
        "DXY": "DX-Y.NYB"
    }
    
    df_yf = yf.download(
        list(yf_tickers.values()), 
        start=START_DATE, 
        end=END_DATE, 
        progress=False
    )["Close"]
    
    
    inv_map = {v: k for k, v in yf_tickers.items()}
    df_yf = df_yf.rename(columns=inv_map)
    
    print("[3/3] Sincronizzazione calendari e pulizia dati...")
    
    df_merged = df_fred.join(df_yf, how="inner")
    df_merged = df_merged.dropna(subset=["Treasury_10Y", "Brent", "SP500_EW", "OVX"])
    
    
    df_merged = df_merged.sort_index()
    
    print(f"Dataset sincronizzato con successo: {len(df_merged)} osservazioni giornaliere.")
    return df_merged

if __name__ == "__main__":
    df = fetch_macro_data()
    print(df.tail())
