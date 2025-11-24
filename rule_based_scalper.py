# Run a lighter demo using the previously defined prototype logic, but with smaller sample size (300 minutes)
# For speed, re-define only necessary functions (simplified) and run backtest on small data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import List, Optional
import math

def generate_sample_data(minutes=300, seed=42):
    rng = np.random.RandomState(seed)
    t0 = pd.Timestamp('2025-01-01 00:00:00')
    times = [t0 + pd.Timedelta(minutes=i) for i in range(minutes)]
    price = 1.1000
    opens, highs, lows, closes, vols = [], [], [], [], []
    for i in range(minutes):
        phase = (i // 60) % 4
        drift = 0.00012 if phase==0 else (-0.00014 if phase==1 else 0.0)
        change = drift + rng.normal(scale=0.0005)
        high = price + abs(rng.normal(scale=0.0006)) + 0.0002
        low = price - abs(rng.normal(scale=0.0006)) - 0.0002
        openp = price
        closep = price + change
        vol = rng.randint(1, 100)
        opens.append(openp); highs.append(high); lows.append(low); closes.append(closep); vols.append(vol)
        price = closep
    df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': vols}, index=times)
    return df

# Minimal candlestick detectors
def is_hammer(row):
    o,h,l,c = row['open'], row['high'], row['low'], row['close']
    body = abs(c-o)
    lower_wick = min(o,c)-l
    upper_wick = h-max(o,c)
    if body==0: body=1e-9
    return (lower_wick > 2*body) and (upper_wick < body*1.5)

def is_inverted_hammer(row):
    o,h,l,c = row['open'], row['high'], row['low'], row['close']
    body = abs(c-o)
    upper_wick = h-max(o,c)
    lower_wick = min(o,c)-l
    if body==0: body=1e-9
    return (upper_wick > 2*body) and (lower_wick < body*1.5)

def is_doji(row, tol=0.001):
    o,c,h,l = row['open'], row['close'], row['high'], row['low']
    rng = h - l
    if rng==0: return False
    return abs(o-c) <= max(tol*(h+l)/2, 0.1*rng)

def detect_trend(df, lookback=20):
    closes = df['close'].values
    if len(closes) < lookback+2:
        return 'sideways'
    recent = closes[-lookback:]
    diffs = np.diff(recent)
    ups = np.sum(diffs>0); downs = np.sum(diffs<0)
    if ups > downs*1.2 and ups > lookback*0.45: return 'up'
    if downs > ups*1.2 and downs > lookback*0.45: return 'down'
    return 'sideways'

@dataclass
class Trade:
    entry_time: pd.Timestamp
    side: str
    entry: float
    stop: float
    target: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    result: Optional[float] = None

def size_for_risk(balance, entry, stop, risk_pct=0.005):
    risk_amount = balance * risk_pct
    pip_risk = abs(entry-stop)
    if pip_risk==0: return 0
    return risk_amount / pip_risk

def run_backtest_simple(df, initial_balance=10000, risk_pct=0.005, rr=2.0):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    # compute simple indicators per bar
    df['is_hammer'] = df.apply(is_hammer, axis=1)
    df['is_inv_hammer'] = df.apply(is_inverted_hammer, axis=1)
    df['is_doji'] = df.apply(is_doji, axis=1)
    trades = []
    for i in range(2, len(df)-5):
        window = df.iloc[max(0,i-40):i+1]
        trend_1m = detect_trend(window, lookback=20)
        # simulate 15m HTF by looking further back coarse
        trend_15m = detect_trend(df.iloc[max(0,i-80):i+1], lookback=20)
        bar = df.iloc[i]
        prev = df.iloc[i-1]
        # simple SR: previous local high/low
        recent_high = df['high'].iloc[max(0,i-20):i].max()
        recent_low = df['low'].iloc[max(0,i-20):i].min()
        # BUY condition: HTF up, LTF retrace into support (near recent_low), hammer/engulf/doji bullish
        if trend_15m=='up' and (bar['low'] <= recent_low * 1.001):
            cond = bar['is_hammer'] or (bar['close']>prev['close'] and bar['close']>bar['open']) or (bar['is_doji'] and bar['close']>prev['close'])
            if cond:
                entry = bar['close']
                stop = recent_low - 0.0005
                target = entry + rr*abs(entry-stop)
                trades.append(Trade(entry_time=bar.name, side='buy', entry=entry, stop=stop, target=target))
        # SELL condition: HTF down, retrace into resistance (near recent_high), inverted hammer / bearish move
        if trend_15m=='down' and (bar['high'] >= recent_high * 0.999):
            cond = bar['is_inv_hammer'] or (bar['close']<prev['close'] and bar['close']<bar['open']) or (bar['is_doji'] and bar['close']<prev['close'])
            if cond:
                entry = bar['close']
                stop = recent_high + 0.0005
                target = entry - rr*abs(entry-stop)
                trades.append(Trade(entry_time=bar.name, side='sell', entry=entry, stop=stop, target=target))
    # simulate exits
    results = []
    for t in trades:
        start = df.index.get_loc(t.entry_time)
        hit=False
        for j in range(start+1, len(df)):
            r = df.iloc[j]
            if t.side=='buy':
                if r['low'] <= t.stop:
                    t.exit_time = r.name; t.exit_price=t.stop; t.result = t.exit_price - t.entry; hit=True; break
                if r['high'] >= t.target:
                    t.exit_time = r.name; t.exit_price=t.target; t.result = t.exit_price - t.entry; hit=True; break
            else:
                if r['high'] >= t.stop:
                    t.exit_time = r.name; t.exit_price=t.stop; t.result = t.entry - t.exit_price; hit=True; break
                if r['low'] <= t.target:
                    t.exit_time = r.name; t.exit_price=t.target; t.result = t.entry - t.exit_price; hit=True; break
        if not hit:
            last = df.iloc[-1]
            t.exit_time = last.name; t.exit_price = last['close']; t.result = (t.exit_price - t.entry) if t.side=='buy' else (t.entry - t.exit_price)
        results.append(t)
    wins = [t for t in results if t.result>0]; losses=[t for t in results if t.result<=0]
    total_pnl = sum([t.result for t in results])
    win_rate = len(wins)/len(results) if results else 0
    print("Trades:", len(results), "Win rate:", f"{win_rate:.2%}", "Total PnL (price units):", total_pnl)
    if results:
        df_log = pd.DataFrame([asdict(t) for t in results])
        print(df_log.head(10).to_string(index=False))
    return {'results': results, 'summary': {'trades': len(results), 'win_rate': win_rate, 'total_pnl': total_pnl}}

# Run demo
df_sample = generate_sample_data(minutes=300)
res = run_backtest_simple(df_sample)
res
