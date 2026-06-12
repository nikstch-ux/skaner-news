"""
Unified Pre-screener — Feeds ALL strategies
============================================
Runs at 9:00 AM ET (before traders at 10:30 AM).
Scans 569+ stocks, pre-computes ALL conditions, saves candidates.json.

Each strategy reads candidates.json and picks its own candidates:
  v4_trader:    ema10>ema20 + ma50 + adx>20 + green
  swing_trader: breakout_20d + rs_spy + volume

Replaces hardcoded universes — all stocks are now eligible.
"""

import os, json, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
from pathlib import Path
import urllib.request
import pytz

ET              = pytz.timezone("America/New_York")
CANDIDATES_FILE = "candidates.json"
NTFY_TOPIC      = os.getenv("NTFY_TOPIC", "nik167privitetrading")

# ── Universe: everything the scanners watch ───────────────────────────────────
# S&P 500 + NASDAQ 100 + our known momentum stocks
UNIVERSE = [
    # Semis
    "NVDA","AMAT","LRCX","AVGO","MU","INTC","AMD","ANET","ON","KLAC","MRVL",
    "ARM","ASML","MCHP","NXPI","ADI","FTNT","TSM","QCOM","TXN",
    # Mega cap
    "AAPL","MSFT","GOOGL","META","AMZN","ORCL","IBM","DELL","CRM","NOW",
    "WDAY","ADBE","NFLX","UBER","ABNB",
    # Finance
    "JPM","GS","MS","BAC","V","MA","COIN","HOOD","SOFI","AFRM","UPST",
    # Industrial + energy
    "HON","GE","CVX","XOM","CAT","DE","PWR","CMI","ROK","GWW","LMT","NOC",
    # Healthcare
    "LLY","UNH","ISRG","DXCM","VRTX","REGN","HIMS",
    # Consumer
    "COST","WMT","SBUX","MCD","NKE","LULU","ONON",
    # Growth / momentum
    "APP","DDOG","PLTR","NET","PANW","CRWD","ZS","SNOW","GTLB",
    "CELH","RKLB","ASTS","LUNR","TSLA","CSCO",
    # High momentum (scanner winners)
    "ALAB","RDW","NTAP","BBY","AMPX","ACHR","JOBY","IONQ","RGTI","QUBT","SOUN",
    # ETFs
    "SPY","QQQ","XLK","XLF","XLE","XAR","SOXX","TQQQ","SPXL",
    # Beaten down recovery watch
    "JBLU","RCL","CCL","HOOD","UPST","DKNG","ZM",
]
UNIVERSE = list(dict.fromkeys(UNIVERSE))

def download_data():
    end   = datetime.today()
    start = end - timedelta(days=310)
    raw   = yf.download(
        UNIVERSE, start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d", auto_adjust=True, progress=False, threads=True
    )
    return raw

def compute_candidate(sym, raw):
    try:
        df = raw.xs(sym, level=1, axis=1) if isinstance(raw.columns, pd.MultiIndex) else raw
        df = df.copy(); df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["open","high","low","close","volume"])
        if len(df) < 55: return None

        c = df["close"]

        # EMAs and MAs
        ema10  = float(c.ewm(span=10, adjust=False).mean().iloc[-1])
        ema20  = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
        ema80  = float(c.ewm(span=80, adjust=False).mean().iloc[-1])
        ema200 = float(c.ewm(span=200, adjust=False).mean().iloc[-1])
        ma50   = float(c.rolling(50).mean().iloc[-1])
        ma200  = float(c.rolling(200).mean().iloc[-1])
        price  = float(c.iloc[-1])

        # ADX
        h, l = df["high"], df["low"]
        tr   = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
        pdm  = (h-h.shift()).clip(lower=0).where((h-h.shift())>(l.shift()-l), 0)
        ndm  = (l.shift()-l).clip(lower=0).where((l.shift()-l)>(h-h.shift()), 0)
        atr  = tr.ewm(span=14, adjust=False).mean()
        pdi  = 100 * pdm.ewm(span=14, adjust=False).mean() / atr
        ndi  = 100 * ndm.ewm(span=14, adjust=False).mean() / atr
        dx   = (100*(pdi-ndi).abs()/(pdi+ndi)).fillna(0)
        adx  = float(dx.ewm(span=14, adjust=False).mean().iloc[-1])
        atr_val = float(atr.iloc[-1])

        # Volume
        avg_vol = float(df["volume"].tail(20).mean())
        cur_vol = float(df["volume"].iloc[-1])
        vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 0

        # Green candle — use PREVIOUS complete daily bar (not today's partial)
        green_prev = float(df["close"].iloc[-2]) > float(df["open"].iloc[-2])

        # 20-day and 55-day breakout
        brk20 = price > float(df["high"].values[-21:-1].max()) if len(df) >= 22 else False
        brk55 = price > float(df["high"].values[-56:-1].max()) if len(df) >= 57 else False

        # Conditions
        cond_v4_ema   = ema10 > ema20
        cond_v4_ma50  = price > ma50
        cond_v4_adx   = adx > 20
        cond_v4_green = green_prev
        cond_v4_vol   = avg_vol >= 1_000_000
        cond_v4_price = price >= 10

        cond_swing_brk  = brk20
        cond_swing_vol  = vol_ratio >= 1.5
        cond_swing_price = price >= 5

        # Overall score (0-10)
        score = sum([
            cond_v4_ema, cond_v4_ma50, cond_v4_adx, cond_v4_green,
            cond_v4_vol, cond_v4_price, cond_swing_brk, cond_swing_vol,
            price > ma200, adx > 30
        ])

        # v4 ready = all v4 conditions met
        v4_ready    = all([cond_v4_ema, cond_v4_ma50, cond_v4_adx,
                           cond_v4_green, cond_v4_vol, cond_v4_price])
        swing_ready = all([cond_swing_brk, cond_swing_vol, cond_swing_price])

        return {
            "symbol":      sym,
            "price":       round(price, 2),
            "adx":         round(adx, 1),
            "atr":         round(atr_val, 2),
            "avg_vol_m":   round(avg_vol / 1e6, 2),
            "vol_ratio":   round(vol_ratio, 1),
            "ema10_gt_20": cond_v4_ema,
            "above_ma50":  cond_v4_ma50,
            "above_ma200": price > ma200,
            "adx_gt_20":   cond_v4_adx,
            "green_prev":  cond_v4_green,
            "brk20d":      cond_swing_brk,
            "brk55d":      brk55,
            "score":       score,
            "v4_ready":    v4_ready,
            "swing_ready": swing_ready,
        }
    except Exception:
        return None

def send_ntfy(title, msg):
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=msg.encode(),
            headers={"Title": title.encode(), "Priority": b"high"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  ntfy failed: {e}")

def main():
    now = datetime.now(ET)
    print(f"\n{'='*60}")
    print(f"  UNIFIED PRE-SCREENER | {now.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"  Scanning {len(UNIVERSE)} stocks -> candidates.json")
    print(f"{'='*60}\n")

    print("  Downloading data...")
    raw = download_data()

    candidates = []
    for sym in UNIVERSE:
        result = compute_candidate(sym, raw)
        if result and result["score"] >= 3:
            candidates.append(result)

    # Sort by score desc, then ADX
    candidates.sort(key=lambda x: (x["score"], x["adx"]), reverse=True)

    v4_ready    = [c for c in candidates if c["v4_ready"]]
    swing_ready = [c for c in candidates if c["swing_ready"]]

    print(f"\n  v4_trader candidates:    {len(v4_ready)}")
    for c in v4_ready[:10]:
        print(f"    {c['symbol']:<7} ${c['price']:>8.2f}  ADX={c['adx']:>5.1f}  vol={c['vol_ratio']:.1f}x")

    print(f"\n  swing_trader candidates: {len(swing_ready)}")
    for c in swing_ready[:10]:
        print(f"    {c['symbol']:<7} ${c['price']:>8.2f}  ADX={c['adx']:>5.1f}  vol={c['vol_ratio']:.1f}x  {'55d' if c['brk55d'] else '20d'}")

    # Save candidates.json
    output = {
        "generated":   now.strftime("%Y-%m-%d %H:%M ET"),
        "total":       len(candidates),
        "v4_ready":    len(v4_ready),
        "swing_ready": len(swing_ready),
        "candidates":  candidates,
    }
    with open(CANDIDATES_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {len(candidates)} candidates -> {CANDIDATES_FILE}")

    # Notify
    lines = []
    if v4_ready:
        lines.append(f"v4 ready: {' '.join(c['symbol'] for c in v4_ready[:5])}")
    if swing_ready:
        lines.append(f"swing ready: {' '.join(c['symbol'] for c in swing_ready[:5])}")
    lines.append(f"Total: {len(candidates)} candidates for 10:30 AM")

    if v4_ready or swing_ready:
        send_ntfy(f"Prescreener: {len(v4_ready)}+{len(swing_ready)} ready",
                  "\n".join(lines))

    print("  Done.")

if __name__ == "__main__":
    main()
