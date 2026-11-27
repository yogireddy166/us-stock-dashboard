#!/usr/bin/env python3
"""
US Stock Dashboard - data builder.

Downloads ~10 years of daily (dividend/split-adjusted) closing prices from
Yahoo Finance for every stock in the chosen universe, computes trailing
returns for 1D, 1W, 1M, 3M, 6M, YTD, 1Y, 1.5Y, 2Y, 3Y, 4Y, 5Y and 10Y,
and writes JSON files that the dashboard page (site/index.html) reads.

Usage
  python scripts/fetch_data.py --universe sp500        # ~500 stocks (fast, ~3 min)
  python scripts/fetch_data.py --universe all          # every NYSE/Nasdaq/AMEX stock (~6,000, ~20-40 min)
  python scripts/fetch_data.py --universe custom       # tickers listed in tickers.txt
  python scripts/fetch_data.py --demo                  # fake data, no internet (for testing the page)
"""
import argparse
import datetime as dt
import io
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "site", "data")
HIST_DIR = os.path.join(OUT_DIR, "history")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# (label, offset) - offset is a pandas DateOffset kwargs dict, "prev" or "ytd"
PERIODS = [
    ("1D", "prev"),
    ("1W", {"weeks": 1}),
    ("1M", {"months": 1}),
    ("3M", {"months": 3}),
    ("6M", {"months": 6}),
    ("YTD", "ytd"),
    ("1Y", {"years": 1}),
    ("1.5Y", {"months": 18}),
    ("2Y", {"years": 2}),
    ("3Y", {"years": 3}),
    ("4Y", {"years": 4}),
    ("5Y", {"years": 5}),
    ("10Y", {"years": 10}),
]

BENCHMARKS = [("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq Composite"), ("^DJI", "Dow Jones"), ("^RUT", "Russell 2000")]

EXCHANGES = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX"}


# --------------------------------------------------------------------------- universes
def _get(url):
    import requests
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    return r.text


def universe_sp500():
    """S&P 500 members with name + GICS sector."""
    try:
        html = _get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = pd.read_html(io.StringIO(html))[0]
    except Exception as e:  # fallback mirror
        print(f"Wikipedia failed ({e}); using GitHub mirror", file=sys.stderr)
        df = pd.read_csv(io.StringIO(_get(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")))
    out = []
    for _, r in df.iterrows():
        sym = str(r["Symbol"]).strip()
        out.append({"t": sym.replace(".", "-"), "n": str(r["Security"]), "s": str(r["GICS Sector"])})
    return out


def universe_all(include_etfs=False):
    """Every listed US common stock (Nasdaq + NYSE + NYSE American + Arca + Cboe)."""
    out, seen = [], set()
    skip_words = (" warrant", " right", " unit", "units", " preferred", " depositary shares", " notes due")

    def keep(name, etf, test):
        if test == "Y" or (etf == "Y" and not include_etfs):
            return False
        low = name.lower()
        return not any(w in low for w in skip_words)

    nas = pd.read_csv(io.StringIO(_get("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt")),
                      sep="|", dtype=str).fillna("")
    for _, r in nas.iterrows():
        sym = r["Symbol"]
        if not sym or sym.startswith("File Creation") or not keep(r["Security Name"], r["ETF"], r["Test Issue"]):
            continue
        if sym not in seen:
            seen.add(sym)
            out.append({"t": sym, "n": r["Security Name"].split(" - ")[0], "s": "Nasdaq"})

    oth = pd.read_csv(io.StringIO(_get("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt")),
                      sep="|", dtype=str).fillna("")
    for _, r in oth.iterrows():
        sym = r["ACT Symbol"]
        if not sym or sym.startswith("File Creation") or "$" in sym or not keep(r["Security Name"], r["ETF"], r["Test Issue"]):
            continue
        ysym = sym.replace(".", "-")
        if ysym not in seen:
            seen.add(ysym)
            out.append({"t": ysym, "n": r["Security Name"].split(" - ")[0],
                        "s": EXCHANGES.get(r["Exchange"], r["Exchange"])})

    # Use S&P sectors where we know them
    try:
        sect = {x["t"]: x["s"] for x in universe_sp500()}
        for x in out:
            if x["t"] in sect:
                x["s"] = sect[x["t"]]
    except Exception:
        pass
    return out


def universe_custom():
    path = os.path.join(ROOT, "tickers.txt")
    out = []
    with open(path) as f:
        for line in f:
            t = line.split("#")[0].strip().upper()
            if t:
                out.append({"t": t.replace(".", "-"), "n": t, "s": "Watchlist"})
    return out


# --------------------------------------------------------------------------- prices
def download_closes(tickers, chunk=150, retries=3):
    """Return (closes DataFrame, volumes DataFrame) indexed by date, one column per ticker."""
    import yfinance as yf
    closes, vols = [], []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        for attempt in range(retries):
            try:
                df = yf.download(batch, period="10y", interval="1d", auto_adjust=True,
                                 group_by="column", threads=True, progress=False)
                if df is None or df.empty:
                    raise RuntimeError("empty result")
                c, v = df["Close"], df["Volume"]
                if isinstance(c, pd.Series):
                    c, v = c.to_frame(batch[0]), v.to_frame(batch[0])
                closes.append(c)
                vols.append(v)
                break
            except Exception as e:
                print(f"  batch {i // chunk + 1} attempt {attempt + 1} failed: {e}", file=sys.stderr)
                time.sleep(10 * (attempt + 1))
        print(f"  downloaded {min(i + chunk, len(tickers))}/{len(tickers)}", flush=True)
        time.sleep(1)
    if not closes:
        raise SystemExit("No price data downloaded - Yahoo Finance may be blocking requests. Try again later.")
    C = pd.concat(closes, axis=1)
    V = pd.concat(vols, axis=1)
    C = C.loc[:, ~C.columns.duplicated()]
    V = V.loc[:, ~V.columns.duplicated()]
    C.index = pd.to_datetime(C.index).tz_localize(None)
    V.index = pd.to_datetime(V.index).tz_localize(None)
    return C.sort_index(), V.sort_index()


def demo_closes(universe, years=10.5, seed=7):
    """Random-walk prices so the page can be tested with no internet."""
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(end - pd.DateOffset(days=int(365 * years)), end)
    C, V = {}, {}
    for k, u in enumerate(universe):
        n = len(idx)
        start = rng.integers(0, n // 2) if k % 9 == 0 else 0  # some "young" stocks
        drift, vol = rng.normal(0.0004, 0.0004), rng.uniform(0.01, 0.035)
        r = rng.normal(drift, vol, n - start)
        p = np.full(n, np.nan)
        p[start:] = rng.uniform(5, 400) * np.exp(np.cumsum(r))
        C[u["t"]] = p
        V[u["t"]] = rng.integers(1e5, 5e7, n).astype(float)
    return pd.DataFrame(C, index=idx), pd.DataFrame(V, index=idx)


# --------------------------------------------------------------------------- math
def base_price(s, target):
    """Last close on or before `target`; None if history doesn't reach back that far."""
    i = s.index.searchsorted(target, side="right") - 1
    if i < 0:
        # allow up to 5 days of slack (holidays at the start of the data)
        if s.index[0] - target <= pd.Timedelta(days=5):
            return float(s.iloc[0])
        return None
    return float(s.iloc[i])


def trailing_returns(s):
    last_date, last = s.index[-1], float(s.iloc[-1])
    res = []
    for label, off in PERIODS:
        if off == "prev":
            b = float(s.iloc[-2]) if len(s) > 1 else None
        elif off == "ytd":
            b = base_price(s, pd.Timestamp(last_date.year - 1, 12, 31))
        else:
            b = base_price(s, last_date - pd.DateOffset(**off))
        res.append(None if not b or b <= 0 else round((last / b - 1) * 100, 2))
    return res


def rnd(x):
    if x is None or not math.isfinite(x):
        return None
    return round(x, 4) if abs(x) < 1 else round(x, 2)


def series_json(s):
    me = s.resample("ME").last().dropna()
    if len(me) and me.index[-1] > s.index[-1]:
        me.index = me.index[:-1].append(pd.DatetimeIndex([s.index[-1]]))
    daily = s[s.index >= s.index[-1] - pd.DateOffset(months=13)]
    return {
        "m": [[d.strftime("%Y-%m-%d"), rnd(float(v))] for d, v in me.items()],
        "d": [[d.strftime("%Y-%m-%d"), rnd(float(v))] for d, v in daily.items()],
    }


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=os.environ.get("UNIVERSE", "sp500"),
                    choices=["sp500", "all", "custom"])
    ap.add_argument("--include-etfs", action="store_true")
    ap.add_argument("--demo", action="store_true", help="fake data, no internet")
    a = ap.parse_args()

    os.makedirs(HIST_DIR, exist_ok=True)

    if a.demo:
        sectors = ["Information Technology", "Health Care", "Financials", "Energy", "Industrials",
                   "Consumer Discretionary", "Utilities", "Materials"]
        universe = [{"t": f"DEMO{i:03d}", "n": f"Demo Company {i}", "s": sectors[i % len(sectors)]}
                    for i in range(120)]
        bench = [{"t": t, "n": n, "s": "Index"} for t, n in BENCHMARKS]
        C, V = demo_closes(universe + bench)
    else:
        print(f"Building universe: {a.universe}", flush=True)
        universe = {"sp500": universe_sp500,
                    "all": lambda: universe_all(a.include_etfs),
                    "custom": universe_custom}[a.universe]()
        # always also include anything in tickers.txt (your personal watchlist)
        if a.universe != "custom" and os.path.exists(os.path.join(ROOT, "tickers.txt")):
            have = {u["t"] for u in universe}
            universe += [u for u in universe_custom() if u["t"] not in have]
        print(f"{len(universe)} tickers", flush=True)
        C, V = download_closes([u["t"] for u in universe] + [t for t, _ in BENCHMARKS])

    meta = {u["t"]: u for u in universe}
    rows, written = [], 0
    for t, u in meta.items():
        if t not in C.columns:
            continue
        s = C[t].dropna()
        s = s[s > 0]
        if len(s) < 2 or (C.index[-1] - s.index[-1]).days > 10:  # skip delisted / stale
            continue
        yr = s[s.index >= s.index[-1] - pd.DateOffset(years=1)]
        vol = V[t].dropna() if t in V.columns else pd.Series(dtype=float)
        rows.append({
            "t": t, "n": u["n"], "s": u["s"],
            "p": rnd(float(s.iloc[-1])),
            "r": trailing_returns(s),
            "hi": rnd(float(yr.max())), "lo": rnd(float(yr.min())),
            "v": int(vol.iloc[-1]) if len(vol) else None,
            "since": s.index[0].strftime("%Y-%m-%d"),
        })
        with open(os.path.join(HIST_DIR, f"{t}.json"), "w") as f:
            json.dump(series_json(s), f, separators=(",", ":"))
        written += 1

    bench_rows = []
    for t, n in BENCHMARKS:
        if t in C.columns and C[t].dropna().size > 1:
            s = C[t].dropna()
            bench_rows.append({"t": t, "n": n, "p": rnd(float(s.iloc[-1])), "r": trailing_returns(s)})
            with open(os.path.join(HIST_DIR, f"{t}.json"), "w") as f:
                json.dump(series_json(s), f, separators=(",", ":"))

    summary = {
        "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "asof": C.index[-1].strftime("%Y-%m-%d"),
        "universe": "demo" if a.demo else a.universe,
        "periods": [p for p, _ in PERIODS],
        "benchmarks": bench_rows,
        "stocks": rows,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, separators=(",", ":"))
    print(f"Wrote {written} stocks, as of {summary['asof']}")


if __name__ == "__main__":
    main()
