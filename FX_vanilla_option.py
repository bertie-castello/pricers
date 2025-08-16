## all files are a WIP - in the process of transferring from past notes 

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ---------- Core (vectorised) ----------
def _to_arr(x):
    return np.asarray(x, dtype=float)

def gk_d1_d2(S, K, T, rd, rf, sigma):
    S, K, T, rd, rf, sigma = map(_to_arr, (S, K, T, rd, rf, sigma))
    if np.any(S <= 0) or np.any(K <= 0) or np.any(T <= 0) or np.any(sigma <= 0):
        raise ValueError("S, K, T, sigma must be strictly positive.")
    vol_sqrtT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma**2) * T) / vol_sqrtT
    d2 = d1 - vol_sqrtT
    return d1, d2

def gk_price(S, K, T, rd, rf, sigma, opt='C'):
    d1, d2 = gk_d1_d2(S, K, T, rd, rf, sigma)
    disc_d = np.exp(-_to_arr(rd) * _to_arr(T))
    disc_f = np.exp(-_to_arr(rf) * _to_arr(T))
    if opt.upper() == 'C':
        return _to_arr(S) * disc_f * norm.cdf(d1) - _to_arr(K) * disc_d * norm.cdf(d2)
    elif opt.upper() == 'P':
        return _to_arr(K) * disc_d * norm.cdf(-d2) - _to_arr(S) * disc_f * norm.cdf(-d1)
    else:
        raise ValueError("opt must be 'C' or 'P'.")

def _bounds_price_fx(S, K, T, rd, rf, opt):
    S, K, T, rd, rf = map(_to_arr, (S, K, T, rd, rf))
    disc_d = np.exp(-rd * T)
    disc_f = np.exp(-rf * T)
    if opt.upper() == 'C':
        lower = np.maximum(S * disc_f - K * disc_d, 0.0)
        upper = S * disc_f
    else:
        lower = np.maximum(K * disc_d - S * disc_f, 0.0)
        upper = K * disc_d
    return lower, upper

# ---------- Implied volatility (vectorised) ----------
def implied_vol_fx(price, S, K, T, rd, rf, opt='C',
                   vol_lo=1e-6, vol_hi=5.0, tol=1e-10, maxiter=100):
    """
    Returns array of implied vols; NaN if price violates bounds or bracketing fails.
    """
    price = _to_arr(price)
    S, K, T, rd, rf = map(_to_arr, (S, K, T, rd, rf))
    P, S_, K_, T_, RD_, RF_ = np.broadcast_arrays(price, S, K, T, rd, rf)
    out = np.full(P.shape, np.nan, dtype=float)

    lower, upper = _bounds_price_fx(S_, K_, T_, RD_, RF_, opt)
    ok = (P >= lower - 1e-12) & (P <= upper + 1e-12)

    def diff(sig, s, k, t, rdd, rff, p):
        return gk_price(s, k, t, rdd, rff, sig, opt) - p

    it = np.nditer([P, S_, K_, T_, RD_, RF_, ok, out],
                   flags=['multi_index', 'refs_ok'],
                   op_flags=[['readonly']]*7 + [['readwrite']])
    for p, s, k, t, rdd, rff, good, res in it:
        if not good:
            res[...] = np.nan
            continue
        f_lo = diff(vol_lo, s, k, t, rdd, rff, p)
        f_hi = diff(vol_hi, s, k, t, rdd, rff, p)
        if f_lo * f_hi > 0:
            res[...] = np.nan
            continue
        try:
            root = brentq(lambda x: diff(x, s, k, t, rdd, rff, p),
                          vol_lo, vol_hi, xtol=tol, maxiter=maxiter)
            res[...] = root
        except Exception:
            res[...] = np.nan
    return out

# ---------- I/O utils ----------
def _parse_float(x: str) -> float:
    return float(x.strip())

def _parse_rate(x: str) -> float:
    v = float(x.strip())
    return v / 100.0 if v > 1.0 else v

def _parse_list_of_floats(x: str, rate=False):
    vals = [float(t) for t in x.replace(";", ",").split(",") if t.strip()]
    if rate:
        vals = [vv / 100.0 if vv > 1.0 else vv for vv in vals]
    return np.asarray(vals, dtype=float)

def print_table(headers, rows):
    colw = [max(len(h), max((len(f"{r[i]}") for r in rows), default=0)) for i,h in enumerate(headers)]
    fmt = "  ".join("{:" + str(w) + "}" for w in colw)
    print(fmt.format(*headers))
    print(fmt.format(*["-"*w for w in colw]))
    for r in rows:
        print(fmt.format(*r))

# ---------- CLI ----------
def main():
    print("FX European Option Pricer (Garman–Kohlhagen)")
    mode = input("Mode: 'single' (price one) or 'smile' (plot IV smile): ").strip().lower()

    S   = _parse_float(input("Spot S (domestic per unit foreign): "))
    T   = _parse_float(input("Maturity T in years (e.g., 0.5): "))
    rd  = _parse_rate(input("Domestic rate r_d (e.g., 5 or 0.05): "))
    rf  = _parse_rate(input("Foreign  rate r_f (e.g., 3 or 0.03): "))
    opt = (input("Option type ('C' or 'P') [default C]: ").strip().upper() or "C")

    if mode == "single":
        K     = _parse_float(input("Strike K: "))
        sigma = _parse_rate(input("Volatility sigma (e.g., 12 or 0.12): "))
        prem  = gk_price(S, K, T, rd, rf, sigma, opt)
        print(f"\n{opt}-option premium: {prem:.8f}")
        return

    if mode == "smile":
        strikes = _parse_list_of_floats(input("Strikes K list (comma-separated): "))
        if strikes.size == 0:
            raise ValueError("Provide at least one strike.")
        src = input("Provide market prices? 'y' to enter prices, else generate from base sigma: ").strip().lower()
        if src == 'y':
            prices = _parse_list_of_floats(input("Option prices list (same length as strikes): "))
            if prices.size != strikes.size:
                raise ValueError("Prices length must match strikes length.")
        else:
            sigma0 = _parse_rate(input("Base sigma to generate theoretical prices (e.g., 10 or 0.10): "))
            prices = gk_price(S, strikes, T, rd, rf, sigma0, opt)

        iv = implied_vol_fx(prices, S, strikes, T, rd, rf, opt)

        rows = [(f"{k:.6f}", f"{p:.8f}", f"{(v if np.isfinite(v) else np.nan):.6f}")
                for k,p,v in zip(strikes, prices, iv)]
        print()
        print_table(["Strike", "Price", "ImpliedVol"], rows)

        # Plot IV vs Strike (and optionally vs moneyness)
        plt.figure()
        plt.plot(strikes, iv, marker='o', linestyle='-')
        plt.xlabel("Strike K")
        plt.ylabel("Implied volatility")
        plt.title(f"FX IV Smile (T={T} yr, opt={opt}, rd={rd:.4f}, rf={rf:.4f})")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()
        return

    raise ValueError("Mode must be 'single' or 'smile'.")

if __name__ == "__main__":
    main()
