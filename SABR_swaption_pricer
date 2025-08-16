import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ---------------- Black-76 core (vectorised) ----------------

def _arr(x): return np.asarray(x, dtype=float)

def black76_price(F, K, T, vol, A, opt="payer"):
    """
    Black-76 price for swaptions.
    F: forward swap rate (>0)
    K: strike (>0)
    T: expiry in years (>0)
    vol: Black vol (annualised)
    A: annuity (PV of 1bp times 10000? No: A = PV of 1 unit; i.e., PVBP in rate units)
    opt: 'payer' (call on rate) or 'receiver' (put)
    """
    F, K, T, vol, A = map(_arr, (F, K, T, vol, A))
    if np.any(F <= 0) or np.any(K <= 0) or np.any(T <= 0) or np.any(vol <= 0) or np.any(A <= 0):
        raise ValueError("F, K, T, vol, A must be > 0 for Black-76.")
    srt = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * srt**2) / srt
    d2 = d1 - srt
    call = A * (F * norm.cdf(d1) - K * norm.cdf(d2))
    if opt.lower().startswith("p"):  # payer
        return call
    elif opt.lower().startswith("r"):  # receiver
        put = A * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        return put
    else:
        raise ValueError("opt must be 'payer' or 'receiver'.")

# ---------------- SABR (Hagan, lognormal implied vol) ----------------

def sabr_implied_vol_black(F, K, T, alpha, beta, rho, nu):
    """
    Hagan et al. (2002) lognormal SABR implied vol. Vectorised.
    F, K > 0 (use a normal/Bachelier model if rates can be ≤ 0).
    Parameters:
      alpha > 0, beta in [0,1], rho in (-1,1), nu > 0
    """
    F, K, T, alpha, beta, rho, nu = map(_arr, (F, K, T, alpha, beta, rho, nu))
    if np.any(T <= 0) or np.any(F <= 0) or np.any(K <= 0) or np.any(alpha <= 0) or np.any(nu <= 0):
        raise ValueError("T,F,K,alpha,nu must be > 0.")
    if np.any((beta < 0) | (beta > 1)):
        raise ValueError("beta must be in [0,1].")
    if np.any((rho <= -1) | (rho >= 1)):
        raise ValueError("rho must be in (-1,1).")

    one_m_beta = 1.0 - beta
    FK_beta = (F * K) ** (0.5 * one_m_beta)
    logFK = np.log(F / K)

    # ATM mask for numerical stability
    atm = np.isclose(logFK, 0.0, atol=1e-12)

    # General case (K != F)
    z = (nu / alpha) * FK_beta * logFK
    # x(z)
    sqrt_term = np.sqrt(1 - 2 * rho * z + z * z)
    x_z = np.log((sqrt_term + z - rho) / (1 - rho))

    # Hagan numerator/denominator terms
    # Pre-log terms
    term_log = 1 + (one_m_beta**2 / 24.0) * (logFK**2) + (one_m_beta**4 / 1920.0) * (logFK**4)
    # Time-dependent correction
    corr = (
        (one_m_beta**2 / 24.0) * (alpha**2) / ((F * K) ** one_m_beta)
        + (rho * beta * nu * alpha) / (4.0 * FK_beta)
        + ((2 - 3 * rho**2) * nu**2) / 24.0
    )

    sigma_gen = (alpha / FK_beta) * (z / x_z) * term_log * (1 + corr * T)

    # ATM closed form
    F_pow = F ** one_m_beta
    sigma_atm = (alpha / F_pow) * (1 + (
        (one_m_beta**2 / 24.0) * (alpha**2) / (F_pow**2)
        + (rho * beta * nu * alpha) / (4.0 * F_pow)
        + ((2 - 3 * rho**2) * nu**2) / 24.0
    ) * T)

    # Blend
    sigma = np.where(atm, sigma_atm, sigma_gen)
    return sigma

# ---------------- Convenience: SABR -> price ----------------

def swaption_price_sabr(F, K, T, A, alpha, beta, rho, nu, opt="payer"):
    vol = sabr_implied_vol_black(F, K, T, alpha, beta, rho, nu)
    return black76_price(F, K, T, vol, A, opt)

# ---------------- Minimal CLI / plotting ----------------

def _parse_floats(s, pct=False):
    xs = [float(t) for t in s.replace(";", ",").split(",") if t.strip()]
    if pct:  # accept "2" as 2% or 0.02
        xs = [x/100.0 if x > 1.0 else x for x in xs]
    return np.asarray(xs if len(xs) > 1 else xs + [ ] or [], dtype=float)

def _parse_float(x, pct=False):
    v = float(x)
    return v/100.0 if pct and v > 1 else v

def main():
    print("Swaption pricer with SABR (lognormal) -> Black-76")
    mode = input("Mode: 'single' price or 'smile' plot over strikes: ").strip().lower()

    F  = _parse_float(input("Forward swap rate F (e.g., 0.025 or 2.5 for 2.5%): "), pct=True)
    A  = _parse_float(input("Annuity A (PV of 1 unit of rate): "))
    T  = _parse_float(input("Expiry T in years (e.g., 2.0): "))
    alpha = _parse_float(input("SABR alpha (e.g., 0.02 or 2 for 2%): "), pct=True)
    beta  = _parse_float(input("SABR beta in [0,1] (e.g., 0.5): "))
    rho   = _parse_float(input("SABR rho in (-1,1) (e.g., -0.2): "))
    nu    = _parse_float(input("SABR nu (vol-of-vol), e.g., 0.5 or 50 for 50%: "), pct=True)
    opt   = (input("Type: 'payer' or 'receiver' [payer]: ").strip().lower() or "payer")

    if mode == "single":
        K = _parse_float(input("Strike K (e.g., 0.025 or 2.5 for 2.5%): "), pct=True)
        price = swaption_price_sabr(F, K, T, A, alpha, beta, rho, nu, opt)
        vol   = sabr_implied_vol_black(F, K, T, alpha, beta, rho, nu)
        print(f"\nBlack vol: {vol:.8f}")
        print(f"{opt.title()} swaption price: {price:.10f}")
        return

    if mode == "smile":
        Ks = _parse_floats(input("Strike grid (comma-separated, % or decimals): "), pct=True)
        if Ks.size == 0:
            raise ValueError("Provide at least one strike.")
        vols = sabr_implied_vol_black(F, Ks, T, alpha, beta, rho, nu)
        prices = black76_price(F, Ks, T, vols, A, opt)
        # Table
        hdr = ("Strike", "Vol", "Price")
        w = [8, 10, 14]
        print(f"\n{hdr[0]:>{w[0]}} {hdr[1]:>{w[1]}} {hdr[2]:>{w[2]}}")
        print(f"{'-'*w[0]} {'-'*w[1]} {'-'*w[2]}")
        for k, v, p in zip(Ks, vols, prices):
            print(f"{k:>{w[0]}.6f} {v:>{w[1]}.8f} {p:>{w[2]}.10f}")

        # Plot
        plt.figure()
        plt.plot(Ks, vols, marker="o", linestyle="-")
        plt.xlabel("Strike K")
        plt.ylabel("Black implied vol")
        plt.title(f"SABR smile @ T={T}y (F={F:.6f}, beta={beta:.2f}, rho={rho:.2f}, nu={nu:.4f})")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()
        return

    raise ValueError("Mode must be 'single' or 'smile'.")

if __name__ == "__main__":
    main()
