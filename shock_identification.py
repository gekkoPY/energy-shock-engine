# shock_identification.py
import numpy as np
import pandas as pd
from data_loader import fetch_macro_data
from config import SHOCK_SIGMA_THRESHOLD

def build_shock_and_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the exogenous oil supply shock series (e_t),
    the discrete Strategic Petroleum Reserve release dummy (D_t),
    and the Federal Reserve monetary policy regime indicator (I_t).
    """
    df = df.copy()

    # 1. Compute daily returns and benchmark rate deltas
    df["ret_Brent"] = df["Brent"].pct_change() * 100        # % return
    df["ret_SP500_EW"] = df["SP500_EW"].pct_change() * 100  # % return
    df["d_OVX"] = df["OVX"].diff()                          # Absolute vol point delta
    df["d_DXY"] = df["DXY"].pct_change() * 100              # % return

    # Yield and breakeven deltas in basis points (bps)
    df["d_DGS2"] = df["Treasury_2Y"].diff() * 100
    df["d_DGS10"] = df["Treasury_10Y"].diff() * 100
    df["d_T5YIE"] = df["Breakeven_5Y"].diff() * 100
    df["d_DFII10"] = df["TIPS_10Y_Real"].diff() * 100

    # 2. Dynamic volatility standardization (60-day rolling window)
    # Strictly backward-looking window to prevent look-ahead bias
    rolling_std = df["ret_Brent"].rolling(window=60, min_periods=30).std()
    shock_condition = (
        (df["ret_Brent"] > SHOCK_SIGMA_THRESHOLD * rolling_std) &
        (df["ret_SP500_EW"] <= 0.0) &
        (df["d_OVX"] > 0.0)
    )

    # Continuous shock vector e_t: jump magnitude on identified dates, 0 elsewhere
    df["e_t"] = np.where(shock_condition, df["ret_Brent"], 0.0)
    df["is_shock"] = shock_condition.astype(int)

    # 3. Discrete Strategic Petroleum Reserve (SPR) control dummy (D_t in {0, 1})
    # Official coordinated release announcement dates (White House / DOE / IEA)
    spr_announcement_dates = pd.to_datetime([
        "2011-06-23",  # Libyan civil war supply disruption (IEA coordinated release)
        "2021-11-23",  # Post-COVID supply recovery release (50M barrels)
        "2022-03-01",  # Ukraine invasion initial multilateral response (60M barrels)
        "2022-03-31"   # Historic emergency release (180M barrels / 1M per day for 6 months)
    ])
    df["D_SPR"] = df.index.isin(spr_announcement_dates).astype(int)

    # 4. Monetary Policy Regime Indicator I_t: Restrictive vs Neutral/Easing Stance
    # Classifies stance as restrictive (I_t = 1) if the policy rate exceeds neutral (~2.50%)
    # without rapid easing, OR if an active rate-hiking trajectory occurred over the prior 2 months
    ff_rate = df["Fed_Funds_Target"].fillna(df["Fed_Funds_Effective"])
    ff_delta_6m = ff_rate - ff_rate.shift(126)
    ff_delta_2m = ff_rate - ff_rate.shift(42)

    is_restrictive_level = (ff_rate >= 2.50) & (ff_delta_6m >= -0.25)
    is_actively_hiking = ff_delta_2m > 0.05

    df["I_t"] = np.where(is_restrictive_level | is_actively_hiking, 1, 0)

    # Drop burn-in rows due to rolling lag operations (126 trading days)
    df = df.dropna().copy()

    num_shocks = df["is_shock"].sum()
    hiking_shocks = df.loc[df["is_shock"] == 1, "I_t"].sum()
    easing_shocks = num_shocks - hiking_shocks

    print(f"Identification pipeline complete. Pure supply shocks identified: {num_shocks} trading days.")
    print("Distribution by Federal Reserve policy stance:")
    print(f" - Restrictive / Hiking Stance (I_t = 1): {hiking_shocks} events")
    print(f" - Neutral / Easing Stance (I_t = 0): {easing_shocks} events")

    return df

if __name__ == "__main__":
    df_raw = fetch_macro_data()
    df_processed = build_shock_and_regimes(df_raw)

    shock_events = df_processed[df_processed["is_shock"] == 1][
        ["ret_Brent", "ret_SP500_EW", "d_OVX", "I_t", "e_t"]
    ]
    print("\nMost recent 5 supply shocks identified in sample:")
    print(shock_events.tail())
