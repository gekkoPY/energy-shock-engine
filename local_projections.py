# local_projections.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config import LP_HORIZONS
from data_loader import fetch_macro_data
from shock_identification import build_shock_and_regimes

def estimate_local_projections(
    df: pd.DataFrame, 
    target_col: str, 
    is_yield: bool = True, 
    lags: int = 2
) -> pd.DataFrame:
    """
    Estimates Jordà (2005) Local Projections for a specified target asset.
    Returns cumulative beta multipliers (unconditional, restrictive, easing),
    HAC Newey-West robust standard errors, p-values, and 90% confidence bands.
    
    Parameters:
    - df: Preprocessed macro panel containing shock indicators and state dummies
    - target_col: Column name of the response asset (e.g., 'Treasury_2Y', 'Breakeven_5Y', 'SP500_EW')
    - is_yield: True for rates/yields (computes basis points delta), False for price indices (% returns)
    - lags: Number of autoregressive dynamic control lags (p)
    """
    results = []

    for h in LP_HORIZONS:
        # 1. Construct cumulative response variable: y_{t+h} - y_{t-1}
        if is_yield:
            # Yield/spread deltas scaled to basis points (bps)
            y_cumulative = (df[target_col].shift(-h) - df[target_col].shift(1)) * 100.0
        else:
            # Cumulative percentage returns for equity indices and FX spot
            y_cumulative = (df[target_col].shift(-h) / df[target_col].shift(1) - 1.0) * 100.0

        reg_data = pd.DataFrame(index=df.index)
        reg_data["Y_h"] = y_cumulative

        # 2. Contemporaneous regressors and policy regime indicators
        reg_data["e_t"] = df["e_t"]
        reg_data["D_SPR"] = df["D_SPR"]
        reg_data["I_t"] = df["I_t"]
        reg_data["I_t_low"] = 1 - df["I_t"]

        # Regime interaction terms for State-Dependent LP
        reg_data["e_t_High"] = df["e_t"] * df["I_t"]
        reg_data["e_t_Low"] = df["e_t"] * (1 - df["I_t"])

        # 3. Dynamic lag controls (k = 1..p) to purge pre-existing momentum
        delta_y = df[target_col].diff() * (100.0 if is_yield else 1.0)
        for k in range(1, lags + 1):
            reg_data[f"lag_y_{k}"] = delta_y.shift(k)
            reg_data[f"lag_shock_{k}"] = df["e_t"].shift(k)

        # Drop NaN observations induced by lead/lag operators
        valid = reg_data.dropna()
        Y = valid["Y_h"]

        # --- MODEL 1: Unconditional Linear LP (Phase 2) ---
        X_linear_cols = ["e_t", "D_SPR"] + [f"lag_y_{k}" for k in range(1, lags + 1)] + [f"lag_shock_{k}" for k in range(1, lags + 1)]
        X_linear = sm.add_constant(valid[X_linear_cols])
        
        # Newey-West HAC covariance correction with bandwidth maxlags = h + 1
        model_linear = sm.OLS(Y, X_linear).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        
        beta_linear = model_linear.params["e_t"]
        se_linear = model_linear.bse["e_t"]
        pval_linear = model_linear.pvalues["e_t"]

        # --- MODEL 2: State-Dependent LP (Phase 3) ---
        X_state_cols = ["I_t", "I_t_low", "e_t_High", "e_t_Low", "D_SPR"] + \
                       [f"lag_y_{k}" for k in range(1, lags + 1)] + \
                       [f"lag_shock_{k}" for k in range(1, lags + 1)]
        # No constant added to prevent dummy variable collinearity with I_t and I_t_low
        X_state = valid[X_state_cols]
        
        model_state = sm.OLS(Y, X_state).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})

        beta_high = model_state.params["e_t_High"]
        se_high = model_state.bse["e_t_High"]
        pval_high = model_state.pvalues["e_t_High"]

        beta_low = model_state.params["e_t_Low"]
        se_low = model_state.bse["e_t_Low"]
        pval_low = model_state.pvalues["e_t_Low"]

        results.append({
            "horizon_days": h,
            "target": target_col,
            "beta_Linear": beta_linear,
            "se_Linear": se_linear,
            "pval_Linear": pval_linear,
            "beta_High": beta_high,
            "se_High": se_high,
            "pval_High": pval_high,
            "beta_Low": beta_low,
            "se_Low": se_low,
            "pval_Low": pval_low,
            # 90% confidence bands (z ~ 1.645) for IRF trajectory mapping
            "ci_high_up": beta_high + 1.645 * se_high,
            "ci_high_down": beta_high - 1.645 * se_high,
            "ci_low_up": beta_low + 1.645 * se_low,
            "ci_low_down": beta_low - 1.645 * se_low
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    df_raw = fetch_macro_data()
    df_clean = build_shock_and_regimes(df_raw)

    print("\n[ESTIMATION] Running Local Projections for US Treasury 2Y Yield (DGS2)...")
    res_2y = estimate_local_projections(df_clean, target_col="Treasury_2Y", is_yield=True)
    cols_to_print = ["horizon_days", "beta_Linear", "beta_High", "pval_High", "beta_Low", "pval_Low"]
    print(res_2y[cols_to_print].round(3).to_string(index=False))

    print("\n[ESTIMATION] Running Local Projections for 5Y Inflation Breakeven (T5YIE)...")
    res_5yie = estimate_local_projections(df_clean, target_col="Breakeven_5Y", is_yield=True)
    print(res_5yie[cols_to_print].round(3).to_string(index=False))
