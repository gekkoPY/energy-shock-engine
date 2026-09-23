# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from config import LP_HORIZONS
from data_loader import fetch_macro_data
from shock_identification import build_shock_and_regimes
from local_projections import estimate_local_projections

# 1. Page Configuration
st.set_page_config(
    page_title="Energy Shock Macro Engine",
    layout="wide"
)

# 2. Custom Terminal Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;600;700&display=swap');

    /* Global background and base typography */
    html, body, .stApp {
        background-color: #000000 !important;
        color: #D4D4D4 !important;
    }

    /* Monospace typography across dashboard elements */
    h1, h2, h3, h4, p, label, .stSelectbox, .stSlider, .stCaption {
        font-family: 'Roboto Mono', monospace !important;
    }

    /* Sidebar widget label formatting */
    [data-testid="stWidgetLabel"] p, 
    [data-testid="stWidgetLabel"] label,
    section[data-testid="stSidebar"] label {
        color: #FF9900 !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    /* Fallback font to prevent Streamlit SVG/Material icon corruption */
    [data-testid="stSidebarCollapseButton"] *,
    button[kind="header"] *,
    span[data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded', 'Source Sans Pro', sans-serif !important;
    }

    /* Dark sidebar surface */
    section[data-testid="stSidebar"] {
        background-color: #0A0A0A !important;
        border-right: 1px solid #1C1C1C !important;
    }

    /* Terminal amber headers */
    h1 {
        color: #FF9900 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        border-bottom: 2px solid #FF9900;
        padding-bottom: 6px;
        letter-spacing: -0.5px;
    }
    h2, h3, h4 {
        color: #FF9900 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
    }
    .stCaption {
        color: #777777 !important;
    }

    /* Trade expression playbook containers */
    .desk-box-cyan {
        background-color: #080808;
        border-left: 3px solid #00E5FF;
        border-top: 1px solid #1A1A1A;
        border-right: 1px solid #1A1A1A;
        border-bottom: 1px solid #1A1A1A;
        padding: 16px 20px;
        margin-top: 10px;
        border-radius: 2px;
        font-family: 'Roboto Mono', monospace;
    }
    .desk-box-green {
        background-color: #080808;
        border-left: 3px solid #00FF66;
        border-top: 1px solid #1A1A1A;
        border-right: 1px solid #1A1A1A;
        border-bottom: 1px solid #1A1A1A;
        padding: 16px 20px;
        margin-top: 10px;
        border-radius: 2px;
        font-family: 'Roboto Mono', monospace;
    }
    .desk-title-cyan {
        color: #00E5FF;
        font-weight: 700;
        margin-bottom: 10px;
        font-size: 13px;
        text-transform: uppercase;
    }
    .desk-title-green {
        color: #00FF66;
        font-weight: 700;
        margin-bottom: 10px;
        font-size: 13px;
        text-transform: uppercase;
    }

    /* Dark matrix table styling */
    .dark-matrix-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Roboto Mono', monospace;
        font-size: 12.5px;
        background-color: #050505;
        border: 1px solid #222222;
        margin-top: 14px;
        margin-bottom: 25px;
    }
    .dark-matrix-table th {
        background-color: #121212;
        color: #FF9900;
        text-align: left;
        padding: 11px 16px;
        border-bottom: 1px solid #333333;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .dark-matrix-table td {
        padding: 10px 16px;
        border-bottom: 1px solid #181818;
        color: #DDDDDD;
    }
    .dark-matrix-table tr:hover {
        background-color: #0F0F0F;
    }
</style>
""", unsafe_allow_html=True)

# 3. Data Ingestion Pipeline
@st.cache_data(show_spinner=False)
def load_all_data():
    raw_df = fetch_macro_data()
    clean_df = build_shock_and_regimes(raw_df)
    return clean_df

with st.spinner("Fetching macro series from FRED & Yahoo Finance..."):
    df = load_all_data()

# 4. Sidebar Controls
st.sidebar.markdown("### MODEL CONTROLS")
st.sidebar.markdown("---")

targets_meta = {
    "Breakeven_5Y": {"label": "5Y Inflation Breakeven (T5YIE)", "is_yield": True, "unit": "bps"},
    "Treasury_2Y": {"label": "US Treasury 2-Year Yield (DGS2)", "is_yield": True, "unit": "bps"},
    "Treasury_10Y": {"label": "US Treasury 10-Year Yield (DGS10)", "is_yield": True, "unit": "bps"},
    "TIPS_10Y_Real": {"label": "10Y Real TIPS Yield (DFII10)", "is_yield": True, "unit": "bps"},
    "SP500_EW": {"label": "S&P 500 Equal Weight (RSP)", "is_yield": False, "unit": "%"},
    "DXY": {"label": "U.S. Dollar Index (DXY)", "is_yield": False, "unit": "%"}
}

selected_target = st.sidebar.selectbox(
    "Target Asset (Local Projections):",
    options=list(targets_meta.keys()),
    format_func=lambda x: targets_meta[x]["label"]
)

lags_choice = st.sidebar.slider("Dynamic Control Lags (p):", min_value=1, max_value=5, value=2)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Telemetry Summary Block
hiking_count = df.loc[df['is_shock'] == 1, 'I_t'].sum()
easing_count = df['is_shock'].sum() - hiking_count
start_str = df.index[0].strftime('%Y-%m')
end_str = df.index[-1].strftime('%Y-%m')
total_shocks = df['is_shock'].sum()

telemetry_box = (
    '<div style="background-color: #080808; border: 1px solid #222222; border-left: 3px solid #FF9900; padding: 14px; border-radius: 2px; font-size: 12px; line-height: 1.8; font-family: \'Roboto Mono\', monospace;">'
    '<div style="color: #777777; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Data Window</div>'
    f'<div style="color: #FFFFFF; font-weight: 600; margin-bottom: 8px;">{start_str} ➔ {end_str}</div>'
    '<div style="color: #777777; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Identified Shocks (e<sub>t</sub>)</div>'
    f'<div style="color: #FFFFFF; font-weight: 600; margin-bottom: 8px;">{total_shocks} Trading Days</div>'
    '<div style="border-top: 1px solid #1C1C1C; padding-top: 8px; font-size: 11px;">'
    f'<div style="color: #CCCCCC;">• Restrictive (I<sub>t</sub> = 1): <span style="color: #FF9900; font-weight: 600;">{hiking_count}</span></div>'
    f'<div style="color: #CCCCCC;">• Easing / Neutral (I<sub>t</sub> = 0): <span style="color: #00FF66; font-weight: 600;">{easing_count}</span></div>'
    '</div></div>'
)
st.sidebar.markdown(telemetry_box, unsafe_allow_html=True)

# 5. Header Section
st.title("Energy Shock Transmission Engine")
st.markdown("<span style='color:#AAAAAA; font-size:14px; font-family: Roboto Mono, monospace;'>CROSS-ASSET LOCAL PROJECTIONS & LEAD-LAG LATENCY BENCHMARK</span>", unsafe_allow_html=True)
st.caption("Econometric Architecture: Jordà (2005) Local Projections | Newey-West HAC Covariance | High-Frequency Filter")

# 6. Latency Matrix
st.markdown("### 01 // CROSS-ASSET LATENCY MATRIX (80% PEAK ABSORPTION)")
st.markdown(
    "<span style='color:#888888; font-size:13px; font-family: Roboto Mono, monospace;'>Trading days (h) required for liquid benchmark assets "
    "to absorb 80% of total cumulative pass-through over the 60-day horizon.</span>",
    unsafe_allow_html=True
)

@st.cache_data(show_spinner=False)
def calculate_latency_matrix(data: pd.DataFrame, lags: int):
    matrix_rows = []
    all_lp_results = {}
    
    for asset, meta in targets_meta.items():
        res = estimate_local_projections(data, target_col=asset, is_yield=meta["is_yield"], lags=lags)
        all_lp_results[asset] = res
        
        betas = res["beta_Linear"].values
        abs_betas = np.abs(betas)
        peak_idx = np.argmax(abs_betas)
        peak_val = betas[peak_idx]
        threshold_80 = 0.80 * abs(peak_val)
        
        reach_80_idx = np.where(abs_betas >= threshold_80)[0]
        h_80 = LP_HORIZONS[reach_80_idx[0]] if len(reach_80_idx) > 0 else 60
        
        speed = "⚡ FAST (1-5D)" if h_80 <= 5 else ("⚖️ INTERMEDIATE (6-10D)" if h_80 <= 10 else "🐢 SLOW (>10D)")
        
        matrix_rows.append({
            "Asset Target": meta["label"],
            "Units": meta["unit"],
            "Cumulative Peak": f"{peak_val:+.2f} {meta['unit']}",
            "Peak Horizon": f"{LP_HORIZONS[peak_idx]} days",
            "80% Absorption": f"{h_80} days",
            "Market Profile": speed
        })
        
    return pd.DataFrame(matrix_rows), all_lp_results

latency_df, lp_cache = calculate_latency_matrix(df, lags=lags_choice)

table_html = "<table class='dark-matrix-table'><thead><tr>"
for col in latency_df.columns:
    table_html += f"<th>{col}</th>"
table_html += "</tr></thead><tbody>"
for _, row in latency_df.iterrows():
    table_html += "<tr>"
    for val in row:
        table_html += f"<td>{val}</td>"
    table_html += "</tr>"
table_html += "</tbody></table>"

st.markdown(table_html, unsafe_allow_html=True)

# 7. Plotly Impulse Response Functions (IRF)
st.markdown("---")
st.markdown(f"### 02 // DYNAMIC IMPULSE RESPONSE FUNCTIONS (IRF) — {targets_meta[selected_target]['label']}")

res_target = lp_cache[selected_target]
unit = targets_meta[selected_target]["unit"]

tab_regime, tab_linear = st.tabs(["[REGIME CONDITIONAL: FED STANCE]", "[AGGREGATE LINEAR SPECIFICATION]"])

# Base layout with high-contrast text and clean margins
bbg_layout_base = dict(
    plot_bgcolor="#000000",
    paper_bgcolor="#000000",
    font=dict(family="Roboto Mono, monospace", color="#CCCCCC", size=12),
    xaxis=dict(
        gridcolor="#181818",
        zerolinecolor="#333333",
        showgrid=True,
        linecolor="#333333"
    ),
    yaxis=dict(
        gridcolor="#181818",
        zerolinecolor="#333333",
        showgrid=True,
        linecolor="#333333"
    ),
    hovermode="x unified",
    height=480,
    margin=dict(l=40, r=40, t=65, b=40)
)

with tab_regime:
    fig_state = go.Figure()
    
    # Easing Path (Terminal Green)
    fig_state.add_trace(go.Scatter(
        x=res_target["horizon_days"],
        y=res_target["beta_Low"],
        mode="lines+markers",
        name="Fed Easing / Neutral (I_t = 0)",
        line=dict(color="#00FF66", width=2.5),
        marker=dict(size=6, symbol="square")
    ))
    fig_state.add_trace(go.Scatter(
        x=list(res_target["horizon_days"]) + list(res_target["horizon_days"][::-1]),
        y=list(res_target["ci_low_up"]) + list(res_target["ci_low_down"][::-1]),
        fill='toself',
        fillcolor='rgba(0, 255, 102, 0.10)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name="90% CI Easing"
    ))

    # Restrictive Path (Amber Orange)
    fig_state.add_trace(go.Scatter(
        x=res_target["horizon_days"],
        y=res_target["beta_High"],
        mode="lines+markers",
        name="Fed Restrictive / Hiking (I_t = 1)",
        line=dict(color="#FF9900", width=2.5, dash="dash"),
        marker=dict(size=6, symbol="circle")
    ))
    fig_state.add_trace(go.Scatter(
        x=list(res_target["horizon_days"]) + list(res_target["horizon_days"][::-1]),
        y=list(res_target["ci_high_up"]) + list(res_target["ci_high_down"][::-1]),
        fill='toself',
        fillcolor='rgba(255, 153, 0, 0.10)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name="90% CI Hiking"
    ))

    fig_state.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=1)
    fig_state.update_layout(
        **bbg_layout_base,
        title=f"STATE-DEPENDENT RESPONSE BY MONETARY REGIME ({unit})",
        xaxis_title="TRADING DAYS POST-SHOCK (HORIZON H)",
        yaxis_title=f"ESTIMATED PASS-THROUGH ({unit})",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#FFFFFF", size=12, family="Roboto Mono, monospace"),
            bgcolor="rgba(10, 10, 10, 0.85)",
            bordercolor="#333333",
            borderwidth=1
        )
    )
    st.plotly_chart(fig_state, use_container_width=True)

with tab_linear:
    fig_lin = go.Figure()
    fig_lin.add_trace(go.Scatter(
        x=res_target["horizon_days"],
        y=res_target["beta_Linear"],
        mode="lines+markers",
        name="Unconditional LP",
        line=dict(color="#00E5FF", width=2.5),
        marker=dict(size=6, symbol="diamond")
    ))
    fig_lin.add_hline(y=0, line_dash="solid", line_color="#333333", line_width=1)
    fig_lin.update_layout(
        **bbg_layout_base,
        title=f"UNCONDITIONAL LINEAR PASS-THROUGH TRAJECTORY ({unit})",
        xaxis_title="TRADING DAYS POST-SHOCK (HORIZON H)",
        yaxis_title=f"ESTIMATED PASS-THROUGH ({unit})",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#FFFFFF", size=12, family="Roboto Mono, monospace"),
            bgcolor="rgba(10, 10, 10, 0.85)",
            bordercolor="#333333",
            borderwidth=1
        )
    )
    st.plotly_chart(fig_lin, use_container_width=True)

# 8. Trade Expression Playbook & Diagnostics
st.markdown("---")
st.markdown("### 03 // TRADE EXPRESSION & RISK MAPPING")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="desk-box-cyan">
        <div class="desk-title-cyan">🎯 QUANTITATIVE LEAD-LAG DIAGNOSTICS</div>
        <ul style="margin: 0; padding-left: 18px; font-size: 13px; line-height: 1.6; color: #CCCCCC;">
            <li><b>Front-End Pricing Speed:</b> 5Y Inflation Breakevens discount mechanical energy pass-through within <b>1 to 5 trading days</b>.</li>
            <li><b>Macro Transmission Drag:</b> Sovereign benchmark curves (2Y/10Y) lag by <b>10 to 20 trading days</b>, reacting only upon realized CPI verification.</li>
            <li><b>Actionable Window:</b> A systematic mispricing window of <b>10 to 15 trading days</b> emerges before policy-rate expectations fully adjust.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="desk-box-green">
        <div class="desk-title-green">💼 TRADE EXPRESSION PLAYBOOK</div>
        <ul style="margin: 0; padding-left: 18px; font-size: 13px; line-height: 1.6; color: #CCCCCC;">
            <li><b>Linear Delta (Curve Spread):</b> 2s10s Bear Flattener structured via CME Treasury Futures (<span style="color:#FF9900; font-weight:700;">TU</span> short vs <span style="color:#FF9900; font-weight:700;">TY</span> long) on a strict <b>DV01-neutral</b> weighting ratio.</li>
            <li><b>Relative Value (Inflation vs Nominal):</b> Synthetic 5Y Breakeven expansion (Long 5Y TIPS vs Short Nominal 5Y Treasuries) isolating CPI drift from outright duration exposure.</li>
            <li><b>Macro FX Proxy:</b> Short currencies of structural net energy importers (e.g., EUR/USD, USD/JPY) against the USD to capture terms-of-trade degradation.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
