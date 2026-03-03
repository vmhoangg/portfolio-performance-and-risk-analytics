# app.py — Tab 1: Overview
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from math import isfinite
import plotly.graph_objects as go

# ---------- CONFIG PATHS ----------
P_PERF = Path("data/analytics/performance_metrics.parquet")     # ret_p, NAV, drawdown, Sharpe_252, VolAnn_20, TE_252
P_RISK = Path("data/analytics/risk_metrics.parquet")            # VaR95_hist, ES95_hist
P_BENCH= Path("data/processed/benchmark_ret.parquet")           # SPY returns (daily)
P_REG  = Path("data/processed/regime_labels.parquet")           # optional: column 'regime'

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

# ---------- LOADERS ----------
@st.cache_data
def load_perf():
    df = pd.read_parquet(P_PERF)
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data
def load_risk():
    if P_RISK.exists():
        df = pd.read_parquet(P_RISK)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()

@st.cache_data
def load_bench_nav():
    if not P_BENCH.exists():
        return pd.Series(dtype=float), pd.Series(dtype=float)
    b = pd.read_parquet(P_BENCH)["SPY"]
    b.index = pd.to_datetime(b.index)
    nav_spy = (1 + b.fillna(0)).cumprod()
    return b, nav_spy.rename("NAV_SPY")

@st.cache_data
def load_regimes():
    if not P_REG.exists():
        return pd.Series(dtype=object)
    lab = pd.read_parquet(P_REG).squeeze()
    # chấp nhận cả DataFrame 1 cột hoặc Series
    if isinstance(lab, pd.DataFrame) and lab.shape[1] == 1:
        lab = lab.iloc[:,0]
    lab.index = pd.to_datetime(lab.index)
    lab.name = "regime"
    return lab

# ---------- DATA ----------
perf = load_perf()
risk = load_risk()
ret_spy, nav_spy = load_bench_nav()
regimes = load_regimes()

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("Filters")
min_d = perf.index.min().date()
max_d = perf.index.max().date()
date_range = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

use_regime = st.sidebar.checkbox("Filter by market regime", value=False)
selected_regimes = []
if use_regime and not regimes.empty:
    all_regs = list(regimes.dropna().unique())
    selected_regimes = st.sidebar.multiselect("Select regimes", options=all_regs, default=all_regs)

# ---------- APPLY FILTER ----------
def apply_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.loc[(df.index.date >= date_range[0]) & (df.index.date <= date_range[1])]
    if use_regime and not regimes.empty and selected_regimes:
        reg = regimes.reindex(df.index)
        df = df[reg.isin(selected_regimes)]
    return df

perf_f = apply_filter(perf)
risk_f = apply_filter(risk) if not risk.empty else risk
nav_spy_f = apply_filter(nav_spy.to_frame()).iloc[:,0] if not nav_spy.empty else nav_spy

# ---------- KPIs ----------
st.title("Overview")

col1, col2, col3, col4, col5 = st.columns(5)

# NAV cuối kỳ (portfolio)
nav_last = float(perf_f["NAV"].iloc[-1]) if not perf_f.empty else np.nan
# Max Drawdown
dd_min = float(perf_f["drawdown"].min()) if not perf_f.empty else np.nan
# Mean Sharpe (252)
sh_mean = float(perf_f["Sharpe_252"].dropna().mean()) if "Sharpe_252" in perf_f and not perf_f["Sharpe_252"].dropna().empty else np.nan
# VaR/ES cuối kỳ (nếu có file risk)
var_last = float(risk_f["VaR95_hist"].dropna().iloc[-1]) if not risk_f.empty and "VaR95_hist" in risk_f and not risk_f["VaR95_hist"].dropna().empty else np.nan
es_last  = float(risk_f["ES95_hist"].dropna().iloc[-1])  if not risk_f.empty and "ES95_hist"  in risk_f and not risk_f["ES95_hist"].dropna().empty  else np.nan

col1.metric("NAV (Portfolio)", f"{nav_last:,.3f}" if np.isfinite(nav_last) else "NA")
col2.metric("Max Drawdown", f"{dd_min:.2%}" if np.isfinite(dd_min) else "NA")
col3.metric("Mean Sharpe (252d)", f"{sh_mean:,.3f}" if np.isfinite(sh_mean) else "NA")
col4.metric("VaR95 (last, daily)", f"{var_last:.2%}" if np.isfinite(var_last) else "NA")
col5.metric("ES95 (last, daily)",  f"{es_last:.2%}"  if np.isfinite(es_last)  else "NA")

st.markdown("---")

# ---------- NAV CHART (Portfolio vs SPY) ----------
fig = go.Figure()

# Portfolio NAV
if not perf_f.empty:
    fig.add_trace(go.Scatter(
        x=perf_f.index, y=perf_f["NAV"], mode="lines", name="Portfolio NAV"
    ))

# SPY NAV
if not nav_spy_f.empty:
    fig.add_trace(go.Scatter(
        x=nav_spy_f.index, y=nav_spy_f.values, mode="lines", name="SPY NAV"
    ))

# Regime shading (nếu có)
if not regimes.empty:
    # Dùng nhãn regime theo filter thời gian
    reg_sel = regimes.reindex(perf_f.index if not perf_f.empty else regimes.index).fillna("Unlabeled")
    # tìm các đoạn liên tục theo cùng regime
    if not reg_sel.empty:
        run_start = reg_sel.index[0]
        run_lab = reg_sel.iloc[0]
        for t, lab in zip(reg_sel.index[1:], reg_sel.iloc[1:]):
            if lab != run_lab:
                fig.add_vrect(x0=run_start, x2=t, fillcolor="lightgrey", opacity=0.15, layer="below", line_width=0)
                run_start = t
                run_lab = lab
        # close last segment
        fig.add_vrect(x0=run_start, x2=reg_sel.index[-1], fillcolor="lightgrey", opacity=0.15, layer="below", line_width=0)

fig.update_layout(
    title="NAV: Portfolio vs SPY",
    xaxis_title="Date",
    yaxis_title="NAV (Base = 1.0)",
    legend=dict(orientation="h", y=1.08, x=0),
    margin=dict(l=20, r=20, t=60, b=20),
    height=480
)
st.plotly_chart(fig, use_container_width=True)

# ---------- QUICK STATS TABLE ----------
from datetime import date, datetime

def _fmt_pct(x):
    return "NA" if x is None or (isinstance(x, float) and (not np.isfinite(x))) else f"{x:.2%}"

def _fmt_f3(x):
    return "NA" if x is None or (isinstance(x, float) and (not np.isfinite(x))) else f"{x:,.3f}"

def _fmt_date(x):
    if x is None:
        return "NA"
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        return str(x)

with st.expander("Summary statistics (current filter)"):
    # Lấy giá trị thô (an toàn với DF rỗng)
    start_d = perf_f.index.min() if not perf_f.empty else None
    end_d   = perf_f.index.max() if not perf_f.empty else None
    obs_n   = len(perf_f)

    if not perf_f.empty and "ret_p" in perf_f:
        rp = perf_f["ret_p"].dropna()
        avg_ret = float(rp.mean()) if not rp.empty else np.nan
        std_ret = float(rp.std(ddof=0)) if not rp.empty else np.nan
    else:
        avg_ret = np.nan
        std_ret = np.nan

    ann_vol = std_ret * np.sqrt(252) if np.isfinite(std_ret) else np.nan
    ann_sh  = ((avg_ret / std_ret) * np.sqrt(252)) if (np.isfinite(avg_ret) and np.isfinite(std_ret) and std_ret > 0) else np.nan

    # Dòng dữ liệu đã format thành CHUỖI -> tránh pandas ép dtype cả cột
    rows = [
        ("Start",            _fmt_date(start_d)),
        ("End",              _fmt_date(end_d)),
        ("Obs",              f"{obs_n}"),
        ("Avg Daily Return", _fmt_pct(avg_ret)),
        ("Std Daily Return", _fmt_pct(std_ret)),
        ("Ann. Vol (≈)",     _fmt_pct(ann_vol)),
        ("Ann. Sharpe (≈)",  _fmt_f3(ann_sh)),
        ("Min Drawdown",     _fmt_pct(dd_min if 'dd_min' in locals() else np.nan)),
        ("VaR95 (last)",     _fmt_pct(var_last if 'var_last' in locals() else np.nan)),
        ("ES95 (last)",      _fmt_pct(es_last  if 'es_last'  in locals() else np.nan)),
    ]

    df_stats = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.dataframe(df_stats, hide_index=True)
# ----- END REPLACE BLOCK -----

# =========================
# TAB 2 — PERFORMANCE
# =========================

# Đường dẫn dùng lại
P_PERF = Path("data/analytics/performance_metrics.parquet")
P_BENCH= Path("data/processed/benchmark_ret.parquet")

# --- Load (dùng lại cache nếu bạn đã có wrapper; nếu không, dùng trực tiếp) ---
def _load_perf_local():
    df = pd.read_parquet(P_PERF)
    df.index = pd.to_datetime(df.index)
    return df

def _load_bench_local():
    b = pd.read_parquet(P_BENCH)["SPY"]
    b.index = pd.to_datetime(b.index)
    return b

perf_all = _load_perf_local()
ret_spy_all = _load_bench_local()

st.header("Performance")

# --- Bộ lọc thời gian cho riêng Tab 2 ---
min_d2 = perf_all.index.min().date()
max_d2 = perf_all.index.max().date()
date_range2 = st.date_input(
    "Choose a date (Tab 2)",
    value=(min_d2, max_d2),
    min_value=min_d2, max_value=max_d2, key="perf_date_range"
)

# Filter theo thời gian
perf = perf_all.loc[(perf_all.index.date >= date_range2[0]) & (perf_all.index.date <= date_range2[1])]
ret_spy = ret_spy_all.loc[(ret_spy_all.index.date >= date_range2[0]) & (ret_spy_all.index.date <= date_range2[1])]

# --- Tuỳ chọn window ---
c1, c2, c3 = st.columns(3)
win_sh = c1.number_input("Rolling window for Sharpe (day)", min_value=60, max_value=756, value=252, step=10, key="win_sharpe")
win_vol = c2.number_input("Rolling window for Volatility (day)", min_value=10, max_value=252, value=20, step=5, key="win_vol")
win_te = c3.number_input("Rolling window for Tracking Error (day)", min_value=60, max_value=756, value=252, step=10, key="win_te")

# --- Tính lại các chỉ số theo window người dùng ---
ret_p = perf["ret_p"].copy()

roll_sharpe = (ret_p.rolling(win_sh).mean() / ret_p.rolling(win_sh).std(ddof=0)) * np.sqrt(252)
roll_sharpe.name = f"Sharpe_{win_sh}"

roll_vol = ret_p.rolling(win_vol).std(ddof=0) * np.sqrt(252)
roll_vol.name = f"VolAnn_{win_vol}"

te = (ret_p - ret_spy.reindex(ret_p.index)).rolling(win_te).std(ddof=0) * np.sqrt(252)
te.name = f"TE_{win_te}"

# --- Hàng KPI nhanh cho khoảng thời gian đã chọn ---
k1, k2, k3, k4 = st.columns(4)
nav_last = perf["NAV"].iloc[-1] if not perf.empty else np.nan
dd_min = perf["drawdown"].min() if "drawdown" in perf and not perf.empty else np.nan
k1.metric("NAV (End of period)", f"{nav_last:,.3f}" if np.isfinite(nav_last) else "NA")
k2.metric("Max Drawdown ", f"{dd_min:.2%}" if np.isfinite(dd_min) else "NA")
k3.metric(f"Sharpe {win_sh} (last, daily)", f"{roll_sharpe.dropna().iloc[-1]:.3f}" if roll_sharpe.dropna().size else "NA")
k4.metric(f"TE {win_te} (last, daily)", f"{te.dropna().iloc[-1]:.3f}" if te.dropna().size else "NA")

st.markdown("---")

# --- Chart 1: Rolling Sharpe ---
fig_sh = go.Figure()
if roll_sharpe.dropna().size:
    fig_sh.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, mode="lines", name=f"Sharpe ({win_sh})"))
fig_sh.update_layout(
    title=f"Rolling Sharpe ({win_sh} days)",
    xaxis_title="Date", yaxis_title="Sharpe",
    height=360, margin=dict(l=20, r=20, t=60, b=20)
)
st.plotly_chart(fig_sh, use_container_width=True)

# --- Chart 2: Rolling Volatility ---
fig_vol = go.Figure()
if roll_vol.dropna().size:
    fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol, mode="lines", name=f"VolAnn ({win_vol})"))
fig_vol.update_layout(
    title=f"Rolling annual Volatility ({win_vol} days)",
    xaxis_title="Date", yaxis_title="Vol (ann.)",
    height=360, margin=dict(l=20, r=20, t=60, b=20)
)
st.plotly_chart(fig_vol, use_container_width=True)

# --- Chart 3: Tracking Error ---
fig_te = go.Figure()
if te.dropna().size:
    fig_te.add_trace(go.Scatter(x=te.index, y=te, mode="lines", name=f"TE ({win_te})"))
fig_te.update_layout(
    title=f"Tracking Error vs SPY ({win_te} days, ann.)",
    xaxis_title="Date", yaxis_title="TE (ann.)",
    height=360, margin=dict(l=20, r=20, t=60, b=20)
)
st.plotly_chart(fig_te, use_container_width=True)

st.markdown("---")

# --- Monthly returns (Portfolio vs SPY) ---
# Dùng month-end ('ME') để tránh cảnh báo 'M' deprecated
ret_p_m = ((1 + ret_p).resample('ME').prod() - 1).rename("Portfolio")
ret_b_m = ((1 + ret_spy.reindex(ret_p.index)).resample('ME').prod() - 1).rename("SPY")
monthly = pd.concat([ret_p_m, ret_b_m], axis=1).dropna(how="all")

st.subheader("Monthly Returns — Portfolio vs SPY")

fig_m = go.Figure()
if not monthly.empty:
    fig_m.add_bar(x=monthly.index, y=monthly["Portfolio"], name="Portfolio")
    if "SPY" in monthly.columns:
        fig_m.add_bar(x=monthly.index, y=monthly["SPY"], name="SPY")
fig_m.update_layout(
    barmode="group",
    title="Monthly Returns (grouped)",
    xaxis_title="Month",
    yaxis_title="Return",
    height=420, margin=dict(l=20, r=20, t=60, b=20)
)
st.plotly_chart(fig_m, use_container_width=True)

# --- Bảng Top/Bottom months theo Portfolio ---
if not monthly.empty and "Portfolio" in monthly.columns:
    top_n = 5
    bot_n = 5
    top_tbl = monthly["Portfolio"].sort_values(ascending=False).head(top_n).to_frame("Return")
    bot_tbl = monthly["Portfolio"].sort_values(ascending=True).head(bot_n).to_frame("Return")

    ctop, cbot = st.columns(2)
    with ctop:
        st.markdown(f"**Top {top_n} months (Portfolio)**")
        st.dataframe(top_tbl.style.format({"Return": "{:.2%}"}))
    with cbot:
        st.markdown(f"**Bottom {bot_n} months (Portfolio)**")
        st.dataframe(bot_tbl.style.format({"Return": "{:.2%}"}))

# --- Gợi ý giải thích nhanh ---
with st.expander("Gợi ý diễn giải nhanh"):
    st.write(
        """
- **Rolling Sharpe** cao & ổn định → khẩu vị rủi ro/hiệu suất tốt trong vùng chọn.
- **Volatility** tăng mạnh ở giai đoạn khủng hoảng (COVID, thắt chặt tiền tệ) → cần so với Tracking Error để biết độ “lệch” so với SPY.
- **Tracking Error** cao → danh mục đang đi khác SPY; thấp → bám sát benchmark.
- **Monthly returns**: xem Top/Bottom để tìm tháng đặc biệt, liên kết với **regime** ở Tab 6.
        """
    )


# =========================
# TAB 3 — RISK
# =========================
import plotly.graph_objects as go
from pathlib import Path

P_PERF = Path("data/analytics/performance_metrics.parquet")   # chứa 'ret_p'
P_REG  = Path("data/processed/regime_labels.parquet")         # optional

def _load_perf_risk_base():
    df = pd.read_parquet(P_PERF)
    df.index = pd.to_datetime(df.index)
    return df

def _load_regimes_local():
    if not P_REG.exists():
        return pd.Series(dtype=object)
    lab = pd.read_parquet(P_REG).squeeze()
    if isinstance(lab, pd.DataFrame) and lab.shape[1] == 1:
        lab = lab.iloc[:,0]
    lab.index = pd.to_datetime(lab.index)
    lab.name = "regime"
    return lab

def var_hist_series(ret: pd.Series, alpha: float = 0.95, win: int = 252) -> pd.Series:
    # Rolling historical VaR (left tail): quantile(1-alpha)
    return ret.rolling(win).quantile(1 - alpha).rename(f"VaR{int(alpha*100)}_hist_{win}")

def es_hist_series(ret: pd.Series, alpha: float = 0.95, win: int = 252) -> pd.Series:
    var = var_hist_series(ret, alpha=alpha, win=win)
    arr = ret.values
    es_vals = np.full(len(ret), np.nan, dtype=float)
    for i in range(len(ret)):
        if i < win - 1:
            continue
        window = arr[i-(win-1):i+1]
        thr = var.iloc[i]
        tail = window[window <= thr]
        es_vals[i] = np.mean(tail) if tail.size > 0 else np.nan
    return pd.Series(es_vals, index=ret.index, name=f"ES{int(alpha*100)}_hist_{win}")

def var_point(ret: pd.Series, alpha: float) -> float:
    # VaR điểm (toàn kỳ lọc) — left-tail quantile
    if ret.dropna().empty: return np.nan
    return float(np.quantile(ret.dropna(), 1 - alpha))

def es_point(ret: pd.Series, alpha: float) -> float:
    v = var_point(ret, alpha)
    if not np.isfinite(v): return np.nan
    tail = ret.dropna()[ret.dropna() <= v]
    return float(tail.mean()) if not tail.empty else np.nan

st.header("Risk")

# --- Load & filter thời gian riêng cho Tab 3 ---
perf_all = _load_perf_risk_base()
regimes_all = _load_regimes_local()

min_d3 = perf_all.index.min().date()
max_d3 = perf_all.index.max().date()
date_range3 = st.date_input(
    "Chọn khoảng thời gian (Tab 3)",
    value=(min_d3, max_d3), min_value=min_d3, max_value=max_d3, key="risk_date_range"
)

perf = perf_all.loc[(perf_all.index.date >= date_range3[0]) & (perf_all.index.date <= date_range3[1])]
regimes = regimes_all.reindex(perf.index) if not regimes_all.empty else regimes_all
ret_p = perf["ret_p"] if "ret_p" in perf else pd.Series(dtype=float, index=perf.index)

# --- Tham số rủi ro ---
c1, c2, c3, c4 = st.columns(4)
alpha_label = c1.selectbox("Confidence level (α)", ["90%", "95%", "97.5%", "99%"], index=1, key="risk_alpha")
alpha = {"90%":0.90, "95%":0.95, "97.5%":0.975, "99%":0.99}[alpha_label]
win = c2.number_input("Cửa sổ rolling (ngày)", min_value=60, max_value=756, value=252, step=10, key="risk_win")
show_ret = c3.checkbox("Hiển thị đường lợi suất (mờ)", value=True, key="risk_show_ret")
per_regime_table = c4.checkbox("Tính bảng theo Regime", value=True, key="risk_reg_tbl")

# --- Rolling VaR/ES ---
var_roll = var_hist_series(ret_p, alpha=alpha, win=int(win))
es_roll  = es_hist_series(ret_p, alpha=alpha, win=int(win))

# --- KPI nhanh ---
cK1, cK2, cK3 = st.columns(3)
var_last = var_roll.dropna().iloc[-1] if var_roll.dropna().size else np.nan
es_last  = es_roll.dropna().iloc[-1]  if es_roll.dropna().size else np.nan
ex_viol  = (ret_p < var_roll).sum() if var_roll.notna().any() else np.nan
tot_obs  = int(min(ret_p.dropna().shape[0], var_roll.dropna().shape[0])) if var_roll.notna().any() else 0
expected_rate = (1 - alpha)
viol_rate = (ex_viol / tot_obs) if tot_obs > 0 else np.nan

cK1.metric(f"VaR{int(alpha*100)} last (daily)", f"{var_last:.2%}" if np.isfinite(var_last) else "NA")
cK2.metric(f"ES{int(alpha*100)} last (daily)",  f"{es_last:.2%}"  if np.isfinite(es_last)  else "NA")
cK3.metric("Violation rate", 
           f"{viol_rate:.2%}" if np.isfinite(viol_rate) else "NA",
           help=f"Kỳ vọng ≈ {(expected_rate):.2%} (theo α)")

st.markdown("---")

# --- Chart 1: Rolling VaR & ES ---
fig_risk = go.Figure()
if show_ret and not perf.empty:
    fig_risk.add_trace(go.Scatter(
        x=ret_p.index, y=ret_p, name="Daily return", mode="lines",
        opacity=0.2
    ))
if var_roll.dropna().size:
    fig_risk.add_trace(go.Scatter(x=var_roll.index, y=var_roll, name=f"VaR{int(alpha*100)} ({win}d)", mode="lines"))
if es_roll.dropna().size:
    fig_risk.add_trace(go.Scatter(x=es_roll.index, y=es_roll, name=f"ES{int(alpha*100)} ({win}d)", mode="lines"))
fig_risk.update_layout(
    title=f"Rolling Historical VaR / ES (α={alpha_label}, window={win}d)",
    xaxis_title="Date", yaxis_title="Return",
    height=420, margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(orientation="h", y=1.08, x=0)
)
st.plotly_chart(fig_risk, use_container_width=True)

# --- Chart 2: Histogram + VaR/ES điểm (toàn vùng lọc) ---
var_p = var_point(ret_p, alpha)
es_p  = es_point(ret_p, alpha)
fig_hist = go.Figure()
hist_vals = ret_p.dropna().values
if hist_vals.size:
    fig_hist.add_histogram(x=hist_vals, nbinsx=60, name="Returns hist", opacity=0.8)
    if np.isfinite(var_p):
        fig_hist.add_vline(x=var_p, line_dash="dash", annotation_text=f"VaR {alpha_label}", annotation_position="top left")
    if np.isfinite(es_p):
        fig_hist.add_vline(x=es_p, line_dash="dot",  annotation_text=f"ES {alpha_label}",  annotation_position="top left")
fig_hist.update_layout(
    title=f"Distribution of returns (with VaR/ES at α={alpha_label})",
    xaxis_title="Daily return", yaxis_title="Frequency",
    height=420, margin=dict(l=20, r=20, t=60, b=20)
)
st.plotly_chart(fig_hist, use_container_width=True)

# --- Bảng 10 ngày tệ nhất + đánh dấu vi phạm ---
if not perf.empty:
    worst = ret_p.sort_values().head(10).to_frame("Return")
    if var_roll.notna().any():
        worst["VaR_roll"] = var_roll.reindex(worst.index)
        worst["Breach?"]  = worst["Return"] < worst["VaR_roll"]
    st.subheader("Worst 10 days (current filter)")
    st.dataframe(worst.style.format({"Return":"{:.2%}", "VaR_roll":"{:.2%}"}))

# --- Bảng theo Regime (nếu có) ---
if per_regime_table and not regimes.empty:
    df = pd.DataFrame({"ret_p": ret_p, "regime": regimes})
    df = df.dropna(subset=["ret_p", "regime"])
    def _agg(g):
        r = g["ret_p"].dropna()
        if r.empty:
            return pd.Series({"Mean":np.nan, "AnnVol":np.nan, f"VaR{int(alpha*100)}":np.nan, f"ES{int(alpha*100)}":np.nan, "Obs":0})
        mean = r.mean()
        volA = r.std(ddof=0) * np.sqrt(252)
        v = var_point(r, alpha)
        e = es_point(r, alpha)
        return pd.Series({"Mean":mean, "AnnVol":volA, f"VaR{int(alpha*100)}":v, f"ES{int(alpha*100)}":e, "Obs":len(r)})
    tbl = df.groupby("regime", dropna=False).apply(_agg)
    # format đẹp
    fmt_cols_pct = ["Mean", f"VaR{int(alpha*100)}", f"ES{int(alpha*100)}"]
    for c in fmt_cols_pct:
        if c in tbl.columns:
            tbl[c] = tbl[c].map(lambda x: f"{x:.2%}" if np.isfinite(x) else "NA")
    if "AnnVol" in tbl.columns:
        tbl["AnnVol"] = tbl["AnnVol"].map(lambda x: f"{x:.2%}" if np.isfinite(x) else "NA")
    st.subheader("Risk by Market Regime")
    st.dataframe(tbl)


# =========================
# TAB 4 — ATTRIBUTION (SECTOR vs SPY)
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ---- Đường dẫn dữ liệu (chỉnh nếu khác) ----
P_ATTR   = Path("data/processed/attribution_results_sector_spy.parquet")  # A/S/I/Total (daily)
P_ACTIVE = Path("data/processed/active_returns_from_weights.parquet")     # Active (daily) - optional
P_RET    = Path("data/processed/returns_linear.parquet")                  # fallback để tự tính Active
P_W      = Path("data/portfolio_weights/ew_daily_wide.parquet")           # khi cần tự tính

st.header("Attribution (Brinson–Fachler • Sector vs SPY)")

# ---------- Load helpers ----------
def _load_attr(path=P_ATTR):
    if not path.exists():
        st.error(f"Không tìm thấy {path}. Hãy chạy scripts 04–06 để tạo dữ liệu attribution.")
        st.stop()
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    need = {"Allocation","Selection","Interaction","Total"}
    miss = need - set(df.columns)
    if miss:
        st.error(f"Thiếu cột trong attribution: {miss}")
        st.stop()
    return df.sort_index()

def _load_active():
    # ưu tiên file active đã tính sẵn; nếu không có thì tự tính từ weights + returns
    if P_ACTIVE.exists():
        act = pd.read_parquet(P_ACTIVE)
        if isinstance(act, pd.DataFrame) and "Active" in act.columns:
            s = act["Active"].copy()
        elif isinstance(act, pd.Series):
            s = act.copy()
        else:
            s = None
        if s is not None:
            s.index = pd.to_datetime(s.index)
            s.name = "Active"
            return s.sort_index()

    # fallback: tự build active = (w_{t-1}·r_assets) − r_SPY
    if not P_RET.exists() or not P_W.exists():
        return None
    rets = pd.read_parquet(P_RET); rets.index = pd.to_datetime(rets.index)
    asset_cols = [c for c in rets.columns if c != "SPY"]
    w = pd.read_parquet(P_W); w.index = pd.to_datetime(w.index)
    w_prev = w.shift(1).reindex(rets.index).ffill()
    ret_p = (w_prev.reindex(columns=asset_cols).fillna(0.0) * rets[asset_cols]).sum(axis=1)
    active = (ret_p - rets["SPY"]).rename("Active")
    return active.sort_index()

def _aggregate(df: pd.DataFrame, mode: str, freq: str) -> pd.DataFrame:
    """mode: Period sum | Cumulative ; freq: D | ME | QE | YE"""
    if mode == "Cumulative":
        if freq == "D":
            return df.cumsum()
        return df.cumsum().resample(freq).last()
    # Period sum
    if freq == "D":
        return df
    return df.resample(freq).sum()

# ---------- Load ----------
attr_all = _load_attr()
active_all = _load_active()

# ---------- Controls ----------
min_d, max_d = attr_all.index.min().date(), attr_all.index.max().date()
date_range = st.date_input(
    "Khoảng thời gian", value=(min_d, max_d), min_value=min_d, max_value=max_d, key="attr_daterange"
)
freq_label = st.selectbox("Kỳ tổng hợp", ["Daily","Monthly","Quarterly","Yearly"], index=1, key="attr_freq")
freq_map = {"Daily":"D","Monthly":"ME","Quarterly":"QE","Yearly":"YE"}
freq = freq_map[freq_label]
view_mode = st.radio("Kiểu hiển thị", ["Period sum","Cumulative"], index=0, horizontal=True, key="attr_mode")

# ---------- Filter ----------
attr = attr_all.loc[(attr_all.index.date >= date_range[0]) & (attr_all.index.date <= date_range[1])]
active = None if active_all is None else active_all.loc[attr.index.min():attr.index.max()]

# ---------- Sanity: A+S+I = Total ----------
comp_sum = attr[["Allocation","Selection","Interaction"]].sum(axis=1)
diff_max = float(np.abs(comp_sum - attr["Total"]).max())
if diff_max > 1e-9:
    st.warning(f"Chênh lệch nhỏ A+S+I vs Total (max |diff| = {diff_max:.2e}).")

# ---------- Aggregate ----------
attr_period = _aggregate(attr[["Allocation","Selection","Interaction","Total"]], view_mode, freq)
active_period = None if active is None else _aggregate(active.to_frame(), view_mode, freq).iloc[:,0]

# ---------- KPI ----------
c1,c2,c3,c4 = st.columns(4)
last = attr_period.tail(1)
c1.metric(f"Allocation ({freq_label})", f"{last['Allocation'].iloc[0]:.2%}" if not last.empty else "NA")
c2.metric(f"Selection ({freq_label})",  f"{last['Selection'].iloc[0]:.2%}"  if not last.empty else "NA")
c3.metric(f"Interaction ({freq_label})",f"{last['Interaction'].iloc[0]:.2%}" if not last.empty else "NA")
c4.metric(f"Total ({freq_label})",      f"{last['Total'].iloc[0]:.2%}"      if not last.empty else "NA")

st.markdown("---")

# ---------- Chart 1: Stacked bars (A/S/I) + Active line ----------
fig = go.Figure()
if not attr_period.empty:
    x = attr_period.index
    for col in ["Allocation","Selection","Interaction"]:
        fig.add_bar(x=x, y=attr_period[col], name=col)
if active_period is not None and not active_period.empty:
    fig.add_trace(go.Scatter(x=active_period.index, y=active_period, name="Active", mode="lines+markers", yaxis="y2"))
fig.update_layout(
    title=f"Attribution — {view_mode} ({freq_label})",
    barmode="relative",
    xaxis_title="Period",
    yaxis_title="Contribution",
    yaxis2=dict(title="Active", overlaying="y", side="right", showgrid=False),
    height=480, margin=dict(l=20,r=20,t=60,b=20),
    legend=dict(orientation="h", y=1.08, x=0)
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Chart 2: Cumulative area (A/S/I) ----------
st.subheader("Cumulative Attribution by Component")
attr_cum = attr[["Allocation","Selection","Interaction"]].cumsum()
fig_cum = go.Figure()
for col in ["Allocation","Selection","Interaction"]:
    fig_cum.add_trace(go.Scatter(x=attr_cum.index, y=attr_cum[col], mode="lines", stackgroup="one", name=col))
fig_cum.update_layout(title="Cumulative Attribution (stacked area)", height=400, margin=dict(l=20,r=20,t=60,b=20))
st.plotly_chart(fig_cum, use_container_width=True)

# ---------- Chart 3: Scatter Allocation vs Selection (theo kỳ) ----------
st.subheader(f"Allocation vs Selection ({freq_label})")
fig_sc = go.Figure()
if not attr_period.empty:
    xs, ys = attr_period["Allocation"], attr_period["Selection"]
    labels = {
        "D":   [d.strftime("%Y-%m-%d") for d in attr_period.index],
        "ME":  [d.strftime("%Y-%m")     for d in attr_period.index],
        "QE":  [f"Q{((d.month-1)//3)+1}-{d.year}" for d in attr_period.index],
        "YE":  [str(d.year)             for d in attr_period.index],
    }[freq]
    fig_sc.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=labels, textposition="top center", name="Periods"))
fig_sc.update_layout(xaxis_title="Allocation effect", yaxis_title="Selection effect",
                     height=420, margin=dict(l=20,r=20,t=60,b=20),
                     title="Relationship between Allocation and Selection")
st.plotly_chart(fig_sc, use_container_width=True)

# ---------- Chart 4: Rolling 6M Total Attribution vs Active ----------
st.subheader("Rolling 6M Total vs Active")
roll_win = 126  # ~6 months
roll_tot = attr["Total"].rolling(roll_win).sum()
fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(x=roll_tot.index, y=roll_tot, name="Total Attribution (6M)"))
if active is not None:
    fig_roll.add_trace(go.Scatter(x=active.index, y=active.rolling(roll_win).sum(), name="Active Return (6M)"))
fig_roll.update_layout(title="Rolling 6-Month Comparison", height=400, margin=dict(l=20,r=20,t=60,b=20),
                       legend=dict(orientation="h", y=1.08, x=0))
st.plotly_chart(fig_roll, use_container_width=True)

# ---------- Bảng Top/Bottom kỳ theo Total ----------
if not attr_period.empty:
    k = 5
    period_df = attr_period.copy()
    if active_period is not None:
        period_df["Active"] = active_period.reindex(period_df.index)
    top_tbl = period_df.sort_values("Total", ascending=False).head(k)
    bot_tbl = period_df.sort_values("Total", ascending=True).head(k)
    ctop, cbot = st.columns(2)
    with ctop:
        st.subheader(f"Top {k} periods by Total")
        st.dataframe(top_tbl.style.format("{:.2%}"))
    with cbot:
        st.subheader(f"Bottom {k} periods by Total")
        st.dataframe(bot_tbl.style.format("{:.2%}"))

# ---------- Kiểm tra đối chiếu Total vs Active (Period sum) ----------
with st.expander("Kiểm tra Total vs Active (Period sum)"):
    if view_mode != "Period sum":
        st.info("Hãy chuyển sang 'Period sum' để đối chiếu chuẩn hơn (cumulative phụ thuộc điểm bắt đầu).")
    elif active_period is None or active_period.empty:
        st.warning("Chưa có chuỗi Active để đối chiếu (thiếu file hoặc dữ liệu).") 
    else:
        cmp = pd.DataFrame({
            "Total": attr_period["Total"],
            "Active": active_period.reindex(attr_period.index)
        }).dropna()
        cmp["Diff"] = cmp["Total"] - cmp["Active"]
        mae = float(cmp["Diff"].abs().mean()) if not cmp.empty else np.nan
        st.write(f"MAE(|Total − Active|) = **{mae:.6f}**  (càng gần 0 càng tốt).")
        st.dataframe(cmp.tail(12).style.format("{:.4%}"))

# ---------- Cảnh báo nếu tất cả gần 0 (trọng số & benchmark giống nhau) ----------
try:
    max_abs = float(attr_period[["Allocation","Selection","Interaction","Total"]].abs().max().max())
    if max_abs < 1e-8:
        st.info("Các giá trị Attribution gần 0. Hãy kiểm tra: danh mục có trùng benchmark không, "
                "hoặc dữ liệu weights/benchmark có bị equal-weight giống nhau không.")
except Exception:
    pass

# ---------- Download ----------
st.download_button(
    "Tải dữ liệu Attribution (bảng theo kỳ)",
    data=attr_period.to_csv(index=True),
    file_name=f"attribution_{freq_label}_{view_mode}.csv",
    mime="text/csv"
)


# =========================
# TAB 5 — Correlation & Dependence
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

P_RET   = Path("data/processed/returns_linear.parquet")
P_W     = Path("data/portfolio_weights/ew_daily_wide.parquet")   # optional, for PORT
P_BENCH = Path("data/processed/benchmark_ret.parquet")           # for regimes (SPY)

st.header("Correlation & Dependence")

# ---------- Load returns ----------
if not P_RET.exists():
    st.error(f"Missing {P_RET}. Please run your data-processing scripts first.")
    st.stop()

rets_all = pd.read_parquet(P_RET)
rets_all.index = pd.to_datetime(rets_all.index)
rets_all = rets_all.sort_index()

cols_all   = list(rets_all.columns)     # includes SPY
asset_cols = [c for c in cols_all if c != "SPY"]

# ---------- Optional: compute portfolio daily return from weights ----------
port_ret = None
if P_W.exists():
    try:
        w = pd.read_parquet(P_W)
        w.index = pd.to_datetime(w.index)
        w_prev = w.shift(1).reindex(rets_all.index).ffill().fillna(0.0)
        port_ret = (w_prev.reindex(columns=asset_cols).fillna(0.0) * rets_all[asset_cols]).sum(axis=1)
        port_ret.name = "PORT"
    except Exception as e:
        st.warning(f"Could not compute portfolio returns from weights: {e}")

# ---------- Optional: benchmark for regimes ----------
bench = None
if P_BENCH.exists():
    tmpb = pd.read_parquet(P_BENCH)
    if isinstance(tmpb, pd.Series):
        bench = tmpb.copy()
    elif isinstance(tmpb, pd.DataFrame):
        if "SPY" in tmpb.columns:
            bench = tmpb["SPY"].copy()
        elif tmpb.shape[1] == 1:
            bench = tmpb.iloc[:, 0].copy()
    if bench is not None:
        bench.index = pd.to_datetime(bench.index)
        bench = bench.sort_index()
        bench.name = "SPY"

# ---------- Controls: date, freq, method ----------
min_d, max_d = rets_all.index.min().date(), rets_all.index.max().date()
c1, c2, c3 = st.columns([2,1,1])

with c1:
    date_range = st.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        key="corr_date",
    )
with c2:
    freq_label = st.selectbox("Return frequency", ["Daily", "Monthly"], index=0)
freq_map = {"Daily": "D", "Monthly": "M"}
freq = freq_map[freq_label]

with c3:
    method = st.selectbox("Correlation method", ["Pearson", "Spearman"], index=0)

# ---------- Select series ----------
available_series = cols_all.copy()
if port_ret is not None:
    available_series = ["PORT"] + available_series

default_list = ["PORT", "SPY"] + asset_cols[:3]
default_list = [x for x in default_list if x in available_series]

selected = st.multiselect(
    "Assets / series for correlation matrix",
    options=available_series,
    default=default_list,
    help="Include PORT, SPY and any other assets.",
)
if len(selected) < 2:
    st.warning("Please select at least two series.")
    st.stop()

# ---------- Build filtered returns ----------
rets = rets_all.copy()
if port_ret is not None:
    rets = rets.assign(PORT=port_ret)

mask = (rets.index.date >= date_range[0]) & (rets.index.date <= date_range[1])
rets = rets.loc[mask, selected].copy()

def to_periodic_returns(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "D":
        return df
    def _compound(x):
        return (1 + x).prod() - 1
    out = df.resample(freq).apply(_compound)
    return out.dropna(how="all")

rets_period = to_periodic_returns(rets, freq)
if rets_period.shape[0] < 2:
    st.warning("Not enough observations for chosen range/frequency.")
    st.stop()

# ========================
# 1) Correlation matrix
# ========================
st.subheader("Correlation matrix")

corr = rets_period.corr(method.lower())
st.dataframe(corr.style.format("{:.2f}"))

fig_heat = go.Figure(
    data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        zmin=-1,
        zmax=1,
        colorscale="RdBu",
        colorbar=dict(title="ρ"),
    )
)
fig_heat.update_layout(
    title=f"{method} correlation ({freq_label} returns)",
    height=500,
    margin=dict(l=60, r=40, t=60, b=40),
)
st.plotly_chart(fig_heat, use_container_width=True)

st.download_button(
    "Download correlation matrix (CSV)",
    data=corr.to_csv(index=True),
    file_name=f"correlation_matrix_{freq_label.lower()}_{method.lower()}.csv",
    mime="text/csv",
)

st.markdown("---")

# ========================
# 2) Rolling correlation
# ========================
st.subheader("Rolling correlation")

choices_roll = selected.copy()
pair1 = st.selectbox("Series 1", choices_roll, index=0, key="roll_s1")
pair2 = st.selectbox("Series 2", choices_roll, index=min(1, len(choices_roll)-1), key="roll_s2")

if pair1 == pair2:
    st.info("Choose two different series for rolling correlation.")
else:
    win_days = st.slider("Rolling window (trading days)", 20, 252, 126, step=5)

    s1 = rets[pair1]
    s2 = rets[pair2]
    roll_corr = s1.rolling(win_days).corr(s2)

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr, mode="lines", name=f"Corr({pair1},{pair2})"))
    fig_roll.update_layout(
        title=f"Rolling correlation ({pair1} vs {pair2}, window = {win_days} days)",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1, 1]),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    st.download_button(
        "Download rolling correlation (CSV)",
        data=roll_corr.to_frame("rolling_corr").to_csv(index=True),
        file_name=f"rolling_corr_{pair1}_{pair2}_w{win_days}.csv",
        mime="text/csv",
    )

st.markdown("---")

# ========================
# 3) Scatter & regression
# ========================
st.subheader("Scatter plot & linear fit")

pair1_s = st.selectbox("X axis", selected, index=0, key="scat_s1")
pair2_s = st.selectbox("Y axis", selected, index=min(1, len(selected)-1), key="scat_s2")

if pair1_s == pair2_s:
    st.info("Choose two different series for scatter plot.")
else:
    # dùng returns đã aggregate theo freq
    x = rets_period[pair1_s]
    y = rets_period[pair2_s]
    df_xy = pd.concat([x, y], axis=1).dropna()
    xv = df_xy.iloc[:,0].values
    yv = df_xy.iloc[:,1].values

    # linear regression y = a + b x
    if len(df_xy) >= 2:
        b, a = np.polyfit(xv, yv, 1)
        y_hat = a + b * xv
        resid = yv - y_hat
        ss_tot = ((yv - yv.mean())**2).sum()
        ss_res = (resid**2).sum()
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        rho = np.corrcoef(xv, yv)[0,1]
    else:
        a = b = r2 = rho = np.nan
        y_hat = np.zeros_like(xv)

    c1s, c2s, c3s = st.columns(3)
    c1s.metric("Slope (beta)", f"{b:.3f}")
    c2s.metric("R²", f"{r2:.3f}")
    c3s.metric("ρ (Pearson, same freq)", f"{rho:.3f}")

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(x=xv, y=yv, mode="markers", name="Points"))
    fig_sc.add_trace(go.Scatter(x=xv, y=y_hat, mode="lines", name="Linear fit"))
    fig_sc.update_layout(
        title=f"{pair2_s} vs {pair1_s} ({freq_label} returns)",
        xaxis_title=f"{pair1_s} return",
        yaxis_title=f"{pair2_s} return",
        height=450,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

# ========================
# 4) Correlation by regime (SPY)
# ========================
st.markdown("---")
st.subheader("Correlation by market regime (SPY)")

if bench is None:
    st.info("Benchmark file not found; cannot classify regimes. Make sure benchmark_ret.parquet exists.")
else:
    # align SPY to daily calendar of rets
    b = bench.reindex(rets.index)

    # Rolling stats (daily)
    roll_ret = (1 + b).rolling(126).apply(np.prod, raw=True) - 1   # ~6M rolling return
    roll_vol = b.rolling(20).std()

    regimes = pd.Series("Neutral", index=b.index)
    regimes[roll_ret > 0] = "Bull"
    regimes[roll_ret < 0] = "Bear"
    hv_th = roll_vol.quantile(0.75)
    regimes[roll_vol > hv_th] = "High-vol"

    # tính correlation matrix theo từng regime với returns_period (đã aggregate)
    # map regime vào index của rets_period
    reg_period = regimes.reindex(rets_period.index)

    reg_results = {}
    for rname in ["Bull", "Bear", "High-vol"]:
        mask_r = reg_period == rname
        df_r = rets_period.loc[mask_r]
        if df_r.shape[0] >= 5:
            reg_results[rname] = df_r.corr(method.lower())

    if not reg_results:
        st.info("Not enough data in each regime with current date range / frequency.")
    else:
        # show matrices
        for rname, cmat in reg_results.items():
            st.markdown(f"**{rname} regime**  (n = {cmat.shape[0]} periods)")
            st.dataframe(cmat.style.format("{:.2f}"))

        # Example: corr(PORT, SPY) across regimes (if both exist)
        if "SPY" in rets_period.columns and ("PORT" in rets_period.columns or "PORT" in selected):
            vals = []
            labs = []
            for rname, cmat in reg_results.items():
                if "PORT" in cmat.columns and "SPY" in cmat.columns:
                    vals.append(cmat.loc["PORT", "SPY"])
                    labs.append(rname)
            if vals:
                fig_bar = go.Figure()
                fig_bar.add_bar(x=labs, y=vals)
                fig_bar.update_layout(
                    title="Correlation(PORT, SPY) by regime",
                    yaxis_title="Correlation",
                    yaxis=dict(range=[-1,1]),
                    height=350,
                    margin=dict(l=40,r=20,t=60,b=40),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
