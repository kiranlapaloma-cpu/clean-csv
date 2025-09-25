import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import io
import math

# ======================= Page config =======================
st.set_page_config(page_title="Race Edge — Manual Mode (PI v3.1 + GCI + Hidden Horses v2)", layout="wide")

# ======================= Small helpers =====================
def as_num(x):
    return pd.to_numeric(x, errors="coerce")

def color_cycle(n):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def mad_std(x):
    # robust sigma from MAD
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def winsorize(s, p_lo=0.10, p_hi=0.90):
    lo = s.quantile(p_lo)
    hi = s.quantile(p_hi)
    return s.clip(lower=lo, upper=hi)

def _lerp(a, b, t):
    return a + (b - a) * float(t)

def _interpolate_weights(dm, a_dm, a_w, b_dm, b_w):
    # linear interpolate between two anchors a_dm -> b_dm
    span = float(b_dm - a_dm)
    t = 0.0 if span <= 0 else (float(dm) - a_dm) / span
    return {
        "F200_idx": _lerp(a_w["F200_idx"], b_w["F200_idx"], t),
        "tsSPI":    _lerp(a_w["tsSPI"],    b_w["tsSPI"],    t),
        "Accel":    _lerp(a_w["Accel"],    b_w["Accel"],    t),
        "Grind":    _lerp(a_w["Grind"],    b_w["Grind"],    t),
    }

# ======================= Sidebar (Manual only) ===========================
with st.sidebar:
    st.markdown("### Manual Mode")
    race_distance_input = st.number_input("Actual Race Distance (m)", min_value=800, max_value=4000, step=50, value=1200)
    # Grid rounding rule: round UP to next 200 for entry convenience
    rounded_grid_distance = int(np.ceil(race_distance_input / 200.0) * 200)
    st.caption(
        f"Entry grid is rounded to **{rounded_grid_distance} m** (200 m splits), "
        f"but all calculations use the **actual distance = {int(race_distance_input)} m**."
    )

    n_horses = st.number_input("Number of horses", min_value=2, max_value=20, value=8, step=1)
    st.markdown("---")
    DEBUG = st.toggle("Debug info", value=False)

def _dbg(label, obj=None):
    if DEBUG:
        st.write(f"🔧 {label}")
        if obj is not None:
            st.write(obj)

# ======================= Build manual entry grid =========================
# Columns represent the END of each 200 m segment from the start, plus the final 200→Finish split.
# Example: for 1000 m grid => 800_Time, 600_Time, 400_Time, 200_Time, Finish_Time
grid_markers = list(range(rounded_grid_distance - 200, 0, -200))  # e.g. 1200 -> [1000,800,600,400,200]
cols = ["Horse", "Finish_Pos"]
for m in grid_markers:
    cols += [f"{m}_Time", f"{m}_Pos"]
cols += ["Finish_Time"]  # explicit 200→Finish segment (last 200 m)

st.markdown("### Enter segment times (seconds) for each 200 m leg; positions optional.")
template = pd.DataFrame([[None, None] + [None] * (2 * len(grid_markers)) + [None] for _ in range(n_horses)], columns=cols)
work = st.data_editor(template, num_rows="dynamic", use_container_width=True, key="manual_grid").copy()

st.success("Manual grid captured.")
st.markdown("### Raw / Converted Table")
st.dataframe(work.head(12), use_container_width=True)
_dbg("Grid markers (end-of-segment labels)", grid_markers)

# ======================= Distance + Context PI weights =======================
def pi_weights_distance_and_context(distance_m: float,
                                    acc_median: float | None,
                                    grd_median: float | None) -> dict:
    """
    Distance logic:
      - 1000m anchor: F200=0.12, tsSPI=0.35, Accel=0.36, Grind=0.17
      - 1100m anchor: F200=0.10, tsSPI=0.36, Accel=0.34, Grind=0.20
      - 1200m anchor: F200=0.08, tsSPI=0.37, Accel=0.30, Grind=0.25
      - >1200m: shift 0.01 per +100m from tsSPI → Grind, cap Grind at 0.40
                (F200 stays 0.08, Accel stays 0.30; tsSPI = 1 - F200 - Accel - Grind)

    Context nudge (tiny, ±0.02 max total) based on race median Accel vs Grind.
    """
    dm = float(distance_m or 1200)

    # ---- distance base weights ----
    if dm <= 1000:
        base = {"F200_idx":0.12, "tsSPI":0.35, "Accel":0.36, "Grind":0.17}
    elif dm < 1100:
        base = _interpolate_weights(
            dm,
            1000, {"F200_idx":0.12, "tsSPI":0.35, "Accel":0.36, "Grind":0.17},
            1100, {"F200_idx":0.10, "tsSPI":0.36, "Accel":0.34, "Grind":0.20}
        )
    elif dm < 1200:
        base = _interpolate_weights(
            dm,
            1100, {"F200_idx":0.10, "tsSPI":0.36, "Accel":0.34, "Grind":0.20},
            1200, {"F200_idx":0.08, "tsSPI":0.37, "Accel":0.30, "Grind":0.25}
        )
    elif dm == 1200:
        base = {"F200_idx":0.08, "tsSPI":0.37, "Accel":0.30, "Grind":0.25}
    else:
        # shift from tsSPI to Grind, +0.01 per +100m, cap Grind at 0.40
        shift_units = max(0.0, (dm - 1200.0) / 100.0) * 0.01
        grind = min(0.25 + shift_units, 0.40)
        F200  = 0.08
        ACC   = 0.30
        ts    = max(0.0, 1.0 - F200 - ACC - grind)
        base  = {"F200_idx":F200, "tsSPI":ts, "Accel":ACC, "Grind":grind}

    # ---- context nudge (very small) ----
    acc_med = float(acc_median) if acc_median is not None else None
    grd_med = float(grd_median) if grd_median is not None else None

    if acc_med is not None and grd_med is not None and math.isfinite(acc_med) and math.isfinite(grd_med):
        bias = acc_med - grd_med  # +ve = kick-leaning, −ve = grind-leaning
        scale = math.tanh(abs(bias) / 6.0)
        max_shift = 0.02 * scale

        F200 = base["F200_idx"]; ts = base["tsSPI"]; ACC = base["Accel"]; GR = base["Grind"]

        if bias > 0:
            delta = min(max_shift, ACC - 0.26)  # floor on Accel
            ACC -= delta; GR += delta
        elif bias < 0:
            delta = min(max_shift, GR - 0.18)   # floor on Grind
            GR  -= delta; ACC += delta

        GR = min(GR, 0.40)
        ts = max(0.0, 1.0 - F200 - ACC - GR)
        base = {"F200_idx":F200, "tsSPI":ts, "Accel":ACC, "Grind":GR}

    s = sum(base.values())
    if abs(s - 1.0) > 1e-6:
        base = {k: v / s for k, v in base.items()}
    return base

# ======================= Core metric build (Manual grid) =================
def build_metrics_manual(df_in: pd.DataFrame, actual_distance_m: float, grid_markers: list[int]):
    """
    df_in has columns:
      Horse, Finish_Pos, <m>_Time, <m>_Pos for m in grid_markers (e.g., 1000,800,...,200), and Finish_Time.
    Internally we add a synthetic '0_Time' for Finish_Time so 'last 200' is measured correctly.
    """
    w = df_in.copy()

    # numeric finish pos (if present)
    if "Finish_Pos" in w.columns:
        w["Finish_Pos"] = as_num(w["Finish_Pos"])

    # build seg markers (include 0 as 'Finish')
    seg_markers = sorted(set([int(m) for m in grid_markers] + [0]), reverse=True)  # e.g., [1000,800,600,400,200,0]

    # make internal 0_Time from Finish_Time (200→Finish) for uniform handling
    if "Finish_Time" in w.columns:
        w["0_Time"] = as_num(w["Finish_Time"])
    else:
        w["0_Time"] = np.nan

    # per-segment speeds (all 200 m legs, including the final 200→Finish)
    for m in seg_markers:
        col = f"{m}_Time" if m != 0 else "0_Time"
        w[f"spd_{m}"] = 200.0 / as_num(w[col])

    # race time = sum of all 200 m splits including the finish split
    sum_cols = [f"{m}_Time" for m in grid_markers if f"{m}_Time" in w.columns]
    sum_cols.append("0_Time")
    w["RaceTime_s"] = w[sum_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # ---------- small-field stabilizers ----------
    def shrink_center(idx_series):
        x = idx_series.dropna().values
        N_eff = len(x)
        if N_eff == 0:
            return 100.0, 0
        med_race = float(np.median(x))
        alpha = N_eff / (N_eff + 6.0)    # gentle pull to 100 when N is small
        return alpha * med_race + (1 - alpha) * 100.0, N_eff

    def dispersion_equalizer(delta_series, N_eff, N_ref=10, beta=0.20, cap=1.20):
        gamma = 1.0 + beta * max(0, N_ref - N_eff) / N_ref
        return delta_series * min(gamma, cap)

    def variance_floor(idx_series, floor=1.5, cap=1.25):
        deltas = idx_series - 100.0
        sigma = mad_std(deltas)
        if not np.isfinite(sigma) or sigma <= 0:
            return idx_series
        if sigma < floor:
            factor = min(cap, floor / sigma)
            return 100.0 + deltas * factor
        return idx_series

    # ---------- F200 = first 200 m from the start ----------
    first_mark = max(seg_markers) if seg_markers else None  # e.g., 1000 for a 1200 grid
    if first_mark and f"spd_{first_mark}" in w.columns:
        base = w[f"spd_{first_mark}"]
        prelim = 100.0 * (base / base.median(skipna=True))
        center, n_eff = shrink_center(prelim)
        f200 = 100.0 * (base / (center / 100.0 * base.median(skipna=True)))
        f200 = 100.0 + dispersion_equalizer(f200 - 100.0, n_eff)
        f200 = variance_floor(f200)
        w["F200_idx"] = f200
    else:
        w["F200_idx"] = np.nan

    # ---------- tsSPI = sustained mid race (exclude first 200 & last 600) ----------
    # With explicit 0 marker, last 600 means markers [600,400,200,0] => drop last three 200s + finish.
    def tsspi_avg(row):
        mids = [m for m in seg_markers if m not in (first_mark, 400, 200, 0)]
        # adaptive fallback for very short races
        if len(mids) < 2:
            mids = [m for m in seg_markers if m not in (first_mark, 200, 0)]
        if len(mids) < 1:
            mids = [m for m in seg_markers if m not in (first_mark, 0)]
        vals = [row.get(f"spd_{m}") for m in mids]
        vals = [v for v in vals if pd.notna(v)]
        return np.nan if not vals else float(np.mean(vals))

    w["_mid_spd"] = w.apply(tsspi_avg, axis=1)
    mid_med = w["_mid_spd"].median(skipna=True)
    w["tsSPI_raw"] = 100.0 * (w["_mid_spd"] / mid_med)
    center_ts, n_ts = shrink_center(w["tsSPI_raw"])
    tsSPI = 100.0 * (w["_mid_spd"] / (center_ts / 100.0 * mid_med))
    tsSPI = 100.0 + dispersion_equalizer(tsSPI - 100.0, n_ts)
    tsSPI = variance_floor(tsSPI)
    w["tsSPI"] = tsSPI

    # ---------- Accel = 600→200 (adaptive up to 3 segments before 200) ----------
    pre_marks = [m for m in seg_markers if m > 0]  # everything before finish
    # typical window: [600, 400, 200] if present
    accel_candidates = [m for m in pre_marks if m <= 600 and m >= 200]
    accel_candidates = sorted(set(accel_candidates), reverse=True)  # [600,400,200]
    if not accel_candidates:
        # fallback: use up to the last 2 segments before finish
        accel_candidates = sorted(pre_marks)[-2:] if len(pre_marks) >= 2 else pre_marks[-1:]
        accel_candidates = list(reversed(accel_candidates))  # keep late-first ordering

    def mean_marks(row, marks):
        vals = [row.get(f"spd_{m}") for m in marks]
        vals = [v for v in vals if pd.notna(v)]
        return np.nan if not vals else float(np.mean(vals))

    w["_accel_spd"] = w.apply(lambda r: mean_marks(r, accel_candidates), axis=1)
    a_med = w["_accel_spd"].median(skipna=True)
    w["Accel_raw"] = 100.0 * (w["_accel_spd"] / a_med)
    center_a, n_a = shrink_center(w["Accel_raw"])
    Accel = 100.0 * (w["_accel_spd"] / (center_a / 100.0 * a_med))
    Accel = 100.0 + dispersion_equalizer(Accel - 100.0, n_a)
    Accel = variance_floor(Accel)
    w["Accel"] = Accel

    # ---------- Grind = last 200 (200→Finish) ----------
    if "spd_0" in w.columns:
        g_base = w["spd_0"]
        g_med = g_base.median(skipna=True)
        w["Grind_raw"] = 100.0 * (g_base / g_med)
        center_g, n_g = shrink_center(w["Grind_raw"])
        Grind = 100.0 * (g_base / (center_g / 100.0 * g_med))
        Grind = 100.0 + dispersion_equalizer(Grind - 100.0, n_g)
        Grind = variance_floor(Grind)
        w["Grind"] = Grind
    else:
        w["Grind"] = np.nan

    # ---------- PI v3.1 (distance- & context-aware weights, robust 0–10 scaling) ----------
    acc_med = w["Accel"].median(skipna=True)
    grd_med = w["Grind"].median(skipna=True)
    PI_W = pi_weights_distance_and_context(float(actual_distance_m), acc_med, grd_med)

    def pi_pts_row(row):
        parts, weights = [], []
        for k, wgt in PI_W.items():
            v = row.get(k, np.nan)
            if pd.notna(v):
                parts.append(wgt * (v - 100.0))
                weights.append(wgt)
        if not weights:
            return np.nan
        return sum(parts) / sum(weights)
    w["PI_pts"] = w.apply(pi_pts_row, axis=1)

    pts = pd.to_numeric(w["PI_pts"], errors="coerce")
    med = float(np.nanmedian(pts)) if np.isfinite(np.nanmedian(pts)) else 0.0
    centered = pts - med
    sigma = mad_std(centered)
    if not np.isfinite(sigma) or sigma < 0.75:
        sigma = 0.75
    w["PI"] = (5.0 + 2.2 * (centered / sigma)).clip(0.0, 10.0).round(2)

    # ---------- GCI (0–10) — aligned with same worldview ----------
    acc_med_g = w["Accel"].median(skipna=True)
    grd_med_g = w["Grind"].median(skipna=True)
    Wg = pi_weights_distance_and_context(float(actual_distance_m), acc_med_g, grd_med_g)

    wT   = 0.25
    wPACE= Wg["Accel"] + Wg["Grind"]
    wSS  = Wg["tsSPI"]
    wEFF = max(0.0, 1.0 - (wT + wPACE + wSS))

    winner_time = None
    if "RaceTime_s" in w.columns and w["RaceTime_s"].notna().any():
        try:
            winner_time = float(w["RaceTime_s"].min())
        except Exception:
            winner_time = None

    def map_pct(x, lo=98.0, hi=104.0):
        if pd.isna(x): return 0.0
        return clamp((float(x) - lo) / (hi - lo), 0.0, 1.0)

    gci_vals = []
    for _, r in w.iterrows():
        T = 0.0
        if winner_time is not None and pd.notna(r.get("RaceTime_s")):
            d = float(r["RaceTime_s"]) - winner_time
            if d <= 0.30:   T = 1.0
            elif d <= 0.60: T = 0.7
            elif d <= 1.00: T = 0.4
            else:           T = 0.2

        LQ = 0.6 * map_pct(r.get("Accel")) + 0.4 * map_pct(r.get("Grind"))
        SS = map_pct(r.get("tsSPI"))

        acc, grd = r.get("Accel"), r.get("Grind")
        if pd.isna(acc) or pd.isna(grd):
            EFF = 0.0
        else:
            dev = (abs(acc - 100.0) + abs(grd - 100.0)) / 2.0
            EFF = clamp(1.0 - dev / 8.0, 0.0, 1.0)

        score01 = (wT * T) + (wPACE * LQ) + (wSS * SS) + (wEFF * EFF)
        gci_vals.append(round(10.0 * score01, 3))

    w["GCI"] = gci_vals

    # tidy rounding
    for c in ["F200_idx", "tsSPI", "Accel", "Grind", "PI", "GCI", "RaceTime_s"]:
        if c in w.columns:
            w[c] = w[c].round(3)

    return w, seg_markers

# ---- compute metrics
try:
    metrics, seg_markers = build_metrics_manual(work, float(race_distance_input), grid_markers)
except Exception as e:
    st.error("Metric computation failed.")
    if DEBUG:
        st.exception(e)
    st.stop()

# ======================= Metrics table =====================
st.markdown("## Sectional Metrics (Manual PI v3.1 & GCI)")
show_cols = ["Horse", "Finish_Pos", "RaceTime_s", "F200_idx", "tsSPI", "Accel", "Grind", "PI", "GCI"]
for c in show_cols:
    if c not in metrics.columns:
        metrics[c] = np.nan
st.dataframe(metrics[show_cols].sort_values(["PI","Finish_Pos"], ascending=[False, True]),
             use_container_width=True)

# ===================== Sectional Shape Map — Accel vs Grind =====================
st.markdown("## Sectional Shape Map — Accel (600→200) vs Grind (200→Finish)")

needed_cols = {"Horse", "Accel", "Grind", "tsSPI", "PI"}
if not needed_cols.issubset(metrics.columns):
    st.warning("Shape Map: required columns missing: " + ", ".join(sorted(needed_cols - set(metrics.columns))))
else:
    dfm = metrics.loc[:, ["Horse", "Accel", "Grind", "tsSPI", "PI"]].copy()
    for c in ["Accel", "Grind", "tsSPI", "PI"]:
        dfm[c] = pd.to_numeric(dfm[c], errors="coerce")
    dfm = dfm.dropna(subset=["Accel", "Grind", "tsSPI"])

    if dfm.empty:
        st.info("Not enough data to draw the shape map.")
    else:
        dfm["AccelΔ"] = dfm["Accel"] - 100.0
        dfm["GrindΔ"] = dfm["Grind"] - 100.0
        dfm["tsSPIΔ"] = dfm["tsSPI"] - 100.0

        names = dfm["Horse"].astype(str).to_list()
        xv = dfm["AccelΔ"].to_numpy()
        yv = dfm["GrindΔ"].to_numpy()
        cv = dfm["tsSPIΔ"].to_numpy()
        piv = dfm["PI"].fillna(0).to_numpy()

        if not np.isfinite(xv).any() or not np.isfinite(yv).any():
            st.info("No valid sectional differentials available for this race.")
        else:
            try:
                span = float(np.nanmax([np.nanmax(np.abs(xv)), np.nanmax(np.abs(yv))]))
            except Exception:
                span = 1.0
            if not np.isfinite(span) or span <= 0:
                span = 1.0
            lim = max(4.5, float(np.ceil(span / 1.5) * 1.5))

            DOT_MIN, DOT_MAX = 40.0, 140.0
            pmin, pmax = float(np.nanmin(piv)), float(np.nanmax(piv))
            if not np.isfinite(pmin) or not np.isfinite(pmax) or abs(pmax - pmin) < 1e-9:
                sizes = np.full_like(xv, DOT_MIN)
            else:
                sizes = DOT_MIN + (piv - pmin) / (pmax - pmin) * (DOT_MAX - DOT_MIN)

            fig, ax = plt.subplots(figsize=(7.6, 6.4))

            TINT = 0.06
            ax.add_patch(Rectangle((0, 0),  lim,  lim, facecolor="#4daf4a", alpha=TINT, edgecolor="none"))
            ax.add_patch(Rectangle((-lim,0), lim,  lim, facecolor="#377eb8", alpha=TINT, edgecolor="none"))
            ax.add_patch(Rectangle((0,-lim), lim, lim, facecolor="#ff7f00", alpha=TINT, edgecolor="none"))
            ax.add_patch(Rectangle((-lim,-lim),lim, lim, facecolor="#984ea3", alpha=TINT, edgecolor="none"))

            ax.axvline(0, color="gray", lw=1.3, ls=(0, (3, 3)))
            ax.axhline(0, color="gray", lw=1.3, ls=(0, (3, 3)))

            vmin = float(np.nanmin(cv)) if np.isfinite(cv).any() else -1.0
            vmax = float(np.nanmax(cv)) if np.isfinite(cv).any() else  1.0
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = -1.0, 1.0
            norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

            sc = ax.scatter(
                xv, yv, s=sizes, c=cv, cmap="coolwarm", norm=norm,
                edgecolor="black", linewidth=0.6, alpha=0.95
            )

            # Simple non-overlapping label helper
            def label_points_neatly(ax, x, y, names):
                try:
                    from adjustText import adjust_text
                    texts = []
                    for xi, yi, nm in zip(x, y, names):
                        texts.append(ax.text(xi, yi, nm, fontsize=8.6,
                                             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.70)))
                    adjust_text(
                        texts, x=x, y=y, ax=ax,
                        only_move={'points': 'y', 'text': 'xy'},
                        force_points=0.6, force_text=0.7,
                        expand_text=(1.05, 1.15), expand_points=(1.05, 1.15),
                        arrowprops=dict(arrowstyle="->", lw=0.75, color="black", alpha=0.9, shrinkA=0, shrinkB=3)
                    )
                except Exception:
                    for xi, yi, nm in zip(x, y, names):
                        ax.text(xi+0.2, yi+0.2, nm, fontsize=8.6,
                                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.70))

            label_points_neatly(ax, xv, yv, names)

            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_xlabel("Acceleration vs field (points)  →")
            ax.set_ylabel("Grind vs field (points)  ↑")
            ax.set_title("Quadrants: +X = late acceleration; +Y = strong last 200. Colour = tsSPI deviation")

            s_ex = [DOT_MIN, 0.5*(DOT_MIN+DOT_MAX), DOT_MAX]
            h_ex = [Line2D([0],[0], marker='o', color='w', markerfacecolor='gray',
                           markersize=np.sqrt(s/np.pi), markeredgecolor='black') for s in s_ex]
            ax.legend(h_ex, ["PI: low", "PI: mid", "PI: high"],
                      loc="upper left", frameon=False, fontsize=8)

            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("tsSPI − 100")

            ax.grid(True, linestyle=":", alpha=0.25)
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            st.download_button("Download shape map (PNG)", buf.getvalue(),
                               file_name="shape_map.png", mime="image/png")

            st.caption(
                "Each bubble is a runner. Size = PI (bigger = stronger overall). "
                "X: late acceleration (600→200) vs field; Y: last-200 grind vs field. "
                "Colour shows cruise strength (tsSPI vs field): red = faster mid-race, blue = slower."
            )

# ======================= Pace Curve — includes Finish split =================
st.markdown("## Pace Curve — field average (black) + Top 8 finishers (Finish split included)")

# Ensure we have a synthetic 0_Time from Finish_Time for the last 200 → Finish
work["0_Time"] = as_num(work.get("Finish_Time", np.nan))

# Build ordered markers including 0 for finish, e.g. [1000, 800, 600, 400, 200, 0]
markers = sorted(set(grid_markers + [0]), reverse=True)

if len(markers) < 2:
    st.info("Not enough segments to draw the pace curve.")
else:
    # Build (start, end, length) pairs; last pair must be (200, 0, 200)
    segs = []
    for i, start in enumerate(markers[:-1]):
        end = markers[i + 1]
        seg_len = float(start - end) if end > 0 else 200.0  # force last split to be 200→Finish
        if seg_len > 0:
            segs.append((start, end, seg_len))

    if not segs:
        st.info("Could not infer segment lengths.")
    else:
        # Assemble times and speeds for each segment, including 0_Time for finish
        times_df = pd.DataFrame(index=work.index)
        for (start, end, seg_len) in segs:
            col = f"{start}_Time" if end > 0 else "0_Time"
            times_df[col] = as_num(work.get(col, np.nan))

        speed_df = pd.DataFrame(index=work.index)
        for (start, end, seg_len) in segs:
            col = f"{start}_Time" if end > 0 else "0_Time"
            speed_df[col] = seg_len / times_df[col]

        # Field average
        field_avg = speed_df.mean(axis=0).to_numpy()

        # Choose top 8: finish pos if present, else PI
        if "Finish_Pos" in metrics.columns and metrics["Finish_Pos"].notna().any():
            top8 = metrics.sort_values("Finish_Pos").head(8)
        else:
            top8 = metrics.sort_values("PI", ascending=False).head(8)

        # X axis labels (show "200–Finish" for the last)
        def seg_label(s, e):
            return "200–Finish" if e == 0 else f"{int(s)}–{int(e)}m"
        x_labels = [seg_label(s, e) for (s, e, _) in segs]
        x_idx = list(range(len(segs)))

        fig2, ax2 = plt.subplots(figsize=(8.6, 5.2))
        # Field average — thicker black
        ax2.plot(x_idx, field_avg, linewidth=2.2, color="black", label="Field average", marker=None)

        # Overlay top 8 — thin lines & small markers
        palette = color_cycle(len(top8))
        for i, (_, r) in enumerate(top8.iterrows()):
            # Build a row-wise time vector that includes 0_Time for finish
            row_times = {}
            for (start, end, seg_len) in segs:
                col = f"{start}_Time" if end > 0 else "0_Time"
                # Try to read from the original work row for this Horse (if names match), fallback to metrics
                if "Horse" in work.columns and "Horse" in metrics.columns:
                    wrow = work[work["Horse"] == r.get("Horse")]
                    if not wrow.empty and col in wrow.columns:
                        row_times[col] = as_num(wrow.iloc[0].get(col))
                    else:
                        row_times[col] = as_num(r.get(col, np.nan))
                else:
                    row_times[col] = as_num(r.get(col, np.nan))

            y_vals = []
            for (start, end, seg_len) in segs:
                col = f"{start}_Time" if end > 0 else "0_Time"
                t = pd.to_numeric(row_times.get(col, np.nan), errors="coerce")
                y_vals.append(seg_len / t if pd.notna(t) and t > 0 else np.nan)

            ax2.plot(x_idx, y_vals, linewidth=1.1, marker="o", markersize=2.5,
                     label=str(r.get("Horse", "")), color=palette[i])

        ax2.set_xticks(x_idx)
        ax2.set_xticklabels(x_labels, rotation=45, ha="right")
        ax2.set_ylabel("Speed (m/s)")
        ax2.set_title("Pace over 200 m segments (left = early, right = home straight) — includes 200→Finish")
        ax2.grid(True, linestyle="--", alpha=0.30)
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=9)
        st.pyplot(fig2)

# ======================= Top-8 PI — stacked contributions =================
st.markdown("## Top-8 PI — stacked contributions")

acc_med_for_bars = metrics["Accel"].median(skipna=True)
grd_med_for_bars = metrics["Grind"].median(skipna=True)
PI_W_BARS = pi_weights_distance_and_context(float(race_distance_input), acc_med_for_bars, grd_med_for_bars)

def parts_scaled_to_total(row, total_pi, weights, zero_floor=True):
    raw = {
        "F200_idx": weights["F200_idx"] * (float(row.get("F200_idx", 100.0)) - 100.0),
        "tsSPI":    weights["tsSPI"]    * (float(row.get("tsSPI",    100.0)) - 100.0),
        "Accel":    weights["Accel"]    * (float(row.get("Accel",    100.0)) - 100.0),
        "Grind":    weights["Grind"]    * (float(row.get("Grind",    100.0)) - 100.0),
    }
    if zero_floor:
        raw = {k: max(0.0, v) for k, v in raw.items()}
    s = sum(raw.values())
    if not np.isfinite(total_pi) or total_pi <= 0 or not np.isfinite(s) or s <= 0:
        return {"F200_idx": 0.0, "tsSPI": 0.0, "Accel": 0.0, "Grind": 0.0}
    scale = float(total_pi) / float(s)
    return {k: v * scale for k, v in raw.items()}

top8_pi = metrics.sort_values(["PI","Finish_Pos"], ascending=[False, True]).head(8).copy()
if not top8_pi.empty:
    horses = []
    stacks = {"F200_idx": [], "tsSPI": [], "Accel": [], "Grind": []}
    totals = []
    is_winner = []

    for _, r in top8_pi.iterrows():
        total_pi = float(r.get("PI", 0.0))
        parts = parts_scaled_to_total(r, total_pi, PI_W_BARS, zero_floor=True)
        for k in stacks:
            stacks[k].append(parts[k])
        totals.append(total_pi)
        horses.append(str(r.get("Horse", "")))
        is_winner.append(int(r.get("Finish_Pos", 0)) == 1)

    fig3, ax3 = plt.subplots(figsize=(max(7.5, 0.95*len(horses)), 4.8))
    x = np.arange(len(horses))
    palette = {"F200_idx": "#6baed6", "tsSPI": "#9e9ac8", "Accel": "#74c476", "Grind": "#fd8d3c"}

    bottoms = np.zeros(len(horses))
    for key, label in [("F200_idx","F200"), ("tsSPI","tsSPI"), ("Accel","Accel"), ("Grind","Grind")]:
        vals = np.array(stacks[key], dtype=float)
        ax3.bar(x, vals, bottom=bottoms, label=label, color=palette[key], edgecolor="black", linewidth=0.4)
        bottoms += vals

    ymax = max(0.1, max(totals)*1.20)
    for i, tot in enumerate(totals):
        if is_winner[i]:
            ax3.add_patch(plt.Rectangle((i-0.5, 0), 1.0, max(tot, bottoms[i]), fill=False, lw=2.0, ec="#d4af37"))
            horses[i] = f"★ {horses[i]}"
        ax3.text(i, tot + ymax*0.03, f"{tot:.2f}", ha="center", va="bottom", fontsize=9)

    ax3.set_xticks(x); ax3.set_xticklabels(horses, rotation=45, ha="right")
    ax3.set_ylim(0, ymax)
    ax3.set_ylabel("PI (stacked contributions)")
    ax3.grid(axis="y", linestyle="--", alpha=0.3)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    st.pyplot(fig3)
    st.caption("Slices rescaled to sum exactly to each horse’s PI. ★ = race winner.")
else:
    st.info("No PI values available to plot the stacked contributions.")

# ======================= Hidden Horses (v2) =================
st.markdown("## Hidden Horses (v2)")

hh = metrics.copy()

need_cols = {"tsSPI", "Accel", "Grind"}
if need_cols.issubset(hh.columns) and len(hh) > 0:
    ts_w = winsorize(pd.to_numeric(hh["tsSPI"], errors="coerce"))
    ac_w = winsorize(pd.to_numeric(hh["Accel"], errors="coerce"))
    gr_w = winsorize(pd.to_numeric(hh["Grind"], errors="coerce"))

    def rz(s: pd.Series) -> pd.Series:
        mu = np.nanmedian(s)
        sd = mad_std(s)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mu) / sd

    z_ts = rz(ts_w).clip(-2.5, 3.5)
    z_ac = rz(ac_w).clip(-2.5, 3.5)
    z_gr = rz(gr_w).clip(-2.5, 3.5)

    hh["SOS_raw"] = 0.45 * z_ts + 0.35 * z_ac + 0.20 * z_gr

    q5, q95 = hh["SOS_raw"].quantile(0.05), hh["SOS_raw"].quantile(0.95)
    denom = (q95 - q5) if (pd.notna(q95) and pd.notna(q5) and (q95 > q5)) else 1.0
    hh["SOS"] = (2.0 * (hh["SOS_raw"] - q5) / denom).clip(lower=0.0, upper=2.0)
else:
    hh["SOS"] = 0.0

acc_med = pd.to_numeric(hh.get("Accel"), errors="coerce").median(skipna=True)
grd_med = pd.to_numeric(hh.get("Grind"), errors="coerce").median(skipna=True)
bias = (acc_med - 100.0) - (grd_med - 100.0)
B = min(1.0, abs(bias) / 4.0)
S = pd.to_numeric(hh.get("Accel"), errors="coerce") - pd.to_numeric(hh.get("Grind"), errors="coerce")
if bias >= 0:
    hh["ASI2"] = (B * (-S).clip(lower=0.0) / 5.0).fillna(0.0)
else:
    hh["ASI2"] = (B * (S).clip(lower=0.0) / 5.0).fillna(0.0)

# Trip friction: late variability over last 3 segments vs mid pace
last3 = []
seg_markers_for_tfs = sorted(set(seg_markers), reverse=False)  # [0, 200, 400, ...]
near_finish = [m for m in seg_markers_for_tfs if m in (600, 400, 200)]
last3 = sorted(near_finish, reverse=True)

def tfs_row(r):
    spds = [r.get(f"spd_{m}") for m in last3]
    spds = [s for s in spds if pd.notna(s)]
    if len(spds) < 2:
        return np.nan
    sigma = float(np.std(spds, ddof=0))
    mid = float(r.get("_mid_spd", np.nan))
    if not np.isfinite(mid) or mid <= 0:
        return np.nan
    return 100.0 * (sigma / mid)

hh["TFS"] = hh.apply(tfs_row, axis=1)

# Distance-aware TFS gate
rounded_distance_for_gate = int(np.ceil(float(race_distance_input) / 200.0) * 200)
if rounded_distance_for_gate <= 1200:
    gate = 4.0
elif rounded_distance_for_gate < 1800:
    gate = 3.5
else:
    gate = 3.0

def tfs_plus(x):
    if pd.isna(x) or x < gate:
        return 0.0
    return min(0.6, (x - gate) / 3.0)

hh["TFS_plus"] = hh["TFS"].apply(tfs_plus)

def uei_row(r):
    ts = pd.to_numeric(r.get("tsSPI"), errors="coerce")
    ac = pd.to_numeric(r.get("Accel"), errors="coerce")
    gr = pd.to_numeric(r.get("Grind"), errors="coerce")
    if pd.isna(ts) or pd.isna(ac) or pd.isna(gr):
        return 0.0
    val = 0.0
    if ts >= 102 and ac <= 98 and gr <= 98:
        gap = min((ts - 102) / 3.0, 1.0)
        val = max(val, 0.3 + 0.3 * gap)
    if ts >= 102 and gr >= 102 and ac <= 100:
        gap = min(((ts - 102) + (gr - 102)) / 6.0, 1.0)
        val = max(val, 0.3 + 0.3 * gap)
    return round(val, 3)

hh["UEI"] = hh.apply(uei_row, axis=1)

hidden = (
    0.55 * pd.to_numeric(hh["SOS"], errors="coerce").fillna(0.0) +
    0.30 * pd.to_numeric(hh["ASI2"], errors="coerce").fillna(0.0) +
    0.10 * pd.to_numeric(hh["TFS_plus"], errors="coerce").fillna(0.0) +
    0.05 * pd.to_numeric(hh["UEI"], errors="coerce").fillna(0.0)
)

if int(hh.shape[0]) <= 6:
    hidden = hidden * 0.90

h_med = float(np.nanmedian(hidden))
h_mad = float(np.nanmedian(np.abs(hidden - h_med)))
h_sigma = max(1e-6, 1.4826 * h_mad)

hh["HiddenScore"] = (1.2 + (hidden - h_med) / (2.5 * h_sigma)).clip(lower=0.0, upper=3.0)

def hh_tier(s):
    if pd.isna(s): return ""
    if s >= 1.8:   return "🔥 Top Hidden"
    if s >= 1.2:   return "🟡 Notable Hidden"
    return ""

hh["Tier"] = hh["HiddenScore"].apply(hh_tier)

def hh_note(r):
    bits = []
    if r.get("Tier", "") != "":
        if pd.to_numeric(r.get("SOS"), errors="coerce") >= 1.2:
            bits.append("sectionals superior")
        asi2 = pd.to_numeric(r.get("ASI2"), errors="coerce")
        if asi2 >= 0.8:
            bits.append("ran against strong bias")
        elif asi2 >= 0.4:
            bits.append("ran against bias")
        if pd.to_numeric(r.get("TFS_plus"), errors="coerce") > 0:
            bits.append("trip friction late")
        if pd.to_numeric(r.get("UEI"), errors="coerce") >= 0.5:
            bits.append("latent potential if shape flips")
    return ("; ".join(bits).capitalize() + ".") if bits else ""

hh["Note"] = hh.apply(hh_note, axis=1)

cols_hh = [
    "Horse", "Finish_Pos", "PI", "GCI",
    "tsSPI", "Accel", "Grind",
    "SOS", "ASI2", "TFS", "UEI",
    "HiddenScore", "Tier", "Note"
]
for c in cols_hh:
    if c not in hh.columns:
        hh[c] = np.nan

st.dataframe(
    hh.sort_values(["Tier", "HiddenScore", "PI"], ascending=[True, False, False])[cols_hh],
    use_container_width=True
)

st.caption(
    "Manual Mode: Enter 200 m split times ending with **Finish_Time** (200→Finish). "
    "F200_idx = first 200 m; tsSPI excludes first 200 & last 600; Accel = 600→200; Grind = last 200 (finish). "
    "PI v3.1 uses distance- & context-aware weights; GCI is aligned; Hidden Horses v2 highlights sectional standouts."
)
