"""
spring_lumped_final.py  —  Foux Karst System
=============================================
Foux spring discharge prediction using the lumped siphon model extracted
from lumped_v3test.py.  Only the spring model is kept here.

Model structure (Epikarst–Matrix–Conduit + Siphon)
  Input   : net water [m³/h] = P − ET, pre-computed per basin
            (from foux_total_water.csv, column Foux_Basin)
  Chain   : Epikarst (E) → Matrix (M) + Conduit (C) → Siphon (S) → spring
  Siphon  : threshold physics — flow only when S > S_crest (explains 97% zeros)
  Output  : Foux spring discharge [L/s], 6-h lead prediction

Calibration: warm-start L-BFGS-B + Differential Evolution (hybrid)
  k-parameters calibrated on log₁₀ scale (span 4+ orders of magnitude).
  Warm seeds anchored to recession analysis: k_sip ≈ 0.025 h⁻¹.

Outputs (saved to final/results/)
  spring_lumped_final_pred.csv
  spring_lumped_final_metrics.csv
  spring_lumped_final.png     ← time-series + scatter in one image
"""

import warnings; warnings.filterwarnings("ignore")
import os, math, gc
import numpy as np
import pandas as pd
from numba import jit
from scipy.optimize import differential_evolution, minimize as _lbfgsb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

PROJECT_DIR = r"D:\foux\Foux\final"
DATA_CSV    = os.path.join(PROJECT_DIR, "foux_total_water.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DT         = 1        # timestep [h]
PRED_LAG   = 6        # prediction horizon [h-ahead]
TRAIN_FRAC = 0.70
EPS_Q      = 0.01     # log-transform offset (avoids log(0))

# Calibration settings (increase for production runs)
DE_MAXITER = 400
DE_POPSIZE = 15
LBFGS_ITER = 1000

# Warmup: short (30 days) because most spring events are in 2019+.
# A long warmup would discard the rare active periods.
WARMUP_SP = 24 * 30   # 30 days

PLOT_START = "2024-02-01"
PLOT_END   = "2024-04-30 23:00"

# Plot aesthetics
BG   = "#FFFFFF"; GRID = "#E2E8F0"; TXT = "#1A202C"; LBL = "#4A5568"
OBS  = "#E53E3E"; SP_C = "#2B6CB0"
ABOX = dict(boxstyle="round,pad=0.35", facecolor="#EBF8FF",
            edgecolor="#3182CE", alpha=0.92)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
})


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════
# foux_total_water.csv contains net water (P − ET) per basin [m³/h].
# The spring basin is the Foux sub-basin (HRUs 5 and 6 combined).

print("[spring_lumped] Loading data …", flush=True)
df = pd.read_csv(DATA_CSV, parse_dates=["date"], index_col="date",
                 dayfirst=False).sort_index()
df.columns = ["HRU_4(m3/h)" if c == "(m3/h)" else c for c in df.columns]

# Net water for the Foux spring basin [m³/h] — can be negative during drought
net_sp = df["Foux_Basin(m3/h)"].values.astype(np.float64)

# Observed spring discharge [L/s] → convert to [m³/h] (1 L/s = 3.6 m³/h)
Q_foux_obs = df["Foux_Charge(l/s)"].values.astype(np.float64) * 3.6

T_total = len(df)
NTR     = int(T_total * TRAIN_FRAC)
print(f"[spring_lumped] T={T_total}  Train={NTR}  Test={T_total-NTR}", flush=True)
print(f"[spring_lumped] Spring max = {Q_foux_obs.max():.1f} m³/h  "
      f"({Q_foux_obs.max()/3.6:.0f} L/s)", flush=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — PHYSICS MODEL (Numba JIT)
# ══════════════════════════════════════════════════════════════════════
# Numba compiles the inner loop to machine code on first call (~3 s),
# then each DE evaluation runs ~100× faster than pure Python.

@jit(nopython=True, cache=True)
def run_model_spring(net_water, k_EM, k_EC, k_MC, k_MS, k_CS,
                     E_loss, S_crest, k_sip, leak_sp):
    """
    SPRING MODEL — Epikarst → Matrix + Conduit → Siphon.

    ── EPIKARST (E) ──────────────────────────────────────────────────
    Shallow weathered zone. Receives net water (P − ET already removed).
    E_loss is the overflow threshold: when E > E_loss the excess becomes
    quickflow Q_ES that bypasses the slow matrix and goes straight to siphon.

      Q_ES   = max(0, E − E_loss)       overflow
      E_eff  = E − Q_ES
      Q_EM   = k_EM × E_eff             slow: epikarst → matrix
      Q_EC   = k_EC × E_eff             fast: epikarst → conduit
      E(t+1) = E_eff − Q_EM − Q_EC + net_water(t)

    ── MATRIX (M) ────────────────────────────────────────────────────
    Slow micropore storage (weeks–months). Pressure gradient drives M→C.

      Q_MC   = k_MC × max(M − C, 0)    pressure exchange
      Q_MS   = k_MS × M                slow baseflow
      M(t+1) = M + Q_EM − Q_MC − Q_MS

    ── CONDUIT (C) ───────────────────────────────────────────────────
    Fast karst drainage network (caves, fractures).

      Q_CS   = k_CS × C                fast drainage to siphon
      C(t+1) = C + Q_EC + Q_MC − Q_CS

    ── SPRING SIPHON (S) ─────────────────────────────────────────────
    The Foux spring exits through a submerged siphon tube.
    Flow only occurs when S > S_crest (explains 97% zero observations).
    Leakage (leak_sp) drains S through rock fractures during droughts.

      Q_into  = Q_CS + Q_MS + Q_ES     all paths merge here
      Q_leak  = leak_sp × S            seepage during drought
      Q_sip   = k_sip × max(S − S_crest, 0)   zero below crest
      S(t+1)  = S + Q_into − Q_sip − Q_leak

    Returns Q_spring [m³/h] for every timestep.
    """
    T = len(net_water)
    Q_out = np.zeros(T)
    E = 0.0; M = 0.0; C = 0.0; S = 0.0

    for t in range(T):
        # Epikarst
        Q_ES  = max(0.0, E - E_loss)
        E_eff = E - Q_ES
        Q_EM  = k_EM * E_eff
        Q_EC  = k_EC * E_eff
        E = E_eff - Q_EM - Q_EC + net_water[t]
        if E < 0.0: E = 0.0

        # Matrix
        Q_MC = k_MC * max(M - C, 0.0)
        Q_MS = k_MS * M
        M = M + Q_EM - Q_MC - Q_MS
        if M < 0.0: M = 0.0

        # Conduit
        Q_CS = k_CS * C
        C = C + Q_EC + Q_MC - Q_CS
        if C < 0.0: C = 0.0

        # Siphon
        Q_into = Q_CS + Q_MS + Q_ES
        Q_leak = leak_sp * S
        Q_sip  = k_sip * max(S - S_crest, 0.0)
        S = S + Q_into - Q_sip - Q_leak
        if S < 0.0: S = 0.0

        Q_out[t] = Q_sip
    return Q_out


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — PARAMETER UNPACKING
# ══════════════════════════════════════════════════════════════════════
# k-values are calibrated on log₁₀ scale (span ~5 orders of magnitude).

def unpack_spring(raw_p):
    lkEM, lkEC, lkMC, lkMS, lkCS, E_loss, S_crest, k_sip, leak = raw_p
    return (10**lkEM, 10**lkEC, 10**lkMC, 10**lkMS, 10**lkCS,
            E_loss, S_crest, k_sip, leak)

def model_spring(raw_p, net_water):
    return run_model_spring(net_water, *unpack_spring(raw_p))


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — METRICS
# ══════════════════════════════════════════════════════════════════════

def nse(o, s):
    d = np.sum((o - o.mean())**2)
    return float(1. - np.sum((o - s)**2) / d) if d > 0 else -np.inf

def kge(o, s, event_only=False):
    """
    Kling-Gupta Efficiency (Gupta et al. 2009). 1 = perfect.
    event_only=True: compute only on non-zero observed timesteps.
    Required for the spring (97% zeros) — with all-zero obs, std(o)≈0
    and mean(o)≈0 make α and β undefined, giving meaninglessly large-negative KGE.
    """
    if event_only:
        mask = o > 0.01 * o.max()
        if mask.sum() < 5:
            return float("nan")
        o = o[mask]; s = s[mask]
    r  = np.corrcoef(o, s)[0, 1]
    al = np.std(s)  / (np.std(o)  + 1e-12)
    be = np.mean(s) / (np.mean(o) + 1e-12)
    return float(1. - math.sqrt((r-1)**2 + (al-1)**2 + (be-1)**2))

def metrics(o, s, event_kge=False):
    return {"NSE":  round(nse(o, s), 3),
            "KGE":  round(kge(o, s, event_only=event_kge), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(o, s))), 3),
            "MAE":  round(float(mean_absolute_error(o, s)), 3),
            "R2":   round(float(r2_score(o, s)), 3)}


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — OBJECTIVE FUNCTION
# ══════════════════════════════════════════════════════════════════════
# Spring-specific challenges:
#   1. 97%+ observations are zero → standard NSE degenerates to zero prediction.
#      Fix: compute scale factor ONLY on non-zero event timesteps.
#   2. Model may fire continuously (low false-positive penalty).
#      Fix: penalise energy at zero-observation times.
#
# Combined loss = 0.45×NSE_loss + 0.45×logNSE_loss + 0.10×FP_energy

def make_obj_spring(net_tr, Q_tr, warmup):
    ev    = Q_tr > 0.01 * Q_tr.max()   # event mask (non-zero timesteps)

    def obj(raw_p):
        Q_mod = model_spring(raw_p, net_tr)[warmup:]
        Q_obs = Q_tr[warmup:]
        ev_w  = ev[warmup:]
        Qe    = Q_obs[ev_w]
        Se    = Q_mod[ev_w]

        # Scale factor from event timesteps only
        denom = np.dot(Se, Se)
        if denom < 1e-20: return 1.0
        sf  = max(0., np.dot(Qe, Se) / denom)
        Sm  = sf * Q_mod

        # NSE loss on events
        o_ev = Q_obs[ev_w]; s_ev = Sm[ev_w]
        dl   = np.sum((o_ev - o_ev.mean())**2)
        ll   = np.sum((o_ev - s_ev)**2) / dl if dl > 1e-12 else 1.

        # Log-NSE loss (weights low-flow periods more)
        ol  = np.log1p(Q_obs + EPS_Q); sl = np.log1p(Sm + EPS_Q)
        dl2 = np.sum((ol - ol.mean())**2)
        lg  = np.sum((ol - sl)**2) / dl2 if dl2 > 1e-12 else 1.

        # False-positive energy penalty (model fires when spring is dry)
        fp       = np.dot(Sm[~ev_w], Sm[~ev_w])
        fp_energy = fp / (np.dot(Sm, Sm) + 1e-30)

        return float(np.clip(0.45*ll + 0.45*lg + 0.10*fp_energy, -1e6, 1e6))
    return obj


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — PARAMETER BOUNDS AND WARM SEEDS
# ══════════════════════════════════════════════════════════════════════
# Bounds based on physical reasoning for Mediterranean karst.
# k-values on log₁₀ scale; thresholds in m³.

BOUNDS_SP = [
    (-5.0,  0.0),    # log10(k_EM)  epikarst → matrix  [h⁻¹]
    (-5.0,  0.0),    # log10(k_EC)  epikarst → conduit [h⁻¹]
    (-6.0, -1.0),    # log10(k_MC)  matrix–conduit pressure exchange [h⁻¹]
    (-6.0, -1.0),    # log10(k_MS)  matrix → siphon (slow baseflow) [h⁻¹]
    (-4.0,  0.0),    # log10(k_CS)  conduit → siphon (fast flow) [h⁻¹]
    (1e3,   5e6),    # E_loss [m³]  epikarst overflow threshold
    (0.0,   2e5),    # S_crest [m³] spring siphon crest level
    (0.001, 0.5),    # k_sip [h⁻¹] siphon discharge rate
    (1e-5,  0.05),   # leak_sp [h⁻¹] siphon leakage/decay during drought
]

# Warm seeds anchored to recession analysis of Foux spring hydrograph tails:
#   k_sip ≈ 0.025 h⁻¹  (log₁₀ ≈ −1.60)
#   At peak Q = 4547 m³/h: S − S_crest ≈ 182,000 m³
#   → S_crest seeds in range 1e4–8e4 m³
#
# Format: [log_k_EM, log_k_EC, log_k_MC, log_k_MS, log_k_CS,
#           E_loss, S_crest, k_sip, leak]
SEEDS_SP = np.array([
    [-3.0, -1.0, -6.0, -2.5, -1.6,  2e5, 3e4, 0.025, 0.003],  # data-anchored baseline
    [-2.5, -0.8, -6.0, -2.0, -1.6,  5e5, 5e4, 0.050, 0.002],  # faster conduit
    [-3.5, -1.5, -6.0, -3.0, -1.6,  1e5, 2e4, 0.025, 0.005],  # deeper matrix
    [-2.0, -0.5, -6.0, -2.0, -1.6,  5e5, 8e4, 0.030, 0.001],  # fast epikarst
    [-4.0, -2.0, -6.0, -3.5, -1.6,  5e4, 1e4, 0.020, 0.008],  # slow E, high leak
    [-3.2, -1.2, -6.0, -2.8, -1.6,  3e5, 4e4, 0.040, 0.004],  # faster siphon
])


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — CALIBRATION
# ══════════════════════════════════════════════════════════════════════
# Strategy: warm-start L-BFGS-B from physical seeds → best seed goes
# into DE population slot 0 → DE global search → polished with L-BFGS-B.

def calibrate():
    net_tr = net_sp[:NTR];  net_te = net_sp[NTR:]
    Q_tr   = Q_foux_obs[:NTR];  Q_te = Q_foux_obs[NTR:]
    obj    = make_obj_spring(net_tr, Q_tr, WARMUP_SP)

    # ── Warm-start L-BFGS-B ──
    best_val = np.inf;  best_x = SEEDS_SP[0].copy()
    for x0 in SEEDS_SP:
        x0c = np.clip(x0, [b[0] for b in BOUNDS_SP], [b[1] for b in BOUNDS_SP])
        try:
            r = _lbfgsb(obj, x0c, method="L-BFGS-B", bounds=BOUNDS_SP,
                        options={"maxiter": LBFGS_ITER, "ftol": 1e-13, "gtol": 1e-9})
            if r.fun < best_val:
                best_val = r.fun;  best_x = r.x.copy()
        except Exception:
            pass
    print(f"[spring_lumped] Warm-start best obj = {best_val:.4f}", flush=True)

    # ── Seed DE population ──
    n_p = len(BOUNDS_SP)
    pop = np.random.uniform(
        [b[0] for b in BOUNDS_SP], [b[1] for b in BOUNDS_SP],
        size=(DE_POPSIZE * n_p, n_p))
    pop[0] = best_x

    # ── Differential Evolution ──
    print(f"[spring_lumped] DE calibration "
          f"({DE_POPSIZE}×{n_p} params, maxiter={DE_MAXITER}) …", flush=True)
    res = differential_evolution(
        obj, BOUNDS_SP, seed=42, init=pop,
        maxiter=DE_MAXITER, popsize=DE_POPSIZE, polish=True,
        mutation=(0.5, 1.5), recombination=0.9,
        updating="immediate", workers=1, disp=False,
    )
    gc.collect()
    p_opt = res.x

    # ── Scale factor from training data ──
    Q_mod_tr = model_spring(p_opt, net_tr)[WARMUP_SP:]
    Q_obs_tr = Q_tr[WARMUP_SP:]
    denom    = np.dot(Q_mod_tr, Q_mod_tr)
    sf       = max(0., np.dot(Q_obs_tr, Q_mod_tr) / denom) if denom > 1e-20 else 1.

    # ── Full-period prediction for LSTM physics prior ──
    # Run model on ALL timesteps (train + test) and save to CSV.
    # LSTM files load this as a physics-prior feature (Q_foux_lumped_lag6).
    Q_mod_full   = sf * model_spring(p_opt, net_sp)   # m³/h, full dataset
    Q_full_ls    = Q_mod_full / 3.6                    # convert to L/s
    df_full      = pd.DataFrame({"Q_foux_lumped_ls": Q_full_ls}, index=df.index)
    df_full.index.name = "date"
    df_full.to_csv(os.path.join(RESULTS_DIR, "spring_lumped_full_predictions.csv"))
    print("[spring_lumped] Saved spring_lumped_full_predictions.csv", flush=True)

    # ── Test set ──
    Q_mod_te = sf * model_spring(p_opt, net_te)
    met      = metrics(Q_te, Q_mod_te, event_kge=True)
    print(f"[spring_lumped] NSE={met['NSE']:.3f}  KGE={met['KGE']:.3f}  "
          f"RMSE={met['RMSE']:.1f} m³/h  (DE obj≈{res.fun:.4f})", flush=True)

    # Return p_opt so the caller can display calibrated parameter values
    return Q_mod_te, Q_te, df.index[NTR:], met, p_opt


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*60, flush=True)
print("[spring_lumped]  SPRING LUMPED CALIBRATION (Foux)", flush=True)
print("="*60, flush=True)

Qp, Qo, idx_te, met, p_opt = calibrate()

# Unpack calibrated parameters to physical values for annotation
(kEM, kEC, kMC, kMS, kCS,
 E_loss, S_crest, k_sip, leak) = unpack_spring(p_opt)

# Save predictions (convert m³/h → L/s for output)
df_pred = pd.DataFrame({
    "observed_ls":  Qo / 3.6,
    "predicted_ls": Qp / 3.6,
}, index=idx_te)
df_pred.index.name = "date"
df_pred.to_csv(os.path.join(RESULTS_DIR, "spring_lumped_final_pred.csv"))
print("[spring_lumped] Saved spring_lumped_final_pred.csv", flush=True)

# Save metrics
pd.DataFrame([{"model": "Spring Lumped V3", **met}]).to_csv(
    os.path.join(RESULTS_DIR, "spring_lumped_final_metrics.csv"), index=False)
print("[spring_lumped] Saved spring_lumped_final_metrics.csv", flush=True)
print(f"\n  NSE={met['NSE']:.3f}  KGE(events)={met['KGE']:.3f}  "
      f"RMSE={met['RMSE']/3.6:.1f} L/s  R²={met['R2']:.3f}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — COMBINED PLOT (time series + scatter in one figure)
# ══════════════════════════════════════════════════════════════════════
# Left panel (75%): time series over PLOT_START → PLOT_END window.
# Right panel (25%): scatter of full test set (all predicted vs observed).
PLOT_START = "2024-02-01"
PLOT_END   = "2024-03-31 23:00"

MODEL_COLOR = "#276749"   # green for river
OBS_COLOR   = "#E53E3E"
BG_MAIN  = "#F7F9FC"; BG_PANEL = "#FFFFFF"
GRID_COL = "#D1D9E6"; TEXT_COL = "#1A202C"; LBL_COL = "#4A5568"
plt.rcParams.update({"font.family": "DejaVu Sans",
                     "figure.facecolor": BG_MAIN, "axes.facecolor": BG_PANEL,
                     "figure.dpi": 150, "axes.grid": True,
                     "grid.color": GRID_COL, "grid.linewidth": 0.6})

# Convert to L/s for display
Qp_ls = Qp / 3.6
Qo_ls = Qo / 3.6

# Build calibrated parameter string for annotation under the plot
param_line = (
    f"Calibrated parameters —  "
    f"kEM = {kEM:.2e} h⁻¹  |  kEC = {kEC:.2e} h⁻¹  |  kMC = {kMC:.2e} h⁻¹  |  "
    f"kMS = {kMS:.2e} h⁻¹  |  kCS = {kCS:.2e} h⁻¹  |  "
    f"E_loss = {E_loss:.2e} m³  |  S_crest = {S_crest:.2e} m³  |  "
    f"k_sip = {k_sip:.4f} h⁻¹  |  leak = {leak:.2e} h⁻¹"
)

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG_MAIN)
gs  = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1],
                        width_ratios=[3, 1], hspace=0.45, wspace=0.12)
ax1   = fig.add_subplot(gs[0, 0])
   # time series
#ax2 = fig.add_subplot(gs[1])   # scatter

# ── Time series panel ──
df_tmp = pd.DataFrame({"obs": Qo_ls, "pred": Qp_ls}, index=idx_te)
try:
    df_win = df_tmp[PLOT_START:PLOT_END]
except Exception:
    df_win = df_tmp

ax1.fill_between(df_win.index, df_win["obs"], alpha=0.12, color=OBS, zorder=2)
ax1.plot(df_win.index, df_win["obs"],  color=OBS,  lw=1.4, label="Observed",  zorder=5)
ax1.plot(df_win.index, df_win["pred"], color=SP_C, lw=1.2, alpha=0.90,
         label="Spring Lumped V3 (E-M-C + Siphon)", zorder=4)

ax1.text(0.01, 0.97,
         f"NSE = {met['NSE']:.3f}     KGE(events) = {met['KGE']:.3f}",
         transform=ax1.transAxes, fontsize=12, fontweight="bold",
         color="#1A365D", va="top", ha="left", bbox=ABOX, zorder=10)

ax1.set_title("Spring Lumped V3 (E-M-C + Siphon)  —  6-h Lead Prediction (test period)",
              fontsize=11, fontweight="bold", color=TXT)
ax1.set_ylabel("Foux Spring Discharge (L/s)", fontsize=9, color=LBL)
ax1.legend(fontsize=9, loc="upper right", facecolor=BG, edgecolor=GRID)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
ax1.xaxis.set_major_locator(
    mdates.DayLocator(interval=3)
)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax1.tick_params(colors=LBL, labelsize=8)
for sp in ax1.spines.values(): sp.set_edgecolor(GRID)
'''
# ── Scatter panel ──
mask  = ~np.isnan(Qp_ls)
om, pm = Qo_ls[mask], Qp_ls[mask]
lim   = max(om.max(), pm.max()) * 1.08

ax2.scatter(om, pm, color=SP_C, alpha=0.25, s=5, rasterized=True, zorder=3)
ax2.plot([0, lim], [0, lim], "k--", lw=0.9, zorder=4)

ax2.text(0.05, 0.95,
         f"R²   = {met['R2']:.3f}\nRMSE = {met['RMSE']/3.6:.1f} L/s\nMAE  = {met['MAE']/3.6:.1f} L/s",
         transform=ax2.transAxes, fontsize=8, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                   edgecolor=GRID, alpha=0.9))

ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
ax2.set_xlabel("Observed (L/s)",  fontsize=9, color=LBL)
ax2.set_ylabel("Predicted (L/s)", fontsize=9, color=LBL)
ax2.set_title("Scatter — Full Test Set", fontsize=9, color=TXT, fontweight="bold")
ax2.set_aspect("equal", adjustable="box")
ax2.tick_params(colors=LBL, labelsize=8)
for sp in ax2.spines.values(): sp.set_edgecolor(GRID)

# Leave 13% of figure height at the bottom for the parameter annotation
plt.tight_layout(rect=[0, 0.13, 1, 1])
'''
# Parameter text bar — centred under both panels
fig.text(0.5, 0, param_line,
         ha="center", va="center", fontsize=10, color=TXT,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF8FF",
                   edgecolor="#3182CE", alpha=0.85))

out_png = os.path.join(RESULTS_DIR, "spring_lumped_final.png")
plt.savefig(out_png, bbox_inches="tight", dpi=150)
plt.show()
plt.close()
print(f"[spring_lumped] Saved {out_png}", flush=True)
print("\n[spring_lumped] ✓  Done. Outputs → final/results/", flush=True)
