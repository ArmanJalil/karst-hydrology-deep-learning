"""
semidistributed_foux_final.py  —  Foux Karst System
=====================================================
Semi-distributed spring (Foux) discharge prediction.

Key design: intrinsic reservoir k-rates are FIXED from the calibrated
lumped spring model.  Only the spatially-explicit parameters are
calibrated here:
    tau_base — base routing time constant [h] (scaled by d_rel per HRU)
    r56      — relative contribution of HRU-5 vs HRU-6 to the siphon
    E_loss   — per-HRU epikarst overflow threshold [m³]
    S_crest  — siphon activation threshold [m³]

Fixed from lumped spring_lumped_final.py (calibrated values):
    kEM   = 2.00e-05 h⁻¹
    kEC   = 9.95e-04 h⁻¹
    kMC   = 1.89e-05 h⁻¹
    kMS   = 8.70e-03 h⁻¹
    kCS   = 5.51e-03 h⁻¹
    k_sip = 5.50e-03 h⁻¹
    leak  = 1.85e-02 h⁻¹

HRU assignments (spring sub-basin):
    HRU 5 and HRU 6

Input  : D:\\foux\\Foux\\final\\foux_total_water.csv
Outputs: final/results/semidist_foux_final_pred.csv
         final/results/semidist_foux_final_metrics.csv
         final/results/semidist_foux_final.png
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

PROJECT_DIR = os.environ.get("FOUX_DATA_DIR", "./data")
DATA_CSV    = os.path.join(PROJECT_DIR, "foux_total_water.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

PREFIX = "semidist_foux_final"

DT         = 1
PRED_LAG   = 6
TRAIN_FRAC = 0.70
EPS_Q      = 0.01

DE_MAXITER = 500
DE_POPSIZE = 18
LBFGS_ITER = 800

WARMUP_SP  = 24 * 30   # 30 days

PLOT_START = "2024-02-01"
PLOT_END   = "2024-03-31 23:00"

# ── Fixed k-parameters from lumped spring_lumped_final.py ────────────
KEM   = 2.00e-05   # epikarst → matrix     [h⁻¹]
KEC   = 9.95e-04   # epikarst → conduit    [h⁻¹]
KMC   = 1.89e-05   # matrix ↔ conduit      [h⁻¹]
KMS   = 8.70e-03   # matrix → siphon       [h⁻¹]
KCS   = 5.51e-03   # conduit → siphon      [h⁻¹]
K_SIP = 5.50e-03   # siphon drainage rate  [h⁻¹]
LEAK  = 1.85e-02   # siphon leakage rate   [h⁻¹]

# ── HRU routing distances (relative travel times to spring outlet) ────
SPRING_HRUS = [5, 6]
D_REL_SP    = {5: 0.8, 6: 1.0}

# Plot aesthetics (match spring_lumped_final.py)
BG_MAIN  = "#F7F9FC"; BG_PANEL = "#FFFFFF"; GRID_COL = "#D1D9E6"
TEXT_COL = "#1A202C"; LBL_COL  = "#4A5568"
SP_COLOR = "#2B6CB0"   # blue for spring
OBS_COL  = "#E53E3E"   # red for observed
ABOX = dict(boxstyle="round,pad=0.35", facecolor="#EBF8FF",
            edgecolor="#3182CE", alpha=0.92)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": BG_MAIN,
    "axes.facecolor":   BG_PANEL,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.color": GRID_COL,
    "grid.linewidth": 0.6,
})


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════

print(f"[{PREFIX}] Loading data …", flush=True)
df = pd.read_csv(DATA_CSV, parse_dates=["date"], index_col="date",
                 dayfirst=False).sort_index()
df.columns = ["HRU_4(m3/h)" if c == "(m3/h)" else c for c in df.columns]

def get_hru_array(hru_list):
    return np.array([df[f"HRU_{h}(m3/h)"].values.astype(np.float64)
                     for h in hru_list])

net_sp_arr = get_hru_array(SPRING_HRUS)   # (2, T)
drel_sp    = np.array([D_REL_SP[h] for h in SPRING_HRUS], dtype=np.float64)

Q_foux_obs = df["Foux_Charge(l/s)"].values.astype(np.float64) * 3.6   # [m³/h]

T_total = len(df)
NTR     = int(T_total * TRAIN_FRAC)
print(f"[{PREFIX}] T={T_total}  Train={NTR}  Test={T_total-NTR}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — PHYSICS MODEL (Numba JIT)
# ══════════════════════════════════════════════════════════════════════
# k-rates are passed as compile-time constants via the closure so the
# JIT signature stays fixed across calibration calls.

@jit(nopython=True, cache=True)
def run_spring_sd(net_arr, tau_arr, r56, E_loss, S_crest,
                  k_EM, k_EC, k_MC, k_MS, k_CS, k_sip, leak):
    """
    Semi-distributed spring model with per-HRU E-M-C chains.

    net_arr : (2, T)  net water [m³/h] for HRU5 (row 0) and HRU6 (row 1)
    tau_arr : (2,)    routing time constants [h]  (tau_base × d_rel_i)
    r56     : weight on HRU-5 output (0=only HRU6, 1=only HRU5)
    E_loss  : per-HRU epikarst overflow threshold [m³]
    S_crest : siphon activation threshold [m³]

    Each HRU runs an independent E→M+C reservoir chain with the SAME
    intrinsic k-rates (material property) but DIFFERENT net water inputs
    and routing delays (geometry property).
    """
    T = net_arr.shape[1]
    E5=0.; M5=0.; C5=0.; R5=0.
    E6=0.; M6=0.; C6=0.; R6=0.
    S  = 0.
    Q_out = np.zeros(T)
    tau5 = max(tau_arr[0], 1.0)
    tau6 = max(tau_arr[1], 1.0)

    for t in range(T):
        # HRU 5
        QES5 = max(0., E5 - E_loss);  Ef5 = E5 - QES5
        QEM5 = k_EM * Ef5;  QEC5 = k_EC * Ef5
        E5   = Ef5 - QEM5 - QEC5 + net_arr[0, t]
        if E5 < 0.: E5 = 0.
        QMC5 = k_MC * max(M5 - C5, 0.);  QMS5 = k_MS * M5
        M5   = M5 + QEM5 - QMC5 - QMS5
        if M5 < 0.: M5 = 0.
        QCS5 = k_CS * C5
        C5   = C5 + QEC5 + QMC5 - QCS5
        if C5 < 0.: C5 = 0.
        Q5   = QCS5 + QMS5 + QES5
        R5   = R5 + Q5 - R5 / tau5
        if R5 < 0.: R5 = 0.
        Qr5  = R5 / tau5

        # HRU 6
        QES6 = max(0., E6 - E_loss);  Ef6 = E6 - QES6
        QEM6 = k_EM * Ef6;  QEC6 = k_EC * Ef6
        E6   = Ef6 - QEM6 - QEC6 + net_arr[1, t]
        if E6 < 0.: E6 = 0.
        QMC6 = k_MC * max(M6 - C6, 0.);  QMS6 = k_MS * M6
        M6   = M6 + QEM6 - QMC6 - QMS6
        if M6 < 0.: M6 = 0.
        QCS6 = k_CS * C6
        C6   = C6 + QEC6 + QMC6 - QCS6
        if C6 < 0.: C6 = 0.
        Q6   = QCS6 + QMS6 + QES6
        R6   = R6 + Q6 - R6 / tau6
        if R6 < 0.: R6 = 0.
        Qr6  = R6 / tau6

        # Combine HRUs → siphon
        Q_in  = r56 * Qr5 + (1.0 - r56) * Qr6
        Q_lk  = leak * S
        Q_sip = k_sip * max(S - S_crest, 0.)
        S     = S + Q_in - Q_sip - Q_lk
        if S < 0.: S = 0.
        Q_out[t] = Q_sip

    return Q_out


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — PARAMETER UNPACKING
# ══════════════════════════════════════════════════════════════════════
# Calibrated params:  [tau_base, r56, E_loss, S_crest]
# All k-rates injected from global constants (fixed).

def unpack(p):
    tau_base, r56, E_loss, S_crest = p
    tau_arr = drel_sp * max(tau_base, 1.0)
    r56     = float(np.clip(r56, 0.01, 0.99))
    E_loss  = max(E_loss, 0.0)
    S_crest = max(S_crest, 0.0)
    return tau_arr, r56, E_loss, S_crest

def model_fn(p, net_arr):
    tau_arr, r56, E_loss, S_crest = unpack(p)
    return run_spring_sd(net_arr, tau_arr, r56, E_loss, S_crest,
                         KEM, KEC, KMC, KMS, KCS, K_SIP, LEAK)


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — METRICS
# ══════════════════════════════════════════════════════════════════════

def nse(o, s):
    d = np.sum((o - o.mean())**2)
    return float(1. - np.sum((o - s)**2) / d) if d > 0 else -np.inf

def kge(o, s, event_only=False):
    if event_only:
        mask = o > 0.01 * o.max()
        if mask.sum() < 5: return float("nan")
        o = o[mask]; s = s[mask]
    r  = np.corrcoef(o, s)[0, 1]
    al = np.std(s)  / (np.std(o)  + 1e-12)
    be = np.mean(s) / (np.mean(o) + 1e-12)
    return float(1. - math.sqrt((r-1)**2 + (al-1)**2 + (be-1)**2))

def metrics(o, s):
    return {"NSE":  round(nse(o, s), 3),
            "KGE":  round(kge(o, s, event_only=True), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(o, s))), 3),
            "MAE":  round(float(mean_absolute_error(o, s)), 3),
            "R2":   round(float(r2_score(o, s)), 3)}


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — OBJECTIVE FUNCTION
# ══════════════════════════════════════════════════════════════════════
# Spring-specific: event-based scale factor + false-positive penalty.

def make_obj(net_arr_tr, Q_tr, warmup):
    ev = Q_tr > 0.01 * Q_tr.max()
    def obj(p):
        Q_mod = model_fn(p, net_arr_tr)[warmup:]
        Q_obs = Q_tr[warmup:]
        Se = Q_mod[ev[warmup:]]; Oe = Q_obs[ev[warmup:]]
        denom = np.dot(Se, Se)
        if denom < 1e-20: return 1.0
        sf = max(0., np.dot(Oe, Se) / denom)
        Sm = sf * Q_mod

        o_ev = Q_obs[ev[warmup:]]; s_ev = Sm[ev[warmup:]]
        dl   = np.sum((o_ev - o_ev.mean())**2)
        ll   = np.sum((o_ev - s_ev)**2) / dl if dl > 1e-12 else 1.

        ol   = np.log1p(Q_obs + EPS_Q); sl = np.log1p(Sm + EPS_Q)
        dl2  = np.sum((ol - ol.mean())**2)
        lg   = np.sum((ol - sl)**2) / dl2 if dl2 > 1e-12 else 1.

        fp   = np.dot(Sm[~ev[warmup:]], Sm[~ev[warmup:]])
        fp_e = fp / (np.dot(Sm, Sm) + 1e-30)
        return float(np.clip(0.35*ll + 0.35*lg + 0.30*fp_e, -1e6, 1e6))
    return obj


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — BOUNDS AND WARM SEEDS
# ══════════════════════════════════════════════════════════════════════
# Only 4 parameters are calibrated:
#   [tau_base (h), r56 (-), E_loss (m³), S_crest (m³)]

BOUNDS = [
    (1.,    300.),     # tau_base [h]
    (0.05,  0.95),     # r56 (HRU-5 weight)
    (1e3,   8e6),      # E_loss [m³]  (can be larger/smaller than lumped per-HRU)
    (5e3,   5e5),      # S_crest [m³] (siphon threshold)
]

# Warm seeds anchored to lumped calibration results
SEEDS = np.array([
    [24.,  0.50, 2.99e6, 1.36e5],   # lumped anchor
    [12.,  0.55, 1.50e6, 1.00e5],   # shorter tau, smaller E_loss
    [48.,  0.45, 4.00e6, 2.00e5],   # longer tau, larger E_loss
    [36.,  0.60, 2.00e6, 8.00e4],   # mid tau, HRU5 heavier
    [6.,   0.40, 5.00e5, 5.00e4],   # very fast routing
    [72.,  0.70, 5.00e6, 1.50e5],   # slow routing, HRU5 dominant
])


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — CALIBRATION
# ══════════════════════════════════════════════════════════════════════

print(f"\n[{PREFIX}] Starting calibration …", flush=True)
net_tr = net_sp_arr[:, :NTR];  net_te = net_sp_arr[:, NTR:]
Q_tr   = Q_foux_obs[:NTR];     Q_te   = Q_foux_obs[NTR:]
obj    = make_obj(net_tr, Q_tr, WARMUP_SP)

# Warm-start L-BFGS-B
best_val = np.inf;  best_x = SEEDS[0].copy()
for x0 in SEEDS:
    x0c = np.clip(x0, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
    try:
        r = _lbfgsb(obj, x0c, method="L-BFGS-B", bounds=BOUNDS,
                    options={"maxiter": LBFGS_ITER, "ftol": 1e-13, "gtol": 1e-9})
        if r.fun < best_val:
            best_val = r.fun;  best_x = r.x.copy()
    except Exception:
        pass
print(f"[{PREFIX}] Warm-start best obj = {best_val:.4f}", flush=True)

# DE (global)
n_p = len(BOUNDS)
pop = np.random.uniform([b[0] for b in BOUNDS], [b[1] for b in BOUNDS],
                         size=(DE_POPSIZE * n_p, n_p))
pop[0] = best_x
print(f"[{PREFIX}] DE ({DE_POPSIZE}×{n_p}, maxiter={DE_MAXITER}) …", flush=True)
res = differential_evolution(obj, BOUNDS, seed=42, init=pop,
                             maxiter=DE_MAXITER, popsize=DE_POPSIZE,
                             polish=True, mutation=(0.5, 1.5),
                             recombination=0.9, updating="immediate",
                             workers=1, disp=False)
gc.collect()
p_opt = res.x

# Scale factor (event-based, same as objective)
Q_mod_tr = model_fn(p_opt, net_tr)[WARMUP_SP:]
Q_tr_wu  = Q_tr[WARMUP_SP:]
ev_mask  = Q_tr_wu > 0.01 * Q_tr_wu.max()
Se = Q_mod_tr[ev_mask]; Oe = Q_tr_wu[ev_mask]
denom_sf = np.dot(Se, Se)
sf = max(0., np.dot(Oe, Se) / denom_sf) if denom_sf > 1e-20 else 1.

# Test set
Q_mod_te = sf * model_fn(p_opt, net_te)
m        = metrics(Q_te, Q_mod_te)
print(f"[{PREFIX}] NSE={m['NSE']:.3f}  KGE={m['KGE']:.3f}  "
      f"RMSE={m['RMSE']/3.6:.1f} L/s  (DE obj≈{res.fun:.4f})", flush=True)

# Unpack final params
tau_arr, r56, E_loss, S_crest = unpack(p_opt)
tau_base = float(p_opt[0])

# Save CSVs
df_pred = pd.DataFrame({
    "date": df.index[NTR:],
    "observed_ls":  Q_te / 3.6,
    "predicted_ls": Q_mod_te / 3.6,
}).set_index("date")
df_pred.to_csv(os.path.join(RESULTS_DIR, f"{PREFIX}_pred.csv"))

pd.DataFrame([{"model": PREFIX, **m}]).to_csv(
    os.path.join(RESULTS_DIR, f"{PREFIX}_metrics.csv"), index=False)
print(f"[{PREFIX}] Saved prediction and metrics CSVs", flush=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — PLOT
# ══════════════════════════════════════════════════════════════════════

Qp_ls = Q_mod_te / 3.6
Qo_ls = Q_te     / 3.6
idx   = df.index[NTR:]

fig = plt.figure(figsize=(18, 10), facecolor=BG_MAIN)
gs  = gridspec.GridSpec(2, 2, width_ratios=[2.5, 1], wspace=0.12)
ax1 = fig.add_subplot(gs[0])
#ax2 = fig.add_subplot(gs[1])

# ── Time-series window ──
tmp = pd.DataFrame({"obs": Qo_ls, "pred": Qp_ls}, index=idx)
try:    tmp = tmp[PLOT_START:PLOT_END]
except Exception: pass

ax1.set_facecolor(BG_PANEL)
ax1.fill_between(tmp.index, tmp["obs"], alpha=0.15, color=OBS_COL)
ax1.plot(tmp.index, tmp["obs"],  color=OBS_COL,  lw=1.4, label="Observed",            zorder=5)
ax1.plot(tmp.index, tmp["pred"], color=SP_COLOR, lw=1.1, alpha=0.88,
         label="Spring Lumped (E-M-C)", zorder=4)
ax1.annotate(f"NSE = {m['NSE']:.3f}     KGE(events) = {m['KGE']:.3f}",
             xy=(0.01, 0.97), xycoords="axes fraction",
             fontsize=12, fontweight="bold", color="#1A365D",
             va="top", ha="left", bbox=ABOX, zorder=10)
ax1.set_title("Semi-Distributed Foux  —  6-h Lead Prediction (test period)",
              fontsize=11, fontweight="bold", color=TEXT_COL, pad=6)
ax1.set_ylabel("Foux Spring Discharge (L/s)", fontsize=9, color=LBL_COL)
ax1.legend(fontsize=8, loc="upper right", facecolor=BG_PANEL, edgecolor=GRID_COL)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
ax1.xaxis.set_major_locator(
    mdates.DayLocator(interval=3)
)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
for sp in ax1.spines.values(): sp.set_edgecolor(GRID_COL)
ax1.tick_params(colors=LBL_COL, labelsize=8)
'''
# ── Scatter (full test set) ──
lim = max(Qo_ls.max(), Qp_ls.max()) * 1.08
ax2.set_facecolor(BG_PANEL)
ax2.scatter(Qo_ls, Qp_ls, color=SP_COLOR, alpha=0.20, s=4, rasterized=True)
ax2.plot([0, lim], [0, lim], "k--", lw=0.9)
ax2.annotate(f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']/3.6:.1f} L/s\nMAE = {m['MAE']/3.6:.1f} L/s",
             xy=(0.05, 0.94), xycoords="axes fraction",
             fontsize=8, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=GRID_COL, alpha=0.85))
ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
ax2.set_xlabel("Observed (L/s)",  fontsize=8, color=LBL_COL)
ax2.set_ylabel("Predicted (L/s)", fontsize=8, color=LBL_COL)
ax2.set_title("Scatter — Full Test Set", fontsize=9, color=TEXT_COL, fontweight="bold")
ax2.set_aspect("equal", adjustable="box")
for sp in ax2.spines.values(): sp.set_edgecolor(GRID_COL)
ax2.tick_params(colors=LBL_COL, labelsize=8)

# ── Parameter annotation bar ──
param_line = (
    f"Calibrated parameters —  "
    f"tau_base = {tau_base:.1f} h  |  r56 = {r56:.3f}  |  "
    f"E_loss = {E_loss:.2e} m³  |  S_crest = {S_crest:.2e} m³"
    f"\n"
    f"Fixed from lumped —  "
    f"kEM = {KEM:.2e} h⁻¹  |  kEC = {KEC:.2e} h⁻¹  |  kMC = {KMC:.2e} h⁻¹  |  "
    f"kMS = {KMS:.2e} h⁻¹  |  kCS = {KCS:.2e} h⁻¹  |  "
    f"k_sip = {K_SIP:.4f} h⁻¹  |  leak = {LEAK:.2e} h⁻¹"
)
plt.tight_layout(rect=[0, 0.12, 1, 1])
fig.text(0.5, 0.01, param_line, ha="center", va="bottom", fontsize=7.5,
         color=TEXT_COL,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF8FF",
                   edgecolor="#3182CE", alpha=0.90))
'''
out_png = os.path.join(RESULTS_DIR, f"{PREFIX}.png")
plt.savefig(out_png, bbox_inches="tight", dpi=150)
plt.show()
plt.close()
print(f"[{PREFIX}] Saved {os.path.basename(out_png)}", flush=True)
print(f"\n[{PREFIX}] ✓  Done.  Outputs → final/results/", flush=True)

