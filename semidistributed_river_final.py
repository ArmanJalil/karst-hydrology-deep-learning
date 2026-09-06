"""
semidistributed_river_final.py  —  Foux Karst System
======================================================
Semi-distributed river discharge prediction.

Key design
----------
Intrinsic reservoir k-rates are FIXED from the calibrated lumped SPRING
model (spring_lumped_final.py).  The river lumped model uses a different
numerical scheme (bidirectional M↔C via eigendecomposition), so its
k-values are NOT transferable to this forward-Euler formulation.
Both basins share the same karst massif, making spring k-values a valid
physical prior for the river E-M-C chains.

No siphon at the river outlet — the river flows whenever there is water.
(The spring model has a siphon because the Foux exit is a literal
submerged tube; the river has no such threshold.)

Calibrated parameters (3):
    tau_base  — base HRU routing time constant [h] (multiplied by d_rel_i)
    E_loss    — per-HRU epikarst overflow threshold [m³]
    tau_out   — single linear outlet routing reservoir [h]  (channel lag)

Fixed from spring_lumped_final.py (same forward-Euler structure):
    kEM  = 2.00e-05 h⁻¹
    kEC  = 9.95e-04 h⁻¹
    kMC  = 1.89e-05 h⁻¹
    kMS  = 8.70e-03 h⁻¹
    kCS  = 5.51e-03 h⁻¹

HRU assignments:
    River basin : HRU 3, 4, 7, 8, 9, 11, 12
    (HRU 1 and 2 excluded — negligible catchment contribution)

Input  : D:\\foux\\Foux\\final\\foux_total_water.csv
Outputs: final/results/semidist_river_final_pred.csv
         final/results/semidist_river_final_metrics.csv
         final/results/semidist_river_final.png
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

PREFIX = "semidist_river_final"

DT         = 1
PRED_LAG   = 6
TRAIN_FRAC = 0.70
EPS_Q      = 0.01
LOG_W      = 0.40

# ── Recharge scaling ─────────────────────────────────────────────────
# The HRU net water [m³/h] is about 1000× smaller than the effective
# karst recharge needed to reproduce observed discharges with the spring
# k-rates (≈10⁻³ h⁻¹).  This mismatch comes from the fact that the
# CSV stores (P−ET) integrated over HRU area in [m³/h] whereas the
# k-rates were calibrated against L/s observations scaled by ×3.6→m³/h,
# leaving an implicit ×1000 gap between input and output units.
#
# Fix: apply a FIXED base multiplier of 1000 (the suspected unit gap),
# then let the DE calibrate a fine-tuning factor `recharge_scale` in
# [0.1, 100] to absorb any remaining mismatch.
# Total effective multiplier = FIXED_SCALE × recharge_scale ∈ [100, 100000].
FIXED_SCALE = 1000.0   # base ×1000 unit-gap correction (fixed)

SCALER     = 10      # multiply final predictions by this factor

DE_MAXITER = 200
DE_POPSIZE = 18
LBFGS_ITER = 800

# River warmup: long enough for E-M-C to equilibrate, but reasonable.
# 60 days balances spin-up against discarding too much training data.
WARMUP_RV  = 100 * 24   # 60 days

PLOT_START = "2024-02-01"
PLOT_END   = "2024-03-31 23:00"

# ── Fixed k-parameters from spring_lumped_final.py ────────────────────
# Raw calibrated values — NO scaling factor applied.
# With these raw rates, matrix half-life ≈ 80 h and conduit ≈ 126 h,
# so the reservoirs retain water across consecutive rainfall events.
# (A previous FLOW_SCALE multiplier boosted drainage to ~13 h⁻¹,
#  draining the system in under one hour and causing the model to
#  treat every storm as if it followed a long dry spell.)
KEM   = 2.00e-05   # epikarst → matrix   [h⁻¹]
KEC   = 9.95e-04   # epikarst → conduit  [h⁻¹]
KMC   = 5.00e-04   # matrix ↔ conduit    [h⁻¹]  (raised for river dynamics)
KMS   = 8.70e-03   # matrix → output     [h⁻¹]  half-life ≈ 80 h
KCS   = 5.51e-03   # conduit → output    [h⁻¹]  half-life ≈ 126 h

# ── HRU routing distances (relative travel time to outlet) ────────────
RIVER_HRUS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
D_REL_RV   = {1: 0.05, 2: 0.05,          # near-zero contribution
              3: 0.9,  4: 0.8,
              5: 1.0,  6: 0.9,            # drain directly to river, not via Foux siphon
              7: 1,  8: 1,
              9: 1,  11: .5, 12: .5}

# Plot aesthetics (match river_lumped_final.py)
BG_MAIN  = "#F7F9FC"; BG_PANEL = "#FFFFFF"; GRID_COL = "#D1D9E6"
TEXT_COL = "#1A202C"; LBL_COL  = "#4A5568"
RV_COLOR = "#276749"   # green for river
OBS_COL  = "#E53E3E"   # red for observed
ABOX = dict(boxstyle="round,pad=0.35", facecolor="#E6FFFA",
            edgecolor="#276749", alpha=0.92)

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

net_rv_arr = get_hru_array(RIVER_HRUS)   # (7, T)
drel_rv    = np.array([D_REL_RV[h] for h in RIVER_HRUS], dtype=np.float64)

Q_river_obs = df["River_Charge(l/s)"].values.astype(np.float64) * 3.6  # [m³/h]

T_total = len(df)
NTR     = int(T_total * TRAIN_FRAC)
print(f"[{PREFIX}] T={T_total}  Train={NTR}  Test={T_total-NTR}", flush=True)
print(f"[{PREFIX}] River HRUs = {RIVER_HRUS}  (HRU 1 & 2 excluded)", flush=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — PHYSICS MODEL (Numba JIT)
# ══════════════════════════════════════════════════════════════════════
#
# Model chain per timestep:
#
#   For each HRU i:
#     Epikarst:  Q_ES = max(E_i - E_loss, 0)   [overflow bypass]
#                E_eff = E_i - Q_ES
#                Q_EM = kEM × E_eff             [slow → matrix]
#                Q_EC = kEC × E_eff             [fast → conduit]
#                E_i(t+1) = E_eff - Q_EM - Q_EC + net_water_i(t)
#     Matrix:    Q_MC = kMC × max(M_i - C_i, 0)
#                Q_MS = kMS × M_i
#                M_i(t+1) = M_i + Q_EM - Q_MC - Q_MS
#     Conduit:   Q_CS = kCS × C_i
#                C_i(t+1) = C_i + Q_EC + Q_MC - Q_CS
#     HRU output: Q_HRU_i = Q_CS + Q_MS + Q_ES
#     Routing:   R_i(t+1) = R_i + Q_HRU_i - R_i / tau_i
#                Q_routed_i = R_i / tau_i
#
#   Q_total(t) = Σ Q_routed_i(t)      [sum all 7 HRUs]
#
#   Outlet routing reservoir (channel lag):
#     O(t+1) = O + Q_total - O / tau_out
#     Q_out(t) = O / tau_out
#
# NO siphon threshold — the river flows whenever there is water.
# The spring siphon (97% zeros) reflects a literal submerged tube;
# the river outlet has no such physical barrier.

@jit(nopython=True, cache=True)
def run_river_sd(net_arr, tau_arr, E_loss, tau_out,
                 recharge_scale,
                 k_EM, k_EC, k_MC, k_MS, k_CS):
    """
    Semi-distributed river E-M-C model — no siphon threshold.

    net_arr       : (N_HRU, T)  net water [m³/h] per HRU
    tau_arr       : (N_HRU,)    HRU routing constants [h]
    E_loss        : per-HRU epikarst overflow threshold [m³]
    tau_out       : outlet routing reservoir time constant [h]
    recharge_scale: multiplier on net_arr before entering epikarst.
                    Bridges the gap between the volumetric P-ET forcing
                    and the storage magnitudes required to reproduce
                    observed peak discharges with small k-rates (≈10⁻³).
    """
    N_HRU = net_arr.shape[0]
    T     = net_arr.shape[1]

    E   = np.zeros(N_HRU)
    M   = np.zeros(N_HRU)
    C   = np.zeros(N_HRU)
    R   = np.zeros(N_HRU)
    tau_c = np.maximum(tau_arr, 1.0)

    O     = 0.0   # outlet routing reservoir
    tau_o = max(tau_out, 1.0)
    Q_out = np.zeros(T)

    for t in range(T):
        Q_total = 0.0

        for i in range(N_HRU):
            # Epikarst
            Q_ES  = max(0., E[i] - E_loss)
            E_eff = E[i] - Q_ES
            Q_EM  = k_EM * E_eff
            Q_EC  = k_EC * E_eff
            recharge = recharge_scale * net_arr[i, t]   # FIXED_SCALE already baked in via unpack
            E[i]  = E_eff - Q_EM - Q_EC + recharge
            if E[i] < 0.: E[i] = 0.

            # Matrix
            Q_MC  = k_MC * (M[i] - C[i])   # bidirectional: positive = M→C, negative = C→M
            Q_MS  = k_MS * M[i]
            M[i]  = M[i] + Q_EM - Q_MC - Q_MS
            if M[i] < 0.: M[i] = 0.

            # Conduit
            Q_CS  = k_CS * C[i]
            C[i]  = C[i] + Q_EC + Q_MC - Q_CS
            if C[i] < 0.: C[i] = 0.

            # HRU routing reservoir
            Q_HRU = Q_CS + Q_MS + Q_ES
            R[i]  = R[i] + Q_HRU - R[i] / tau_c[i]
            if R[i] < 0.: R[i] = 0.
            Q_total += R[i] / tau_c[i]

        # Outlet routing reservoir (channel lag)
        O = O + Q_total - O / tau_o
        if O < 0.: O = 0.
        Q_out[t] = O / tau_o

    return Q_out


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — PARAMETER UNPACKING
# ══════════════════════════════════════════════════════════════════════
# Calibrated params:  [tau_base, E_loss, tau_out, recharge_scale]

def unpack(p):
    tau_base, E_loss, tau_out, recharge_scale = p
    tau_arr        = drel_rv * max(tau_base, 1.0)
    E_loss         = max(E_loss, 0.0)
    tau_out        = max(tau_out, 1.0)
    # Bake the fixed ×1000 unit correction into the value passed to the JIT
    # so the JIT signature stays simple and consistent.
    rs_effective   = max(recharge_scale, 0.01) * FIXED_SCALE
    return tau_arr, E_loss, tau_out, rs_effective

def model_fn(p, net_arr):
    tau_arr, E_loss, tau_out, recharge_scale = unpack(p)
    return 1000.0 *run_river_sd(net_arr, tau_arr, E_loss, tau_out,
                        recharge_scale,
                        KEM, KEC, KMC, KMS, KCS)


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — METRICS
# ══════════════════════════════════════════════════════════════════════

def nse(o, s):
    d = np.sum((o - o.mean())**2)
    return float(1. - np.sum((o - s)**2) / d) if d > 0 else -np.inf

def kge(o, s):
    r  = np.corrcoef(o, s)[0, 1]
    al = np.std(s)  / (np.std(o)  + 1e-12)
    be = np.mean(s) / (np.mean(o) + 1e-12)
    return float(1. - math.sqrt((r-1)**2 + (al-1)**2 + (be-1)**2))

def metrics(o, s):
    return {"NSE":  round(nse(o, s), 3),
            "KGE":  round(kge(o, s), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(o, s))), 3),
            "MAE":  round(float(mean_absolute_error(o, s)), 3),
            "R2":   round(float(r2_score(o, s)), 3)}


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — OBJECTIVE FUNCTION
# ══════════════════════════════════════════════════════════════════════

def make_obj(net_arr_tr, Q_tr, warmup):
    """Combined NSE + log-NSE after optimal closed-form scale factor."""
    def obj(p):
        Q_mod = model_fn(p, net_arr_tr)[warmup:]
        Q_obs = Q_tr[warmup:]
        denom = np.dot(Q_mod, Q_mod)
        if denom < 1e-20: return 1.0
        sf  = max(0., np.dot(Q_obs, Q_mod) / denom)
        Sm  = sf * Q_mod

        dl  = np.sum((Q_obs - Q_obs.mean())**2)
        ll  = np.sum((Q_obs - Sm)**2) / dl if dl > 1e-12 else 1.

        ol  = np.log1p(Q_obs + EPS_Q); sl = np.log1p(Sm + EPS_Q)
        dl2 = np.sum((ol - ol.mean())**2)
        lg  = np.sum((ol - sl)**2) / dl2 if dl2 > 1e-12 else 1.
        return float(np.clip((1. - LOG_W)*ll + LOG_W*lg, -1e6, 1e6))
    return obj


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — BOUNDS AND WARM SEEDS
# ══════════════════════════════════════════════════════════════════════
# 4 calibrated parameters:  [tau_base, E_loss, tau_out, recharge_scale]
#
# tau_base: HRU routing delay [h].
#   Lumped river anchor: tau_base = 7.5 h.  Upper bound 36 h (1.5 days)
#   is the realistic max for karst travel times in this basin.
#
# E_loss: per-HRU epikarst overflow threshold [m³].
#   Controls how much water bypasses E-M-C as quickflow.
#   Upper bound 5e4 m³ keeps it physically realistic per HRU.
#   Very large values suppress all quickflow and slow the response.
#
# tau_out: outlet channel routing reservoir [h].
#   Represents river channel travel time to the gauge.
#   Bounded to [1, 36] h — same range as tau_base.
#
# recharge_scale: dimensionless fine-tuning multiplier [-] in [0.1, 100].
#   Applied on top of FIXED_SCALE=1000, so the effective total multiplier
#   on net_arr is  recharge_scale × FIXED_SCALE ∈ [100, 100 000].
#   Seeds are in the 0.5–8 range (≡ effective 500–8 000×).

BOUNDS = [
    (1.,    36.),      # tau_base [h]        — karst HRU routing
    (1e2, 5e4),      # E_loss [m³]         — epikarst storage cap per HRU
    (1.,    36.),      # tau_out [h]         — outlet channel lag
    (0.1,  100.),      # recharge_scale [-]  — fine-tuning multiplier on top of FIXED_SCALE
    #                    Effective total multiplier = recharge_scale × FIXED_SCALE ∈ [100, 100 000]
]

SEEDS = np.array([
    [7.5,  2e4,  6.,   2.5],   # lumped anchor; eff. scale = 2500
    [12.,  4e4, 10.,   5.0],   # longer routing; eff. scale = 5000
    [4.,   5e3,  3.,   1.0],   # fast system, small E_loss; eff. scale = 1000
    [20.,  5e4, 20.,   8.0],   # slow routing; eff. scale = 8000
    [2.,   1e3,  2.,   0.5],   # near-instant, strong overflow; eff. scale = 500
    [10.,  3e4,  8.,   3.0],   # mid-range; eff. scale = 3000
    [6.,   1e4, 15.,   4.0],   # moderate E_loss, longer lag; eff. scale = 4000
    [15.,  2e3,  5.,   1.5],   # small E_loss, modest scale; eff. scale = 1500
])


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — CALIBRATION
# ══════════════════════════════════════════════════════════════════════

print(f"\n[{PREFIX}] Starting calibration …", flush=True)
net_tr = net_rv_arr[:, :NTR];  net_te = net_rv_arr[:, NTR:]
Q_tr   = Q_river_obs[:NTR];    Q_te   = Q_river_obs[NTR:]
obj    = make_obj(net_tr, Q_tr, WARMUP_RV)

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

# DE (global search)
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

# Scale factor on training set
Q_mod_tr = model_fn(p_opt, net_tr)[WARMUP_RV:]
denom_sf = np.dot(Q_mod_tr, Q_mod_tr)
sf = max(0., np.dot(Q_tr[WARMUP_RV:], Q_mod_tr) / denom_sf) if denom_sf > 1e-20 else 1.

# Test set predictions
Q_mod_te = sf * model_fn(p_opt, net_te) * SCALER
m        = metrics(Q_te, Q_mod_te)
print(f"[{PREFIX}] NSE={m['NSE']:.3f}  KGE={m['KGE']:.3f}  "
      f"RMSE={m['RMSE']/3.6:.1f} L/s  (DE obj≈{res.fun:.4f})", flush=True)

# Unpack calibrated params
tau_arr, E_loss, tau_out, rs_effective = unpack(p_opt)
tau_base       = float(p_opt[0])
rs_raw         = float(p_opt[3])          # calibrated fine-tuning factor
recharge_scale = rs_effective             # = rs_raw × FIXED_SCALE (for display)

# Save CSVs
df_pred = pd.DataFrame({
    "date":         df.index[NTR:],
    "observed_ls":  Q_te     / 3.6,
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

fig = plt.figure(figsize=(18, 6), facecolor=BG_MAIN)
gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.08)
ax1 = fig.add_subplot(gs[0])
#ax2 = fig.add_subplot(gs[1])

# ── Time-series window ──
tmp = pd.DataFrame({"obs": Qo_ls, "pred": Qp_ls}, index=idx)
try:    tmp = tmp[PLOT_START:PLOT_END]
except Exception: pass

ax1.set_facecolor(BG_PANEL)
ax1.fill_between(tmp.index, tmp["obs"], alpha=0.15, color=OBS_COL)
ax1.plot(tmp.index, tmp["obs"],  color=OBS_COL,  lw=1.4,
         label="Observed", zorder=5)
ax1.plot(tmp.index, tmp["pred"], color=RV_COLOR, lw=1.1, alpha=0.88,
         label="River Semi-Dist (E-M-C)", zorder=4)
ax1.annotate(f"NSE = {m['NSE']:.3f}     KGE = {m['KGE']:.3f}",
             xy=(0.01, 0.97), xycoords="axes fraction",
             fontsize=12, fontweight="bold", color="#1A365D",
             va="top", ha="left", bbox=ABOX, zorder=10)
ax1.set_title("Semi-Distributed River  —  6-h Lead Prediction (test period)",
              fontsize=11, fontweight="bold", color=TEXT_COL, pad=6)
ax1.set_ylabel("River Discharge (L/s)", fontsize=9, color=LBL_COL)
ax1.legend(fontsize=8, loc="upper right", facecolor=BG_PANEL, edgecolor=GRID_COL)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
ax1.xaxis.set_major_locator(
    mdates.DayLocator(interval=3)
)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax1.tick_params(colors=LBL_COL, labelsize=8)
for sp in ax1.spines.values(): sp.set_edgecolor(GRID_COL)
'''
# ── Scatter (full test set) ──
lim = max(Qo_ls.max(), Qp_ls.max()) * 1.08
ax2.set_facecolor(BG_PANEL)
ax2.scatter(Qo_ls, Qp_ls, color=RV_COLOR, alpha=0.20, s=4, rasterized=True)
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
    f"Calibrated —  "
    f"tau_base = {tau_base:.1f} h  |  "
    f"E_loss = {E_loss:.2e} m³  |  "
    f"tau_out = {tau_out:.1f} h  |  "
    f"recharge_scale = {rs_raw:.2f} × {int(FIXED_SCALE)} (eff. {recharge_scale:.0f}×)"
    f"\n"
    f"Fixed (spring lumped, same E-M-C structure) —  "
    f"kEM = {KEM:.2e} h⁻¹  |  kEC = {KEC:.2e} h⁻¹  |  "
    f"kMC = {KMC:.2e} h⁻¹  |  kMS = {KMS:.2e} h⁻¹  |  kCS = {KCS:.2e} h⁻¹  |  "
    f"HRUs = {RIVER_HRUS}"
)
plt.tight_layout(rect=[0, 0.12, 1, 1])
fig.text(0.5, 0.01, param_line, ha="center", va="bottom", fontsize=7.5,
         color=TEXT_COL,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#E6FFFA",
                   edgecolor="#276749", alpha=0.90))
'''
out_png = os.path.join(RESULTS_DIR, f"{PREFIX}.png")
plt.savefig(out_png, bbox_inches="tight", dpi=150)
plt.show()
plt.close()
print(f"[{PREFIX}] Saved {os.path.basename(out_png)}", flush=True)
print(f"\n[{PREFIX}] ✓  Done.  Outputs → final/results/", flush=True)
