"""
river_lumped_final.py  —  Foux Karst System
=============================================
River discharge prediction using the E-M-C lumped model extracted from
09_karstmod_physics.py.  Only the River Lumped configuration is kept here.

Model structure (Epikarst–Matrix–Conduit, Geyer et al. / Tritz et al. style)
  Input   : area-weighted precipitation [mm/h] from all HRUs + PET [ERA5-Land]
  Chain   : Epikarst (E) → Matrix (M) + Conduit (C) → River discharge
  Output  : River discharge [L/s], 6-h lead prediction

Data sources
  variables_foux.csv  — precipitation per HRU, ERA5 PET, River discharge
  hrusfinalf.shp      — HRU geometries for area-weighted precipitation

Outputs (saved to results/)
  river_lumped_final_pred.csv
  river_lumped_final_metrics.csv
  river_lumped_final.png            ← time-series + scatter in one image
"""

import warnings; warnings.filterwarnings("ignore")
import os, math, gc
import numpy as np, pandas as pd
import geopandas as gpd
from scipy.signal import lfilter
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

PROJECT_DIR = os.environ.get("FOUX_DATA_DIR", "./data")
SHP_PATH = os.path.join(PROJECT_DIR, "study_area", "hrusfinalf.shp")
RAW_CSV     = os.path.join(PROJECT_DIR, "variables_foux.csv")
RESULTS_DIR = os.path.join(PROJECT_DIR, "final", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DT         = 1        # hourly time step [h]
PRED_LAG   = 6        # 6-hour ahead prediction
TRAIN_FRAC = 0.70
WARMUP_RV  = 6 * 30 * 24   # 180 days — slow matrix reservoir needs time to spin up

PLOT_START = "2024-02-01"
PLOT_END   = "2024-03-31 23:00"

# Plot aesthetics
BG_MAIN  = "#F7F9FC"; BG_PANEL = "#FFFFFF"; GRID_COL = "#D1D9E6"
TEXT_COL = "#1A202C"; LBL_COL  = "#4A5568"
RV_COLOR = "#276749"   # green for river
OBS_COL  = "#E53E3E"   # red for observed
ABOX = dict(boxstyle="round,pad=0.35", facecolor="#E6FFFA",
            edgecolor="#276749", alpha=0.92)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": BG_MAIN,
    "axes.facecolor": BG_PANEL,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.color": GRID_COL,
    "grid.linewidth": 0.6,
})

# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════

print("[river_lumped] Loading data …", flush=True)
df = pd.read_csv(RAW_CSV, encoding="latin-1", parse_dates=["date"],
                 index_col="date", dayfirst=False).sort_index().fillna(0)

# River discharge target
RIVER_COL = [c for c in df.columns if "river" in c.lower() and "stationnary" in c.lower()][0]
Q_river   = df[RIVER_COL].values   # [L/s]

# HRU precipitation columns (COMEPHORE, mm/h)
hru_cols = sorted([c for c in df.columns if "comephore_precipitation" in c.lower()])
import re
hru_nums = [int(re.match(r"HRU(\d+)_", c).group(1))
            for c in hru_cols if re.match(r"HRU(\d+)_", c)]

# ERA5-Land potential evapotranspiration [mm/h]
pet_cols = [c for c in df.columns if "potential_evaporation" in c.lower() and "era5" in c.lower()]
PET = df[pet_cols].mean(axis=1).values if pet_cols else np.zeros(len(df))
if len(pet_cols) > 0 and np.nanmean(np.abs(PET)) < 0.1:
    PET *= 1000.0   # unit correction if values look like m/h rather than mm/h

T_total = len(df)
NTR = int(T_total * TRAIN_FRAC)
print(f"[river_lumped] T={T_total}  Train={NTR}  Test={T_total-NTR}", flush=True)

# ── HRU geometries from shapefile (needed for area-weighted precipitation) ──
print("[river_lumped] Loading shapefile …", flush=True)
gdf = gpd.read_file(SHP_PATH).to_crs("EPSG:2154")
gdf["area_m2"] = gdf.geometry.area
gdf = gdf.sort_values("HRU").reset_index(drop=True)
area_map = dict(zip(gdf.HRU, gdf.area_m2))

areas    = np.array([area_map[h] for h in hru_nums])   # all HRU areas [m²]
RA_river = areas.sum()                                  # total river basin area [m²]

# Precipitation matrix: shape (N_HRU, T)
rv_P = np.stack([df[hru_cols[i]].values for i in range(len(hru_nums))])

# Lumped (area-weighted average): shape (1, T)  — single input source
rv_P_lump = (rv_P * areas[:, None]).sum(axis=0, keepdims=True) / RA_river

print(f"[river_lumped] RA_river = {RA_river/1e6:.2f} km²", flush=True)
print(f"[river_lumped] Q_river max = {Q_river.max():.1f} L/s", flush=True)

# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — E-M-C MODEL
# ══════════════════════════════════════════════════════════════════════
# Physical chain (per hourly timestep):
#
#   Precipitation [mm/h] → per-HRU routing reservoir (tau) → area-weighted P_eff
#   P_eff − λ·PET → Epikarst (E) storage
#   E → Matrix (M)  via k_EM  (slow, weeks-to-months timescale)
#   E → Conduit (C) via k_EC  (fast, hours-to-days timescale)
#   M ↔ C via k_MC exchange (pressure gradient)
#   Discharge = k_MS·M + k_CS·C  (baseflow from matrix + fast conduit drainage)
#
# The M-C coupled system is solved analytically via eigendecomposition of the
# transfer matrix to avoid numerical instability in the discrete-time scheme.
#
# Returns Q_shape [mm/h] — scale factor added externally (closed-form LSQ).

def run_emc(P_src, PET_arr, tau_arr, src_areas, RA,
            kEM, kEC, kMS, kCS, kMC, lam, E0, M0, C0):
    """
    E-M-C model.  P_src shape: (N_src, T).
    Returns Q_shape [mm/h] (shape only; scale factor applied outside).
    The returned series is shifted PRED_LAG hours ahead (6-h lead time).
    """
    N, T = P_src.shape

    # 1. Per-source linear routing reservoir: dR/dt = (P - R) / tau
    #    Equivalent to exponential decay with time constant tau [h].
    R = np.zeros((N, T))
    for i in range(N):
        tau_i = max(float(tau_arr[i]), 1.0)
        a_r   = max(0.0, 1.0 - DT / tau_i)
        R[i], _ = lfilter([DT / tau_i], [1.0, -a_r],
                           P_src[i], zi=np.array([P_src[i, 0] * a_r]))
    R = np.maximum(R, 0.0)

    # 2. Area-weighted effective precipitation [mm/h]
    P_eff = (R * src_areas[:, None]).sum(axis=0) / RA

    # 3. Epikarst E: net recharge = max(P_eff − λ·PET, 0)
    #    Solved as a linear reservoir: aE is the discrete decay factor.
    f  = np.maximum(0.0, P_eff - lam * np.abs(PET_arr))
    kE = kEM + kEC
    aE = max(0.0, 1.0 - kE * DT)
    E, _ = lfilter([DT], [1.0, -aE], f, zi=np.array([E0 * aE]))
    E = np.maximum(E, 0.0)

    # 4. Coupled M-C via eigendecomposition of discrete transfer matrix.
    #    Using np.linalg.eigh (symmetric → guaranteed real eigenvalues).
    kM = kMS + kMC;  kC = kCS + kMC
    A  = np.array([[1.0 - kM * DT,  kMC * DT],
                   [kMC * DT,        1.0 - kC * DT]])
    eigvals, V = np.linalg.eigh(A)

    u  = V.T @ np.stack([kEM * DT * E, kEC * DT * E])
    z0 = V.T @ np.array([M0, C0])

    z_list = []
    for j in range(2):
        lj = float(np.clip(eigvals[j], -0.9999, 0.9999))
        zj, _ = lfilter([1.0], [1.0, -lj], u[j], zi=np.array([z0[j] * lj]))
        z_list.append(zj)

    MC = V @ np.stack(z_list)
    M  = np.maximum(MC[0], 0.0)
    C  = np.maximum(MC[1], 0.0)

    # 5. Total drainage [mm/h]
    Q_shape = kMS * M + kCS * C

    # 6. Shift 6 h ahead (prediction lead time)
    Qp = np.full(T, np.nan)
    if PRED_LAG < T:
        Qp[:-PRED_LAG] = Q_shape[PRED_LAG:]
    return Qp


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — METRICS
# ══════════════════════════════════════════════════════════════════════

def nse(o, s):
    d = np.sum((o - o.mean())**2)
    return float(1 - np.sum((o - s)**2) / d) if d > 0 else -np.inf

def kge_fn(o, s):
    r  = np.corrcoef(o, s)[0, 1]
    al = np.std(s)  / (np.std(o)  + 1e-12)
    be = np.mean(s) / (np.mean(o) + 1e-12)
    return float(1 - math.sqrt((r-1)**2 + (al-1)**2 + (be-1)**2))

def compute_metrics(o, s):
    o, s = np.asarray(o), np.asarray(s)
    return {"RMSE": round(float(np.sqrt(mean_squared_error(o, s))), 3),
            "MAE":  round(float(mean_absolute_error(o, s)), 3),
            "R2":   round(float(r2_score(o, s)), 3),
            "NSE":  round(nse(o, s), 3),
            "KGE":  round(kge_fn(o, s), 3)}


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — OBJECTIVE FUNCTION AND CALIBRATION
# ══════════════════════════════════════════════════════════════════════
# Objective: minimise (1 – NSE) after applying an optimal closed-form
# least-squares scale factor.  This decouples the shape calibration
# from magnitude, letting DE focus on the hydrograph timing/dynamics.

# Parameter order: kEM, kEC, kMS, kCS, kMC, lam, tau_base, E0, M0, C0
BOUNDS_RV = [
    (1e-4, 0.50),   # kEM  — epikarst → matrix transfer rate [h⁻¹]
    (1e-4, 0.90),   # kEC  — epikarst → conduit transfer rate [h⁻¹]
    (1e-4, 0.98),   # kMS  — matrix → river (slow baseflow) [h⁻¹]
    (1e-4, 0.98),   # kCS  — conduit → river (fast drainage) [h⁻¹]
    (0.00, 0.30),   # kMC  — matrix–conduit exchange rate [h⁻¹]
    (0.00, 0.95),   # lam  — PET efficiency factor [-]
    (1.0,  500.0),  # tau_base — routing reservoir time constant [h]
    (0.0,  80.0),   # E0   — initial epikarst storage [mm]
    (0.0,  500.0),  # M0   — initial matrix storage [mm]
    (0.0,  100.0),  # C0   — initial conduit storage [mm]
]

def make_obj(P_src_tr, PET_tr, Q_tr, src_areas, RA):
    """Objective: 1 − NSE after optimal LSQ scale factor."""
    dist_rel = np.ones(len(src_areas))   # lumped → single source, no routing spread

    def obj(p):
        kEM, kEC, kMS, kCS, kMC, lam, tau_base = p[:7]
        E0, M0, C0 = p[7], p[8], p[9]
        tau_arr = tau_base * dist_rel

        Qs = run_emc(P_src_tr, PET_tr, tau_arr, src_areas, RA,
                     kEM, kEC, kMS, kCS, kMC, lam, E0, M0, C0)
        mask = ~np.isnan(Qs)
        if mask.sum() < 100: return 1.0
        o, s = Q_tr[mask], Qs[mask]

        denom = np.dot(s, s)
        if denom < 1e-15: return 1.0
        sf  = max(0.0, np.dot(o, s) / denom)
        ss  = sf * s
        d   = np.sum((o - o.mean())**2)
        if d < 1e-12: return 1.0
        return float(np.clip(np.sum((o - ss)**2) / d, -1e6, 1e6))
    return obj


def run_calibration():
    """Calibrate the river lumped model and return predictions + metrics."""
    P_tr   = rv_P_lump[:, :NTR];  P_te   = rv_P_lump[:, NTR:]
    PET_tr = PET[:NTR];            PET_te = PET[NTR:]
    Q_tr   = Q_river[:NTR];        Q_te   = Q_river[NTR:]

    src_areas = np.array([RA_river])   # single lumped source
    dist_rel  = np.ones(1)             # no routing variation for lumped
    tau_arr_fn = lambda p: p[6] * dist_rel

    obj = make_obj(P_tr, PET_tr, Q_tr, src_areas, RA_river)

    print("[river_lumped] Calibrating (DE) …", flush=True)
    res = differential_evolution(
        obj, BOUNDS_RV, seed=42,
        maxiter=300, popsize=10,
        tol=1e-7, mutation=(0.5, 1.5), recombination=0.9,
        polish=True, updating="immediate", workers=1,
        disp=False,
    )
    gc.collect()
    p = res.x

    kEM, kEC, kMS, kCS, kMC, lam, tau_base = p[:7]
    E0, M0, C0 = p[7], p[8], p[9]
    tau_arr    = tau_base * dist_rel

    # Re-compute scale factor on training set
    Qs_tr  = run_emc(P_tr, PET_tr, tau_arr, src_areas, RA_river,
                     kEM, kEC, kMS, kCS, kMC, lam, E0, M0, C0)
    mask_tr = ~np.isnan(Qs_tr)
    o_tr, s_tr = Q_tr[mask_tr], Qs_tr[mask_tr]
    sf = max(0.0, np.dot(o_tr, s_tr) / (np.dot(s_tr, s_tr) + 1e-15))

    # ── Full-period prediction for LSTM physics prior ──
    # Run model on ALL timesteps (train + test) and save to CSV.
    # LSTM files load this as a physics-prior feature (Q_river_lumped_lag6).
    Qs_full = run_emc(rv_P_lump, PET, tau_arr, src_areas, RA_river,
                      kEM, kEC, kMS, kCS, kMC, lam, E0, M0, C0)
    Qp_full = sf * Qs_full
    df_full = pd.DataFrame({"Q_river_lumped_ls": Qp_full}, index=df.index)
    df_full.index.name = "date"
    df_full.to_csv(os.path.join(RESULTS_DIR, "river_lumped_full_predictions.csv"))
    print("[river_lumped] Saved river_lumped_full_predictions.csv", flush=True)

    # Predict on test set
    Qs_te = run_emc(P_te, PET_te, tau_arr, src_areas, RA_river,
                    kEM, kEC, kMS, kCS, kMC, lam, E0, M0, C0)
    Qp_te = sf * Qs_te

    mask_te = ~np.isnan(Qp_te)
    m       = compute_metrics(Q_te[mask_te], Qp_te[mask_te])

    nse_cal = 1.0 - res.fun
    print(f"[river_lumped] NSE_cal={nse_cal:.4f}  "
          f"NSE_test={m['NSE']:.3f}  KGE={m['KGE']:.3f}  RMSE={m['RMSE']:.1f} L/s",
          flush=True)
    print(f"[river_lumped] tau_base={tau_base:.1f}h  kEM={kEM:.4f}  kEC={kEC:.4f}  "
          f"kMS={kMS:.4f}  kCS={kCS:.4f}  kMC={kMC:.4f}  lam={lam:.3f}",
          flush=True)

    idx_te = df.index[NTR:]
    # Return calibrated physical params for plot annotation
    calib = dict(kEM=kEM, kEC=kEC, kMS=kMS, kCS=kCS,
                 kMC=kMC, lam=lam, tau_base=tau_base)
    return Qp_te, Q_te, idx_te, m, nse_cal, calib


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

print("\n" + "="*60, flush=True)
print("[river_lumped]  RIVER LUMPED CALIBRATION", flush=True)
print("="*60, flush=True)

Qp, Qo, idx_te, met, nse_cal, calib = run_calibration()

# Save predictions CSV
df_pred = pd.DataFrame({"observed_ls": Qo, "predicted_ls": Qp}, index=idx_te)
df_pred.index.name = "date"
df_pred.to_csv(os.path.join(RESULTS_DIR, "river_lumped_final_pred.csv"))
print("[river_lumped] Saved river_lumped_final_pred.csv", flush=True)

# Save metrics CSV
pd.DataFrame([{"model": "River Lumped", **met}]).to_csv(
    os.path.join(RESULTS_DIR, "river_lumped_final_metrics.csv"), index=False)
print("[river_lumped] Saved river_lumped_final_metrics.csv", flush=True)
print(f"\n  NSE={met['NSE']:.3f}  KGE={met['KGE']:.3f}  "
      f"RMSE={met['RMSE']:.1f} L/s  R²={met['R2']:.3f}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — COMBINED PLOT (time series + scatter in one figure)
# ══════════════════════════════════════════════════════════════════════
# Layout: left 75% → time-series window; right 25% → scatter of full test set.

# Build calibrated parameter string for annotation under the plot
# Build calibrated parameter string for annotation under the plot
# Build calibrated parameter string for annotation under the plot
param_line = (
    f"Calibrated parameters —  "
    f"kEM = {calib['kEM']:.4f} h⁻¹  |  kEC = {calib['kEC']:.4f} h⁻¹  |  "
    f"kMS = {calib['kMS']:.4f} h⁻¹  |  kCS = {calib['kCS']:.4f} h⁻¹  |  "
    f"kMC = {calib['kMC']:.4f} h⁻¹  |  "
    f"λ (PET eff.) = {calib['lam']:.3f}  |  τ_base = {calib['tau_base']:.1f} h"
)

fig = plt.figure(figsize=(18, 6), facecolor=BG_MAIN)
gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.08)
ax1 = fig.add_subplot(gs[0])
# ── Time series panel ──
df_tmp = pd.DataFrame({"obs": Qo, "pred": Qp}, index=idx_te)
try:
    df_win = df_tmp[PLOT_START:PLOT_END]
except Exception:
    df_win = df_tmp

ax1.set_facecolor(BG_PANEL)
ax1.fill_between(df_win.index, df_win["obs"], alpha=0.15, color=OBS_COL, zorder=2)
ax1.plot(df_win.index, df_win["obs"],  color=OBS_COL,  lw=1.4,
         label="Observed",              zorder=5)
ax1.plot(df_win.index, df_win["pred"], color=RV_COLOR, lw=1.1, alpha=0.88,
         label="River Lumped (E-M-C)", zorder=4)

ax1.annotate(f"NSE = {met['NSE']:.3f}     KGE = {met['KGE']:.3f}",
             xy=(0.01, 0.97), xycoords="axes fraction",
             fontsize=15, fontweight="bold", color="#1A365D",  # ← 12 → 15
             va="top", ha="left", bbox=ABOX, zorder=10)

ax1.set_title("River Lumped E-M-C Model  —  6-h Lead Prediction (test period)",
              fontsize=14, fontweight="bold", color=TEXT_COL, pad=6)  # ← 11 → 14

ax1.set_ylabel("River Discharge (L/s)", fontsize=12, color=LBL_COL)  # ← 9 → 12
ax1.set_xlabel("Date",                  fontsize=12, color=LBL_COL)  # ← missing → added

ax1.legend(fontsize=11, loc="upper right",                            # ← 8 → 11
           facecolor=BG_PANEL, edgecolor=GRID_COL)

ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax1.tick_params(colors=LBL_COL, labelsize=11)                         # ← 8 → 11
for sp in ax1.spines.values(): sp.set_edgecolor(GRID_COL)

# Parameter text bar — centred under the panel
plt.tight_layout(rect=[0, 0.10, 1, 1])
fig.text(0.5, 0.02, param_line,
         ha="center", va="center", fontsize=11, color=TEXT_COL,       # ← 10 → 11
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#E6FFFA",
                   edgecolor="#276749", alpha=0.85))

out_png = os.path.join(RESULTS_DIR, "river_lumped_final.png")
plt.savefig(out_png, bbox_inches="tight", dpi=150)
plt.show()
plt.close()
print(f"[river_lumped] Saved {out_png}", flush=True)
print("\n[river_lumped] ✓  Done.", flush=True)