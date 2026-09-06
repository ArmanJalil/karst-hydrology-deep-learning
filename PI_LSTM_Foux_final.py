"""
PI_LSTM_Foux_final.py  —  Foux Karst System
============================================
Physics-Informed LSTM (PI-LSTM) for Foux spring discharge prediction (6-h ahead).

Physics prior feature:
  Q_foux_lumped_lag6  — output of spring_lumped_final.py for the full
  period, joined as an additional input feature AND used directly by the
  physics branch of the PI-LSTM.

Prerequisites:
  Run spring_lumped_final.py first so that
  final/results/spring_lumped_full_predictions.csv exists.

Architecture (FouxPILSTM):
  LSTM branch  : 3-layer stacked LSTM (128→64→32) + Tanh/ReLU/Tanh
  Physics branch: Linear(input_size → 1) on the LAST time-step features.
                  Approximates: Q(t+6) ≈ α·Q(t) + β·P(t) + γ·Q_lumped(t) + b
                  Because the Foux spring has 97% zero observations, the
                  physics branch provides a crucial structural constraint
                  (siphon threshold behaviour) that LSTM alone may miss.
  Gate         : Sigmoid(Linear(32 → 1)) on last LSTM hidden state
  Output       : gate × Q_lstm + (1 – gate) × Q_phys
    gate → 1 : trusts LSTM (active spring flow events)
    gate → 0 : trusts physics (siphon filling / dry periods)

Settings:
  Lookback : 48 h  |  Prediction lag : 6 h ahead
  Split    : 70 % train / 15 % val / 15 % test (temporal)
  Loss     : Huber (delta=1.0, normalised space)
  Device   : CUDA if available, else CPU

Outputs (saved to final/results/ and final/models/):
  PI_LSTM_Foux_final_pred.csv
  PI_LSTM_Foux_final_metrics.csv
  PI_LSTM_Foux_final_history.csv
  PI_LSTM_Foux_final_loss.png
  PI_LSTM_Foux_final_scatter.png
  PI_LSTM_Foux_final_timeseries.png
  models/PI_LSTM_Foux_final.pth
  models/PI_LSTM_Foux_final_scaler_X.pkl
  models/PI_LSTM_Foux_final_scaler_y.pkl
  models/PI_LSTM_Foux_final_config.json
"""

import warnings; warnings.filterwarnings("ignore")
import os, json, pickle, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(42); np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

PROJECT_DIR = os.environ.get("FOUX_DATA_DIR", "./data")
DATA_CSV     = os.path.join(PROJECT_DIR, "foux_total_water.csv")
LUMPED_CSV   = os.path.join(PROJECT_DIR, "results", "spring_lumped_full_predictions.csv")
RESULTS_DIR  = os.path.join(PROJECT_DIR, "results")
MODELS_DIR   = os.path.join(PROJECT_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

PREFIX = "PI_LSTM_Foux_final"   # used for all output filenames

# ── Device ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
print(f"[{PREFIX}] PyTorch {torch.__version__}  |  device = {DEVICE}")

# ── Hyper-parameters ─────────────────────────────────────────────────
INPUT_WINDOW = 48
PRED_LAG     = 6
LUMPED_LAG   = 0      # shift applied to lumped feature (0 = no extra shift;
                      # karst routing in the spring model already imposes delay)
HIDDEN1      = 128
HIDDEN2      = 64
HIDDEN3      = 32
DROPOUT      = 0.2
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 128
EPOCHS       = 200
PATIENCE     = 20
CLIP_GRAD    = 1.0
HUBER_DELTA  = 1.0

PLOT_START = "2024-02-01"
PLOT_END   = "2024-03-31 23:00"

MODEL_COLOR = "#2B6CB0"   # blue — matches spring_lumped_final.py
OBS_COLOR   = "#E53E3E"
BG_MAIN  = "#F7F9FC"; BG_PANEL = "#FFFFFF"
GRID_COL = "#D1D9E6"; TEXT_COL = "#1A202C"; LBL_COL = "#4A5568"
plt.rcParams.update({"font.family": "DejaVu Sans",
                     "figure.facecolor": BG_MAIN, "axes.facecolor": BG_PANEL,
                     "figure.dpi": 150, "axes.grid": True,
                     "grid.color": GRID_COL, "grid.linewidth": 0.6})


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════

print(f"[{PREFIX}] Loading foux_total_water.csv …")
df = pd.read_csv(DATA_CSV, parse_dates=["date"], index_col="date",
                 dayfirst=False).sort_index()
df.columns = ["HRU_4(m3/h)" if c == "(m3/h)" else c for c in df.columns]

TARGET_COL = "Foux_Charge(l/s)"
RIVER_COL  = "River_Charge(l/s)"
print(f"[{PREFIX}] Rows={len(df)}  Target: {TARGET_COL}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD SPRING LUMPED MODEL OUTPUTS (physics prior)
# ══════════════════════════════════════════════════════════════════════
# The physics branch of PI-LSTM uses Q_foux_lumped_lag6 as part of
# the last-timestep feature vector to learn a linear siphon equation:
#   Q(t+6) ≈ α·Q(t) + β·net_water(t) + γ·Q_foux_lumped(t) + bias
# This constraint is especially valuable for the Foux spring where
# 97% of observations are zero (siphon physics dominate dry periods).

print(f"[{PREFIX}] Loading spring lumped physics prior …")
if not os.path.exists(LUMPED_CSV):
    raise FileNotFoundError(
        f"'{LUMPED_CSV}' not found.\n"
        "Please run spring_lumped_final.py first to generate the physics predictions.")

df_lump = pd.read_csv(LUMPED_CSV, parse_dates=["date"], index_col="date")
df = df.join(df_lump[["Q_foux_lumped_ls"]], how="left")
df["Q_foux_lumped_ls"] = df["Q_foux_lumped_ls"].fillna(0.0)

# shift(LUMPED_LAG=0): value at time t is the lumped prediction at t
df["Q_foux_lumped_lag6"] = df["Q_foux_lumped_ls"].shift(LUMPED_LAG).fillna(0.0)


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════

hru_cols   = [c for c in df.columns if c.startswith("HRU_") and "m3/h" in c]
basin_cols = ["Foux_Basin(m3/h)", "River_Basin(m3/h)"]

df["foux_roll6"] = df["Foux_Basin(m3/h)"].rolling(6,  min_periods=1).sum()
df["rv_roll6"]   = df["River_Basin(m3/h)"].rolling(6,  min_periods=1).sum()
df["sin_hour"]   = np.sin(2 * np.pi * df.index.hour / 24)
df["cos_hour"]   = np.cos(2 * np.pi * df.index.hour / 24)

for lag in [1, 3, 6]:
    df[f"rl{lag}"] = df[RIVER_COL].shift(lag).fillna(0)
    df[f"fl{lag}"] = df[TARGET_COL].shift(lag).fillna(0)

df = df.fillna(0.0)

FEATURES = (
    hru_cols
    + basin_cols
    + ["foux_roll6", "rv_roll6"]
    + ["sin_hour", "cos_hour"]
    + [f"rl{l}" for l in [1, 3, 6]]
    + [f"fl{l}" for l in [1, 3, 6]]
    + ["Q_foux_lumped_lag6"]
)
FEATURES = [c for c in FEATURES if c in df.columns]
NF = len(FEATURES)
print(f"[{PREFIX}] Features ({NF}): {FEATURES}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — SEQUENCE SLICING
# ══════════════════════════════════════════════════════════════════════

def make_sequences(F_arr, y_arr, lookback, pred_lag):
    N = len(F_arr) - lookback - pred_lag + 1
    X = np.lib.stride_tricks.as_strided(
            F_arr,
            shape=(N, lookback, F_arr.shape[1]),
            strides=(F_arr.strides[0], F_arr.strides[0], F_arr.strides[1])
        ).copy().astype(np.float32)
    y = y_arr[lookback + pred_lag - 1:
              lookback + pred_lag - 1 + N].astype(np.float32)
    return X, y


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — MODEL ARCHITECTURE: FouxPILSTM
# ══════════════════════════════════════════════════════════════════════

def _xavier_init(module):
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


class FouxPILSTM(nn.Module):
    """
    Physics-Informed LSTM — blends two branches via a learned gate.

    LSTM branch  : 3-layer stacked LSTM (128→64→32) + Tanh/ReLU/Tanh
    Physics branch: Linear(input_size → 1) on the LAST time-step features.
                    Learns: Q(t+6) ≈ α·Q(t) + β·P(t) + γ·Q_lumped(t) + b
    Gate          : Sigmoid(Linear(32 → 1)) on last LSTM hidden state.
                    gate→1: trust LSTM (complex events)
                    gate→0: trust physics (recession, baseflow, dry siphon)

    Output: gate × Q_lstm + (1 – gate) × Q_phys
    """
    def __init__(self, input_size):
        super().__init__()
        # LSTM branch
        self.enc1    = nn.LSTM(input_size, HIDDEN1, batch_first=True)
        self.tanh1   = nn.Tanh()
        self.enc2    = nn.LSTM(HIDDEN1,    HIDDEN2, batch_first=True)
        self.relu    = nn.ReLU()
        self.enc3    = nn.LSTM(HIDDEN2,    HIDDEN3, batch_first=True)
        self.tanh2   = nn.Tanh()
        self.drop    = nn.Dropout(DROPOUT)
        self.fc_lstm = nn.Linear(HIDDEN3, 1)
        # Physics branch: linear combination of last-timestep features
        self.fc_phys = nn.Linear(input_size, 1)
        # Gate: sigmoid on last LSTM hidden state
        self.fc_gate = nn.Linear(HIDDEN3, 1)
        _xavier_init(self)
        nn.init.zeros_(self.fc_gate.bias)   # start at gate=0.5 (equal blend)

    def forward(self, x):
        out, _       = self.enc1(x)
        out          = self.drop(self.tanh1(out))
        out, _       = self.enc2(out)
        out          = self.drop(self.relu(out))
        out, (hn, _) = self.enc3(out)
        out          = self.tanh2(out)
        last_h       = self.drop(hn[-1])              # (B, H3)

        q_lstm = self.fc_lstm(last_h)                 # (B, 1)
        q_phys = self.fc_phys(x[:, -1, :])            # (B, 1) — last timestep
        gate   = torch.sigmoid(self.fc_gate(last_h))  # (B, 1)

        q_out = gate * q_lstm + (1.0 - gate) * q_phys
        return q_out.squeeze(-1)                       # (B,)


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — METRICS
# ══════════════════════════════════════════════════════════════════════

def nse(obs, sim):
    d = np.sum((obs - obs.mean())**2)
    return float(1.0 - np.sum((obs - sim)**2) / d) if d > 0 else np.nan

def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    a = np.std(sim)  / np.std(obs)   if np.std(obs)  > 0 else 0.
    b = np.mean(sim) / np.mean(obs)  if np.mean(obs) > 0 else 0.
    return float(1. - math.sqrt((r-1)**2 + (a-1)**2 + (b-1)**2))

def all_metrics(obs, sim):
    return {"RMSE": float(np.sqrt(mean_squared_error(obs, sim))),
            "MAE":  float(mean_absolute_error(obs, sim)),
            "R2":   float(r2_score(obs, sim)),
            "NSE":  nse(obs, sim),
            "KGE":  kge(obs, sim)}


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5, min_lr=1e-6)
    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    best_val, best_state, wait = np.inf, None, 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tl = 0.
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            tl += loss.item() * len(yb)
        tl /= len(train_loader.dataset)

        model.eval()
        vl = 0.
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl += criterion(model(xb), yb).item() * len(yb)
        vl /= len(val_loader.dataset)

        scheduler.step(vl)
        lr_now = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["lr"].append(lr_now)

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stop @ epoch {epoch}  (best val={best_val:.6f})")
                break

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Ep {epoch:>4d}  train={tl:.5f}  val={vl:.5f}  "
                  f"lr={lr_now:.1e}  ({time.time()-t0:.0f}s)")

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# ══════════════════════════════════════════════════════════════════════
# SECTION 10 — PREDICT
# ══════════════════════════════════════════════════════════════════════

def predict(model, X_arr):
    model.eval()
    ds  = TimeSeriesDataset(X_arr, np.zeros(len(X_arr), np.float32))
    ldr = DataLoader(ds, batch_size=BATCH_SIZE * 8, shuffle=False,
                     pin_memory=torch.cuda.is_available())
    out = []
    with torch.no_grad():
        for xb, _ in ldr:
            out.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(out)


# ══════════════════════════════════════════════════════════════════════
# SECTION 11 — MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"[{PREFIX}]  TARGET : Foux Spring Discharge (L/s)")
print(f"{'='*60}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

raw_X = df[FEATURES].values
raw_y = df[[TARGET_COL]].values.ravel()

# Temporal split
N_seq = len(raw_X) - INPUT_WINDOW - PRED_LAG + 1
NTR   = int(N_seq * 0.70)
NVA   = int(N_seq * 0.85)
N_te  = N_seq - NVA

t_offset        = INPUT_WINDOW + PRED_LAG - 1
test_ts_start   = NVA + t_offset
test_ts_end     = NVA + t_offset + N_te
test_timestamps = df.index[test_ts_start:test_ts_end]
print(f"  Test timestamps: {test_timestamps[0]} → {test_timestamps[-1]}")

# Scalers fit on training rows only (no data leakage)
train_end_row = NTR + t_offset
scaler_X = StandardScaler().fit(raw_X[:train_end_row])
scaler_y = StandardScaler().fit(raw_y[:train_end_row].reshape(-1, 1))

F_sc = scaler_X.transform(raw_X).astype(np.float32)
y_sc = scaler_y.transform(raw_y.reshape(-1, 1)).ravel().astype(np.float32)

X_all, y_all = make_sequences(F_sc, y_sc, INPUT_WINDOW, PRED_LAG)
X_tr, y_tr   = X_all[:NTR],    y_all[:NTR]
X_va, y_va   = X_all[NTR:NVA], y_all[NTR:NVA]
X_te, y_te   = X_all[NVA:],    y_all[NVA:]
print(f"  Seqs: {N_seq}  train={NTR}  val={NVA-NTR}  test={N_te}")

obs = scaler_y.inverse_transform(y_te.reshape(-1, 1)).ravel()

train_ldr = DataLoader(
    TimeSeriesDataset(X_tr, y_tr),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    pin_memory=torch.cuda.is_available(), num_workers=0)
val_ldr = DataLoader(
    TimeSeriesDataset(X_va, y_va),
    batch_size=BATCH_SIZE * 4, shuffle=False,
    pin_memory=torch.cuda.is_available(), num_workers=0)

# Save scalers
with open(os.path.join(MODELS_DIR, f"{PREFIX}_scaler_X.pkl"), "wb") as f:
    pickle.dump(scaler_X, f)
with open(os.path.join(MODELS_DIR, f"{PREFIX}_scaler_y.pkl"), "wb") as f:
    pickle.dump(scaler_y, f)

# ── Train ──
print(f"\n  ── PI-LSTM ──")
model = FouxPILSTM(NF).to(DEVICE)
n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {n_par:,}")

model, df_hist = train_model(model, train_ldr, val_ldr)

pred_sc = predict(model, X_te)
pred    = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).ravel()

m = all_metrics(obs, pred)
print(f"  [TEST] RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  "
      f"R²={m['R2']:.3f}  NSE={m['NSE']:.3f}  KGE={m['KGE']:.3f}")

# ── Save model ──
torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{PREFIX}.pth"))

# ── Save config ──
cfg = {"target": "Foux Q", "model": "PI-LSTM", "physics_prior": "Q_foux_lumped_lag6",
       "INPUT_WINDOW": INPUT_WINDOW, "PRED_LAG": PRED_LAG, "LUMPED_LAG": LUMPED_LAG,
       "NF": NF, "FEATURES": FEATURES,
       "HIDDEN1": HIDDEN1, "HIDDEN2": HIDDEN2, "HIDDEN3": HIDDEN3,
       "DROPOUT": DROPOUT, "HUBER_DELTA": HUBER_DELTA,
       "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
       "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS,
       "PATIENCE": PATIENCE, "CLIP_GRAD": CLIP_GRAD,
       "epochs_trained": len(df_hist), "metrics": m}
with open(os.path.join(RESULTS_DIR, f"{PREFIX}_config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

# ── Save history and predictions ──
df_hist.to_csv(os.path.join(RESULTS_DIR, f"{PREFIX}_history.csv"), index=False)

df_pred_out = pd.DataFrame({
    "date": test_timestamps,
    "observed_ls":  obs,
    "predicted_ls": pred
})
df_pred_out.to_csv(os.path.join(RESULTS_DIR, f"{PREFIX}_pred.csv"), index=False)

pd.DataFrame([{"model": PREFIX, **m}]).to_csv(
    os.path.join(RESULTS_DIR, f"{PREFIX}_metrics.csv"), index=False)
print(f"[{PREFIX}] Saved predictions and metrics CSVs")


# ══════════════════════════════════════════════════════════════════════
# SECTION 12 — COMBINED PLOT (single page)
# ══════════════════════════════════════════════════════════════════════

def _style_ax(ax, title="", xl="", yl=""):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold", pad=5)
    ax.set_xlabel(xl, color=LBL_COL, fontsize=9)
    ax.set_ylabel(yl, color=LBL_COL, fontsize=9)
    ax.tick_params(colors=LBL_COL, labelsize=9)
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)

ABOX_TS = dict(boxstyle="round,pad=0.35", facecolor="#EBF8FF", edgecolor="#3182CE", alpha=0.92)

metrics_line = (f"Test metrics —  NSE = {m['NSE']:.3f}  |  KGE = {m['KGE']:.3f}  |  "
                f"RMSE = {m['RMSE']:.1f} L/s  |  MAE = {m['MAE']:.1f} L/s  |  "
                f"R² = {m['R2']:.3f}  |  Epochs trained = {len(df_hist)}")

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG_MAIN)
gs  = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1],
                        width_ratios=[3, 1], hspace=0.45, wspace=0.12)
ax_ts   = fig.add_subplot(gs[0, 0])
'''ax_sc   = fig.add_subplot(gs[0, 1])
ax_loss = fig.add_subplot(gs[1, 0])
ax_null = fig.add_subplot(gs[1, 1])
ax_null.set_visible(False)'''

# ── Time series (top-left) ──
df_te_idx = pd.DataFrame({"observed": obs, "PI-LSTM": pred}, index=test_timestamps)
try:
    window = df_te_idx[PLOT_START:PLOT_END]
except Exception:
    window = df_te_idx

ax_ts.set_facecolor(BG_PANEL)
ax_ts.fill_between(window.index, window["observed"], alpha=0.18, color=OBS_COLOR)
ax_ts.plot(window.index, window["observed"], color=OBS_COLOR,   lw=1.3, label="Observed", zorder=5)
ax_ts.plot(window.index, window["PI-LSTM"],  color=MODEL_COLOR, lw=1.0, alpha=0.88,
           label="PI-LSTM (6-h ahead)", zorder=4)
ax_ts.set_title("PI-LSTM — Foux Spring Discharge  |  6-h Lead Prediction (test period)",
                color=TEXT_COL, fontsize=11, fontweight="bold", pad=6)
ax_ts.set_ylabel("Foux Spring Discharge (L/s)", color=LBL_COL, fontsize=9)
ax_ts.tick_params(colors=LBL_COL, labelsize=8)
for s in ax_ts.spines.values():
    s.set_edgecolor(GRID_COL)
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
ax_ts.xaxis.set_major_locator(
    mdates.DayLocator(interval=3)
)
plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha="right")
leg = ax_ts.legend(fontsize=8, loc="upper right", facecolor=BG_PANEL, edgecolor=GRID_COL)
for t in leg.get_texts():
    t.set_color(TEXT_COL)
ax_ts.text(
    0.01, 0.97,
    f"NSE = {m['NSE']:.3f}     KGE = {m['KGE']:.3f}",
    transform=ax_ts.transAxes,
    fontsize=12,
    fontweight="bold",
    color="#1A365D",
    va="top",
    ha="left",
    bbox=ABOX_TS,
    zorder=10)
'''
# ── Scatter (top-right) ──
ax_sc.set_facecolor(BG_PANEL)
mx = max(obs.max(), pred.max()) * 1.05
ax_sc.scatter(obs, pred, color=MODEL_COLOR, alpha=0.25, s=4, rasterized=True)
ax_sc.plot([0, mx], [0, mx], "k--", lw=0.8, label="1:1")
ax_sc.set_title("Observed vs Predicted", color=TEXT_COL, fontsize=10, fontweight="bold", pad=5)
ax_sc.set_xlabel("Observed (L/s)", color=LBL_COL, fontsize=9)
ax_sc.set_ylabel("Predicted (L/s)", color=LBL_COL, fontsize=9)
ax_sc.tick_params(colors=LBL_COL, labelsize=8)
for s in ax_sc.spines.values():
    s.set_edgecolor(GRID_COL)
sc_leg = ax_sc.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COL)
for t in sc_leg.get_texts():
    t.set_color(TEXT_COL)
ax_sc.annotate(f"R² = {m['R2']:.3f}\nRMSE = {m['RMSE']:.1f} L/s\nMAE = {m['MAE']:.1f} L/s",
               xy=(0.05, 0.95), xycoords="axes fraction",
               va="top", ha="left", fontsize=8, color=TEXT_COL, bbox=ABOX_TS)

# ── Loss curve (bottom-left) ──
ax_loss.set_facecolor(BG_PANEL)
ax_loss.plot(df_hist["epoch"], df_hist["train_loss"], color=MODEL_COLOR, lw=1.5, label="Train loss")
ax_loss.plot(df_hist["epoch"], df_hist["val_loss"],   color=MODEL_COLOR, lw=1.5,
             ls="--", alpha=0.7, label="Val loss")
ax_loss.set_title("Training History", color=TEXT_COL, fontsize=10, fontweight="bold", pad=5)
ax_loss.set_xlabel("Epoch", color=LBL_COL, fontsize=9)
ax_loss.set_ylabel("Huber Loss", color=LBL_COL, fontsize=9)
ax_loss.tick_params(colors=LBL_COL, labelsize=8)
for s in ax_loss.spines.values():
    s.set_edgecolor(GRID_COL)
loss_leg = ax_loss.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COL)
for t in loss_leg.get_texts():
    t.set_color(TEXT_COL)

plt.tight_layout(rect=[0, 0.07, 1, 1])
fig.text(0.5, 0.01, metrics_line, ha="center", va="bottom", fontsize=9,
         color=TEXT_COL,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#EBF8FF",
                   edgecolor="#3182CE", alpha=0.92))
'''
combined_png = os.path.join(RESULTS_DIR, f"{PREFIX}_combined.png")
plt.savefig(combined_png, bbox_inches="tight", dpi=150)
plt.show()
plt.close()
print(f"  [plot] {os.path.basename(combined_png)}")

print(f"\n[{PREFIX}] ✓  Done.  Outputs → final/results/ and final/models/")
