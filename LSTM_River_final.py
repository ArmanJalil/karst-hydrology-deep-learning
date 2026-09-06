"""
LSTM_River_final.py  —  Foux Karst System
==========================================
Standard 3-layer stacked LSTM for River discharge prediction (6-h ahead).

Physics prior feature:
  Q_river_lumped_lag6  — output of river_lumped_final.py for the full
  period, joined as an additional input feature. Gives the LSTM a
  physics-consistent estimate of current hydrological state.

Prerequisites:
  Run river_lumped_final.py first so that
  final/results/river_lumped_full_predictions.csv exists.

Architecture (FouxLSTM):
  Layer 1 : LSTM(input_size → 128) + Tanh + Dropout(0.2)
  Layer 2 : LSTM(128 → 64)         + ReLU + Dropout(0.2)
  Layer 3 : LSTM(64  → 32)         + Tanh
  Output  : Linear(32 → 1)

Settings:
  Lookback : 48 h  |  Prediction lag : 6 h ahead
  Split    : 70 % train / 15 % val / 15 % test (temporal)
  Loss     : Huber (delta=1.0, normalised space)
  Device   : CUDA if available, else CPU

Outputs (saved to final/results/ and final/models/):
  LSTM_River_final_pred.csv
  LSTM_River_final_metrics.csv
  LSTM_River_final_history.csv
  LSTM_River_final_loss.png
  LSTM_River_final_scatter.png
  LSTM_River_final_timeseries.png
  models/LSTM_River_final.pth
  models/LSTM_River_final_scaler_X.pkl
  models/LSTM_River_final_scaler_y.pkl
  models/LSTM_River_final_config.json
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
LUMPED_CSV   = os.path.join(PROJECT_DIR, "results", "river_lumped_full_predictions.csv")
RESULTS_DIR  = os.path.join(PROJECT_DIR, "results")
MODELS_DIR   = os.path.join(PROJECT_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

PREFIX = "LSTM_River_final"   # used for all output filenames

# ── Device ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
print(f"[{PREFIX}] PyTorch {torch.__version__}  |  device = {DEVICE}")

# ── Hyper-parameters ─────────────────────────────────────────────────
INPUT_WINDOW = 24
PRED_LAG     = 6
LUMPED_LAG   = 0      # shift applied to lumped feature (0 = no extra shift)
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

MODEL_COLOR = "#276749"   # green for river
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

TARGET_COL = "River_Charge(l/s)"
FOUX_COL   = "Foux_Charge(l/s)"
print(f"[{PREFIX}] Rows={len(df)}  Target: {TARGET_COL}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD RIVER LUMPED MODEL OUTPUTS (physics prior)
# ══════════════════════════════════════════════════════════════════════

print(f"[{PREFIX}] Loading river lumped physics prior …")
if not os.path.exists(LUMPED_CSV):
    raise FileNotFoundError(
        f"'{LUMPED_CSV}' not found.\n"
        "Please run river_lumped_final.py first to generate the physics predictions.")

df_lump = pd.read_csv(LUMPED_CSV, parse_dates=["date"], index_col="date")
df = df.join(df_lump[["Q_river_lumped_ls"]], how="left")
df["Q_river_lumped_ls"] = df["Q_river_lumped_ls"].fillna(0.0)

# Apply lag: shift(LUMPED_LAG) — with LUMPED_LAG=0, value at time t
# is the lumped prediction at t (run_emc already embeds 6-h shift internally)
df["Q_river_lumped_lag6"] = df["Q_river_lumped_ls"].shift(LUMPED_LAG).fillna(0.0)


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
    df[f"rl{lag}"] = df[TARGET_COL].shift(lag).fillna(0)
    df[f"fl{lag}"] = df[FOUX_COL].shift(lag).fillna(0)

df = df.fillna(0.0)

FEATURES = (
    hru_cols
    + basin_cols
    + ["foux_roll6", "rv_roll6"]
    + ["sin_hour", "cos_hour"]
    + [f"rl{l}" for l in [1, 3, 6]]
    + [f"fl{l}" for l in [1, 3, 6]]
    + ["Q_river_lumped_lag6"]
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
# SECTION 7 — MODEL ARCHITECTURE: FouxLSTM
# ══════════════════════════════════════════════════════════════════════

def _xavier_init(module):
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


class FouxLSTM(nn.Module):
    """
    Standard 3-layer stacked LSTM.
    Layer 1 : LSTM(input → 128) + Tanh + Dropout
    Layer 2 : LSTM(128  →  64) + ReLU + Dropout
    Layer 3 : LSTM(64   →  32) + Tanh
    Output  : Linear(32 → 1)
    """
    def __init__(self, input_size):
        super().__init__()
        self.enc1  = nn.LSTM(input_size, HIDDEN1, batch_first=True)
        self.tanh1 = nn.Tanh()
        self.enc2  = nn.LSTM(HIDDEN1,    HIDDEN2, batch_first=True)
        self.relu  = nn.ReLU()
        self.enc3  = nn.LSTM(HIDDEN2,    HIDDEN3, batch_first=True)
        self.tanh2 = nn.Tanh()
        self.drop  = nn.Dropout(DROPOUT)
        self.fc    = nn.Linear(HIDDEN3, 1)
        _xavier_init(self)

    def forward(self, x):
        out, _       = self.enc1(x)
        out          = self.drop(self.tanh1(out))
        out, _       = self.enc2(out)
        out          = self.drop(self.relu(out))
        out, (hn, _) = self.enc3(out)
        out          = self.tanh2(out)
        last_h       = self.drop(hn[-1])
        return self.fc(last_h).squeeze(-1)


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
print(f"[{PREFIX}]  TARGET : River Discharge (L/s)")
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
print(f"\n  ── LSTM ──")
model = FouxLSTM(NF).to(DEVICE)
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
cfg = {"target": "River Q", "model": "LSTM", "physics_prior": "Q_river_lumped_lag6",
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
# SECTION 12 — COMBINED PLOT (time series + scatter + loss in one figure)
# ══════════════════════════════════════════════════════════════════════
# Layout mirrors river_lumped_final.py:
#   Top row (tall)    : time-series window (75%) + scatter full test set (25%)
#   Bottom row (short): training loss curve (75%), right panel hidden
#   Annotation bar at very bottom: test metrics text

ABOX_TS = dict(boxstyle="round,pad=0.35", facecolor="#E6FFFA",
               edgecolor="#276749", alpha=0.92)

metrics_line = (
    f"Test metrics —  "
    f"NSE = {m['NSE']:.3f}  |  KGE = {m['KGE']:.3f}  |  "
    f"RMSE = {m['RMSE']:.1f} L/s  |  MAE = {m['MAE']:.1f} L/s  |  "
    f"R² = {m['R2']:.3f}  |  Epochs trained = {len(df_hist)}"
)

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor(BG_MAIN)
gs  = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1],
                        width_ratios=[3, 1], hspace=0.45, wspace=0.12)
ax_ts   = fig.add_subplot(gs[0, 0])
'''ax_sc   = fig.add_subplot(gs[0, 1])
ax_loss = fig.add_subplot(gs[1, 0])
ax_null = fig.add_subplot(gs[1, 1])
ax_null.set_visible(False)'''

# ── Time series panel ──
df_te_idx = pd.DataFrame({"observed": obs, "predicted": pred}, index=test_timestamps)
try:
    window = df_te_idx[PLOT_START:PLOT_END]
except Exception:
    window = df_te_idx

ax_ts.fill_between(window.index, window["observed"], alpha=0.12, color=OBS_COLOR, zorder=2)
ax_ts.plot(window.index, window["observed"],  color=OBS_COLOR,   lw=1.4, label="Observed",      zorder=5)
ax_ts.plot(window.index, window["predicted"], color=MODEL_COLOR, lw=1.2, alpha=0.90,
           label="LSTM (6-h ahead)", zorder=4)
ax_ts.text(0.01, 0.97,
           f"NSE = {m['NSE']:.3f}     KGE = {m['KGE']:.3f}",
           transform=ax_ts.transAxes, fontsize=12, fontweight="bold",
           color="#1A365D", va="top", ha="left", bbox=ABOX_TS, zorder=10)
ax_ts.set_title("LSTM — River Discharge  |  6-h Lead Prediction (test period)",
                fontsize=11, fontweight="bold", color=TEXT_COL)
ax_ts.set_ylabel("River Discharge (L/s)", fontsize=9, color=LBL_COL)
ax_ts.legend(fontsize=9, loc="upper right", facecolor=BG_PANEL, edgecolor=GRID_COL)
ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%d %b '%y"))
ax_ts.xaxis.set_major_locator(
    mdates.DayLocator(interval=3)
)
plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax_ts.tick_params(colors=LBL_COL, labelsize=8)
for sp in ax_ts.spines.values(): sp.set_edgecolor(GRID_COL)
'''
# ── Scatter panel ──
mx = max(obs.max(), pred.max()) * 1.08
ax_sc.scatter(obs, pred, color=MODEL_COLOR, alpha=0.20, s=5, rasterized=True, zorder=3)
ax_sc.plot([0, mx], [0, mx], "k--", lw=0.9, zorder=4)
ax_sc.text(0.05, 0.95,
           f"R²   = {m['R2']:.3f}\nRMSE = {m['RMSE']:.1f} L/s\nMAE  = {m['MAE']:.1f} L/s",
           transform=ax_sc.transAxes, fontsize=8, va="top", ha="left",
           bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                     edgecolor=GRID_COL, alpha=0.9))
ax_sc.set_xlim(0, mx); ax_sc.set_ylim(0, mx)
ax_sc.set_xlabel("Observed (L/s)",  fontsize=9, color=LBL_COL)
ax_sc.set_ylabel("Predicted (L/s)", fontsize=9, color=LBL_COL)
ax_sc.set_title("Scatter — Full Test Set", fontsize=9, color=TEXT_COL, fontweight="bold")
ax_sc.set_aspect("equal", adjustable="box")
ax_sc.tick_params(colors=LBL_COL, labelsize=8)
for sp in ax_sc.spines.values(): sp.set_edgecolor(GRID_COL)

# ── Loss curve panel ──
ax_loss.plot(df_hist["epoch"], df_hist["train_loss"], color=MODEL_COLOR, lw=1.5, label="Train loss")
ax_loss.plot(df_hist["epoch"], df_hist["val_loss"],   color=MODEL_COLOR, lw=1.5,
             ls="--", alpha=0.7, label="Val loss")
ax_loss.set_xlabel("Epoch", fontsize=9, color=LBL_COL)
ax_loss.set_ylabel("Huber Loss", fontsize=9, color=LBL_COL)
ax_loss.set_title("Training History", fontsize=9, color=TEXT_COL, fontweight="bold")
ax_loss.legend(fontsize=8, facecolor=BG_PANEL, edgecolor=GRID_COL)
ax_loss.tick_params(colors=LBL_COL, labelsize=8)
for sp in ax_loss.spines.values(): sp.set_edgecolor(GRID_COL)

plt.tight_layout(rect=[0, 0.07, 1, 1])

fig.text(0.5, 0.01, metrics_line,
         ha="center", va="center", fontsize=8, color=TEXT_COL,
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#E6FFFA",
                   edgecolor="#276749", alpha=0.85))
'''
out_png = os.path.join(RESULTS_DIR, f"{PREFIX}_combined.png")
plt.savefig(out_png, bbox_inches="tight", dpi=150)
plt.show()
plt.close()
print(f"[{PREFIX}] Saved {os.path.basename(out_png)}")

print(f"\n[{PREFIX}] ✓  Done.  Outputs → final/results/ and final/models/")
