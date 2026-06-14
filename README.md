# Karstic System Prediction — Foux Spring & River Discharge

A hybrid modelling framework for **6-hour-ahead discharge prediction** in the Foux karstic system (Southern France). The pipeline couples physics-based lumped/semi-distributed reservoir models with data-driven LSTM and Physics-Informed LSTM (PI-LSTM) networks.

---

## Repository Structure

```
├── spring_lumped_final.py          # Lumped siphon model — Foux spring
├── river_lumped_final.py           # Lumped E-M-C model  — River
├── semidistributed_foux_final.py   # Semi-distributed model — Foux spring
├── semidistributed_river_final.py  # Semi-distributed model — River
├── LSTM_Foux_final.py              # Standard stacked LSTM — Foux spring
├── LSTM_River_final.py             # Standard stacked LSTM — River
├── PI_LSTM_Foux_final.py           # Physics-Informed LSTM — Foux spring
├── PI_LSTM_River_final.py          # Physics-Informed LSTM — River
└── README.md
```

---

## Scientific Background

The **Foux karstic system** is a complex Mediterranean karst aquifer characterised by:
- An intermittent spring (Foux) controlled by a **submerged siphon** — active flow only when the conduit reservoir exceeds a threshold (explains ~97 % zero-discharge observations).
- A perennial **karstic river** fed by the same massif without a siphon threshold.

The dual-outlet structure makes it an ideal benchmark for testing how well physics constraints can be injected into neural sequence models.

---

## Modelling Approach

### 1. Physics baseline — Lumped models

Both outlets are modelled with the **Epikarst–Matrix–Conduit (E-M-C)** reservoir cascade (Geyer *et al.* / Tritz *et al.* style):

```
Precipitation / Net Water
        ↓
  [Epikarst  E]  — overflow threshold E_loss
        ↓
  [Matrix   M] ←→ [Conduit  C]
        ↓               ↓
  [Siphon  S]  (Foux only — threshold S_crest)
        ↓
    Discharge Q
```

Calibration uses a **warm-start hybrid strategy**: L-BFGS-B seeds + Differential Evolution global search on log-scaled k-parameters.

| Model | Script |
|---|---|
| Spring lumped | `spring_lumped_final.py` |
| River lumped  | `river_lumped_final.py`  |

### 2. Semi-distributed extension

The lumped k-rates are **fixed** (transferred from spring calibration) and spatial heterogeneity is introduced through per-HRU routing time constants (τ scaled by relative distance d_rel). Only a small number of spatial parameters are re-calibrated.

| Model | HRUs used | Script |
|---|---|---|
| Spring semi-distributed | HRU 5, 6 | `semidistributed_foux_final.py` |
| River semi-distributed  | HRU 3, 4, 7, 8, 9, 11, 12 | `semidistributed_river_final.py` |

### 3. Standard LSTM

A **3-layer stacked LSTM** takes the lagged lumped-model output as an additional physics-prior feature alongside observed meteorological inputs:

```
Architecture (FouxLSTM):
  Layer 1 : LSTM(input_size → 128) + Tanh  + Dropout(0.2)
  Layer 2 : LSTM(128 → 64)         + ReLU  + Dropout(0.2)
  Layer 3 : LSTM(64  → 32)         + Tanh
  Output  : Linear(32 → 1)

Lookback : 48 h  |  Prediction horizon : 6 h ahead
Loss     : Huber (δ = 1.0, normalised space)
Split    : 70 % train / 15 % val / 15 % test (temporal)
```

| Model | Script |
|---|---|
| Spring LSTM | `LSTM_Foux_final.py` |
| River LSTM  | `LSTM_River_final.py` |

### 4. Physics-Informed LSTM (PI-LSTM)

The PI-LSTM adds a **physics branch** that runs in parallel with the LSTM branch. A learnable gate blends the two outputs adaptively:

```
Architecture (FouxPILSTM):
  LSTM branch   : same 3-layer LSTM as above (128→64→32)
  Physics branch: Linear(input_size → 1) on last time-step
                  Q_phys(t+6) ≈ α·Q(t) + β·P(t) + γ·Q_lumped(t) + b
  Gate          : Sigmoid(Linear(32 → 1))  — learned from LSTM hidden state
  Output        : gate × Q_lstm + (1 – gate) × Q_phys
```

The gate behaviour is interpretable:
- **gate → 1** : LSTM dominates (active spring events, flood peaks)
- **gate → 0** : physics branch dominates (siphon filling, recession, baseflow)

| Model | Script |
|---|---|
| Spring PI-LSTM | `PI_LSTM_Foux_final.py` |
| River PI-LSTM  | `PI_LSTM_River_final.py` |

---

## Data Requirements

| File | Description |
|---|---|
| `variables_foux.csv` | Hourly precipitation per HRU, ERA5-Land PET, spring & river discharge |
| `foux_total_water.csv` | Pre-computed net water (P − ET) per basin |
| `hrusfinalf.shp` | HRU geometries (area-weighted precipitation) |

All paths are configured at the top of each script under **SECTION 1 — CONFIGURATION**.

---

## Execution Order

The LSTM models depend on the lumped model outputs. Run scripts in this order:

```bash
# Step 1 — Physics baselines (independent)
python spring_lumped_final.py
python river_lumped_final.py

# Step 2 — Semi-distributed extensions (independent)
python semidistributed_foux_final.py
python semidistributed_river_final.py

# Step 3 — LSTM models (require Step 1 outputs)
python LSTM_Foux_final.py
python LSTM_River_final.py

# Step 4 — PI-LSTM models (require Step 1 outputs)
python PI_LSTM_Foux_final.py
python PI_LSTM_River_final.py
```

---

## Dependencies

```bash
pip install numpy pandas matplotlib scikit-learn torch geopandas scipy numba
```

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA optional — models fall back to CPU automatically)
- Numba (JIT compilation for reservoir forward-Euler loops in physics models)

---

## Outputs

Each script saves results to `final/results/` and model artefacts to `final/models/`:

| Type | Files produced |
|---|---|
| Predictions | `*_pred.csv` |
| Metrics (NSE, KGE, RMSE, MAE, R²) | `*_metrics.csv` |
| Training history | `*_history.csv` (LSTM/PI-LSTM only) |
| Plots | `*_loss.png`, `*_scatter.png`, `*_timeseries.png` |
| Saved models | `*.pth`, `*_scaler_X.pkl`, `*_scaler_y.pkl`, `*_config.json` |

---

## Key Results Summary

| Model | Target | Notes |
|---|---|---|
| Lumped siphon | Foux spring | Captures 97 % zero-flow periods via threshold physics |
| Lumped E-M-C  | River | Bidirectional M↔C eigendecomposition for recession curves |
| Semi-dist (Foux) | Foux spring | Fixes k-rates; calibrates 4 spatial parameters |
| Semi-dist (River) | River | Fixes k-rates from spring; calibrates 3 spatial parameters |
| LSTM | Both | Physics prior via lagged lumped output as input feature |
| PI-LSTM | Both | Gated fusion — interpretable blend of LSTM and linear physics |

---

## Citation

If you use this code, please cite the associated thesis/report (reference to be added upon publication).

---

## Author

**Arman Jalil Sargaleh**  
MSc Water Science — Aix-Marseille Université  
MSc Environmental Engineering (Land Assessment & Planning)  
