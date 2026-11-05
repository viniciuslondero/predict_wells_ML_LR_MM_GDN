# Bulk Density (RHOB/DEN) Prediction from Well Logs

Machine-learning workflow to predict the bulk density log (DEN / RHOB) from other well logs
(gamma ray, P-wave velocity and lithology), and to fill gaps in wells where density was not
recorded. The model is trained on wells that have a measured density log and then applied to new wells.

## Overview

| File | Purpose |
|------|---------|
| `train_rhob_model.py` | Script 1: loads the training wells, runs exploratory analysis, compares models and saves the final one |
| `predict_rhob.py` | Script 2: applies the saved model to new wells and exports the results |
| `rhob_common.py` | Shared helpers: LAS reading, resampling, lithology mapping, feature building, metrics, plots, LAS export |

### Training (`train_rhob_model.py`)

1. Loads the training wells, resamples them to a regular depth step and keeps the interval of interest.
2. Builds per-well statistics tables and exploratory plots (scatter matrices, histograms, crossplots).
3. Compares models with leave-one-well-out (LOWO) cross-validation:
   - XGBoost (GR + VP + lithology)
   - XGBoost without lithology
   - Mean density per lithology
   - Gardner relation (VP to DEN)
   - Linear regression per lithology
   - Mineral mixture model
4. Trains the final XGBoost on all training wells and saves it with its metadata.

### Prediction (`predict_rhob.py`)

1. Applies the same preparation used in training to each new well.
2. Predicts DEN and flags samples outside the training range or with an unseen lithology.
3. Builds `DEN_MERGED`: measured DEN where it exists, predicted DEN elsewhere.
4. If a well has some measured DEN, computes blind-test statistics and plots.
5. Exports one LAS file per well, a CSV with all results, and figures.

## Requirements

- Python 3.11
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`
- `xgboost`, `lasio`, `openpyxl`
- `scikit-optimize` (optional; falls back to a randomized search if missing)

Example setup with conda:

```bash
conda create -n rhob python=3.11
conda activate rhob
pip install numpy pandas matplotlib seaborn scikit-learn scipy xgboost lasio scikit-optimize openpyxl
```

## Usage

1. Put the LAS files of the training wells and of the wells to predict in separate folders.
2. Edit the `CONFIG` block at the top of each script:
   - `BASE_DIR`, the LAS input folders and the output folders
   - `TRAIN_WELLS` and the interval markers (`ARIRI_MARKERS`) for each well
   - Optional settings: valid ranges, normalization, search iterations, plots
3. Run the training script, then the prediction script:

```bash
python train_rhob_model.py
python predict_rhob.py
```

Both scripts can also be run cell by cell in Spyder or VS Code (cells are marked with `# %%`).
`rhob_common.py` must stay in the same folder as the scripts, or be pointed to with `CODE_DIR`.

## Outputs

- `model_output/` (training): saved model, metadata, tables and figures
- `predictions/` (prediction): LAS files with predicted and merged DEN, a results CSV and figures

## Data

Input LAS files and generated outputs are not versioned. The folders holding them are listed in
`.gitignore`, so the data stays local.
