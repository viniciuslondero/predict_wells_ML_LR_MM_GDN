# -*- coding: utf-8 -*-
"""
train_rhob_model.py  -  SCRIPT 1 of 2: build the DEN (RHOB) model
====================================================================
Formacao Ariri, Bacia de Campos.  Features: GR + VP + lithology  ->  target: DEN

Steps
  1. Load the training wells, resample to 0.1524 m, keep only the Ariri interval
     and only rows that have lithology, DEN, VP and GR.
  2. Statistics table per well (GR, VP, DEN).
  3. Scatter-plot matrix GR x VP x DEN coloured by lithology.
  4. Histograms per log coloured by lithology + extra correlation plots.
  5. Model comparison with leave-one-well-out (LOWO): XGBoost vs. lithology mean,
     Gardner (VP->DEN) and linear regression per lithology.
  6. Train the final XGBoost on all wells and save it for predict_rhob.py.

Run in Spyder with F5, or cell by cell (Ctrl+Enter). Needs rhob_common.py in the
same folder.  Dependencies (Anaconda Prompt):
    pip install lasio xgboost scikit-optimize seaborn openpyxl
"""

# %% ===========================================================================
# CONFIG  (edit here)
# ==============================================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()   # project folder (edit if needed)
TRAIN_LAS_DIR = BASE_DIR / "PocosSantos"     # folder with the training .las files
OUT_DIR = BASE_DIR / "model_output"          # everything produced by this script
CODE_DIR = None   # folder with rhob_common.py; None = same folder as this script

TRAIN_WELLS = ['3-BRSA-944A-RJS', '9-BRSA-716-RJS', '1-BRSA-976-RJS',
               '3-BRSA-883-RJS', '3-BRSA-821-RJS']

# Ariri Fm. top and base (MD, m) for each well.
# A value set to None means that side is not cut (a warning is printed).
ARIRI_MARKERS = {
    '3-BRSA-944A-RJS': {'top': 3444, 'base': 5487},
    '9-BRSA-716-RJS':  {'top': 2818, 'base': 4836},
    '1-BRSA-976-RJS':  {'top': 3228, 'base': 4961},
    '3-BRSA-883-RJS':  {'top': 3198, 'base': 4805},
    '3-BRSA-821-RJS':  {'top': 3118, 'base': 4788},
}

# Lithology source. True: if a LAS has an upscaled lithology text column
# (LITOUPSC, e.g. exported from Colab with your flag-based reclassification),
# use it; elsewhere the numeric LITO code is mapped as before.
# False: always use the LITO code. Saved with the model -> predict_rhob.py uses
# the same choice.
USE_FILE_LITHOLOGY = True

# Physically valid limits: values outside are set to NaN (row then dropped).
# Only removes impossible readings (e.g. DEN = 4.6 g/cc). None = disable.
VALID_RANGES = {'DEN': (1.5, 3.1), 'VP': (2500, 7500)}

# Samples closer than this (m) to a lithology change are NOT used to TRAIN the
# models (lithology log vs. wireline mismatch at contacts / thin beds).
# They are still predicted and still counted in the evaluation. 0 = disable.
BOUNDARY_BUFFER_M = 0.5

# Light GR normalisation: each well is shifted (constant API offset) so that its
# clean-halite median GR equals a common reference. GR_REF = None -> median of
# the training wells' halite medians. The same reference is applied to new wells.
GR_NORMALIZE = True
GR_REF = None

# MINERAL_MIX model (mineral end-members and mineral sets per lithology are in
# rhob_common.py -> MINERALS, MINERAL_SETS). Mineral densities are fine-tuned on
# the training wells within +/- MINERAL_MAX_CHANGE g/cc of the chart values.
MINERAL_CALIBRATE = True
MINERAL_MAX_CHANGE = 0.25

NUM_FEATURES = ['GR', 'VP']   # numeric model inputs (+ lithology one-hot)
USE_LITHOLOGY = True
TARGET = 'DEN'

N_ITER_SEARCH = 25            # hyper-parameter search iterations
NESTED_TUNING = False         # True = re-tune inside every LOWO fold (unbiased, ~5x slower)
RANDOM_STATE = 42

SHOW_FIGS = True              # show figures in Spyder
SAVE_FIGS = True              # save PNGs to OUT_DIR/figures
PLOT_PER_WELL = True
PAIRPLOT_MAX_PER_LITHO = 3000  # points per lithology in the scatter matrix (speed)
POINT_SIZE = 12               # marker size in the lithology scatter plots
TOP_LITHO = 'HALITE'          # lithology drawn on top of the others (None = random order)
LITO_ALPHA = 0.25             # transparency of the lithology shading behind the depth tracks


# %% ===========================================================================
# Imports
# ==============================================================================
import sys
import json
import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, KFold

_code_dir = CODE_DIR or (Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd())
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))
import rhob_common as rc   # noqa: E402

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    HAS_SKOPT = True
except Exception:
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform, loguniform
    HAS_SKOPT = False
    print("scikit-optimize not available -> RandomizedSearchCV will be used.")

mpl.rcParams['figure.dpi'] = 110
mpl.rcParams['figure.max_open_warning'] = 0
sns.set_style("whitegrid")

FIG_DIR = OUT_DIR / "figures"
MODEL_DIR = OUT_DIR / "model"
for d in (OUT_DIR, FIG_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

LOGS = ['GR', 'VP', 'DEN']
PAL = rc.UPSCALE_COLORS


def fin(fig, name):
    rc.finish_fig(fig, name, FIG_DIR, show=SHOW_FIGS, save=SAVE_FIGS)


# %% ===========================================================================
# STEP 1 - Load, resample to 0.1524 m, Ariri cut, complete rows only
# ==============================================================================
if not TRAIN_LAS_DIR.exists():
    raise FileNotFoundError(f"'{TRAIN_LAS_DIR}' not found - edit BASE_DIR / TRAIN_LAS_DIR.")

files = rc.list_las_files(TRAIN_LAS_DIR, TRAIN_WELLS)
print(f"{len(files)} LAS files found for {len(TRAIN_WELLS)} training wells.")

frames, report = [], []
for f in files:
    print(f"Reading {Path(f).name} ...")
    try:
        d, info = rc.prepare_well(f, ARIRI_MARKERS, required=['LITO', 'DEN', 'VP', 'GR'],
                                  valid_ranges=VALID_RANGES, use_file_lithology=USE_FILE_LITHOLOGY)
    except Exception as e:
        # a training well must not be silently dropped -> stop with a clear message
        raise RuntimeError(f"Could not read training well file '{Path(f).name}': {e}") from e
    if info.get('read_note', 'ok') != 'ok':
        print(f"  note: {info['read_note']}")
    if d.empty:
        warnings.warn(f"{info['WELL']}: no rows with LITO + DEN + VP + GR - check curve names, "
                      "Ariri markers and units. This well adds nothing to training.")
    frames.append(d)
    report.append(info)

prep_report = pd.DataFrame(report).set_index('WELL')
rc.show_table(prep_report, "Preparation report (check units, steps and rows)", "{:.4f}")

missing = set(TRAIN_WELLS) - set(prep_report.index)
if missing:
    warnings.warn(f"Wells with no LAS file found (check the file/WELL names): {sorted(missing)}")

data = pd.concat(frames, ignore_index=True)
if data.empty:
    raise ValueError("No rows with LITO + DEN + VP + GR. Check curve names in rc.CURVES.")
wells = sorted(data['WELL'].unique())

# ---- Light GR normalisation (halite baseline shift per well)
gr_ref_used = None
gr_shift_table = pd.DataFrame()
if GR_NORMALIZE:
    data, gr_ref_used, gr_shift_table = rc.normalize_gr(data, ref=GR_REF,
                                                        min_dist=max(BOUNDARY_BUFFER_M, 0.5))
    rc.show_table(gr_shift_table, f"GR normalisation (reference halite GR = {gr_ref_used:.2f} API)",
                  "{:.2f}")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, col, ttl in zip(axes, ['GR_RAW', 'GR'], ['before', 'after']):
        sns.kdeplot(data=data[data['LITO_UPSCALED'] == 'HALITE'], x=col, hue='WELL',
                    palette='tab10', common_norm=False, ax=ax, clip=(-20, 100))
        ax.axvline(gr_ref_used, color='k', ls='--', lw=1)
        ax.set_title(f"Halite GR per well - {ttl} normalisation", fontweight='bold')
        ax.set_xlim(-10, 60); ax.set_xlabel('GR (API)')
    plt.tight_layout()
    fin(fig, "01b_gr_normalisation_halite")

lito_present = [c for c in rc.LITO_CATEGORIES if c in data['LITO_UPSCALED'].unique()]

print(f"\nTraining dataset: {len(data)} rows, {len(wells)} wells")
print("Sampling step check (should all be 0.1524):",
      data.groupby('WELL')['MD'].apply(lambda s: s.diff().mode().iloc[0]).round(4).to_dict())

# Samples used to TRAIN (away from lithology contacts)
data['TRAIN_OK'] = data['DIST_LITHO_BOUNDARY'] > BOUNDARY_BUFFER_M
buf = (data.groupby('WELL')['TRAIN_OK'].agg(total='size', used_for_training='sum'))
buf['excluded_near_contacts_%'] = (1 - buf['used_for_training'] / buf['total']) * 100
buf_lito = data.groupby('LITO_UPSCALED')['TRAIN_OK'].agg(total='size', used_for_training='sum')
buf_lito['excluded_%'] = (1 - buf_lito['used_for_training'] / buf_lito['total']) * 100
rc.show_table(buf, f"Boundary buffer {BOUNDARY_BUFFER_M} m - samples per well", "{:.1f}")
rc.show_table(buf_lito, "Boundary buffer - samples per lithology", "{:.1f}")
if (buf_lito['used_for_training'] < 50).any():
    warnings.warn("Some lithologies have < 50 training samples after the boundary buffer "
                  "(thin beds). Consider a smaller BOUNDARY_BUFFER_M.")

data.to_csv(OUT_DIR / "training_dataset_0.1524m.csv", index=False)


# %% ===========================================================================
# STEP 2 - Statistics per well
# ==============================================================================
def describe_logs(d):
    rows = {}
    for log in LOGS:
        s = d[log].dropna()
        rows[log] = {'count': s.count(), 'mean': s.mean(), 'std': s.std(),
                     'CV %': s.std() / s.mean() * 100 if s.mean() else np.nan,
                     'min': s.min(), 'P10': s.quantile(.10), 'P25': s.quantile(.25),
                     'median': s.median(), 'P75': s.quantile(.75), 'P90': s.quantile(.90),
                     'max': s.max(), 'skewness': s.skew(), 'kurtosis': s.kurt()}
    return pd.DataFrame(rows).T


per_well = {w: describe_logs(g) for w, g in data.groupby('WELL')}
per_well['ALL WELLS'] = describe_logs(data)
stats_well = pd.concat(per_well, names=['WELL', 'LOG'])
rc.show_table(stats_well, "Statistics per well (GR API | VP m/s | DEN g/cc)", "{:.3f}")

# Interval summary
interval = data.groupby('WELL').agg(MD_top=('MD', 'min'), MD_base=('MD', 'max'), rows=('MD', 'size'))
interval['thickness_with_data_m'] = interval['rows'] * rc.STEP
rc.show_table(interval, "Interval used per well", "{:.1f}")

# Lithology proportions per well (%)
lito_prop = pd.crosstab(data['WELL'], data['LITO_UPSCALED'], normalize='index')[lito_present] * 100
lito_prop.loc['ALL WELLS'] = data['LITO_UPSCALED'].value_counts(normalize=True)[lito_present] * 100
rc.show_table(lito_prop, "Lithology proportion per well (%)", "{:.1f}")

# DEN / VP / GR mean and std per well and lithology
stats_well_lito = data.groupby(['WELL', 'LITO_UPSCALED'])[LOGS].agg(['count', 'mean', 'std'])
stats_lito = data.groupby('LITO_UPSCALED')[LOGS].agg(['count', 'mean', 'std', 'min', 'max'])
rc.show_table(stats_lito, "Statistics per lithology (all wells)", "{:.3f}")

rc.save_tables({'stats_per_well': stats_well, 'interval': interval,
                'litho_proportion_%': lito_prop, 'stats_per_lithology': stats_lito,
                'stats_well_x_lithology': stats_well_lito, 'preparation': prep_report,
                'boundary_buffer_well': buf, 'boundary_buffer_litho': buf_lito,
                'GR_normalisation': gr_shift_table},
               OUT_DIR / "01_statistics.xlsx")


# %% ===========================================================================
# STEP 3 - Scatter-plot matrix (GR, VP, DEN) coloured by lithology
# ==============================================================================
fig = rc.fig_scatter_matrix(data, 'LITO_UPSCALED', lito_present, PAL,
                            f"Scatter-plot matrix - all wells (max {PAIRPLOT_MAX_PER_LITHO} pts per lithology)",
                            n_max=PAIRPLOT_MAX_PER_LITHO, point_size=POINT_SIZE, top=TOP_LITHO, seed=RANDOM_STATE)
fin(fig, "03_scatter_matrix")

# Same matrix, coloured by WELL -> checks whether wells overlap (model transferability)
fig = rc.fig_scatter_matrix(data, 'WELL', wells, 'tab10', "Scatter-plot matrix coloured by well",
                            n_max=PAIRPLOT_MAX_PER_LITHO, point_size=POINT_SIZE, seed=RANDOM_STATE)
fin(fig, "03b_scatter_matrix_by_well")


# %% ===========================================================================
# STEP 4 - Histograms per log coloured by lithology (all wells)
# Left: counts (true abundance). Right: density per lithology (shape of each
# class - small classes like K_SALT become visible).
# ==============================================================================
fig = rc.fig_histograms(data, lito_present, PAL, LOGS)
fin(fig, "04_histograms_by_lithology")


# %% Extra 1: VP x DEN crossplot with Gardner fit and mineral reference points
fig, (gard_a, gard_b) = rc.fig_vp_den(data, lito_present, PAL, n_max=PAIRPLOT_MAX_PER_LITHO,
                                      point_size=POINT_SIZE, top=TOP_LITHO, seed=RANDOM_STATE)
fin(fig, "04b_vp_den_gardner_minerals")


# %% Extra 2: Pearson and Spearman correlation per lithology
pairs = [('DEN', 'VP'), ('DEN', 'GR'), ('VP', 'GR')]
corr_rows = {}
for lit, gg in [('ALL', data)] + [(l, data[data['LITO_UPSCALED'] == l]) for l in lito_present]:
    row = {'n': len(gg)}
    for a, b in pairs:
        if len(gg) > 10:
            row[f'Pearson {a}-{b}'] = gg[a].corr(gg[b])
            row[f'Spearman {a}-{b}'] = gg[a].corr(gg[b], method='spearman')
    corr_rows[lit] = row
corr_tab = pd.DataFrame(corr_rows).T
rc.show_table(corr_tab, "Correlation coefficients per lithology", "{:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 0.6 * len(corr_tab) + 2))
for ax, meth in zip(axes, ['Pearson', 'Spearman']):
    cols = [c for c in corr_tab.columns if c.startswith(meth)]
    sns.heatmap(corr_tab[cols].astype(float), annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1,
                ax=ax, cbar=ax is axes[1])
    ax.set_title(f"{meth} correlation by lithology", fontweight='bold')
    ax.set_xticklabels([c.replace(meth + ' ', '') for c in cols], rotation=0)
plt.tight_layout()
fin(fig, "04c_correlation_by_lithology")


# %% Extra 3: box plots per lithology and well (are the wells consistent?)
fig = rc.fig_boxplots(data, lito_present, LOGS)
fin(fig, "04d_boxplots_lithology_well")


# %% Extra 4: lithology proportions per well
fig = rc.fig_lito_proportions(lito_prop, PAL)
fin(fig, "04e_lithology_proportions")


# %% Extra 5: depth tracks of the training data
if PLOT_PER_WELL:
    for w, wd in data.groupby('WELL'):
        mk = ARIRI_MARKERS.get(w, {})
        fig = rc.plot_well_tracks(wd, f"{w} - training data (0.1524 m)",
                                  overlays={'GR': [('GR_RAW', 'grey', ':', 'GR raw')]} if GR_NORMALIZE else None,
                                  markers={'Ariri top': mk.get('top'), 'Ariri base': mk.get('base')})
        fin(fig, f"04f_tracks_{w}")


# %% ===========================================================================
# STEP 5 - Model comparison with leave-one-well-out (LOWO)
# ------------------------------------------------------------------------------
# Each well is predicted by models trained ONLY on the other wells, so the
# scores measure performance on an unseen well (your real use case).
#   LITHO_MEAN    : DEN = mean DEN of the lithology               (baseline)
#   GARDNER       : DEN = a * VP^b fitted on the training wells    (classic)
#   LINREG_LITHO  : DEN = c0 + c1*GR + c2*VP, one fit per lithology
#   XGBOOST       : gradient boosted trees on GR, VP + lithology
# ==============================================================================
X_all = rc.build_X(data, NUM_FEATURES, USE_LITHOLOGY)
y_all = data[TARGET]
groups = data['WELL']
train_ok = data['TRAIN_OK']           # boundary buffer: used for fitting only
FEATURE_COLS = X_all.columns.tolist()


def make_xgb(**p):
    return xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist',
                            random_state=RANDOM_STATE, n_jobs=-1, **p)


def tune_xgb(X, y, grp):
    n_g = grp.nunique()
    cv = LeaveOneGroupOut() if n_g >= 2 else KFold(5, shuffle=False)
    if HAS_SKOPT:
        s = BayesSearchCV(make_xgb(),
                          {'n_estimators': Integer(50, 500),
                           'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                           'max_depth': Integer(2, 8),
                           'subsample': Real(0.5, 1.0),
                           'colsample_bytree': Real(0.6, 1.0),
                           'min_child_weight': Integer(1, 100)},
                          n_iter=N_ITER_SEARCH, cv=cv, scoring='neg_mean_absolute_error',
                          random_state=RANDOM_STATE, n_jobs=1)
    else:
        s = RandomizedSearchCV(make_xgb(),
                               {'n_estimators': randint(50, 500), 'learning_rate': loguniform(0.01, 0.3),
                                'max_depth': randint(2, 9), 'subsample': uniform(0.5, 0.5),
                                'colsample_bytree': uniform(0.6, 0.4), 'min_child_weight': randint(1, 101)},
                               n_iter=N_ITER_SEARCH, cv=cv, scoring='neg_mean_absolute_error',
                               random_state=RANDOM_STATE, n_jobs=1)
    s.fit(X, y, groups=grp if n_g >= 2 else None)
    return dict(s.best_params_), -s.best_score_


MINERAL_NAMES = list(rc.MINERALS)
MINERAL_COLORS = {'Halite': '#BDBDBD', 'Anhydrite': '#FC349A', 'Sylvite': '#E31A1C',
                  'Carnallite': '#B000FF', 'Tachyhydrite': '#6A3D9A', 'Calcite': '#1F78B4',
                  'Dolomite': '#A6CEE3', 'Clay': '#33A02C'}


def fit_baselines(tr):
    """Fit the reference models (all except XGBoost) on a training DataFrame."""
    m = {'litho_mean': tr.groupby('LITO_UPSCALED')[TARGET].mean().to_dict(),
         'global_mean': tr[TARGET].mean()}
    c = np.polyfit(np.log(tr['VP']), np.log(tr[TARGET]), 1)
    m['gardner'] = (np.exp(c[1]), c[0])
    m['linreg'] = {l: LinearRegression().fit(g[NUM_FEATURES], g[TARGET])
                   for l, g in tr.groupby('LITO_UPSCALED') if len(g) >= 20}
    m['linreg_global'] = LinearRegression().fit(tr[NUM_FEATURES], tr[TARGET])
    # MINERAL_MIX: GR of clean halite = GR of the non-radioactive end-members
    hal = tr['LITO_UPSCALED'] == 'HALITE'
    m['gr_clean'] = float(tr.loc[hal, 'GR'].median()) if hal.any() else 0.0
    fr = rc.mineral_fractions(tr, gr_clean=m['gr_clean'])
    if MINERAL_CALIBRATE:
        m['mineral_rho'], m['mineral_present'] = rc.calibrate_mineral_densities(
            fr, tr[TARGET], max_change=MINERAL_MAX_CHANGE)
    else:
        m['mineral_rho'] = {k: v['RHOB'] for k, v in rc.MINERALS.items()}
        m['mineral_present'] = {k: False for k in rc.MINERALS}
    return m


def predict_baselines(m, te):
    """Returns (predictions DataFrame, mineral fractions DataFrame)."""
    out = pd.DataFrame(index=te.index)
    out['LITHO_MEAN'] = te['LITO_UPSCALED'].map(m['litho_mean']).fillna(m['global_mean'])
    a, b = m['gardner']
    out['GARDNER'] = a * te['VP'] ** b
    lr = pd.Series(np.nan, index=te.index)
    for l, g in te.groupby('LITO_UPSCALED'):
        mdl = m['linreg'].get(l, m['linreg_global'])
        lr.loc[g.index] = mdl.predict(g[NUM_FEATURES])
    out['LINREG_LITHO'] = lr
    fr = rc.mineral_fractions(te, gr_clean=m['gr_clean'])
    out['MINERAL_MIX'] = rc.mineral_mix_density(fr, m['mineral_rho'])
    return out, fr


MODELS = ['LITHO_MEAN', 'GARDNER', 'LINREG_LITHO', 'MINERAL_MIX', 'XGB_NO_LITHO', 'XGBOOST']
XGB_MODELS = ['XGB_NO_LITHO', 'XGBOOST']
BASELINES = [m for m in MODELS if m not in XGB_MODELS]
MODEL_COLORS = {'LITHO_MEAN': '#999999', 'GARDNER': '#E69F00', 'LINREG_LITHO': '#56B4E9',
                'MINERAL_MIX': '#009E73', 'XGB_NO_LITHO': '#8C564B', 'XGBOOST': '#D55E00'}

# XGB_NO_LITHO: same XGBoost but only the numeric logs (GR + VP), no lithology
X_nl = rc.build_X(data, NUM_FEATURES, use_lito=False)
FEATURE_COLS_NL = X_nl.columns.tolist()

# ---- Hyper-parameters (tuned with LOWO folds on all training wells; one search per XGBoost model)
print(f"\nTuning XGBoost ({N_ITER_SEARCH} iterations, LOWO folds)... this can take a few minutes.")
best_params, best_cv_mae = tune_xgb(X_all[train_ok], y_all[train_ok], groups[train_ok])
print(f"XGBOOST      best parameters: {best_params}\n             best LOWO MAE during tuning: {best_cv_mae:.4f} g/cc")
print(f"Tuning XGB_NO_LITHO ({N_ITER_SEARCH} iterations)...")
best_params_nl, best_cv_mae_nl = tune_xgb(X_nl[train_ok], y_all[train_ok], groups[train_ok])
print(f"XGB_NO_LITHO best parameters: {best_params_nl}\n             best LOWO MAE during tuning: {best_cv_mae_nl:.4f} g/cc")
if not NESTED_TUNING:
    print("Note: the same folds were used for tuning, so the XGBoost LOWO score is slightly "
          "optimistic. Set NESTED_TUNING = True for a strictly unbiased estimate.")

# %% ---- LOWO loop
preds = pd.DataFrame(index=data.index, columns=MODELS, dtype=float)
fracs = pd.DataFrame(index=data.index, columns=MINERAL_NAMES + ['MIX_MISFIT'], dtype=float)
fold_params = {}
if len(wells) < 2:
    raise ValueError("LOWO needs at least 2 training wells.")

for w in wells:
    te_m = groups == w                       # test: ALL samples of the held-out well
    tr_m = (groups != w) & train_ok          # train: other wells, away from contacts
    base = fit_baselines(data[tr_m])
    bp, fr = predict_baselines(base, data[te_m])
    preds.loc[te_m, BASELINES] = bp[BASELINES].values
    fracs.loc[te_m] = fr[fracs.columns].values
    if NESTED_TUNING:
        p, _ = tune_xgb(X_all[tr_m], y_all[tr_m], groups[tr_m])
        p_nl, _ = tune_xgb(X_nl[tr_m], y_all[tr_m], groups[tr_m])
    else:
        p, p_nl = best_params, best_params_nl
    fold_params[(w, 'XGBOOST')] = p
    fold_params[(w, 'XGB_NO_LITHO')] = p_nl
    mdl = make_xgb(**p).fit(X_all[tr_m], y_all[tr_m])
    preds.loc[te_m, 'XGBOOST'] = mdl.predict(X_all[te_m])
    mdl_nl = make_xgb(**p_nl).fit(X_nl[tr_m], y_all[tr_m])
    preds.loc[te_m, 'XGB_NO_LITHO'] = mdl_nl.predict(X_nl[te_m])
    print(f"  fold {w:18s} done (test rows: {te_m.sum()})")

for mname in MODELS:
    data[f'PRED_{mname}'] = preds[mname]
    data[f'RES_{mname}'] = data[f'PRED_{mname}'] - data[TARGET]
for mn in MINERAL_NAMES:
    data[f'V_{mn.upper()}'] = fracs[mn]          # LOWO mineral fractions (0-1)
data['MIX_MISFIT'] = fracs['MIX_MISFIT']


# %% ---- Comparison statistics
overall = pd.DataFrame({m: rc.regression_metrics(data[TARGET], data[f'PRED_{m}']) for m in MODELS}).T
rc.show_table(overall, "LOWO - overall, ALL samples (all wells pooled)", "{:.4f}")
interior = data[train_ok]
overall_interior = pd.DataFrame({m: rc.regression_metrics(interior[TARGET], interior[f'PRED_{m}'])
                                 for m in MODELS}).T
rc.show_table(overall_interior, f"LOWO - overall, samples > {BOUNDARY_BUFFER_M} m from a lithology contact",
              "{:.4f}")

per_well_rows = []
for w, g in data.groupby('WELL'):
    for m in MODELS:
        per_well_rows.append({'WELL': w, 'MODEL': m, **rc.regression_metrics(g[TARGET], g[f'PRED_{m}'])})
per_well_metrics = pd.DataFrame(per_well_rows).set_index(['WELL', 'MODEL'])
mae_well = per_well_metrics['MAE'].unstack('MODEL')[MODELS]
r2_well = per_well_metrics['R2'].unstack('MODEL')[MODELS]
mae_well.loc['MEAN of wells'] = mae_well.mean()
r2_well.loc['MEAN of wells'] = r2_well.mean()
rc.show_table(mae_well, "LOWO - MAE per test well (g/cc)", "{:.4f}")
rc.show_table(r2_well, "LOWO - R2 per test well", "{:.3f}")

per_lito_rows = []
for l, g in data.groupby('LITO_UPSCALED'):
    for m in MODELS:
        per_lito_rows.append({'LITO': l, 'MODEL': m, **rc.regression_metrics(g[TARGET], g[f'PRED_{m}'])})
per_lito_metrics = pd.DataFrame(per_lito_rows).set_index(['LITO', 'MODEL'])
mae_lito = per_lito_metrics['MAE'].unstack('MODEL')[MODELS]
bias_lito = per_lito_metrics['Bias (pred-meas)'].unstack('MODEL')[MODELS]
rc.show_table(mae_lito, "LOWO - MAE per lithology (g/cc)", "{:.4f}")
rc.show_table(bias_lito, "LOWO - bias per lithology (pred - meas, g/cc)", "{:.4f}")

best_baseline = overall.loc[BASELINES, 'MAE'].idxmin()
_mw = mae_well.drop('MEAN of wells')
lines = []
for xm in XGB_MODELS:
    gain = (1 - overall.loc[xm, 'MAE'] / overall.loc[best_baseline, 'MAE']) * 100
    wins = int((_mw[xm] <= _mw.min(axis=1)).sum())
    lines.append(f"{xm}: MAE = {overall.loc[xm, 'MAE']:.4f} g/cc vs best non-XGBoost model "
                 f"{best_baseline} = {overall.loc[best_baseline, 'MAE']:.4f} g/cc "
                 f"({abs(gain):.1f}% {'better' if gain >= 0 else 'worse'}); "
                 f"best of all models in {wins} of {len(wells)} wells.")
d_lit = overall.loc['XGB_NO_LITHO', 'MAE'] - overall.loc['XGBOOST', 'MAE']
lines.append(f"Effect of the lithology input on XGBoost: MAE {'+' if d_lit >= 0 else ''}{d_lit:.4f} g/cc "
             f"when it is removed ({'lithology helps' if d_lit > 0 else 'lithology does not help'}).")
summary_txt = "\n".join(lines)
print("\n" + summary_txt)

rc.save_tables({'overall': overall, 'overall_away_from_contacts': overall_interior,
                'MAE_per_well': mae_well, 'R2_per_well': r2_well,
                'metrics_well_x_model': per_well_metrics, 'MAE_per_lithology': mae_lito,
                'bias_per_lithology': bias_lito, 'metrics_litho_x_model': per_lito_metrics,
                'xgb_params_per_fold': pd.DataFrame(fold_params).T.rename_axis(['WELL', 'MODEL'])},
               OUT_DIR / "02_model_comparison_LOWO.xlsx")
data.to_csv(OUT_DIR / "training_dataset_with_LOWO_predictions.csv", index=False)


# %% ---- Plot: measured x predicted, one panel per model
_nc = 3 if len(MODELS) > 4 else len(MODELS)
_nr = int(np.ceil(len(MODELS) / _nc))
fig, axes = plt.subplots(_nr, _nc, figsize=(5.4 * _nc, 5.2 * _nr), sharex=True, sharey=True, squeeze=False)
axes = axes.ravel()
# Axis limits include measured AND predicted values (nothing hidden off-plot);
# extreme 0.05% on each side is trimmed and counted in the panel text.
_allv = pd.concat([data[TARGET]] + [data[f'PRED_{m}'] for m in MODELS])
lims = (min(1.5, _allv.quantile(.0005)) - 0.05, max(3.1, _allv.quantile(.9995)) + 0.05)
for ax, m in zip(axes, MODELS):
    n_out = int(((data[f'PRED_{m}'] < lims[0]) | (data[f'PRED_{m}'] > lims[1]) |
                 (data[TARGET] < lims[0]) | (data[TARGET] > lims[1])).sum())
    sns.scatterplot(data=rc.plot_order(data, TOP_LITHO), x=TARGET, y=f'PRED_{m}', hue='LITO_UPSCALED',
                    hue_order=lito_present, palette=PAL, s=POINT_SIZE * 0.6, alpha=0.4, edgecolor=None,
                    ax=ax, legend=(ax is axes[_nc - 1]))
    ax.plot(lims, lims, 'k--', lw=1.2, label='1:1')
    _ok = data[[TARGET, f'PRED_{m}']].dropna()
    s_, i_ = np.polyfit(_ok[TARGET], _ok[f'PRED_{m}'], 1)         # pred = s*meas + i (y on x)
    pp = np.array(lims)
    ax.plot(pp, s_ * pp + i_, color='red', lw=1.5)
    met = overall.loc[m]
    ax.text(0.03, 0.97, f"MAE {met['MAE']:.3f}\nRMSE {met['RMSE']:.3f}\nR² {met['R2']:.3f}\n"
                        f"bias {met['Bias (pred-meas)']:+.3f}\nfit slope {met['Fit slope']:.2f}\n"
                        f"off-plot pts: {n_out}",
            transform=ax.transAxes, va='top', fontsize=9, bbox=dict(fc='white', alpha=0.8))
    ax.set_title(m, fontweight='bold'); ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Measured DEN (g/cc)'); ax.set_ylabel('Predicted DEN (g/cc)')
for ax in axes[len(MODELS):]:
    ax.set_visible(False)
if axes[_nc - 1].get_legend():
    sns.move_legend(axes[_nc - 1], 'upper left', bbox_to_anchor=(1.02, 1), markerscale=4)
fig.suptitle("Leave-one-well-out: measured vs predicted (black dashed = 1:1, red = least-squares fit)", fontweight='bold', y=1.02)
plt.tight_layout()
fin(fig, "05a_measured_vs_predicted_models")

# %% ---- Plot: MAE per well and per lithology
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
mae_well.drop('MEAN of wells').plot(kind='bar', color=[MODEL_COLORS[m] for m in MODELS], ax=axes[0], edgecolor='k', lw=0.3)
axes[0].set_title('MAE per test well (LOWO)', fontweight='bold'); axes[0].set_ylabel('MAE (g/cc)')
axes[0].tick_params(axis='x', rotation=20)
mae_lito.loc[[l for l in lito_present if l in mae_lito.index]].plot(
    kind='bar', color=[MODEL_COLORS[m] for m in MODELS], ax=axes[1], edgecolor='k', lw=0.3)
axes[1].set_title('MAE per lithology (LOWO)', fontweight='bold'); axes[1].set_ylabel('MAE (g/cc)')
axes[1].tick_params(axis='x', rotation=20)
plt.tight_layout()
fin(fig, "05b_mae_per_well_and_lithology")

# %% ---- Plot: residual distributions
res_long = data.melt(id_vars=['LITO_UPSCALED', 'WELL'], value_vars=[f'RES_{m}' for m in MODELS],
                     var_name='MODEL', value_name='Residual')
res_long['MODEL'] = res_long['MODEL'].str.replace('RES_', '')
fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), gridspec_kw={'width_ratios': [1, 1.6]})
for m in MODELS:
    sns.kdeplot(data[f'RES_{m}'], ax=axes[0], color=MODEL_COLORS[m], lw=2, label=m)
axes[0].axvline(0, color='k', lw=1); axes[0].set_xlim(-0.4, 0.4)
axes[0].set_xlabel('Residual pred - meas (g/cc)'); axes[0].legend()
axes[0].set_title('Residual distribution', fontweight='bold')
sns.boxplot(data=res_long, x='LITO_UPSCALED', y='Residual', hue='MODEL', order=lito_present,
            palette=MODEL_COLORS, showfliers=False, ax=axes[1])
axes[1].axhline(0, color='k', lw=1); axes[1].set_title('Residuals per lithology', fontweight='bold')
axes[1].set_ylabel('pred - meas (g/cc)')
plt.tight_layout()
fin(fig, "05c_residuals")

# %% ---- Plot: depth tracks per well with LOWO predictions
# One track per model (measured DEN in black + the model), lithology shaded behind
# every track, MAE of each model in its header, mineral fractions at the end.
if PLOT_PER_WELL:
    model_tracks = [(f'PRED_{m}', m, MODEL_COLORS[m]) for m in MODELS]
    vcols = [f'V_{m.upper()}' for m in MINERAL_NAMES]
    for w, wd in data.groupby('WELL'):
        best_w = mae_well.loc[w, MODELS].astype(float).idxmin()
        fig = rc.plot_model_tracks(
            wd, f"{w} - leave-one-well-out (well not used in training) | best: {best_w}",
            models=model_tracks, den_col=TARGET, lito_alpha=LITO_ALPHA,
            stack=(vcols, [MINERAL_COLORS[m] for m in MINERAL_NAMES], 'Mineral fractions'),
            markers={'Ariri top': ARIRI_MARKERS.get(w, {}).get('top'),
                     'Ariri base': ARIRI_MARKERS.get(w, {}).get('base')})
        fin(fig, f"05d_LOWO_tracks_{w}")


# %% ===========================================================================
# STEP 6 - Final model on ALL training wells + save
# ==============================================================================
final_model = make_xgb(**best_params).fit(X_all[train_ok], y_all[train_ok])
final_model_nl = make_xgb(**best_params_nl).fit(X_nl[train_ok], y_all[train_ok])
final_base = fit_baselines(data[train_ok])
# Per-lithology linear equations of the final LINREG_LITHO (easy to report in a paper)
lin_eq = pd.DataFrame({l: {'intercept': m_.intercept_, **dict(zip([f'coef_{f}' for f in NUM_FEATURES], m_.coef_)),
                           'n_train': int((data[train_ok]['LITO_UPSCALED'] == l).sum())}
                       for l, m_ in final_base['linreg'].items()}).T
rc.show_table(lin_eq, "LINREG_LITHO equations (all wells): DEN = intercept + coef_GR*GR + coef_VP*VP",
              "{:.6g}")
lin_eq.to_csv(OUT_DIR / "03_linreg_litho_equations.csv")

# Mineral end-members: chart values vs. values fitted on the training wells
min_tab = pd.DataFrame({m: {'DT_us_ft': v['DT'],
                            'GR_API': final_base['gr_clean'] if m in rc.NON_RADIOACTIVE else v['GR'],
                            'RHOB_chart': v['RHOB'],
                            'RHOB_fitted': final_base['mineral_rho'][m],
                            'fitted?': final_base['mineral_present'][m]}
                        for m, v in rc.MINERALS.items()}).T
min_tab['change'] = min_tab['RHOB_fitted'].astype(float) - min_tab['RHOB_chart'].astype(float)
rc.show_table(min_tab, "MINERAL_MIX end-members (fitted = calibrated on all training wells)", "{:.3f}")
print("Changes hitting the limit (+/-{:.2f}) mean the chart value or the mineral set "
      "does not fit your data well.".format(MINERAL_MAX_CHANGE))
min_tab.to_csv(OUT_DIR / "03_mineral_mix_endmembers.csv")

# Mean mineral fractions per lithology (LOWO), a sanity check of the mixing model
vcols = [f'V_{m.upper()}' for m in MINERAL_NAMES]
frac_lito = data.groupby('LITO_UPSCALED')[vcols].mean()
frac_lito = frac_lito.loc[:, frac_lito.sum() > 0]
rc.show_table(frac_lito, "Mean mineral fractions per lithology label (LOWO)", "{:.2f}")

imp = pd.Series(final_model.get_booster().get_score(importance_type='gain')).reindex(FEATURE_COLS).fillna(0)
imp = (imp / imp.sum()).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
imp.plot(kind='barh', ax=ax, color='#D55E00')
ax.set_title('XGBoost feature importance (gain, normalised)', fontweight='bold')
plt.tight_layout()
fin(fig, "06_feature_importance")

# Training ranges per lithology (used by predict_rhob.py to flag extrapolation)
ranges = {}
for l, g in data[train_ok].groupby('LITO_UPSCALED'):
    ranges[l] = {f: [float(g[f].quantile(.005)), float(g[f].quantile(.995))] for f in NUM_FEATURES}
    ranges[l]['n'] = int(len(g))

model_path = MODEL_DIR / "rhob_xgb_model.json"
final_model.save_model(str(model_path))
model_path_nl = MODEL_DIR / "rhob_xgb_nolitho_model.json"
final_model_nl.save_model(str(model_path_nl))

meta = {
    'created': datetime.datetime.now().isoformat(timespec='seconds'),
    'target': TARGET,
    'num_features': NUM_FEATURES,
    'use_lithology': USE_LITHOLOGY,
    'lito_categories': rc.LITO_CATEGORIES,
    'feature_columns': FEATURE_COLS,
    'step_m': rc.STEP,
    'training_wells': wells,
    'ariri_markers': ARIRI_MARKERS,
    'n_training_rows': int(train_ok.sum()),
    'valid_ranges': VALID_RANGES,
    'use_file_lithology': USE_FILE_LITHOLOGY,
    'boundary_buffer_m': BOUNDARY_BUFFER_M,
    'gr_normalize': GR_NORMALIZE,
    'gr_reference_api': gr_ref_used,
    'gr_shifts_training_wells': gr_shift_table['shift_subtracted_API'].to_dict() if GR_NORMALIZE else {},
    'linreg_litho': {l: {'intercept': float(m_.intercept_),
                         'coef': dict(zip(NUM_FEATURES, map(float, m_.coef_)))}
                     for l, m_ in final_base['linreg'].items()},
    'best_params': {k: (v.item() if hasattr(v, 'item') else v) for k, v in best_params.items()},
    'xgb_no_litho': {'model_file': model_path_nl.name, 'feature_columns': FEATURE_COLS_NL,
                     'best_params': {k: (v.item() if hasattr(v, 'item') else v)
                                     for k, v in best_params_nl.items()}},
    'lowo_overall': {m: {k: float(v) for k, v in overall.loc[m].items()} for m in MODELS},
    # Reference models fitted exactly like in the LOWO comparison (all wells, training samples)
    'gardner_all_wells': {'a': float(final_base['gardner'][0]), 'b': float(final_base['gardner'][1])},
    'litho_mean_den': {k: float(v) for k, v in final_base['litho_mean'].items()},
    'global_mean_den': float(final_base['global_mean']),
    'linreg_global': {'intercept': float(final_base['linreg_global'].intercept_),
                      'coef': dict(zip(NUM_FEATURES, map(float, final_base['linreg_global'].coef_)))},
    'mineral_mix': {'minerals': rc.MINERALS, 'mineral_sets': rc.MINERAL_SETS,
                    'non_radioactive': rc.NON_RADIOACTIVE, 'sigma': rc.MIX_SIGMA,
                    'gr_clean': float(final_base['gr_clean']),
                    'rho_fitted': final_base['mineral_rho']},
    'training_ranges_P0.5_P99.5': ranges,
    'versions': {'xgboost': xgb.__version__, 'sklearn': sklearn.__version__, 'pandas': pd.__version__},
}
with open(MODEL_DIR / "rhob_model_meta.json", 'w', encoding='utf-8') as fh:
    json.dump(meta, fh, indent=2, ensure_ascii=False)

print(f"\nModel saved:    {model_path}")
print(f"Model saved:    {model_path_nl}  (XGB_NO_LITHO)")
print(f"Metadata saved: {MODEL_DIR / 'rhob_model_meta.json'}")
print(f"Tables/figures: {OUT_DIR}")
print("\n" + summary_txt)

if SHOW_FIGS:
    plt.show()   # keep all figure windows open until they are closed
