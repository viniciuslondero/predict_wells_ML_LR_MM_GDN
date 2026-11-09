# -*- coding: utf-8 -*-
"""
predict_rhob.py  -  SCRIPT 2 of 2: apply the saved DEN model to new wells
=========================================================================
Uses the model created by train_rhob_model.py.

For each new well:
  * same preparation as training (0.1524 m, Ariri interval, lithology mapping)
  * keeps rows that have lithology, VP and GR (DEN is optional)
  * predicts DEN with XGBoost (+ Gardner as a reference curve)
  * flags samples outside the training range or with an unseen lithology
  * DEN_MERGED = measured DEN where it exists, prediction elsewhere
  * where some measured DEN exists -> blind-test statistics and plots
  * exports one LAS per well + a CSV with everything

Run in Spyder with F5 or cell by cell. Needs rhob_common.py in the same folder.
"""

# %% ===========================================================================
# CONFIG  (edit here)
# ==============================================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()   # project folder (edit if needed)
NEW_LAS_DIR = BASE_DIR / "PocosCampos"           # .las files of the wells to predict
MODEL_DIR = BASE_DIR / "model_output" / "model"    # created by train_rhob_model.py
OUT_DIR = BASE_DIR / "predictions"
CODE_DIR = None   # folder with rhob_common.py; None = same folder as this script

PREDICT_WELLS = None   # None = every .las in NEW_LAS_DIR, or a list of well names

# Ariri Fm. top/base (MD, m) per well - fill in later (None = not cut)
ARIRI_MARKERS = {
    # 'WELL-NAME': {'top': None, 'base': None},
}

# Model used to fill the gaps in DEN_MERGED (measured DEN where it exists):
#   'auto' -> the model with the lowest blind MAE on the new wells
#             (if no new well has measured DEN: best LOWO model of the training)
#   or a fixed name: 'XGBOOST', 'XGB_NO_LITHO', 'LINREG_LITHO', 'MINERAL_MIX', 'GARDNER'
MERGE_MODEL = 'auto'

SHOW_FIGS = True
SAVE_FIGS = True
PLOT_PER_WELL = True
LITO_ALPHA = 0.25      # transparency of the lithology shading behind the depth tracks
PLOT_EXPLORATION = True       # scatter matrices, histograms, VP x DEN, box plots, proportions
PAIRPLOT_MAX_PER_LITHO = 3000  # points per lithology / well in the scatter plots (speed)
POINT_SIZE = 12
TOP_LITHO = 'HALITE'          # lithology drawn on top in scatter plots (None = random order)


# %% ===========================================================================
# Imports and model loading
# ==============================================================================
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

_code_dir = CODE_DIR or (Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd())
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))
import rhob_common as rc   # noqa: E402

mpl.rcParams['figure.dpi'] = 110
mpl.rcParams['figure.max_open_warning'] = 0
sns.set_style("whitegrid")
FIG_DIR = OUT_DIR / "figures"
LAS_OUT = OUT_DIR / "las"
for d in (OUT_DIR, FIG_DIR, LAS_OUT):
    d.mkdir(parents=True, exist_ok=True)


def fin(fig, name):
    rc.finish_fig(fig, name, FIG_DIR, show=SHOW_FIGS, save=SAVE_FIGS)


with open(MODEL_DIR / "rhob_model_meta.json", encoding='utf-8') as fh:
    meta = json.load(fh)
model = xgb.XGBRegressor()
model.load_model(str(MODEL_DIR / "rhob_xgb_model.json"))
NL = meta.get('xgb_no_litho')                 # XGBoost with GR + VP only
model_nl = None
if NL and (MODEL_DIR / NL['model_file']).exists():
    model_nl = xgb.XGBRegressor()
    model_nl.load_model(str(MODEL_DIR / NL['model_file']))

NUM_FEATURES = meta['num_features']
USE_LITHOLOGY = meta['use_lithology']
CATEGORIES = meta['lito_categories']
FEATURE_COLS = meta['feature_columns']
RANGES = meta['training_ranges_P0.5_P99.5']
GA, GB = meta['gardner_all_wells']['a'], meta['gardner_all_wells']['b']
VALID_RANGES = {k: tuple(v) for k, v in (meta.get('valid_ranges') or {}).items()}  # same limits as training
LINREG = meta.get('linreg_litho', {})
USE_FILE_LITHOLOGY = meta.get('use_file_lithology', True)      # same choice as in training
if abs(meta['step_m'] - rc.STEP) > 1e-6:
    raise ValueError("Model was trained with a different sampling step.")

print(f"Model trained on {meta['training_wells']} ({meta['n_training_rows']} rows) - {meta['created']}")
print(f"Features: {FEATURE_COLS}")
print(f"Training LOWO MAE (XGBoost): {meta['lowo_overall']['XGBOOST']['MAE']:.4f} g/cc")


# %% ===========================================================================
# 1. Prepare new wells (rows with LITO + VP + GR)
# ==============================================================================
files = rc.list_las_files(NEW_LAS_DIR, PREDICT_WELLS)
if not files:
    raise FileNotFoundError(f"No .las files in '{NEW_LAS_DIR}'.")

frames, report, skipped = [], [], {}
all_rows = {}   # every row in the interval, incl. those without VP etc. (LAS export only)
for f in files:
    print(f"Reading {os.path.basename(f)} ...")
    try:
        d, info, d_all = rc.prepare_well(f, ARIRI_MARKERS, required=['LITO', 'VP', 'GR'],
                                         valid_ranges=VALID_RANGES, use_file_lithology=USE_FILE_LITHOLOGY,
                                         return_all=True)
    except Exception as e:                       # unreadable / broken file -> skip it, keep going
        skipped[os.path.basename(f)] = str(e).splitlines()[0][:200]
        print(f"  !! SKIPPED: {skipped[os.path.basename(f)]}")
        continue
    if d.empty:
        skipped[os.path.basename(f)] = ("no rows with LITO + VP + GR (check curve names in "
                                        "rhob_common.CURVES, the Ariri markers and the units)")
        print(f"  !! SKIPPED: {skipped[os.path.basename(f)]}")
        continue
    if info['WELL'] in meta['training_wells']:
        print(f"  NOTE: {info['WELL']} is a TRAINING well -> predicted in-sample, excluded from the blind test.")
    if info.get('read_note', 'ok') != 'ok':
        print(f"  note: {info['read_note']}")
    frames.append(d)
    report.append(info)
    all_rows[info['WELL']] = d_all

if skipped:
    rc.show_table(pd.Series(skipped, name='reason').to_frame(), "Files SKIPPED (could not be read)")
if not frames:
    raise ValueError("None of the LAS files could be read.")
prep_report = pd.DataFrame(report).set_index('WELL')
rc.show_table(prep_report, "Preparation report", "{:.4f}")
pdata = pd.concat(frames, ignore_index=True)
if pdata.empty:
    raise ValueError("No rows with LITO + VP + GR in the new wells.")

# Same GR normalisation as the training wells (shift to the training halite reference)
gr_shift_table = pd.DataFrame()
if meta.get('gr_normalize'):
    pdata, _, gr_shift_table = rc.normalize_gr(pdata, ref=meta['gr_reference_api'],
                                               min_dist=max(meta.get('boundary_buffer_m', 1.0), 0.5))
    rc.show_table(gr_shift_table, f"GR normalisation to training reference "
                                  f"({meta['gr_reference_api']:.2f} API)", "{:.2f}")


# %% ===========================================================================
# 1b. Data exploration of the new wells (same figures as the training script)
#     DEN-based plots use only the samples where density was measured.
# ==============================================================================
if PLOT_EXPLORATION:
    LOGS = ['GR', 'VP', 'DEN']
    PAL = rc.UPSCALE_COLORS
    lito_present_new = [c for c in CATEGORIES if c in pdata['LITO_UPSCALED'].unique()]
    wells_new = sorted(pdata['WELL'].unique())
    n_den = int(pdata['DEN'].notna().sum())
    print(f"\nExploration plots: {len(pdata)} samples, {n_den} with measured DEN.")

    fig = rc.fig_scatter_matrix(pdata, 'LITO_UPSCALED', lito_present_new, PAL,
                                f"New wells - scatter-plot matrix (max {PAIRPLOT_MAX_PER_LITHO} pts per "
                                f"lithology; DEN only where measured)",
                                n_max=PAIRPLOT_MAX_PER_LITHO, point_size=POINT_SIZE, top=TOP_LITHO)
    fin(fig, "01_scatter_matrix")

    fig = rc.fig_scatter_matrix(pdata, 'WELL', wells_new, 'tab10',
                                "New wells - scatter-plot matrix coloured by well",
                                n_max=PAIRPLOT_MAX_PER_LITHO, point_size=POINT_SIZE)
    fin(fig, "01b_scatter_matrix_by_well")

    fig = rc.fig_histograms(pdata, lito_present_new, PAL, LOGS)
    fin(fig, "02_histograms_by_lithology")

    if n_den >= 3:
        fig, _ = rc.fig_vp_den(pdata, lito_present_new, PAL, gardner=(GA, GB),
                               gardner_label='Gardner (training wells, used by the model)',
                               n_max=PAIRPLOT_MAX_PER_LITHO, point_size=POINT_SIZE, top=TOP_LITHO)
        fin(fig, "02b_vp_den_gardner_minerals")

    fig = rc.fig_boxplots(pdata, lito_present_new, LOGS)
    fin(fig, "02c_boxplots_lithology_well")

    lito_prop_new = rc.lithology_proportions(pdata, lito_present_new)
    rc.show_table(lito_prop_new, "Lithology proportion per new well (%)", "{:.1f}")
    fig = rc.fig_lito_proportions(lito_prop_new, PAL, 'New wells - lithology proportion (rows used)')
    fin(fig, "02d_lithology_proportions")


# %% ===========================================================================
# 2. Predict
# ==============================================================================
# Four models from train_rhob_model.py, fitted on all training wells.
# Fallbacks are the same as in the LOWO comparison.
X = rc.build_X(pdata, NUM_FEATURES, USE_LITHOLOGY, CATEGORIES)[FEATURE_COLS]

# XGBOOST
pdata['DEN_PRED'] = model.predict(X)

# GARDNER: DEN = a * VP^b
pdata['DEN_GARDNER'] = GA * pdata['VP'] ** GB

# LINREG_LITHO: one equation per lithology (global equation if lithology not fitted)
LINREG_GLOBAL = meta.get('linreg_global')
pdata['DEN_LINREG'] = np.nan
for l, g in pdata.groupby('LITO_UPSCALED'):
    eq = LINREG.get(l, LINREG_GLOBAL)
    if eq:
        pdata.loc[g.index, 'DEN_LINREG'] = eq['intercept'] + sum(eq['coef'][f] * g[f] for f in NUM_FEATURES)

# MINERAL_MIX: mineral fractions from VP + GR, then DEN = sum(fraction * mineral density)
MM = meta.get('mineral_mix')
MINERAL_NAMES = list(MM['minerals']) if MM else []
MINERAL_COLORS = {'Halite': '#BDBDBD', 'Anhydrite': '#FC349A', 'Sylvite': '#E31A1C',
                  'Carnallite': '#B000FF', 'Tachyhydrite': '#6A3D9A', 'Calcite': '#1F78B4',
                  'Dolomite': '#A6CEE3', 'Clay': '#33A02C'}
if MM:
    fr = rc.mineral_fractions(pdata, minerals=MM['minerals'], sets=MM['mineral_sets'],
                              gr_clean=MM['gr_clean'], sigma=MM['sigma'],
                              non_radioactive=MM['non_radioactive'])
    pdata['DEN_MINMIX'] = rc.mineral_mix_density(fr, MM['rho_fitted'])
    for m in MINERAL_NAMES:
        pdata[f'V_{m.upper()}'] = fr[m]
    pdata['MIX_MISFIT'] = fr['MIX_MISFIT']
else:
    warnings.warn("Model metadata has no MINERAL_MIX - rerun train_rhob_model.py to include it.")

# XGB_NO_LITHO: XGBoost on GR + VP only
if model_nl is not None:
    pdata['DEN_PRED_NOLITHO'] = model_nl.predict(pdata[NL['feature_columns']].astype(float))
else:
    warnings.warn("No XGB_NO_LITHO model found - rerun train_rhob_model.py to include it.")

MODELS = {'GARDNER': 'DEN_GARDNER', 'LINREG_LITHO': 'DEN_LINREG', 'MINERAL_MIX': 'DEN_MINMIX',
          'XGB_NO_LITHO': 'DEN_PRED_NOLITHO', 'XGBOOST': 'DEN_PRED'}       # name -> column
if not MM:
    MODELS.pop('MINERAL_MIX')
if model_nl is None:
    MODELS.pop('XGB_NO_LITHO')
MODEL_COLORS = {'GARDNER': '#E69F00', 'LINREG_LITHO': '#56B4E9', 'MINERAL_MIX': '#009E73',
                'XGB_NO_LITHO': '#8C564B', 'XGBOOST': '#D55E00'}
SHORT = {'GARDNER': 'GARDNER', 'LINREG_LITHO': 'LINREG', 'MINERAL_MIX': 'MINMIX',
         'XGB_NO_LITHO': 'XGB-noLith', 'XGBOOST': 'XGB'}
pdata['DEN_SOURCE'] = np.where(pdata['DEN'].notna(), 1, 2)    # 1 = measured, 2 = predicted

# Quality flags: 0 = OK, 1 = feature outside training P0.5-P99.5 for that lithology,
#                2 = lithology not present in training
flag = np.zeros(len(pdata), dtype=int)
for l, g in pdata.groupby('LITO_UPSCALED'):
    if l not in RANGES:
        flag[pdata.index.get_indexer(g.index)] = 2
        continue
    out = np.zeros(len(g), dtype=bool)
    for f in NUM_FEATURES:
        lo, hi = RANGES[l][f]
        out |= (g[f] < lo).to_numpy() | (g[f] > hi).to_numpy()
    flag[pdata.index.get_indexer(g.index)[out]] = 1
pdata['PRED_FLAG'] = flag

summary = pdata.groupby('WELL').agg(
    rows=('MD', 'size'), MD_top=('MD', 'min'), MD_base=('MD', 'max'),
    measured_DEN=('DEN', 'count'),
    DEN_PRED_mean=('DEN_PRED', 'mean'), DEN_PRED_min=('DEN_PRED', 'min'), DEN_PRED_max=('DEN_PRED', 'max'),
    flag_out_of_range=('PRED_FLAG', lambda s: (s == 1).sum()),
    flag_unseen_litho=('PRED_FLAG', lambda s: (s == 2).sum()))
summary['%_flagged'] = (summary['flag_out_of_range'] + summary['flag_unseen_litho']) / summary['rows'] * 100
rc.show_table(summary, "Prediction summary per well", "{:.3f}")

stats_new = pdata.groupby('WELL')[['GR', 'VP', 'DEN'] + list(MODELS.values())].describe(
    percentiles=[.1, .5, .9]).T
rc.show_table(stats_new, "Log statistics of the new wells", "{:.3f}")


# %% ===========================================================================
# 3. Blind test where measured DEN exists
# ==============================================================================
ev = pdata.dropna(subset=['DEN'])
# Training wells are predicted in-sample -> NOT a blind test. They are still
# predicted and exported, but left out of the blind statistics and of the
# automatic DEN_MERGED model choice.
in_sample = sorted(set(ev['WELL']) & set(meta['training_wells']))
if in_sample:
    print(f"\nNOTE: training well(s) {in_sample} excluded from the blind test (their prediction is in-sample).")
    ev = ev[~ev['WELL'].isin(in_sample)]
blind_tables = {}
if len(ev) >= 3:
    # ---- Tables: overall, per well, per lithology - all models
    blind = pd.DataFrame({n: rc.regression_metrics(ev['DEN'], ev[c]) for n, c in MODELS.items()}).T
    rc.show_table(blind, "Blind test - all new wells with measured DEN", "{:.4f}")

    rows = [{'WELL': w, 'MODEL': n, **rc.regression_metrics(g['DEN'], g[c])}
            for w, g in ev.groupby('WELL') if len(g) >= 3 for n, c in MODELS.items()]
    bw_long = pd.DataFrame(rows).set_index(['WELL', 'MODEL'])
    bw_mae = bw_long['MAE'].unstack('MODEL')[list(MODELS)]
    bw_r2 = bw_long['R2'].unstack('MODEL')[list(MODELS)]
    rc.show_table(bw_mae, "Blind test - MAE per well (g/cc)", "{:.4f}")
    rc.show_table(bw_r2, "Blind test - R2 per well", "{:.3f}")

    rows = [{'LITO': l, 'MODEL': n, **rc.regression_metrics(g['DEN'], g[c])}
            for l, g in ev.groupby('LITO_UPSCALED') if len(g) >= 3 for n, c in MODELS.items()]
    bl_long = pd.DataFrame(rows).set_index(['LITO', 'MODEL'])
    bl_mae = bl_long['MAE'].unstack('MODEL')[list(MODELS)]
    bl_bias = bl_long['Bias (pred-meas)'].unstack('MODEL')[list(MODELS)]
    rc.show_table(bl_mae, "Blind test - MAE per lithology (g/cc)", "{:.4f}")
    rc.show_table(bl_bias, "Blind test - bias per lithology (pred - meas, g/cc)", "{:.4f}")

    best = blind['MAE'].idxmin()
    print(f"\nBest model on the new wells: {best} (MAE {blind.loc[best, 'MAE']:.4f} g/cc). "
          f"Training LOWO MAE for comparison: " +
          ", ".join(f"{n} {meta['lowo_overall'][n]['MAE']:.4f}" for n in MODELS if n in meta['lowo_overall']))

    # Side by side: LOWO (training wells) vs blind (new wells)
    lowo_vs_blind = pd.DataFrame({
        'MAE LOWO (training wells)': {n: meta['lowo_overall'].get(n, {}).get('MAE', np.nan) for n in MODELS},
        'MAE blind (new wells)': blind['MAE'],
        'R2 LOWO (training wells)': {n: meta['lowo_overall'].get(n, {}).get('R2', np.nan) for n in MODELS},
        'R2 blind (new wells)': blind['R2']})
    rc.show_table(lowo_vs_blind, "Training-well LOWO vs new-well blind test", "{:.4f}")

    blind_tables = {'blind_overall': blind, 'LOWO_vs_blind': lowo_vs_blind,
                    'blind_MAE_per_well': bw_mae, 'blind_R2_per_well': bw_r2,
                    'blind_well_x_model': bw_long, 'blind_MAE_per_lithology': bl_mae,
                    'blind_bias_per_lithology': bl_bias, 'blind_litho_x_model': bl_long}

    # ---- Crossplots: measured vs predicted, one panel per model
    _allv = pd.concat([ev['DEN']] + [ev[c] for c in MODELS.values()])
    lims = (min(1.5, _allv.quantile(.0005)) - 0.05, max(3.1, _allv.quantile(.9995)) + 0.05)
    lit_present = [c for c in CATEGORIES if c in ev['LITO_UPSCALED'].unique()]
    _nc = 3 if len(MODELS) > 4 else len(MODELS)
    _nr = int(np.ceil(len(MODELS) / _nc))
    fig, axes = plt.subplots(_nr, _nc, figsize=(5.4 * _nc, 5.2 * _nr), sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, (name, col) in zip(axes, MODELS.items()):
        sns.scatterplot(data=rc.plot_order(ev), x='DEN', y=col, hue='LITO_UPSCALED', hue_order=lit_present,
                        palette=rc.UPSCALE_COLORS, s=10, alpha=0.5, edgecolor=None, ax=ax,
                        legend=(ax is axes[_nc - 1]))
        ax.plot(lims, lims, 'k--', lw=1)
        ok = ev[[col, 'DEN']].dropna()
        if len(ok) > 2:
            s_, i_ = np.polyfit(ok['DEN'], ok[col], 1)            # pred = s*meas + i (y on x)
            ax.plot(np.array(lims), s_ * np.array(lims) + i_, color='red', lw=1.5)
        m = blind.loc[name]
        ax.text(0.03, 0.97, f"MAE {m['MAE']:.3f}\nRMSE {m['RMSE']:.3f}\nR² {m['R2']:.3f}\n"
                            f"bias {m['Bias (pred-meas)']:+.3f}\nfit slope {m['Fit slope']:.2f}",
                transform=ax.transAxes, va='top', fontsize=9, bbox=dict(fc='white', alpha=0.8))
        ax.set_title(f"Blind test - {name}", fontweight='bold')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Measured DEN (g/cc)'); ax.set_ylabel('Predicted DEN (g/cc)')
    for ax in axes[len(MODELS):]:
        ax.set_visible(False)
    if axes[_nc - 1].get_legend():
        sns.move_legend(axes[_nc - 1], 'upper left', bbox_to_anchor=(1.02, 1), markerscale=3)
    plt.tight_layout()
    fin(fig, "blind_test_crossplot")

    # ---- MAE per well and per lithology (bars)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    bw_mae.plot(kind='bar', color=[MODEL_COLORS[n] for n in MODELS], ax=axes[0], edgecolor='k', lw=0.3)
    axes[0].set_title('Blind test - MAE per new well', fontweight='bold'); axes[0].set_ylabel('MAE (g/cc)')
    axes[0].tick_params(axis='x', rotation=20)
    bl_mae.loc[[l for l in lit_present if l in bl_mae.index]].plot(
        kind='bar', color=[MODEL_COLORS[n] for n in MODELS], ax=axes[1], edgecolor='k', lw=0.3)
    axes[1].set_title('Blind test - MAE per lithology', fontweight='bold'); axes[1].set_ylabel('MAE (g/cc)')
    axes[1].tick_params(axis='x', rotation=20)
    plt.tight_layout()
    fin(fig, "blind_test_mae_bars")
else:
    blind = None
    print("No measured DEN in the new wells -> no blind test.")


# %% ---- DEN_MERGED: measured DEN where it exists, best model elsewhere
if MERGE_MODEL != 'auto':
    if MERGE_MODEL not in MODELS:
        raise ValueError(f"MERGE_MODEL must be 'auto' or one of {list(MODELS)}")
    merge_model, merge_reason = MERGE_MODEL, "chosen in CONFIG"
elif blind is not None and blind['MAE'].notna().any():
    merge_model = blind['MAE'].idxmin()
    merge_reason = f"lowest blind MAE on the new wells ({blind.loc[merge_model, 'MAE']:.4f} g/cc)"
else:
    lowo = {n: v['MAE'] for n, v in meta['lowo_overall'].items() if n in MODELS}
    merge_model = min(lowo, key=lowo.get)
    merge_reason = f"no measured DEN in new wells -> best training LOWO MAE ({lowo[merge_model]:.4f} g/cc)"

pdata['DEN_MERGED'] = pdata['DEN'].combine_first(pdata[MODELS[merge_model]])
print(f"\nDEN_MERGED = measured DEN + {merge_model} where DEN is missing  [{merge_reason}]")


# %% ===========================================================================
# 4. Plots per well
# ==============================================================================
# One track per model (measured DEN in black + the model), lithology shaded behind
# every track, MAE in each header where measured DEN exists, FLAG track
# (cyan = outside training range, red = unseen lithology), DEN_MERGED and
# mineral fractions at the end.
# Plots and LAS export show the WHOLE interval: rows without a prediction (e.g. VP
# missing) keep their GR / VP / DEN / lithology, model curves are empty there.
lito_code = {l: i for i, l in enumerate(CATEGORIES)}
pdata['LITO_UPS_CODE'] = pdata['LITO_UPSCALED'].map(lito_code)


def full_well(w, wd):
    """Predicted rows of well `w` + the interval rows that were not predicted,
    with the input logs prepared the same way (same GR shift). Plots/export only."""
    g = all_rows.get(w)
    if g is None:
        return wd
    g = g[~g['MD'].round(4).isin(wd['MD'].round(4))]
    g = g[g[['GR', 'VP', 'DEN', 'LITO_CODE']].notna().any(axis=1)].copy()
    if g.empty:
        return wd
    g['GR_RAW'] = g['GR']
    if 'GR_RAW' in wd.columns and len(gr_shift_table) and w in gr_shift_table.index:
        g['GR'] = g['GR'] - gr_shift_table.loc[w, 'shift_subtracted_API']
    g['LITO_UPS_CODE'] = g['LITO_UPSCALED'].map(lito_code)
    g['DEN_MERGED'] = g['DEN']                                   # measured DEN only
    g['DEN_SOURCE'] = np.where(g['DEN'].notna(), 1, np.nan)
    return pd.concat([wd, g], ignore_index=True).sort_values('MD').reset_index(drop=True)


if PLOT_PER_WELL:
    model_tracks = [(c, n, MODEL_COLORS[n]) for n, c in MODELS.items()]
    model_tracks.append(('DEN_MERGED', f'DEN_MERGED\n({SHORT.get(merge_model, merge_model)})', '#000080'))
    for w, wd in pdata.groupby('WELL'):
        mk = ARIRI_MARKERS.get(w, {})
        tag = ""
        if wd['DEN'].notna().sum() >= 3:
            tag = " | IN-SAMPLE (training well)" if w in meta['training_wells'] else " | blind test"
        fig = rc.plot_model_tracks(
            full_well(w, wd), f"{w} - predicted DEN{tag}", models=model_tracks, den_col='DEN',
            lito_alpha=LITO_ALPHA, flag_col='PRED_FLAG',
            stack=(([f'V_{m.upper()}' for m in MINERAL_NAMES],
                    [MINERAL_COLORS.get(m, 'k') for m in MINERAL_NAMES], 'Mineral fractions') if MM else None),
            markers={'Ariri top': mk.get('top'), 'Ariri base': mk.get('base')})
        fin(fig, f"tracks_{w}")

# Distribution of predicted DEN per lithology vs training mean
fig, ax = plt.subplots(figsize=(11, 5))
lit_present = [c for c in CATEGORIES if c in pdata['LITO_UPSCALED'].unique()]
sns.boxplot(data=pdata, x='LITO_UPSCALED', y='DEN_PRED', hue='WELL', order=lit_present,
            showfliers=False, palette='tab10', ax=ax)
for i, l in enumerate(lit_present):
    if l in meta['litho_mean_den']:
        ax.plot([i - 0.45, i + 0.45], [meta['litho_mean_den'][l]] * 2, 'k--', lw=1.5)
ax.set_title("Predicted DEN per lithology (dashed = training-wells mean)", fontweight='bold')
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
plt.tight_layout()
fin(fig, "predicted_den_by_lithology")


# %% ===========================================================================
# 5. Export
# ==============================================================================
las_curves = {
    'GR_RAW': ('GRRAW', 'API', 'Gamma ray as loaded'),
    'GR': ('GR', 'API', 'Gamma ray (halite-normalised, model input)'),
    'VP': ('VP', 'm/s', 'P-wave velocity'),
    'DEN': ('DEN', 'g/cc', 'Measured density'),
    'DEN_PRED': ('DENPRED', 'g/cc', 'XGBoost predicted density (GR+VP+lithology)'),
    'DEN_PRED_NOLITHO': ('DENPREDNL', 'g/cc', 'XGBoost predicted density (GR+VP only)'),
    'DEN_GARDNER': ('DENGARD', 'g/cc', 'Gardner density (reference)'),
    'DEN_LINREG': ('DENLINREG', 'g/cc', 'Linear regression per lithology (reference)'),
    'DEN_MINMIX': ('DENMINMIX', 'g/cc', 'Mineral mixing model density'),
    **{f'V_{m.upper()}': (f'V{m.upper()[:6]}', 'v/v', f'{m} fraction (mineral mixing model)')
       for m in MINERAL_NAMES},
    'MIX_MISFIT': ('MIXMISFIT', '', 'Mineral mixing misfit (weighted, 0 = exact)'),
    'DEN_MERGED': ('DENMERGED', 'g/cc', f'Measured DEN, {merge_model} where missing'),
    'DEN_SOURCE': ('DENSRC', '', f'1=measured 2={merge_model}'),
    'PRED_FLAG': ('PREDFLAG', '', '0=ok 1=out of training range 2=unseen lithology'),
    'LITO_CODE': ('LITO', '', 'Original lithology code'),
    'LITO_UPS_CODE': ('LITOUPS', '', 'Upscaled lithology code (see ~Other)'),
}
other = ("LITOUPS codes: " + ", ".join(f"{v}={k}" for k, v in lito_code.items()) +
         f"\nModels trained on {', '.join(meta['training_wells'])}; XGBoost inputs: {', '.join(FEATURE_COLS)}"
         f"\nDENMERGED gaps filled with {merge_model} ({merge_reason})"
         f"\nSampling 0.1524 m, continuous. Depths without LITO+VP+GR have no prediction: "
         f"their input logs (GR, VP, DEN, LITO) are kept, DENMERGED = measured DEN if any, "
         f"everything else -9999.25.")

merge_info = pd.DataFrame({'value': {'DEN_MERGED model': merge_model, 'reason': merge_reason}})
blind_tables['DEN_MERGED_choice'] = merge_info

for w, wd in pdata.groupby('WELL'):
    p = LAS_OUT / f"{rc.safe_name(w)}_DEN_pred.las"
    rc.export_las(full_well(w, wd), p, las_curves, w, other)
    print(f"  saved {p}")

pdata.to_csv(OUT_DIR / "all_wells_DEN_predicted.csv", index=False)
rc.save_tables({'summary': summary, 'log_stats': stats_new, 'preparation': prep_report,
                'GR_normalisation': gr_shift_table,
                **({'litho_proportion_%': lito_prop_new} if PLOT_EXPLORATION else {}), **blind_tables},
               OUT_DIR / "prediction_summary.xlsx")
print("\nDone.")
if SHOW_FIGS:
    plt.show()   # keep all figure windows open until they are closed
