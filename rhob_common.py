# -*- coding: utf-8 -*-
"""
rhob_common.py - shared functions for the RHOB (DEN) prediction workflow.

Used by BOTH scripts:
    train_rhob_model.py   -> builds, evaluates and saves the model
    predict_rhob.py       -> applies the saved model to new wells

Keeping loading / resampling / lithology mapping here guarantees that the
new wells are prepared exactly like the training wells.
Keep this file in the same folder as the two scripts.
"""
import io
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mpl_ticker

try:
    import lasio
except ImportError as e:
    raise ImportError("lasio not installed. In Anaconda Prompt: pip install lasio") from e


# ==============================================================================
# Constants
# ==============================================================================
STEP = 0.1524          # target sampling step (m) = 0.5 ft
FT2M = 0.3048

# Curve mnemonics searched in each LAS (first one with data wins)
CURVES = {
    'DT':   ['DTP', 'DT', 'DTCO', 'DTC', 'DT4P', 'SONIC', 'AC'],
    'VP':   ['VP', 'VELP', 'PVEL'],                     # used only if no DT exists
    'DEN':  ['RHOB', 'DEN', 'DENS', 'RHOZ', 'RHO8', 'ZDEN'],
    'GR':   ['GR', 'ECGR', 'SGR', 'CGR', 'GRC', 'HSGR'],
    'LITO': ['LITHOLOGYRESAMPLE', 'LITHOLOGY', 'LITO', 'LITH', 'LITOLOGIA'],
}
# Text curves with an already-upscaled lithology (HALITE, ANHYDRITE, K_SALT, ...)
FILE_UPSCALED_CURVES = ['LITOUPSC', 'LITO_UPSCALED', 'LITOUPSCALED']

LITO_TABLE = '''LITO,ABREVIACAO,COR,CODIGO
CALCARIO,CAL,#7FBFFF,0
CALCILUTITO,CLU,#85FCFF,1
CALCARENITO,CRE,#4441D8,2
CHERT,CHT,#FCC13F,3
DOLOMITO,DOL,#0BFAFF,4
VULCANICA NAO IDENTIFICADA,VNI,#FF7F7F,5
CONGLOMERADO,CGL,#FBC13F,6
ARENITO,ARN,#FEFF01,7
SILTITO,SLT,#7D4200,8
ARGILITO,AGT,#08FB41,9
FOLHELHO,FLH,#037E01,10
MARGA,MRG,#7D0281,11
IGNEA INTRUSIVA NAO IDENTIFICADA,INT,#FF0000,12
DIABASIO,DIA,#F97F71,13
BASALTO,BAS,#FE0000,14
ANIDRITA,AND,#FC349A,15
SAL POTASSICO,KSL,#FF7FFF,16
HALITA,HAL,#BEBEBE,17
SILVITA,SYL,#9089E7,18
CARNALITITA,CRN,#6D89E7,19
TAQUIDRITA,TQD,#6E1FDE,20
'''
LITO_MAP = pd.read_csv(io.StringIO(LITO_TABLE))
CODE_TO_NAME = LITO_MAP.set_index('CODIGO')['LITO'].to_dict()

# Upscaling of the 21 original classes (everything not listed -> OTHER)
LITH_UPSCALE = {
    "HALITA": "HALITE",
    "ANIDRITA": "ANHYDRITE",
    "CARNALITITA": "K_SALT", "SILVITA": "K_SALT",
    "TAQUIDRITA": "K_SALT", "SAL POTASSICO": "K_SALT",
    "CALCARENITO": "CARBONATE", "CALCARIO": "CARBONATE",
    "DOLOMITO": "CARBONATE", "CALCILUTITO": "CARBONATE",
    "MARGA": "MARL_CLAY", "ARGILITO": "MARL_CLAY", "FOLHELHO": "MARL_CLAY",
}
LITO_CATEGORIES = ['HALITE', 'ANHYDRITE', 'K_SALT', 'CARBONATE', 'MARL_CLAY', 'OTHER']
UPSCALE_COLORS = {
    'HALITE': '#6E6E6E', 'ANHYDRITE': '#FC349A', 'K_SALT': '#B000FF',
    'CARBONATE': '#0000FF', 'MARL_CLAY': '#037E01', 'OTHER': '#FBC13F',
}
TOP_LITHO = 'HALITE'   # lithology drawn ON TOP in scatter plots


def plot_order(df, top=TOP_LITHO):
    """Reorder rows so `top` lithology is drawn last (on top) in scatter plots.
    Seaborn draws points in row order, not in legend order."""
    if top is None or 'LITO_UPSCALED' not in df:
        return df
    m = df['LITO_UPSCALED'] == top
    return pd.concat([df[~m], df[m]])

# Approximate LOG-APPARENT values for pure minerals (Schlumberger chart book).
# Plotted only as a visual reference on the VP x DEN crossplot - verify before citing.
MINERAL_REF = {   # name: (DT us/ft, RHOB g/cc)
    'Halite': (67.0, 2.04), 'Anhydrite': (50.0, 2.98), 'Sylvite': (74.0, 1.86),
    'Carnallite': (83.0, 1.57), 'Calcite': (47.6, 2.71), 'Dolomite': (43.5, 2.88),
}

LOG_UNITS = {'GR': 'API', 'VP': 'm/s', 'DEN': 'g/cc'}
LOG_RANGES = {'GR': (0, 200), 'VP': (2000, 7000), 'DEN': (1.4, 3.1)}
LOG_COLORS = {'GR': 'green', 'VP': 'blue', 'DEN': 'black'}


# ==============================================================================
# Small helpers
# ==============================================================================
def show_table(table, title=None, floatfmt="{:.3f}"):
    """Print a DataFrame nicely in the Spyder console."""
    if title:
        print(f"\n{title}\n" + "-" * len(title))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                           'display.width', 250, 'display.float_format', floatfmt.format):
        print(table.to_string() if hasattr(table, 'to_string') else table)


def safe_name(name):
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))


def fit_window_to_screen(fig):
    """Shrink and reposition a Qt figure window that does not fit on its screen.

    Other backends (e.g. Spyder's inline plots) have no such window and are left alone.
    """
    manager = fig.canvas.manager
    window = getattr(manager, 'window', None)
    if window is None or not hasattr(window, 'frameGeometry'):
        return
    screen = window.screen().availableGeometry()
    frame = window.frameGeometry()
    canvas_w, canvas_h = fig.canvas.width(), fig.canvas.height()
    # Title bar, borders and toolbar do not scale with the figure
    extra_w, extra_h = frame.width() - canvas_w, frame.height() - canvas_h
    scale = min((screen.width() - extra_w) / canvas_w, (screen.height() - extra_h) / canvas_h)
    if scale < 1:
        window.resize(window.width() - canvas_w + int(canvas_w * scale),
                      window.height() - canvas_h + int(canvas_h * scale))
    if scale < 1 or not screen.contains(window.frameGeometry()):
        window.move(screen.topLeft())


def finish_fig(fig, name, fig_dir=None, show=True, save=True):
    """Save the figure (PNG) and either open its window without waiting or close it.

    Scripts must call plt.show() at their end to keep the windows open.
    """
    if save and fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, f"{safe_name(name)}.png"), dpi=150, bbox_inches='tight')
    if show:
        plt.show(block=False)
        plt.pause(0.001)   # let the GUI draw the window before the script continues
        fit_window_to_screen(fig)
    else:
        plt.close(fig)


def save_tables(tables, path_xlsx):
    """Save {sheet_name: DataFrame} to Excel; falls back to one CSV per table."""
    try:
        with pd.ExcelWriter(path_xlsx) as xw:
            for sheet, t in tables.items():
                t.to_excel(xw, sheet_name=sheet[:31])
        print(f"Saved: {path_xlsx}")
    except Exception as e:  # openpyxl missing, file open in Excel, ...
        print(f"Excel export failed ({e}); writing CSV files instead.")
        base = os.path.splitext(path_xlsx)[0]
        for sheet, t in tables.items():
            t.to_csv(f"{base}_{safe_name(sheet)}.csv")


def find_curve(df, names):
    for n in names:
        if n in df.columns and df[n].notna().any():
            return df[n].astype(float), n
    return pd.Series(np.nan, index=df.index), None


def _unit(las, name):
    try:
        return str(las.curves[name].unit).upper().replace(' ', '') if name else ''
    except Exception:
        return ''


# ==============================================================================
# 1. Loading
# ==============================================================================
def list_las_files(folder, wells=None):
    """All .las files in folder; if `wells` is given keep only files whose name
    contains one of those well names."""
    files = sorted({os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith('.las')})
    if wells:
        files = [f for f in files if any(w in os.path.basename(f) for w in wells)]
    return files


def _las_null(las):
    try:
        return float(las.well['NULL'].value)
    except Exception:
        return -999.25


def _clean_types(raw, null):
    """Numeric columns -> float with the LAS NULL value as NaN; text columns kept as text.
    (When a LAS has text columns, lasio returns EVERY column as text and does not
    replace the NULL value - this undoes that.)"""
    raw = raw.copy()
    idx = pd.to_numeric(pd.Series(raw.index), errors='coerce').to_numpy(float)
    raw.index = idx
    for c in raw.columns:
        col = raw[c]
        if not pd.api.types.is_numeric_dtype(col):
            num = pd.to_numeric(col, errors='coerce')
            if num.notna().sum() >= 0.5 * col.notna().sum():      # numeric stored as text
                raw[c] = num
            else:                                                   # real text column
                raw[c] = col.astype(str).str.strip()
                continue
        raw[c] = raw[c].astype(float).mask(np.isclose(raw[c].astype(float), null))
    return raw[~np.isnan(raw.index)]


def read_las_robust(path):
    """
    Returns (raw DataFrame indexed by depth, las header object, note).
    Numeric curves are float (NULL -> NaN); text curves (e.g. LITONAME, LITOUPSC)
    are kept as text.
    If lasio cannot read the ~A section (text values with spaces such as
    'VULCANICA NAO IDENTIFICADA', rows with extra/missing values, numbers stuck
    together like '2.34-999.25'), it is re-read line by line.
    """
    import re
    import logging
    lg = logging.getLogger('lasio')
    old_level = lg.level
    lg.setLevel(logging.ERROR)                       # silence "Could not convert curve #n"
    try:
        las = lasio.read(path)
        raw0 = las.df()
        # If the file has TEXT columns, lasio's whitespace split can silently shift
        # rows when a text value contains spaces (e.g. 'VULCANICA NAO IDENTIFICADA')
        # and the total count happens to divide evenly. Never trust it then: use the
        # line-by-line reader below, which handles text with spaces.
        has_text = (not pd.api.types.is_numeric_dtype(pd.Series(raw0.index))) or \
                   any(not pd.api.types.is_numeric_dtype(raw0[c]) for c in raw0.columns)
        if not has_text:
            return _clean_types(raw0, _las_null(las)), las, ''
        first_error = "file has text columns"
    except Exception as e:
        first_error = str(e).splitlines()[0][:120]
    finally:
        lg.setLevel(old_level)

    lg.setLevel(logging.ERROR)
    try:
        las = lasio.read(path, ignore_data=True)
    finally:
        lg.setLevel(old_level)
    try:
        wrapped = str(las.version['WRAP'].value).strip().upper().startswith('Y')
    except Exception:
        wrapped = False
    if wrapped:
        raise ValueError(f"lasio could not read this wrapped LAS ({first_error}). "
                         "Unwrap it (e.g. in your log software) and try again.")
    names = [c.mnemonic for c in las.curves]
    n = len(names)
    null = _las_null(las)
    glued = re.compile(r'(?<=\d)-(?=\d)')            # '2.34-999.25' -> '2.34 -999.25' (not 1E-5)
    number = re.compile(r'[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?')

    lines, in_data = [], False
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            s = line.strip()
            if not in_data:
                in_data = s.upper().startswith('~A')
                continue
            if not s or s.startswith('#'):
                continue
            if s.startswith('~'):
                break
            lines.append(glued.sub(' -', s).replace(',', ' ').split())

    def is_num(t):
        return bool(number.fullmatch(t)) or t.lower() in ('nan', '-nan', 'null', 'inf', '-inf')

    # Which columns are text: a column is text if ANY line with exactly n tokens has
    # a non-numeric value there. (A missing name can be written as -9999.25 or 'nan',
    # so a single line is not enough to decide.)
    exact = [toks for toks in lines if len(toks) == n]
    text_cols = sorted({i for toks in exact for i, t in enumerate(toks) if not is_num(t)})
    if 0 in text_cols:
        raise ValueError(f"Depth column is not numeric ({first_error}).")
    num_cols = [i for i in range(n) if i not in text_cols]

    # Parse every row by POSITION. Only the first text column (e.g. LITONAME) may contain
    # spaces: extra tokens are joined into it.
    t0 = text_cols[0] if text_cols else None
    num_rows, txt_rows, bad = [], [], 0
    for toks in lines:
        extra = len(toks) - n
        if extra < 0 or (extra > 0 and t0 is None):
            bad += 1
            continue
        if extra > 0:
            toks = toks[:t0] + [' '.join(toks[t0:t0 + extra + 1])] + toks[t0 + extra + 1:]
        try:
            vals = [float(toks[i]) for i in num_cols]
        except ValueError:
            bad += 1
            continue
        num_rows.append(vals)
        txt_rows.append([toks[i] for i in text_cols])
    if not num_rows:
        raise ValueError(f"No readable data rows ({first_error}).")
    arr = np.array(num_rows, dtype=float)
    raw = pd.DataFrame(arr[:, 1:], columns=[names[i] for i in num_cols[1:]], index=arr[:, 0])
    raw = raw.mask(np.isclose(raw, null))
    for j, i in enumerate(text_cols):
        col = pd.Series([r[j] for r in txt_rows], index=raw.index, dtype=object)
        nulls = col.str.lower().isin(['nan', '-nan', 'null', 'none', '']) | \
            col.map(lambda t: is_num(t) and np.isclose(float(t), null) if is_num(t) else False)
        raw[names[i]] = col.mask(nulls)                      # missing text -> NaN
    raw = raw[names[1:]]
    raw.index.name = names[0]
    k = len(text_cols)
    txt = f"; text columns: {', '.join(names[i] for i in text_cols)}" if k else ""
    note = (f"read line by line ({first_error}): {len(num_rows)} rows kept, "
            f"{bad} damaged rows skipped{txt}")
    if bad:                                    # only warn when rows were actually lost
        warnings.warn(f"{os.path.basename(path)}: {note}")
    return raw, las, note


def load_las_well(path, well_name=None):
    """
    Read one LAS and return a standard DataFrame: MD (m), WELL, LITO_CODE, GR, VP (m/s), DEN (g/cc)
    plus an info dict with the curves/units that were used.
    """
    raw, las, read_note = read_las_robust(path)
    raw = raw[~raw.index.duplicated(keep='first')]

    if well_name is None:
        try:
            well_name = str(las.well['WELL'].value).strip()
        except Exception:
            well_name = ''
        well_name = well_name or os.path.splitext(os.path.basename(path))[0]

    md = raw.index.to_numpy(dtype=float)
    depth_unit = str(getattr(las, 'index_unit', '') or '').upper()
    if depth_unit.startswith('F'):
        md = md * FT2M
        warnings.warn(f"{well_name}: depth in feet -> converted to meters.")

    out = pd.DataFrame({'MD': md, 'WELL': well_name})
    info = {'WELL': well_name, 'file': os.path.basename(path), 'depth_unit': depth_unit or '?',
            'read_note': read_note or 'ok'}

    # ---- VP from DT (preferred) or from a velocity curve
    dt, dt_name = find_curve(raw, CURVES['DT'])
    if dt_name:
        u = _unit(las, dt_name)
        k = 1e6 if ('/M' in u and '/F' not in u) else 304800.0   # us/m or us/ft
        with np.errstate(divide='ignore', invalid='ignore'):
            vp = k / dt.to_numpy()
        info.update({'VP source': dt_name, 'VP src unit': u or '?'})
    else:
        v, v_name = find_curve(raw, CURVES['VP'])
        vp = v.to_numpy()
        if v_name and np.isfinite(vp).any() and np.nanmedian(vp) < 20:      # km/s -> m/s
            vp = vp * 1000
        info.update({'VP source': v_name, 'VP src unit': _unit(las, v_name) or '?'})
    vp = np.where(np.isfinite(vp) & (vp > 0), vp, np.nan)
    out['VP'] = vp

    # ---- DEN
    den, den_name = find_curve(raw, CURVES['DEN'])
    den = den.to_numpy()
    if den_name and np.isfinite(den).any() and np.nanmedian(den) > 100:      # kg/m3 -> g/cc
        den = den / 1000
    out['DEN'] = den
    info['DEN source'] = den_name

    # ---- GR
    gr, gr_name = find_curve(raw, CURVES['GR'])
    out['GR'] = gr.to_numpy()
    info['GR source'] = gr_name

    # ---- Lithology code
    lito, lito_name = find_curve(raw, CURVES['LITO'])
    lito = lito.to_numpy()
    lito = np.where(lito < 0, np.nan, lito)        # -999 etc.
    out['LITO_CODE'] = lito
    info['LITO source'] = lito_name

    # ---- Upscaled lithology already in the file (e.g. LITOUPSC exported from Colab,
    #      including your flag-based reclassification). Stored as a code so it can be
    #      resampled; used by add_lithology() when use_file_lithology=True.
    out['LITO_UPS_FILE'] = np.nan
    info['LITO_UPS source'] = None
    for name in FILE_UPSCALED_CURVES:
        if name in raw.columns and not pd.api.types.is_numeric_dtype(raw[name]):
            txt = raw[name].astype(str).str.strip().str.upper()
            out['LITO_UPS_FILE'] = txt.map({c: i for i, c in enumerate(LITO_CATEGORIES)}).to_numpy(float)
            info['LITO_UPS source'] = name
            other = sorted({str(v) for v in txt.dropna().unique()} - set(LITO_CATEGORIES)
                           - {'NAN', 'NONE', ''})
            if other:
                info['LITO_UPS ignored values'] = ', '.join(other)   # e.g. UNKNOWN -> code is used
            break

    out = out.dropna(subset=['MD']).sort_values('MD').reset_index(drop=True)
    info['orig_rows'] = len(out)
    return out, info


# ==============================================================================
# 2. Resampling to 0.1524 m
# ==============================================================================
def resample_well(df, step=STEP, cont_cols=('GR', 'VP', 'DEN'), cat_cols=('LITO_CODE', 'LITO_UPS_FILE')):
    """
    Put a well on a regular grid of multiples of `step` (0.1524 m).

    * Already at 0.1524 m          -> kept as is (no interpolation).
    * Finer than 0.1524 m          -> continuous logs: mean of the samples in each
                                      0.1524 m bin; lithology: sample closest to bin centre.
    * Coarser than 0.1524 m        -> continuous logs: linear interpolation (not across
                                      gaps); lithology: nearest sample.
    """
    df = df.dropna(subset=['MD']).sort_values('MD').drop_duplicates('MD')
    md = df['MD'].to_numpy(float)
    orig = float(np.median(np.diff(md))) if len(md) > 1 else step
    info = {'orig_step_m': round(orig, 5)}

    if abs(orig - step) <= 1e-3 * step:
        info['resampling'] = 'none (already 0.1524 m)'
        out = df.copy()
        out['MD'] = out['MD'].round(4)
        return out.reset_index(drop=True), info

    k = np.round(md / step).astype(np.int64)
    k_grid = np.arange(k.min(), k.max() + 1)
    grid = k_grid * step
    out = pd.DataFrame({'MD': np.round(grid, 4)})

    if orig < step:
        info['resampling'] = f'bin average ({orig:.4f} -> {step} m)'
        tmp = df.assign(_k=k, _d=np.abs(md - k * step))
        means = tmp.groupby('_k')[list(cont_cols)].mean()
        for c in cont_cols:
            out[c] = means[c].reindex(k_grid).to_numpy()
        for c in cat_cols:
            nearest = (tmp.dropna(subset=[c]).sort_values(['_k', '_d'])
                       .drop_duplicates('_k').set_index('_k')[c])
            out[c] = nearest.reindex(k_grid).to_numpy()
    else:
        info['resampling'] = f'linear interpolation ({orig:.4f} -> {step} m)'
        for c in cont_cols:
            v = df[c].to_numpy(float)
            ok = ~np.isnan(v)
            if ok.sum() < 2:
                out[c] = np.nan
                continue
            x, y = md[ok], v[ok]
            vals = np.interp(grid, x, y, left=np.nan, right=np.nan)
            j = np.searchsorted(x, grid)
            jl, jr = np.clip(j - 1, 0, len(x) - 1), np.clip(j, 0, len(x) - 1)
            gap = (x[jr] - x[jl]) > 1.5 * orig          # do not bridge missing intervals
            vals[gap] = np.nan
            out[c] = vals
        for c in cat_cols:
            v = df[c].to_numpy(float)
            ok = ~np.isnan(v)
            if ok.sum() < 2:
                out[c] = np.nan
                continue
            x, y = md[ok], v[ok]
            j = np.clip(np.searchsorted(x, grid), 1, len(x) - 1)
            idx = np.where((grid - x[j - 1]) <= (x[j] - grid), j - 1, j)
            vals = y[idx].astype(float)
            vals[np.abs(x[idx] - grid) > orig] = np.nan
            out[c] = vals

    out['WELL'] = df['WELL'].iloc[0]
    return out, info


# ==============================================================================
# 3. Interval cut and lithology
# ==============================================================================
def cut_interval(df, top=None, base=None):
    """Keep TOP <= MD <= BASE. None = no limit on that side."""
    m = pd.Series(True, index=df.index)
    if top is not None:
        m &= df['MD'] >= top
    if base is not None:
        m &= df['MD'] <= base
    return df[m].reset_index(drop=True)


def add_lithology(df, use_file_lithology=True):
    """LITO_CODE -> LITO_NAME (21 classes) -> LITO_UPSCALED (6 classes).
    If use_file_lithology and the file had an upscaled lithology (LITOUPSC),
    that value wins wherever it exists; the code-based value is kept as
    LITO_UPSCALED_CODE for comparison. Returns (df, n_samples_changed_by_file)."""
    df = df.copy()
    code = df['LITO_CODE'].round()
    df['LITO_NAME'] = code.map(CODE_TO_NAME)
    df.loc[code.notna() & df['LITO_NAME'].isna(), 'LITO_NAME'] = 'CODE_NOT_IN_TABLE'
    df['LITO_UPSCALED'] = df['LITO_NAME'].map(LITH_UPSCALE)
    df.loc[df['LITO_NAME'].notna() & df['LITO_UPSCALED'].isna(), 'LITO_UPSCALED'] = 'OTHER'
    df['LITO_UPSCALED_CODE'] = df['LITO_UPSCALED']
    n_changed = 0
    if use_file_lithology and 'LITO_UPS_FILE' in df and df['LITO_UPS_FILE'].notna().any():
        f = df['LITO_UPS_FILE'].round().map(dict(enumerate(LITO_CATEGORIES)))
        n_changed = int((f.notna() & (f != df['LITO_UPSCALED'])).sum())
        df['LITO_UPSCALED'] = f.combine_first(df['LITO_UPSCALED'])
    return df, n_changed


def apply_valid_ranges(df, valid_ranges):
    """Set values outside physically valid limits to NaN.
    valid_ranges = {'DEN': (1.5, 3.1), 'VP': (2500, 7500)}  (None = no limit)
    Returns the DataFrame and {log: number of values removed}."""
    df = df.copy()
    removed = {}
    for log, (lo, hi) in (valid_ranges or {}).items():
        if log not in df:
            continue
        s = df[log]
        bad = s.notna() & (((s < lo) if lo is not None else False) | ((s > hi) if hi is not None else False))
        removed[log] = int(bad.sum())
        df.loc[bad, log] = np.nan
    return df, removed


def lithology_boundary_distance(df, step=STEP):
    """Distance (m) from each sample to the nearest change of LITO_UPSCALED.
    Only changes between two known lithologies in contiguous samples count."""
    d = df.sort_values('MD')
    md = d['MD'].to_numpy(float)
    lit = d['LITO_UPSCALED'].to_numpy(object)
    prev_lit = np.r_[[None], lit[:-1]]
    contiguous = np.r_[False, np.diff(md) <= 1.5 * step]
    known = pd.notna(lit) & pd.notna(prev_lit)
    change = contiguous & known & (lit != prev_lit)
    b = (md[change] + md[np.r_[change[1:], False]]) / 2 if change.any() else np.array([])
    if len(b) == 0:
        return pd.Series(np.inf, index=d.index).reindex(df.index)
    j = np.clip(np.searchsorted(b, md), 1, len(b)) if len(b) > 1 else np.ones(len(md), int)
    left = b[np.clip(j - 1, 0, len(b) - 1)]
    right = b[np.clip(j, 0, len(b) - 1)]
    dist = np.minimum(np.abs(md - left), np.abs(md - right))
    return pd.Series(dist, index=d.index).reindex(df.index)


def normalize_gr(df, ref=None, min_dist=1.0, min_samples=50):
    """
    Light GR normalisation: per-well constant SHIFT so the median GR of clean
    halite (> min_dist m from any lithology contact) equals `ref`.
    ref=None -> ref = median of the per-well halite medians (training wells).
    The original curve is kept as GR_RAW. Returns (df, ref, table of shifts).
    """
    df = df.copy()
    df['GR_RAW'] = df['GR']
    hal = df[(df['LITO_UPSCALED'] == 'HALITE') & (df['DIST_LITHO_BOUNDARY'] > min_dist)]
    med = hal.groupby('WELL')['GR'].median()
    cnt = hal.groupby('WELL').size()
    ok = cnt[cnt >= min_samples].index
    if ref is None:
        if len(ok) == 0:
            raise ValueError("No well has enough clean halite to define the GR reference.")
        ref = float(med[ok].median())
    rows = []
    for w in df['WELL'].unique():
        n = int(cnt.get(w, 0))
        if n >= min_samples:
            shift = float(med[w] - ref)
        else:
            shift = 0.0
            print(f"  NOTE {w}: only {n} clean halite samples (> boundary buffer from contacts) "
                  f"-> GR not normalised for this well.")
        df.loc[df['WELL'] == w, 'GR'] = df.loc[df['WELL'] == w, 'GR'] - shift
        rows.append({'WELL': w, 'clean_halite_samples': n,
                     'halite_GR_median_raw': float(med.get(w, np.nan)),
                     'reference_GR': ref, 'shift_subtracted_API': shift})
    return df, ref, pd.DataFrame(rows).set_index('WELL')


def prepare_well(path, markers, required, step=STEP, valid_ranges=None, use_file_lithology=True,
                 return_all=False):
    """
    Full preparation of one well: load -> resample 0.1524 m -> Ariri cut ->
    lithology mapping -> distance to lithology boundary -> (optional) valid-range
    filter -> keep rows where all `required` columns exist.
    `markers` = {'WELL': {'top': .., 'base': ..}}
    return_all=True -> also returns every row in the interval BEFORE the required-column
    filter (used to keep GR / lithology in the exported LAS where VP is missing).
    """
    df, info = load_las_well(path)
    well = info['WELL']
    df, rinfo = resample_well(df, step=step)
    info.update(rinfo)

    mk = markers.get(well, {}) if markers else {}
    top, base = mk.get('top'), mk.get('base')
    if top is None and base is None:
        print(f"  {well}: Ariri top/base not set -> whole file used (fine if the file is already cut)")
    df = cut_interval(df, top, base)
    info.update({'ariri_top': top, 'ariri_base': base, 'rows_in_interval': len(df)})

    df, n_changed = add_lithology(df, use_file_lithology)
    if info.get('LITO_UPS source'):
        info['lithology used'] = (f"file {info['LITO_UPS source']} ({n_changed} samples differ from code)"
                                  if use_file_lithology else "LITO code (file lithology ignored)")
    else:
        info['lithology used'] = 'LITO code'
    # boundaries computed BEFORE dropping incomplete rows, so hidden contacts still count
    df['DIST_LITHO_BOUNDARY'] = lithology_boundary_distance(df, step)
    if valid_ranges:
        df, removed = apply_valid_ranges(df, valid_ranges)
        for log, nrem in removed.items():
            info[f'{log} out of range (removed)'] = nrem
    req_cols = ['LITO_UPSCALED' if c in ('LITO', 'LITO_CODE') else c for c in required]
    all_rows = df.copy()                       # every row in the interval (for export only)
    df = df.dropna(subset=req_cols).reset_index(drop=True)
    info['rows_used'] = len(df)
    if len(df):
        info['md_min'], info['md_max'] = df['MD'].min(), df['MD'].max()
        info['final_step_m'] = round(float(df['MD'].diff().mode().iloc[0]), 5) if len(df) > 1 else np.nan
    if return_all:
        return df, info, all_rows
    return df, info


# ==============================================================================
# 4. Model features
# ==============================================================================
def build_X(df, num_features, use_lito=True, categories=LITO_CATEGORIES):
    """Numeric features + one-hot lithology with a FIXED column set."""
    X = df[list(num_features)].astype(float).copy()
    if use_lito:
        cat = pd.Categorical(df['LITO_UPSCALED'], categories=categories)
        d = pd.get_dummies(cat, prefix='LITO').astype(int)
        d.index = df.index
        X = pd.concat([X, d], axis=1)
    return X


# ==============================================================================
# 4b. Mineral mixing model (MINERAL_MIX)
# ------------------------------------------------------------------------------
# Evaporites have ~zero porosity, so each sample is treated as a mix of up to
# three minerals.  Fractions f_i come from:
#       DT = sum(f_i * DT_i)      (sonic slowness mixes ~linearly, Wyllie)
#       GR = sum(f_i * GR_i)
#       sum(f_i) = 1,   0 <= f_i <= 1
# and then  DEN = sum(f_i * RHOB_i).
# Approximate LOG-APPARENT end-members (Schlumberger chart book); 'Clay' is a
# generic shale/marl point - adjust to your data.  The GR of non-radioactive
# minerals is replaced by the clean-halite GR of the training wells.
# ==============================================================================
MINERALS = {          # DT (us/ft), GR (API), RHOB (g/cc)
    'Halite':       {'DT': 67.0, 'GR': 0.0,   'RHOB': 2.04},
    'Anhydrite':    {'DT': 50.0, 'GR': 0.0,   'RHOB': 2.98},
    'Sylvite':      {'DT': 74.0, 'GR': 500.0, 'RHOB': 1.86},
    'Carnallite':   {'DT': 83.0, 'GR': 220.0, 'RHOB': 1.57},
    'Tachyhydrite': {'DT': 92.0, 'GR': 0.0,   'RHOB': 1.66},
    'Calcite':      {'DT': 47.6, 'GR': 0.0,   'RHOB': 2.71},
    'Dolomite':     {'DT': 43.5, 'GR': 0.0,   'RHOB': 2.88},
    'Clay':         {'DT': 90.0, 'GR': 120.0, 'RHOB': 2.50},
}
NON_RADIOACTIVE = ['Halite', 'Anhydrite', 'Tachyhydrite', 'Calcite', 'Dolomite']

# Minerals allowed for each upscaled lithology (max 3: two logs + closure).
# HALITE and ANHYDRITE share the same set (no Clay), so a halite/anhydrite label
# error does not matter: VP decides how much anhydrite there is. Carnallite
# captures thin K-salt beds not resolved by the lithology log (GR high + VP low).
MINERAL_SETS = {
    'HALITE':    ['Halite', 'Anhydrite', 'Carnallite'],
    'ANHYDRITE': ['Halite', 'Anhydrite', 'Carnallite'],
    'K_SALT':    ['Halite', 'Carnallite', 'Tachyhydrite'],
    'CARBONATE': ['Calcite', 'Anhydrite', 'Clay'],
    'MARL_CLAY': ['Clay', 'Calcite', 'Halite'],
    'OTHER':     ['Halite', 'Anhydrite', 'Clay'],
}
MIX_SIGMA = {'DT': 2.0, 'GR': 8.0}    # weights of the two log equations (us/ft, API)


def _subsets(items):
    from itertools import combinations
    return [list(c) for k in range(1, len(items) + 1) for c in combinations(items, k)]


def mineral_fractions(df, minerals=None, sets=None, gr_clean=0.0, sigma=None, non_radioactive=None):
    """
    Bounded least-squares mineral fractions for every row (vectorised).
    Returns a DataFrame with one column per mineral (0 where not in the set) and
    'MIX_MISFIT' (weighted residual). The constrained optimum is found by
    testing every subset of the mineral set (1, 2 or 3 minerals) and keeping the
    valid solution (all fractions in [0, 1]) with the lowest misfit.
    """
    minerals = minerals or MINERALS
    sets = sets or MINERAL_SETS
    sigma = sigma or MIX_SIGMA
    non_rad = non_radioactive if non_radioactive is not None else NON_RADIOACTIVE
    names = list(minerals)
    out = pd.DataFrame(0.0, index=df.index, columns=names)
    out['MIX_MISFIT'] = np.nan
    dt = 304800.0 / df['VP'].to_numpy(float)
    gr = df['GR'].to_numpy(float)
    W = np.diag([1 / sigma['DT'], 1 / sigma['GR']])

    def endm(m):
        g = gr_clean if m in non_rad else minerals[m]['GR']
        return np.array([minerals[m]['DT'], g])

    default_set = sets.get('OTHER', ['Halite', 'Anhydrite', 'Clay'])
    for lit in df['LITO_UPSCALED'].fillna('OTHER').unique():
        rows = np.where(df['LITO_UPSCALED'].fillna('OTHER').to_numpy() == lit)[0]
        mset = sets.get(lit, default_set)
        b = np.vstack([dt[rows], gr[rows]])                       # 2 x n
        best_mis = np.full(len(rows), np.inf)
        best_f = np.zeros((len(mset), len(rows)))
        for sub in _subsets(mset):
            A = np.column_stack([endm(m) for m in sub])           # 2 x k
            if len(sub) == 1:
                f = np.ones((1, len(rows)))
            else:
                Ar = A[:, :-1] - A[:, [-1]]                       # closure substituted
                br = b - A[:, [-1]]
                g = np.linalg.pinv(W @ Ar) @ (W @ br)             # (k-1) x n
                f = np.vstack([g, 1 - g.sum(axis=0)])
            mis = np.sum((W @ (A @ f - b)) ** 2, axis=0)
            valid = np.all((f >= -1e-9) & (f <= 1 + 1e-9), axis=0) & (mis < best_mis)
            if valid.any():
                best_mis[valid] = mis[valid]
                full = np.zeros((len(mset), len(rows)))
                for i, m in enumerate(sub):
                    full[mset.index(m)] = f[i]
                best_f[:, valid] = full[:, valid]
        for i, m in enumerate(mset):
            out.iloc[rows, names.index(m)] = np.clip(best_f[i], 0, 1)
        out.iloc[rows, out.columns.get_loc('MIX_MISFIT')] = np.sqrt(best_mis)
    bad = ~np.isfinite(dt) | np.isnan(gr)
    out.loc[df.index[bad], :] = np.nan
    return out


def calibrate_mineral_densities(fractions, den, minerals=None, max_change=0.25, min_weight=20.0):
    """Least-squares mineral densities: DEN = F @ rho, each rho within
    +/- max_change of the chart value. Minerals barely present keep the chart value."""
    from scipy.optimize import lsq_linear
    minerals = minerals or MINERALS
    names = list(minerals)
    ok = fractions[names].notna().all(axis=1) & pd.Series(np.isfinite(np.asarray(den, float)),
                                                          index=fractions.index)
    F = fractions.loc[ok, names].to_numpy()
    y = np.asarray(den, float)[ok.to_numpy()]
    chart = np.array([minerals[m]['RHOB'] for m in names])
    present = F.sum(axis=0) >= min_weight                       # equivalent samples of pure mineral
    rho = chart.copy()
    if present.any():
        y_adj = y - F[:, ~present] @ chart[~present]
        res = lsq_linear(F[:, present], y_adj,
                         bounds=(chart[present] - max_change, chart[present] + max_change))
        rho[present] = res.x
    return dict(zip(names, map(float, rho))), dict(zip(names, map(bool, present)))


def mineral_mix_density(fractions, rho):
    names = list(rho)
    return fractions[names].to_numpy() @ np.array([rho[m] for m in names])


def regression_metrics(y_true, y_pred):
    """MAE, RMSE, R2, bias, MAPE, Pearson r and the calibration regression
    (least-squares line  predicted = slope * measured + intercept, i.e. the red line
    on the crossplots; ideal slope 1, intercept 0. Slope < 1 = predictions compressed)."""
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    n = len(y)
    if n < 3:
        return {'n': n}
    res = p - y
    const = np.std(p) < 1e-9 or np.std(y) < 1e-9     # e.g. LITHO_MEAN inside one lithology
    slope, intercept = (np.nan, np.nan) if const else np.polyfit(y, p, 1)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {'n': n,
            'MAE': np.mean(np.abs(res)),
            'RMSE': np.sqrt(np.mean(res ** 2)),
            'R2': 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            'Pearson r': np.nan if const else np.corrcoef(y, p)[0, 1],
            'Bias (pred-meas)': res.mean(),
            'MAPE %': np.mean(np.abs(res / y)) * 100,
            'P90 |err|': np.percentile(np.abs(res), 90),
            'Fit slope': slope, 'Fit intercept': intercept}


# ==============================================================================
# 5. Plotting
# ==============================================================================
def plot_well_tracks(wdf, title, curves=('GR', 'VP', 'DEN'), overlays=None,
                     extra_tracks=None, ranges=None, colors=None, markers=None):
    """
    Lithology column + one track per curve.
    overlays: {'DEN': [(column, color, linestyle, label), ...]}
    extra_tracks: [(column, label, (xmin, xmax), color, 'residual'|'line')]
    Returns the figure.
    """
    ranges = {**LOG_RANGES, **(ranges or {})}
    colors = {**LOG_COLORS, **(colors or {})}
    overlays = overlays or {}
    extra_tracks = extra_tracks or []
    wdf = wdf.sort_values('MD')
    step = wdf['MD'].diff().median() if len(wdf) > 1 else STEP
    step = step if step and step > 0 else STEP
    # Re-insert missing depths as NaN rows so curves are NOT drawn across gaps
    key = np.round((wdf['MD'] - wdf['MD'].iloc[0]) / step).astype(int)
    wplot = wdf.set_index(key)
    wplot = wplot[~wplot.index.duplicated()].reindex(np.arange(key.min(), key.max() + 1))
    wplot['MD'] = wdf['MD'].iloc[0] + wplot.index.to_numpy() * step
    n = 1 + len(curves) + len(extra_tracks)
    widths = [0.5] + [1] * (n - 1)
    fig, axes = plt.subplots(1, n, figsize=(1.0 + 2.2 * (n - 1), 10), sharey=True,
                             gridspec_kw={'width_ratios': widths})
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Lithology column
    ax = axes[0]
    blk =(wdf['LITO_UPSCALED'] != wdf['LITO_UPSCALED'].shift()) | (wdf['MD'].diff() > 1.5 * step)
    used = []
    for _, g in wdf.groupby(blk.cumsum()):
        lit = g['LITO_UPSCALED'].iloc[0]
        ax.axhspan(g['MD'].min() - step / 2, g['MD'].max() + step / 2,
                   color=UPSCALE_COLORS.get(lit, 'white'), lw=0)
        if lit not in used:
            used.append(lit)
    ax.set_xticks([])
    ax.set_xlabel('LITHO', fontweight='bold', fontsize=9)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('MD (m)')

    # Curves (drawn from the gap-aware copy)
    wdf = wplot
    for ax, c in zip(axes[1:], curves):
        if c in wdf and wdf[c].notna().any():
            ax.plot(wdf[c], wdf['MD'], color=colors.get(c, 'k'), lw=0.8, label=f'{c} measured')
        for col, colr, ls, lab in overlays.get(c, []):
            if col in wdf and wdf[col].notna().any():
                ax.plot(wdf[col], wdf['MD'], color=colr, ls=ls, lw=0.9, label=lab)
        ax.set_xlim(ranges.get(c))
        ax.set_xlabel(f"{c} ({LOG_UNITS.get(c, '')})", color=colors.get(c, 'k'),
                      fontweight='bold', fontsize=9)
        if overlays.get(c):
            ax.legend(fontsize=7, loc='lower left')

    for ax, (col, lab, rng, colr, kind) in zip(axes[1 + len(curves):], extra_tracks):
        if kind == 'stack':      # col = list of fraction columns, colr = list of colours
            cols = [c for c in col if c in wdf and wdf[c].fillna(0).sum() > 0]
            left = np.zeros(len(wdf))
            for c in cols:
                v = wdf[c].to_numpy(float)
                right = left + np.nan_to_num(v)
                ax.fill_betweenx(wdf['MD'], np.where(np.isnan(v), np.nan, left),
                                 np.where(np.isnan(v), np.nan, right),
                                 color=colr[col.index(c)], lw=0, label=c)
                left = right
            ax.legend(fontsize=6, loc='lower right', framealpha=0.8)
        elif col in wdf:
            if kind == 'residual':
                r = wdf[col]
                ax.fill_betweenx(wdf['MD'], 0, r, color=colr, alpha=0.6, lw=0)
                ax.axvline(0, color='k', lw=0.8)
            else:
                ax.plot(wdf[col], wdf['MD'], color=colr, lw=0.8)
        ax.set_xlim(rng)
        ax.set_xlabel(lab, fontweight='bold', fontsize=9)

    for ax in axes[1:]:
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(mpl_ticker.MaxNLocator(4))
        ax.grid(True, ls=':', alpha=0.5)

    if markers:
        for name, d in markers.items():
            if d is not None:
                for ax in axes:
                    ax.axhline(d, color='k', ls='--', lw=1)
                axes[-1].text(axes[-1].get_xlim()[1], d, f" {name}", va='center', fontsize=8,
                              fontweight='bold')

    axes[0].set_ylim(wdf['MD'].max() + 5, wdf['MD'].min() - 5)
    fig.legend(handles=[mpatches.Patch(color=UPSCALE_COLORS.get(l, 'white'), label=l) for l in
                        [c for c in LITO_CATEGORIES if c in used]],
               loc='lower center', ncol=len(used), fontsize=8, bbox_to_anchor=(0.5, 0.0))
    plt.subplots_adjust(wspace=0.15, bottom=0.06, top=0.9)
    return fig


def _gap_aware(wdf):
    """Copy on a regular grid with NaN rows in gaps, so lines are not drawn across them."""
    wdf = wdf.sort_values('MD')
    step = wdf['MD'].diff().median() if len(wdf) > 1 else STEP
    step = step if step and step > 0 else STEP
    key = np.round((wdf['MD'] - wdf['MD'].iloc[0]) / step).astype(int)
    w = wdf.set_index(key)
    w = w[~w.index.duplicated()].reindex(np.arange(key.min(), key.max() + 1))
    w['MD'] = wdf['MD'].iloc[0] + w.index.to_numpy() * step
    return w, step


def _lito_blocks(wdf, step):
    """List of (top, base, lithology) blocks of constant LITO_UPSCALED."""
    wdf = wdf.sort_values('MD')
    blk = (wdf['LITO_UPSCALED'] != wdf['LITO_UPSCALED'].shift()) | (wdf['MD'].diff() > 1.5 * step)
    return [(g['MD'].min() - step / 2, g['MD'].max() + step / 2, g['LITO_UPSCALED'].iloc[0])
            for _, g in wdf.groupby(blk.cumsum())]


def plot_model_tracks(wdf, title, models, den_col='DEN', logs=('GR', 'VP'), lito_alpha=0.25,
                      den_range=(1.4, 3.1), flag_col=None, stack=None, markers=None):
    """
    One track per model: measured density (black) + that model's prediction.
    The lithology is shaded behind EVERY track (transparency = lito_alpha).
      models : list of (column, name, colour)
      logs   : input logs shown before the model tracks
      flag_col: optional 0/1/2 column drawn as a narrow 'FLAG' track
      stack  : optional (columns, colours, label) for a stacked fraction track
    Track headers show the MAE of each model where measured density exists.
    """
    w, step = _gap_aware(wdf)
    blocks = _lito_blocks(wdf, step)
    used = []
    for _, _, lit in blocks:
        if lit not in used:
            used.append(lit)

    spec = [('LITHO', 0.45)] + [(l, 1.0) for l in logs]
    if flag_col and flag_col in w:
        spec.append(('FLAG', 0.3))
    spec += [(m[0], 1.0) for m in models]
    if stack:
        spec.append(('STACK', 1.0))
    fig, axes = plt.subplots(1, len(spec), figsize=(0.9 + 1.9 * sum(s[1] for s in spec), 10.5),
                             sharey=True, gridspec_kw={'width_ratios': [s[1] for s in spec]})
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.985)

    has_den = den_col in w and w[den_col].notna().any()
    for ax, (kind, _) in zip(axes, spec):
        alpha = 1.0 if kind == 'LITHO' else lito_alpha
        if kind not in ('FLAG', 'STACK'):        # no lithology shading behind FLAG / mineral fractions
            for z1, z2, lit in blocks:
                ax.axhspan(z1, z2, color=UPSCALE_COLORS.get(lit, 'white'), alpha=alpha, lw=0, zorder=0)
        if kind == 'LITHO':
            ax.set_xticks([])
            ax.set_xlabel('LITHO', fontweight='bold', fontsize=9)
        elif kind in logs:
            if w[kind].notna().any():
                ax.plot(w[kind], w['MD'], color=LOG_COLORS.get(kind, 'k'), lw=0.7, zorder=5)
            ax.set_xlim(LOG_RANGES.get(kind))
            ax.set_xlabel(f"{kind} ({LOG_UNITS.get(kind, '')})", color=LOG_COLORS.get(kind, 'k'),
                          fontweight='bold', fontsize=9)
        elif kind == 'FLAG':
            f = w[flag_col]
            for val, col in [(1, 'cyan'), (2, 'red')]:
                ax.fill_betweenx(w['MD'], 0, 1, where=(f == val).to_numpy(), color=col, lw=0, zorder=5)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.grid(False)
            ax.set_xlabel('FLAG', fontweight='bold', fontsize=8)
        elif kind == 'STACK':
            cols, colours, lab = stack
            left = np.zeros(len(w))
            for c, colr in zip(cols, colours):
                if c not in w or w[c].fillna(0).sum() == 0:
                    continue
                v = w[c].to_numpy(float)
                right = left + np.nan_to_num(v)
                ax.fill_betweenx(w['MD'], np.where(np.isnan(v), np.nan, left),
                                 np.where(np.isnan(v), np.nan, right), color=colr, lw=0, label=c, zorder=5)
                left = right
            ax.set_xlim(0, 1)
            ax.set_xlabel(lab, fontweight='bold', fontsize=9)
            ax.legend(fontsize=6, loc='lower right', framealpha=0.85)
        else:                                            # a model track
            col, name, colr = next(m for m in models if m[0] == kind)
            if has_den:
                ax.plot(w[den_col], w['MD'], color='black', lw=0.7, zorder=5, label='measured')
            if col in w and w[col].notna().any():
                ax.plot(w[col], w['MD'], color=colr, lw=0.9, zorder=6, label=name)
            lab = name
            if has_den and col in wdf and col != 'DEN_MERGED':    # merged = measured where it exists
                ok = wdf[[den_col, col]].dropna()
                if len(ok):
                    lab += f"\nMAE {np.mean(np.abs(ok[col] - ok[den_col])):.3f}"
            ax.set_xlim(den_range)
            ax.set_xlabel(lab, color=colr, fontweight='bold', fontsize=9)
        if kind not in ('LITHO', 'FLAG'):
            ax.xaxis.set_major_locator(mpl_ticker.MaxNLocator(4))
            ax.grid(True, ls=':', alpha=0.5, zorder=1)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(labelsize=8)

    if markers:
        for name, d in markers.items():
            if d is not None:
                for ax in axes:
                    ax.axhline(d, color='k', ls='--', lw=1, zorder=10)
                axes[-1].text(axes[-1].get_xlim()[1], d, f" {name}", va='center', fontsize=8,
                              fontweight='bold')

    axes[0].set_ylabel('MD (m)')
    axes[0].set_ylim(wdf['MD'].max() + 5, wdf['MD'].min() - 5)
    handles = [mpatches.Patch(color=UPSCALE_COLORS.get(l, 'white'), label=l)
               for l in LITO_CATEGORIES if l in used]
    if has_den:
        handles.append(plt.Line2D([0], [0], color='black', lw=1.2, label=f'{den_col} measured'))
    if flag_col and flag_col in w:
        handles += [mpatches.Patch(color='cyan', label='flag: outside training range'),
                    mpatches.Patch(color='red', label='flag: unseen lithology')]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 8), fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    plt.subplots_adjust(wspace=0.12, bottom=0.06, top=0.9)
    return fig


# ==============================================================================
# 6. Data-exploration figures (shared by train_rhob_model.py and predict_rhob.py)
# ==============================================================================
def _sample_groups(data, by, n_max, seed=42):
    return data.sample(frac=1, random_state=seed).groupby(by).head(n_max)


def fig_scatter_matrix(data, hue, order, palette, title, logs=('GR', 'VP', 'DEN'),
                       n_max=3000, point_size=12, top=TOP_LITHO, seed=42):
    """Scatter-plot matrix of the logs, coloured by `hue` (lithology or well)."""
    import seaborn as sns
    s = _sample_groups(data, hue, n_max, seed)
    if hue == 'LITO_UPSCALED':
        s = plot_order(s, top)
    lito = hue == 'LITO_UPSCALED'
    g = sns.pairplot(s, vars=list(logs), hue=hue, hue_order=order, palette=palette, diag_kind='kde',
                     height=3, plot_kws={'s': point_size, 'alpha': 0.5 if lito else 0.4, 'edgecolor': None},
                     diag_kws={'common_norm': False, 'fill': lito, **({'alpha': 0.3} if lito else {})})
    g.figure.suptitle(title, y=1.02, fontweight='bold')
    return g.figure


def fig_histograms(data, order, palette, logs=('GR', 'VP', 'DEN')):
    """Histograms per log coloured by lithology. Left: counts; right: normalised per lithology."""
    import seaborn as sns
    fig, axes = plt.subplots(len(logs), 2, figsize=(15, 4.2 * len(logs)), squeeze=False)
    for i, log in enumerate(logs):
        d = data.dropna(subset=[log])
        for j, (stat, norm, ttl) in enumerate([('count', True, 'counts'),
                                               ('density', False, 'normalised per lithology')]):
            if len(d):
                sns.histplot(data=d, x=log, hue='LITO_UPSCALED',
                             hue_order=[o for o in order if o in d['LITO_UPSCALED'].unique()],
                             palette=palette, element='step', stat=stat, common_norm=norm, bins=80,
                             alpha=0.25, ax=axes[i, j])
            axes[i, j].set_title(f"{log} ({LOG_UNITS.get(log, '')}) - {ttl}  (n={len(d)})",
                                 fontweight='bold')
    plt.tight_layout()
    return fig


def fig_vp_den(data, order, palette, gardner=None, gardner_label='Gardner fit',
               n_max=3000, point_size=12, top=TOP_LITHO, seed=42):
    """VP x DEN crossplot with a Gardner curve and approximate pure-mineral log values.
    gardner=(a, b) to draw a given relation; None = fit it to these data.
    Returns (figure, (a, b))."""
    import seaborn as sns
    d = data.dropna(subset=['VP', 'DEN'])
    if gardner is None:
        c = np.polyfit(np.log(d['VP']), np.log(d['DEN']), 1)       # ln(DEN) = b ln(VP) + ln(a)
        gardner = (float(np.exp(c[1])), float(c[0]))
    a, b = gardner
    s = plot_order(_sample_groups(d, 'LITO_UPSCALED', n_max, seed), top)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(data=s, x='VP', y='DEN', hue='LITO_UPSCALED',
                    hue_order=[o for o in order if o in s['LITO_UPSCALED'].unique()],
                    palette=palette, s=point_size, alpha=0.5, edgecolor=None, ax=ax)
    vv = np.linspace(d['VP'].quantile(.001), d['VP'].quantile(.999), 200)
    ax.plot(vv, a * vv ** b, 'k--', lw=2, label=f'{gardner_label}: DEN={a:.3g}·VP^{b:.3f}')
    ax.plot(vv, 0.31 * vv ** 0.25, color='grey', ls=':', lw=2, label='Gardner (1974): 0.31·VP^0.25')
    for name, (dt, rho) in MINERAL_REF.items():
        vpm = 304800 / dt
        ax.scatter(vpm, rho, marker='*', s=250, color='gold', edgecolor='k', zorder=10)
        ax.annotate(name, (vpm, rho), xytext=(6, 6), textcoords='offset points', fontsize=9,
                    fontweight='bold')
    ax.set_title(f"VP x DEN with Gardner relation and pure-mineral log values (approx.)  n={len(d)}",
                 fontweight='bold')
    ax.set_xlabel('VP (m/s)')
    ax.set_ylabel('DEN (g/cc)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    return fig, (a, b)


def fig_boxplots(data, order, logs=('GR', 'VP', 'DEN')):
    """Box plots of each log per lithology and well (outliers hidden)."""
    import seaborn as sns
    fig, axes = plt.subplots(len(logs), 1, figsize=(14, 4 * len(logs)), sharex=True, squeeze=False)
    axes = axes.ravel()
    wells = sorted(data['WELL'].unique())
    for ax, log in zip(axes, logs):
        d = data.dropna(subset=[log])
        if len(d):
            sns.boxplot(data=d, x='LITO_UPSCALED', y=log, hue='WELL', order=order, hue_order=wells,
                        palette='tab10', showfliers=False, ax=ax)
        ax.set_ylabel(f"{log} ({LOG_UNITS.get(log, '')})")
        if ax.get_legend():
            if ax is axes[0]:
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
            else:
                ax.get_legend().remove()
    axes[0].set_title("Distribution per lithology and well (outliers hidden)", fontweight='bold')
    plt.tight_layout()
    return fig


def lithology_proportions(data, order):
    prop = pd.crosstab(data['WELL'], data['LITO_UPSCALED'], normalize='index')
    prop = prop.reindex(columns=order, fill_value=0) * 100
    # Wells sorted by salt thickness (MD base - MD top of the rows used), thickest first
    thick = data.groupby('WELL')['MD'].agg(lambda md: md.max() - md.min())
    prop = prop.loc[thick.reindex(prop.index).sort_values(ascending=False).index]
    prop.loc['ALL WELLS'] = data['LITO_UPSCALED'].value_counts(normalize=True).reindex(order, fill_value=0) * 100
    return prop


def fig_lito_proportions(prop, palette, title='Lithology proportion per well (rows used)'):
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(prop) + 2))
    prop.plot(kind='barh', stacked=True, color=[palette.get(c, '#DDDDDD') for c in prop.columns],
              ax=ax, edgecolor='k', lw=0.3)
    ax.set_xlabel('% of samples')
    ax.set_title(title, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def export_las(wdf, path, curves, well_name, other_text='', step=STEP, null=-9999.25):
    """curves: {column: (mnemonic, unit, description)}
    Written on a continuous 0.1524 m grid from the first to the last MD: depths with no
    row (e.g. missing VP) are kept and filled with the NULL value."""
    wdf = wdf.sort_values('MD')
    k = np.round(wdf['MD'].to_numpy(float) / step).astype(np.int64)
    wdf = wdf.assign(_k=k).drop_duplicates('_k').set_index('_k')
    k_grid = np.arange(k.min(), k.max() + 1) if len(k) else k
    wdf = wdf.reindex(k_grid)
    wdf['MD'] = np.round(k_grid * step, 4)
    las = lasio.LASFile()
    las.well['NULL'].value = null
    las.well['WELL'].value = well_name
    las.append_curve('DEPT', wdf['MD'].to_numpy(float), unit='m', descr='Measured depth')
    for col, (mn, unit, descr) in curves.items():
        if col in wdf.columns:
            las.append_curve(mn, pd.to_numeric(wdf[col], errors='coerce').to_numpy(float),
                             unit=unit, descr=descr)
    las.other = other_text
    las.write(str(path), version=2.0)
