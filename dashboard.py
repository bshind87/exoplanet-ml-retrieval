"""
INARA Exoplanet Atmospheric Retrieval — Interactive Dashboard

Run:
    streamlit run dashboard.py

Sections:
  1. Dataset Overview  — molecule distributions, spectral channels
  2. Model Metrics     — val/test R², RMSE, MAE for Baseline vs Deep Model
  3. Prediction Detail — scatter plots and residual distributions per molecule
  4. Training History  — deep model loss curves
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import streamlit as st

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title='INARA — Exoplanet Atmospheric Retrieval',
    page_icon='🌍',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ------------------------------------------------------------------
# Academic colour palette  (Nature / AAS journal style — light theme)
# ------------------------------------------------------------------
MOLECULE_NAMES = ['H2O', 'CO2', 'O2', 'O3', 'CH4', 'N2', 'N2O', 'CO', 'H2', 'H2S', 'SO2', 'NH3']
EXCLUDED_FROM_COMPARISON = {'H2O', 'NH3'}

# Two-model palette: muted steel-blue vs warm burnt-orange (colorblind-safe)
C_BASELINE = '#2166AC'   # deep steel blue
C_DEEP     = '#B2182B'   # deep crimson
C_GOOD     = '#1A7A3C'   # forest green
C_WARN     = '#B8860B'   # dark goldenrod
C_BAD      = '#CC3311'   # dark red
C_GRID     = 'rgba(26,95,168,0.12)'
C_ZERO     = 'rgba(26,95,168,0.35)'
C_BG       = '#EAF1FA'   # light blue plot area
C_PAPER    = '#D6E6F5'   # slightly deeper blue plot frame

PALETTE = {'baseline': C_BASELINE, 'deep': C_DEEP,
           'good': C_GOOD, 'warn': C_WARN, 'bad': C_BAD}

# Per-molecule colours — 12 dark, perceptually distinct hues readable on white
MOL_COLORS = [
    '#2166AC', '#4DAC26', '#B2182B', '#7B2D8B',
    '#8C6D31', '#1A8CA8', '#D55E00', '#117733',
    '#3690C0', '#C51B7D', '#8B6914', '#666666',
]

PLOT_LAYOUT = dict(
    plot_bgcolor=C_BG,
    paper_bgcolor=C_PAPER,
    font=dict(family='Arial, sans-serif', size=13, color='#1A1A2E'),
    title_font=dict(size=14, color='#0D0D1A'),
    legend=dict(
        bgcolor='rgba(214,230,245,0.95)',
        bordercolor='rgba(60,60,80,0.25)',
        borderwidth=1,
    ),
    xaxis=dict(
        showgrid=True, gridcolor=C_GRID, gridwidth=1,
        linecolor='rgba(60,60,80,0.4)', linewidth=1,
        tickfont=dict(size=12, color='#2A2A3E'),
        title_font=dict(color='#1A1A2E'),
    ),
    yaxis=dict(
        showgrid=True, gridcolor=C_GRID, gridwidth=1,
        linecolor='rgba(60,60,80,0.4)', linewidth=1,
        tickfont=dict(size=12, color='#2A2A3E'),
        title_font=dict(color='#1A1A2E'),
    ),
    margin=dict(t=55, b=45, l=60, r=20),
)

# ------------------------------------------------------------------
# Dataset registry
# ------------------------------------------------------------------
DATASETS = {
    'INARA ATMOS': {
        'data_dir':    Path('inara_data/processed'),
        'results_dir': Path('results/processed'),
        'label':       'INARA-ATMOS',
        'x_label':     'Altitude level (index)',
        'x_type':      'clima',
    },
}

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
st.sidebar.title('INARA Dashboard')
st.sidebar.markdown('NASA Exoplanet Atmospheric Retrieval')
st.sidebar.divider()

dataset_name = st.sidebar.selectbox('Dataset', list(DATASETS.keys()), index=0)
ds = DATASETS[dataset_name]
DATA_DIR    = ds['data_dir']
RESULTS_DIR = ds['results_dir']

section = st.sidebar.radio(
    'Section',
    ['📊 Dataset Overview', '📈 Model Metrics', '🔬 Prediction Detail', '📉 Training History'],
)
st.sidebar.divider()
st.sidebar.caption(dataset_name)

# ------------------------------------------------------------------
# Cached data loaders (mtime-busted so new files always load)
# ------------------------------------------------------------------
@st.cache_data
def load_raw_data(data_dir_str, _mtime=0):
    d = Path(data_dir_str)
    if not d.exists():
        return None, None, None
    spectra     = np.load(d / 'spectra.npy')
    molecules   = np.load(d / 'molecules.npy')
    wavelengths = np.load(d / 'wavelengths.npy')
    return spectra, molecules, wavelengths


@st.cache_data
def load_results(results_dir_str, _mtime=0):
    r = Path(results_dir_str)

    def _csv(fname):
        p = r / fname
        return pd.read_csv(p) if p.exists() else None

    def _npy(fname):
        p = r / fname
        return np.load(p) if p.exists() else None

    history = None
    hp = r / 'deep_training_history.csv'
    if hp.exists():
        history = pd.read_csv(hp)

    return (
        _csv('baseline_val_metrics.csv'),
        _csv('baseline_test_metrics.csv'),
        _csv('deep_val_metrics.csv'),
        _csv('deep_test_metrics.csv'),
        _npy('test_targets.npy'),
        _npy('baseline_test_pred.npy'),
        _npy('deep_test_pred.npy'),
        history,
    )


@st.cache_data
def load_dataset_info(data_dir_str):
    p = Path(data_dir_str) / 'dataset_info.json'
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def r2_color(r2):
    if not isinstance(r2, (int, float)) or np.isnan(r2): return C_WARN
    if r2 >= 0.7:  return C_GOOD
    if r2 >= 0.4:  return C_WARN
    return C_BAD


def _clip_r2(df):
    out = df.copy()
    if 'R2' in out.columns:
        out['R2'] = out['R2'].clip(lower=-1.5)
    return out


def apply_layout(fig, **overrides):
    """Apply the shared academic layout to any figure."""
    layout = {**PLOT_LAYOUT, **overrides}
    fig.update_layout(**layout)
    return fig


def metrics_bar_chart(bl_df, dl_df, metric='R2', split='Test', n_train_cnn=0):
    mols = MOLECULE_NAMES
    bl_vals = bl_df[bl_df['molecule'].isin(mols)].set_index('molecule')[metric].reindex(mols)
    dl_vals = (dl_df[dl_df['molecule'].isin(mols)].set_index('molecule')[metric].reindex(mols)
               if dl_df is not None else None)

    if metric == 'R2':
        bl_vals = bl_vals.clip(lower=-1.5)
        if dl_vals is not None:
            dl_vals = dl_vals.clip(lower=-1.5)

    fig = go.Figure()
    fig.add_bar(name='Baseline RF (10k)', x=mols, y=bl_vals,
                marker_color=C_BASELINE, opacity=0.85,
                marker_line=dict(color='rgba(60,60,80,0.12)', width=0.5))
    cnn_label = f'1D CNN ({n_train_cnn:,})' if n_train_cnn > 0 else '1D CNN'
    if dl_vals is not None:
        fig.add_bar(name=cnn_label, x=mols, y=dl_vals,
                    marker_color=C_DEEP, opacity=0.85,
                    marker_line=dict(color='rgba(60,60,80,0.12)', width=0.5))

    fig = apply_layout(fig,
        title=f'{split} {metric} per Molecule',
        barmode='group',
        xaxis_title='Molecule', yaxis_title=metric,
        legend=dict(orientation='h', y=1.14, x=0.5, xanchor='center',
                    bgcolor='rgba(214,230,245,0.95)', bordercolor='rgba(60,60,80,0.25)',
                    borderwidth=1),
        height=380,
    )
    if metric == 'R2':
        fig.add_hline(y=0,   line_dash='dot',  line_color=C_ZERO, line_width=1)
        fig.add_hline(y=0.7, line_dash='dash', line_color=C_GOOD, line_width=1,
                      annotation_text='R²=0.7', annotation_font_color=C_GOOD,
                      opacity=0.5)
    return fig


# ------------------------------------------------------------------
# Status banner (suppressed if log is older than 2h — run is done)
# ------------------------------------------------------------------
def show_status_banner():
    import time as _time
    log = Path('/tmp/inara_full_extract.log')
    if not log.exists():
        return
    # Don't show banner for stale logs
    if _time.time() - log.stat().st_mtime > 7200:
        return
    lines = log.read_text().splitlines()
    info_lines = [l for l in lines if 'INFO' in l or 'Done:' in l or 'complete' in l.lower()]
    if not info_lines:
        return
    for l in info_lines:
        if 'Processing complete' in l:
            st.success('✅ Extraction complete. Data ready.', icon='✅')
            return
    last = info_lines[-1]
    st.info(f'⏳ **Extraction in progress** — {last.split("INFO")[-1].strip()}', icon='🔄')


show_status_banner()

# ------------------------------------------------------------------
# Load active dataset
# ------------------------------------------------------------------
_data_mtime = int((DATA_DIR / 'spectra.npy').stat().st_mtime) if (DATA_DIR / 'spectra.npy').exists() else 0
_res_mtime  = int((RESULTS_DIR / 'deep_test_metrics.csv').stat().st_mtime) if (RESULTS_DIR / 'deep_test_metrics.csv').exists() else 0

spectra, molecules, wavelengths = load_raw_data(str(DATA_DIR), _mtime=_data_mtime)
bl_val, bl_test, dl_val, dl_test, targets, bl_pred, dl_pred, history = \
    load_results(str(RESULTS_DIR), _mtime=_res_mtime)
ds_info = load_dataset_info(str(DATA_DIR))

has_data    = spectra is not None
has_results = bl_test is not None
has_deep    = dl_test is not None

_n_samples = len(spectra) if has_data else 0
_n_train   = _n_samples - 2 * int(round(_n_samples * 0.15)) if _n_samples > 0 else 0

if has_data:
    st.sidebar.metric('Samples loaded', f'{_n_samples:,}')


# ====================================================================
# SECTION 1 — DATASET OVERVIEW
# ====================================================================
if section == '📊 Dataset Overview':
    st.title('Dataset Overview')
    st.markdown(f'**{dataset_name}**')

    if not has_data:
        st.warning(f'Data not found at `{DATA_DIR}`. Run `process_inara.py` first.')
        st.stop()

    n_samples, n_channels, seq_len = spectra.shape

    # --- Header metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Samples',    f'{n_samples:,}')
    c2.metric('Input Channels',   f'{n_channels}')
    c3.metric('Sequence Length',  f'{seq_len}')
    c4.metric('Target Molecules', '12')

    if ds_info:
        st.caption(
            f"Processed in {ds_info.get('processing_time_s', '?')}s  ·  "
            f"Source: {ds_info.get('source_dir', '?')}"
        )

    st.divider()

    # --- Molecule abundance violin plot ---
    st.subheader('Molecule Abundance Distributions (log₁₀ VMR)')
    mol_df = pd.DataFrame(molecules, columns=MOLECULE_NAMES)

    # Identify degenerate molecules (constant or quasi-constant)
    degenerate = {m: len(np.unique(np.round(mol_df[m].values, 4)))
                  for m in MOLECULE_NAMES}
    constant_mols   = [m for m, u in degenerate.items() if u == 1]
    quasiconstant_mols = [m for m, u in degenerate.items() if 1 < u <= 5]

    # Exclude constant molecules from axis calculation
    valid_cols = [m for m in MOLECULE_NAMES if m not in constant_mols]
    y_lo = np.floor(np.percentile(mol_df[valid_cols].values, 1))
    y_hi = np.ceil(np.percentile(mol_df[valid_cols].values, 99)) + 0.5

    outlier_mols = [m for m in valid_cols
                    if mol_df[m].min() < y_lo or mol_df[m].max() > y_hi]

    fig_vio = go.Figure()
    for i, mol in enumerate(MOLECULE_NAMES):
        if mol in constant_mols:
            # Show a single point marker instead of violin for constant molecules
            fig_vio.add_trace(go.Box(
                y=[mol_df[mol].iloc[0]],
                name=mol,
                x=[mol],
                marker=dict(color='#AAAAAA', size=8, symbol='x'),
                line_color='#AAAAAA',
                fillcolor='rgba(0,0,0,0)',
                showlegend=False,
            ))
            continue
        clipped = mol_df[mol].clip(lower=y_lo, upper=y_hi)
        c = MOL_COLORS[i]
        fig_vio.add_trace(go.Violin(
            y=clipped,
            name=mol,
            x=[mol] * len(clipped),
            box_visible=True,
            meanline_visible=True,
            meanline_color='#1A1A2E',
            fillcolor=c,
            line_color=c,
            box_fillcolor='rgba(60,60,80,0.06)',
            box_line_color='rgba(60,60,80,0.5)',
            points=False,
            opacity=0.82,
            scalemode='width',
            width=0.8,
            spanmode='soft',
        ))

    fig_vio = apply_layout(fig_vio,
        height=440,
        showlegend=False,
        xaxis_title='Molecule',
        yaxis_title='log₁₀ Volume Mixing Ratio',
        yaxis=dict(range=[y_lo, y_hi], showgrid=True, gridcolor=C_GRID,
                   tickfont=dict(size=12)),
        xaxis=dict(showgrid=False, tickfont=dict(size=13)),
        violingap=0.15,
        violingroupgap=0,
    )
    st.plotly_chart(fig_vio, use_container_width=True)

    notes = []
    if constant_mols:
        notes.append(
            f'**{", ".join(constant_mols)}** — constant sentinel value (×, grey): '
            f'always at the detection-limit floor, not a learnable target.'
        )
    if quasiconstant_mols:
        notes.append(
            f'**{", ".join(quasiconstant_mols)}** — quasi-constant (≤5 unique values): '
            f'R² is unreliable for this molecule.'
        )
    if outlier_mols:
        notes.append(
            f'Y-axis clipped to [{y_lo:.0f}, {y_hi:.1f}] (1st–99th percentile of valid molecules). '
            f'Values outside window: **{", ".join(outlier_mols)}** (display only).'
        )
    for note in notes:
        st.caption(note)

    # --- Summary statistics table ---
    st.subheader('Descriptive Statistics (log₁₀ VMR)')
    stats = mol_df.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    stats.columns = ['Mean', 'Std', 'Min', 'Q1', 'Median', 'Q3', 'Max']
    st.dataframe(stats.style.format('{:.3f}'), use_container_width=True)

    st.divider()

    # --- Train / Val / Test split ---
    st.subheader('Train / Validation / Test Split')
    n_test  = len(targets) if targets is not None else int(round(n_samples * 0.15))
    n_val   = len(bl_val[bl_val['molecule'] == 'H2O']) if (bl_val is not None and 'molecule' in bl_val.columns) else int(round(n_samples * 0.15))
    # Derive val count from actual data split: both val/test are 15% of full dataset
    n_val   = int(round(n_samples * 0.15))
    n_train_full = n_samples - n_val - n_test
    n_train_bl   = 10_000   # baseline was capped at 10k
    split_df = pd.DataFrame({
        'Split':      ['Train (RF, capped)', 'Train (CNN, full)', 'Validation', 'Test'],
        'Samples':    [n_train_bl, n_train_full, n_val, n_test],
        'Fraction':   [
            f'{n_train_bl/n_samples:.1%}',
            f'{n_train_full/n_samples:.1%}',
            f'{n_val/n_samples:.1%}',
            f'{n_test/n_samples:.1%}',
        ],
    })
    st.dataframe(split_df, hide_index=True, use_container_width=False)

    st.divider()

    # --- Channel samples ---
    st.subheader(f'Input Channel Samples ({ds["x_label"]})')
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n_samples, size=min(5, n_samples), replace=False)
    x_axis = wavelengths if len(wavelengths) == seq_len else np.arange(seq_len)

    channel_labels = (
        ['Ch0: Star+Planet', 'Ch1: Stellar residual', 'Ch2: Transit depth']
        if ds['x_type'] == 'spectra' else
        [f'Ch{i}: {n}' for i, n in enumerate(
            ds_info.get('clima_channels', [f'Ch{i}' for i in range(n_channels)])
        )]
    )
    cols_to_show = min(3, n_channels)
    ch_cols = st.columns(cols_to_show)
    for c_idx, col in enumerate(ch_cols):
        with col:
            fig = go.Figure()
            for si, s in enumerate(sample_idx):
                fig.add_trace(go.Scatter(
                    x=x_axis, y=spectra[s, c_idx, :],
                    mode='lines',
                    line=dict(width=1.2, color=MOL_COLORS[si % len(MOL_COLORS)]),
                    opacity=0.75, showlegend=False,
                ))
            label = channel_labels[c_idx] if c_idx < len(channel_labels) else f'Ch{c_idx}'
            fig = apply_layout(fig,
                title=label, height=280,
                xaxis_title=ds['x_label'], yaxis_title='Normalised value',
                margin=dict(t=45, b=40, l=50, r=15),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Correlation heatmap ---
    st.subheader('Inter-Molecule Correlation (log₁₀ VMR)')
    corr = mol_df.corr()
    fig_corr = px.imshow(
        corr, text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1, aspect='auto', height=430,
    )
    fig_corr.update_traces(textfont_size=11)
    fig_corr.update_layout(
        paper_bgcolor='#D6E6F5',
        font=dict(family='Arial, sans-serif', size=12, color='#1A1A2E'),
        margin=dict(t=20, b=20, l=20, r=20),
        coloraxis_colorbar=dict(title='r', tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ====================================================================
# SECTION 2 — MODEL METRICS
# ====================================================================
elif section == '📈 Model Metrics':
    st.title('Model Metrics')

    if not has_results:
        st.warning(f'No results found at `{RESULTS_DIR}`. Run the training scripts first.')
        st.code(f'python run_baseline.py --data-dir {DATA_DIR} --max-train-samples 10000\n'
                f'python run_deep_model.py --data-dir {DATA_DIR} '
                f'--in-channels {ds_info.get("in_channels", 3)}')
        st.stop()

    def _mean_r2(df, exclude=None):
        if df is None: return float('nan')
        exclude = set(exclude or [])
        keep = df['molecule'].isin(MOLECULE_NAMES) & ~df['molecule'].isin(exclude)
        return df[keep]['R2'].clip(lower=-1.5).mean()

    # --- Summary cards ---
    st.subheader('Overall Performance Summary')
    cols = st.columns(4)
    cols[0].metric('Baseline Val R²',    f"{_mean_r2(bl_val, EXCLUDED_FROM_COMPARISON):.4f}"   if bl_val  is not None else 'N/A')
    cols[1].metric('Baseline Test R²',   f"{_mean_r2(bl_test, EXCLUDED_FROM_COMPARISON):.4f}"  if bl_test is not None else 'N/A')
    cols[2].metric('1D CNN Val R²', f"{_mean_r2(dl_val, EXCLUDED_FROM_COMPARISON):.4f}" if dl_val is not None else 'N/A')
    cols[3].metric(
        '1D CNN Test R²',
        f"{_mean_r2(dl_test, EXCLUDED_FROM_COMPARISON):.4f}" if dl_test is not None else 'N/A',
        delta=(f"{_mean_r2(dl_test, EXCLUDED_FROM_COMPARISON)-_mean_r2(bl_test, EXCLUDED_FROM_COMPARISON):+.4f} vs baseline"
               if (dl_test is not None and bl_test is not None) else None),
    )
    st.caption('Summary cards and comparison metrics exclude H2O and NH3 because both are degenerate on this split and can distort mean R².')

    st.divider()

    metric = st.selectbox('Metric', ['R2', 'RMSE', 'MAE'], index=0)
    split  = st.radio('Split', ['Validation', 'Test'], horizontal=True)

    bl_df = bl_val  if split == 'Validation' else bl_test
    dl_df = dl_val  if split == 'Validation' else dl_test

    if bl_df is not None:
        st.plotly_chart(
            metrics_bar_chart(
                _clip_r2(bl_df) if metric == 'R2' else bl_df,
                (_clip_r2(dl_df) if metric == 'R2' else dl_df) if dl_df is not None else None,
                metric=metric, split=split, n_train_cnn=_n_train,
            ),
            use_container_width=True,
        )

    st.divider()
    st.subheader(f'Per-Molecule {split} Metrics')

    def style_r2(df):
        if df is None: return None
        df = _clip_r2(df).copy()
        def color_r2(val):
            if not isinstance(val, float) or np.isnan(val): return ''
            c = r2_color(val)
            return f'background-color: {c}20; color: {c}'
        return df.style.map(color_r2, subset=['R2']).format(
            {'R2': '{:.4f}', 'RMSE': '{:.4f}', 'MAE': '{:.4f}'}
        )

    col_bl, col_dl = st.columns(2)
    with col_bl:
        st.markdown('**Baseline RF** *(10 000 training samples)*')
        if bl_df is not None:
            st.dataframe(style_r2(bl_df), hide_index=True, use_container_width=True)
        else:
            st.info('No baseline results.')
    with col_dl:
        cnn_train_label = f'{_n_train:,}' if _n_train > 0 else 'full'
        st.markdown(f'**1D CNN** *({cnn_train_label} training samples)*')
        if dl_df is not None:
            st.dataframe(style_r2(dl_df), hide_index=True, use_container_width=True)
        else:
            st.info('Deep model not trained yet.')

    # --- Comparison delta (always recomputed live) ---
    if bl_test is not None and dl_test is not None and split == 'Test':
        _bl = (bl_test[bl_test['molecule'].isin(MOLECULE_NAMES)][['molecule', 'R2']]
               .rename(columns={'R2': 'Baseline_R2'}))
        _dl = (dl_test[dl_test['molecule'].isin(MOLECULE_NAMES)][['molecule', 'R2']]
               .rename(columns={'R2': 'DeepModel_R2'}))
        cmp_live = pd.merge(_bl, _dl, on='molecule')
        cmp_live['DeepModel_R2'] = cmp_live['DeepModel_R2'].clip(lower=-1.5)
        cmp_live['Delta_R2']     = cmp_live['DeepModel_R2'] - cmp_live['Baseline_R2']

        st.divider()
        st.subheader('R² Gain: 1D CNN vs Baseline RF (Test Set)')

        cmp_live = cmp_live[~cmp_live['molecule'].isin(EXCLUDED_FROM_COMPARISON)].copy()

        def color_delta(val):
            if not isinstance(val, float) or np.isnan(val): return ''
            if val > 0.05:  return f'color: {C_GOOD}'
            if val < -0.05: return f'color: {C_BAD}'
            return f'color: {C_WARN}'

        styled = cmp_live.style.map(color_delta, subset=['Delta_R2']).format(
            {'Baseline_R2': '{:.4f}', 'DeepModel_R2': '{:.4f}', 'Delta_R2': '{:+.4f}'}
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)

        colors = [C_GOOD if v > 0 else C_BAD for v in cmp_live['Delta_R2']]
        fig_delta = go.Figure(go.Bar(
            x=cmp_live['molecule'], y=cmp_live['Delta_R2'],
            marker_color=colors,
            marker_line=dict(color='rgba(60,60,80,0.12)', width=0.5),
            opacity=0.88,
            text=cmp_live['Delta_R2'].apply(lambda v: f'{v:+.3f}'),
            textposition='outside',
            textfont=dict(size=11),
        ))
        fig_delta.add_hline(y=0, line_color='rgba(60,60,80,0.4)', line_width=1)
        fig_delta = apply_layout(fig_delta,
            title='ΔR² = 1D CNN − Baseline RF  (positive = CNN advantage)',
            xaxis_title='Molecule', yaxis_title='ΔR²', height=360,
            margin=dict(t=60, b=45, l=60, r=20),
        )
        st.plotly_chart(fig_delta, use_container_width=True)


# ====================================================================
# SECTION 3 — PREDICTION DETAIL
# ====================================================================
elif section == '🔬 Prediction Detail':
    from sklearn.metrics import r2_score, mean_absolute_error

    st.title('Prediction Detail — Test Set')

    if targets is None or bl_pred is None:
        st.warning('Run baseline training first.')
        st.stop()

    mol_sel = st.selectbox('Select Molecule', MOLECULE_NAMES, index=0)
    mol_idx = MOLECULE_NAMES.index(mol_sel)

    y_true = targets[:, mol_idx]
    y_bl   = bl_pred[:, mol_idx]
    y_var  = y_true.var()

    r2_bl = r2_score(y_true, y_bl) if y_var > 1e-10 else float('nan')

    if dl_pred is not None:
        y_dl  = dl_pred[:, mol_idx]
        r2_dl = r2_score(y_true, y_dl) if y_var > 1e-10 else float('nan')
    else:
        y_dl, r2_dl = None, float('nan')

    delta_r2 = (r2_dl - r2_bl) if (not np.isnan(r2_bl) and not np.isnan(r2_dl)) else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Baseline R²',         f'{r2_bl:.4f}'  if not np.isnan(r2_bl) else 'N/A')
    c2.metric('1D CNN R²',   f'{r2_dl:.4f}'  if not np.isnan(r2_dl) else 'N/A',
              delta=f'{delta_r2:+.4f} vs baseline'   if delta_r2 is not None else None)
    c3.metric('Baseline MAE',        f'{mean_absolute_error(y_true, y_bl):.4f}')
    c4.metric('CNN MAE',          f'{mean_absolute_error(y_true, y_dl):.4f}' if y_dl is not None else 'N/A')
    c5.metric('Test samples',        f'{len(y_true):,}')

    # --- Degenerate feature detection ---
    n_unique    = len(np.unique(np.round(y_true, 6)))
    is_constant = y_var < 1e-10                        # NH3: literally one value
    is_quasi    = (not is_constant) and n_unique <= 5  # H2O: only 2 discrete values

    if is_constant:
        st.warning(
            f'**{mol_sel} is a degenerate target.** '
            f'All {len(y_true):,} test samples share the same value '
            f'({y_true[0]:.4f} log₁₀ = 10^{y_true[0]:.0f}). '
            f'This is a sentinel/floor value — {mol_sel} is always below the detection '
            f'limit in this dataset and carries no predictive information. '
            f'It should be excluded from model evaluation.'
        )
        st.stop()

    if is_quasi:
        st.info(
            f'**{mol_sel} is quasi-constant.** '
            f'Only {n_unique} discrete values exist in the test set '
            f'(std = {y_true.std():.2e}). '
            f'R² is unreliable for near-binary targets. '
            f'The scatter plot shows discrete clusters, not a continuous distribution — '
            f'this reflects limited variability in the source dataset, not a model failure.'
        )

    st.divider()

    def scatter_pred(y_true, y_pred, label):
        valid_r2 = y_var > 1e-10
        r2_val   = r2_score(y_true, y_pred) if valid_r2 else float('nan')
        r2_txt   = f'R² = {r2_val:.4f}' if valid_r2 else 'R² = N/A'
        pad  = max((y_true.max() - y_true.min()) * 0.08, 0.02)
        lims = [min(y_true.min(), y_pred.min()) - pad,
                max(y_true.max(), y_pred.max()) + pad]
        residuals = y_pred - y_true
        r_abs = np.abs(residuals).max() or 1.0

        # For quasi-constant molecules use jitter to reveal stacked points
        jitter = np.random.default_rng(0).normal(0, pad * 0.04, size=len(y_true)) if is_quasi else 0

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lims, y=lims, mode='lines',
            line=dict(dash='dash', color='rgba(60,60,80,0.35)', width=1.2),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=y_true + jitter, y=y_pred,
            mode='markers',
            marker=dict(
                color=residuals, colorscale='RdBu_r',
                size=4 if not is_quasi else 3,
                opacity=0.55 if is_quasi else 0.65,
                cmin=-r_abs, cmax=r_abs,
                colorbar=dict(title='Residual', len=0.55, thickness=12,
                              tickfont=dict(size=10)),
                line=dict(width=0),
            ),
            hovertemplate='True: %{x:.4f}<br>Pred: %{y:.4f}<extra></extra>',
            showlegend=False,
        ))
        fig = apply_layout(fig,
            title=f'{label}  —  {r2_txt}',
            xaxis_title=f'{mol_sel} true (log₁₀)' + (' [jittered]' if is_quasi else ''),
            yaxis_title=f'{mol_sel} predicted (log₁₀)',
            height=400,
            xaxis=dict(range=lims, showgrid=True, gridcolor=C_GRID),
            yaxis=dict(range=lims, showgrid=True, gridcolor=C_GRID),
            margin=dict(t=50, b=50, l=60, r=70),
        )
        return fig

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(scatter_pred(y_true, y_bl, 'Baseline RF'),
                        use_container_width=True)
    with col_r:
        if y_dl is not None:
            y_dl_c = np.clip(y_dl, y_true.min() - 3, y_true.max() + 3)
            st.plotly_chart(scatter_pred(y_true, y_dl_c, '1D CNN'),
                            use_container_width=True)
        else:
            st.info('Deep model predictions not available.')

    # --- Residual distribution ---
    st.subheader('Residual Distribution (Predicted − True)')
    fig_res = go.Figure()
    res_bl = y_bl - y_true
    fig_res.add_trace(go.Histogram(
        x=res_bl, name='Baseline RF',
        marker_color=C_BASELINE, opacity=0.70,
        nbinsx=60, histnorm='probability density',
        marker_line=dict(color='rgba(60,60,80,0.06)', width=0.5),
    ))
    if y_dl is not None:
        res_dl = np.clip(y_dl, y_true.min() - 3, y_true.max() + 3) - y_true
        fig_res.add_trace(go.Histogram(
            x=res_dl, name='1D CNN',
            marker_color=C_DEEP, opacity=0.70,
            nbinsx=60, histnorm='probability density',
            marker_line=dict(color='rgba(60,60,80,0.06)', width=0.5),
        ))
    fig_res.add_vline(x=0, line_dash='dash', line_color='rgba(60,60,80,0.4)', line_width=1)
    fig_res = apply_layout(fig_res,
        barmode='overlay', height=300,
        xaxis_title='Residual (log₁₀)', yaxis_title='Probability density',
        legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center',
                    bgcolor='rgba(214,230,245,0.95)', borderwidth=1),
        margin=dict(t=45, b=45, l=60, r=20),
    )
    st.plotly_chart(fig_res, use_container_width=True)
    if is_quasi:
        st.caption(
            f'Residual histogram shows discrete spikes due to {mol_sel} having only '
            f'{n_unique} unique values — this is a dataset property, not a model artefact.'
        )

    # --- All-molecule R² heatmap ---
    st.divider()
    st.subheader('All-Molecule R² — Test Set')

    from sklearn.metrics import r2_score as _r2
    rows, row_names = [], []

    bl_r2_all = [max(_r2(targets[:, i], bl_pred[:, i]), -1.5)
                 if targets[:, i].var() > 1e-10 else float('nan')
                 for i in range(12)]
    rows.append(bl_r2_all); row_names.append('Baseline RF')

    if dl_pred is not None:
        dl_r2_all = [max(_r2(targets[:, i], dl_pred[:, i]), -1.5)
                     if targets[:, i].var() > 1e-10 else float('nan')
                     for i in range(12)]
        rows.append(dl_r2_all); row_names.append('1D CNN')

    fig_hm = px.imshow(
        np.array(rows, dtype=float),
        x=MOLECULE_NAMES, y=row_names,
        color_continuous_scale=[
            [0.0, '#CC3311'], [0.5, '#E6A817'], [0.8, '#2CA02C'], [1.0, '#1a5c1a']
        ],
        zmin=0.6, zmax=1.0,
        text_auto='.3f', aspect='auto',
        height=200 if len(rows) == 2 else 150,
    )
    fig_hm.update_traces(textfont_size=11)
    fig_hm.update_layout(
        paper_bgcolor='#D6E6F5',
        font=dict(family='Arial, sans-serif', size=12, color='#1A1A2E'),
        margin=dict(t=20, b=20, l=20, r=20),
        coloraxis_colorbar=dict(title='R²', tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_hm, use_container_width=True)


# ====================================================================
# SECTION 4 — TRAINING HISTORY
# ====================================================================
elif section == '📉 Training History':
    st.title('Deep Model Training History')

    if history is None:
        st.warning(f'No training history found at `{RESULTS_DIR}`.')
        if not has_deep:
            st.code(f'python run_deep_model.py --data-dir {DATA_DIR} '
                    f'--in-channels {ds_info.get("in_channels", 3)}')
        st.stop()

    n_epochs   = len(history)
    best_epoch = history['val_loss'].idxmin() + 1
    best_val   = history['val_loss'].min()

    c1, c2, c3 = st.columns(3)
    c1.metric('Epochs trained', n_epochs)
    c2.metric('Best epoch',     best_epoch)
    c3.metric('Best val loss',  f'{best_val:.5f}')

    st.divider()
    epochs = list(range(1, n_epochs + 1))

    # --- Loss curves ---
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=history['train_loss'], name='Training loss',
        line=dict(color=C_BASELINE, width=2),
    ))
    fig_loss.add_trace(go.Scatter(
        x=epochs, y=history['val_loss'], name='Validation loss',
        line=dict(color=C_DEEP, width=2),
    ))
    fig_loss.add_vline(
        x=best_epoch, line_dash='dash', line_color=C_GOOD, line_width=1.2,
        annotation_text=f'Best epoch {best_epoch}',
        annotation_font_color=C_GOOD, annotation_position='top right',
    )
    fig_loss = apply_layout(fig_loss,
        title='Training & Validation Loss (Weighted MSE)',
        xaxis_title='Epoch', yaxis_title='Loss', height=400,
        legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center',
                    bgcolor='rgba(214,230,245,0.95)', borderwidth=1),
    )
    st.plotly_chart(fig_loss, use_container_width=True)

    # --- Log-scale ---
    fig_log = go.Figure()
    fig_log.add_trace(go.Scatter(x=epochs, y=history['train_loss'],
                                 name='Training loss',
                                 line=dict(color=C_BASELINE, width=2)))
    fig_log.add_trace(go.Scatter(x=epochs, y=history['val_loss'],
                                 name='Validation loss',
                                 line=dict(color=C_DEEP, width=2)))
    fig_log = apply_layout(fig_log,
        title='Loss — log scale',
        xaxis_title='Epoch', yaxis_title='Loss (log)', height=320,
        yaxis=dict(type='log', showgrid=True, gridcolor=C_GRID),
        legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center',
                    bgcolor='rgba(214,230,245,0.95)', borderwidth=1),
    )
    st.plotly_chart(fig_log, use_container_width=True)

    with st.expander('Raw history (last 20 epochs)'):
        tail = history.tail(20).copy()
        tail.index = range(n_epochs - len(tail) + 1, n_epochs + 1)
        tail.index.name = 'epoch'
        st.dataframe(tail.style.format('{:.6f}'))
