#!/usr/bin/env python
"""
process_inara.py — Extract and process N samples from the full INARA ATMOS dataset.

Dataset layout (inside /path/to/full_inara/):
  pyatmos_summary.csv      — master index: hash, concentrations, pressure, temperature
  dir_0.tar.gz ... dir_9.tar.gz — each ~7800 simulation samples
  Dir_alpha.tar.gz              — additional ~46 000 samples

Each sample directory (hash-named) contains:
  parsed_clima_final.npy.npz          → (101, 12) climate profile  [FEATURES]
  parsed_photochem_mixing_ratios.npy.npz → (102, 66) chem profile [TARGETS]
  mixing_ratios.dat                   → surface N2 concentration   [TARGET]
  run_metadata.json                   → input fluxes + conditions  [AUX]

Output (saved to --output-dir):
  spectra.npy       (N, 12, 101) float32  — CLIMA profile (channels=12, seq=101)
  molecules.npy     (N, 12)      float32  — log10 surface mol. concentrations
  aux_params.npy    (N, 11)      float32  — metadata scalars
  system_ids.npy    (N,)         object   — sample hash strings
  wavelengths.npy   (101,)       float64  — altitude axis (km)
  dataset_info.json               — dataset statistics + column names

Molecule order (same as existing model):
  H2O, CO2, O2, O3, CH4, N2, N2O, CO, H2, H2S, SO2, NH3

Usage:
  python process_inara.py \\
    --source-dir /Users/bhalchandra/Downloads/full_inara \\
    --output-dir inara_data/processed_inara \\
    --n-samples 10000 \\
    --n-workers 4 \\
    --seed 42

M5 MacBook Pro recommendations (unified RAM):
  --n-samples 5000   → quick dev/test  (~30s, 12 MB output)
  --n-samples 20000  → good baseline   (~2 min, 50 MB)
  --n-samples 50000  → strong model    (~5 min, 120 MB)
  --n-samples 124000 → full dataset    (~12 min, 300 MB, needs 3+ GB RAM)
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOLECULE_NAMES = ['H2O', 'CO2', 'O2', 'O3', 'CH4', 'N2', 'N2O', 'CO', 'H2', 'H2S', 'SO2', 'NH3']

# Column indices in parsed_photochem_mixing_ratios.npy.npz arr_0 (102, 66)
# Col 0 = row-index stored in array; col 1 = Z (altitude); then species follow.
# Separator (Z) columns repeat altitude at positions 1, 20, 39, 58.
PHOTOCHEM_COLS = {
    'H2O': 4, 'CO2': 48, 'O2': 3, 'O3': 37, 'CH4': 17,
    'N2O': 42, 'CO': 10, 'H2': 9, 'H2S': 24, 'SO2': 28, 'NH3': 47,
}
# N2 comes from mixing_ratios.dat (background carrier gas, not in photochem array)

# CLIMA column indices in parsed_clima_final.npy.npz arr_0 (101, 12)
# J, P, ALT, T, CONVEC, DT, TOLD, FH20, FSAVE, FO3, TCOOL, THEAT
CLIMA_COLS = ['J', 'P', 'ALT', 'T', 'CONVEC', 'DT', 'TOLD', 'FH20', 'FSAVE', 'FO3', 'TCOOL', 'THEAT']
N_CLIMA_VARS = 12   # channels for the CNN

# Aux-param metadata keys from run_metadata.json
AUX_KEYS = ['flux_CH4', 'flux_CO', 'flux_CO2', 'flux_H2O', 'flux_NH3',
            'flux_O3', 'pressure', 'temperature',
            'input_CH4', 'input_CO2', 'input_O2']

LOG_FLOOR = -40.0   # minimum log10 value (replace -inf / very negative logs)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(output_dir: Path, verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = '%(asctime)s  %(levelname)-8s  %(message)s'
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / 'process_inara.log', mode='a'),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Build hash → archive index
# ---------------------------------------------------------------------------
def build_archive_index(source_dir: Path, index_cache: Path, logger) -> dict:
    """
    Scan all tar.gz archives to build {hash: archive_path} mapping.
    Result is cached to avoid re-scanning on subsequent runs.
    """
    if index_cache.exists():
        logger.info(f'Loading cached archive index from {index_cache}')
        with open(index_cache) as f:
            raw = json.load(f)
        return {k: Path(v) for k, v in raw.items()}

    logger.info('Building archive index (one-time scan of all tar.gz files) ...')
    archives = sorted(source_dir.glob('*.tar.gz'))
    index = {}
    for archive in archives:
        logger.info(f'  Scanning {archive.name} ...')
        t0 = time.time()
        count = 0
        with tarfile.open(archive, 'r:gz') as tf:
            for member in tf.getmembers():
                if member.name.endswith('/run_metadata.json'):
                    # path: dir_0/07c2d2.../run_metadata.json
                    parts = member.name.split('/')
                    if len(parts) >= 2:
                        hash_id = parts[1]
                        index[hash_id] = str(archive)
                        count += 1
        logger.info(f'    {count:6d} samples  ({time.time()-t0:.1f}s)')

    logger.info(f'Total samples in all archives: {len(index):,}')
    with open(index_cache, 'w') as f:
        json.dump(index, f)
    logger.info(f'Archive index cached to {index_cache}')
    return {k: Path(v) for k, v in index.items()}


# ---------------------------------------------------------------------------
# Step 2: Sample N hashes from pyatmos_summary.csv
# ---------------------------------------------------------------------------
def sample_hashes(source_dir: Path, archive_index: dict, n_samples: int,
                  seed: int, logger) -> list:
    """
    Read pyatmos_summary.csv, keep only hashes that exist in archives,
    and return a random sample of n_samples (hash, archive_path) tuples.
    """
    summary_path = source_dir / 'pyatmos_summary.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f'pyatmos_summary.csv not found at {source_dir}')

    df = pd.read_csv(summary_path)
    logger.info(f'pyatmos_summary.csv: {len(df):,} rows')

    # Keep only samples that have been indexed (archive exists)
    available = df[df['hash'].isin(archive_index)].copy()
    logger.info(f'Samples with matching archive entry: {len(available):,}')

    if n_samples > len(available):
        logger.warning(f'Requested {n_samples} but only {len(available):,} available. '
                       f'Using all.')
        n_samples = len(available)

    sampled = available.sample(n=n_samples, random_state=seed)
    pairs = [(row['hash'], archive_index[row['hash']]) for _, row in sampled.iterrows()]
    logger.info(f'Sampled {len(pairs):,} hashes across '
                f'{len(set(str(p) for _, p in pairs))} archives')
    return pairs


# ---------------------------------------------------------------------------
# Step 3: Per-sample processing helpers
# ---------------------------------------------------------------------------
def _parse_n2_from_dat(dat_bytes: bytes) -> float:
    """Parse surface N2 mixing ratio from mixing_ratios.dat byte content."""
    for line in dat_bytes.decode('utf-8', errors='replace').splitlines():
        parts = line.strip().split()
        # Match exactly "!Nitrogen" (not "!Nitrogen Dioxide")
        if len(parts) >= 2 and parts[1] == '!Nitrogen':
            try:
                return float(parts[0])
            except ValueError:
                pass
    return np.nan


def _safe_log10(arr: np.ndarray) -> np.ndarray:
    """log10 with floor at LOG_FLOOR to avoid -inf."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.log10(np.abs(arr))
    return np.clip(result, LOG_FLOOR, None).astype(np.float32)


def _process_file_contents(hash_id: str, file_contents: dict):
    """
    Process pre-loaded file bytes for one sample.
    Returns (hash_id, clima_arr, mol_arr, aux_arr) or None on failure.
    """

    # Use .get() for all keys — not every file may be present in the archive
    clima_bytes    = file_contents.get('clima')
    photochem_bytes = file_contents.get('photochem')
    mixing_bytes   = file_contents.get('mixing')
    meta_bytes     = file_contents.get('meta')

    # ---- CLIMA profile (features) ----------------------------------------
    if not clima_bytes:
        return None
    try:
        clima_npz = np.load(io.BytesIO(clima_bytes))
        clima_arr = clima_npz['arr_0'].astype(np.float32)  # (101, 12)
        if clima_arr.shape != (101, 12):
            return None
        # Transpose to (12, 101) for CNN: (channels, seq_len)
        clima_arr = clima_arr.T   # (12, 101)
    except Exception:
        return None

    # ---- Photochem surface concentrations (targets) -----------------------
    if not photochem_bytes:
        return None
    try:
        pc_npz = np.load(io.BytesIO(photochem_bytes))
        pc_arr = pc_npz['arr_0']   # (102, 66)
        if pc_arr.shape[1] < 49:
            return None
        surface = pc_arr[0]        # first row = surface layer

        mol_vals = []
        for mol in MOLECULE_NAMES:
            if mol == 'N2':
                mol_vals.append(np.nan)   # filled below
            else:
                col = PHOTOCHEM_COLS[mol]
                mol_vals.append(float(surface[col]))
    except Exception:
        return None

    # ---- N2 from mixing_ratios.dat ----------------------------------------
    n2 = np.nan
    if mixing_bytes:
        n2 = _parse_n2_from_dat(mixing_bytes)
    if np.isnan(n2):
        # Fallback: N2 ≈ 1 - H2O(col4) - O2(col3) (major species complement)
        try:
            n2 = max(0.0, 1.0 - float(surface[4]) - float(surface[3]))
        except Exception:
            n2 = 0.79

    mol_vals[MOLECULE_NAMES.index('N2')] = n2
    mol_arr = _safe_log10(np.array(mol_vals, dtype=np.float64))  # (12,)

    # ---- Aux params from run_metadata.json --------------------------------
    aux = np.full(len(AUX_KEYS), np.nan, dtype=np.float32)
    if meta_bytes:
        try:
            meta = json.loads(meta_bytes.decode('utf-8'))
            for i, key in enumerate(AUX_KEYS):
                if key in meta:
                    aux[i] = float(meta[key])
                elif key.startswith('input_') and 'input_species_concentrations' in meta:
                    species = key.replace('input_', '')
                    aux[i] = float(meta['input_species_concentrations'].get(species, np.nan))
        except Exception:
            pass

    return hash_id, clima_arr, mol_arr, aux


# ---------------------------------------------------------------------------
# Step 4: Single-pass archive extraction (key performance optimization)
# ---------------------------------------------------------------------------
WANTED_FILENAMES = frozenset([
    'parsed_clima_final.npy.npz',
    'parsed_photochem_mixing_ratios.npy.npz',
    'mixing_ratios.dat',
    'run_metadata.json',
])

FILE_KEY_MAP = {
    'parsed_clima_final.npy.npz':               'clima',
    'parsed_photochem_mixing_ratios.npy.npz':   'photochem',
    'mixing_ratios.dat':                         'mixing',
    'run_metadata.json':                         'meta',
}


def process_archive_single_pass(archive_path: Path, target_hashes: set,
                                 logger) -> list:
    """
    ONE sequential scan of the archive to collect all needed files,
    then process each sample in memory.

    gzip archives do not support random access — opening the archive N times
    for N samples restarts decompression each time. A single pass is always
    faster regardless of sample count.
    """
    # Buffer: hash_id → {key: bytes}
    buffers = {h: {} for h in target_hashes}
    found = 0

    with tarfile.open(str(archive_path), 'r:gz') as tf:
        with tqdm(desc=archive_path.name[:20], leave=False, unit='entry') as pbar:
            for member in tf:
                pbar.update(1)
                if not member.isfile():
                    continue
                parts = member.name.split('/')
                if len(parts) < 3:
                    continue
                hash_id = parts[1]
                filename = parts[-1]

                if hash_id not in buffers or filename not in WANTED_FILENAMES:
                    continue

                try:
                    buf = tf.extractfile(member)
                    if buf is not None:
                        buffers[hash_id][FILE_KEY_MAP[filename]] = buf.read()
                        found += 1
                except Exception:
                    pass

    # Process each collected sample
    results = []
    for hash_id, file_contents in buffers.items():
        result = _process_file_contents(hash_id, file_contents)
        if result is not None:
            results.append(result)

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Extract and process N samples from the INARA ATMOS dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--source-dir', type=Path,
                        default=Path('/Users/bhalchandra/Downloads/full_inara'),
                        help='Directory containing tar.gz archives + pyatmos_summary.csv')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('inara_data/processed'),
                        help='Output directory for processed numpy arrays (overwrites existing files)')
    parser.add_argument('--n-samples', type=int, default=10000,
                        help='Number of samples to extract (max ~124 000)')
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Parallel workers for processing (per-archive parallelism)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint if previous run was interrupted')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation checks on the extracted data')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir, args.verbose)
    logger.info('=' * 60)
    logger.info('INARA Data Processing Pipeline')
    logger.info(f'  source_dir  : {args.source_dir}')
    logger.info(f'  output_dir  : {args.output_dir}')
    logger.info(f'  n_samples   : {args.n_samples:,}')
    logger.info(f'  n_workers   : {args.n_workers}')
    logger.info(f'  seed        : {args.seed}')
    logger.info('=' * 60)

    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Build / load archive index
    # ------------------------------------------------------------------
    index_cache = args.output_dir / 'archive_index.json'
    archive_index = build_archive_index(args.source_dir, index_cache, logger)

    # ------------------------------------------------------------------
    # 2. Sample hashes
    # ------------------------------------------------------------------
    sample_pairs = sample_hashes(
        args.source_dir, archive_index, args.n_samples, args.seed, logger
    )

    # ------------------------------------------------------------------
    # 3. Resume: skip already-processed hashes
    # ------------------------------------------------------------------
    checkpoint_path = args.output_dir / 'checkpoint.json'
    done_hashes = set()
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            done_hashes = set(json.load(f).get('done_hashes', []))
        sample_pairs = [(h, a) for h, a in sample_pairs if h not in done_hashes]
        logger.info(f'Resuming: {len(done_hashes)} already done, '
                    f'{len(sample_pairs)} remaining')

    # ------------------------------------------------------------------
    # 4. Group by archive for efficient sequential reading
    # ------------------------------------------------------------------
    from collections import defaultdict
    by_archive = defaultdict(list)
    for hash_id, archive_path in sample_pairs:
        by_archive[archive_path].append((hash_id, archive_path))

    logger.info(f'Processing {len(sample_pairs):,} samples across '
                f'{len(by_archive)} archives ...')

    # ------------------------------------------------------------------
    # 5. Process archive by archive, collect results
    # ------------------------------------------------------------------
    all_clima = []
    all_mol   = []
    all_aux   = []
    all_ids   = []

    # Load previously checkpointed data if resuming
    if args.resume and (args.output_dir / 'spectra_partial.npy').exists():
        all_clima = [np.load(args.output_dir / 'spectra_partial.npy')]
        all_mol   = [np.load(args.output_dir / 'molecules_partial.npy')]
        all_aux   = [np.load(args.output_dir / 'aux_partial.npy')]
        with open(args.output_dir / 'ids_partial.json') as f:
            all_ids = json.load(f)
        logger.info(f'Loaded {len(all_ids)} previously processed samples')

    archive_list = sorted(by_archive.keys(), key=lambda p: str(p))

    for archive_path in archive_list:
        pairs = by_archive[archive_path]
        target_hashes = {h for h, _ in pairs}
        logger.info(f'Archive {Path(archive_path).name}: '
                    f'{len(target_hashes)} samples (single-pass scan)')
        t0 = time.time()

        results = process_archive_single_pass(
            Path(archive_path), target_hashes, logger
        )
        done_hashes.update(r[0] for r in results)

        if results:
            batch_clima = np.stack([r[1] for r in results])   # (B, 12, 101)
            batch_mol   = np.stack([r[2] for r in results])   # (B, 12)
            batch_aux   = np.stack([r[3] for r in results])   # (B, 11)
            batch_ids   = [r[0] for r in results]

            all_clima.append(batch_clima)
            all_mol.append(batch_mol)
            all_aux.append(batch_aux)
            all_ids.extend(batch_ids)

        elapsed = time.time() - t0
        rate = len(results) / elapsed if elapsed > 0 else 0
        logger.info(f'  Done: {len(results)}/{len(pairs)} valid  '
                    f'({rate:.0f} samples/s, {elapsed:.1f}s)')

        # Save checkpoint after each archive
        np.save(args.output_dir / 'spectra_partial.npy',
                np.concatenate(all_clima, axis=0))
        np.save(args.output_dir / 'molecules_partial.npy',
                np.concatenate(all_mol, axis=0))
        np.save(args.output_dir / 'aux_partial.npy',
                np.concatenate(all_aux, axis=0))
        with open(args.output_dir / 'ids_partial.json', 'w') as f:
            json.dump(all_ids, f)
        with open(checkpoint_path, 'w') as f:
            json.dump({'done_hashes': list(done_hashes)}, f)

    # ------------------------------------------------------------------
    # 6. Concatenate and save final arrays
    # ------------------------------------------------------------------
    logger.info('Finalising and saving arrays ...')
    spectra   = np.concatenate(all_clima, axis=0).astype(np.float32)   # (N, 12, 101)
    molecules = np.concatenate(all_mol, axis=0).astype(np.float32)     # (N, 12)
    aux_params = np.concatenate(all_aux, axis=0).astype(np.float32)    # (N, 11)
    system_ids = np.array(all_ids, dtype=object)                       # (N,)

    # Altitude axis (CLIMA row index maps to altitude: rows 0-100)
    wavelengths = np.arange(101, dtype=np.float64)  # placeholder altitude indices

    N = len(spectra)
    logger.info(f'Final dataset: {N:,} samples')
    logger.info(f'  spectra    : {spectra.shape}  (CLIMA channels × altitude levels)')
    logger.info(f'  molecules  : {molecules.shape}  (log10 surface mol. concentrations)')
    logger.info(f'  aux_params : {aux_params.shape}  (metadata scalars)')

    np.save(args.output_dir / 'spectra.npy',    spectra)
    np.save(args.output_dir / 'molecules.npy',  molecules)
    np.save(args.output_dir / 'aux_params.npy', aux_params)
    np.save(args.output_dir / 'system_ids.npy', system_ids)
    np.save(args.output_dir / 'wavelengths.npy', wavelengths)

    # ------------------------------------------------------------------
    # 7. Dataset info JSON (for downstream scripts)
    # ------------------------------------------------------------------
    mol_stats = {}
    for i, mol in enumerate(MOLECULE_NAMES):
        col = molecules[:, i]
        mol_stats[mol] = {
            'mean': float(col.mean()), 'std': float(col.std()),
            'min':  float(col.min()),  'max': float(col.max()),
        }

    info = {
        'n_samples':       N,
        'spectra_shape':   list(spectra.shape),
        'molecules_shape': list(molecules.shape),
        'molecule_names':  MOLECULE_NAMES,
        'clima_channels':  CLIMA_COLS,
        'aux_keys':        AUX_KEYS,
        'in_channels':     N_CLIMA_VARS,
        'seq_len':         101,
        'log_floor':       LOG_FLOOR,
        'seed':            args.seed,
        'source_dir':      str(args.source_dir),
        'molecule_stats':  mol_stats,
        'processing_time_s': round(time.time() - t_start, 1),
    }
    with open(args.output_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    # ------------------------------------------------------------------
    # 8. Optional validation
    # ------------------------------------------------------------------
    if args.validate:
        _validate(spectra, molecules, aux_params, logger)

    # Clean up partial checkpoints on success
    for p in ['spectra_partial.npy', 'molecules_partial.npy',
              'aux_partial.npy', 'ids_partial.json']:
        try:
            (args.output_dir / p).unlink()
        except FileNotFoundError:
            pass

    total_time = time.time() - t_start
    logger.info('=' * 60)
    logger.info(f'Processing complete in {total_time:.1f}s')
    logger.info(f'Output saved to: {args.output_dir}')
    logger.info(f'Ready to train:')
    logger.info(f'  python run_baseline.py   --data-dir {args.output_dir}')
    logger.info(f'  python run_deep_model.py --data-dir {args.output_dir} '
                f'--in-channels {N_CLIMA_VARS}')
    logger.info('=' * 60)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(spectra, molecules, aux_params, logger):
    logger.info('Running validation checks ...')
    ok = True

    # Check no NaN in spectra
    if np.isnan(spectra).any():
        logger.warning(f'  spectra has {np.isnan(spectra).sum()} NaNs')
        ok = False
    else:
        logger.info('  spectra: no NaNs ✓')

    # Check molecule ranges
    inf_count = (molecules <= LOG_FLOOR + 0.1).sum()
    logger.info(f'  molecules: {inf_count} values at or near log-floor '
                f'({100*inf_count/molecules.size:.1f}%)')

    # Check each molecule has reasonable range
    for i, mol in enumerate(MOLECULE_NAMES):
        col = molecules[:, i]
        if col.std() < 0.01:
            logger.warning(f'  {mol}: near-constant (std={col.std():.4f}) — may not be learnable')
        else:
            logger.info(f'  {mol}: range=[{col.min():.2f}, {col.max():.2f}] '
                        f'mean={col.mean():.2f} std={col.std():.2f} ✓')

    if ok:
        logger.info('  All checks passed ✓')


if __name__ == '__main__':
    main()
