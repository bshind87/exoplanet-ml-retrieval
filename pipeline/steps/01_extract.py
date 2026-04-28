#!/usr/bin/env python
"""
Step 1 — Data Extraction
========================
Extracts N samples from INARA ATMOS tar.gz archives and saves processed
numpy arrays to processed_dir/.

Input  (raw_dir):
  pyatmos_summary.csv, dir_0.tar.gz … dir_9.tar.gz, Dir_alpha.tar.gz

Output (processed_dir):
  spectra.npy       (N, 12, 101)  — CLIMA profiles
  molecules.npy     (N, 12)       — log10 molecular surface abundances
  aux_params.npy    (N, 11)       — input flux / condition metadata
  system_ids.npy    (N,)          — sample hash strings
  wavelengths.npy   (101,)        — altitude axis (km)
  dataset_info.json               — dataset statistics + column names

Skip this step (steps.extract: false in config.yaml) if processed/ already
exists from a previous run.

Usage:
  python pipeline/steps/01_extract.py [--config pipeline/config.yaml] [--profile local|hpc]
  python pipeline/steps/01_extract.py --resume          # continue a partial extraction
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.steps.config_loader import get_parser, load_config, resolve_path


def main() -> None:
    parser = get_parser('Step 1: Extract INARA ATMOS tar.gz archives')
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume a partial extraction (skips already-processed hashes)',
    )
    parser.add_argument(
        '--n-samples', type=int, default=None,
        help='Override extraction n_samples from config',
    )
    args = parser.parse_args()

    cfg   = load_config(args.config, args.profile)
    paths = cfg['paths']
    ext   = cfg['extraction']

    raw_dir       = resolve_path(paths['raw_dir'],       args.profile)
    processed_dir = resolve_path(paths['processed_dir'], args.profile)
    n_samples     = args.n_samples or ext['n_samples']
    n_workers     = paths.get('n_workers', ext['n_workers'])
    seed          = ext['seed']

    print('=' * 60)
    print('  INARA Pipeline — Step 1: Data Extraction')
    print('=' * 60)
    print(f'  Profile       : {args.profile}')
    print(f'  Source        : {raw_dir}')
    print(f'  Output        : {processed_dir}')
    print(f'  n_samples     : {n_samples:,}')
    print(f'  n_workers     : {n_workers}')
    print(f'  seed          : {seed}')
    if args.resume:
        print('  resume        : yes')
    print()

    # Delegate to the existing process_inara.py (logic unchanged)
    cmd = [
        sys.executable, str(ROOT / 'process_inara.py'),
        '--source-dir',  str(raw_dir),
        '--output-dir',  str(processed_dir),
        '--n-samples',   str(n_samples),
        '--n-workers',   str(n_workers),
        '--seed',        str(seed),
        '--validate',
    ]
    if args.resume:
        cmd.append('--resume')

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
