"""
Shared configuration loader for all INARA pipeline steps.

Each step script calls:
    parser = get_parser("Step N description")
    args   = parser.parse_args()
    cfg    = load_config(args.config, args.profile)

cfg keys after loading:
    cfg['steps']        — dict of step enable/disable flags
    cfg['data']         — val_frac, test_frac, seed
    cfg['extraction']   — n_samples, n_workers, seed
    cfg['model']        — in_channels, pca_components
    cfg['baseline']     — max_train_samples, seed
    cfg['training']     — epochs, batch_size, lr, weight_decay, patience
    cfg['paths']        — merged profile: raw_dir, processed_dir, engineered_dir,
                          results_dir, models_dir, device, n_workers
    cfg['slurm']        — SLURM settings (only relevant in job scripts)
    cfg['profile']      — 'local' or 'hpc'
"""

import argparse
from pathlib import Path

# Default config path relative to project root
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'pipeline' / 'config.yaml'


def get_parser(description: str) -> argparse.ArgumentParser:
    """Return a base ArgumentParser with --config and --profile arguments."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config', type=str,
        default=str(_DEFAULT_CONFIG),
        help='Path to pipeline/config.yaml',
    )
    parser.add_argument(
        '--profile', type=str,
        default='local',
        choices=['local', 'hpc'],
        help='Environment profile (local = Mac, hpc = Northeastern Explorer)',
    )
    return parser


def load_config(config_path: str, profile: str) -> dict:
    """
    Load config.yaml and merge the chosen environment profile.

    Returns a flat cfg dict — see module docstring for keys.
    """
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if profile not in raw.get('profiles', {}):
        raise ValueError(
            f"Profile '{profile}' not found in {config_path}. "
            f"Available: {list(raw.get('profiles', {}).keys())}"
        )

    cfg: dict = {}
    for section in ('steps', 'data', 'extraction', 'model', 'baseline', 'training'):
        cfg[section] = raw.get(section, {})

    cfg['paths']   = raw['profiles'][profile]
    cfg['slurm']   = raw.get('slurm', {})
    cfg['profile'] = profile

    return cfg


def resolve_path(p: str, profile: str) -> Path:
    """
    Return an absolute Path.
    Relative paths are anchored to the project root when running locally,
    and returned as-is (already absolute) on HPC.
    """
    path = Path(p)
    if path.is_absolute():
        return path
    # Relative → anchor at project root (two levels up from pipeline/steps/)
    project_root = Path(__file__).resolve().parents[2]
    return project_root / path
