"""Load one explicit HOTC2020 profile; do not infer settings from output paths."""
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path).resolve(strict=True)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"data_root", "output_root", "device", "seed", "backbone_checkpoint",
                "backbone_config", "motion_checkpoint", "tabr", "sdmr"}
    if (not isinstance(config, dict) or not required.issubset(config)
            or set(config) - required - {"training"}):
        raise ValueError(f"Configuration requires {sorted(required)}, plus optional training")
    for key in ("data_root", "output_root", "backbone_checkpoint", "motion_checkpoint"):
        value = Path(config[key]).expanduser()
        config[key] = str((ROOT / value).resolve() if not value.is_absolute() else value.resolve())
    if config["backbone_config"] != "sam21pp_hiera_l.yaml":
        raise ValueError("This clean repository includes only DAM4SAM SAM2.1-L")
    for name in ("object_threshold", "ema_alpha"):
        if not 0 <= config["sdmr"][name] <= 1:
            raise ValueError(f"Invalid SDMR {name}")
    if set(config["sdmr"]) != {"object_threshold", "ema_alpha"}:
        raise ValueError("Unsupported SDMR configuration key")
    return config
