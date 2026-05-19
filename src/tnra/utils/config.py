"""Configuration loading with hierarchical YAML merging and CLI overrides.

The config system works in three layers, merged in order (later wins):
    1. `base.yaml`  — defaults shared across all stages (paths, logging, seed)
    2. `<stage>.yaml` — stage-specific config (ingestion, retrieval, eval)
    3. CLI overrides — runtime tweaks like `chunking.chunk_size=256`

Validation (via Pydantic) happens at the module level, not here. Each consumer
module defines its own schema and validates the sub-config it cares about.
This keeps config loading lightweight and decouples modules from each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from tnra.utils.paths import CONFIGS_DIR, ensure_dir

_BASE_CONFIG_NAME = "base.yaml"


def load_config(
    stage_config: str | Path,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load and merge base + stage config, then apply CLI overrides.

    Args:
        stage_config: Either a filename relative to `configs/` (e.g. "ingestion.yaml")
            or an absolute/relative path to a YAML file.
        overrides: List of `key.path=value` strings (OmegaConf dotlist format),
            typically forwarded from CLI args. Examples:
                ["chunking.chunk_size=256", "embeddings.batch_size=64"]

    Returns:
        A fully merged OmegaConf DictConfig, accessible via dot notation
        (e.g. `cfg.chunking.chunk_size`) or dict notation (`cfg["chunking"]`).

    Raises:
        FileNotFoundError: If base.yaml or the stage config doesn't exist.
    """
    base_path = CONFIGS_DIR / _BASE_CONFIG_NAME
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    stage_path = _resolve_stage_path(stage_config)
    if not stage_path.exists():
        raise FileNotFoundError(f"Stage config not found: {stage_path}")

    # Load both YAML files. OmegaConf.load returns a DictConfig that supports
    # dot-access, type coercion, and interpolation (${other.field} references).
    base_cfg = OmegaConf.load(base_path)
    stage_cfg = OmegaConf.load(stage_path)

    # Merge: stage overrides base. OmegaConf.merge does a deep merge, so
    # nested keys are combined rather than replaced wholesale.
    cfg = OmegaConf.merge(base_cfg, stage_cfg)

    # Apply CLI overrides last (highest priority). from_dotlist parses
    # "a.b.c=value" strings into a DictConfig that we merge on top.
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Type narrowing for the type checker: after merging two DictConfigs we
    # always get a DictConfig back, but the static type is the union DictConfig|ListConfig.
    assert isinstance(cfg, DictConfig)
    return cfg


def save_config(cfg: DictConfig, output_dir: Path, filename: str = "config.yaml") -> Path:
    """Snapshot a resolved config to disk for reproducibility.

    Call this at the start of each pipeline run so you can always answer
    "which exact config produced this index/eval/model 6 months later?".

    Args:
        cfg: The resolved OmegaConf config to save.
        output_dir: Directory to write into (created if missing).
        filename: Name of the YAML file.

    Returns:
        Path to the written file.
    """
    ensure_dir(output_dir)
    output_path = output_dir / filename
    OmegaConf.save(cfg, output_path)
    return output_path


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert an OmegaConf config to a plain Python dict.

    Useful when passing config to libraries that don't accept OmegaConf objects
    (e.g. some HuggingFace components, or JSON serialization).
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    return container  # type: ignore


def _resolve_stage_path(stage_config: str | Path) -> Path:
    """Accept either a filename ('ingestion.yaml') or a full path."""
    path = Path(stage_config)
    if path.is_absolute() or path.exists():
        return path
    return CONFIGS_DIR / path
