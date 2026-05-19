"""Project path resolution.

All paths in the project are derived from a single source of truth: the location
of this file. This makes the code immune to the current working directory and to
the machine it runs on — paths resolve identically whether you launch from VS Code,
PowerShell, a Docker container, or a CI runner.
"""

from pathlib import Path

# This file lives at: <repo>/src/tnra/utils/paths.py
# Going up 3 levels (utils → tnra → src → <repo>) gives the repo root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

# Top-level directories (relative to the repo root)
SRC_DIR: Path = PROJECT_ROOT / "src"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"
EVAL_DIR: Path = PROJECT_ROOT / "eval"

# Runtime directories (created on demand, gitignored)
DATA_DIR: Path = PROJECT_ROOT / "data"
CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
HF_CACHE_DIR: Path = PROJECT_ROOT / ".hf_cache"
MODELS_DIR: Path = PROJECT_ROOT / "models"
EVAL_OUTPUTS_DIR: Path = EVAL_DIR / "outputs"


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it doesn't exist, then return it.

    Args:
        path: Directory to create.

    Returns:
        The same path, guaranteed to exist as a directory after this call.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
