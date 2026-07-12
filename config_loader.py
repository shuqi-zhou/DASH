"""config_loader.py - load config.yaml and resolve all paths."""
import yaml
from pathlib import Path

_CODE_DIR = Path(__file__).parent.resolve()


def get_root(cfg: dict) -> Path:
    r = cfg.get("root", "")
    if not r:
        return _CODE_DIR.parent
    # Unix-style absolute path (e.g., /workspace) - keep as-is for Docker
    if str(r).startswith('/'):
        return Path(r)
    p = Path(r)
    return p.expanduser().resolve()


def load_config(path: str = None) -> dict:
    """Return parsed config with all paths resolved to absolute Path objects."""
    config_path = Path(path) if path else _CODE_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = get_root(cfg)
    cfg["root"] = root

    data_mode = cfg.get("preprocess", {}).get("data_mode", "train")

    def resolve(key_path: str) -> Path:
        key_path_str = str(key_path)
        if '{data_mode}' in key_path_str:
            key_path_str = key_path_str.replace('{data_mode}', data_mode)
        # Unix-style absolute path (e.g., /workspace) - keep as-is for Docker
        if str(key_path_str).startswith('/'):
            return Path(key_path_str)
        p = Path(key_path_str)
        return p if p.is_absolute() else (root / p).resolve()

    cfg["preprocess"]["raw_dir"]       = resolve(cfg["preprocess"]["raw_dir"])
    cfg["preprocess"]["processed_dir"] = resolve(cfg["preprocess"]["processed_dir"])
    return cfg


def get_root_str(cfg: dict = None) -> str:
    """Return root as a trailing-slash string (drop-in for the old `root` variable)."""
    if cfg is None:
        cfg = load_config()
    return str(cfg["root"]) + "/"
