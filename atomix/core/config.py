"""Configuration handling for atomix."""

from pathlib import Path
from typing import Any

import yaml


class Config:
    """Configuration manager for atomix settings.

    Parameters
    ----------
    config_path : Path | str | None
        Path to YAML configuration file.
    """

    DEFAULT_CONFIG: dict[str, Any] = {
        "vasp": {
            "potcar_dir": None,
            "default_functional": "PBE",
            "default_encut": 400,
        },
        "slurm": {
            "partition": None,
            "nodes": 1,
            "ntasks_per_node": 32,
        },
        "llm": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
        },
    }

    def __init__(self, config_path: Path | str | None = None) -> None:
        self._config = self.DEFAULT_CONFIG.copy()
        if config_path is not None:
            self.load(config_path)

    def load(self, config_path: Path | str) -> None:
        """Load configuration from YAML file."""
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    self._update_nested(self._config, user_config)

    def _update_nested(self, base: dict, update: dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_nested(base[key], value)
            else:
                base[key] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value.

        Parameters
        ----------
        *keys : str
            Sequence of keys to traverse.
        default : Any
            Default value if key not found.

        Returns
        -------
        Any
            Configuration value.
        """
        result = self._config
        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default
        return result

    def set(self, *keys: str, value: Any) -> None:
        """Set nested configuration value."""
        if not keys:
            return
        target = self._config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    def save(self, config_path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)
