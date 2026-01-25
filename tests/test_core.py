"""Tests for core atomix functionality."""

import pytest

from atomix.core.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = Config()
        assert config.get("vasp", "default_functional") == "PBE"
        assert config.get("vasp", "default_encut") == 400

    def test_get_nested(self) -> None:
        """Test nested key access."""
        config = Config()
        assert config.get("slurm", "nodes") == 1
        assert config.get("nonexistent", default="default") == "default"

    def test_set_nested(self) -> None:
        """Test setting nested values."""
        config = Config()
        config.set("vasp", "default_encut", value=500)
        assert config.get("vasp", "default_encut") == 500

    def test_set_new_key(self) -> None:
        """Test setting new nested keys."""
        config = Config()
        config.set("new", "nested", "key", value="value")
        assert config.get("new", "nested", "key") == "value"
