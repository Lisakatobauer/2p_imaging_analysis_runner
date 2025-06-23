import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional


class Suite2pConfig:
    def __init__(self,
                 config_path: Path,
                 raw_path: Path,
                 processed_path: Path,
                 suite2p_ops: Dict[str, Any] = None,
                 classifier_file: Optional[Path] = None,
                 downsampling_factor: int = 5,
                 bidirectional_scanning: bool = True):
        self.one_fish = False
        self._fish_configs_by_name = None
        self.config_path = Path(config_path)
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.suite2p_ops = suite2p_ops
        self.classifier_file = classifier_file
        self.downsampling_factor = downsampling_factor
        self.bidirectional_scanning = bidirectional_scanning

        self._fish_configs_cache = {}

    def _load_fish_config(self, filepath: Path) -> Dict[str, Any]:
        """Dynamically import a fish config .py file and return its globals."""
        if filepath in self._fish_configs_cache:
            return self._fish_configs_cache[filepath]

        spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore

        # Collect relevant attributes (skip builtins and imports)
        config_vars = {k: v for k, v in vars(module).items()
                       if not k.startswith('_') and not callable(v)}

        self._fish_configs_cache[filepath] = config_vars
        return config_vars

    def load_all_fish_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all .py config files in the config path."""
        fish_configs = {}
        for filepath in self.config_path.glob("*.py"):
            config_data = self._load_fish_config(filepath)
            fish_name = filepath.stem
            fish_configs[fish_name] = config_data
        self._fish_configs_by_name = fish_configs
        return fish_configs

    def get_fish_config(self, fish_id: str) -> Dict[str, Any]:
        """Load a single fish config by ID (filename without .py)."""
        file_path = self.config_path / f"fish_{fish_id}.py"
        if not file_path.exists():
            raise FileNotFoundError(f"No config found for fish ID: {fish_id}")
        self.one_fish = True
        self._fish_configs_by_name = fish_id
        return self._load_fish_config(file_path)

    def clear_cache(self):
        """Reset all cached config files."""
        self._fish_configs_cache.clear()

    def get_loaded_fish_ids(self) -> list[str]:
        """Return list of fish IDs currently cached (from load_all_fish_configs)."""
        if self.one_fish:
            fish_ids = [self._fish_configs_by_name]
        else:
            fish_ids = list(self._fish_configs_by_name.keys())

        return fish_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config_path': str(self.config_path),
            'raw_path': str(self.raw_path),
            'processed_path': str(self.processed_path),
            'suite2p_ops': self.suite2p_ops,
            'classifier_file': str(self.classifier_file) if self.classifier_file else None,
            'downsampling_factor': self.downsampling_factor,
            'bidirectional_scanning': self.bidirectional_scanning
        }

