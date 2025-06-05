from core.suite2p_processor import classifier_file, Suite2pProcessor
from utils.config import Suite2pConfig
from utils.utils import get_git_root

from config.base_config import raw_path, processed_path

# 1. Config
config_path = get_git_root() / 'config' / 'configlist'

suite2p_ops = {'framerate': 30.0, 'number_planes': 6, 'classifier_path': classifier_file}

config = Suite2pConfig(
    config_path,
    raw_path,
    processed_path,
    suite2p_ops)

config.load_all_fish_configs()

# 2. Initialize and run the processor

for fishnum in config.get_loaded_fish_ids():
    processor = Suite2pProcessor(config, fishnum)
    processor.run_extraction()
