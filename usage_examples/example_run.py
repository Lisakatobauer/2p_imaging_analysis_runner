from core.suite2p_processor import classifier_file, Suite2pProcessor
from core.suite2p_traces import Suite2pTraces
from core.suite2p_visualiser import Suite2pVisualiser
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

config.get_fish_config(125)
# config.load_all_fish_configs()
fish_ids = config.get_loaded_fish_ids()

# 2. Initialize and run the processor

for fishnum in fish_ids:
    processor = Suite2pProcessor(config, fishnum)
    processor.run_extraction()

# 3. Do some standard processing on the traces

for fishnum in fish_ids:
    traces = Suite2pTraces(config, fishnum)
    traces.process_all()



