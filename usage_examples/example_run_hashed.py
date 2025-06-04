# Example configuration - replace with your actual paths and parameters
from core.suite2p_processor import classifier_file, Suite2pProcessor

config = {
    'suite2ppath_raw': 'J:\\_Projects\\Lisa\\rawdata',
    'suite2ppath_processed': 'J:\\_Projects\\Lisa\\processed',
    'fishnum': 125,
    'framerate': 30.0,
    'number_planes': 6,
    'experiments': {
        "20240425": {
            '001': 'dark', '002': 'background', '003': 'okr', '004': 'pseudosaccade', '005': 'circleJJ'}},
    'experiment_lengths': {'dark': 1800, 'background': 1800, 'okr': 900, 'pseudosaccade': 1500, 'circleJJ': 1390.8},
    'downsampling_factor': 5,
    'classifierfile': classifier_file
}

# Initialize and run the pipeline
pipeline = Suite2pProcessor(**config)
pipeline.run_extraction_hashed(config)