# Example configuration - replace with your actual paths and parameters
from core.suite2p_traces import Suite2pTraces
from core.suite2p_run import classifier_file, Suite2pProcessor


# Generally, you should only run together the data that has the same image acquisition.
# E.g. framerate/number planes/resolution.
# Because this defines your suite2p settings.

config = {
    'suite2ppath_raw': 'J:\\_Projects\\Lisa\\rawdata',
    'suite2ppath_processed': 'J:\\_Projects\\Lisa\\processed',
    'suite2p_ops_settings': {'framerate': 30.0,  'number_planes': 6, 'classifier_path': classifier_file},
    'fishnum': 125,
    'experiments': {
        "20240425": {
            '001': 'dark', '002': 'background', '003': 'okr', '004': 'pseudosaccade', '005': 'circleJJ'}},
    'experiment_lengths': {'dark': 1800, 'background': 1800, 'okr': 900, 'pseudosaccade': 1500, 'circleJJ': 1390.8},
}

# Initialize and run the pipeline
pipeline = Suite2pProcessor(**config)
pipeline.run_extraction()
