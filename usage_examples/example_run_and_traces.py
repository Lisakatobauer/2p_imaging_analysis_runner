# Example configuration - replace with your actual paths and parameters
from core.suite2p_F_process import Suite2pTraces
from core.suite2p_run import classifier_file, Suite2pProcessor

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
pipeline.run_extraction()

# perform the F trace processing

# Initialize processor
trace_processor = Suite2pTraces(
)

# Process all planes (automatically saves results)
trace_processor.process_all_planes()

# Get specific trace types
dff_traces = trace_processor.get_processed_traces()  # Default is dff
zscore_traces = trace_processor.get_processed_traces(trace_type='zscore')

# Get traces for a specific plane
plane_2_traces = trace_processor.get_processed_traces(plane=2)

#
activity_plot(plane_2_traces)
cell_location()
