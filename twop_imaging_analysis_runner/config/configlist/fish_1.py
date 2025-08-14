fishnum = 1

notes = """
A example config file of an example fish. 
"""

# the dates and types of experiments on those dates
experiments = {
    "20240425":
        {'001': 'experiment_type_1', '002': 'experiment_type_2', '003': 'experiment_type_3'},
    "20240531":
        {'001': 'experiment_type_2'}}

# in seconds
experiment_lengths = {'experiment_type_1': 1800, 'experiment_type_2': 900, 'experiment_type_3': 1500}

# If you have different planes for the same fish, you will have to specify in your run configuration.
nplanes = 6
