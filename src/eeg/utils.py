import numpy as np

EEG_10_20_CHANNELS = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']
EEG_10_10_CHANNELS = ['FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ']

EEG_ALL_CHANNELS = sorted(list(set(EEG_10_20_CHANNELS + EEG_10_10_CHANNELS)))

EEG_10_20_MAP_OLD_TO_NEW = {'T1': 'T7', 'T2': 'T8', 'T3': 'T9', 'T4': 'T10', 'T5': 'T11', 'T6': 'T12'}

def align_data_to_standard_channels(input_data, channel_locations):
    """
    Map input data to a new zero-filled numpy array based on channel locations.

    Parameters:
    - input_data (numpy.ndarray): The input data array where each row corresponds to a channel in `channel_locations`.
    - channel_locations (list): List of channel names corresponding to the rows in `input_data`.

    Returns:
    - numpy.ndarray: A 2D array where each row corresponds to a channel in `EEG_ALL_CHANNELS` and is filled with data from `input_data` or zeros if no data is available for that channel.
    """
    num_channels = len(EEG_ALL_CHANNELS)
    num_data_points = input_data.shape[1]
    mapped_array = np.zeros((num_channels, num_data_points))

    channel_index_map = {channel: index for index, channel in enumerate(EEG_ALL_CHANNELS)}

    for i, channel in enumerate(channel_locations):
        if channel in channel_index_map:
            mapped_array[channel_index_map[channel]] = input_data[i]

    return mapped_array