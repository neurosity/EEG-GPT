import numpy as np
import argparse
import os
import json
from datetime import datetime
import glob

EEG_10_20_CHANNELS = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2', 'T1', 'T2']
EEG_10_10_CHANNELS = ['FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ']

# EEG_ALL_CHANNELS = sorted(list(set(EEG_10_20_CHANNELS + EEG_10_10_CHANNELS)))
EEG_ALL_CHANNELS = sorted(list(set(EEG_10_20_CHANNELS)))
NUM_CHANNELS = len(EEG_ALL_CHANNELS)

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


def find_latest_timestamp(input_directory):
    latest_timestamp = None

    json_files = glob.glob(os.path.join(input_directory, '**', 'session.json'), recursive=True)

    for json_file in json_files:
        with open(json_file) as f:
            session_data = json.load(f)
            if 'startTime' in session_data:
                if session_data['startTime'] is not None:
                    if latest_timestamp is None or session_data['startTime'] > latest_timestamp:
                        latest_timestamp = session_data['startTime']
                

    return latest_timestamp

def main():
    # Example for CSV
    # python3 src/eeg/utils.
    print(
        f'Utilities'
    )
    parser = argparse.ArgumentParser(description='Find the latest session completed in a folder')
    parser.add_argument('--input_directory', type=str, help='The directory containing the json session.js files')
    parser.add_argument('--find_latest_timestamp', action='store_true', help='Find the latest timestamp in the directory')

    args = parser.parse_args()

    # Traverse the directory structure
    print(
        f'Processing {args.input_directory}'
    )

    if args.find_latest_timestamp:
        latest_timestamp = find_latest_timestamp(args.input_directory)
        if latest_timestamp:
            time = datetime.fromtimestamp(latest_timestamp/1000.0).strftime('%Y-%m-%d_%H-%M-%S')
            print(f'Latest timestamp: {latest_timestamp} or {time}')
        else:
            print('No timestamps found in the directory')


if __name__ == '__main__':
    main()