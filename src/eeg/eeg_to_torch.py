"""
This script converts CSV files to PyTorch .pt files. It applies preprocessing steps to the data, including a notch filter and a bandpass filter.

Usage:
    python script.py --input_directory INPUT_DIRECTORY --output_directory OUTPUT_DIRECTORY --sampling_rate SAMPLING_RATE [--include_timestamp] [--notch_filter NOTCH_FILTER [NOTCH_FILTER ...]] [--bandpass_filter LOWCUT HIGHCUT]

Arguments:
    --input_directory: The directory containing the CSV files.
    --output_directory: The directory where the .pt files will be saved.
    --sampling_rate: The sampling rate of the data.
    --include_timestamp: Include a timestamp in the output file names.
    --notch_filter: The frequencies for the notch filter.
    --bandpass_filter: The lowcut and highcut frequencies for the bandpass filter.

Example:
    python3 crown-eeg-to-torch.py --input_directory data/sessions --output_directory data/pt_sessions --sampling_rate 256 --notch_filter 50 60 --bandpass_filter 1 48
"""
import os
import shutil
import pandas as pd
import torch
from scipy.signal import iirnotch, butter, filtfilt, resample
import argparse
from datetime import datetime
import scipy.io
import mne
import glob
import json
from utils import EEG_ALL_CHANNELS

mne.set_log_level('WARNING')

# Function to apply preprocessing steps
def apply_preprocessing(data, recording_sample_rate=256.0, target_sampling_rate=128.0, notch_filter=[50.0, 60.0], bandpass_filter=[1.0,45.0]):
    # Apply the notch filter
    for freq in notch_filter:
        nyq = 0.5 * recording_sample_rate
        q = 30
        w0 = freq / nyq
        b, a = iirnotch(w0, Q=q)
        # Apply the filter to each column of the 2D array
        for i in range(data.shape[1]):
            data[:, i] = filtfilt(b, a, data[:, i])
    # Apply the bandpass filter
    lowcut, highcut = bandpass_filter
    nyq = 0.5 * recording_sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='band')
    for i in range(data.shape[1]):
        data[:, i] = filtfilt(b, a, data[:, i])

    # Downsample the data to standarized sample rate
    data = downsample_data(data, recording_sample_rate, target_sampling_rate)
    return data



def process_tuh_edf_directory(input_directory, output_directory, include_timestamp, notch_filter, bandpass_filter):
    # Create a dictionary to store metadata on each file
    file_metadata = {}
    print(f'Searching {input_directory} for .edf files')
    # Get all .edf files in the directory recursively
    edf_files = glob.glob(os.path.join(input_directory, '**', '*.edf'), recursive=True)
    print(f'Found {len(edf_files)} .edf files in {input_directory}')

    # Make output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    count = 0
    for edf_file in edf_files:
        # Create a descriptive file name from the path
        descriptive_file_name = edf_file.split('/edf/', 1)[1].replace('/', '_').replace('.edf', '')
        # Convert to pt and save file
        output_file = os.path.join(output_directory, descriptive_file_name + '.pt')
        data, recording_sample_rate, channel_locations = convert_to_pt(edf_file, output_file, include_timestamp=include_timestamp, notch_filter=notch_filter, bandpass_filter=bandpass_filter)
        rows = data.shape[0]
        # Save metadata to the file_metadata dictionary
        file_metadata[descriptive_file_name] = {
            'sample_rate': recording_sample_rate,
            'channel_locations': channel_locations,
            'rows': rows
        }

        # save data to a csv log file with count as index
        count += 1
        print(f'{count}: Processed {rows} rows and sample rate of {recording_sample_rate} and saved to {output_file}')
    
    # Summary of the process
    print(f'Processed {count} EDF files.')

    # Descriptive log file name with date and time
    descriptive_log_file_name = input_directory.replace('/', '_').replace('.edf', '') + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".json"
    # Create a file path for the JSON file using the descriptive_log_file_name
    json_file_path = os.path.join(output_directory, descriptive_log_file_name)
    # Write the file_metadata dictionary to the file
    with open(json_file_path, 'w') as json_file:
        json.dump(file_metadata, json_file)
    print(f'Saved file metadata to {json_file_path}')


# Resample the data to target Hz
def downsample_data(data, original_sampling_rate=256.0, target_sampling_rate=128.0):
    # Calculate the number of samples in the downsampled data
    num_samples = int(data.shape[0] * target_sampling_rate / original_sampling_rate)
    # Use scipy's resample function to downsample the data
    downsampled_data = resample(data, num_samples)
    return downsampled_data

# Modify the convert_to_pt function to include downsampling
def convert_to_pt(input_file, output_file, recording_sample_rate=None, target_sampling_rate=128.0, channel_locations=None, include_timestamp=False, notch_filter=[50, 60], bandpass_filter=[1,45]):
    # Check file extension and read data accordingly
    if input_file.endswith('.csv'):
        data = pd.read_csv(input_file)
        data = data.iloc[:, 1:9].to_numpy(dtype='float32')
    elif input_file.endswith('.mat'):
        data = read_mat_file(input_file)
    elif input_file.endswith('.edf'):
        data, sample_rate, channel_location = read_edf_file(input_file)
        recording_sample_rate = sample_rate
        channel_locations = channel_location

    if recording_sample_rate is None:
        raise ValueError('Recording sample rate is not set.')

    # Apply preprocessing steps
    data = apply_preprocessing(data, recording_sample_rate, target_sampling_rate, notch_filter, bandpass_filter) 
    # Convert to PyTorch tensor
    tensor = torch.tensor(data)
    # Save the tensor to a .pt file
    torch.save(tensor, output_file)

    return data, target_sampling_rate, channel_locations

def process_crown_directory(input_directory, output_directory, sampling_rate, include_timestamp, notch_filter, bandpass_filter):
    # Get the name of the directory
    directory_name = os.path.basename(input_directory)
    # Check if there are any CSV files in the directory
    csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]
    if csv_files:
        # Process each CSV file in the directory
        for filename in csv_files:
            input_file = os.path.join(input_directory, filename)
            # Extract the dataset name from the input file path
            dataset_name = os.path.basename(os.path.dirname(input_file))
            # Use the dataset name as the name of the output file
            output_file = os.path.join(output_directory, dataset_name + '.pt')
            # Check if the output file already exists
            if not os.path.exists(output_file):
                convert_to_pt(input_file, output_file, sampling_rate, include_timestamp, notch_filter, bandpass_filter)
                print(f'Processed {input_file} and saved to {output_file}')
            else:
                print(f'Output file {output_file} already exists. Skipping processing for {input_file}')

def read_mat_file(file_path):
    # Load .mat file
    mat = scipy.io.loadmat(file_path)
    # Assuming the EEG data is stored under the key 'data'
    data = mat['data']
    return data


def read_edf_file(file_path):
    # Load EDF file
    raw = mne.io.read_raw_edf(file_path, preload=True)
    # Get channel locations
    channel_locations = raw.ch_names
    # Clean the data, take only channels with "EEG" and map these to another list with just the channel name
    eeg_channels = [ch for ch in channel_locations if 'EEG' in ch]
    # Clean the data, remove the "EEG " and "-REF" or "-LE" or anything else
    eeg_channels_clean = [ch.split('-')[0].replace('EEG ', '') for ch in eeg_channels]  
    # Only take channels found in the EEG 10-10 system
    eeg_channels_picks = [ch for ch in eeg_channels_clean if ch in EEG_ALL_CHANNELS]
    # go through original list with new confirmed channels and make a picks array to later use to pick the data array
    picks = []
    for ch in eeg_channels_picks:
        for suffix in ['-LE', '-REF']:
            if 'EEG ' + ch + suffix in channel_locations:
                picks.append(channel_locations.index('EEG ' + ch + suffix))
    # Extract data as a numpy array
    data = raw.get_data().T  # Transpose to align samples along rows
    # Take only the channels in the 10-20 system
    data = data[:, picks]
    # Get the sampling rate
    sampling_rate = raw.info['sfreq']
    return (data, sampling_rate, eeg_channels_picks)

# Main function
def main():
    # Example for CSV
    # python3 src/eeg/eeg_to_torch.py --input_directory data/sessions --output_directory data/pt_sessions --sampling_rate 256 --notch_filter 50 60 --bandpass_filter 1 48

    # Example for TUH EDF
    # python3 src/eeg/eeg_to_torch.py --input_directory data/tuh_eeg --output_directory data/pt_tuh_edf --tuh_edf --notch_filter 50 60 --bandpass_filter 1 48

    print(
        f'Converting CSV files to PyTorch .pt files'
    )
    parser = argparse.ArgumentParser(description='Convert Crown CSV or TUH EDF files to PyTorch .pt files')
    parser.add_argument('--input_directory', type=str, help='The directory containing the CSV files')
    parser.add_argument('--output_directory', type=str, help='The directory where the .pt files will be saved')
    parser.add_argument('--recording_sample_rate', type=float, help='The sampling rate of the data', default=None)
    parser.add_argument('--include_timestamp', action='store_true', help='Include a timestamp in the output file names')
    parser.add_argument('--notch_filter', nargs='+', type=float, help='The frequencies for the notch filter')
    parser.add_argument('--bandpass_filter', nargs=2, type=float, help='The lowcut and highcut frequencies for the bandpass filter')
    parser.add_argument('--tuh_eeg', action='store_true', help='Process TUH EEG files')
    args = parser.parse_args()

    # Traverse the directory structure
    print(
        f'Processing {args.input_directory}'
    )
    if args.tuh_eeg is True:
        process_tuh_edf_directory(args.input_directory, args.output_directory, args.include_timestamp, args.notch_filter, args.bandpass_filter) 
    else:
        for root, dirs, files in os.walk(args.input_directory):
            for directory in dirs:
                process_crown_directory(input_directory=os.path.join(root, directory), output_directory=args.output_directory, sampling_rate=args.recording_sample_rate, include_timestamp=args.include_timestamp, notch_filter=args.notch_filter, bandpass_filter=args.bandpass_filter)

if __name__ == '__main__':
    main()

