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
from scipy.signal import iirnotch, butter, filtfilt
import argparse
from datetime import datetime
import scipy.io
import mne

# Function to apply preprocessing steps
def apply_preprocessing(data, recording_sample_rate=None, target_sampling_rate=100, include_timestamp=False, notch_filter=[50, 60], bandpass_filter=[1,45]):
    # Remove the timestamp, sample count, and marker channels

    print(f'Shape of data: {data.shape}')
    print(f'First row of data: {data[0]}')

    # Apply the notch filter
    for freq in notch_filter:
        nyq = 0.5 * sampling_rate
        q = 30
        w0 = freq / nyq
        b, a = iirnotch(w0, Q=q)
        # Apply the filter to each column of the 2D array
        for i in range(data.shape[1]):
            data[:, i] = filtfilt(b, a, data[:, i])
    # Apply the bandpass filter
    lowcut, highcut = bandpass_filter
    nyq = 0.5 * sampling_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='band')
    for i in range(data.shape[1]):
        data[:, i] = filtfilt(b, a, data[:, i])

    # Downsample the data to standarized sample rate
    data = downsample_data(data, target_sampling_rate, target_sampling_rate)
    return data

# Downsample the data to 100 Hz
def downsample_data(data, original_sampling_rate, target_sampling_rate):
    # Calculate the downsampling factor
    downsample_factor = int(original_sampling_rate / target_sampling_rate)
    # Downsample the data by taking every nth sample
    downsampled_data = data[::downsample_factor, :]
    return downsampled_data

# Modify the convert_to_pt function to include downsampling
def convert_to_pt(input_file, output_file, recording_sample_rate=None, target_sampling_rate=100, include_timestamp=False, notch_filter=[50, 60], bandpass_filter=[1,45]):
    # Check file extension and read data accordingly
    if input_file.endswith('.csv'):
        data = pd.read_csv(input_file)
        data = data.iloc[:, 1:9].to_numpy(dtype='float32')
    elif input_file.endswith('.mat'):
        data = read_mat_file(input_file)
    elif input_file.endswith('.edf'):
        data, sample_rate = read_edf_file(input_file)
        recording_sample_rate = sample_rate
    elif input_file.endswith('.pt'):
        # Load tensor file
        data = torch.load(input_file)

    if recording_sample_rate is None:
        raise ValueError('Recording sample rate is not set.')

    # Apply preprocessing steps
    data = apply_preprocessing(data, target_sampling_rate, notch_filter, bandpass_filter) 
    # Convert to PyTorch tensor
    tensor = torch.tensor(data)
    # Save the tensor to a .pt file
    torch.save(tensor, output_file)


def convert_to_pt(input_file, output_file, sampling_rate, include_timestamp, notch_filter, bandpass_filter):
    # Check file extension and read data accordingly
    if input_file.endswith('.csv'):
        data = pd.read_csv(input_file)
        data = data.iloc[:, 1:9].to_numpy(dtype='float32')
    elif input_file.endswith('.mat'):
        data = read_mat_file(input_file)
    elif input_file.endswith('.edf'):
        data = read_edf_file(input_file)
    elif input_file.endswith('.pt'):
        # Load tensor file
        data = torch.load(input_file)

    # Apply preprocessing steps
    data = apply_preprocessing(data, sampling_rate, notch_filter, bandpass_filter)
    # Convert to PyTorch tensor
    tensor = torch.tensor(data)
    # Save the tensor to a .pt file
    torch.save(tensor, output_file)

def process_directory(input_directory, output_directory, sampling_rate, include_timestamp, notch_filter, bandpass_filter):
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
    # Extract data as a numpy array
    data = raw.get_data().T  # Transpose to align samples along rows
    sampling_rate = raw.info['sfreq']
    return (data, sampling_rate)

# Main function
def main():
    print(
        f'Converting CSV files to PyTorch .pt files'
    )
    parser = argparse.ArgumentParser(description='Convert CSV files to PyTorch .pt files')
    parser.add_argument('--input_directory', type=str, help='The directory containing the CSV files')
    parser.add_argument('--output_directory', type=str, help='The directory where the .pt files will be saved')
    parser.add_argument('--sampling_rate', type=float, help='The sampling rate of the data')
    parser.add_argument('--include_timestamp', action='store_true', help='Include a timestamp in the output file names')
    parser.add_argument('--notch_filter', nargs='+', type=float, help='The frequencies for the notch filter')
    parser.add_argument('--bandpass_filter', nargs=2, type=float, help='The lowcut and highcut frequencies for the bandpass filter')
    args = parser.parse_args()

    # Traverse the directory structure
    print(
        f'Processing {args.input_directory}'
    )
    for root, dirs, files in os.walk(args.input_directory):
        for directory in dirs:
            process_directory(os.path.join(root, directory), args.output_directory, args.sampling_rate, args.include_timestamp, args.notch_filter, args.bandpass_filter)

if __name__ == '__main__':
    main()

