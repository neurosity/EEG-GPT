"""
This script converts CSV files to NumPy .npy files. It applies preprocessing steps to the data, including a notch filter and a bandpass filter.

Usage:
    python preprocess.py --input_directory INPUT_DIRECTORY --output_directory OUTPUT_DIRECTORY --sampling_rate SAMPLING_RATE [--include_timestamp] [--notch_filter NOTCH_FILTER [NOTCH_FILTER ...]] [--bandpass_filter LOWCUT HIGHCUT] [--min_sample_size MIN_SAMPLE_SIZE] [--cutoff_samples CUTOFF_SAMPLES] [--parallel]

Arguments:
    --input_directory: The directory containing the CSV files.
    --output_directory: The directory where the .npy files will be saved.
    --sampling_rate: The sampling rate of the data.
    --include_timestamp: Include a timestamp in the output file names.
    --notch_filter: The frequencies for the notch filter.
    --bandpass_filter: The lowcut and highcut frequencies for the bandpass filter.
    --min_sample_size: The minimum number of seconds required for processing a file. Defaults to 4 seconds.
    --cutoff_samples: The number of samples to cut off from the beginning and end of the data to account for filter ringing. Defaults to 18.
    --parallel: Process files in parallel.

Example:
    python3 preprocess.py --input_directory data/sessions --output_directory data/npy_sessions --sampling_rate 256 --notch_filter 50 60 --bandpass_filter 1 48 --min_sample_size 4 --cutoff_samples 18
    python3 preprocess.py --input_directory edf/ --output_directory data/npy_tuh_eeg/ --notch_filter 50 60 --bandpass_filter 1 48 --verbose --tuh_eeg --min_sample_size 4 --cutoff_samples 18

"""

import os
import shutil
import pandas as pd
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt, resample
import argparse
from datetime import datetime
import scipy.io
import mne
import glob
import json
from utils import EEG_ALL_CHANNELS, align_data_to_standard_channels
from multiprocessing import Pool


mne.set_log_level("WARNING")

# Function to apply preprocessing steps


def apply_preprocessing(
    data,
    recording_sample_rate=256.0,
    target_sampling_rate=128.0,
    notch_filter=[50.0, 60.0],
    bandpass_filter=[1.0, 45.0],
    cutoff_samples=18,
):
    # Apply the notch filter
    for freq in notch_filter:
        nyq = 0.5 * recording_sample_rate
        q = 30
        w0 = freq / nyq
        b, a = iirnotch(w0, Q=q)
        # Apply the filter to each row of the 2D array
        for i in range(data.shape[0]):
            data[i, :] = filtfilt(b, a, data[i, :])
    # Apply the bandpass filter
    lowcut, highcut = bandpass_filter
    nyq = 0.5 * recording_sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype="band")
    for i in range(data.shape[0]):
        data[i, :] = filtfilt(b, a, data[i, :])

    # Cut off the beginning and end samples to account for filter ringing
    data = data[:, cutoff_samples:-cutoff_samples]

    # Downsample the data to standardized sample rate
    data = downsample_data(data, recording_sample_rate, target_sampling_rate)
    return data


def process_file(
    file_path,
    output_directory,
    include_timestamp,
    notch_filter,
    bandpass_filter,
    verbose,
    min_sample_size=4,  # in seconds
    cutoff_samples=18,
    recording_sample_rate=256.0,
    target_sampling_rate=128.0,
):
    descriptive_file_name = (
        file_path.replace("/", "_").replace(".edf", "").replace(".bdf", "")
    )
    output_file = os.path.join(output_directory, descriptive_file_name + ".npy")
    if not os.path.exists(output_file):
        data, recording_sample_rate, channel_locations = convert_to_npy(
            file_path,
            output_file,
            include_timestamp=include_timestamp,
            notch_filter=notch_filter,
            bandpass_filter=bandpass_filter,
            cutoff_samples=cutoff_samples,
            recording_sample_rate=recording_sample_rate,
            target_sampling_rate=target_sampling_rate,
        )
        num_samples = data.shape[1]
        min_samples_required = int(recording_sample_rate * min_sample_size)
        if num_samples < min_samples_required:
            if verbose:
                print(f"Skipping {file_path} due to insufficient samples: {num_samples} (minimum required: {min_samples_required})")
            return None, None, None
        # Additional processing and metadata storage logic here
        if verbose:
            print(f"Processed {file_path} into {output_file}")
        return num_samples, recording_sample_rate, descriptive_file_name
    else:
        if verbose:
            print(f"Skipping {file_path} because it already exists")
    return None, None, None


def process_directory_serial(
    edf_bdf_files,
    output_directory,
    include_timestamp,
    notch_filter,
    bandpass_filter,
    verbose,
    min_sample_size=4,  # in seconds
    cutoff_samples=18,
    recording_sample_rate=256.0,
    target_sampling_rate=128.0,
):
    file_metadata = {}
    print(f"Processing {len(edf_bdf_files)} files in serial mode.")
    for file_path in edf_bdf_files:
        num_samples, recording_sample_rate, descriptive_file_name = process_file(
            file_path,
            output_directory,
            include_timestamp,
            notch_filter,
            bandpass_filter,
            verbose,
            min_sample_size,
            cutoff_samples,
            recording_sample_rate,
            target_sampling_rate,
        )
        if num_samples is not None:
            file_metadata[descriptive_file_name] = {
                "recording_sample_rate": recording_sample_rate,
                "num_samples": num_samples,
                "notch_filter": notch_filter,
                "bandpass_filter": bandpass_filter,
                "target_sampling_rate": target_sampling_rate,
            }
    return file_metadata


def process_directory_parallel(
    edf_bdf_files,
    output_directory,
    include_timestamp,
    notch_filter,
    bandpass_filter,
    verbose,
    min_sample_size=4,  # in seconds
    cutoff_samples=18,
):
    file_metadata = {}
    print(f"Processing {len(edf_bdf_files)} files in parallel mode.")
    with Pool() as pool:
        results = pool.starmap(
            process_file,
            [
                (
                    file_path,
                    output_directory,
                    include_timestamp,
                    notch_filter,
                    bandpass_filter,
                    verbose,
                    min_sample_size,
                    cutoff_samples,
                )
                for file_path in edf_bdf_files
            ],
        )
    for result in results:
        num_samples, recording_sample_rate, descriptive_file_name = result
        if num_samples is not None:
            file_metadata[descriptive_file_name] = {
                "sample_rate": recording_sample_rate,
                "num_samples": num_samples,
                "notch_filter": notch_filter,
                "bandpass_filter": bandpass_filter,
            }
    return file_metadata


def process_directory(
    input_directory,
    output_directory,
    include_timestamp,
    notch_filter,
    bandpass_filter,
    verbose,
    min_sample_size=4,  # in seconds
    cutoff_samples=18,
    parallel=False,
    recording_sample_rate=256.0,
    target_sampling_rate=128.0,
):
    #update the edf_bdf_files file with all edf files in input folder
    edf_bdf_files = (
        glob.glob(os.path.join(input_directory, "**", "*.edf"), recursive=True)
        + glob.glob(os.path.join(input_directory, "**", "*.bdf"), recursive=True)
    )
    
    edf_bdf_files_path = os.path.join(input_directory, "__edf_bdf_files.txt")

    # if file exists, delete and overwrite
    if os.path.exists(edf_bdf_files_path):
        os.remove(edf_bdf_files_path)

    with open(edf_bdf_files_path, "w") as file:
        for file_path in edf_bdf_files:
            file.write(f"{file_path}\n")

    if len(edf_bdf_files) > 0:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    if parallel:
        if verbose:
            print("Processing files in parallel mode.")
        file_metadata = process_directory_parallel(
            edf_bdf_files,
            output_directory,
            include_timestamp,
            notch_filter,
            bandpass_filter,
            verbose,
            min_sample_size,
            cutoff_samples,
            recording_sample_rate,
            target_sampling_rate,
        )
    else:
        if verbose:
            print("Processing files in serial mode.")
        file_metadata = process_directory_serial(
            edf_bdf_files,
            output_directory,
            include_timestamp,
            notch_filter,
            bandpass_filter,
            verbose,
            min_sample_size,
            cutoff_samples,
            recording_sample_rate,
            target_sampling_rate,
        )

    descriptive_log_file_name = (
        input_directory.replace("/", "_").replace(".edf", "").replace(".bdf", "")
        + "_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + ".json"
    )
    metadata_directory = os.path.join(output_directory, "metadata")
    if not os.path.exists(metadata_directory):
        os.makedirs(metadata_directory)
    json_file_path = os.path.join(metadata_directory, descriptive_log_file_name)
    with open(json_file_path, "w") as json_file:
        json.dump(file_metadata, json_file)
    if verbose:
        print(f"Saved file metadata to {json_file_path}")




# Resample the data to target Hz
def downsample_data(data, original_sampling_rate=256.0, target_sampling_rate=128.0):
    # Calculate the number of samples in the downsampled data
    num_samples = int(data.shape[1] * target_sampling_rate / original_sampling_rate)
    # Use scipy's resample function to downsample the data
    try:
        downsampled_data = np.zeros((data.shape[0], num_samples))
        for i in range(data.shape[0]):
            downsampled_data[i, :] = resample(data[i, :], num_samples)
        return downsampled_data

    except Exception as e:
        print("Error during resampling:", e)


# Modify the convert_to_npy function to include downsampling


def convert_to_npy(
    input_file,
    output_file,
    recording_sample_rate=None,
    target_sampling_rate=128.0,
    channel_locations=None,
    include_timestamp=False,
    notch_filter=[50, 60],
    bandpass_filter=[1, 45],
    cutoff_samples=18,
):
    # Check file extension and read data accordingly
    if input_file.endswith(".csv"):
        data = pd.read_csv(input_file)
        data = data.iloc[:, 1:9].to_numpy(dtype="float32").T
    elif input_file.endswith(".mat"):
        data = read_mat_file(input_file)
    elif input_file.endswith(".edf") or input_file.endswith(".bdf"):
        data, sample_rate, channel_location = read_edf_file(input_file)
        recording_sample_rate = sample_rate
        channel_locations = channel_location

    if recording_sample_rate is None:
        raise ValueError("Recording sample rate is not set.")
    if channel_locations is None:
        raise ValueError("Channel locations are not set.")

    # Apply preprocessing steps
    data = apply_preprocessing(
        data, recording_sample_rate, target_sampling_rate, notch_filter, bandpass_filter, cutoff_samples
    )

    # Map each data to a zero filled array with the channels in the 10-20 system
    data = align_data_to_standard_channels(data, channel_locations)
    # Save the data to a .npy file
    np.save(output_file, data)

    return data, target_sampling_rate, channel_locations


def process_crown_directory(
    input_directory=None,
    output_directory=None,
    recording_sample_rate=256.0,
    target_sampling_rate=128.0,
    channel_locations=None,
    include_timestamp=False,
    notch_filter=[50, 60],
    bandpass_filter=[1, 45],
    verbose=False,
    cutoff_samples=18,
    parallel=False,
):
    file_metadata = {}
    count = 0

    if verbose:
        print(f"Searching {input_directory} for .csv files")
    # Get all .edf files in the directory recursively
    csv_files = glob.glob(os.path.join(input_directory, "**", "*.csv"), recursive=True)
    if verbose:
        print(f"Found {len(csv_files)} .csv files in {input_directory}")

    # Convert channel locations into an array, preserve order
    channel_locations = channel_locations.split(", ")
    channel_locations = [ch for ch in channel_locations if ch in EEG_ALL_CHANNELS]
    # Make output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if parallel:
        with Pool() as pool:
            results = pool.starmap(
                convert_to_npy,
                [
                    (
                        csv_file,
                        os.path.join(output_directory, csv_file.replace("/", "_").replace(".csv", "") + ".npy"),
                        recording_sample_rate,
                        target_sampling_rate,
                        channel_locations,
                        include_timestamp,
                        notch_filter,
                        bandpass_filter,
                        cutoff_samples,
                    )
                    for csv_file in csv_files
                ],
            )
        for result, csv_file in zip(results, csv_files):
            data, final_sample_rate, channel_locations = result
            num_samples = data.shape[1]
            descriptive_file_name = csv_file.replace("/", "_").replace(".csv", "")
            file_metadata[descriptive_file_name] = {
                "sample_rate": final_sample_rate,
                "channel_locations": channel_locations,
                "num_samples": num_samples,
                "file_type": "npy",
            }
            count += 1
            if verbose:
                print(
                    f"{count} of {len(csv_files)}: Processed {num_samples} @ {recording_sample_rate} Hz to {final_sample_rate} Hz and saved to {os.path.join(output_directory, descriptive_file_name + '.npy')}"
                )
    else:
        for csv_file in csv_files:
            descriptive_file_name = csv_file.replace("/", "_").replace(".csv", "")
            output_file = os.path.join(output_directory, descriptive_file_name + ".npy")
            if os.path.exists(output_file):
                if verbose:
                    print(
                        f"{count} of {len(csv_files)}: Output file {output_file} already exists. Skipping processing for {csv_file}"
                    )
                    count += 1
                continue
            data, final_sample_rate, channel_locations = convert_to_npy(
                csv_file,
                output_file,
                channel_locations=channel_locations,
                recording_sample_rate=recording_sample_rate,
                include_timestamp=include_timestamp,
                notch_filter=notch_filter,
                bandpass_filter=bandpass_filter,
                cutoff_samples=cutoff_samples,
            )
            num_samples = data.shape[1]
            file_metadata[descriptive_file_name] = {
                "sample_rate": final_sample_rate,
                "channel_locations": channel_locations,
                "num_samples": num_samples,
                "file_type": "npy",
            }
            count += 1
            if verbose:
                print(
                    f"{count} of {len(csv_files)}: Processed {num_samples} @ {recording_sample_rate} Hz to {final_sample_rate} Hz and saved to {output_file}"
                )

    if verbose:
        print(f"Processed {count} CSV files.")

    descriptive_log_file_name = (
        input_directory.replace("/", "_").replace(".csv", "")
        + "_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + ".json"
    )
    json_file_path = os.path.join(output_directory, descriptive_log_file_name)
    with open(json_file_path, "w") as json_file:
        json.dump(file_metadata, json_file)
    if verbose:
        print(f"Saved file metadata to {json_file_path}")


def read_mat_file(file_path):
    # Load .mat file
    mat = scipy.io.loadmat(file_path)
    # Assuming the EEG data is stored under the key 'data'
    data = mat["data"]
    return data


def read_edf_file(file_path):
    # Load EDF file
    # if ends with .edf
    if file_path.endswith(".edf"):
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif file_path.endswith(".bdf"):
        raw = mne.io.read_raw_bdf(file_path, preload=True)

    # Get channel locations
    channel_locations = raw.ch_names
    # Capitalize the channel names
    channel_locations = [ch.upper() for ch in channel_locations]
    # Clean the data, take only channels with "EEG" and map these to another list with just the channel name
    eeg_channels = [ch for ch in channel_locations if "EEG" in ch]
    # Clean the data, remove the "EEG " and "-REF" or "-LE" or anything else
    eeg_channels_clean = [ch.split("-")[0].replace("EEG ", "") for ch in eeg_channels]
    if len(eeg_channels_clean) == 0 and len(channel_locations) > 0:
        eeg_channels_clean = channel_locations
    # Only take channels found in the EEG 10-10 system
    eeg_channels_picks = [ch for ch in eeg_channels_clean if ch in EEG_ALL_CHANNELS]
    # go through original list with new confirmed channels and make a picks array to later use to pick the data array
    picks = []

    for ch in eeg_channels_picks:
        if "EEG" in channel_locations[eeg_channels_picks.index(ch)]:
            for suffix in ["-LE", "-REF"]:
                if "EEG " + ch + suffix in channel_locations:
                    picks.append(channel_locations.index("EEG " + ch + suffix))
        elif ch in eeg_channels_clean:
            picks.append(eeg_channels_clean.index(ch))

    # Extract data as a numpy array
    data = raw.get_data()
    # Take only the channels in the 10-20 system
    data = data[picks, :]
    # Get the sampling rate
    sampling_rate = raw.info["sfreq"]
    return (data, sampling_rate, eeg_channels_picks)

# Main function
def main():
    # Example for CSV
    # python3 preprocess.py --input_directory data/sessions --output_directory data/npy_sessions --sampling_rate 256 --notch_filter 50 60 --bandpass_filter 1 48

    # Example for TUH EDF
    # python3 preprocess.py --input_directory data/tuh_eeg --output_directory data/npy_tuh_edf --tuh_edf --notch_filter 50 60 --bandpass_filter 1 48

    print(f"Converting CSV or EDF files to NumPy .npy files")
    parser = argparse.ArgumentParser(
        description="Convert Crown CSV, TUH EDF files to NumPy .npy files"
    )
    parser.add_argument(
        "--input_directory", type=str, help="The directory containing the CSV files"
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        help="The directory where the .npy files will be saved",
    )
    parser.add_argument(
        "--recording_sample_rate",
        type=float,
        help="The sampling rate of the data",
        default=None,
    )
    parser.add_argument(
        "--target_sampling_rate",
        type=float,
        help="The target sampling rate",
        default=128.0,
    )
    parser.add_argument(
        "--include_timestamp",
        action="store_true",
        help="Include a timestamp in the output file names",
    )
    parser.add_argument(
        "--notch_filter",
        nargs="+",
        type=float,
        help="The frequencies for the notch filter",
    )
    parser.add_argument(
        "--bandpass_filter",
        nargs=2,
        type=float,
        help="The lowcut and highcut frequencies for the bandpass filter",
    )
    parser.add_argument(
        "--channel_locations",
        type=str,
        help='The channel locations "CP3, C3, F5, PO3, PO4, F6, C4, CP4"',
        default=None,
    )
    parser.add_argument("--tuh_eeg", action="store_true", help="Process TUH EEG files")
    parser.add_argument("--verbose", action="store_true", help="Verbose", default=False)
    parser.add_argument(
        "--cutoff_samples",
        type=int,
        help="The number of samples to cut off from the beginning and end of the data to account for filter ringing",
        default=18,
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process files in parallel",
        default=False,
    )

    args = parser.parse_args()

    # Traverse the directory structure
    print(f"Processing {args.input_directory}")
    if args.tuh_eeg is True:
        process_directory(
            input_directory=args.input_directory,
            output_directory=args.output_directory,
            include_timestamp=args.include_timestamp,
            notch_filter=args.notch_filter,
            bandpass_filter=args.bandpass_filter,
            verbose=args.verbose,
            cutoff_samples=args.cutoff_samples,
            parallel=args.parallel,
            recording_sample_rate=args.recording_sample_rate,
            target_sampling_rate=args.target_sampling_rate,
        )
    else:
        process_crown_directory(
            input_directory=args.input_directory,
            output_directory=args.output_directory,
            include_timestamp=args.include_timestamp,
            notch_filter=args.notch_filter,
            bandpass_filter=args.bandpass_filter,
            recording_sample_rate=args.recording_sample_rate,
            target_sampling_rate=args.target_sampling_rate,
            channel_locations=args.channel_locations,
            verbose=args.verbose,
            cutoff_samples=args.cutoff_samples,
            parallel=args.parallel,
        )
        # for root, dirs, files in os.walk(args.input_directory):
        #     for directory in dirs:
        #         process_crown_directory(input_directory=os.path.join(root, directory), output_directory=args.output_directory, sampling_rate=args.recording_sample_rate, include_timestamp=args.include_timestamp, notch_filter=args.notch_filter, bandpass_filter=args.bandpass_filter)


if __name__ == "__main__":
    main()

