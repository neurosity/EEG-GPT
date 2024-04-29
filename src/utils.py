import os
import pdb
import shutil

import h5py
import numpy as np
import gzip
import pickle
import time
import pandas as pd

EEG_10_20_CHANNELS = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']
EEG_10_10_CHANNELS = ['FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ']

EEG_ALL_CHANNELS = list(set(EEG_10_20_CHANNELS + EEG_10_10_CHANNELS))
EEG_10_20_REF_MAP = {
    'FP1-REF': 'FP1', 'FP2-REF': 'FP2', 'F3-REF': 'F3', 'F4-REF': 'F4',
    'C3-REF': 'C3', 'C4-REF': 'C4', 'P3-REF': 'P3', 'P4-REF': 'P4',
    'O1-REF': 'O1', 'O2-REF': 'O2', 'F7-REF': 'F7', 'F8-REF': 'F8',
    'T3-REF': 'T3', 'T4-REF': 'T4', 'T5-REF': 'T5', 'T6-REF': 'T6',
    'T1-REF': 'T1', 'T2-REF': 'T2', 'FZ-REF': 'FZ', 'CZ-REF': 'CZ',
    'PZ-REF': 'PZ'
}

def load_tuh_all(path):
    # files = os.listdir(path)
    filepath = []
    file=""
    # for file in files:
    groups = os.listdir(path)
    for group in groups:
        if os.path.isdir(os.path.join(path, group)):
            subs = os.listdir(os.path.join(path, file, group))
        else:
            continue
        for sub in subs:
            sessions = os.listdir(os.path.join(path, file, group, sub))
            for sess in sessions:
                montages = os.listdir(os.path.join(path, file, group, sub, sess))
                for mont in montages:
                    edf_files = os.listdir(os.path.join(path, file, group, sub, sess, mont))
                    for edf in edf_files:
                        full_path = os.path.join(path, file, group, sub, sess, mont, edf)
                        filepath.append(full_path)
                        # pdb.set_trace()
                        shutil.move(full_path, os.path.join(path, group, sess + "_" + mont + "_" + edf))
                        # pdb.set_trace()
                # load_eeg(filepath[-1])
    return filepath


def load_pickle(filename):
    start_time = time.time()
    with gzip.open(filename, "rb") as file:
        data = pickle.load(file)
    print(data)
    end_time = time.time()
    print("Compressed Elapsed time:", end_time - start_time, "seconds")
    
    return data['data'], np.array(data['channel'])
  

def read_threshold_sub(csv_file, lower_bound=2599, upper_bound=1000000):
    df_read = pd.read_csv(csv_file)
    # Access the list of filenames and time_len
    filenames = df_read['filename'].tolist()
    time_lens = df_read['time_len'].tolist()
    filtered_files = []
    for fn, tlen in zip(filenames, time_lens):
        if (tlen > lower_bound) and (tlen < upper_bound):
            filtered_files.append(fn)
    return filtered_files

def get_epi_files(path, epi_csv, nonepi_csv, lower_bound=2599, upper_bound=1000000):
    epi_full_path = []
    nonepi_full_path = []
    if epi_csv is not None:
        epi_filtered_files = read_threshold_sub(epi_csv, lower_bound, upper_bound)
        epi_full_path = [path + "/epilepsy_edf/" + fn for fn in epi_filtered_files]
    if nonepi_csv is not None:
        nonepi_filtered_files = read_threshold_sub(nonepi_csv, lower_bound, upper_bound)
        nonepi_full_path = [path + "/no_epilepsy_edf/" + fn for fn in nonepi_filtered_files]

    return epi_full_path + nonepi_full_path

def read_sub_list(epi_list):
    with open(epi_list, 'r') as file:
        items = file.readlines()
    # Remove newline characters
    epi_subs = [item.strip() for item in items]
    return epi_subs

def exclude_epi_subs(csv_file, epi_list, lower_bound=2599, upper_bound=1000000, files_all=None):
    epi_subs = read_sub_list(epi_list)
    group_epi_subs = epi_subs
    if files_all is None:
        all_files = read_threshold_sub(csv_file, lower_bound, upper_bound)
    else:
        all_files = files_all
    filtered_files = [f for f in all_files if not any(sub_id in f for sub_id in group_epi_subs)]
    # pdb.set_trace()
    return filtered_files

def exclude_sz_subs(csv_file, lower_bound=2599, upper_bound=1000000, files_all=None):
    if files_all is None:
        all_files = read_threshold_sub(csv_file, lower_bound, upper_bound)
    else:
        all_files = files_all
    with open('sz_subs.txt', 'r') as f:
        sz_subs = f.readlines()
    filtered_files = [f for f in all_files if not any(sub_id in f for sub_id in sz_subs)]
    # pdb.set_trace()
    return filtered_files        

def cv_split_bci(filenames):
    train_folds = []
    val_folds = []
    for i in range(9):
        train_files = filenames[0:i*2] + filenames[i*2+2:]
        validation_files = filenames[i*2 : i*2+2]
        train_folds.append(train_files)
        val_folds.append(validation_files)
    return train_folds, val_folds

def reorder_to_ten_ten(self, data, chann_labels):
    reordered = np.zeros((len(self.ten_ten_labels), data.shape[1]))
    for label, idx in self.ten_ten_labels.items():
        if label in chann_labels:
            mapped_idx = chann_labels[label]
            reordered[idx, :] = data[mapped_idx, :]
        else:
            reordered[idx, :] = np.zeros((1, data.shape[1]))
    return reordered



def map_ref_channels_to_ten_ten(data, chann_labels):
    # Define the mapping from the dataset channel names to the 10-10 system names
    channel_map = {
        'FP1-REF': 'FP1', 'FP2-REF': 'FP2', 'F3-REF': 'F3', 'F4-REF': 'F4',
        'C3-REF': 'C3', 'C4-REF': 'C4', 'P3-REF': 'P3', 'P4-REF': 'P4',
        'O1-REF': 'O1', 'O2-REF': 'O2', 'F7-REF': 'F7', 'F8-REF': 'F8',
        'T3-REF': 'T3', 'T4-REF': 'T4', 'T5-REF': 'T5', 'T6-REF': 'T6',
        'T1-REF': 'T1', 'T2-REF': 'T2', 'FZ-REF': 'FZ', 'CZ-REF': 'CZ',
        'PZ-REF': 'PZ'
    }

    # Initialize an array for the reordered data
    reordered = np.zeros((len(chann_labels), data.shape[1]))

    # Map the channels based on the provided channel_map
    for original_label, new_label in channel_map.items():
        if new_label in chann_labels:
            original_idx = chann_labels[original_label]
            new_idx = chann_labels[new_label]
            reordered[new_idx, :] = data[original_idx, :]
        else:
            # Handle the case where the new label is not found in chann_labels
            print(f"Label {new_label} not found in chann_labels")

    return reordered
