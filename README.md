# WARN: UNDER ACTIVE DEVELOPMENT

# Intro

## What is this?

This is a project to build a foundation model for EEG data. It is based on the NeuroGPT model by Wenhui Cui et al.

## What is EEG?

EEG is short for Electroencephalography. It is a non-invasive method of measuring brain activity. It is used to detect brain signals that are not easily captured by other methods, such as seizures and emotional states.

## Who is Neurosity?

Neurosity is a technology company that specializes in creating brain-computer interfaces. They have developed a device called the [Crown](neurosity.co), which is a wearable EEG headset that can measure brain activity. The data collected by the Crown can be used for a variety of applications, including mental health monitoring, cognitive enhancement, and controlling devices with your mind. Neurosity's mission is to empower individuals with the ability to understand and enhance their mental state.

# About this Repo

## Standardizing EEG

1. The model allows any electrode found in the 10-10 or 10-20 system.
2. The model expects 128.0 Hz sampling rate

## Preprocessing 

The preprocessing script (`preprocess.py`) converts CSV or EDF files to NumPy .npy files. It applies various preprocessing steps to the data, including notch filtering and bandpass filtering. Here are the available arguments for the preprocessing script:

```bash
python3 src/eeg/preprocess.py [arguments]
```

### Arguments:

- `--input_directory`: The directory containing the input files (CSV or EDF).
- `--output_directory`: The directory where the processed .npy files will be saved.
- `--recording_sample_rate`: The original sampling rate of the data (default: None).
- `--include_timestamp`: Include a timestamp in the output file names (flag).
- `--notch_filter`: The frequencies for the notch filter (e.g., 50 60 for both 50Hz and 60Hz).
- `--bandpass_filter`: The lowcut and highcut frequencies for the bandpass filter (e.g., 1 48 for 1-48Hz).
- `--channel_locations`: The channel locations (e.g., "CP3, C3, F5, PO3, PO4, F6, C4, CP4").
- `--tuh_eeg`: Process TUH EEG files (flag).
- `--verbose`: Enable verbose output (flag).
- `--cutoff_samples`: The number of samples to cut off from the beginning and end of the data to account for filter ringing (default: 18).
- `--parallel`: Process files in parallel (flag).

### Example Usage:

For Crown CSV files:

```bash
python3 src/eeg/preprocess.py --input_directory data/sessions --output_directory data/npy_sessions --recording_sample_rate 256 --notch_filter 50 60 --bandpass_filter 1 48 --cutoff_samples 18
```

For TUH EEG files:

```bash
python3 src/eeg/preprocess.py --input_directory edf/ --output_directory data/npy_tuh_eeg --notch_filter 50 60 --bandpass_filter 1 48 --verbose --tuh_eeg --cutoff_samples 18
```

# Based on

## NeuroGPT

[NeuroGPT](https://github.com/wenhui0206/NeuroGPT/blob/main/src/batcher/downstream_dataset.py) by Wenhui Cui et al.

```bibtex
@misc{cui2024neurogpt,
      title={Neuro-GPT: Towards A Foundation Model for EEG},
      author={Wenhui Cui and Woojae Jeong and Philipp Thölke and Takfarinas Medani and Karim Jerbi and Anand A. Joshi and Richard M. Leahy},
      year={2024},
      eprint={2311.03764},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Neurosity Foundational Model

[JeremyNixon/neurosity)](https://github.com/JeremyNixon/neurosity)

```bibtex
@misc{neurosity_eeg_dataset,
  title={Neurosity EEG Dataset},
  author={Nixon, Jeremy and Keller, AJ},
  year={2024},
  url={https://github.com/JeremyNixon/neurosity}
}
```
