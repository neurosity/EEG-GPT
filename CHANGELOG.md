# v0.1.0

## Initial Release (v0.1.0)

### Features
- Added preprocessing script (`preprocess.py`) to convert CSV or EDF files to NumPy .npy files with various preprocessing steps including notch filtering and bandpass filtering.
- Implemented parallel processing for preprocessing using the `--parallel` flag.
- Added support for TUH EEG files in preprocessing.
- Included detailed README with instructions for using `preprocess.py` and example usage for Crown CSV files and TUH EEG files.
- Integrated `wandb` for experiment tracking.
- Added `CSVLogCallback` class in `src/trainer/make.py` for logging training and evaluation metrics to CSV files.
- Provided `train_parallel.sh` script for distributed training using PyTorch with multiple GPUs.
- Included dependencies in `requirements.txt` and `requirements-dev.txt` for easy setup.

### Documentation
- Comprehensive README with setup instructions, preprocessing details, and example usage.
- Links to external resources for additional help with tools like `tmux`.

### References
- Based on NeuroGPT by Wenhui Cui et al.
- Inspired by Neurosity Foundational Model by Jeremy Nixon and AJ Keller.
