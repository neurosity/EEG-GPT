python src/eeg/preprocess.py \
    --input_directory data/tuh_eeg \
    --output_directory data/npy_tuh_eeg \
    --notch_filter 50 60 \
    --bandpass_filter 1 48 \
    --tuh_eeg \
    --verbose \
    --parallel