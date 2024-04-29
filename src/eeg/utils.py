#add requirements
import scipy.io
import mne

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