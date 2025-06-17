
"""
NeuroInsight: Neontal EEG Preprocessing By  Geletaw Sahle Tegenaw(Ph.D), Norman Delanty (Prof), Tomas Ward (Prof)

- The EEG manifestation of neonatal seizures is defined as: ’clear ictal events characterized by the appearance of sudden, repetitive, evolving stereotyped waveforms with a definite beginning, middle, and end’.

- A key component of neonatal seizure detection is, therefore, the detection of evolving repetition or periodicity in the EEG.

- Periodicity within any signal, x(t), is defined as, x(t) = x(t + T), where t is time and T is the period.

- Source
    - Neonatal intensive care unit (NICU) at the Helsinki University Hospital
- Time period
    - Between 2010 and 2014
- Total number of recordings
    - 79 neonatal EEG recordings
- Data format
    - 79 raw EDF (European Data Format) files
    - 3 annotation files in CSV and Matlab MAT formats
- EEG recording details
    - 19 electrodes positioned as per the international 10-20 standard
- Sampling frequency
    - 256 Hz
    - Signals recorded in microvolts
- Bipolar montage
    - 18 channels generated from electrode pairs
- Annotations
    - Made by 3 independent human experts (labeled A, B, C)
    - Sampled with one second resolution
- Dataset subsets:
    - 40 neonates with seizures annotated by all 3 experts
    - 22 neonates that were seizure-free
    - 17 neonates with seizures annotated by only 1 or 2 experts (excluded from analysis)
- Total number of annotated seizures
    - 1,379
- Dataset availability
    - https://zenodo.org/record/4940267
"""


# Infant IDs which have seizures annotated by all 3 experts
sIDS = [1,4,5,7,9,11,13,14,15,16,17,19,20,21,22,25,31,34,36,38,39,40,41,44,47,50,51,52,62,63,66,67,69,71,73,75,76,77,78,79]
# Infant IDs which are seizure free.
nsIDs = [3,10,18,27,28,29,30,32,35,37,42,45,48,49,53,55,57,58,59,60,70,72]



# edfdirectly_path = "/Users/geletawsahle/Desktop/MNE_EEG/Neontal_eeg_dataset"
edfdirectly_path = '/Users/geletawsahle/Desktop/mlflow_eeg/DataUnpreprocessed'
annotation_path = '/Users/geletawsahle/Desktop/mlflow_eeg/seizures_chunks.csv'

import os
import h5py
import mne
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import re
from scipy import signal
import scipy.io


def load_edf_file(file_path):
    """
    Load an EDF file using MNE with robust error handling.

    Args:
        file_path (str): Path to the EDF file

    Returns:
        mne.io.Raw: Loaded raw EEG data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    if file_path.suffix.lower() not in ['.edf']:
        raise ValueError(f"Unsupported file type: {file_path.suffix}. Only .edf files are supported.")

    try:
        raw = mne.io.read_raw_edf(str(file_path), preload=True)
        print(f"Successfully loaded {file_path}")
        return raw
    except Exception as e:
        print(f"Error reading EDF file {file_path}: {str(e)}")
        raise


# Function to rename EEG channels
def rename_eeg_channels(raw):
    """
    Input: Raw edf file with channel names like 'EEG FP1-REF', 'EEG FP2-Ref', etc.
    Process: Remove prefix and suffix to create standard names like 'FP1', 'FP2', etc. Uses regular expressions for case-insensitivity.
    Output: Standard channel names (e.g., FP1, FP2)
    """
    current_ch_names = raw.ch_names
    name_map = {}
    for ch_name in current_ch_names:
        # Use case-insensitive regex to match channel names
        if re.match(r'^eeg\s', ch_name, re.IGNORECASE) and re.search(r'-ref$', ch_name, re.IGNORECASE):
            new_name = re.sub(r'^eeg\s|-ref$', '', ch_name, flags=re.IGNORECASE)
            name_map[ch_name] = new_name
    raw.rename_channels(name_map)
    return raw

# Function to create bipolar montage
def create_bipolar_montage(raw):
    """
    Create a bipolar montage based on the 10-20 system for consistency across EDF files.
    """
    # Define bipolar channel pairs
    bipolar_pairs = [
        ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
        ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
        ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
        ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
        ('Fz', 'Cz'), ('Cz', 'Pz')
    ]

    # Create new bipolar channel names
    ch_names_bipolar = [f"{pair[0]}-{pair[1]}" for pair in bipolar_pairs]
    data = raw.get_data()
    bipolar_data = []

    for anode, cathode in bipolar_pairs:
        # Get the index of each channel in the pair
        anode_idx = raw.ch_names.index(anode)
        cathode_idx = raw.ch_names.index(cathode)
        # Subtract signals to form bipolar channels
        bipolar_data.append(data[anode_idx] - data[cathode_idx])

    # Convert bipolar data to array
    bipolar_data = np.array(bipolar_data)

    # Create a new Info structure for the bipolar montage
    info = mne.create_info(ch_names=ch_names_bipolar, sfreq=raw.info['sfreq'], ch_types='eeg')
    raw_bipolar = mne.io.RawArray(bipolar_data, info)

    return raw_bipolar


def downsample_eeg(raw_bipolar, downsample_factor=4):
    """
    Down-sample the EEG data.
    """
    data = raw_bipolar.get_data()
    sfreq = raw_bipolar.info['sfreq']
    
    data_downsampled = signal.decimate(data, downsample_factor, axis=1, zero_phase=True)
    
    info_downsampled = mne.create_info(ch_names=raw_bipolar.ch_names, 
                                    sfreq=sfreq/downsample_factor, 
                                    ch_types='eeg')
    raw_downsampled = mne.io.RawArray(data_downsampled, info_downsampled)
    
    return raw_downsampled

def get_segment_label(start_time, end_time, intervals):
    """
    Determine if a segment is no seizure or seizure based on intervals.
    """
    for _, row in intervals.iterrows():
        if start_time >= row['from_sec'] and end_time <= row['to_sec']:
            return 'seizure'
    return 'noseizure'

# # def get_filter_coefficients(raw):
#     """
#     Preprocess EEG data from an EDF file with bandpass (0.3-30 Hz),
#     notch (50 Hz), and custom filters from .mat file.

#     Parameters:
#     raw (mne.io.Raw): The raw EEG data to be filtered.

#     Returns:
#     mne.io.Raw: The preprocessed EEG data.

#     Steps
#     - Added a 50 Hz notch filter using signal.iirnotch()
#     - Implemented a bandpass filter (0.3-30 Hz) using signal.butter()
#     - Used signal.filtfilt() instead of signal.lfilter() for the new filters to achieve zero-phase filtering
#     - Kept the original filters from the .mat file
#     - Added necessary imports
#     - Updated the documentation

#     Considerations:

#     - The 50 Hz notch filter helps remove power line interference
#     - The 0.3-30 Hz bandpass filter helps remove DC offset and high-frequency noise
#     - Using filtfilt() prevents phase distortion in the signal
#     - The original filters are maintained for compatibility
#     """

#     # Get the raw EEG data (numpy array) and sampling frequency
#     dat_eeg = raw.get_data()  # Shape: (n_channels, n_times)
#     sfreq = raw.info['sfreq']  # Sampling frequency

#     # Load the filter coefficients from the .mat file
#     filters_data = scipy.io.loadmat('/Users/geletawsahle/Desktop/mlflow_eeg/filters_db_256.mat')
#     Bn, An = filters_data['Bn'][0], filters_data['An'][0]  # Notch filter coefficients
#     NumL, DenL = filters_data['NumL'][0], filters_data['DenL'][0]  # Low-pass filter
#     NumH, DenH = filters_data['NumH'][0], filters_data['DenH'][0]  # High-pass filter

#     # Design new filters
#     # Notch filter at 50 Hz
#     notch_freq = 50.0  # Frequency to remove
#     quality_factor = 30.0  # Quality factor
#     b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, sfreq)

#     # Bandpass filter (0.3-30 Hz)
#     nyquist = sfreq / 2.0
#     low_cut = 0.3  # High-pass frequency
#     high_cut = 30.0  # Low-pass frequency
#     order = 4  # Filter order

#     b_band, a_band = signal.butter(order, [low_cut/nyquist, high_cut/nyquist],
#                                   btype='band')

#     # Apply the filters to each EEG channel
#     for i in range(dat_eeg.shape[0]):  # Iterate over channels
#         # Apply existing filters from .mat file
#         dat_eeg[i] = signal.lfilter(Bn, An, dat_eeg[i])  # Original notch filter
#         dat_eeg[i] = signal.lfilter(NumL, DenL, dat_eeg[i])  # Original low-pass filter
#         dat_eeg[i] = signal.lfilter(NumH, DenH, dat_eeg[i])  # Original high-pass filter

#         # Apply new filters
#         dat_eeg[i] = signal.filtfilt(b_notch, a_notch, dat_eeg[i])  # 50 Hz notch filter
#         dat_eeg[i] = signal.filtfilt(b_band, a_band, dat_eeg[i])  # 0.3-30 Hz bandpass filter

#     # Convert filtered data back to RawArray
#     info = mne.create_info(ch_names=raw.ch_names, sfreq=sfreq, ch_types='eeg')
#     filtered_raw = mne.io.RawArray(dat_eeg, info)

#     return filtered_raw


def get_filter_coefficients(raw):
    """
    Preprocess EEG data from an EDF file with bandpass (0.3-30 Hz),
    optional notch (50 Hz), and custom filters from a .mat file.

    Parameters:
        raw (mne.io.Raw): The raw EEG data to be filtered.

    Returns:
        mne.io.Raw: The preprocessed EEG data.
    """
    import scipy.io
    from scipy import signal
    import mne

    # Get the raw EEG data (numpy array) and sampling frequency
    dat_eeg = raw.get_data()  # Shape: (n_channels, n_times)
    sfreq = raw.info['sfreq']  # Sampling frequency

    # Load the filter coefficients from the .mat file
    filters_data = scipy.io.loadmat('/Users/geletawsahle/Desktop/mlflow_eeg/filters_db_256.mat')
    Bn, An = filters_data['Bn'][0], filters_data['An'][0]  # Notch filter coefficients
    NumL, DenL = filters_data['NumL'][0], filters_data['DenL'][0]  # Low-pass filter
    NumH, DenH = filters_data['NumH'][0], filters_data['DenH'][0]  # High-pass filter

    # Design new filters
    nyquist = sfreq / 2.0  # Nyquist frequency

    # Bandpass filter (0.3-30 Hz)
    low_cut = 0.3  # High-pass frequency
    high_cut = 30.0  # Low-pass frequency
    b_band, a_band = signal.butter(4, [low_cut / nyquist, high_cut / nyquist], btype='band')

    # Apply the filters to each EEG channel
    for i in range(dat_eeg.shape[0]):  # Iterate over channels
        # Apply existing filters from .mat file
        dat_eeg[i] = signal.lfilter(Bn, An, dat_eeg[i])  # Original notch filter
        dat_eeg[i] = signal.lfilter(NumL, DenL, dat_eeg[i])  # Original low-pass filter
        dat_eeg[i] = signal.lfilter(NumH, DenH, dat_eeg[i])  # Original high-pass filter

        # Apply new filters
        # Conditional 50 Hz notch filter
        notch_freq = 50.0  # Frequency to remove
        quality_factor = 30.0  # Quality factor
        if notch_freq < nyquist:
            b_notch, a_notch = signal.iirnotch(notch_freq / nyquist, quality_factor)
            dat_eeg[i] = signal.filtfilt(b_notch, a_notch, dat_eeg[i])  # Apply notch filter
        else:
            print(f"Skipping 50 Hz notch filter as notch frequency {notch_freq} exceeds Nyquist frequency {nyquist}")

        # Apply bandpass filter
        dat_eeg[i] = signal.filtfilt(b_band, a_band, dat_eeg[i])  # 0.3-30 Hz bandpass filter

    # Convert filtered data back to RawArray
    info = mne.create_info(ch_names=raw.ch_names, sfreq=sfreq, ch_types='eeg')
    filtered_raw = mne.io.RawArray(dat_eeg, info)

    return filtered_raw


def preprocess_eeg_data(raw, intervals=None, sfreq=64, duration=1, eeg_file_id="eeg1", expert="A"):
    """
    Preprocess EEG data and save segments.
    If no intervals (annotations) are provided, segments entire recording as noseizure.
    """
    print(f"Processing file {eeg_file_id} for expert {expert}")

    # Resample the data
    raw.resample(sfreq)
    data = raw.get_data()
    n_samples = int(sfreq * duration)

    # Create output directories
    # normal_dir = Path("/Users/geletawsahle/Desktop/mlflow_eeg/data/DownSampled") / "C1_noseizure"
    # notnormal_dir = Path("/Users/geletawsahle/Desktop/mlflow_eeg/data/DownSampled") / "C1_seizure"

    normal_dir = Path("/home/geletaw/eeg_llm/data") / "C1_noseizure"
    notnormal_dir = Path("/home/geletaw/eeg_llm/data") / "C1_seizure"

    

    # output_dir_noseizure = "/Users/geletawsahle/Desktop/mlflow_eeg/data/DownSampled/C4_noseizure"
    # #"C1_noseizure, C2_noseizure, C3_noseizure, C4_noseizure"
    # output_dir_seizure = "/Users/geletawsahle/Desktop/mlflow_eeg/data/DownSampled/C4_seizure"
    # # "C1_seizure, C2_seizure, C3_seizure, C4_seizure"
    
    normal_dir.mkdir(parents=True, exist_ok=True)
    notnormal_dir.mkdir(parents=True, exist_ok=True)

    segments_info = {
        'noseizure': 0,
        'seizure': 0,
        'total': 0
    }

    # Process segments
    for i in range(0, data.shape[1] - n_samples + 1, n_samples):
        segment = data[:, i:i+n_samples]
        start_time = i / sfreq
        end_time = (i + n_samples) / sfreq

        # If no intervals provided or empty intervals, mark all as noseizure
        if intervals is None or len(intervals) == 0:
            label = 'noseizure'
        else:
            label = get_segment_label(start_time, end_time, intervals)

        filename = f"expert_{expert}_{eeg_file_id}_{label}_{int(start_time)}-{int(end_time)}.h5"
        output_dir = normal_dir if label == 'noseizure' else notnormal_dir
        file_path = output_dir / filename

        with h5py.File(file_path, 'w') as f:
            f.create_dataset('eeg', data=segment)
            f.create_dataset('label', data=0 if label == 'noseizure' else 1)

        segments_info[label] += 1
        segments_info['total'] += 1

    print(f"File {eeg_file_id} processing complete:")
    print(f"- No seizure segments: {segments_info['noseizure']}")
    print(f"- Seizure segments: {segments_info['seizure']}")
    print(f"- Total segments: {segments_info['total']}")

    return segments_info

def process_edf_files(edf_directory, csv_annotations_path, experts=['A', 'B', 'C']):
    """
    Process all EDF files in a directory.
    """
    edf_directory = Path(edf_directory)
    print(f"Looking for EDF files in: {edf_directory.absolute()}")

    # Find all EDF files
    edf_files = list(edf_directory.glob('eeg*.edf'))
    print(f"Found {len(edf_files)} EDF files: {[f.name for f in edf_files]}")

    if not edf_files:
        raise FileNotFoundError(f"No EDF files found in {edf_directory}")

    # Load annotations
    try:
        print(f"Loading annotations from: {csv_annotations_path}")
        all_annotations = pd.read_csv(csv_annotations_path)
        print(f"Loaded {len(all_annotations)} annotation records")
    except Exception as e:
        print(f"Error reading annotations file: {str(e)}")
        raise

    # Validate columns
    required_columns = ['Expert', 'Id', 'from_sec', 'to_sec']
    if not all(col in all_annotations.columns for col in required_columns):
        raise ValueError(f"Missing columns in annotations file. Required: {required_columns}")

    processing_summary = {
        'total_files': len(edf_files),
        'processed_files': 0,
        'total_segments': 0,
        'normal_segments': 0,
        'notnormal_segments': 0,
        'file_details': []
    }

    # Process each file
    for edf_file in sorted(edf_files):
        try:
            # Extract file ID
            file_id = int(edf_file.stem.replace('eeg', ''))
            print(f"\nProcessing file: {edf_file.name} (ID: {file_id})")

            # Load EDF
            raw = load_edf_file(edf_file)
            print("Before renaming channel", raw.ch_names)

            # Rename the edf files
            raw = rename_eeg_channels(raw)
            print("after renaming channel", raw.ch_names)

            # Create bipolar montage
            raw = create_bipolar_montage(raw)
            print("after bipolar montage", raw.ch_names)

            #Downsample  by a factor of 4  such as 256  to 64
            raw = downsample_eeg(raw, downsample_factor=4)
            raw.plot()

            print("After preprocessing EEG")
            # preprocess
            filtered_raw = get_filter_coefficients(raw)

            # Process for each expert
            for expert in experts:
                # Filter annotations
                expert_annotations = all_annotations[
                    (all_annotations['Expert'] == expert) &
                    (all_annotations['Id'] == file_id)
                ]

                if len(expert_annotations) == 0:
                    print(f"No annotations found for expert {expert}, file ID {file_id}")
                    print(f"Processing entire file as noseizure segments")
                    # Process without annotations
                    segments_info = preprocess_eeg_data(
                        filtered_raw,
                        intervals=None,  # Pass None to indicate no annotations
                        eeg_file_id=edf_file.stem,
                        expert=expert
                    )
                else:
                    print(f"Found {len(expert_annotations)} annotations for expert {expert}")
                    # Process with annotations
                    segments_info = preprocess_eeg_data(
                        filtered_raw,
                        expert_annotations,
                        eeg_file_id=edf_file.stem,
                        expert=expert
                    )

                # Update summary
                processing_summary['total_segments'] += segments_info['total']
                processing_summary['normal_segments'] += segments_info['noseizure']
                processing_summary['notnormal_segments'] += segments_info['seizure']

            processing_summary['processed_files'] += 1
            processing_summary['file_details'].append({
                'file': edf_file.name,
                'status': 'success'
            })

        except Exception as e:
            print(f"Error processing {edf_file.name}: {str(e)}")
            processing_summary['file_details'].append({
                'file': edf_file.name,
                'status': 'error',
                'error': str(e)
            })

    return processing_summary

# Function to count and visualize the number of files in "seizures" and "noseizuers" folders
def count_and_visualize_files(data_dir):
    # Define directories for seizures and noseizures files
    noseizures_dir = os.path.join(data_dir, "noseizure")
    seizures_dir = os.path.join(data_dir, "seizure")

    # Count the .h5 files in each directory
    noseizures_files_count = len([f for f in os.listdir(noseizures_dir) if f.endswith('.h5')])
    seizures_files_count = len([f for f in os.listdir(seizures_dir) if f.endswith('.h5')])

    # Store counts in a dictionary for easy plotting
    counts = {'No Seizures': noseizures_files_count, 'Seizures': seizures_files_count}

    # Create bar graph with counts
    plt.figure(figsize=(8, 6))
    bars = plt.bar(counts.keys(), counts.values(), color=['blue', 'red'])

    # Add counts on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Set graph labels and title
    plt.xlabel('Label')
    plt.ylabel('Number of Files')
    plt.title('Number of .h5 Files in Seizures and No Seizures Folders')
    # Save the figure before showing it
    plt.savefig("seizures_vs_noseizures_segement.png")  
    plt.show()

    # Print counts for reference
    print(f"Number of no seizures files: {noseizures_files_count}")
    print(f"Number of seizures files: {seizures_files_count}")

def main():
    """
    Main execution function.
    """
    # Get current directory
    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")

    # Set up paths
    edf_directory = edfdirectly_path
    csv_annotations_path = annotation_path


    print("\nInput Paths:")
    print(f"EDF Directory: {edf_directory}")
    print(f"Annotations File: {csv_annotations_path}")

    # Process files
    try:
        summary = process_edf_files(edf_directory, csv_annotations_path)

        print("\nProcessing Summary:")
        print(f"Total files found: {summary['total_files']}")
        print(f"Files processed successfully: {summary['processed_files']}")
        print(f"Total segments created: {summary['total_segments']}")
        print(f"No seizure segments: {summary['normal_segments']}")
        print(f"Seizure segments: {summary['notnormal_segments']}")

        print("\nFile Processing Details:")
        for file_detail in summary['file_details']:
            status = "✓" if file_detail['status'] == 'success' else "✗"
            print(f"{status} {file_detail['file']}")
            if file_detail['status'] == 'error':
                print(f"  Error: {file_detail['error']}")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    count_and_visualize_files("/home/geletaw/eeg_llm/data")
