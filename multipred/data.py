import os

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import norm


def extract_slice(data, axis, position, flip_axial=False):
    """Extract a 2D slice from 3D data and optionally flip axial slices."""
    if axis == "x":
        slice_ = data[int(position), :, :]
        if flip_axial:
            slice_ = np.rot90(slice_, k=3)
    elif axis == "y":
        slice_ = data[:, int(position), :]
        if flip_axial:
            slice_ = np.rot90(slice_, k=3)
    elif axis == "z":
        slice_ = data[:, :, int(position)]
        if flip_axial:
            slice_ = np.rot90(slice_, k=3)  # Flip the slice vertically
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return slice_


def validate_mri_input(data):
    """
    Validates the input and converts it to a NumPy array if necessary.

    Parameters:
    - data: str (path to file), nib.Nifti1Image, or numpy.ndarray

    Returns:
    - NumPy array representation of the data.

    Raises:
    - ValueError: If the input is not a valid path, Nifti1Image, or NumPy array.
    """
    if isinstance(data, np.ndarray):
        # Already a NumPy array
        return data
    elif isinstance(data, nib.Nifti1Image):
        # Convert Nifti1Image to NumPy array
        return data.get_fdata()
    elif isinstance(data, str) and os.path.exists(data):
        # Load data from file path
        img = nib.load(data)
        return img.get_fdata()
    else:
        raise ValueError(
            f"Invalid input type: {type(data)}. Input must be a file path, Nifti1Image, or NumPy array."
        )



def create_timefiles(subj_df, regressors, onset_var, duration, path):
    # This functions creates a timefile for each combination of regressors specified, based on the timings contained in the "onset_var" column of the input dataframe
    # The timefiles are saved in the path specified as text files
    # The timefiles are formatted to be used with FSL. They are 3 columns: onset, duration, and parametric modulations (ones)
    
    # Check if path exists, if not create it
    if not os.path.exists(path): os.makedirs(path)

    # Group dataframe by specified regressors
    grouped_data = subj_df.groupby(regressors)

    # Dictionary to store the results
    grouped_onsets = {}

    for name, group in grouped_data:
        grouped_onsets[name] = group[onset_var].unique().tolist()

    for key in list(grouped_onsets.keys()):
        onsets = np.round(grouped_onsets[key], 2)
        durations = np.repeat(duration, len(onsets))
        ones = np.ones(len(onsets))
        filename = '_'.join(map(str, key)); filename = filename + '.txt'
        to_txt = pd.DataFrame({"onset": onsets, "dur" : durations, 'ones': ones})
        np.savetxt(os.path.join(path, filename) , to_txt.values, fmt='%1.2f')

    return




def create_bids_event_files(subj_df, regressors, onset_var, duration, task, run, path):
    """
    Creates a BIDS-compatible event file (.tsv) for each subject/run.
    
    Parameters:
    - subj_df: DataFrame containing event data.
    - regressors: List of columns to group by (e.g., ['condition']).
    - onset_var: Column name indicating event onset times.
    - duration: Fixed duration (or column name for variable duration).
    - path: Directory where .tsv files will be saved.

    Output:
    - Saves BIDS-compatible event files named `sub-XX_task-YYY_run-ZZ_events.tsv`
    """

    # Ensure output directory exists
    os.makedirs(path, exist_ok=True)

    # Get normal trials
    # Group dataframe by specified regressors
    grouped_data = subj_df[subj_df["catch"]==0].groupby(regressors)

    # Dictionary to store grouped onset data
    grouped_onsets = {}

    for name, group in grouped_data:
        grouped_onsets[name] = group[onset_var].unique().tolist()

    # Create a single BIDS-compatible event file
    event_list = []

    for key, onsets in grouped_onsets.items():
        trial_type = '_'.join(map(str, key))  # Combine regressor values into a label
        for onset in np.round(onsets, 2):
            event_list.append([onset, duration, trial_type])

    # Get visual catch trials
    vcatch = subj_df[subj_df["catch"]==1]
    vcatch_onsets = np.round(vcatch[vcatch["catch"]==1][onset_var].unique(), 2)
    for onset in vcatch_onsets:
        event_list.append([onset, duration, "vcatch"])

    # Get auditory catch trials
    acatch = subj_df[subj_df["catch"]==2]
    acatch_onsets = np.round(acatch[acatch["catch"]==2][onset_var].unique(), 2)
    for onset in acatch_onsets:
        event_list.append([onset, duration, "acatch"])


    # Convert to DataFrame
    events_df = pd.DataFrame(event_list, columns=["onset", "duration", "trial_type"])

    # Sort by onset time
    events_df = events_df.sort_values(by="onset")

    # Generate BIDS-compliant filename
    bids_filename = f"{subj_df['id'].iloc[0]}_task-{task}_run-0{run + 1}_events.tsv"

    # Save as tab-separated values (.tsv)
    events_df.to_csv(os.path.join(path, bids_filename), sep="\t", index=False)

    return


def compute_dprime(data, signal_col, condition_cols=None):
    """
    Generalized function to compute d' and criterion for given conditions.

    Parameters:
    - data (pd.DataFrame): The input dataframe containing trial data.
    - signal_col (str): Column name indicating signal/noise classification (e.g., 'signal_noise').
    - condition_cols (list, optional): List of column names for experimental conditions (e.g., ['v_pred', 'run']).
      If None, computes overall d' and criterion for each subject.

    Returns:
    - pd.DataFrame: A dataframe with d' and criterion for each subject and condition combinations (if specified).
    """
    results = []

    # Group by subject and specified condition columns
    group_by_cols = ['subj'] + (condition_cols if condition_cols else [])
    grouped = data.groupby(group_by_cols)

    for group_keys, group in grouped:
        # Determine the subject and conditions
        keys = group_keys if isinstance(group_keys, tuple) else (group_keys,)
        subj = keys[0]
        conditions = keys[1:] if condition_cols else None

        # Count signal and noise trials
        signal_trials = group[group[signal_col] == 'signal']
        noise_trials = group[group[signal_col] == 'noise']

        # Calculate hit and false alarm rates
        n_signal = len(signal_trials)
        n_noise = len(noise_trials)
        hit_rate = len(signal_trials[signal_trials['decision_type'] == 'hit']) / n_signal if n_signal > 0 else 0
        fa_rate = len(noise_trials[noise_trials['decision_type'] == 'fa']) / n_noise if n_noise > 0 else 0
        miss_rate = len(signal_trials[signal_trials['decision_type'] == 'miss']) / n_signal if n_signal > 0 else 0
        cr_rate = len(noise_trials[noise_trials['decision_type'] == 'CR']) / n_noise if n_noise > 0 else 0

        # Apply corrections to rates of 0 or 1
        hit_rate = min(max(hit_rate, 0.5 / max(n_signal, 1)), 1 - 0.5 / max(n_signal, 1))
        fa_rate = min(max(fa_rate, 0.5 / max(n_noise, 1)), 1 - 0.5 / max(n_noise, 1))

        # Calculate d' and criterion
        d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
        criterion = -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))

        # Prepare result dictionary
        result = {'subj': subj, 'd_prime': d_prime, 'criterion': criterion, 'hit_rate': hit_rate, 'fa_rate': fa_rate, 'miss_rate': miss_rate, 'cr_rate': cr_rate}
        if condition_cols:
            for col, val in zip(condition_cols, conditions):
                result[col] = val
        results.append(result)

    return pd.DataFrame(results)