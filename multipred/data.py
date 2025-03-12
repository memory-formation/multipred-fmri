import numpy as np
import nibabel as nib
import os


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



