"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
__all__ = ["load_landmarks"]

import pathlib

import numpy as np
import pandas as pd
import scipy.io


def load_landmarks(path: pathlib.Path, **kwargs) -> np.ndarray:
    """Load landmarks from a file.

    Automatically detects the file type and calls the corresponding function.

    Parameters
    ----------
    path : pathlib.Path
        Path to the bnd file

    Returns
    -------
    np.ndarray
        Array of landmarks
    """
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_landmarks_txt(path, **kwargs)

    if suffix == ".csv":
        return load_landmarks_csv(path, **kwargs)

    if suffix == ".npy":
        return load_landmarks_npy(path, **kwargs)

    if suffix == ".mat":
        return load_landmarks_mat(path, **kwargs)

    if suffix == ".bnd":
        return load_landmarks_bnd(path, **kwargs)

    raise ValueError(f"Unknown file type: {suffix}")


def load_landmarks_txt(path: pathlib.Path, **kwargs) -> np.ndarray:
    """Load landmarks from bnd file

    Parameters
    ----------
    path : pathlib.Path
        Path to the bnd file

    Returns
    -------
    np.ndarray
        Array of landmarks
    """
    landmarks = pd.read_csv(
        path, delim_whitespace=True, header=None, index_col=0
    ).values
    return landmarks.reshape((-1, 3))


def load_landmarks_npy(path: pathlib.Path, **kwargs) -> np.ndarray:
    """Load landmarks from bnd file

    Parameters
    ----------
    path : pathlib.Path
        Path to the bnd file

    Returns
    -------
    np.ndarray
        Array of landmarks
    """
    return np.load(path)


def load_landmarks_mat(path: pathlib.Path, **kwargs) -> np.ndarray:
    """Load landmarks from bnd file

    Parameters
    ----------
    path : pathlib.Path
        Path to the bnd file

    Returns
    -------
    np.ndarray
        Array of landmarks
    """

    temp = scipy.io.loadmat(path)
    landmarks = temp["stereo"][0][:][1]
    return landmarks.reshape((-1, 3))


def load_landmarks_csv(path: pathlib.Path, **kwargs) -> np.ndarray:
    """Load landmarks from csv file

    Parameters
    ----------
    path : pathlib.Path
        Path to the csv file

    Returns
    -------
    np.ndarray
        Array of landmarks
    """
    landmarks = pd.read_csv(path, sep=kwargs.get("sep", ","), index_col=None)
    x = landmarks["x"].values
    y = landmarks["y"].values
    z = landmarks["z"].values
    return np.stack((x, y, z), axis=1)


def load_landmarks_bnd(path: pathlib.Path, **kwargs) -> np.ndarray:
    """Load landmarks from bnd file

    Parameters
    ----------
    path : pathlib.Path
        Path to the bnd file

    Returns
    -------
    np.ndarray
        Array of landmarks
    """

    # This file type is especially for the BU-3DFE data set
    # https://stackoverflow.com/questions/12643009/regular-expression-for-floating-point-numbers
    # 3rd answer
    # regex_float = "[+-]?(\d+([.]\d*)?(e[+-]?\d+)?|[.]\d+(e[+-]?\d+)?)"
    # regex_float = "-?\d+\.?\d*"
    landmarks_temp = []
    with open(path, "r", encoding="utf-8") as bnd_file:
        for i, line in enumerate(bnd_file):
            # the values a separated by tabs, thus split them like this
            # remove all empty strings as two tabs have been used
            # after that 4 values should remain whereas the last 3 values are
            # the coordinates
            landmark = list(filter(lambda x: x != "", line.split("\t")))
            if len(landmark) == 4:
                landmarks_temp.append([float(x) for x in landmark[1:]])
            else:
                raise SyntaxError(f"Unknown format :{i} with {landmark} in file {path}")
    landmarks = np.asarray(landmarks_temp, dtype=np.float32).reshape((-1, 3))
    return landmarks
