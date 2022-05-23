"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
__all__ = ["volume_grading"]

import numpy as np


def volume_grading(curves: np.ndarray, n_curves: int, n_points: int) -> float:
    """Compute the CORC volume grading.

    Parameters
    ----------
    curves : np.ndarray
        The curves.
    n_curves : int
        The number of curves.
    n_points : int
        The number of points per curve.

    Returns
    -------
    float
        The volume grading.
    """
    temp_curves = curves.copy().reshape(n_curves, n_points, 3)

    values = []
    for i in range(n_curves // 2):
        values.append(
            np.sum(np.abs(temp_curves[i, :, 2] - temp_curves[-i, :, 2])) / n_points
        )

    return float(np.mean(values))
