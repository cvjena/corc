"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
import numpy as np


def compute_curvature_2d_spline(curve_2d: np.ndarray) -> np.ndarray:
    """Compute the curvature of a two-dimensional curve

    This function uses the numerical differentiation to compute the
    first and second gradient in x and y direction. These will be used
    to estimate the curvature of the given 2d curve.

    Parameters
    ----------
    curve_2d : np.ndarray
        2d curve

    Returns
    -------
    np.ndarray
        curvature of the given 2d curve
    """
    dx_dt = np.gradient(curve_2d[:, 0])
    dy_dt = np.gradient(curve_2d[:, 1])
    d2x_dx = np.gradient(dx_dt)
    d2y_dy = np.gradient(dy_dt)
    return (
        np.abs(d2x_dx * d2x_dx - d2y_dy * d2y_dy)
        / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    )
