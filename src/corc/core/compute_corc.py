"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
__all__ = ["compute_corc"]

import numpy as np

from corc import landmarks as lm

from .postprocess import humphrey
from .preprocess import preprocess_point_cloud
from .process import corc_feature


def compute_corc(
    point_cloud: np.ndarray,
    landmarks: lm.Landmarks,
    delta: float = 0.015,
    **kwargs,
) -> np.ndarray:
    """Function to extract the curvate features from a point cloud

    Parameters
    ----------
    point_cloud : np.ndarray
        The point cloud of the face
    landmarks : lm.Landmarks
        The landmarks of the face
    delta : float, optional
        The delta to use for the radial slices. Defaults to 0.015.
    kwargs : dict, optional
        Additional arguments to pass to the estimate_facial_curvature function.
        n_curves : int, optional
            Number of curves/slices to extract. Defaults to 128.
        n_points : int, optional
            Number of points to use for the spline. Defaults to 130.
        debug_vars : bool, optional
            If should return debug variables. Defaults to False.

    preprocessing kwargs: dict, optional
        perimeter_nose_tip: float
            the perimeter of the nose tip estimation. Defaults to 0.015
        threshold_z_axis: float
            the threshold for the z-axis. Defaults to 0.1

    processing kwargs: dict, optional
        fix_end_point: bool, optional
            fix the end point of the spline. Default to True.

    Returns
    -------
    np.ndarray
        The radial curve representation of the face
    """
    kwargs["n_curves"] = kwargs.get("n_curves", 128)
    kwargs["n_points"] = kwargs.get("n_points", 128) + 2
    kwargs["debug_vars"] = kwargs.get("debug_vars", False)

    point_cloud, crop_radius = preprocess_point_cloud(point_cloud, landmarks, **kwargs)
    # calculate the curvature and get the sclice (for visual)
    points_3d_fitted, points_2d_original = corc_feature(
        point_cloud, crop_radius, delta=delta, **kwargs
    )
    points_3d_fitted = humphrey(
        points_3d_fitted,
        n_curves=kwargs["n_curves"],
        n_points=kwargs["n_points"],
        iterations=10,
    )

    points_3d_fitted = points_3d_fitted.reshape(
        (kwargs["n_curves"], kwargs["n_points"], 3)
    )

    points_3d_fitted = np.delete(
        points_3d_fitted, obj=kwargs.get("n_points") - 2, axis=1
    )
    points_3d_fitted = np.delete(
        points_3d_fitted, obj=kwargs.get("n_points") - 2, axis=1
    )
    points_3d_fitted = points_3d_fitted.reshape(
        (kwargs["n_curves"] * (kwargs["n_points"] - 2), 3)
    )

    if kwargs["debug_vars"]:
        return points_3d_fitted, (points_2d_original, point_cloud)
    return points_3d_fitted
