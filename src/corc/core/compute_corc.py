"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
__all__ = ["compute_corc"]

import numpy as np
import copy
from corc import landmarks as lm

from .postprocess import humphrey
from .preprocess import preprocess_point_cloud
from .process import corc_feature


def compute_corc(
    point_cloud: np.ndarray,
    landmarks: lm.Landmarks,
    n_curves: int = 128,
    n_points: int = 128,
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
    n_curves : int, optional
        Number of curves/slices to extract. Defaults to 128.
    n_points : int, optional
        Number of points to use for the spline. Defaults to 130.

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
    n_points += 2
    
    point_cloud_ = copy.deepcopy(point_cloud)
    landmarks_ = copy.deepcopy(landmarks)

    point_cloud_, crop_radius, (R, s, T1, T2) = preprocess_point_cloud(point_cloud_, landmarks_, **kwargs)
    # calculate the curvature and get the sclice (for visual)
    points_3d_fitted, points_2d_original = corc_feature(
        point_cloud_, crop_radius, delta=delta, n_curves=n_curves, n_points=n_points, **kwargs
    )
    points_3d_fitted = humphrey(
        points_3d_fitted,
        n_curves=n_curves,
        n_points=n_points,
        iterations=10,
    )

    points_3d_fitted = points_3d_fitted.reshape((n_curves, n_points, 3))
    points_3d_fitted = np.delete(points_3d_fitted, obj=n_points - 2, axis=1)
    points_3d_fitted = np.delete(points_3d_fitted, obj=n_points - 2, axis=1)
    points_3d_fitted = points_3d_fitted.reshape((n_curves * (n_points - 2)), 3)

    points_3d_fitted += T2
    points_3d_fitted += T1
    points_3d_fitted = np.dot(points_3d_fitted, R.T)
    points_3d_fitted /= s
    points_3d_fitted += landmarks.nose_tip()

    return points_3d_fitted
