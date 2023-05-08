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
    """
    Compute the radial curve representation of the face.
    
    This function computes the radial curve representation of the face.
    We use the landmarks to estimate the head pose and the nose tip location.
    The point cloud will be normalized and cropped automatically using these
    facial information. 
    Then, we use the radial slices to extract the radial curves of the face.
    The radial curves will be fitted using a spline and the spline will be
    smoothed using the Humphrey algorithm.
    
    The user can specify the number of curves and points to use for the
    radial slices and the spline fitting. (n_curves, n_points)
    
    The user can also specify the delta to use for the radial slices.
    (delta)

    As the curves need a normlized face point cloud to work properly,
    we afterwards reverse the euclidean transformation and return the
    curves in the original coordinate system.

    TODO: Add option to autocompute the delta based on the face size!        

    Parameters
    ----------
    point_cloud : np.ndarray
        The 3D point cloud of the face. The point cloud will be
        normalized and cropped automatically. We use the landmarks
        to estimate the nose tip location and use that as the center
        of the point cloud.
    landmarks : lm.Landmarks
        The 3D landmarks of the face. The landmarks will be used to
        estimate the nose tip location and head pose.
    delta : float, optional
        The delta to use for the radial slices. Defaults to 0.015.
    n_curves : int, optional
        Number of curves/slices to extract. Defaults to 128.
    n_points : int, optional
        Number of points to use for the spline. Defaults to 128.
        Please note, that during the process two points will be
        added and removed again, for the spline fitting.

    preprocessing kwargs: dict, optional
        perimeter_nose_tip: float
            the perimeter of the nose tip estimation. Defaults to 0.015
        threshold_z_axis: float
            the threshold for the z-axis. Defaults to 0.1

    processing kwargs: dict, optional
        fix_end_point: bool, optional
            fix the end point of the spline. Default to True.
            
    smoothing kwargs: dict, optional
        Used in the Humphrey algorithm. Please note we only smooth in the 
        z-axis direction.
        
        smooth_alpha: float, optional
            the alpha value for the smoothing. Defaults to 0.5.
        smooth_beta: float, optional
            the beta value for the smoothing. Defaults to 0.5.
        smooth_iterations: int, optional
            the number of iterations for the smoothing. Defaults to 5.

    Returns
    -------
    np.ndarray
        The radial curve representation of the face
        Shape: (n_curves * n_points, 3)
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
        alpha=kwargs.get("smooth_alpha", 0.5),
        beta=kwargs.get("smooth_beta", 0.5),
        iterations=kwargs.get("smooth_iterations", 5),
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
