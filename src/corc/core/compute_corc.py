"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
__all__ = ["compute_corc"]

import copy
from typing import Optional, Union

import numpy as np

from corc import landmarks as lm

from .postprocess import humphrey
from .preprocess import preprocess_point_cloud
from .process import corc_feature


def inverse_tranform(
    pcd: np.ndarray,
    *transforms: Union[np.ndarray, float],
) -> np.ndarray:
    """This function applies the inverse transformation to the point cloud.
    
    We given transforms are assumed to be the ones used for the point cloud
    normalization. We reverse the normalization and return the point cloud.
    
    Thus all tranforms are applied in reverse order.

    Args:
        pcd (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    if len(transforms) == 0:
        return pcd
    
    for transform in transforms[::-1]:
        # if it is a rotation matrix
        if isinstance(transform, np.ndarray):
            if transform.shape == (3, 3):
                pcd = np.dot(pcd, transform.T)
            else:
                pcd += transform 
        elif isinstance(transform, float):
            pcd = pcd / transform 
        else:
            raise ValueError(f"Unknown transform type: [{type(transform)}]")
    
    return pcd


def compute_corc(
    point_cloud: np.ndarray,
    landmarks: lm.Landmarks,
    n_curves: int = 128,
    n_points: int = 128,
    delta: Optional[float] = None,
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
    
    The parameter delta, used in the calcuation, describes the area, perpendicular,
    around the radial curve which point to include in the spline computation.
    We estimate a fitting delta values based on the existing points.
    However, the value can be overwritten by the user if needed.

    As the curves need a normlized face point cloud to work properly,
    we afterwards reverse the euclidean transformation and return the
    curves in the original coordinate system.

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
    n_curves : int, optional
        Number of curves/slices to extract. Defaults to 128.
    n_points : int, optional
        Number of points to use for the spline. Defaults to 128.
        Please note, that during the process two points will be
        added and removed again, for the spline fitting.
    delta : float, optional
        The area around a spline to includes the points for estimating
        the spline. Defaults to None.

    preprocessing kwargs: dict, optional
        perimeter_nose_tip: float
            the perimeter of the nose tip estimation. Defaults to 0.015
        threshold_z_axis: float
            the threshold for the z-axis. Defaults to 0.1
        crop_radius_unit: float
            the radius for the cropping in the unit of the face. Defaults to None.
            If given, we expect it to be in the same unit as the original point cloud.

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
        Shape: (n_curves, n_points, 3)
    """
    n_points += 2# + kwargs.get("fix_end_point", False)

    # TODO maybe we can add the option to do this inplace
    point_cloud_ = copy.deepcopy(point_cloud)
    landmarks_ = copy.deepcopy(landmarks)

    point_cloud_, crop_radius, transforms = preprocess_point_cloud(point_cloud_, landmarks_, **kwargs)

    points_3d_fitted = corc_feature(point_cloud_, crop_radius, delta=delta, n_curves=n_curves, n_points=n_points, **kwargs)
    points_3d_fitted = humphrey(
        points_3d_fitted,
        n_curves=n_curves,
        n_points=n_points,
        alpha=kwargs.get("smooth_alpha", 0.5),
        beta=kwargs.get("smooth_beta", 0.5),
        iterations=kwargs.get("smooth_iterations", 5),
    )

    # Remove the two points we added for the spline fitting
    points_3d_fitted = points_3d_fitted.reshape((n_curves, n_points, 3))[..., :-2, :]
    return inverse_tranform(points_3d_fitted, *transforms, landmarks_.nose_tip())
