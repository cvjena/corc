"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
__all__ = ["compute_corc", "compute_corc_time", "inverse_tranform"]

import time
from typing import Optional

import numpy as np
from corc import landmarks as lm

from corc.core.postprocess import humphrey
from corc.core.preprocess import preprocess_point_cloud
from corc.core.process import corc_feature

class _TIMER:
    def __init__(self) -> None:
        self._start = None
        self._end = None
        self._verbose = False

    def set_verbose(self, verbose: bool) -> None:
        self._verbose = verbose

    def tick(self) -> None:
        self._start = time.time()

    def tock(self, msg: str) -> None:
        self._end = time.time()
        if self._verbose:
            print(f"{msg}: {self._end - self._start:4.5f} s")

timer = _TIMER()

def inverse_tranform(
    pcd: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """This function applies the inverse transformation to the point cloud.
    
    We given transforms are assumed to be the ones used for the point cloud
    normalization. We reverse the normalization and return the point cloud.
    
    Thus all tranforms are applied in reverse order.

    Args:
        pcd (np.ndarray): The point cloud to transform.
        translation (np.ndarray): The translation vector.
        rotation (np.ndarray): The rotation matrix.

    Returns:
        np.ndarray: The transformed point cloud. Should be in the original
        orientation and position of the 3d surface scan.
    """
    pcd = pcd @ rotation.T
    pcd = pcd + translation
    return pcd

def compute_corc(
    point_cloud: np.ndarray,
    landmarks: lm.Landmarks,
    n_curves: int = 128,
    n_points: int = 128,
    delta: Optional[float] = None,
    palsy_right: bool = False,
    verbose: bool = False,
    **kwargs,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], float]:
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
        angle_offset: int, optional
            the angle offset for the spline fitting. Defaults to 0.
            
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
    tuple[np.ndarray, np.ndarray]
        The translation and rotation vectors used for the normalization.
        The translation vector is the nose tip location.
        The rotation vector is the head pose.
    float
        The radius used for the cropping.
    """
    
    timer.set_verbose(verbose)
    
    timer.tick()
    add_points = 1 + int(kwargs.get("fix_end_point", False))
    n_points += add_points
    timer.tock("[CORC] setup")

    timer.tick() 
    point_cloud, crop_radius, transforms = preprocess_point_cloud(point_cloud, landmarks, **kwargs)
    if palsy_right:
        point_cloud[:, 0] *= -1  # Flip the point cloud on the x-axis
    timer.tock("[CORC] preprocess_point_cloud")

    timer.tick() 
    curves = corc_feature(point_cloud, crop_radius, delta=delta, n_curves=n_curves, n_points=n_points, **kwargs)
    timer.tock("[CORC] corc_feature")
    
    timer.tick() 
    if (iter := kwargs.get("smooth_iterations", 1)) != 0:
        curves = humphrey(
            curves,
            n_curves=n_curves,
            n_points=n_points,
            alpha=kwargs.get("smooth_alpha", 0.5),
            beta=kwargs.get("smooth_beta", 0.5),
            iterations=iter,
        )
    timer.tock("[CORC] humphrey")

    timer.tick()
    # Remove the two points we added for the spline fitting
    curves = curves.reshape((n_curves, n_points, 3))[..., :-add_points, :]
    timer.tock("[CORC] reshape")
    
    if kwargs.get("do_offset", False):
        curves += np.array([0.0, 0.0, 2.0])
    
    return curves, transforms, crop_radius

def compute_corc_time(
    **kwargs,
) -> np.ndarray:
    return compute_corc(verbose=True, **kwargs)
