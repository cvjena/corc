"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""

import multiprocessing as mp
from typing import Optional

import numpy as np

from corc import utils

from corc.core.process_spline import to_spline_3d


def corc_feature(
    points: np.ndarray,
    crop_radius: float,
    n_curves: int = 128,
    n_points: int = 130,
    delta: Optional[float] = None,
    **kwargs,
) -> np.ndarray:
    """Estimate the curvature of a face

    Extract radial slices around the nose tip and fit splines inside these points.
    The algorithm computes the corc feature.

    Parameters
    ----------
    points : np.ndarray
        vertices of the point cloud
    crop_radius : float
        radius of the crop circle
    n_curves : int, optional
        the number of radial stripes to extract. Defaults to 128.
    n_points : int, optional
        the number of points to use for the spline fitting. Defaults to 130.
    delta : float, optional
        delta area do decide which points to include. Default to None.
    kwargs : dict, optional
        additional arguments for `to_spline_3d`
        fix_end_point: bool, optional
            fix the end point of the spline. Default to True.
        angle_offset: int, optional
            the angle offset for the spline fitting. Defaults to 0.

    Returns
    -------
    np.ndarray
        the radial curve feature
    """
    # we interpolate a bit more and cut the last part to avoid spikes in the end
    # this is not a problem as the end of spline does not contain a lot of information
    offset = kwargs.get("angle_offset", 0)
    rotations = utils.rotz_v(np.linspace(0+offset, 360+offset, num=n_curves, endpoint=False)) 
    sample_space = np.linspace(0, 1, n_points, endpoint=True)
    radial_slices = to_radial_slices(points, rotations, n_curves, delta)

    with mp.Pool() as pool:
        args = [((s, r, sample_space, crop_radius), kwargs) for s, r in zip(radial_slices, rotations)]
        spline_results = np.asarray(list(pool.imap(mp_to_spline_3d, args)))
    return spline_results.reshape((n_curves * n_points, 3))


def to_radial_slices(
    points: np.ndarray,
    rotations: np.ndarray,
    n_curves: int = 128,
    delta: Optional[float] = None,
) -> np.ndarray:
    """Extract the radial slices around nose tip

    This function extracts the radial slices around the nose tip.
    We move with a rotating virtual plane around the nose tip and
    collect all points close the plane.

    Parameters
    ----------
    points : np.ndarray
        vertices of the point cloud
    rotations : np.ndarray
        rotation matrices for the virtual planes
    n_curves : int, optional
        the number of radial stripes to extract. Defaults to 128.
    delta : float, optional
        delta area do decide which points to include. Default to None.
        Is computed based on the amount of points if None.

    Returns
    -------
    np.ndarray
        the radial slices
    """

    radial_slices: list[np.ndarray] = [None] * n_curves  # type: ignore
    points_projection_x: np.ndarray = np.zeros_like(points)
    points_projection_y: np.ndarray = np.zeros_like(points)

    delta = delta or 0.5

    for i in range(n_curves):
        # intersect the point cloud with a plane whose normal vector is given by the
        # current rotation
        points_projection_x = np.dot(points, np.dot(rotations[i], utils.AXIS_X))
        points_projection_y = np.dot(points, np.dot(rotations[i], utils.AXIS_Y))
        # extract only the points which are closer than a given delta
        radial_slice_indices = np.logical_and(np.abs(points_projection_x) < delta, points_projection_y > 0)
        # rotate the indicated points and only look at y and z coordinate value
        radial_slices[i] = np.dot(points[radial_slice_indices], rotations[i])[:, 1:]
    return np.asarray(radial_slices, dtype="object")


def mp_to_spline_3d(args: tuple) -> np.ndarray:
    """Multiprocessing wrapping function for esimating the splines

    This a wrapper function for using `to_spline_3d` to be able to use the
    faster imap function of the multiprocessing library

    Parameters
    ----------
    args : tuple
        first argument is a tuple of the form
            (radial_slice, rotation, sample_space, crop_radius)
        the other tuple argument is the kwargs for `to_spline_3d`

    Returns
    -------
        tuple[np.ndarray, np.ndarray]: same retuns as `to_spline_3d`
    """
    args, kwargs = args
    return to_spline_3d(*args, **kwargs)
