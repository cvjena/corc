"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""

import math

import igraph
import numpy as np
from scipy import interpolate, ndimage
from scipy.spatial import distance, qhull


def to_spline_3d(
    radial_slice: np.ndarray,
    rotation: np.ndarray,
    sample_space: np.ndarray,
    crop_radius: float,
    **kwargs
) -> np.ndarray:
    """Estimate the splines of a 3d slice point cloud

    Parameters
    ----------
    radial_slice : np.ndarray
        radial slice which contains the points (unordered is ok)
    rotation : np.ndarray
        rotation matrix of the original extraction position
    sample_space : np.ndarray
        sample space from which the spline points will be sampled
    crop_radius : float
        radius of the crop circle
    kwargs : dict, optional
        keyword arguments, see below
        fix_end_point : bool, optional
            if True, the end point of the radial slice will be estimated

    Returns
    -------
    np.ndarray
        spline points describing the 3d slice
    """

    kwargs["fix_end_point"] = kwargs.get("fix_end_point", True)

    # estimate the best fitting pline for the given radial slice points
    # for faster memory allocation the slice contains more points than neeeded
    # thus we disard these values and reshape everything
    radial_slice = np.insert(radial_slice, 0, [[0.0, 0.0]], axis=0)

    if kwargs["fix_end_point"]:
        radial_slice = fix_end_point(radial_slice, crop_radius)

    p_spline_2d = spline_2d(radial_slice, sample_space, **kwargs)

    # add the x-dimension back again to the points
    p_spline_3d: np.ndarray = np.hstack(
        (np.zeros((p_spline_2d.shape[0], 1)), p_spline_2d)
    ).T
    # rotate the points to the desired location in the face
    return np.dot(rotation, p_spline_3d).T


def fix_end_point(radial_slice: np.ndarray, crop_radius: float) -> np.ndarray:
    """Estimate the end point of a radial slice

    The end point is the point which is furthest to the center of the coordinate system
    and is at the end of the spline.

    Parameters
    ----------
    radial_slice : np.ndarray
        radial slice which contains the points (unordered is ok)
    crop_radius : float
        radius of the crop circle

    Returns
    -------
    np.ndarray
        radial slice with the end point added
    """
    current_end_point = radial_slice[np.argmax(radial_slice[:, 0])]
    scale_factor = crop_radius / math.sqrt(
        current_end_point[0] ** 2 + current_end_point[1] ** 2
    )
    strechted_end_point = current_end_point * scale_factor
    return np.append(radial_slice, [strechted_end_point], axis=0)


def spline_2d(
    radial_slice: np.ndarray, sample_space: np.ndarray, **kwargs
) -> np.ndarray:
    """Find the best 2d spline for a given radial slice

    Compute the best fitting spline for a given radial slice.

    Parameters
    ----------
    radial_slice : np.ndarray
        radial slice which contains the points (unordered is ok)
    sample_space : np.ndarray
        sample space from which the spline points will be sampled
    kwargs : dict, optional
        keyword arguments, see below
        graph_fit : bool, optional
            if True, the graph_fit will be used instead of the distance_fit
            this is part of the newly developed spline fitting algorithm

    Returns
    -------
    np.ndarray
        spline points describing the 2d slice
    """
    if kwargs.get("graph_fit", True):
        return graph_fit(radial_slice, sample_space)
    return distance_fit(radial_slice, sample_space)


def graph_fit(radial_slice: np.ndarray, sample_space: np.ndarray) -> np.ndarray:
    """Fit a spline to the radial slice using the graph_fit algorithm

    This a new spline fitting algorithm which is based on the graph_fit algorithm.

    Parameters
    ----------
    radial_slice : np.ndarray
        radial slice which contains the points (unordered is ok)
    sample_space : np.ndarray
        sample space from which the spline points will be sampled

    Returns
    -------
    np.ndarray
        spline points describing the 2d slice
    """
    tess = qhull.Delaunay(radial_slice, incremental=False)
    idx0 = 0
    idx1 = len(radial_slice) - 1

    tess_amt = len(tess.simplices)
    edge_lst = np.zeros((tess_amt * 3, 2), dtype=np.int32)
    for i in range(tess_amt):
        # fmt: off
        edge_lst[i * 3    ] = [tess.simplices[i, 0], tess.simplices[i, 1]] # noqa
        edge_lst[i * 3 + 1] = [tess.simplices[i, 1], tess.simplices[i, 2]]
        edge_lst[i * 3 + 2] = [tess.simplices[i, 2], tess.simplices[i, 0]]
        # fmt: on

    # compute shortest weighted path
    edge_lengths = np.linalg.norm(
        tess.points[edge_lst[:, 0], :] - tess.points[edge_lst[:, 1], :], axis=1
    )
    graph = igraph.Graph(n=len(radial_slice), edges=edge_lst)
    graph = graph.spanning_tree(weights=edge_lengths)
    path_s = graph.get_shortest_paths(v=idx0, to=idx1)[0]

    curve = tess.points[path_s]
    dist = np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
    dist = np.hstack(([0], dist)) / dist[-1]
    splines = interpolate.interp1d(dist, curve, kind="linear", axis=0, copy=False)
    values = splines(sample_space)
    values[:, 0] = ndimage.gaussian_filter1d(values[:, 0], 1, axis=0)
    values[:, 1] = ndimage.gaussian_filter1d(values[:, 1], 1, axis=0)
    return values


def distance_fit(radial_slice: np.ndarray, sample_space: np.ndarray) -> np.ndarray:
    """Fit a spline to the radial slice

    Compute the best fitting spline for a given radial slice based on the points
    ordering.

    Parameters
    ----------
    radial_slice : np.ndarray
        radial slice which contains the points (unordered is ok)
    sample_space : np.ndarray
        sample space from which the spline points will be sampled

    Returns
    -------
    np.ndarray
        spline points describing the 2d slice
    """
    fitted_by_x = spline_2d_simple_fit(argsort_by_xdist(radial_slice), sample_space)
    fitted_by_o = spline_2d_simple_fit(argsort_by_odist(radial_slice), sample_space)

    # check which computed spline is closest to the original radial points
    # with the Hausdorff distance
    hd_xdist = min(
        distance.directed_hausdorff(u=fitted_by_x, v=radial_slice),
        distance.directed_hausdorff(u=radial_slice, v=fitted_by_x),
    )
    hd_odist = min(
        distance.directed_hausdorff(u=fitted_by_o, v=radial_slice),
        distance.directed_hausdorff(u=radial_slice, v=fitted_by_o),
    )

    return fitted_by_x if hd_xdist[0] < hd_odist[0] else fitted_by_o


def argsort_by_xdist(values: np.ndarray) -> np.ndarray:
    """Sort an array by the x-coordinate

    Parameters
    ----------
    values : np.ndarray
        array to be sorted

    Returns
    -------
    np.ndarray
        sorted array
    """
    return values[np.argsort(values[:, 0])]


def argsort_by_odist(values: np.ndarray):
    """Sort an array by the distance to the center of coordinate system

    Parameters
    ----------
    values : np.ndarray
        array to be sorted

    Returns
    -------
    np.ndarray
        sorted array
    """
    dist = np.sqrt(values[:, 0] * values[:, 0] + values[:, 1] * values[:, 1])
    return values[np.argsort(dist)]


def spline_2d_simple_fit(
    radical_slice: np.ndarray, sample_space: np.ndarray
) -> np.ndarray:
    """Fit a 2d spline into a 2d curve

    Parameters
    ----------
    radical_slice : np.ndarray
        array of points which describe the 2d slice
    sample_space : np.ndarray
        sample space from which the spline points will be sampled

    Returns
    -------
    np.ndarray
        spline points describing the 2d slice
    """

    last_point = radical_slice[-1]

    radical_slice[:, 0] = ndimage.gaussian_filter1d(radical_slice[:, 0], 2, axis=0)
    radical_slice[:, 1] = ndimage.gaussian_filter1d(radical_slice[:, 1], 2, axis=0)
    radical_slice[0] = 0
    radical_slice[-1] = last_point

    dist = np.cumsum(np.linalg.norm(np.diff(radical_slice, axis=0), axis=1))
    dist = np.hstack(([0], dist)) / dist[-1]
    splines = interpolate.interp1d(
        dist, radical_slice, kind="cubic", axis=0, copy=False
    )
    values = splines(sample_space)
    values[:, 1] = ndimage.gaussian_filter1d(values[:, 1], 1, axis=0)
    return values[:128]
