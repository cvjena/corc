"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""

__all__ = ["lap_calc", "humphrey"]

from typing import Any, Optional

import numpy as np
from scipy import sparse


def __neigbours_for_face_mesh(
    n_curves: int = 128, n_points: int = 128
) -> list[list[int]]:
    """Calculates the neighbours for the smooting algorithm.

    Parameters
    ----------
    n_curves : int, optional
        Number of curves/slices to extract. Defaults to 128.
    n_points : int, optional
        Number of points to use for the spline. Defaults to 128.

    Returns
    -------
    list[int]
        The neighbours for the smooting algorithm for each index

    """

    total = n_curves * n_points
    neighbors: list[Any] = [None] * total

    neighbors[0] = np.arange(start=1, stop=total, step=n_points)

    for idx in range(1, total):
        below = idx - 1
        above = idx + 1
        left = (idx + n_points) % total
        right = (idx - n_points) % total
        above_left = left + 1
        above_right = right + 1
        below_left = left - 1
        below_right = right - 1

        # center points
        if idx % n_points == 0:
            neighbors[idx] = [0]
        # first point
        elif idx % n_points == 1:
            neighbors[idx] = [0, above, left, right, above_left, above_right]
        # outer point
        elif idx % n_points == (n_points - 1):
            neighbors[idx] = [below, left, right, below_left, below_right]
        # every other point in the middle
        else:
            neighbors[idx] = [
                below,
                above,
                left,
                right,
                above_left,
                above_right,
                below_left,
                below_right,
            ]
    return neighbors


def lap_calc(
    vertexes: np.ndarray,
    neighbors: list[list[int]],
    fixed_points: Optional[list[int]] = None,
) -> sparse.csr_matrix:
    """An implementation of the Laplacian smoothing algorithm.
    based on
    https://github.com/mikedh/trimesh/blob/master/trimesh/smoothing.py

    Parameters
    ----------
    vertexes : np.ndarray
        The vertexes of the mesh
    neighbors : list[list[int]]
        The neighbours for the smooting algorithm for each index
    fixed_points : list[int], optional
        The fixed points for the smooting algorithm. Defaults to None.

    Returns
    -------
    sparse.csr_matrix
        The Laplacian matrix
    """
    fixed_points = fixed_points or []

    for idx in fixed_points:
        neighbors[idx] = [idx]

    col = np.concatenate(neighbors)
    row = np.concatenate([[i] * len(n) for i, n in enumerate(neighbors)])

    data = np.concatenate([[1.0 / len(n)] * len(n) for n in neighbors])

    matrix = sparse.coo_matrix((data, (row, col)), shape=[len(vertexes)] * 2)
    return matrix


def humphrey(
    vertexes: np.ndarray,
    alpha: float = 0.1,
    beta: float = 0.5,
    iterations: int = 5,
    n_curves: int = 128,
    n_points: int = 128,
    fixed_points: Optional[list[int]] = None,
):
    """Humphrey's smoothing algorithm only on the z-Axis.

    Same as the original implementation, but only on the z-Axis.

    Parameters
    ----------
    vertexes : np.ndarray
        The vertexes of the mesh
    alpha : float, optional
        The alpha value for the smoothing algorithm. Defaults to 0.1.
    beta : float, optional
        The beta value for the smoothing algorithm. Defaults to 0.5.
    iterations : int, optional
        The number of iterations for the smoothing algorithm. Defaults to 5.
    n_curves : int, optional
        Number of curves/slices to extract. Defaults to 128.
    n_points : int, optional
        Number of points to use for the spline. Defaults to 128.
    fixed_points : list[int], optional
        The fixed points for the smoothing algorithm. Defaults to None.

    Returns
    -------
    np.ndarray
        The smoothed vertexes of the mesh in the z-Axis
    """
    lap_op = lap_calc(
        vertexes,
        __neigbours_for_face_mesh(n_curves, n_points),
        fixed_points=fixed_points,
    )

    vertices = vertexes.copy()
    original = vertexes.copy()

    for _ in range(iterations):
        vert_q = vertices.copy()
        vertices = lap_op.dot(vertices)
        vert_b = vertices - (alpha * original + (1.0 - alpha) * vert_q)
        vertices -= beta * vert_b + (1.0 - beta) * lap_op.dot(vert_b)

    return vertices
