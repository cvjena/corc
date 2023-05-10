__all__ = ["compute_volume", "make_mesh", "split_left_right"]

import numpy as np


def split_left_right(
    curves: np.ndarray,
    n_curves: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """This function splits the curves of the face into the left and right side.
    
    We assume that the curves are ordered from left to right. The middle curve
    is included in the left side and also in the right side. To later be able to
    compute the volume of the face, we need to split the curves into the left and
    right side.

    Args:
        curves (np.ndarray): The curves of the face
        n_curves (int, optional): Amount of curves. Defaults to 128.

    Returns:
        tuple[np.ndarray, np.ndarray]: The left and right side of the face.
        Order is left, right.
    """
    # reshape the curves, this might not be necessary, but it makes ensures that
    curves = curves.reshape(n_curves, -1, 3)

    # compute the index of curves left and right of the face
    idx_l = np.arange(n_curves//2, n_curves)
    idx_r = np.arange(0, n_curves//2+1) # +1 to include the middle curve
    idx_l = np.append(idx_l, [0]) # add the middle curve to the left side

    # get the left side of the face
    return curves[idx_l], curves[idx_r]


def make_mesh():
    pass


def compute_volume():
    pass