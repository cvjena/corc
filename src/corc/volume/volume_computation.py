__all__ = ["compute_volume", "make_mesh", "split_left_right", "compute_lower_curves_center"]

import numpy as np


def compute_angle_segment(
    center_point,
    intersection_point,
    end_point,
    distance,
) -> tuple[np.ndarray, np.ndarray]:
    direction = center_point - intersection_point
    direction = -(direction / np.linalg.norm(direction))
    # outer point is the point on the circle
    segment_point = center_point + direction * distance 
    # compute the new center point as these two points form a segment
    height = np.linalg.norm(intersection_point - segment_point)
    length = np.linalg.norm(intersection_point - end_point)
    r = (height**2 + length**2) / (2*height)
    # compute the new center point
    center_point_n = segment_point - r * direction
    return segment_point, center_point_n


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


def compute_lower_curves_center(curves_l: np.ndarray, factor: float=1.0) -> tuple[np.ndarray, np.ndarray]:
    """This function computes the center and intersection point of the lower curves.
    
    This than can later be used to compute the lower bounds 

    Args:
        curves (np.ndarray): Upper curves of the face
        factor (float): Scaling factor of the lower curves radius

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    temp_l = curves_l[ 0]
    temp_r = curves_l[-1]

    center_point = temp_l[ 0]
    end_point_l  = temp_l[-1]
    end_point_r  = temp_r[-1]

    dist_l = np.linalg.norm(center_point - end_point_l)
    dist_r = np.linalg.norm(center_point - end_point_r)

    dir_l = center_point - end_point_l
    dir_r = center_point - end_point_r

    dir_l = - (dir_l / np.linalg.norm(dir_l))
    dir_r = - (dir_r / np.linalg.norm(dir_r))

    intersection = (end_point_l + end_point_r) / 2

    dir_c = center_point - intersection
    dir_c = -(dir_c / np.linalg.norm(dir_c))

    radius = (dist_l + dist_r) / 2
    radius *= factor
    segment_point, center_point_n = compute_angle_segment(center_point, intersection, end_point_l, radius)

    return segment_point, center_point_n


def make_mesh():
    pass


def compute_volume():
    pass