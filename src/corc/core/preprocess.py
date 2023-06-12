"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
from typing import Union
import numpy as np

from corc import landmarks as lms
from corc import utils

def __estimate_nosetip_location(
    points: np.ndarray, landmarks: lms.Landmarks
) -> np.ndarray:
    """Estimate the nose tip location from given nose landmarks

    This script estimates the nose tip location based on some given
    landmarks, first we average the the location and crop all points
    around this center point. from the remaining points we estimate
    the one which is most far in one direction

    Parameters
    ----------
    points : np.ndarray
        the vertexes of the point cloud
    landmarks : corc.landmarks.Landmarks
        the landmark information about the point cloud/mesh

    Returns
    -------
    np.ndarray
        the estimated nose tip location
    """

    # calculate the center of the nose landmarks
    nose_loction_avg = np.nanmean(landmarks.nose(), axis=0)

    # normalize around average nosePoint, then remove all points which are too far
    # away from it
    points[:] -= nose_loction_avg
    landmarks.translate(-nose_loction_avg)

    # compute the distances to the center point
    vertex_distance = np.linalg.norm(points, axis=1)

    # get the maximum distance between all landmarks and use that to
    # crop around the center
    # NOTE as the face should already been rotated, dont use a sphere, instead
    # use a cylinder which finds the best fitting point
    points = points[vertex_distance < 0.6]
    # point with highest z-coordinate
    nosetip_loction = np.argmax(points[:, 2])
    return points[nosetip_loction]


def compute_scale_between_points(p_a: np.ndarray, p_b: np.ndarray) -> float:
    """Get the scale of two points

    Use the distance between two points as a scale.
    After that the distance between these two points is 1

    Parameters
    ----------
    p_a : np.ndarray
        the first point
    p_b : np.ndarray
        the second point

    Returns
    -------
    float
        the scale of the two points
    """
    if np.all(p_a == p_b):
        return 0
    return float(1 / np.linalg.norm(p_a - p_b))


def preprocess_point_cloud(
    points: np.ndarray, 
    lm: lms.Landmarks, 
    **kwargs
) -> tuple[np.ndarray, float, tuple[Union[np.ndarray, float], ...]]:
    """This function preprocess the head point cloud.

    These steps include:
    1. estimate head position
    2. normalize head location by nose position
    3. normalize head rotation by estimated head position
    4. remove points which are too far away from the nose

    Parameters
    ----------
    points: np.ndarray
        the vertexes of the point cloud
    landmarks: corc.landmarks.Landmarks
        the landmark information about the point cloud/mesh
    kwargs:
        additional arguments to pass to the algorithm
        perimeter_nose_tip: float
            the perimeter of the nose tip estimation. Defaults to 0.015
        threshold_z_axis: float
            the threshold for the z-axis. Defaults to 0.1

    Returns
    -------
    tuple[np.ndarray, float, tuple[np.ndarray, float, np.ndarray, np.ndarray]]
        the preprocessed point cloud and the radius of the head
        the tuple contains the following information:
            1. Rotation matrix
            2. Scale
            3. Translation vector 1
            4. Translation vector 2
    """
    nose_tip = lm.nose_tip().copy() if lm.nose_tip() is not None else __estimate_nosetip_location(points, lm)
    rotation_matrix = utils.rotation_from_euler_angles(*lm.get_head_pose())

    points[:] -= nose_tip
    points = np.dot(points, rotation_matrix)

    vertex_distance = np.linalg.norm(points, axis=1)
    crop_radius = kwargs.get("crop_radius_unit", None)
    if crop_radius is None:
        # we use the lower chin as the cropping boarder
        if lm.jaw() is not None and not np.isnan(np.sum(lm.jaw())):
            crop_radius = np.linalg.norm(np.abs(lm.nose_tip() - lm.jaw()))
        elif lm.jaw_lower() is not None and not np.isnan(lm.jaw_lower()).all():
            crop_radius = np.linalg.norm(np.abs(lm.nose_tip() - np.nanmean(lm.jaw_lower(), axis=0)))
        else:
            crop_radius = 85 #mm

    return points[vertex_distance < crop_radius, :], float(crop_radius), (rotation_matrix, nose_tip)
