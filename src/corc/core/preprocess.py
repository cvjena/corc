"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
from typing import Union
import numpy as np

from corc import landmarks as lm
from corc import utils


def __estimate_nosetip_location(
    points: np.ndarray, landmarks: lm.Landmarks
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
    landmarks: lm.Landmarks, 
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

    kwargs["perimeter_nose_tip"] = kwargs.get("perimeter_nose_tip", 0.15)
    kwargs["threshold_z_axis"] = kwargs.get("threshold_z_axis", 0.1)

    # scale everything
    scale = compute_scale_between_points(
        np.nanmean(landmarks.eye_left(), axis=0),
        np.nanmean(landmarks.eye_right(), axis=0),
    )
    points *= scale
    landmarks.rescale(scale)

    if landmarks.nose_tip() is None:
        nose_tip_1 = __estimate_nosetip_location(points, landmarks)
    else:
        nose_tip_1 = landmarks.nose_tip()

    rotation_matrix = utils.rotation_from_euler_angles(*landmarks.get_head_pose())

    # center the all the points around the nose point
    # rotate the point cloud with the rotation matrix
    points[:] -= nose_tip_1
    landmarks.translate(-nose_tip_1)

    points = np.dot(points, rotation_matrix)
    landmarks.rotate(rotation_matrix)

    z_side = np.sum(points[:, 2] > 0)
    # if the majority of points is on the positive side
    # mirror them to the other side
    if z_side > points.shape[0] // 4:
        points[:, 1] *= -1
        points[:, 2] *= -1
        landmarks.flip_z()

    # check for better nose tip location: we look around the current nose tip
    # and check if there is a point which is further into the z axis
    # this works as the points area already moved and rotated such that the nose
    # is the  center of the coordinate system
    vertex_distance = np.linalg.norm(points, axis=1)
    points_around_nosetip = points[vertex_distance < kwargs["perimeter_nose_tip"]]
    nose_tip_2 = points_around_nosetip[np.argmax(points_around_nosetip[:, 2])]
    points[:] -= nose_tip_2
    landmarks.translate(-nose_tip_2)

    # remove all points infront of the nosetip
    # iow only keep ones which have lower z value as 0.1
    mask_z = points[:, 2] < kwargs["threshold_z_axis"]
    points = points[mask_z]

    # calculate the distance (easy because centered around the nose)
    vertex_distance = np.linalg.norm(points, axis=1)

    crop_radius = kwargs.get("crop_radius_unit", None)
    if crop_radius is None:
        # we use the lower chin as the cropping boarder
        if landmarks.jaw() is not None and not np.isnan(np.sum(landmarks.jaw())):
            crop_radius = np.linalg.norm(np.abs(nose_tip_2 - landmarks.jaw()))
        elif landmarks.jaw_lower() is not None and not np.isnan(landmarks.jaw_lower()).all():
            crop_radius = np.linalg.norm(np.abs(nose_tip_2 - np.nanmean(landmarks.jaw_lower(), axis=0)))
        else:
            crop_radius = 1.5
    else:
        crop_radius *= scale

    return points[vertex_distance < crop_radius, :], float(crop_radius), (nose_tip_2, rotation_matrix, nose_tip_1, scale)
