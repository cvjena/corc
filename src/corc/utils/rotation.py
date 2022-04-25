"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""

__all__ = [
    "rotation_from_euler_angles",
    "rotx",
    "roty",
    "rotz",
    "rotx_v",
    "roty_v",
    "rotz_v",
    "AXIS_X",
    "AXIS_Y",
    "AXIS_Z",
]
import numpy as np

# Euclidean default axis
AXIS_X: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float32)
AXIS_Y: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32)
AXIS_Z: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def rotation_from_euler_angles(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Compute the rotation matrix for given angles (arc)

    Parameters
    ----------
    yaw : float
        Rotation around y-axis (degrees)
    pitch : float
        Rotation around x-axis (degrees)
    roll : float
        Rotation around z-axis (degrees)

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    roll_matrix = np.array(
        [
            [np.cos(roll), -np.sin(roll), 0.0],
            [np.sin(roll), np.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    yaw_matrix = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )

    pitch_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float32,
    )

    return yaw_matrix @ pitch_matrix @ roll_matrix


def rotz(angle: float) -> np.ndarray:
    """Rotation matrix for angle (degrees) around z-axis

    Parameters
    ----------
    angle : float
        Rotation around z-axis (degrees)

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    angle = angle * np.pi / 180
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def roty(angle: float) -> np.ndarray:
    """Rotation matrix for angle (degrees) around z-axis

    Parameters
    ----------
    angle : float
        Rotation around y-axis (degrees)

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    angle = angle * np.pi / 180
    return np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float32,
    )


def rotx(angle: float) -> np.ndarray:
    """Rotation matrix for angle (degrees) around z-axis

    Parameters
    ----------
    angle : float
        Rotation around x-axis (degrees)

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    angle = angle * np.pi / 180
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float32,
    )


# the rotations functions but also as vectorized numpy functions
rotz_v = np.frompyfunc(rotz, nin=1, nout=1)
roty_v = np.frompyfunc(roty, nin=1, nout=1)
rotx_v = np.frompyfunc(rotx, nin=1, nout=1)
