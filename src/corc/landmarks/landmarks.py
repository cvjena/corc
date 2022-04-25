"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
__all__ = ["Landmarks"]

import abc

import numpy as np
from sklearn.decomposition import PCA


class Landmarks(abc.ABC):
    """Abstract landmarks model class"""

    def __init__(self, landmarks: np.ndarray):
        """Create a new landmarks model from a given np.array

        Parameters
        ----------
        landmarks : np.ndarray
            The landmarks to use for this model
        """
        self.landmarks = landmarks
        if self.landmarks.shape[1] == 3:
            self.head_pose = self.estimate_head_pose()
        else:
            self.head_pose = [0.0, 0.0, 0.0]

    def estimate_head_pose(self) -> tuple[float, float, float]:
        """Estimate the pose (orientation) of a 3D head

        This is the function implemented by Thümmel which was translated from matlab in:
        [3] Dmytro Derkach, et al. "Head Pose Estimation Based on 3-D Facial Landmarks
        Localization and Regression". FG'2017
        The code was converted from matlab and is taken from [3] and explained in Fig. 2
        in [1] landmarks = 2 x n matrix of landmarks coordinates

        However, this function was adapted to be used with different landmark systems.

        Returns
        -------
        tuple[float, float, float]
            The estimated head pose (orientation)
            yaw, pitch, roll
        """

        # consider only the inner/outer eye landmark indices for head pose roll angle
        # estimation calculate the direction of a line through both mean eye positions
        # or the remaining outer/inner eye landmarks if some of them are nan
        # use nan<funcname> to still get some values even though some values are nan
        # this should be a 3x2 Matrix
        eye_left = self.eye_left()
        eye_right = self.eye_right()
        mouth = self.mouth()

        line_ab = np.vstack(
            (np.nanmean(eye_left, axis=0), np.nanmean(eye_right, axis=0))
        )

        # calculate only in the XY plane, ignore Z dimension
        f_ab = line_ab[:, :2]
        # solve the linear equation y=mx + n for m
        f_ab[0, :] -= f_ab[1, :]
        roll = (
            np.arctan(f_ab[0, 1] / f_ab[0, 0]) if np.all(np.isfinite(f_ab)) else np.nan
        )

        # consider only the mouth and inner/outer eye landmark indices for head pose yaw
        # and pitch angle estimation
        plane_eyes_mouth = np.vstack((eye_left, eye_right, mouth))
        # keep only finite values from the points (remove nans)
        plane_eyes_mouth = plane_eyes_mouth[np.isfinite(plane_eyes_mouth[:, 0])]

        # calculate the normal vector of this plane
        # first normalize the plane
        rot_matrix = plane_eyes_mouth - np.mean(plane_eyes_mouth, axis=0)
        # estimate the directions of the first two eigenvectors
        # eg. fitting a plane into a subset of points
        ev_1, ev_2 = PCA(n_components=2).fit(rot_matrix).components_
        # calc the the normal vector from the eigen vectors
        normal_vector = np.cross(ev_1, ev_2)
        # correct the normal vector direction
        if normal_vector[2] < 0:
            normal_vector *= -1

        # obtain the angles from the normal vector
        yaw = np.arctan(normal_vector[0])
        pitch = -np.arctan(normal_vector[1])
        return yaw, pitch, roll

    def rescale(self, scale_factor: float) -> None:
        """Rescale the landmarks by a given factor

        Parameters
        ----------
        scale_factor : float
            The factor to scale the landmarks by
        """
        self.landmarks *= scale_factor

    def translate(self, vector: np.ndarray) -> None:
        """Translate the landmarks by a given vector

        Parameters
        ----------
        vector : np.ndarray
            The vector to translate the landmarks by
        """
        self.landmarks += vector

    def flip_z(self) -> None:
        """Flip the landmarks along the z axis."""
        self.landmarks[:, 1] *= -1
        self.landmarks[:, 2] *= -1

    def rotate(self, rotation: np.ndarray) -> None:
        """Rotate the landmarks by a given rotation matrix

        Parameters
        ----------
        rotation : np.ndarray
            The rotation matrix to rotate the landmarks by
        """
        self.landmarks = np.dot(self.landmarks, rotation)

    def get_head_pose(self) -> tuple[float, float, float]:
        """Get the estimated head pose

        Returns
        -------
        tuple[float, float, float]
            The estimated head pose (orientation)
            yaw, pitch, roll
        """
        return self.head_pose

    @abc.abstractmethod
    def eye_left(self) -> np.ndarray:
        """Get the left eye landmark indices"""

    @abc.abstractmethod
    def eye_right(self) -> np.ndarray:
        """Get the right eye landmark indices"""

    @abc.abstractmethod
    def nose_tip(self) -> np.ndarray:
        """Get the nose tip landmark indices"""

    @abc.abstractmethod
    def nose(self) -> np.ndarray:
        """Get the nose landmark indices"""

    @abc.abstractmethod
    def mouth(self) -> np.ndarray:
        """Get the mouth landmark indices"""

    @abc.abstractmethod
    def eyebrow_left(self) -> np.ndarray:
        """Get the left eyebrow landmark indices"""

    @abc.abstractmethod
    def eyebrow_right(self) -> np.ndarray:
        """Get the right eyebrow landmark indices"""

    @abc.abstractmethod
    def jawline(self) -> np.ndarray:
        """Get the jawline landmark indices"""

    @abc.abstractmethod
    def jaw(self) -> np.ndarray:
        """Get the jaw landmark indices"""

    @abc.abstractmethod
    def jaw_lower(self) -> np.ndarray:
        """Get the jaw lower landmark indices"""
