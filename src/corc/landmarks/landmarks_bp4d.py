"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
__all__ = ["LandmarksBP4D"]

import numpy as np

from .landmarks import Landmarks


class LandmarksBP4D(Landmarks):
    """Landmarks for the BP4D model"""

    def eye_left(self) -> np.ndarray:
        return self.landmarks[20:28]

    def eye_right(self) -> np.ndarray:
        return self.landmarks[28:36]

    def nose_tip(self) -> np.ndarray:
        return None

    def nose(self) -> np.ndarray:
        return self.landmarks[36:48]

    def mouth(self) -> np.ndarray:
        return self.landmarks[48:67]

    def eyebrow_left(self) -> np.ndarray:
        return self.landmarks[0:10]

    def eyebrow_right(self) -> np.ndarray:
        return self.landmarks[10:20]

    def jawline(self) -> np.ndarray:
        return self.landmarks[68:83]

    def jaw(self) -> np.ndarray:
        return None

    def jaw_lower(self) -> np.ndarray:
        return None
