"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
__all__ = ["Landmarks68"]

import numpy as np

from .landmarks import Landmarks


class Landmarks68(Landmarks):
    """Landmarks for the 68-point model"""

    def eye_left(self) -> np.ndarray:
        return self.landmarks[42:48]

    def eye_right(self) -> np.ndarray:
        return self.landmarks[36:42]

    def nose_tip(self) -> np.ndarray:
        return self.landmarks[30:31]

    def nose(self) -> np.ndarray:
        return self.landmarks[27:36]

    def mouth(self) -> np.ndarray:
        return self.landmarks[48:67]

    def eyebrow_left(self) -> np.ndarray:
        return self.landmarks[17:22]

    def eyebrow_right(self) -> np.ndarray:
        return self.landmarks[22:27]

    def jawline(self) -> np.ndarray:
        return self.landmarks[0:17]

    def jaw(self) -> np.ndarray:
        return self.landmarks[8:9]

    def jaw_lower(self) -> np.ndarray:
        return self.landmarks[7:10]
