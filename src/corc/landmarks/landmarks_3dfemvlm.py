"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""
__all__ = ["Landmarks3DFEMVLM"]

import numpy as np

from .landmarks import Landmarks


class Landmarks3DFEMVLM(Landmarks):
    """Landmarks for the 3DFEMVLM model"""

    def eye_left(self) -> np.ndarray:
        return self.landmarks[8:16]

    def eye_right(self) -> np.ndarray:
        return self.landmarks[0:8]

    def nose_tip(self) -> np.ndarray:
        return self.landmarks[-1]

    def nose(self) -> np.ndarray:
        return self.landmarks[36:48]

    def mouth(self) -> np.ndarray:
        return self.landmarks[48:68]

    def eyebrow_left(self) -> np.ndarray:
        return self.landmarks[16:26]

    def eyebrow_right(self) -> np.ndarray:
        return self.landmarks[26:36]

    def jawline(self) -> np.ndarray:
        return self.landmarks[69:83]

    def jaw(self) -> np.ndarray:
        return self.landmarks[75:76]

    def jaw_lower(self) -> np.ndarray:
        return self.landmarks[74:77]
