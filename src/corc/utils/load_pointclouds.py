"""
Copyright (c) Computer Vision Group - FSU Jena

Author: Tim Büchner
Email: tim.buechner@uni-jena.de
"""

__all__ = ["load_pointcloud"]
import os
import pathlib
import sys
from contextlib import contextmanager

import numpy as np
import open3d as o3d


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def load_pointcloud(path: pathlib.Path, subdivide:bool = True) -> np.ndarray:
    """Load pointcloud from a file

    Automatically detects the file type and loads the pointcloud.

    Parameters
    ----------
    path : pathlib.Path
        Path to the pointcloud file

    Returns
    -------
    np.ndarray
        Pointcloud as numpy array
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    path_s = str(path)
    
    geometry_type = o3d.io.read_file_geometry_type(path_s)
    mesh = None
    if geometry_type & o3d.io.CONTAINS_TRIANGLES:
        mesh = o3d.io.read_triangle_mesh(path_s, print_progress=True)
        if subdivide:
            with suppress_stdout():
                mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        return np.asarray(mesh.vertices)

    if mesh is None:
        cloud = None
        try:
            cloud = o3d.io.read_point_cloud(path_s)
            return np.asarray(cloud.points)
        except Exception:
            raise ValueError(f"Failed to load pointcloud from {path}")
    raise ValueError(f"Failed to load pointcloud from {path}")

# def load_pointcloud_obj(path: pathlib.Path) -> np.ndarray:
#     """Load pointcloud from obj file


#     Parameters
#     ----------
#     path : pathlib.Path
#         Path to the pointcloud file
# i
#     Returns
#     -------
#     np.ndarray
#         Pointcloud as numpy array
#     """
#     mesh: trimesh.Trimesh = trimesh.load_mesh(path.as_posix())
#     mesh = mesh.subdivide()
#     return mesh.vertices


# def load_pointcloud_ply(path: pathlib.Path) -> np.ndarray:
#     """Load pointcloud from ply file


#     Parameters
#     ----------
#     path : pathlib.Path
#         Path to the pointcloud file

#     Returns
#     -------
#     np.ndarray
#         Pointcloud as numpy array
#     """
#     ply_file = plyfile.PlyData.read(path.as_posix())
#     points = (
#         np.vstack(
#             (
#                 ply_file["vertex"].data["x"],
#                 ply_file["vertex"].data["y"],
#                 ply_file["vertex"].data["z"],
#             )
#         )
#         .reshape(3, -1)
#         .T
#     )
#     return points


# def load_pointcloud_wrl(path: pathlib.Path) -> np.ndarray:
#     """Load pointcloud from wrl file


#     Parameters
#     ----------
#     path : pathlib.Path
#         Path to the pointcloud file

#     Returns
#     -------
#     np.ndarray
#         Pointcloud as numpy array
#     """
#     # https://scicomp.stackexchange.com/questions/11528/wrl-and-vrml-to-matplotlib-numpy
#     coords = []
#     faces = []
#     mode = 0
#     with open(path, "r", encoding="utf-8") as wrl_file:
#         for line in wrl_file:
#             line = line.strip()
#             if "coord Coordinate" in line:
#                 mode = 1
#                 continue
#             if "coordIndex" in line:
#                 mode = 2
#                 continue
#             if "texCoord" in line:
#                 break

#             if mode == 0:
#                 continue

#             if mode == 1:
#                 if "point" in line:
#                     continue
#                 line = line.replace(",", "")
#                 spl = line.split(" ")
#                 # coord = re.findall("-?\d{1,3}\.\d{0,6}", line)
#                 if len(spl) == 3:
#                     coords.append(spl)
#             if mode == 2:
#                 face = re.findall(r"[^-]\d+", line)
#                 if len(face) == 3:
#                     faces.append(face)

#     mesh = trimesh.base.Trimesh(
#         np.asarray(coords, dtype=np.float32), np.asarray(faces, dtype=np.int32)
#     )
#     mesh = mesh.subdivide()
#     return mesh.vertices
