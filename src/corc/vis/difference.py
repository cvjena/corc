__all__ = ["volume_diff_area_map", "volume_diff_heat_map"]

import open3d as o3d
import numpy as np
import palettable
from matplotlib.colors import ListedColormap

cmap = ListedColormap(palettable.scientific.sequential.Imola_20.mpl_colors)

def volume_diff_area_map(
    distances: np.ndarray,
    points_l: np.ndarray,
    points_r: np.ndarray,
    max_distance_mm: float = 10.0,
    line_width: int = 10,
) -> dict:
    colors = cmap(distances / max_distance_mm)[:, :3]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.concatenate([points_l, points_r]))
    line_set.lines = o3d.utility.Vector2iVector(np.array([[i, i + points_l.shape[0]] for i in range(points_l.shape[0])]))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "unlitLine"
    mat.line_width = line_width

    return {
        "name": "Volume Difference",
        "geometry": line_set,
        "material": mat,
    }

def volume_diff_heat_map(
    distances: np.ndarray,
    points: np.ndarray,
    n_points: int = 128,
    max_distance_mm: float = 10.0,
) -> o3d.geometry.TriangleMesh:
    assert points.ndim == 2

    colors = cmap(distances / max_distance_mm)[:, :3]
    mesh_points = np.asarray(points).reshape(-1, n_points, 3)

    index_curves = np.arange(points.shape[0])
    index_curves = index_curves.reshape(-1, n_points)
    triangles = []
    for p1, p2 in zip(index_curves, index_curves[1:]):
        triangles.append([0, p1[1], p2[1]])
        for i in range(1, len(p1) - 1):
            bl = p1[i]
            br = p2[i]
            tl = p1[i+1]
            tr = p2[i+1]
            triangles.append([bl, tl, br])
            triangles.append([tl, tr, br])

    triangles = np.array(triangles, dtype=np.int32)

    mesh_diff = o3d.geometry.TriangleMesh()
    mesh_diff.vertices = o3d.utility.Vector3dVector(mesh_points.reshape(-1, 3))
    mesh_diff.triangles = o3d.utility.Vector3iVector(triangles)
    mesh_diff.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh_diff.compute_vertex_normals()
    
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.base_color = (1.0, 1.0, 1.0, 0.5)
    
    return {
        "name" : "Difference Heatmap",
        "geometry" : mesh_diff,
        "mat" : mat,
    }