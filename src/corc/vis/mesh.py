__all__ = ["mirror_mesh"]
import numpy as np
import open3d as o3d

def mirror_mesh(
    mesh: o3d.geometry.TriangleMesh, 
    translation: np.ndarray, 
    rotation: np.ndarray, 
    inverse: bool=True
) -> None:
    """
    This function mirrors a original face surface mesh along the 
    vertical face axis.
    This function is required if the patient has a facial palsy on the
    right side of the face.
    """

    verts = np.asarray(mesh.vertices)

    verts = verts - translation
    verts = verts @ rotation

    verts *= np.array([-1, 1, 1]) # mirror along the vertical face axis (x-axis)

    if inverse:
        verts = verts @ rotation.T
        verts = verts + translation

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    # change the triangle winding order
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])