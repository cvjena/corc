__all__ = ["mirror_mesh"]


def mirror_mesh() -> None:
    """
    This function mirrors a original face surface mesh along the 
    vertical face axis.
    This function is required if the patient has a facial palsy on the
    right side of the face.
    """
    pass
#   if mesh is not None:
#             verts = np.asarray(mesh.vertices)

#             verts -= landmarks.nose_tip() 
#             verts *= transforms[-1]
#             verts = verts @ transforms[-3]

#             verts *= np.array([-1, 1, 1])

#             verts = verts @ transforms[-3].T
#             verts /= transforms[-1]
#             verts += landmarks.nose_tip() 
            
#             mesh.vertices = o3d.utility.Vector3dVector(verts)