__all__ = ["compute_volume", "make_mesh", "split_left_right", "compute_lower_curves_center"]

import matplotlib.path as mpath
import numpy as np
from scipy import spatial


def compute_angle_segment(
    C: np.ndarray, # center point
    I: np.ndarray, # intersection point  # noqa: E741
    E: np.ndarray, # end point
    R: float,      # radius of the circle
) -> tuple[np.ndarray, np.ndarray]:
    """Compute angle segment for a given point set.
    
    This function computes for a set of points how the angle segment
    would look like if the radius of circle is changable.
    The angle segment is defined by the center point, the intersection point
    and the end point. The radius is the radius of the circle that goes through
    the center point and the intersection point.
    
    We then compute where the new center point would be if the radius is changed.
    This is done using the Intersecting Chords Theorem.
    
    1.  Given C, E, and I, with R Compute where O is.
                    C
             h      | <|
         ----|----- |  |
         v        v |  |
      E . - - - - - I  |-r
         '        > |  |
          '   l -|  |  |
            ' ____>_| <| 
                    S
                    
    2.  Then use O and E to compute the new center point, if they would be on the same circle.
        Compute h (between E and I) and l (between O and E).
        This is done using the Intersecting Chords Theorem.
    
             , - ~ ~ ~ - ,
         , '               ' ,
       ,                       ,
      ,                         ,
     ,             CN            ,
     ,             |             ,
     ,             |             ,
      E . - - - -  I             ,
       ,           |            ,
         ,         |         , '
           ' - , _ S _ ,  '
    

    Args:
        center_point (_type_): Center point which is on the original circle where E currently resides.
        intersection_point (_type_): Intersection point is the point where C and E are perpendicular to each other.
        end_point (_type_): End point is the point defining the angle segment on the original circle.
        radius (_type_): Radius of the of the new circle.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two points to define the angle segment.
            First point is the outer point (O) of the angle segment.
            The second point is the new center point (CN) of the circle that goes through O and E.
    """
    
    direction = C - I
    direction = -(direction / np.linalg.norm(direction))
    # outer point is the point on the circle
    S = C + direction * R 
    # compute the new center point as these two points form a segment
    height = np.linalg.norm(I - S)
    length = np.linalg.norm(I - E)
    r = (height**2 + length**2) / (2*height)
    # compute the new center point
    center_point_n = S - r * direction
    return S, center_point_n


def slerp(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two direction vectors.

    Args:
        p0 (np.ndarray): First direction.
        p1 (np.ndarray): Second direction.
        t (float): Interpolation factor.

    Returns:
        np.ndarray: Direction vector on the sphere between p0 and p1.
    """
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega) / so * p1


def points_on_sphere(
    A: np.ndarray, 
    B: np.ndarray, 
    C: np.ndarray, 
    num_points: int
) -> np.ndarray:
    """Compute points on the sphere between two points with a given center point.

    Args:
        A (np.ndarray): A point on the sphere.
        B (np.ndarray): A point on the sphere.
        C (np.ndarray): Center point of the sphere.
        num_points (int): Number of points to compute.

    Returns:
        np.ndarray: Points on the sphere between A and B.
    """
    # compute the direction vectors
    A_dir = A - C
    B_dir = B - C
    points = np.empty((num_points, 3))
    for i, t in enumerate(np.linspace(0, 1, num_points)):
        points[i] = C + slerp(A_dir, B_dir, t)
    return points


def center(a, b, c):
    # compute the center of the triangle
    return (a + b + c) / 3


def rotation_matrix_from_vectors(vec1, vec2):
    # Normalize input vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Compute rotation axis and angle
    axis = np.cross(vec1, vec2)
    angle = np.arccos(np.dot(vec1, vec2))
    
    # Compute rotation matrix using axis-angle representation
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ]) # type: ignore
    identity = np.eye(3)
    R = identity + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return R


def rotate_plane_to_base_plane(normal, base_plane_normal):
    # Ensure input vectors are numpy arrays
    normal = np.array(normal)
    base_plane_normal = np.array(base_plane_normal)
    # Compute the rotation matrix
    R = rotation_matrix_from_vectors(normal, base_plane_normal)
    return R


def triangulate_face_side(curves, curves_underground, right:bool) -> tuple[np.ndarray, np.ndarray]:
    _, n_points, _ = curves.shape
    index_curves = np.arange(curves.reshape(-1, 3).shape[0])
    index_curves = index_curves.reshape(-1, n_points)
    
    index_underground = np.arange(curves_underground.reshape(-1, 3).shape[0])
    index_underground = index_underground.reshape(-1, n_points)
    index_underground += index_curves.max() + 1

    triangles = [] #np.zeros((faces_per_slice * total_slices , 3), dtype=np.int32)

    # always go pair by pair, with stride 1
    for p1, p2 in zip(index_curves, index_curves[1:]):
        # the first triangle is always idx 0 and seconds indices of the pairs
        #   
        #  p1[2] - p2[2]
        #   |    /   |
        #  p1[1] - p2[1] 
        #     \    /
        #        0
        triangles.append([0, p1[1], p2[1]])
        for i in range(1, len(p1) - 1):
            bl = p1[i]
            br = p2[i]
            tl = p1[i+1]
            tr = p2[i+1]
            triangles.append([bl, tl, br])
            triangles.append([tl, tr, br])

    for p1, p2 in zip(index_underground, index_underground[1:]):
        triangles.append([index_underground[0][-1], p2[-2], p1[-2]])
        for i in range(1, len(p1) - 2):
            bl = p1[i]
            br = p2[i]
            tl = p1[i+1]
            tr = p2[i+1]
            triangles.append([bl, tl, br])
            triangles.append([tl, tr, br])

    # add the connection between the curves and the underground
    for (p1, p2), (u1, u2) in zip(zip(index_curves, index_curves[1:]), zip(index_underground, index_underground[1:])):
        triangles.append([p1[-1], u1[1], p2[-1]])
        triangles.append([p2[-1], u1[1], u2[1]])


    outer_index = np.concatenate([np.flip(index_curves[-1, 1:]), index_curves[0], index_underground[0], np.flip(index_underground[-1, :-1])], axis=0)
    points = np.concatenate([curves.reshape(-1, 3), curves_underground.reshape(-1, 3)], axis=0)
    outer_points = points[outer_index]
    # compute the plane these outer_points are on
    # then rotate them such that the z coordinate is 0 for all points
    # then compute the delaunay triangulation inside the 2d plane
    # then rotate the points back to the original position

    # 1. make the outerpoints like A * x = B
    A = np.concatenate([outer_points[:, :2], np.ones((outer_points.shape[0], 1))], axis=1)
    B = outer_points[:, 2]
    x = np.linalg.lstsq(A, B, rcond=None)[0]

    # compute the normal of the plane
    normal = np.array([x[0], x[1], -1])
    normal /= np.linalg.norm(normal)

    target_normal = np.array([0, 0, 1])

    R = rotate_plane_to_base_plane(normal, target_normal)
    outer_points_rotated = outer_points @ R.T
    # print(outer_points_rotated[:10])

    points_2d = outer_points_rotated[:, :2]
    # compute the delaunay triangulation
    tri = spatial.Delaunay(points_2d)

    # check if each triangle is in the convex hull
    # if not, remove it
    path = mpath.Path(points_2d, closed=True)
    tri_centers = []
    for triangle in tri.simplices:
        circumcenter_ = center(points_2d[triangle[0]], points_2d[triangle[1]], points_2d[triangle[2]])
        tri_centers.append(circumcenter_)
    tri_centers = np.array(tri_centers)
    inside = path.contains_points(tri_centers)
    # Filter the triangles that are inside the perimeter
    inner_triangles = tri.simplices[inside]
    
    # now compute the original index
    for i in range(inner_triangles.shape[0]):
        for j in range(inner_triangles.shape[1]):
            inner_triangles[i, j] = outer_index[inner_triangles[i, j]]

    if not right:
        # the triangles are now in the correct order, but are inverted (clockwise instead of counter-clockwise) so we need to invert them
        for i in range(inner_triangles.shape[0]):
            inner_triangles[i] = np.flip(inner_triangles[i])

    triangles.extend(inner_triangles)
    triangles = np.array(triangles, dtype=np.int32)
    return points, triangles


### Public functions ###


def split_left_right(
    curves: np.ndarray,
    n_curves: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """This function splits the curves of the face into the left and right side.
    
    We assume that the curves are ordered from left to right. The middle curve
    is included in the left side and also in the right side. To later be able to
    compute the volume of the face, we need to split the curves into the left and
    right side.

    Args:
        curves (np.ndarray): The curves of the face
        n_curves (int, optional): Amount of curves. Defaults to 128.

    Returns:
        tuple[np.ndarray, np.ndarray]: The left and right side of the face.
        Order is left, right.
    """
    # reshape the curves, this might not be necessary, but it makes ensures that
    curves = curves.reshape(n_curves, -1, 3)

    # compute the index of curves left and right of the face
    idx_l = np.arange(n_curves//2, n_curves)
    idx_r = np.arange(0, n_curves//2+1) # +1 to include the middle curve
    idx_l = np.append(idx_l, [0]) # add the middle curve to the left side

    # get the left side of the face
    return curves[idx_l], curves[idx_r]


def compute_lower_curves_center(curves_l: np.ndarray, factor: float=1.0) -> tuple[np.ndarray, np.ndarray]:
    """This function computes the center and intersection point of the lower curves.
    
    This than can later be used to compute the lower bounds 

    Args:
        curves (np.ndarray): Upper curves of the face
        factor (float): Scaling factor of the lower curves radius

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    temp_l = curves_l[ 0]
    temp_r = curves_l[-1]

    center_point = temp_l[ 0]
    end_point_l  = temp_l[-1]
    end_point_r  = temp_r[-1]

    dist_l = np.linalg.norm(center_point - end_point_l)
    dist_r = np.linalg.norm(center_point - end_point_r)

    dir_l = center_point - end_point_l
    dir_r = center_point - end_point_r

    dir_l = - (dir_l / np.linalg.norm(dir_l))
    dir_r = - (dir_r / np.linalg.norm(dir_r))

    intersection = (end_point_l + end_point_r) / 2

    dir_c = center_point - intersection
    dir_c = -(dir_c / np.linalg.norm(dir_c))

    radius = (dist_l + dist_r) / 2
    radius *= factor
    segment_point, center_point_n = compute_angle_segment(center_point, intersection, end_point_l, float(radius))

    return segment_point, center_point_n

def make_mesh(
    curves: np.ndarray, 
    segment_point: np.ndarray, 
    center_point: np.ndarray, 
    right: bool=True
) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        curves (np.ndarray): _description_

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    curves_lower = np.zeros_like(curves)
    for i in range(curves.shape[0]):
        end_point = curves[i][-1]
        curves_lower[i] = points_on_sphere(end_point, segment_point, center_point, curves.shape[1])
    
    points, triangles = triangulate_face_side(curves, curves_lower, right=right)
    return points, triangles
    

def compute_volume(points, triangles):
    try: 
        import open3d as o3d
    except ImportError:
        print("Open3D not installed, cannot compute volume. Please install with `pip install corc[volume]`")
        return None, None
    
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(points), o3d.utility.Vector3iVector(triangles))
    if not mesh.is_watertight():
        print("Mesh is not watertight, cannot compute volume.")
        return mesh, None
    
    volume = mesh.get_volume()
    return mesh, volume