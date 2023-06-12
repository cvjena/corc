__all__ = ["display_face_with_area_map", "display_mesh_with_heatmap"]

import open3d as o3d

def display_face_with_area_map(
    area_map: o3d.geometry.LineSet,
    mesh: o3d.geometry.TriangleMesh = None,
    **kwargs
) -> None:
    geometries = [area_map]
    if mesh is not None:
        geometries.append({"name" : "face", "geometry" : mesh})

    o3d.visualization.draw(
        geometry=geometries,
        title='Open3D', 
        width=1024, 
        height=1024, 
        actions=None, 
        lookat=None, 
        eye=None, 
        up=None, 
        field_of_view=60.0, 
        bg_color=(1.0, 1.0, 1.0, 1.0), 
        bg_image=None, 
        ibl=None, 
        ibl_intensity=None, 
        show_skybox=False, 
        show_ui=False, 
        raw_mode=False, 
        point_size=None, 
        line_width=None, 
        animation_time_step=1.0, 
        animation_duration=None, 
        rpc_interface=False, 
        on_init=None, 
        on_animation_frame=None, 
        on_animation_tick=None, 
        non_blocking_and_return_uid=False
    )


def display_mesh_with_heatmap(
    heatmap: o3d.geometry.TriangleMesh,
    mesh: o3d.geometry.TriangleMesh = None,
    **kwargs
) -> None:

    geometries = [heatmap]
    if mesh is not None:
        geometries.append({"name" : "face", "geometry" : mesh})

    o3d.visualization.draw_geometries([g["geometry"] for g in geometries])