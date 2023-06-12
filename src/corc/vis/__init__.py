__all__ = ["volume_diff_area_map", "display_face_with_area_map", "volume_diff_heat_map", "display_mesh_with_heatmap", "BLUE", "RED", "LEFT", "RIGHT", "color_from_dist", "mirror_mesh"]

from corc.vis.difference import volume_diff_area_map, volume_diff_heat_map, color_from_dist
from corc.vis.visuals import display_face_with_area_map, display_mesh_with_heatmap
from corc.vis.mesh import mirror_mesh

BLUE = [ 44/255,  111/255,  187/255] # matte blue
RED  = [201/255,   44/255,   17/255] # thunderbird red

# add some just as call for LEFT and RIGHT
LEFT = BLUE
RIGHT = RED