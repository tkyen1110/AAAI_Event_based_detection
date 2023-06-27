import numpy as np
from models.functions.point_cloud_ops import points_to_voxel


class VoxelGenerator:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32) # [0, 0, 0, 512, 512, 4]
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32) # [1, 1, 4]
        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size # [512, 512, 1]

        grid_size = np.round(grid_size).astype(np.int64) # [512, 512, 1]

        self._voxel_size = voxel_size # [1, 1, 4]
        self._point_cloud_range = point_cloud_range # [0, 0, 0, 512, 512, 4]
        self._max_num_points = max_num_points # N = 5
        self._max_voxels = max_voxels # Pmax = 100000
        self._grid_size = grid_size # [512, 512, 1]

    def generate(self, points, max_voxels):
        return points_to_voxel(
            points, self._voxel_size, self._point_cloud_range,
            self._max_num_points, True, max_voxels)

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points


    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size