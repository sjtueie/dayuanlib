import numpy as np
from scipy.spatial.distance import cdist


class GridFinder:
    """
    输入:
    coord (list or array): 目标坐标列表，每个元素是一个包含纬度和经度 [纬度, 经度]。
    n (int): 正方形网格的大小（n*n）。
    lat_range (tuple): 纬度范围，形式为 (纬度最小值, 纬度最大值)，如 (54, 18)。
    lon_range (tuple): 经度范围，形式为 (经度最小值, 经度最大值)，如 (74, 134)。
    step (float): 网格的步长，表示每个网格的经纬度差异（例如 0.25 表示每个网格单元为 0.25 度）。

    输出:
    最优网格的坐标，每个目标坐标对应一个最优正方形网格的经纬度坐标索引。
    """
    def __init__(self, lat_range=(54, 18), lon_range=(74, 134), step=0.25):
    # def __init__(self, lat_range=(32.25, 18), lon_range=(95.75, 122.25), step=0.25):
    # def __init__(self, lat_range=(53.25, 37), lon_range=(97, 126.75), step=0.25):
        self.lat_max, self.lat_min = lat_range
        self.lon_min, self.lon_max = lon_range
        self.step = step
        
        # 创建网格
        self.grid = np.array([[(y, x) for x in np.arange(self.lon_min, self.lon_max + self.step, self.step)] 
                              for y in np.arange(self.lat_max, self.lat_min - self.step, -self.step)])
        self.grid_shape = self.grid.shape[:2]

    def find_optimal_square_grid(self, coord, n):
        """
        找到最优正方形网格，对于每个目标坐标，找到距离最小的正方形网格
        """
        results = []

        for target in coord:
            min_diff = float('inf')
            best_square = None

            for i in range(self.grid_shape[0] - n + 1):
                for j in range(self.grid_shape[1] - n + 1):
                    corners = [
                        self.grid[i, j],
                        self.grid[i, j + n - 1],
                        self.grid[i + n - 1, j],
                        self.grid[i + n - 1, j + n - 1]
                    ]

                    distances = cdist([target], corners).flatten()
                    diff = np.var(distances)
                    if diff < min_diff:
                        min_diff = diff
                        best_square = [(i + di, j + dj) for di in range(n) for dj in range(n)]
            results.append(best_square)

        return np.array(results)
    

    def get_lat_lon_ranges(self, coords):
        """
        从最优网格索引获取经纬度范围
        """
        optimal_squares = self.find_optimal_square_grid(coords, 8)

        lat_range = (optimal_squares[:, :, 0].min()  , optimal_squares[:, :, 0].max()  )
        lon_range = (optimal_squares[:, :, 1].min()  , optimal_squares[:, :, 1].max()  )
        return lat_range, lon_range

    def get_square_coordinates(self, best_square):
        """
        从最优网格索引获取实际经纬度坐标
        """
        square_coords = []
        for idx in best_square:
            square_coords.append(self.grid[idx[0], idx[1]])
        return np.array(square_coords)

    def batch_extract_patches(self, data, optimal_squares):
        """
        根据最优网格提取数据
        """
        data = data[:, optimal_squares[:, :, 0], optimal_squares[:, :, 1], :]
        data = np.swapaxes(data, 1, 0)
        return data


# if __name__ == "__main__":
#     coords = [[42, 111]]
#     grid_finder = GridFinder()
#     lat_range, lon_range = grid_finder.get_lat_lon_ranges(coords)
#     print(f"纬度范围: {lat_range}, 经度范围: {lon_range}")
        

