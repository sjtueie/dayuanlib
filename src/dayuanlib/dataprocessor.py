# logger setup
import logging
import os

import numpy as np

logger = logging.getLogger('DataProcessor')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class DataProcessor:
    
    VARS = [
        '100u_hres','100v_hres','200u_hres','200v_hres','tp_hres','sp_hres','2t_hres','10u_hres',
        '10v_hres','tcc_hres','ssr_hres','u10_fw','v10_fw','t2m_fw','z850_fw','z1000_fw','q850_fw','q1000_fw',
        'u850_fw','u1000_fw','v850_fw','v1000_fw','t850_fw','t1000_fw','u10_pg','v10_pg','t2m_pg','z1000_pg',
        'z850_pg', 'q1000_pg', 'q850_pg', 't1000_pg', 't850_pg', 'u1000_pg', 'u850_pg', 'v1000_pg','v850_pg',
        'u100_ens','v100_ens','t2m_ens','sp_ens','tp_ens','ssrd_ens','tcc_ens','u25_ens','u50_ens','u75_ens',
        'v25_ens','v50_ens','v75_ens'
    ]
    
    def __init__(self):
        pass
      
    @staticmethod
    def initial_time2timestr(initial_time):
        """将初始时间转换为时间字符串格式"""
        time_parts = initial_time.split('-')
        time_parts[-1] = time_parts[-1][:2]
        timestr = ''.join(time_parts)
        return timestr
    
    def quick_cut(self, arr,
                  step="0:150",
                  lat_range=(54.0, 18.0),
                  lon_range=(74.0, 134.0),
                  var_list=None):
        """
        快速切片数组数据
        
        Args:
            arr: 输入数组 (nt, ny, nx, nv)
            step: 时间步切片，可以是字符串 "i:j"、元组 (i,j) 或 slice 对象
            lat_range: 纬度范围 (上, 下)，降序
            lon_range: 经度范围 (左, 右)
            var_list: 变量列表
            
        Returns:
            sub: 切片后的子数组
            coords: 坐标信息字典
        """
        if var_list is None:
            var_list = self.VARS
            
        nt, ny, nx, nv = arr.shape
        assert (ny, nx, nv) == (145, 241, 50), f"形状不符：{arr.shape}"

        # 构造坐标（不改变顺序）
        lat = 54.0 - 0.25 * np.arange(ny)   # 降序：54, 53.75, ..., 18
        lon = 74.0 + 0.25 * np.arange(nx)   # 升序：74, ..., 134
        assert lat[0] > lat[-1], "纬度轴必须是降序"

        # 时间步切片
        if isinstance(step, str):
            a, b = step.split(':')
            t_sl = slice(int(a), int(b))
        elif isinstance(step, (tuple, list)):
            t_sl = slice(step[0], step[1])
        elif isinstance(step, slice):
            t_sl = step
        else:
            raise ValueError("step 需为 'i:j'、(i,j) 或 slice")

        # 纬度（降序坐标）的闭区间切片
        top, bottom = lat_range
        if top < bottom:
            raise ValueError("lat_range 必须按降序给，例如 (54, 37)")
        asc = lat[::-1]
        start_r = np.searchsorted(asc, bottom, side='left')
        stop_r = np.searchsorted(asc, top, side='right')
        y_sl = slice(ny - stop_r, ny - start_r)

        # 经度（升序坐标）的闭区间切片
        left, right = min(lon_range), max(lon_range)
        x_start = np.searchsorted(lon, left, side='left')
        x_stop = np.searchsorted(lon, right, side='right')
        x_sl = slice(x_start, x_stop)

        # 变量索引（保持原顺序）
        name2idx = {n: i for i, n in enumerate(self.VARS)}
        missing = [n for n in var_list if n not in name2idx]
        if missing:
            raise ValueError(f"未知变量: {missing}\n可用变量: {self.VARS}")
        v_idx = [name2idx[n] for n in var_list]

        # 只做原数组的连续切片，不翻转不重排
        sub = arr[t_sl, y_sl, x_sl, :][..., v_idx]
        coords = {
            "lat": lat[y_sl],
            "lon": lon[x_sl],
            "step": np.arange(nt)[t_sl],
            "vars": [self.VARS[i] for i in v_idx]
        }
        return sub, coords

    def load_arr(self, output_final, timestr):
        # 构建路径
        arr_path = os.path.join(output_final, f'{timestr}.npy')
        logger.info(f"Loading array from {arr_path}")
        arr = np.load(arr_path)
        return arr
    
    
    def process(self,arr,timestr,step="6:54", lat_range=(42, 33), lon_range=(109, 116), var_list=None,outpath='./processed'):
        """处理数据的主方法"""
        sub, coords = self.quick_cut(
            arr,
            step=step,
            lat_range=lat_range,
            lon_range=lon_range,
            var_list=var_list
        )
        
        # 转置和提取场站数据
        sub = sub.transpose(0, 2, 1, 3)  # (48, 29, 37, 27)
        # 创建 self.outpath 目录（如果不存在）
        os.makedirs(outpath, exist_ok=True)
        MID_SAVE_PATH = os.path.join(outpath, f'processed_{timestr}.npy')
        np.save(MID_SAVE_PATH, sub)
        


    def quick_cut_1x1(self,
                arr,
                step="0:150",
                lat_range=(54.0, 18.0),
                lon_range=(74.0, 134.0),
                var_list=VARS,
                lat_point=None,
                lon_point=None,
                nearest=True,
                squeeze=False):
        """
        从 0.25° 网格数据中切子区域或单点。
        - 若同时提供 lat_point 和 lon_point，则按“单点抽取”；
        否则按原来的 lat_range/lon_range 闭区间切片。
        - nearest=True 时，单点会对齐到最近网格点；False 则要求正好落在网格上。
        - squeeze=True 时，若抽到的是单点，会挤掉 y/x 两个长度为 1 的维度。
        返回: sub, coords
        sub.shape = (t, y, x, v)（或 squeeze 后 (t, v)）
        coords = {"lat":..., "lon":..., "step":..., "vars":...}
        """
        nt, ny, nx, nv = arr.shape
        assert (ny, nx, nv) == (145, 241, 50), f"形状不符：{arr.shape}"

        # 构造坐标（不改变顺序）
        dlat = 0.25; dlon = 0.25
        lat0, lon0 = 54.0, 74.0
        lat = lat0 - dlat*np.arange(ny)   # 降序：54, 53.75, ..., 18
        lon = lon0 + dlon*np.arange(nx)   # 升序：74, ..., 134
        assert lat[0] > lat[-1], "纬度轴必须是降序"

        # 时间步切片
        if isinstance(step, str):
            a, b = step.split(':'); t_sl = slice(int(a), int(b))  # 右端开区间
        elif isinstance(step, (tuple, list)): t_sl = slice(step[0], step[1])
        elif isinstance(step, slice): t_sl = step
        else: raise ValueError("step 需为 'i:j'、(i,j) 或 slice")

        # ——空间切片：支持“单点”或“区间”——
        if (lat_point is not None) and (lon_point is not None):
            # 计算最近网格索引
            y_idx_f = (lat0 - float(lat_point)) / dlat   # 由于纬度降序
            x_idx_f = (float(lon_point) - lon0) / dlon   # 经度升序
            if nearest:
                y_idx = int(np.rint(y_idx_f))
                x_idx = int(np.rint(x_idx_f))
            else:
                eps = 1e-6
                if not (abs(y_idx_f - round(y_idx_f)) < eps and abs(x_idx_f - round(x_idx_f)) < eps):
                    raise ValueError(f"给定点不在网格上（lat={lat_point}, lon={lon_point}），"
                                    f"可将 nearest=True 或改为正好落在 0.25° 网格。")
                y_idx = int(round(y_idx_f))
                x_idx = int(round(x_idx_f))
            # 边界保护
            if not (0 <= y_idx < ny and 0 <= x_idx < nx):
                raise ValueError(f"点超出范围：lat={lat_point}, lon={lon_point}")
            y_sl = slice(y_idx, y_idx+1)
            x_sl = slice(x_idx, x_idx+1)
        else:
            # ——纬度（降序坐标）的闭区间切片——
            top, bottom = lat_range  # 期待传入“大到小”
            if top < bottom:
                raise ValueError("lat_range 必须按降序给，例如 (54, 37)")
            asc = lat[::-1]  # 升序副本，仅用于定位索引
            start_r = np.searchsorted(asc, bottom, side='left')
            stop_r  = np.searchsorted(asc, top,    side='right')
            y_sl = slice(ny - stop_r, ny - start_r)

            # ——经度（升序坐标）的闭区间切片——
            left, right = min(lon_range), max(lon_range)
            x_start = np.searchsorted(lon, left,  side='left')
            x_stop  = np.searchsorted(lon, right, side='right')
            x_sl = slice(x_start, x_stop)

        # 变量索引（保持原顺序）
        name2idx = {n:i for i,n in enumerate(self.VARS)}
        missing = [n for n in var_list if n not in name2idx]
        if missing:
            raise ValueError(f"未知变量: {missing}\n可用变量: {self.VARS}")
        v_idx = [name2idx[n] for n in var_list]

        # 只做原数组的连续切片，不翻转不重排
        sub = arr[t_sl, y_sl, x_sl, :][..., v_idx]

        # 构造坐标
        step_idx = np.arange(nt)[t_sl]
        lat_sel = lat[y_sl]
        lon_sel = lon[x_sl]
        coords = {
            "lat": (float(lat_sel[0]) if len(lat_sel)==1 else lat_sel),
            "lon": (float(lon_sel[0]) if len(lon_sel)==1 else lon_sel),
            "step": step_idx,
            "vars": [self.VARS[i] for i in v_idx]
        }

        # 单点可选 squeeze 到 (t, v)
        if squeeze and (len(lat_sel)==1) and (len(lon_sel)==1):
            sub = np.squeeze(sub, axis=(1,2))  # (t, v)

        return sub, coords

    def quick_cut_8x8(
        self,
        arr,
        coords,
        n=8,
        step="0:150",
        var_list=None,
    ):
        """
        从 0.25° 网格数据中，为若干目标经纬度坐标提取 n×n 的最优子网格 patch。

        逻辑：
        - 对每个目标点，在所有可行的 n×n 正方形窗口中，
          计算“四个角点到目标点距离的方差”，选择方差最小的那个窗口
          （对应 GridFinder.find_optimal_square_grid 的逻辑）。
        - 再根据这些最优窗口，从原始数据中批量提取 patch
          （对应 GridFinder.batch_extract_patches 的逻辑）。

        参数：
        - arr: 原始数据数组，shape=(t, y, x, v)，其中 (y,x,v)=(145,241,50)
        - coords: 目标经纬度列表，形如 [[lat1, lon1], [lat2, lon2], ...]
        - n: 正方形网格边长（默认 8，即 8×8）
        - step: 时间步切片，支持 "i:j" 或 (i,j) 或 slice 对象
        - var_list: 变量名列表；为 None 时使用 self.VARS

        返回：
        - patches: shape = (N, T, n*n, V_sel)
            N: 目标点个数
            T: 时间步个数（切片后）
            n*n: 展平后的网格点（如 8×8=64）
            V_sel: 选择的变量个数
        - info: 字典，包含：
            - "optimal_squares": shape=(N, n*n, 2)，每个网格点的 (y,x) 索引
            - "step": 时间步索引数组
            - "vars": 实际变量名列表
            - "centers": 输入的经纬度列表
            - "lat_patch": shape=(N, n, n) 对应网格的纬度
            - "lon_patch": shape=(N, n, n) 对应网格的经度
        """
        nt, ny, nx, nv = arr.shape
        assert (ny, nx, nv) == (145, 241, 50), f"形状不符：{arr.shape}"

        coords = np.atleast_2d(np.asarray(coords, dtype=float))
        N = coords.shape[0]

        # ---------------- 1. 构造经纬度坐标 ----------------
        dlat = 0.25
        dlon = 0.25
        lat0, lon0 = 54.0, 74.0   # 与 lat_range=(54,18), lon_range=(74,134) 一致

        # 1D 坐标
        lat = lat0 - dlat * np.arange(ny)   # 降序：54, 53.75, ..., 18
        lon = lon0 + dlon * np.arange(nx)   # 升序：74, ..., 134

        # 2D 坐标网格，方便后面拿 patch 的经纬度
        lat2d = np.repeat(lat[:, None], nx, axis=1)   # (ny, nx)
        lon2d = np.repeat(lon[None, :], ny, axis=0)   # (ny, nx)

        # ---------------- 2. 时间步切片 ----------------
        if isinstance(step, str):
            a, b = step.split(':')
            t_sl = slice(int(a), int(b))
        elif isinstance(step, (tuple, list)):
            t_sl = slice(step[0], step[1])
        elif isinstance(step, slice):
            t_sl = step
        else:
            raise ValueError("step 需为 'i:j'、(i,j) 或 slice")

        # 安全裁剪时间范围
        t_sl = slice(
            0 if t_sl.start is None else max(0, t_sl.start),
            nt if t_sl.stop  is None else min(nt, t_sl.stop),
        )
        T = t_sl.stop - t_sl.start
        step_idx = np.arange(nt)[t_sl]

        # ---------------- 3. 找每个目标点的最优 n×n 正方形 ----------------
        optimal_squares = []  # 每个元素：shape=(n*n, 2)，存 (y, x)

        for k in range(N):
            target_lat, target_lon = coords[k]
            best_var = None
            best_top = None  # (i, j) 左上角

            # 穷举所有 n×n 窗口左上角索引 (i, j)
            for i in range(ny - n + 1):
                for j in range(nx - n + 1):
                    # 四个角点经纬度
                    corners = [
                        (lat[i],         lon[j]),         # 左上
                        (lat[i],         lon[j + n - 1]), # 右上
                        (lat[i + n - 1], lon[j]),         # 左下
                        (lat[i + n - 1], lon[j + n - 1])  # 右下
                    ]

                    # 计算距离的平方（不开方即可，不影响方差排序）
                    d2 = []
                    for (clat, clon) in corners:
                        d2.append((target_lat - clat) ** 2 + (target_lon - clon) ** 2)
                    d2 = np.asarray(d2)
                    var = np.var(d2)

                    if (best_var is None) or (var < best_var):
                        best_var = var
                        best_top = (i, j)

            if best_top is None:
                raise RuntimeError("未找到最优网格（理论上不应发生）")

            i0, j0 = best_top
            # 生成该 n×n 方格内所有 (y, x) 索引，展平为 n*n 个点
            square_idx = [(i0 + di, j0 + dj) for di in range(n) for dj in range(n)]
            optimal_squares.append(square_idx)

        optimal_squares = np.asarray(optimal_squares, dtype=int)  # (N, n*n, 2)

        # ---------------- 4. 变量维度选择 ----------------
        all_vars = self.VARS
        if var_list is None:
            var_list = all_vars

        name2idx = {n_: i for i, n_ in enumerate(all_vars)}
        missing = [n_ for n_ in var_list if n_ not in name2idx]
        if missing:
            raise ValueError(f"未知变量: {missing}\n可用变量: {all_vars}")
        v_idx = [name2idx[n_] for n_ in var_list]

        # ---------------- 5. 按 optimal_squares 批量提取 patch ----------------
        # 对应你原来的 batch_extract_patches:
        # data = data[:, optimal_squares[:, :, 0], optimal_squares[:, :, 1], :]
        # data = np.swapaxes(data, 1, 0)
        data_sel = arr[t_sl, :, :, :]  # (T, ny, nx, nv)

        y_idx = optimal_squares[:, :, 0]  # (N, n*n)
        x_idx = optimal_squares[:, :, 1]  # (N, n*n)

        # 利用广播：先在时间维切，再用 (N, n*n) 的索引
        # 结果形状： (T, N, n*n, nv)
        data_patches = data_sel[:, y_idx, x_idx, :]

        # swapaxes(1,0) → (N, T, n*n, nv)
        data_patches = np.swapaxes(data_patches, 1, 0)

        # 再选变量维度：V_sel
        data_patches = data_patches[..., v_idx]  # (N, T, n*n, V_sel)

        # ---------------- 6. 附带每个 patch 的经纬度网格 ----------------
        # lat_patch[k] 取出该目标点对应 n×n 网格的纬度
        lat_patch = np.empty((N, n*n), dtype=float)
        lon_patch = np.empty((N, n*n), dtype=float)
        for k in range(N):
            yi = optimal_squares[k, :, 0]
            xi = optimal_squares[k, :, 1]
            lat_patch[k] = lat2d[yi, xi]
            lon_patch[k] = lon2d[yi, xi]

        lat_patch = lat_patch.reshape(N, n, n)
        lon_patch = lon_patch.reshape(N, n, n)

        info = {
            "optimal_squares": optimal_squares,   # (N, n*n, 2)
            "step": step_idx,                     # (T,)
            "vars": [all_vars[i] for i in v_idx],
            "centers": coords,                    # (N, 2) 输入的经纬度
            "lat_patch": lat_patch,               # (N, n, n)
            "lon_patch": lon_patch,               # (N, n, n)
        }

        return data_patches, info
