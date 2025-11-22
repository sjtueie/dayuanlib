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

    def quick_cut_8x8(self,arr,
                  step="0:150",
                  center_lat=None,
                  center_lon=None,
                  var_list=VARS,
                  squeeze=False):
        """
        从 0.25° 网格数据中切 8×8 的子网格。
        以指定点为中心，向四周各延伸 1° (4个网格点)，形成 8×8 网格。
        
        参数：
        - arr: 原始数据数组，shape=(t, y, x, v)，其中 (y,x,v)=(145,241,50)
        - step: 时间步切片，支持 "i:j" 或 (i,j) 或 slice 对象
        - center_lat: 中心纬度（必需）
        - center_lon: 中心经度（必需）
        - var_list: 变量名列表
        - squeeze: 是否压缩时间/空间维度为1的情况
        
        返回：
        - sub: 切片后的数据，shape=(t, 8, 8, v) 或 squeeze 后的形状
        - coords: 坐标字典 {"lat": array, "lon": array, "step": array, "vars": list}
        """
        nt, ny, nx, nv = arr.shape
        assert (ny, nx, nv) == (145, 241, 50), f"形状不符：{arr.shape}"
        
        if center_lat is None or center_lon is None:
            raise ValueError("必须提供 center_lat 和 center_lon")
        
        # 构造坐标
        dlat = 0.25
        dlon = 0.25
        lat0, lon0 = 54.0, 74.0
        lat = lat0 - dlat * np.arange(ny)   # 降序：54, 53.75, ..., 18
        lon = lon0 + dlon * np.arange(nx)   # 升序：74, ..., 134
        
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
        
        # 计算中心点的网格索引
        y_center_f = (lat0 - float(center_lat)) / dlat
        x_center_f = (float(center_lon) - lon0) / dlon
        
        y_center = int(np.rint(y_center_f))
        x_center = int(np.rint(x_center_f))
        
        # 计算 8×8 网格的起止索引（向四周各延伸 4 个网格点）
        y_start = y_center - 4
        y_end = y_center + 4  # Python 切片右端开区间，所以是 +4
        x_start = x_center - 4
        x_end = x_center + 4
        
        # 边界保护
        if not (0 <= y_start and y_end <= ny and 0 <= x_start and x_end <= nx):
            raise ValueError(
                f"8×8 网格超出范围。中心点: ({center_lat}, {center_lon}), "
                f"索引: ({y_center}, {x_center}), "
                f"需求范围: y=[{y_start}:{y_end}], x=[{x_start}:{x_end}], "
                f"有效范围: y=[0:{ny}], x=[0:{nx}]"
            )
        
        y_sl = slice(y_start, y_end)
        x_sl = slice(x_start, x_end)
        
        # 变量索引
        name2idx = {n: i for i, n in enumerate(self.VARS)}
        missing = [n for n in var_list if n not in name2idx]
        if missing:
            raise ValueError(f"未知变量: {missing}\n可用变量: {self.VARS}")
        v_idx = [name2idx[n] for n in var_list]
        
        # 切片
        sub = arr[t_sl, y_sl, x_sl, :][..., v_idx]
        
        # 验证是否确实是 8×8
        assert sub.shape[1:3] == (8, 8), f"切片后不是 8×8：{sub.shape}"
        
        # 构造坐标
        step_idx = np.arange(nt)[t_sl]
        lat_sel = lat[y_sl]
        lon_sel = lon[x_sl]
        coords = {
            "lat": lat_sel,
            "lon": lon_sel,
            "step": step_idx,
            "vars": [self.VARS[i] for i in v_idx],
            "center": (float(center_lat), float(center_lon)),
        }
        
        # return sub, coords
        if squeeze and (len(lat_sel)==1) and (len(lon_sel)==1):
            sub = np.squeeze(sub, axis=(1,2)) 
        return sub, coords
