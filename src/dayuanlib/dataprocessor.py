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
        



