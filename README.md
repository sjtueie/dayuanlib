
<p align="center">
  <a href="https://github.com/sjtueie/dayuanlib">dayuanlib</a>
</p>
<p align="center">
<a href="https://pypi.org/project/dayuanlib/" target="_blank">
    <img src="https://img.shields.io/pypi/v/dayuanlib?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/dayuanlib/" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/dayuanlib.svg?color=%2334D058" alt="Supported Python versions">
</a>

</p>


### dataprocessor
```python
from dayuanlib import dataprocessor

processor = dataprocessor.DataProcessor() 

# 加载全域数据
arr = processor.load_arr(settings.data_output_final_dir, timeStr)

vars = [
        '100u_hres','100v_hres','200u_hres','200v_hres','tp_hres','sp_hres','2t_hres','10u_hres',
        '10v_hres','tcc_hres','ssr_hres','u10_fw','v10_fw','t2m_fw','z850_fw','z1000_fw','q850_fw','q1000_fw',
        'u850_fw','u1000_fw','v850_fw','v1000_fw','t850_fw','t1000_fw','u10_pg','v10_pg','t2m_pg','z1000_pg',
        'z850_pg', 'q1000_pg', 'q850_pg', 't1000_pg', 't850_pg', 'u1000_pg', 'u850_pg', 'v1000_pg','v850_pg',
        'u100_ens','v100_ens','t2m_ens','sp_ens','tp_ens','ssrd_ens','tcc_ens','u25_ens','u50_ens','u75_ens',
        'v25_ens','v50_ens','v75_ens'
    ]

# 数据切割
processor.process(arr, timestr=timeStr,
                                  step="6:54",
                                  lat_range=(42, 33),
                                  lon_range=(109, 116),
                                  var_list=vars,
                                  outpath="processed")
```


### GridFinder
```python
from dayuanlib import tools
grid_finder = tools.GridFinder()
# 现在返回的是index 不是经纬度 还需要更新一次
lat_range, lon_range = grid_finder.get_lat_lon_ranges([[42, 109]])
```

### xmlzip
```python
from dayuanlib import tools

tools.zip2xml("script/2025-03-09_08_00_00.zip", "script/2025-03-09_08_00_00.xml")

tools.xml2zip("script/2025-03-09_08_00_00.xml", "script/2025-03-09_08_00_00_exml.zip")

```

---

版本号由 git tag 自动管理
