# NWM - 数值天气模式 ZTD 生成器

基于数值天气预报（NWM/ERA5）三维气象数据计算 Zenith Tropospheric Delay (ZTD) 的高性能 Python 框架。

---

## 安装

```bash
# 使用 uv（推荐）
uv sync

# 使用 pip
pip install -e .
```

---

## 快速入门

```python
from nwm.ztd_nwm import ZTDNWMGenerator
import pandas as pd

# 1. 准备站点坐标（必须包含 lat, lon, alt, site 列）
location = pd.DataFrame({
    "site": ["BJFS", "WUHN"],
    "lat": [39.6, 30.5],
    "lon": [115.9, 114.4],
    "alt": [87.4, 25.8],  # 椭球高，单位：米
})

# 2. 创建 ZTD 生成器并运行
zg = ZTDNWMGenerator(
    nwm_path="era5_pl_native_2023010100.nc",
    location=location,
)
df = zg.run()
print(df)
```

**输出示例**：

| time | site | ztd_simpson (mm) |
|------|------|------------------|
| 2023-01-01 00:00:00 | BJFS | 2312.456 |
| 2023-01-01 00:00:00 | WUHN | 2398.123 |

---

## 输入数据

**气象数据文件**

支持 NetCDF 或 GRIB 格式，必须包含：

| 变量 | 说明 | 单位 |
|------|------|------|
| `t` | 温度 | K |
| `z` | 位势 $\Phi$ | m²/s² |
| `q` | 比湿 | kg/kg |

**站点坐标**

| 列名 | 说明 | 单位 |
|------|------|------|
| `site` | 站点标识符 | - |
| `lat` | 纬度 | ° |
| `lon` | 经度 | ° |
| `alt` | **WGS84椭球高** | m |

---

## API

### *class* **ZTDNWMGenerator**

从数值天气模式数据计算天顶对流层延迟 (ZTD) 的生成器。

```python
ZTDNWMGenerator(
    nwm_path,
    location=None,
    egm_type="egm96-5",
    vertical_level="pressure_level",
    n_jobs=-1,
    batch_size=100_000,
    horizontal_interpolation_method="linear",
    resample_h=(None, None, 50),
    interp_to_site=True,
    refractive_stats=None,
)
```

**Parameters**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `nwm_path` | *str* / *Path* | 必需 | NWM/ERA5 气象数据文件路径 |
| `location` | *DataFrame* | `None` | 站点坐标，包含 `lat`, `lon`, `alt`, `site` 列 |
| `egm_type` | *str* | `"egm96-5"` | 大地水准面模型：`"egm96-5"` / `"egm2008-1"` |
| `vertical_level` | *str* | `"pressure_level"` | 垂直坐标模式（见[配置选项](#垂直坐标模式)） |
| `n_jobs` | *int* | `-1` | 并行核心数，`-1` 使用全部 |
| `batch_size` | *int* | `100_000` | 向量化/并行切换阈值 |
| `horizontal_interpolation_method` | *str* | `"linear"` | 水平插值方法（传递给 `RegularGridInterpolator`） |
| `resample_h` | *tuple* | `(None, None, 50)` | 高度重采样 `(h_min, h_max, interval)`，单位：米 |
| `interp_to_site` | *bool* | `True` | `True` 输出站点 ZTD，`False` 输出垂直剖面 |
| `refractive_index_constants` | *tuple* | `(77.689, 71.2952, 375463.0)` | 自定义折射率常数 `(k1, k2, k3)`，单位见下文 |



### *method* ZTDNWMGenerator.**run**()

执行完整 ZTD 计算流程。

**Returns**

当 `interp_to_site=True` 时：

| 列名 | 类型 | 说明 |
|------|------|------|
| `time` | *datetime64* | 时间戳 |
| `site` | *str* | 站点标识符 |
| `ztd_simpson` | *float* | ZTD 值，单位：**mm** |

当 `interp_to_site=False` 时，额外包含 `h` 列（高度层，单位：m）。

**Raises**

若最终结果中存在 NaN 值，抛出 `ValueError`。



## 计算步骤

### 高程系统转换

NWM 数据使用位势，需转换为椭球高
$$
\Phi(位势) \xrightarrow{} H_{gp}(位势高度) \xrightarrow{}  H(正高) \xrightarrow{} h(椭球高)
$$

| 符号 | 名称 | 单位 | 说明 |
|------|------|------|------|
| $\Phi$ | 位势 | m²/s² | NWM 原始变量 `z` |
| $H_{gp}$ | 位势高度 | m | IFS 定义 $H_{gp}= \Phi / g_0$ |
| $H$ | 正高 | m | 相对于大地水准面 |
| $h$ | 椭球高 | m | 相对于 WGS84 椭球面 |

**位势 → 正高**

首先将位势 $\Phi$ 转换为位势高度：

$$
H_{gp} = \frac{\Phi}{g_0}
$$

> 这一步完全符合 ECMWF IFS 文档，其中明确规定 $g_0$ 为固定值。
>
> *Source: ECMWF IFS documentation CY47R3 – Part IV: Physical processes, Chapter 12.*

然后转换为正高（Geometric Height）。**注意：** 此处 NWM 采用了比 ERA5 官方文档该更严密的 **Mahoney (2001)** 算法，使用随纬度变化的重力和椭球半径，以配合 GNSS 的 WGS84 椭球系统（ERA5 文档仅提供球体近似公式）。

$$
H = \frac{R(\varphi) \cdot H_{gp}}{\dfrac{g(\varphi)}{g_0} \cdot R(\varphi) - H_{gp}}
$$

其中，$g(\varphi) $ 为重力加速度：

$$
g(\varphi) = g_e \cdot \frac{1 + k \sin^2\varphi}{\sqrt{1 - e^2 \sin^2\varphi}}
$$

$R(\varphi)$ 为有效地球半径：

$$
R(\varphi) = \frac{a}{1 + f + m - 2f \sin^2\varphi}
$$

| 常数 | 值 | 描述 | 来源 |
|------|-----|------|------|
| $g_0$ | $9.80665 \, \text{m/s}^2$ | WMO 标准重力 | ECMWF (2021), Mahoney (2001) |
| $g_e$ | $9.7803253359 \, \text{m/s}^2$ | WGS84 赤道重力 | Mahoney (2001) |
| $k$ | $1.931853 \times 10^{-3}$ | Somigliana 常数 | Mahoney (2001) |
| $e$ | $0.081819$ | WGS84 第一偏心率 | Mahoney (2001) |
| $a$ | $6378137.0 \, \text{m}$ | WGS84 长半轴 | Mahoney (2001) |
| $f$ | $0.003352811$ | WGS84 地球扁率 | Mahoney (2001) |
| $m$ | $0.003449787$ | WGS84 重力比 | Mahoney (2001) |

**正高 → 椭球高**
$$
h = H + N
$$

其中 $N$ 为大地水准面差距，通过 EGM96 或 EGM2008 模型查表获得。默认采用 EGM96（以保持与 ECMWF IFS 模式一致）。

> ECMWF (2021). *IFS documentation CY47R3 – Part IV: Physical processes*. Chapter 12.
>
> Mahoney, M. J. (2001). A discussion of various measures of altitude. *NASA Jet Propulsion Laboratory*.

---

### 水汽压计算

水汽压 $e$ 由从比湿 $q$ 和气压 $p$ 计算：

$$
e = \frac{q \cdot p}{\epsilon + (1 - \epsilon) \cdot q}
$$

其中 $\epsilon = R_d / R_v \approx 0.622$（水汽与干空气气体常数比）。

> ECMWF (2021). *IFS documentation CY47R3 – Part IV: Physical processes*. Chapter 12. 

---

### 折射率计算

折射率由静力学和非静力学部分组成：

$$
N = N_h + N_w
$$

**静力学折射率**
$$
N_h = k_1 R_d \rho_m
$$

其中 $\rho_m = \rho_d + \rho_v$ 为湿空气密度。

**非流体静力学折射率**
$$
N_w = k_2' \frac{e}{T} + k_3 \frac{e}{T^2}
$$

其中：

$$
k_2' = k_2 - k_1 \frac{R_d}{R_v}
$$

| 常数 | 默认值 | 单位 | 来源 |
|------|-----|------|------|
| $k_1$ | $77.689$ | K/hPa | Rüeger (2002) |
| $k_2$ | $71.2952$ | K/hPa | Rüeger (2002) |
| $k_3$ | $375463$ | K²/hPa | Rüeger (2002) |
| $R_d$ | $287.0597$ | J/(kg·K) | ECMWF (2021) |
| $R_v$ | $461.5250$ | J/(kg·K) | ECMWF (2021) |



通过 `refractive_index_constants` 参数可传入自定义折射率常数覆盖默认值：

```python
from nwm.ztd_nwm import ZTDNWMGenerator

# 使用 Bevis et al. (1994) 的常数 (k1, k2, k3)
custom = (77.60, 70.40, 373900.0)

zg = ZTDNWMGenerator(
    nwm_path="era5_data.nc",
    location=location,
    refractive_index_constants=custom,
)
```

> Rüeger, J. M. (2002). Refractive index formulae for radio waves. *Proceedings of the FIG XXII International Congress*.
>
> ECMWF (2021). *IFS documentation CY47R3 – Part IV: Physical processes*. Chapter 12.

---

### 数值积分与边界条件

ZTD 由模型层内的数值积分和模型顶层以上的边界条件两部分组成：

$$
\text{ZTD}(h) = 10^{-6} \int_h^{h_{top}} N \, dh + \text{ZTD}_{top}
$$

积分节点可配置：

- `vertical_level = "pressure_level"`：直接使用模型定义的原始气压层作为积分节点
- `vertical_level = "h"`：先将气象参数重采样到固定间隔的椭球高网格，然后在该网格上执行积分。该模式需要额外的气象参数垂直插值和外推

顶层 ZTD 仅考虑流体静力学延迟 ZHD，采用 Davis 改进的 Saastamoinen 模型计算：
$$
\text{ZTD}_{top}=\text{ZHD} = \frac{0.0022768 \cdot p}{1 - 0.00266 \cos(2\varphi) - 0.00028 \cdot h \times 10^{-3}}
$$

其中 $p$ 为气压（hPa），$\varphi$ 为纬度（rad），$h$ 为椭球高（m）。

> Davis, J. L., et al. (1985). Geodesy by radio interferometry: Effects of atmospheric modeling errors on estimates of baseline length. *Radio Science*, 20(6).

---

### 气象参数垂直插值和外推

设置 `vertical_level = "h"` 时执行此步骤。将原始气压层上的气象参数 $(T, p, e)$ 重采样到等间距椭球高网格 $h_k$

**插值** $(h_k \ge h_{bottom})$

- 气压 $p$ 和水汽压 $e$ 随高度呈指数变化，采用对数线性插值

- 温度 $T$ 随高度近似线性变化，采用线性插值

**外推** $(h_k < h_{bottom})$

- 水汽压保持定值：

$$
e = e_{bottom}
$$

- 温度采用标准递减率 6.5 K/km (WMO, 2024)：

$$
T = T_{bottom} + 0.0065 \cdot (h_{bottom} - h_k)
$$

- 气压采用虚温气压计公式 (WMO, 2024)：

$$
p = p_{bottom} \cdot \exp\left(\frac{g_0 \cdot (h_{bottom} - h_k)}{R_d \cdot T_{mv}}\right)
$$

​	其中虚温
$$
T_{mv} = 0.5(T_{bottom} + T) + 0.12 \cdot e_{bottom}
$$

> World Meteorological Organization. (2024). *Guide to Instruments and Methods of Observation: Volume I – Measurement of Meteorological Variables* (WMO-No. 8, 2024 ed.). Geneva, Switzerland: WMO.

---

### 对流层延迟插值与外推

数值积分可生成所有 `location` 经纬度坐标、所有高度层三维 ZTD 数据。设置 `interp_to_site=True` 时将 ZTD 插值或外推至到站点高程

**插值**（站点在相邻高度层之间，$h_{site} \ge h_{bottom}$）

- 采用对数线性插值。设 $h_i$ 和 $h_{i+1}$ 为包围站点的相邻高度层，则：

$$
\ln \text{ZTD}(h_{site}) = \ln \text{ZTD}_i + \frac{h_{site} - h_i}{h_{i+1} - h_i} \cdot \left( \ln \text{ZTD}_{i+1} - \ln \text{ZTD}_i \right)
$$

**外推**（站点低于最低高度层，$h_{site} < h_{bottom}$）

- 先按上节方法外推底层气象参数到站点高度，计算站点折射率 $N_{site}$，然后梯形积分外推层的延迟增量：

$$
\Delta \text{ZTD} = 10^{-6} \cdot \frac{N_{bottom} + N_{site}}{2} \cdot (h_{bottom} - h_{site})
$$

- 站点 $ZTD_{site}$ 即为底层 $ZTD_{bottom}$ 附加增量
  $$
  ZTD_{site}=ZTD_{bottom}+\Delta ZTD
  $$
  

