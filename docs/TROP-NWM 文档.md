# TROP-NWM 文档

## 概述

TROP-NWM 从三维数值天气模式（NWM）数据计算天顶对流层延迟（ZTD）。输入气象数据可从 [ECMWF Climate Data Store](https://cds.climate.copernicus.eu/) 获取。输入文件须包含：

- **温度**（`t`，K）
- **位势**（`z`，m^2/s^2）
- **比湿**（`q`，kg/kg）

数据需位于气压层上，包含纬度、经度和时间维度。

---

## ZTDNWMGenerator

### 构造函数

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
    refractive_index_constants=(77.689, 71.2952, 375463.0),
    progress_mode="rich",
    time_batch_size=None,
)
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `nwm_path` | str / Path | 必需 | NWM 气象数据文件路径 |
| `location` | DataFrame | `None` | 站点坐标，包含 `lat`、`lon`、`alt`、`site` 列。为 `None` 时在 NWM 原始网格上计算 |
| `egm_type` | str | `"egm96-5"` | 大地水准面模型：`"egm96-5"` 或 `"egm2008-1"` |
| `vertical_level` | str | `"pressure_level"` | 垂直坐标类型（详见[数值积分与边界条件](#数值积分与边界条件)） |
| `n_jobs` | int | `-1` | 并行核心数，`-1` 使用全部 CPU |
| `batch_size` | int | `100_000` | 向量化/并行计算切换阈值 |
| `horizontal_interpolation_method` | str | `"linear"` | 水平插值方法，传递给 `scipy.interpolate.RegularGridInterpolator` |
| `resample_h` | tuple | `(None, None, 50)` | 高度重采样参数 `(h_min, h_max, interval)`，单位米。`None` 自动确定 |
| `interp_to_site` | bool | `True` | `True`：输出站点 ZTD；`False`：输出完整垂直剖面 |
| `refractive_index_constants` | tuple | `(77.689, 71.2952, 375463.0)` | 自定义折射率常数 `(k1, k2, k3)`，默认 Rueger (2002) |
| `progress_mode` | str | `"rich"` | 进度显示模式：`"rich"`（进度条）或 `"simple"`（日志输出） |
| `time_batch_size` | int / None | `None` | 设置后按此大小分批处理时间维度，降低内存峰值 |

### run()

执行 ZTD 计算流程。

**返回** `DataFrame`，包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `time` | datetime64 | 时间戳 |
| `site` | str | 站点标识符 |
| `ztd` | float | 天顶对流层延迟（mm） |

当 `interp_to_site=False` 时，额外包含 `h` 列（高度层，单位 m）。

处理集合产品（如包含多个成员的 EDA）时，输出会包含 `number` 列标识各集合成员。

若最终结果中存在 NaN 值，抛出 `ValueError`。

---

## 计算方法

### 高程系统转换

NWM 提供位势，需转换为椭球高。转换路径为：

$$
\Phi \text{ (位势)} \to H_{gp} \text{ (位势高度)} \to H \text{ (正高)} \to h \text{ (椭球高)}
$$

| 符号 | 名称 | 单位 | 说明 |
|------|------|------|------|
| $\Phi$ | 位势 | m^2/s^2 | NWM 原始变量 `z` |
| $H_{gp}$ | 位势高度 | m | IFS 定义 $H_{gp} = \Phi / g_0$ |
| $H$ | 正高 | m | 相对于大地水准面 |
| $h$ | 椭球高 | m | 相对于 WGS84 椭球面 |

**第一步**：位势转位势高度（ECMWF, 2021）：

$$
H_{gp} = \frac{\Phi}{g_0}
$$

**第二步**：位势高度转正高（Mahoney, 2001）：

$$
H = \frac{R(\varphi) \cdot H_{gp}}{\dfrac{g(\varphi)}{g_0} \cdot R(\varphi) - H_{gp}}
$$

其中：

$$
g(\varphi) = g_e \cdot \frac{1 + k \sin^2\varphi}{\sqrt{1 - e^2 \sin^2\varphi}}, \quad R(\varphi) = \frac{a}{1 + f + m - 2f \sin^2\varphi}
$$

| 常数 | 值 | 描述 | 来源 |
|------|------|------|------|
| $g_0$ | 9.80665 m/s^2 | WMO 标准重力 | ECMWF (2021) |
| $g_e$ | 9.7803253359 m/s^2 | WGS84 赤道重力 | Mahoney (2001) |
| $k$ | 1.931853 x 10^-3 | Somigliana 常数 | Mahoney (2001) |
| $e$ | 0.081819 | WGS84 第一偏心率 | Mahoney (2001) |
| $a$ | 6378137.0 m | WGS84 长半轴 | Mahoney (2001) |
| $f$ | 0.003352811 | WGS84 地球扁率 | Mahoney (2001) |
| $m$ | 0.003449787 | WGS84 重力比 | Mahoney (2001) |

**第三步**：正高转椭球高：

$$
h = H + N
$$

其中 $N$ 为大地水准面差距，使用 EGM96 模型计算，以保持与 ECMWF IFS 模式定义一致。

### 水汽压计算

水汽压 $e$ 由比湿 $q$ 和气压 $p$ 计算：

$$
e = \frac{q \cdot p}{\epsilon + (1 - \epsilon) \cdot q}
$$

其中 $\epsilon = R_d / R_v \approx 0.622$（干空气与水汽气体常数比）。

### 大气折射率

总折射率为静力学和非静力学（湿）分量之和：

$$
N = N_h + N_w
$$

静力学折射率：

$$
N_h = k_1 R_d \rho_m
$$

其中 $\rho_m$ 为湿空气密度。

非静力学（湿）折射率：

$$
N_w = k_2' \frac{e}{T} + k_3 \frac{e}{T^2}, \quad k_2' = k_2 - k_1 \frac{R_d}{R_v}
$$

| 常数 | 默认值 | 单位 | 来源 |
|------|--------|------|------|
| $k_1$ | 77.689 | K/hPa | Rueger (2002) |
| $k_2$ | 71.2952 | K/hPa | Rueger (2002) |
| $k_3$ | 375463 | K^2/hPa | Rueger (2002) |
| $R_d$ | 287.0597 | J/(kg K) | ECMWF (2021) |
| $R_v$ | 461.5250 | J/(kg K) | ECMWF (2021) |

通过 `refractive_index_constants` 参数可传入自定义折射率常数：

```python
# 例：使用 Bevis et al. (1994) 的常数 (k1, k2, k3)
custom = (77.60, 70.40, 373900.0)
zg = ZTDNWMGenerator(nwm_path="data.nc", location=loc, refractive_index_constants=custom)
```

### 数值积分与边界条件

ZTD 由模型层内的数值积分和模型顶层以上的边界条件两部分组成：

$$
\text{ZTD}(h) = 10^{-6} \int_h^{h_{top}} N \, dh + \text{ZTD}_{top}
$$

积分节点可配置：

- `vertical_level = "pressure_level"`：直接使用模型定义的原始气压层作为积分节点。
- `vertical_level = "h"`：先将气象参数重采样到固定间隔的椭球高网格（间隔由 `resample_h` 指定），然后在该网格上执行积分。该模式需执行气象参数垂直插值和外推步骤。

顶层 ZTD 仅考虑静力学延迟 ZHD，采用 Davis 改进的 Saastamoinen 模型：

$$
\text{ZTD}_{top} = \text{ZHD} = \frac{0.0022768 \cdot p}{1 - 0.00266 \cos(2\varphi) - 0.00028 \cdot h \times 10^{-3}}
$$

其中 $p$ 为气压（hPa），$\varphi$ 为纬度（rad），$h$ 为椭球高（m）。

### 气象参数垂直插值和外推

设置 `vertical_level = "h"` 时执行此步骤。将原始气压层上的气象参数 $(T, p, e)$ 重采样到等间距椭球高网格 $h_k$。

**插值**（$h_k \ge h_{bottom}$）：

- 气压 $p$ 和水汽压 $e$ 随高度呈指数变化，采用对数线性插值
- 温度 $T$ 随高度近似线性变化，采用线性插值

**外推**（$h_k < h_{bottom}$，低于模型底层）：

- 水汽压保持定值：

$$
e = e_{bottom}
$$

- 温度采用标准递减率 6.5 K/km（WMO, 2024）：

$$
T = T_{bottom} + 0.0065 \cdot (h_{bottom} - h_k)
$$

- 气压采用虚温气压计公式（WMO, 2024）：

$$
p = p_{bottom} \cdot \exp\left(\frac{g_0 \cdot (h_{bottom} - h_k)}{R_d \cdot T_{mv}}\right)
$$

其中虚温：

$$
T_{mv} = 0.5(T_{bottom} + T) + 0.12 \cdot e_{bottom}
$$

### 对流层延迟垂直插值与外推

数值积分生成所有站点坐标和高度层上的三维 ZTD 数据。设置 `interp_to_site=True` 时将 ZTD 插值或外推至站点高程。

**插值**（站点在相邻高度层之间，$h_{site} \ge h_{bottom}$）：

采用对数线性插值。设 $h_i$ 和 $h_{i+1}$ 为包围站点的相邻高度层：

$$
\ln \text{ZTD}(h_{site}) = \ln \text{ZTD}_i + \frac{h_{site} - h_i}{h_{i+1} - h_i} \cdot \left( \ln \text{ZTD}_{i+1} - \ln \text{ZTD}_i \right)
$$

**外推**（站点低于最低高度层，$h_{site} < h_{bottom}$）：

1. 按上节方法外推底层气象参数到站点高度，计算站点折射率 $N_{site}$。
2. 梯形积分外推层的延迟增量：

$$
\Delta \text{ZTD} = 10^{-6} \cdot \frac{N_{bottom} + N_{site}}{2} \cdot (h_{bottom} - h_{site})
$$

$$
\text{ZTD}_{site} = \text{ZTD}_{bottom} + \Delta \text{ZTD}
$$

---

## 参考文献

- ECMWF (2021). *IFS documentation CY47R3 -- Part IV: Physical processes*. Chapter 12.
- Mahoney, M. J. (2001). A discussion of various measures of altitude. *NASA Jet Propulsion Laboratory*.
- Rueger, J. M. (2002). Refractive index formulae for radio waves. *Proceedings of the FIG XXII International Congress*.
- Davis, J. L., et al. (1985). Geodesy by radio interferometry: Effects of atmospheric modeling errors on estimates of baseline length. *Radio Science*, 20(6).
- World Meteorological Organization. (2024). *Guide to Instruments and Methods of Observation: Volume I -- Measurement of Meteorological Variables* (WMO-No. 8, 2024 ed.). Geneva, Switzerland: WMO.
