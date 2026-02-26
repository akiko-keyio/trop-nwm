# TROP-NWM Documents

## Overview

TROP-NWM computes Zenith Tropospheric Delay (ZTD) from three-dimensional numerical weather model (NWM) data. Input meteorological data can be obtained from the [ECMWF Climate Data Store](https://cds.climate.copernicus.eu/). The input file must contain:

- **Temperature** (`t`, K)
- **Geopotential** (`z`, m^2/s^2)
- **Specific humidity** (`q`, kg/kg)

on pressure levels with latitude, longitude, and time dimensions.

---

## ZTDNWMGenerator

### Constructor

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

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nwm_path` | str / Path | required | Path to the NWM meteorological data file |
| `location` | DataFrame | `None` | Station coordinates with `lat`, `lon`, `alt`, `site` columns. If `None`, computes ZTD on the original NWM grid |
| `egm_type` | str | `"egm96-5"` | Geoid model: `"egm96-5"` or `"egm2008-1"` |
| `vertical_level` | str | `"pressure_level"` | Vertical coordinate type (see [Numerical Integration](#numerical-integration-and-boundary-conditions)) |
| `n_jobs` | int | `-1` | Number of parallel workers. `-1` uses all CPUs |
| `batch_size` | int | `100_000` | Threshold for switching between vectorized and parallel processing |
| `horizontal_interpolation_method` | str | `"linear"` | Method passed to `scipy.interpolate.RegularGridInterpolator` |
| `resample_h` | tuple | `(None, None, 50)` | Height resampling parameters `(h_min, h_max, interval)` in meters. `None` values are auto-determined |
| `interp_to_site` | bool | `True` | `True`: output station ZTD. `False`: output full vertical profile |
| `refractive_index_constants` | tuple | `(77.689, 71.2952, 375463.0)` | Custom refractivity constants `(k1, k2, k3)`. Default from Rueger (2002) |
| `progress_mode` | str | `"rich"` | Progress display: `"rich"` (spinners/bars) or `"simple"` (logger.info) |
| `time_batch_size` | int / None | `None` | If set, splits the time dimension into batches of this size to reduce peak memory |

### run()

Execute the ZTD computation pipeline.

**Returns** a `DataFrame` with columns:

| Column | Type | Description |
|--------|------|-------------|
| `time` | datetime64 | Timestamp |
| `site` | str | Station identifier |
| `ztd` | float | Zenith tropospheric delay (mm) |

When `interp_to_site=False`, an additional `h` column (height in meters) is included.

For ensemble products (e.g., EDA with multiple members), the output includes an additional `number` column identifying each ensemble member. 

**Raises** `ValueError` if the final result contains NaN values.

---

## Computational Methodology

### Height System Conversion

NWM data provides geopotential, which must be converted to ellipsoidal height. The conversion path is:

$$
\Phi \text{ (geopotential)} \to H_{gp} \text{ (geopotential height)} \to H \text{ (orthometric height)} \to h \text{ (ellipsoidal height)}
$$

| Symbol | Name | Unit | Description |
|--------|------|------|-------------|
| $\Phi$ | Geopotential | m^2/s^2 | NWM raw variable `z` |
| $H_{gp}$ | Geopotential height | m | IFS definition: $H_{gp} = \Phi / g_0$ |
| $H$ | Orthometric height | m | Height above the geoid |
| $h$ | Ellipsoidal height | m | Height above the WGS84 ellipsoid |

**Step 1**: Geopotential to geopotential height (ECMWF, 2021):

$$
H_{gp} = \frac{\Phi}{g_0}
$$

**Step 2**: Geopotential height to orthometric height (Mahoney, 2001):

$$
H = \frac{R(\varphi) \cdot H_{gp}}{\dfrac{g(\varphi)}{g_0} \cdot R(\varphi) - H_{gp}}
$$

where:

$$
g(\varphi) = g_e \cdot \frac{1 + k \sin^2\varphi}{\sqrt{1 - e^2 \sin^2\varphi}}, \quad R(\varphi) = \frac{a}{1 + f + m - 2f \sin^2\varphi}
$$

| Constant | Value | Description | Source |
|----------|-------|-------------|--------|
| $g_0$ | 9.80665 m/s^2 | WMO standard gravity | ECMWF (2021) |
| $g_e$ | 9.7803253359 m/s^2 | WGS84 equatorial gravity | Mahoney (2001) |
| $k$ | 1.931853 x 10^-3 | Somigliana constant | Mahoney (2001) |
| $e$ | 0.081819 | WGS84 first eccentricity | Mahoney (2001) |
| $a$ | 6378137.0 m | WGS84 semi-major axis | Mahoney (2001) |
| $f$ | 0.003352811 | WGS84 flattening | Mahoney (2001) |
| $m$ | 0.003449787 | WGS84 gravity ratio | Mahoney (2001) |

**Step 3**: Orthometric to ellipsoidal height:

$$
h = H + N
$$

where $N$ is the geoid undulation from the EGM96 model, consistent with the ECMWF IFS model definition.

### Water Vapor Pressure

Water vapor pressure $e$ is computed from specific humidity $q$ and pressure $p$:

$$
e = \frac{q \cdot p}{\epsilon + (1 - \epsilon) \cdot q}
$$

where $\epsilon = R_d / R_v \approx 0.622$ is the ratio of dry air to water vapor gas constants.

### Atmospheric Refractivity

Total refractivity is the sum of hydrostatic and non-hydrostatic (wet) components:

$$
N = N_h + N_w
$$

Hydrostatic refractivity:

$$
N_h = k_1 R_d \rho_m
$$

where $\rho_m$ is the moist air density.

Non-hydrostatic (wet) refractivity:

$$
N_w = k_2' \frac{e}{T} + k_3 \frac{e}{T^2}, \quad k_2' = k_2 - k_1 \frac{R_d}{R_v}
$$

| Constant | Default Value | Unit | Source |
|----------|---------------|------|--------|
| $k_1$ | 77.689 | K/hPa | Rueger (2002) |
| $k_2$ | 71.2952 | K/hPa | Rueger (2002) |
| $k_3$ | 375463 | K^2/hPa | Rueger (2002) |
| $R_d$ | 287.0597 | J/(kg K) | ECMWF (2021) |
| $R_v$ | 461.5250 | J/(kg K) | ECMWF (2021) |

Custom constants can be passed via the `refractive_index_constants` parameter:

```python
# Example: Bevis et al. (1994) constants (k1, k2, k3)
custom = (77.60, 70.40, 373900.0)
zg = ZTDNWMGenerator(nwm_path="data.nc", location=loc, refractive_index_constants=custom)
```

### Numerical Integration and Boundary Conditions

ZTD is the sum of numerical integration through model layers and a top-level boundary condition:

$$
\text{ZTD}(h) = 10^{-6} \int_h^{h_{top}} N \, dh + \text{ZTD}_{top}
$$

The integration grid depends on the `vertical_level` setting:

- `"pressure_level"`: Uses the original model pressure levels as integration nodes directly.
- `"h"`: Resamples meteorological parameters to a uniform ellipsoidal height grid (with `resample_h` spacing) before integration. This mode requires vertical interpolation and extrapolation of meteorological parameters.

The top-level ZTD uses the Davis-modified Saastamoinen model for hydrostatic delay:

$$
\text{ZTD}_{top} = \text{ZHD} = \frac{0.0022768 \cdot p}{1 - 0.00266 \cos(2\varphi) - 0.00028 \cdot h \times 10^{-3}}
$$

where $p$ is pressure (hPa), $\varphi$ is latitude (rad), and $h$ is ellipsoidal height (m).

### Meteorological Parameter Extrapolation

When `vertical_level="h"`, meteorological parameters $(T, p, e)$ are resampled from the original pressure levels to a uniform height grid $h_k$.

**Interpolation** ($h_k \ge h_{bottom}$):

- Pressure $p$ and water vapor pressure $e$: log-linear interpolation (exponential height dependence)
- Temperature $T$: linear interpolation

**Extrapolation** ($h_k < h_{bottom}$, below model bottom):

- Water vapor pressure: held constant at bottom-level value

$$
e = e_{bottom}
$$

- Temperature: standard lapse rate of 6.5 K/km (WMO, 2024)

$$
T = T_{bottom} + 0.0065 \cdot (h_{bottom} - h_k)
$$

- Pressure: barometric formula with virtual temperature (WMO, 2024)

$$
p = p_{bottom} \cdot \exp\left(\frac{g_0 \cdot (h_{bottom} - h_k)}{R_d \cdot T_{mv}}\right)
$$

where the mean virtual temperature is:

$$
T_{mv} = 0.5(T_{bottom} + T) + 0.12 \cdot e_{bottom}
$$

### ZTD Vertical Interpolation and Extrapolation

After numerical integration, a three-dimensional ZTD field is available at all station coordinates and height levels. When `interp_to_site=True`, ZTD is interpolated or extrapolated to exact station altitudes.

**Interpolation** (station within model grid, $h_{site} \ge h_{bottom}$):

Log-linear interpolation between bounding height levels $h_i$ and $h_{i+1}$:

$$
\ln \text{ZTD}(h_{site}) = \ln \text{ZTD}_i + \frac{h_{site} - h_i}{h_{i+1} - h_i} \cdot \left( \ln \text{ZTD}_{i+1} - \ln \text{ZTD}_i \right)
$$

**Extrapolation** (station below model bottom, $h_{site} < h_{bottom}$):

1. Extrapolate bottom-level meteorological parameters to site height using WMO guidelines (see above).
2. Compute site-level refractivity $N_{site}$.
3. Add trapezoidal delay increment:

$$
\Delta \text{ZTD} = 10^{-6} \cdot \frac{N_{bottom} + N_{site}}{2} \cdot (h_{bottom} - h_{site})
$$

$$
\text{ZTD}_{site} = \text{ZTD}_{bottom} + \Delta \text{ZTD}
$$

---

## References

- ECMWF (2021). *IFS documentation CY47R3 -- Part IV: Physical processes*. Chapter 12.
- Mahoney, M. J. (2001). A discussion of various measures of altitude. *NASA Jet Propulsion Laboratory*.
- Rueger, J. M. (2002). Refractive index formulae for radio waves. *Proceedings of the FIG XXII International Congress*.
- Davis, J. L., et al. (1985). Geodesy by radio interferometry: Effects of atmospheric modeling errors on estimates of baseline length. *Radio Science*, 20(6).
- World Meteorological Organization. (2024). *Guide to Instruments and Methods of Observation: Volume I -- Measurement of Meteorological Variables* (WMO-No. 8, 2024 ed.). Geneva, Switzerland: WMO.
