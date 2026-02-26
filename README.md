# TROP-NWM

Python framework for computing Zenith Tropospheric Delay (ZTD) from various Numerical Weather Model (NWM) data.

## Features

- **Multi-product & multi-format**
  - Supports deterministic (ERA5, HRES, …) and ensemble (EDA, EPS) ECMWF products
  - Reads both GRIB and NetCDF via xarray IO backends

- **Multiple computation modes**
  - Compute refractivity and ZTD on different vertical coordinates (pressure levels or height levels)
  - Configurable refractivity constants and geoid models

- **Flexible output**
  - Compute ZTD on the native NWM grid or at any given location via physically meaningful interpolation/extrapolation
  - Optionally output full vertical ZTD profiles for in-depth analysis

- **Optimized for performance**
  - Fully vectorized NumPy computation
  - Optional parallel processing for large-scale tasks

## Installation

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -e .
```

## Quick Start

```python
from trop_nwm import ZTDNWMGenerator
import pandas as pd

# Station coordinates
location = pd.DataFrame({
    "site": ["BJFS", "WUHN"],
    "lat": [39.6, 30.5],       # latitude (deg)
    "lon": [115.9, 114.4],     # longitude (deg)
    "alt": [87.4, 25.8],       # WGS84 ellipsoidal height (m)
})

zg = ZTDNWMGenerator(
    nwm_path="era5_pl_native_2023010100.nc",  # NWM file with t (K), z (m2/s2), q (kg/kg)
    location=location,
)
df = zg.run()
print(df)
```

| time | site | ztd (mm) |
|:-----|:-----|:---------|
| 2023-01-01 00:00:00 | BJFS | 2312.456 |
| 2023-01-01 00:00:00 | WUHN | 2398.123 |

For ensemble products (e.g., EDA), the output includes an additional `number` column identifying each ensemble member.

## Documentation

- [TROP-NWM Documents](docs/TROP-NWM%20Documents.md)
- [TROP-NWM 文档](docs/TROP-NWM%20文档.md)
