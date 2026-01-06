from __future__ import annotations


from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy.integrate import cumulative_simpson
from scipy.interpolate import (
    RegularGridInterpolator,
    interp1d,
)

from trop_nwm.geoid import GeoidHeight
from trop_nwm.log_utils import logger, track_step, joblib_rich_progress


# ============================================================================
# Physical Constants
# ============================================================================
# References:
#   ECMWF (2021). IFS documentation CY47R3 – Part IV: Physical processes. Chapter 12.
#   Mahoney, M. J. (2001). A discussion of various measures of altitude. NASA JPL.

# WGS84 / ECMWF constants for height conversion
_G0 = 9.80665  # m/s², WMO standard gravity
_GE = 9.7803253359  # m/s², WGS84 equatorial gravity
_K = 1.931853e-3  # Somigliana constant
_E = 0.081819  # WGS84 first eccentricity
_A = 6378137.0  # m, WGS84 semi-major axis
_F = 0.003352811  # WGS84 flattening
_M = 0.003449787  # WGS84 gravity ratio

# Gas constants for moist air thermodynamics
_RD = 287.0597  # J/(kg·K), dry air gas constant
_RV = 461.5250  # J/(kg·K), water vapor gas constant


# ============================================================================
# Height Conversion Functions
# ============================================================================
def geopotential_to_orthometric(latitude, geopotential_height):
    """Convert geopotential height to orthometric height.

    Implements the conversion formula from ECMWF IFS documentation:
        H = (R(φ) · H_gp) / ((g(φ)/g₀) · R(φ) - H_gp)

    Parameters
    ----------
    latitude : array_like
        Latitude in degrees.
    geopotential_height : array_like
        Geopotential height in meters (H_gp = Φ / g₀).

    Returns
    -------
    H : array_like
        Geometric (orthometric) height in meters.
    """
    lat_rad = np.deg2rad(latitude)
    sin2_lat = np.sin(lat_rad) ** 2
    e2 = _E**2

    # Gravity at latitude: g(φ) = g_e · (1 + k·sin²φ) / √(1 - e²·sin²φ)
    g_phi = _GE * (1 + _K * sin2_lat) / np.sqrt(1 - e2 * sin2_lat)

    # Effective Earth radius: R(φ) = a / (1 + f + m - 2f·sin²φ)
    r_phi = _A / (1 + _F + _M - 2 * _F * sin2_lat)

    # Convert: H = (R·H_gp) / ((g/g₀)·R - H_gp)
    g_ratio = g_phi / _G0
    H = (r_phi * geopotential_height) / (g_ratio * r_phi - geopotential_height)

    return H


# ============================================================================
# Meteorological Extrapolation Functions
# ============================================================================


def extrapolate_met_parameters(
    p_bottom: np.ndarray | float,
    t_bottom: np.ndarray | float,
    e_bottom: np.ndarray | float,
    h_bottom: np.ndarray | float,
    h_site: np.ndarray | float,
) -> Tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Extrapolate met parameters below model grid using WMO No.8 guidelines."""
    dh = h_bottom - h_site

    e_site = e_bottom  # Constant assumption
    t_site = t_bottom + 0.0065 * dh  # Standard lapse rate

    # Barometric formula with virtual temperature
    t_mean = 0.5 * (t_bottom + t_site)
    t_mv = t_mean + 0.12 * e_bottom
    p_site = p_bottom * np.exp((_G0 * dh) / (_RD * t_mv))

    return p_site, t_site, e_site


def calculate_refractivity(
    p: np.ndarray | float,
    e: np.ndarray | float,
    t: np.ndarray | float,
    refractive_index_constants: tuple[float, float, float] = (
        77.689,
        71.2952,
        375463.0,
    ),
) -> Tuple[np.ndarray | float, np.ndarray | float]:
    """Calculate atmospheric refractivity components N_h and N_w.

    Uses hydrostatic/wet separation with density formulation.
    Total N = N_h + N_w equals the Smith-Weintraub formula.

    Parameters
    ----------
    p : array or float
        Total pressure (hPa)
    e : array or float
        Water vapor pressure (hPa)
    t : array or float
        Temperature (K)
    refractive_index_constants : tuple, default (77.689, 71.2952, 375463.0)
        Refractive index constants (k1, k2, k3) with units (K/hPa, K/hPa, K²/hPa)

    Returns
    -------
    n_h : array or float
        Hydrostatic refractivity (dimensionless N-units)
    n_w : array or float
        Non-hydrostatic (Wet) refractivity (dimensionless N-units)
    """
    k1, k2, k3 = refractive_index_constants
    p_d = p - e  # Dry air pressure
    k2_prime = k2 - k1 * _RD / _RV

    # Hydrostatic refractivity: N_h = k1*p_d/T + k1*(R_d/R_v)*e/T
    n_h = k1 * p_d / t + k1 * (_RD / _RV) * e / t

    # Non-hydrostatic (Wet) refractivity: N_w = k2'*e/T + k3*e/T²
    n_w = k2_prime * e / t + k3 * e / (t**2)

    return n_h, n_w


def calc_wmo_ztd_extrapolation(
    h_bottom: np.ndarray | float,
    ztd_bottom: np.ndarray | float,
    p_bottom: np.ndarray | float,
    t_bottom: np.ndarray | float,
    e_bottom: np.ndarray | float,
    h_site: np.ndarray | float,
    refractive_index_constants: tuple[float, float, float] = (
        77.689,
        71.2952,
        375463.0,
    ),
) -> np.ndarray | float:
    """Compute ZTD at site below model grid using WMO physical extrapolation.

    Parameters
    ----------
    h_bottom : array or float
        Height at model bottom level (m)
    ztd_bottom : array or float
        ZTD at model bottom level (m)
    p_bottom : array or float
        Pressure at model bottom level (hPa)
    t_bottom : array or float
        Temperature at model bottom level (K)
    e_bottom : array or float
        Water vapor pressure at model bottom level (hPa)
    h_site : array or float
        Site height (m)
    refractive_index_constants : tuple, default (77.689, 71.2952, 375463.0)
        Refractive index constants (k1, k2, k3) with units (K/hPa, K/hPa, K²/hPa)

    Returns
    -------
    ztd_site : array or float
        ZTD at site height (m)
    """
    p_site, t_site, e_site = extrapolate_met_parameters(
        p_bottom, t_bottom, e_bottom, h_bottom, h_site
    )

    n_h_bottom, n_w_bottom = calculate_refractivity(
        p_bottom, e_bottom, t_bottom, refractive_index_constants
    )
    n_h_site, n_w_site = calculate_refractivity(
        p_site, e_site, t_site, refractive_index_constants
    )

    n_bottom = n_h_bottom + n_w_bottom
    n_site = n_h_site + n_w_site

    # Trapezoidal integration of layer delay
    dh = h_bottom - h_site
    delta_ztd = 1e-6 * 0.5 * (n_bottom + n_site) * dh
    return ztd_bottom + delta_ztd


class ZTDNWMGenerator:
    """Generator for Zenith Tropospheric Delay from Numerical Weather Model data."""

    def __init__(
        self,
        nwm_path: str | Path,
        location: pd.DataFrame | xr.Dataset | None = None,
        egm_type: str = "egm96-5",
        vertical_level: str = "pressure_level",
        n_jobs: int = -1,
        batch_size: int = 100_000,
        horizontal_interpolation_method: str = "linear",
        resample_h: tuple = (None, None, 50),
        interp_to_site=True,
        refractive_index_constants: tuple[float, float, float] = (
            77.689,
            71.2952,
            375463.0,
        ),
        progress_mode: str = "rich",
    ):
        """Initialize the ZTD generator.

        Parameters
        ----------
        nwm_path : str or Path
            Path to the NWM/ERA5 NetCDF or GRIB file containing meteorological data
            (temperature, geopotential, specific humidity).
        location : pd.DataFrame or xr.Dataset, optional
            DataFrame or Dataset containing station coordinates with variables 'lat', 'lon', 'alt',
            and 'site'. If None, computes ZTD on the original NWM grid.
        egm_type : str, default "egm96-5"
            Geoid model for orthometric to ellipsoidal height conversion.
            Options: "egm96-5", "egm2008-1".
        vertical_level : str, default "pressure_level"
            Vertical coordinate type. Use "pressure_level" for pressure-level data
            or "h" for height-resampled data.
        n_jobs : int, default -1
            Number of parallel jobs for heavy computations. -1 uses all CPUs.
        batch_size : int, default 100_000
            Threshold for switching between vectorized and parallel processing.
            Tasks with fewer elements use vectorized operations.
        horizontal_interpolation_method : str, default "linear"
            Interpolation method for horizontal fields passed to scipy.interpolate.RegularGridInterpolator.
            e.g. "linear", "nearest", "slinear", "cubic", "quintic", "pchip".
        resample_h : tuple, default (None, None, 50)
            Height resampling parameters (h_min, h_max, interval) in meters.
            None values are auto-determined from data.
        interp_to_site : bool, default True
            If True, interpolate ZTD to exact station altitudes.
            If False, return ZTD on the vertical grid.
        refractive_index_constants : tuple, default (77.689, 71.2952, 375463.0)
            Custom refractivity constants (k1, k2, k3).
            Default values from Rüeger (2002).
        progress_mode : str, default "rich"
            Progress display mode. Options: "rich" (fancy progress bars) or "simple" (logger.info).
            Use "simple" to avoid conflicts with other frameworks that use rich.
        """
        # Set progress mode
        from trop_nwm.log_utils import set_progress_mode
        set_progress_mode(progress_mode)
        
        self.nwm_path = Path(nwm_path)
        self.location = location.copy() if location is not None else None
        self.egm_type = egm_type
        self.vertical_dimension = vertical_level
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.horizontal_interpolation_method = horizontal_interpolation_method
        self.resample_h = resample_h
        self.interp_to_site = interp_to_site
        self.refractive_index_constants = refractive_index_constants

        self.ds: xr.Dataset | None = None
        self.top_level: xr.Dataset | None = None

    @track_step("1/10: Load and Format dataset")
    def read_met_file(self) -> None:
        """Load NWM file, verify variables, and adjust dimensions."""
        try:
            self.ds = xr.load_dataset(self.nwm_path)
        except Exception as e:
            logger.error(f"Error Loading {self.nwm_path} For {e}")
            raise
        required = ["z", "t", "q"]
        for var in required:
            if var not in self.ds:
                raise ValueError(f"Required variable '{var}' not found in dataset")

        rename_map = {
            "level": "pressure_level",
            "isobaricInhPa": "pressure_level",
            "valid_time": "time",
            "longitude": "lon",
            "latitude": "lat",
        }
        exist = {k: v for k, v in rename_map.items() if k in self.ds}
        if exist:
            self.ds = self.ds.rename(exist)
        if "time" not in self.ds.dims:
            self.ds = self.ds.expand_dims("time")
        if "number" not in self.ds.dims:
            self.ds = self.ds.expand_dims("number")
        if "lon" in self.ds.data_vars:
            self.ds = self.ds.drop_vars("lon")
        if "pressure_level" in self.ds.dims:
            self.ds["p"] = self.ds.pressure_level

    @track_step("2/10: Interpolate horizontally")
    def horizontal_interpolate(self) -> None:
        """Interpolate NWM fields to station coordinates."""
        method = self.horizontal_interpolation_method

        # No target sites: stack original grid
        if self.location is None:
            self.ds = self.ds.stack(site_index=("lat", "lon"))
            self.ds["site"] = self.ds.site_index
            return

        ds, loc = self.ds, self.location
        lat_grid = ds.lat.values
        lon_grid = ds.lon.values
        pts_lat = loc["lat"].values
        pts_lon_orig = loc["lon"].values

        # Convert location longitude to match data range [0, 360) if needed
        if lon_grid.max() > 180:
            pts_lon = pts_lon_orig % 360
        else:
            pts_lon = pts_lon_orig

        query_pts = np.column_stack((pts_lat, pts_lon))

        new_vars = {}

        for vn, da in ds.data_vars.items():
            dims = da.dims

            if {"lat", "lon"} <= set(dims):
                lat_ax, lon_ax = dims.index("lat"), dims.index("lon")
                other_axes = [i for i in range(da.ndim) if i not in (lat_ax, lon_ax)]

                # Move lat/lon to front for RegularGridInterpolator
                arr2 = np.moveaxis(
                    da.values,
                    (lat_ax, lon_ax) + tuple(other_axes),
                    (0, 1) + tuple(2 + np.arange(len(other_axes))),
                )
                other_shape = arr2.shape[2:]

                # Main interpolation on the grid
                interp = RegularGridInterpolator(
                    (lat_grid, lon_grid),
                    arr2,
                    method=method,
                    bounds_error=False,
                    fill_value=np.nan,
                )
                res = interp(query_pts)

                # Patch boundary values (e.g. crossing 360/0 degree line)
                # If points are outside the grid longitude range, interpolate using wrap-around
                gap_mask = pts_lon > lon_grid[-1]
                if np.any(gap_mask):
                    b_lons = np.array([lon_grid[-1], lon_grid[0] + 360])
                    b_data = np.stack([arr2[:, -1, ...], arr2[:, 0, ...]], axis=1)

                    b_interp = RegularGridInterpolator(
                        (lat_grid, b_lons),
                        b_data,
                        method="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    res[gap_mask] = b_interp(query_pts[gap_mask])

                if res.ndim > 1:
                    res2 = res.reshape(len(pts_lat), *other_shape)
                    res2 = np.moveaxis(res2, range(1, 1 + len(other_axes)), other_axes)
                else:
                    res2 = res[:, None]

                new_dims = tuple(d for d in dims if d not in ("lat", "lon")) + (
                    "site_index",
                )
                coords = {d: ds.coords[d] for d in new_dims if d != "site_index"}
                coords["site_index"] = loc.index
                new_vars[vn] = xr.DataArray(res2, dims=new_dims, coords=coords, name=vn)
            else:
                new_vars[vn] = da

        ds2 = xr.Dataset(new_vars)
        ds2["lat"] = ("site_index", pts_lat)
        ds2["lon"] = ("site_index", pts_lon_orig)
        ds2["alt"] = ("site_index", loc["alt"].values)
        ds2["site"] = ("site_index", loc["site"].values)
        ds2.coords["alt"] = ds2.alt

        self.ds = ds2

    @track_step("3/10: Convert Geopotential -> Orthometric")
    def geopotential_to_orthometric(self) -> None:
        """Convert geopotential (m^2/s^2) to orthometric height (m)."""
        geop_height = self.ds.z / _G0  # m^2/s^2 / (m/s^2) = m
        lat_vals = self.ds.lat.values
        if lat_vals.ndim == 1:
            lat_vals = np.expand_dims(lat_vals, axis=(0, 1))

        self.ds["h"] = geopotential_to_orthometric(
            latitude=lat_vals, geopotential_height=geop_height
        )  # h in meters

    @track_step("4/10: Convert Orthometric -> Ellipsoidal")
    def orthometric_to_ellipsoidal(self) -> None:
        """Add geoid undulation to convert orthometric to ellipsoidal height (m)."""
        geoid = GeoidHeight(egm_type=self.egm_type)
        anomaly = np.array(
            [
                geoid.get(float(la), float(lo))
                for la, lo in zip(self.ds.lat.values, self.ds.lon.values)
            ]
        )  # meters
        anom_da = xr.DataArray(
            anomaly, dims=("site_index",), coords={"site_index": self.ds.site_index}
        )
        self.ds["h"] = self.ds["h"] + anom_da  # still in meters

    @track_step("5/10: Compute water-vapor pressure")
    def compute_e(self) -> None:
        """Compute water vapor pressure (e) from specific humidity (q) and pressure (p).

        Uses the more accurate physical relationship: e = q * p / (0.622 + 0.378 * q)
        """
        epsilon = _RD / _RV
        self.ds["e"] = (self.ds.q * self.ds.p) / (epsilon + (1 - epsilon) * self.ds.q)

    def resample_to_ellipsoidal(self) -> None:
        if self.vertical_dimension == "pressure_level":
            return

        ds, loc = self.ds, self.location
        h_min_cfg, h_max_cfg, interval_cfg = self.resample_h

        if h_min_cfg is None:
            h_min = float(self.ds.alt.min())
        elif isinstance(h_min_cfg, (int, float)):
            h_min = h_min_cfg
        else:
            raise ValueError("resample_h_min (resample_h[0]) must be float or None")

        if h_max_cfg is None:
            h_max = float(
                ds.sel(
                    pressure_level=(ds.pressure_level[ds.pressure_level > 0]).min()
                ).h.min()
            )
        elif isinstance(h_max_cfg, (int, float)):
            h_max = h_max_cfg
        else:
            raise ValueError("resample_h_max (resample_h[1]) must be float or None")

        if isinstance(interval_cfg, (int, float)):
            interval = interval_cfg
        else:
            raise ValueError("resample_h_interval (resample_h[2]) must be int or float")

        target_h_vals = np.arange(h_min, h_max, interval)  # meters

        # Create output dataset with number, site_index, time and h
        ds_new = xr.Dataset(
            coords={
                "number": ds.number,
                "site_index": ds.site_index,
                "time": ds.time,
                "h": target_h_vals,
            }
        )

        def _interp_profile(n_idx, site_idx, time0):
            # Select profile (retains vertical dimension 'pressure_level')
            profile = ds.sel(number=n_idx, site_index=site_idx, time=time0)

            h_src = np.squeeze(profile["h"].data)
            p_src = np.squeeze(profile["p"].data)  # hPa
            t_src = np.squeeze(profile["t"].data)  # K
            e_src = np.squeeze(profile["e"].data)  # hPa

            # Helper to remove NaNs
            valid_mask = np.isfinite(h_src) & np.isfinite(p_src)
            if not np.any(valid_mask):
                nan_arr = np.full_like(target_h_vals, np.nan)
                return (n_idx, site_idx, time0, nan_arr, nan_arr, nan_arr)

            h_src = h_src[valid_mask]
            p_src = p_src[valid_mask]
            t_src = t_src[valid_mask]
            e_src = e_src[valid_mask]

            # Sort by height for interpolation (ascending)
            sort_idx = np.argsort(h_src)
            h_s = h_src[sort_idx]
            p_s = p_src[sort_idx]
            t_s = t_src[sort_idx]
            e_s = e_src[sort_idx]

            # Determine interpolation zones (below vs inside/above model grid)
            h1 = h_s[0]
            mask_extrap = target_h_vals < h1
            mask_interp = ~mask_extrap

            # Prepare Result Arrays
            p_res = np.empty_like(target_h_vals)
            t_res = np.empty_like(target_h_vals)
            e_res = np.empty_like(target_h_vals)

            # Interpolation (within model bounds)
            if np.any(mask_interp):
                h_in = target_h_vals[mask_interp]

                # T: Linear
                fn_t = interp1d(h_s, t_s, kind="linear", fill_value="extrapolate")
                t_res[mask_interp] = fn_t(h_in)

                # P: Log-Linear
                log_p_s = np.log(np.maximum(p_s, 1e-10))
                fn_p = interp1d(h_s, log_p_s, kind="linear", fill_value="extrapolate")
                p_res[mask_interp] = np.exp(fn_p(h_in))

                # e: Log-Linear
                log_e_s = np.log(np.maximum(e_s, 1e-10))
                fn_e = interp1d(h_s, log_e_s, kind="linear", fill_value="extrapolate")
                e_res[mask_interp] = np.exp(fn_e(h_in))

            # Extrapolation (below bottom level)
            if np.any(mask_extrap):
                h_ex = target_h_vals[mask_extrap]

                # Use values at lowest model level as base
                p1 = p_s[0]
                t1 = t_s[0]
                e1 = e_s[0]

                # Call helper function for consistent physics
                p2, t2, e2 = extrapolate_met_parameters(p1, t1, e1, h1, h_ex)

                # Store results
                e_res[mask_extrap] = e2
                t_res[mask_extrap] = t2
                p_res[mask_extrap] = p2

            return (n_idx, site_idx, time0, p_res, t_res, e_res)

        # Resample all profiles in parallel using joblib
        # We iterate over profiles, not parameters, and compute all 3 params at once
        n_tasks = len(ds.number) * len(ds.site_index) * len(ds.time)
        with joblib_rich_progress("6/10: Resample to Ellipsoidal (WMO)", total=n_tasks):
            results = Parallel(n_jobs=self.n_jobs, batch_size=self.batch_size)(
                delayed(_interp_profile)(n, s, t)
                for n in ds.number.values
                for s in ds.site_index.values
                for t in ds.time.values
            )

        # Unpack results into buffers
        shape = (len(ds.number), len(ds.site_index), len(ds.time), len(target_h_vals))
        buf_p = np.full(shape, np.nan)
        buf_t = np.full(shape, np.nan)
        buf_e = np.full(shape, np.nan)

        n_map = {n: i for i, n in enumerate(ds.number.values)}
        s_map = {s: i for i, s in enumerate(ds.site_index.values)}
        t_map = {t: i for i, t in enumerate(ds.time.values)}

        for n_idx, s_idx, t_idx, p_arr, t_arr, e_arr in results:
            ni, si, ti = n_map[n_idx], s_map[s_idx], t_map[t_idx]
            buf_p[ni, si, ti, :] = p_arr
            buf_t[ni, si, ti, :] = t_arr
            buf_e[ni, si, ti, :] = e_arr

        # Create DataArrays
        dims = ("number", "site_index", "time", "h")
        coords = {
            "number": ds.number,
            "site_index": ds.site_index,
            "time": ds.time,
            "h": target_h_vals,
        }

        ds_new["p"] = xr.DataArray(buf_p, dims=dims, coords=coords)
        ds_new["p"].attrs["units"] = "hPa"

        ds_new["t"] = xr.DataArray(buf_t, dims=dims, coords=coords)
        ds_new["t"].attrs["units"] = "K"

        ds_new["e"] = xr.DataArray(buf_e, dims=dims, coords=coords)
        ds_new["e"].attrs["units"] = "hPa"

        # Copy site information and sort by height for consistency
        ds_new["lon"], ds_new["lat"], ds_new["site"] = ds.lon, ds.lat, ds.site
        self.ds = ds_new.sortby("h", ascending=False)

    @track_step("7/10: Compute refractive index")
    def compute_refractive_index(self) -> None:
        """Compute refractive indices N = N_h + N_w.

        Calls the module-level calculate_refractivity function.
        Input units: p in hPa, t in K, e in hPa
        Output: n, n_h, n_w (dimensionless refractivity N)
        """
        ds = self.ds
        t = ds.t  # K
        e = ds.e  # hPa
        p = ds.p  # hPa

        # Broadcast p to match t shape if needed
        if p.dims != t.dims:
            p, _ = xr.broadcast(p, t)

        n_h, n_w = calculate_refractivity(p, e, t, self.refractive_index_constants)

        self.ds["n"] = n_h + n_w
        self.ds["n_h"] = n_h
        self.ds["n_w"] = n_w

    @track_step("8/10: Compute top-level delays")
    def compute_top_level_delay(self) -> None:
        """Compute ZHD at top model level using Saastamoinen model (boundary condition).

        Input: p (hPa), lat (deg), h (m)
        Output: zhd, zwd, ztd (m)
        """

        def zhd_saastamoinen(p, lat, h):
            """Saastamoinen-Davis model: p (hPa), lat (deg), h (m) -> zhd (m)"""
            dlat = np.radians(lat)
            return (
                0.0022768
                * p
                / (1.0 - 0.00266 * np.cos(2 * dlat) - 0.00028 * h * 1.0e-03)
            )

        ds = self.ds
        top = (
            ds.sel(pressure_level=ds.pressure_level[ds.pressure_level > 0].min())
            if self.vertical_dimension == "pressure_level"
            else ds.sel(h=ds.h.max())
        )
        top = top.transpose("number", "time", "site_index")

        # ZTD_top = ZHD (no wet delay at top boundary)
        zhd_values = zhd_saastamoinen(
            p=top.p.values,
            lat=top.lat.values,
            h=top.h.values,
        )  # meters
        top["zhd"] = (("number", "time", "site_index"), zhd_values)
        top["zwd"] = xr.zeros_like(top["zhd"])
        top["ztd"] = top.zhd
        self.top_level = top

    @track_step("9/10: Perform Simpson integration")
    def simpson_numerical_integration(self) -> None:
        """Integrate refractivity from top to each level using Simpson's rule.

        Output: ztd_simpson, zhd_simpson, zwd_simpson (m)
        """
        ds = (
            self.ds.sortby("pressure_level")
            if self.vertical_dimension == "pressure_level"
            else self.ds
        )

        # Reorder dimensions for integration
        dim_order = ("number", "time", "site_index", self.vertical_dimension)

        # Handle h: when vertical_level="h", h is a 1D coordinate; otherwise it's a variable
        if self.vertical_dimension == "h":
            # h is a 1D coordinate, broadcast to match other variables
            h_vals = ds.h.values  # 1D array
            # Get shape from a variable that has full dims
            ref_var = ds["n"].transpose(*dim_order)
            h = np.broadcast_to(h_vals, ref_var.shape)
        else:
            h = ds.h.transpose(*dim_order).values

        x = -h  # Negative for downward integration

        for zxd, n in [("ztd", "n"), ("zwd", "n_w"), ("zhd", "n_h")]:
            y = (
                ds[n].transpose(*dim_order).values * 1e-6
            )  # refractivity -> delay factor
            val = cumulative_simpson(y=y, x=x, axis=-1, initial=0)
            # Add top-level delay (broadcast from [n,t,s] to [n,t,s,level])
            top_val = self.top_level[zxd].values[..., np.newaxis]
            ds[f"{zxd}_simpson"] = (dim_order, val + top_val)

        self.ds = ds

    def _vertical_interp_vectorized(self, ds, alt_da, log_cubic_interp) -> xr.Dataset:
        """Vectorized vertical interpolation for small workloads."""
        result = xr.apply_ufunc(
            log_cubic_interp,
            ds.h,
            ds.ztd_simpson,
            alt_da,
            input_core_dims=[
                [self.vertical_dimension],
                [self.vertical_dimension],
                [],
            ],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        da = result.expand_dims(h=[0]).transpose("number", "site_index", "time", "h")
        return xr.Dataset({"ztd_simpson": da})

    def _vertical_interp_parallel(
        self,
        ds,
        h_all,
        ztd_all,
        alt,
        log_cubic_interp,
        num_dim,
        time_dim,
        site_dim,
        total_tasks,
    ) -> xr.Dataset:
        """Parallel vertical interpolation for heavy workloads."""
        out = np.empty((num_dim, time_dim, site_dim), dtype=float)

        def _interp_one(n, t, s):
            return log_cubic_interp(h_all[n, t, s], ztd_all[n, t, s], alt[s])

        with joblib_rich_progress("10/10: Vertical Interpolation", total=total_tasks):
            flat = Parallel(n_jobs=self.n_jobs, batch_size=self.batch_size)(
                delayed(_interp_one)(n, t, s)
                for n in range(num_dim)
                for t in range(time_dim)
                for s in range(site_dim)
            )
        out[:] = np.array(flat).reshape(num_dim, time_dim, site_dim)
        da = xr.DataArray(
            out,
            dims=("number", "time", "site_index"),
            coords={
                "number": ds.number,
                "time": ds.time,
                "site_index": ds.site_index,
            },
        ).transpose("number", "site_index", "time")
        return xr.Dataset({"ztd_simpson": da})

    @track_step("10/10: Interpolate to site height")
    def vertical_interpolate_to_site(self) -> xr.Dataset:
        ds = self.ds

        def log_linear_interp(
            x: np.ndarray, y: np.ndarray, xnew: np.ndarray
        ) -> np.ndarray:
            """Log-linear interpolation assuming the input is sorted."""
            if x[0] > x[-1]:
                x, y = x[::-1], y[::-1]
            mask = np.isfinite(y)
            x_filt, y_filt = x[mask], y[mask]
            if x_filt.size < 2:
                raise ValueError(
                    f"Not enough valid points: only {x_filt.size} available"
                )
            log_y = np.log(np.maximum(y_filt, 1e-12))
            return np.exp(np.interp(xnew, x_filt, log_y))

        # Broadcast h and ztd_simpson when vertical_dimension is h
        if self.vertical_dimension == "h":
            h_da, ztd_da = xr.broadcast(ds.h, ds.ztd_simpson)
        else:
            h_da = ds.h
            ztd_da = ds.ztd_simpson

        h_all = h_da.transpose(
            "number", "time", "site_index", self.vertical_dimension
        ).values
        ztd_all = ztd_da.transpose(
            "number", "time", "site_index", self.vertical_dimension
        ).values
        alt = ds.alt.values
        num_dim, time_dim, site_dim, _ = ztd_all.shape
        total_tasks = num_dim * time_dim * site_dim

        # Pressure Level Mode: Hybrid Vectorized Scheme
        # Uses Log-Linear Interpolation for sites within grid and WMO Physics Extrapolation for sites below grid
        if self.vertical_dimension == "pressure_level":
            p_values = ds.pressure_level.values
            t_all = ds.t.transpose(
                "number", "time", "site_index", "pressure_level"
            ).values
            e_all = ds.e.transpose(
                "number", "time", "site_index", "pressure_level"
            ).values

            # Compute Bottom Heights (vectorized)
            h_bottom_all = np.nanmin(h_all, axis=-1)

            # Identify sites needing extrapolation (below model bottom)
            alt_broadcast = alt[np.newaxis, np.newaxis, :]
            mask_extrap = alt_broadcast < h_bottom_all
            n_extrap = np.sum(mask_extrap)
            n_interp = total_tasks - n_extrap
            logger.debug(f"Interpolation: {n_interp}, Extrapolation: {n_extrap}")

            # Initialize output buffer
            out = np.empty((num_dim, time_dim, site_dim), dtype=float)

            # A. Vectorized Interpolation Path
            if n_interp > 0:
                interp_indices = np.where(~mask_extrap)
                n_idx, t_idx, s_idx = interp_indices

                h_in = h_all[n_idx, t_idx, s_idx, :]
                ztd_in = ztd_all[n_idx, t_idx, s_idx, :]
                h_out = alt[s_idx]

                def _vectorized_interp_kernel(h_batch, z_batch, h_target_batch):
                    """Vectorized Log-Linear Interpolation."""
                    if h_batch[0, 0] > h_batch[0, -1]:
                        h_batch = np.flip(h_batch, axis=1)
                        z_batch = np.flip(z_batch, axis=1)

                    log_z = np.log(np.maximum(z_batch, 1e-12))

                    # Find bounding indices
                    mask_ge = h_batch >= h_target_batch[:, None]
                    idx = np.clip(np.argmax(mask_ge, axis=1), 1, h_batch.shape[1] - 1)

                    rows = np.arange(len(idx))
                    x1, x2 = h_batch[rows, idx - 1], h_batch[rows, idx]
                    y1, y2 = log_z[rows, idx - 1], log_z[rows, idx]

                    denom = x2 - x1
                    denom[denom == 0] = 1e-12
                    w = (h_target_batch - x1) / denom

                    return np.exp(y1 + w * (y2 - y1))

                # Execute based on batch size
                if n_interp <= self.batch_size:
                    res_interp = _vectorized_interp_kernel(h_in, ztd_in, h_out)
                else:
                    n_chunks = int(np.ceil(n_interp / self.batch_size))
                    h_chunks = np.array_split(h_in, n_chunks)
                    z_chunks = np.array_split(ztd_in, n_chunks)
                    t_chunks = np.array_split(h_out, n_chunks)

                    results_chunks = Parallel(n_jobs=self.n_jobs)(
                        delayed(_vectorized_interp_kernel)(hc, zc, tc)
                        for hc, zc, tc in zip(h_chunks, z_chunks, t_chunks)
                    )
                    res_interp = np.concatenate(results_chunks)

                out[n_idx, t_idx, s_idx] = res_interp
                logger.debug(
                    f"Vectorized interpolation completed for {n_interp} points"
                )

            # B. Vectorized Extrapolation Path (WMO Physics)
            if n_extrap > 0:
                extrap_indices = np.where(mask_extrap)

                # Bottom level index (highest pressure = lowest height)
                bottom_level_idx = np.nanargmin(h_all, axis=-1)

                n_idx, t_idx, s_idx = extrap_indices
                bottom_idx_flat = bottom_level_idx[n_idx, t_idx, s_idx]

                h_bottom_ext = h_bottom_all[n_idx, t_idx, s_idx]
                ztd_bottom_ext = ztd_all[n_idx, t_idx, s_idx, bottom_idx_flat]
                p_bottom_ext = p_values[bottom_idx_flat]
                t_bottom_ext = t_all[n_idx, t_idx, s_idx, bottom_idx_flat]
                e_bottom_ext = e_all[n_idx, t_idx, s_idx, bottom_idx_flat]
                h_site_ext = alt[s_idx]

                ztd_site = calc_wmo_ztd_extrapolation(
                    h_bottom=h_bottom_ext,
                    ztd_bottom=ztd_bottom_ext,
                    p_bottom=p_bottom_ext,
                    t_bottom=t_bottom_ext,
                    e_bottom=e_bottom_ext,
                    h_site=h_site_ext,
                    refractive_index_constants=self.refractive_index_constants,
                )

                out[n_idx, t_idx, s_idx] = ztd_site
                logger.debug(f"WMO extrapolation completed for {n_extrap} points")

            da = xr.DataArray(
                out,
                dims=("number", "time", "site_index"),
                coords={
                    "number": ds.number,
                    "time": ds.time,
                    "site_index": ds.site_index,
                },
            ).transpose("number", "site_index", "time")
            ds_site = xr.Dataset({"ztd_simpson": da})

        # Height Mode (No extrapolation needed usually as resampling handles it)
        else:
            if total_tasks <= self.batch_size:
                alt_da = (
                    xr.DataArray(
                        alt, dims="site_index", coords={"site_index": ds.site_index}
                    )
                    .expand_dims(number=ds.number, time=ds.time)
                    .transpose("number", "site_index", "time")
                )
                ds_site = self._vertical_interp_vectorized(
                    ds, alt_da, log_linear_interp
                )

            else:
                ds_site = self._vertical_interp_parallel(
                    ds,
                    h_all,
                    ztd_all,
                    alt,
                    log_linear_interp,
                    num_dim,
                    time_dim,
                    site_dim,
                    total_tasks,
                )

        ds_site["site"] = ds.site
        return ds_site

    def run(self) -> pd.DataFrame:
        """Execute the ZTD computation pipeline and return results.

        Returns:
            DataFrame with columns: time, site, [number], ztd_simpson (mm)
        """
        logger.debug(f"Start ZTD computation (vertical='{self.vertical_dimension}')")
        self.read_met_file()
        self.horizontal_interpolate()
        self.geopotential_to_orthometric()
        self.orthometric_to_ellipsoidal()
        self.compute_e()
        if self.vertical_dimension != "pressure_level":
            self.resample_to_ellipsoidal()
        self.compute_refractive_index()
        self.compute_top_level_delay()
        self.simpson_numerical_integration()

        if self.interp_to_site:
            ds_site = self.vertical_interpolate_to_site()
            df = ds_site.to_dataframe().reset_index()[
                ["time", "site", "number", "ztd_simpson"]
            ]
        else:
            df = self.ds.to_dataframe().reset_index()[
                ["time", "site", "h", "number", "ztd_simpson"]
            ]

        if len(df.number.drop_duplicates()) == 1:
            df = df.drop(columns=["number"])

        df["ztd_simpson"] = df["ztd_simpson"] * 1000  # m -> mm

        if df.isnull().values.any():
            raise ValueError("NaN values found in final ZTD results")

        logger.debug("ZTD computation finished")
        return df
