"""
4D Spatio-Temporal Metocean Vector Interpolator for TideTrace.
Performs bilinear interpolation in space (lat, lon) and linear interpolation in time (t)
to return exact vector components (u_c, v_c, u_10, v_10) at any query coordinate (t, lat, lon).
"""

from __future__ import annotations

from datetime import datetime
from typing import NamedTuple, Tuple, Union

import numpy as np

from app.metocean.store import MetoceanGrid, parse_timestamp


class MetoceanVector(NamedTuple):
    """
    Interpolated 4D Metocean vector components at a specific point in spacetime.

    Fields:
        u_current: Eastward ocean current velocity in m/s.
        v_current: Northward ocean current velocity in m/s.
        u_wind: Eastward 10m wind velocity in m/s.
        v_wind: Northward 10m wind velocity in m/s.
    """
    u_current: float
    v_current: float
    u_wind: float
    v_wind: float

    @property
    def current_speed(self) -> float:
        """Return scalar ocean current speed in m/s."""
        return float(np.hypot(self.u_current, self.v_current))

    @property
    def wind_speed(self) -> float:
        """Return scalar 10m wind speed in m/s."""
        return float(np.hypot(self.u_wind, self.v_wind))


class MetoceanInterpolator:
    """
    4D spatio-temporal bilinear-spatial / linear-temporal interpolator.
    Optimized for high-throughput scalar and batch particle advection queries.
    """

    def __init__(self, grid: MetoceanGrid):
        self.grid = grid
        self.times = grid.times
        self.lats = grid.lats
        self.lons = grid.lons
        self.u_c = grid.u_current
        self.v_c = grid.v_current
        self.u_w = grid.u_wind
        self.v_w = grid.v_wind

        self.t_min = float(self.times[0])
        self.t_max = float(self.times[-1])
        self.lat_min = float(self.lats[0])
        self.lat_max = float(self.lats[-1])
        self.lon_min = float(self.lons[0])
        self.lon_max = float(self.lons[-1])

        self.n_t = len(self.times)
        self.n_lat = len(self.lats)
        self.n_lon = len(self.lons)

    def _get_interp_weights_1d(
        self, arr: np.ndarray, val: float, n_len: int
    ) -> Tuple[int, int, float]:
        """
        Compute lower index, upper index, and linear fraction for 1D coordinate array.
        Clamps to boundary if coordinate is outside array range.
        """
        if val <= arr[0]:
            return 0, 0, 0.0
        if val >= arr[-1]:
            last = n_len - 1
            return last, last, 0.0

        idx = int(np.searchsorted(arr, val)) - 1
        idx = max(0, min(idx, n_len - 2))
        x0 = arr[idx]
        x1 = arr[idx + 1]
        denom = x1 - x0
        frac = (val - x0) / denom if denom > 0 else 0.0
        return idx, idx + 1, float(frac)

    def interpolate(
        self, t: Union[float, int, str, datetime], lat: float, lon: float
    ) -> MetoceanVector:
        """
        Interpolate (u_c, v_c, u_w, v_w) at query point (t, lat, lon).
        """
        ts = parse_timestamp(t)

        it0, it1, tau = self._get_interp_weights_1d(self.times, ts, self.n_t)
        ilat0, ilat1, mu = self._get_interp_weights_1d(self.lats, lat, self.n_lat)
        ilon0, ilon1, nu = self._get_interp_weights_1d(self.lons, lon, self.n_lon)

        # Bilinear weights for spatial grid
        w00 = (1.0 - mu) * (1.0 - nu)
        w10 = mu * (1.0 - nu)
        w01 = (1.0 - mu) * nu
        w11 = mu * nu

        def _interp_var(var_3d: np.ndarray) -> float:
            # Spatial slice at it0
            val_t0 = (
                w00 * var_3d[it0, ilat0, ilon0]
                + w10 * var_3d[it0, ilat1, ilon0]
                + w01 * var_3d[it0, ilat0, ilon1]
                + w11 * var_3d[it0, ilat1, ilon1]
            )
            if it0 == it1 or tau == 0.0:
                return float(val_t0)

            # Spatial slice at it1
            val_t1 = (
                w00 * var_3d[it1, ilat0, ilon0]
                + w10 * var_3d[it1, ilat1, ilon0]
                + w01 * var_3d[it1, ilat0, ilon1]
                + w11 * var_3d[it1, ilat1, ilon1]
            )

            # Linear blend in time
            return float((1.0 - tau) * val_t0 + tau * val_t1)

        u_c_val = _interp_var(self.u_c)
        v_c_val = _interp_var(self.v_c)
        u_w_val = _interp_var(self.u_w)
        v_w_val = _interp_var(self.v_w)

        return MetoceanVector(
            u_current=u_c_val,
            v_current=v_c_val,
            u_wind=u_w_val,
            v_wind=v_w_val,
        )

    def interpolate_batch(
        self,
        t_arr: Union[float, int, np.ndarray],
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized batch interpolation for N particles at coordinate vectors (lats, lons).
        t_arr can be a scalar float timestamp or an array of timestamps [N].

        Returns:
            Tuple of (u_current, v_current, u_wind, v_wind) as 1D float arrays of length N.
        """
        lats_arr = np.asarray(lats, dtype=np.float64)
        lons_arr = np.asarray(lons, dtype=np.float64)
        N = len(lats_arr)

        if np.isscalar(t_arr):
            ts_arr = np.full(N, float(t_arr), dtype=np.float64)
        else:
            ts_arr = np.asarray(t_arr, dtype=np.float64)

        # Time indices and weights
        it0 = np.clip(np.searchsorted(self.times, ts_arr) - 1, 0, self.n_t - 2)
        it1 = it0 + 1
        t_denom = self.times[it1] - self.times[it0]
        tau = np.where(t_denom > 0, (ts_arr - self.times[it0]) / t_denom, 0.0)
        tau = np.clip(tau, 0.0, 1.0)
        # Check out-of-bounds time clamping
        tau = np.where(ts_arr <= self.times[0], 0.0, tau)
        tau = np.where(ts_arr >= self.times[-1], 0.0, tau)
        it0 = np.where(ts_arr >= self.times[-1], self.n_t - 1, it0)
        it1 = np.where(ts_arr >= self.times[-1], self.n_t - 1, it1)
        it0 = np.where(ts_arr <= self.times[0], 0, it0)
        it1 = np.where(ts_arr <= self.times[0], 0, it1)

        # Lat indices and weights
        ilat0 = np.clip(np.searchsorted(self.lats, lats_arr) - 1, 0, self.n_lat - 2)
        ilat1 = ilat0 + 1
        lat_denom = self.lats[ilat1] - self.lats[ilat0]
        mu = np.where(lat_denom > 0, (lats_arr - self.lats[ilat0]) / lat_denom, 0.0)
        mu = np.clip(mu, 0.0, 1.0)
        mu = np.where(lats_arr <= self.lats[0], 0.0, mu)
        mu = np.where(lats_arr >= self.lats[-1], 0.0, mu)
        ilat0 = np.where(lats_arr >= self.lats[-1], self.n_lat - 1, ilat0)
        ilat1 = np.where(lats_arr >= self.lats[-1], self.n_lat - 1, ilat1)
        ilat0 = np.where(lats_arr <= self.lats[0], 0, ilat0)
        ilat1 = np.where(lats_arr <= self.lats[0], 0, ilat1)

        # Lon indices and weights
        ilon0 = np.clip(np.searchsorted(self.lons, lons_arr) - 1, 0, self.n_lon - 2)
        ilon1 = ilon0 + 1
        lon_denom = self.lons[ilon1] - self.lons[ilon0]
        nu = np.where(lon_denom > 0, (lons_arr - self.lons[ilon0]) / lon_denom, 0.0)
        nu = np.clip(nu, 0.0, 1.0)
        nu = np.where(lons_arr <= self.lons[0], 0.0, nu)
        nu = np.where(lons_arr >= self.lons[-1], 0.0, nu)
        ilon0 = np.where(lons_arr >= self.lons[-1], self.n_lon - 1, ilon0)
        ilon1 = np.where(lons_arr >= self.lons[-1], self.n_lon - 1, ilon1)
        ilon0 = np.where(lons_arr <= self.lons[0], 0, ilon0)
        ilon1 = np.where(lons_arr <= self.lons[0], 0, ilon1)

        # 4 spatial corner weights
        w00 = (1.0 - mu) * (1.0 - nu)
        w10 = mu * (1.0 - nu)
        w01 = (1.0 - mu) * nu
        w11 = mu * nu

        def _interp_array(var_3d: np.ndarray) -> np.ndarray:
            val_t0 = (
                w00 * var_3d[it0, ilat0, ilon0]
                + w10 * var_3d[it0, ilat1, ilon0]
                + w01 * var_3d[it0, ilat0, ilon1]
                + w11 * var_3d[it0, ilat1, ilon1]
            )
            val_t1 = (
                w00 * var_3d[it1, ilat0, ilon0]
                + w10 * var_3d[it1, ilat1, ilon0]
                + w01 * var_3d[it1, ilat0, ilon1]
                + w11 * var_3d[it1, ilat1, ilon1]
            )
            return (1.0 - tau) * val_t0 + tau * val_t1

        u_c_out = _interp_array(self.u_c)
        v_c_out = _interp_array(self.v_c)
        u_w_out = _interp_array(self.u_w)
        v_w_out = _interp_array(self.v_w)

        return u_c_out, v_c_out, u_w_out, v_w_out
