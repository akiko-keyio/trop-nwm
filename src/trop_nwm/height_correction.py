"""ZTD 高程修正与网格重采样模块。

本模块提供两个核心类：
- ZTDHeightCorrection: 对 ZTD 进行垂直方向的高程修正
- ZTDGridResampler: 将 ZTD 从源网格重采样到目标位置，同时进行高程修正
"""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import delayed
from scipy.interpolate import griddata
from tqdm_joblib import ParallelPbar

# 默认模型文件路径查找函数
def _get_default_model_path():
    """获取默认模型文件路径，支持开发和安装环境。"""
    possible_paths = [
        Path(__file__).parent / "data" / "ztdht_model_01.dat",  # 安装位置
        Path(__file__).parent.parent.parent / "reference" / "ztdht_model_01.dat",  # 开发位置
    ]
    for path in possible_paths:
        if path.exists():
            return path
    # 如果都找不到，返回第一个路径（会在后续使用时报错）
    return possible_paths[0]

_DEFAULT_MODEL_PATH = _get_default_model_path()


class ZTDHeightCorrection:
    """ZTD 高程修正计算器。

    使用指数模型对 ZTD 进行垂直方向的高程修正：
        ZTD_new = ZTD_old * exp(-c * dh)

    其中衰减系数 c 是时空变化的，由模型文件提供的年周期和半年周期
    傅里叶系数通过空间插值和时间计算得到。

    Attributes:
        df_ztdhc: 模型系数数据框。
        coef_: 时变模型系数 (mean, annu_cos, annu_sin, semiannu_cos, semiannu_sin)。
        origin_lonlat_: 模型格点的经纬度。
    """

    def __init__(self, filepath=None):
        """初始化高程修正计算器。

        Args:
            filepath: 模型系数文件路径。如果为 None，使用默认路径。
        """
        if filepath is None:
            filepath = _DEFAULT_MODEL_PATH
        self.df_ztdhc = pd.read_csv(filepath, sep=r"\s+")
        self.coef_ = self.df_ztdhc[
            ["mean", "annu_cos", "annu_sin", "semiannu_cos", "semiannu_sin"]
        ].values
        self.origin_lonlat_ = self.df_ztdhc[["lon", "lat"]].values

    def _interpolate(self, interpolate_lonlat):
        interpolated_values_linear = griddata(
            self.origin_lonlat_, self.coef_, interpolate_lonlat, method="linear"
        )
        interpolated_values_nearest = griddata(
            self.origin_lonlat_, self.coef_, interpolate_lonlat, method="nearest"
        )
        interpolated_values = np.where(
            np.isnan(interpolated_values_linear),
            interpolated_values_nearest,
            interpolated_values_linear,
        )
        return interpolated_values

    def _calculate_coef_exp(self, time, coef_temporal):
        assert coef_temporal.shape[0] == 5

        doy = time.dayofyear + (time.hour + time.minute / 60) / 24
        rad = np.asarray(doy) / 365.25 * 2 * np.pi
        rad = rad[:, np.newaxis]

        return (
            coef_temporal[0]
            + coef_temporal[1] * np.cos(rad)
            + coef_temporal[2] * np.sin(rad)
            + coef_temporal[3] * np.cos(2 * rad)
            + coef_temporal[4] * np.sin(2 * rad)
        )

    def process(self, ztd_matrix, dh, time, location):
        """对 ZTD 矩阵进行高程修正。

        Args:
            ztd_matrix: ZTD 数据 DataFrame，index 为时间，columns 为站点。
            dh: 高程差 (目标高程 - 源高程)，单位 m。
            time: 时间索引 (DatetimeIndex)。
            location: 位置 DataFrame，需包含 'lon', 'lat' 列。

        Returns:
            修正后的 ZTD 矩阵 (numpy array)。
        """
        coef_temporal = self._interpolate(
            interpolate_lonlat=location[["lon", "lat"]].values
        )
        coef_exp = self._calculate_coef_exp(time, coef_temporal.T)
        dh = np.asarray(dh)[np.newaxis, :]
        ztd_matrix = ztd_matrix.values * np.exp(-coef_exp * dh)
        return ztd_matrix


class ZTDGridResampler:
    """ZTD 网格重采样器。

    将 ZTD 数据从源网格位置重采样到目标位置（网格或站点），
    同时进行高程修正以消除地形起伏带来的误差。

    工作流程：
        1. 对每个目标点，找到其周围最近的 4 个源网格点。
        2. 利用 ZTDHeightCorrection 将这 4 个源点的 ZTD 修正到目标点高程面。
        3. 在统一的高程面上进行双线性插值，得到目标点的 ZTD。

    Attributes:
        location_from: 源网格位置 DataFrame (需包含 lon, lat, alt, site 列)。
        location_to: 目标位置 DataFrame (需包含 lon, lat, alt, site 列)。
        origin_spacing: 源网格间距 (0.25, 0.5, 或 1 度)。
        df_convert_info: 预计算的插值权重和索引信息。

    Example:
        >>> resampler = ZTDGridResampler(location_from=grid_100, location_to=grid_50)
        >>> ztd_resampled = resampler.interpolate(X_origin=ztd_data)
    """

    def __init__(self, location_from, location_to):
        """初始化网格重采样器。

        Args:
            location_from: 源网格位置 DataFrame (需包含 lon, lat, alt, site 列)。
            location_to: 目标位置 DataFrame (需包含 lon, lat, alt, site 列)。
        """
        self.location_to = location_to
        self.location_from = location_from
        self.origin_spacing = self.get_spacing(location_from)
        self.df_convert_info = self._generate_convert_info_multiprocessing(
            location_from, location_to
        )

    def get_spacing(self, location):
        """计算网格间距。"""
        def calcualte_spacing(arr):
            return np.unique(np.diff(np.sort(np.unique(arr))))

        dlon = calcualte_spacing(location["lon"].values)
        dlat = calcualte_spacing(location["lat"].values)
        if dlon != dlat or dlon not in [0.25, 0.5, 1]:
            raise ValueError("网格间距必须为 0.25, 0.5 或 1 度，且经纬度间距需相等。")
        return dlon

    def _generate_convert_info_multiprocessing(self, location_from, location_to):
        """多进程生成转换信息。"""

        def find_nearest4(lon, lat, origin_spacing):
            inverse_origin_spacing = 1 / origin_spacing
            lon1 = np.floor(lon * inverse_origin_spacing) / inverse_origin_spacing
            lon2 = np.ceil(lon * inverse_origin_spacing) / inverse_origin_spacing
            lat1 = np.floor(lat * inverse_origin_spacing) / inverse_origin_spacing
            lat2 = np.ceil(lat * inverse_origin_spacing) / inverse_origin_spacing
            return lon1, lon2, lat1, lat2

        lon_origin = location_from.lon.values
        lat_origin = location_from.lat.values

        def process_site(site_interpolate, site_location):
            lon_site = site_location.lon
            lat_site = site_location.lat
            alt_site = site_location.alt

            lon1, lon2, lat1, lat2 = find_nearest4(
                lon_site, lat_site, origin_spacing=self.origin_spacing
            )

            def find_nearest1():
                distance = (location_from.lon - lon_site) ** 2 + (
                    location_from.lat - lat_site
                ) ** 2
                return location_from.loc[[distance.idxmin()]].copy()

            if lon1 != lon2 and lat1 != lat2:
                mask = ((lon_origin == lon1) | (lon_origin == lon2)) & (
                    (lat_origin == lat1) | (lat_origin == lat2)
                )
                convert_info = location_from.loc[mask].copy()
                if len(convert_info) != 4:
                    convert_info = find_nearest1()

            elif lon1 == lon2 and lat1 != lat2:
                mask = (lon_origin == lon1) & (
                    (lat_origin == lat1) | (lat_origin == lat2)
                )
                convert_info = location_from.loc[mask].copy()
                if len(convert_info) != 2:
                    convert_info = find_nearest1()

            elif lon1 != lon2 and lat1 == lat2:
                mask = ((lon_origin == lon1) | (lon_origin == lon2)) & (
                    lat_origin == lat1
                )
                convert_info = location_from.loc[mask].copy()
                if len(convert_info) != 2:
                    convert_info = find_nearest1()

            else:  # lon1 == lon2 and lat1 == lat2
                mask = (lon_origin == lon1) & (lat_origin == lat1)
                convert_info = location_from.loc[mask].copy()
                if len(convert_info) != 1:
                    convert_info = find_nearest1()

            convert_info["site_interpolate"] = site_interpolate
            convert_info["dh"] = alt_site - convert_info.alt
            return convert_info

        list_convert_info = ParallelPbar("Generating Conversion INFO")(n_jobs=-2)(
            delayed(process_site)(site_interpolate, site_location)
            for site_interpolate, site_location in location_to.set_index(
                "site"
            ).iterrows()
        )

        df_convert_info = pd.concat(list_convert_info, ignore_index=True).reset_index(
            drop=True
        )
        return df_convert_info

    def _check_dataframe(self, data):
        if isinstance(data, pd.DataFrame):
            return data
        raise TypeError("Data must be a pandas dataframe")

    def _height_correction(self, X_origin):
        """对源数据进行高程修正。"""
        ztd_matrix = X_origin[self.df_convert_info.site.values]
        values_interpolate = ZTDHeightCorrection().process(
            ztd_matrix=ztd_matrix,
            dh=self.df_convert_info["dh"],
            time=ztd_matrix.index,
            location=self.df_convert_info,
        )
        return pd.DataFrame(
            values_interpolate,
            columns=self.df_convert_info.site.values,
            index=X_origin.index,
        )

    def _interpolate_multiprocessing(self, values_interpolate):
        """多进程进行双线性插值。"""

        def linear_interpolate(x1, x2, f1, f2, xi):
            if xi == x1:
                return f1
            if xi == x2:
                return f2

            d1i = xi - x1
            di2 = x2 - xi
            d12 = x2 - x1
            if d1i * di2 < 0:
                return f1 if np.abs(d1i) > np.abs(di2) else f2

            return (f1 * di2 + f2 * d1i) / d12

        def bilinear_interpolate(x1, x2, y1, y2, xi, yi, f11, f21, f12, f22):
            xiy1 = linear_interpolate(x1=x1, x2=x2, f1=f11, f2=f21, xi=xi)
            xiy2 = linear_interpolate(x1=x1, x2=x2, f1=f12, f2=f22, xi=xi)
            xiyi = linear_interpolate(x1=y1, x2=y2, f1=xiy1, f2=xiy2, xi=yi)
            return xiyi

        df_convert_info = self.df_convert_info.copy()
        df_convert_info["id"] = range(len(df_convert_info))
        df_convert_info = df_convert_info.set_index("site_interpolate")
        location_to = self.location_to
        location_to_site = location_to["site"]

        def interpolate_site(site_interpolate, site_convert_info):
            site_convert_info = site_convert_info.set_index(["lon", "lat"]).sort_index()

            site_info = location_to.set_index("site").loc[site_interpolate]
            lon1, lat1 = site_convert_info.index[0]
            lon2, lat2 = site_convert_info.index[-1]

            def get_values(lon, lat):
                id = site_convert_info.loc[(lon, lat)]["id"]
                return values_interpolate.iloc[:, id]

            bi = bilinear_interpolate(
                x1=lon1,
                x2=lon2,
                y1=lat1,
                y2=lat2,
                f11=get_values(lon1, lat1),
                f21=get_values(lon2, lat1),
                f12=get_values(lon1, lat2),
                f22=get_values(lon2, lat2),
                xi=site_info["lon"],
                yi=site_info["lat"],
            )
            bi.name = site_interpolate
            return bi

        interpolate_list = ParallelPbar("Interpolating sites")(n_jobs=4)(
            delayed(interpolate_site)(site, df_convert_info.loc[[site]].copy())
            for site in location_to_site
        )

        data_interpolate = pd.concat(interpolate_list, axis=1)
        return data_interpolate

    def interpolate(self, X_origin):
        """执行重采样。

        Args:
            X_origin: 源 ZTD 数据 DataFrame，index 为时间，columns 为站点名。

        Returns:
            重采样后的 ZTD DataFrame，columns 为目标站点名。
        """
        values_interpolate = self._height_correction(X_origin)
        return self._interpolate_multiprocessing(values_interpolate)
