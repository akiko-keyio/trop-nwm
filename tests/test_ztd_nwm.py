import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from trop_nwm.ztd_nwm import ZTDNWMGenerator, _G0


class ReadMetFileTests(unittest.TestCase):
    def test_converts_geopotential_height_to_geopotential(self):
        geopotential_height = np.array([[[[100.0]], [[200.0]]]])
        dataset = xr.Dataset(
            data_vars={
                "gh": (("time", "level", "latitude", "longitude"), geopotential_height),
                "t": (("time", "level", "latitude", "longitude"), np.full_like(geopotential_height, 280.0)),
                "q": (("time", "level", "latitude", "longitude"), np.full_like(geopotential_height, 0.01)),
            },
            coords={
                "time": [np.datetime64("2025-01-01")],
                "level": [1000, 900],
                "latitude": [0.0],
                "longitude": [0.0],
            },
        )

        with patch("trop_nwm.ztd_nwm.xr.load_dataset", return_value=dataset):
            generator = ZTDNWMGenerator("ens.nc")
            generator.read_met_file()

        self.assertNotIn("gh", generator.ds)
        np.testing.assert_allclose(
            generator.ds["z"].isel(number=0), geopotential_height * _G0
        )
        self.assertEqual(generator.ds["z"].attrs["units"], "m**2 s**-2")


if __name__ == "__main__":
    unittest.main()
