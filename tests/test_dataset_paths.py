"""Small fixtures for the repository's data/ layout."""

import os
import tempfile
import unittest
from pathlib import Path

from wind_dashboard_core.loaders import available_dataset_keys, load_dataset


class DatasetPathTests(unittest.TestCase):
    def test_all_datasets_load_from_data_even_with_another_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            data = root / "data"
            data.mkdir(parents=True)
            (data / "Wind_Turbine_SCADA_tratado_consolidado.csv").write_text(
                "Turbine_ID,Timestamp,Amb_WindSpeed_Avg,Grd_Prod_Pwr_Avg\n"
                "T01,2017-01-01 00:00:00,5,100\n",
                encoding="utf-8",
            )
            (data / "Location1.csv").write_text(
                "Time,windspeed_100m,Power\n2017-01-01 00:00:00,5,0.5\n",
                encoding="utf-8",
            )
            (data / "NREL_Reference_5MW_126 (1).csv").write_text(
                "Wind Speed [m/s],Power [kW]\n5,100\n",
                encoding="utf-8",
            )
            (data / "Turbine_Data_Penmanshiel_01_2020-01-01_-_2021-01-01_1042.csv").write_text(
                "# Turbine: Penmanshiel 01\n"
                "# Turbine type: Senvion MM82\n"
                "# Date and time,Wind speed (m/s),Power (kW),Available Capacity for Production (kW)\n"
                "2020-01-01 00:00:00,5,100,2050\n",
                encoding="utf-8",
            )
            (data / "Status_Penmanshiel_01_2020-01-01_-_2021-01-01_1042.csv").write_text(
                "Timestamp start,Timestamp end,Status,Code,Message\n"
                "2019-12-31 23:59:00,2020-01-01 00:10:00,Running,1,Normal\n",
                encoding="utf-8",
            )

            original_cwd = Path.cwd()
            elsewhere = Path(temporary) / "elsewhere"
            elsewhere.mkdir()
            try:
                os.chdir(elsewhere)
                self.assertEqual(available_dataset_keys(root), ["edp", "kaggle", "nrel", "penmanshiel"])
                expected_values = {
                    "edp": ("power_kw", 100),
                    "kaggle": ("power_norm", 0.5),
                    "nrel": ("power_kw", 100),
                    "penmanshiel": ("power_kw", 100),
                }
                for key in ("edp", "kaggle", "nrel", "penmanshiel"):
                    with self.subTest(dataset=key):
                        frame = load_dataset(key, root)
                        self.assertEqual(len(frame), 1)
                        self.assertEqual(frame.loc[0, "dataset"], key)
                        column, value = expected_values[key]
                        self.assertEqual(frame.loc[0, column], value)
                        if key == "penmanshiel":
                            self.assertEqual(frame.loc[0, "status"], "Running")
            finally:
                os.chdir(original_cwd)

    def test_edp_turbine_fallback_uses_data_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "Turbina_T01_data.csv").write_text(
                "Turbine_ID;Timestamp;Amb_WindSpeed_Avg;Grd_Prod_Pwr_Avg\n"
                "T01;2017-01-01 00:00:00;5;100\n",
                encoding="utf-8",
            )
            self.assertEqual(available_dataset_keys(root), ["edp"])
            frame = load_dataset("edp", root)
            self.assertEqual(frame.loc[0, "entity"], "T01")
            self.assertEqual(frame.loc[0, "power_kw"], 100)

    def test_root_level_csv_is_not_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Location1.csv").write_text("Time,Power\n2017-01-01,0.5\n", encoding="utf-8")
            self.assertEqual(available_dataset_keys(root), [])


if __name__ == "__main__":
    unittest.main()
