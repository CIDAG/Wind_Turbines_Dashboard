"""Extraction of the unchanged Zenodo archive into a flat data directory."""

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.extract_penmanshiel import extract_penmanshiel


class ExtractPenmanshielTests(unittest.TestCase):
    def test_extracts_nested_csvs_into_data_without_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "datasets2020-penmanshiel.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("datasets2020-penmanshiel/Turbine_Data_Penmanshiel_01.csv", "wind,power\n5,100\n")
                archive.writestr("datasets2020-penmanshiel/Status_Penmanshiel_01.csv", "status\nNormal\n")
                archive.writestr("datasets2020-penmanshiel/README.txt", "metadata")

            data = root / "data"
            self.assertEqual(extract_penmanshiel(archive_path, data), 2)
            self.assertEqual((data / "Turbine_Data_Penmanshiel_01.csv").read_text(), "wind,power\n5,100\n")
            self.assertTrue((data / "Status_Penmanshiel_01.csv").is_file())
            self.assertFalse((data / "datasets2020-penmanshiel").exists())

            with self.assertRaises(FileExistsError):
                extract_penmanshiel(archive_path, data)


if __name__ == "__main__":
    unittest.main()
