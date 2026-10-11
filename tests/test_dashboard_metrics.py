import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from wind_dashboard_core.loaders import DATASET_SPECS
from wind_dashboard_core.metrics import (
    build_profiles,
    count_duplicate_rows,
    dataset_profile,
    infer_resolution_minutes,
)


class DuplicateRowsTests(unittest.TestCase):
    def assert_matches_pandas(self, frame, block_rows=37):
        columns = list(frame.columns)
        expected = int(frame.duplicated().sum())
        original_duplicated = pd.DataFrame.duplicated
        sizes = []

        def bounded_duplicated(block, *args, **kwargs):
            sizes.append(len(block))
            self.assertLessEqual(len(block), block_rows)
            return original_duplicated(block, *args, **kwargs)

        before = frame.copy(deep=True)
        with patch.object(pd.DataFrame, 'duplicated', bounded_duplicated):
            actual = count_duplicate_rows(frame, columns, block_rows=block_rows)
        self.assertEqual(actual, expected)
        pd.testing.assert_frame_equal(frame, before)
        return sizes

    def test_duplicates_across_blocks_and_nonunique_index(self):
        frame = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=500, freq='10min'),
            'entity': ['T01', 'T02'] * 250,
            'power_kw': np.arange(500, dtype=float),
        })
        frame = pd.concat([frame, frame.iloc[::3], frame.iloc[::-5]])
        self.assertTrue(self.assert_matches_pandas(frame))

    def test_skewed_partitions_missing_values_and_signed_zero(self):
        frame = pd.DataFrame({
            'timestamp': [pd.NaT] * 2_000,
            'entity': pd.Categorical(['T01'] * 1_900 + ['T02'] * 100),
            'power_kw': np.resize([0.0, -0.0, np.nan, 1.0, 2.0], 2_000),
            'status': pd.array(np.resize([1, None, 2], 2_000), dtype='Int64'),
        })
        self.assert_matches_pandas(frame)

    def test_constant_partition_needs_no_multi_column_duplicate_array(self):
        frame = pd.DataFrame({'entity': ['T01'] * 2_000, 'power_kw': [np.nan] * 2_000})
        self.assertEqual(self.assert_matches_pandas(frame), [])

    def test_mixed_object_equality(self):
        frame = pd.DataFrame({
            'entity': np.resize(np.array([1, True, '1', None, np.nan, pd.NA], dtype=object), 500),
            'power_kw': np.resize([0.0, -0.0, 1.0, np.nan], 500),
        })
        self.assert_matches_pandas(frame)

    def test_single_column_keeps_pandas_missing_value_semantics(self):
        frame = pd.DataFrame({'entity': np.resize(np.array([None, np.nan, pd.NA, 'T01'], dtype=object), 500)})
        self.assertEqual(count_duplicate_rows(frame, ['entity'], 37), int(frame.duplicated().sum()))

    def test_unique_and_empty_frames(self):
        self.assert_matches_pandas(pd.DataFrame({'entity': range(200), 'power_kw': [1.0] * 200}))
        self.assertEqual(count_duplicate_rows(pd.DataFrame(columns=['entity', 'power_kw']), ['entity', 'power_kw']), 0)


class ProfileTests(unittest.TestCase):
    def test_full_record_quality_and_resolution_are_preserved(self):
        frame = pd.DataFrame({
            'timestamp': pd.to_datetime(['2020-01-01 00:00', '2020-01-01 00:10', '2020-01-01 00:20', '2020-01-01 00:10', None]),
            'entity': ['T01'] * 5,
            'wind_speed_ms': [1.0, 2.0, 3.0, 2.0, np.nan],
            'power_kw': [10.0, 20.0, 30.0, 20.0, np.nan],
            'unrelated_payload': ['x' * 1_000] * 5,
        })
        profile = dataset_profile(frame, DATASET_SPECS['edp'])
        self.assertEqual(profile['rows'], 5)
        self.assertEqual(profile['duplicated_rows_pct'], 20.0)
        self.assertEqual(profile['missing_cells_pct'], 15.0)
        self.assertEqual(profile['resolution_minutes'], 10.0)
        self.assertEqual(profile['coverage_start'], pd.Timestamp('2020-01-01'))
        self.assertEqual(infer_resolution_minutes(frame), 10.0)

    def test_profile_generator_matches_mapping(self):
        frame = pd.DataFrame({'entity': ['T01', 'T01'], 'power_kw': [1.0, 1.0]})
        profiles, table = build_profiles({'edp': frame}, DATASET_SPECS)
        streamed_profiles, streamed_table = build_profiles(iter([('edp', frame)]), DATASET_SPECS)
        self.assertEqual(streamed_profiles.keys(), profiles.keys())
        pd.testing.assert_frame_equal(streamed_table, table)


if __name__ == '__main__':
    unittest.main()
