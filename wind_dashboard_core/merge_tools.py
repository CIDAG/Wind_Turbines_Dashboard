"""Deterministic merge planning and execution for canonical wind datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .loaders import CANONICAL_FEATURES, DatasetSpec
from .metrics import infer_resolution_minutes


MERGE_OK = "MERGE_OK"
MERGE_CAUTION = "MERGE_CAUTION"
MERGE_BLOCKED = "MERGE_BLOCKED"

KEY_ENTITY_TIME = "KEY_ENTITY_TIME"
KEY_TIME_WINDOW = "KEY_TIME_WINDOW"
KEY_WIND_BIN = "KEY_WIND_BIN"
KEY_METADATA = "KEY_METADATA"
KEY_NONE = "KEY_NONE"


@dataclass(frozen=True)
class MergePlan:
    left_key: str
    right_key: str
    left_label: str
    right_label: str
    merge_flag: str
    compatibility_flag: str
    key_flag: str
    strategy_id: str
    strategy_label: str
    join_keys: tuple[str, ...]
    score: int
    preferred_power: str
    window_minutes: int | None
    wind_bin_width_ms: float | None
    temporal_overlap_pct: float | None
    entity_overlap_pct: float | None
    resolution_ratio: float | None
    common_feature_count: int
    warnings: tuple[str, ...]
    details: tuple[str, ...]
    can_execute: bool

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["join_keys"] = ", ".join(self.join_keys) if self.join_keys else "n/a"
        data["warnings"] = " | ".join(self.warnings) if self.warnings else ""
        data["details"] = " | ".join(self.details) if self.details else ""
        return data


def _has_data(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and df[column].notna().any()


def _canonical_common_fields(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    return [
        field
        for field in CANONICAL_FEATURES
        if _has_data(left, field) and _has_data(right, field)
    ]


def _preferred_power(left: pd.DataFrame, right: pd.DataFrame) -> str:
    if _has_data(left, "power_kw") and _has_data(right, "power_kw"):
        return "power_kw"
    if _has_data(left, "power_norm") and _has_data(right, "power_norm"):
        return "power_norm"
    return "n/a"


def _series_range(series: pd.Series) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    ts = pd.to_datetime(series, errors="coerce").dropna()
    if ts.empty:
        return None, None
    return ts.min(), ts.max()


def _temporal_overlap_pct(left: pd.DataFrame, right: pd.DataFrame) -> float | None:
    if not (_has_data(left, "timestamp") and _has_data(right, "timestamp")):
        return None

    left_start, left_end = _series_range(left["timestamp"])
    right_start, right_end = _series_range(right["timestamp"])
    if left_start is None or right_start is None or left_end is None or right_end is None:
        return None

    overlap_start = max(left_start, right_start)
    overlap_end = min(left_end, right_end)
    if overlap_end <= overlap_start:
        return 0.0

    left_span = max((left_end - left_start).total_seconds(), 1.0)
    right_span = max((right_end - right_start).total_seconds(), 1.0)
    overlap_span = (overlap_end - overlap_start).total_seconds()
    return round(float(100.0 * overlap_span / min(left_span, right_span)), 2)


def _entity_overlap_pct(left: pd.DataFrame, right: pd.DataFrame) -> float | None:
    if not (_has_data(left, "entity") and _has_data(right, "entity")):
        return None

    left_values = set(left["entity"].dropna().astype(str).str.strip())
    right_values = set(right["entity"].dropna().astype(str).str.strip())
    if not left_values or not right_values:
        return None

    return round(float(100.0 * len(left_values.intersection(right_values)) / min(len(left_values), len(right_values))), 2)


def _resolution_ratio(left: pd.DataFrame, right: pd.DataFrame) -> float | None:
    if not (_has_data(left, "timestamp") and _has_data(right, "timestamp")):
        return None

    left_resolution = infer_resolution_minutes(left)
    right_resolution = infer_resolution_minutes(right)
    if pd.isna(left_resolution) or pd.isna(right_resolution):
        return None

    low = max(min(float(left_resolution), float(right_resolution)), 1e-9)
    high = max(float(left_resolution), float(right_resolution))
    return round(float(high / low), 2)


def _window_minutes(left: pd.DataFrame, right: pd.DataFrame) -> int:
    resolutions = [
        value
        for value in [infer_resolution_minutes(left), infer_resolution_minutes(right)]
        if pd.notna(value) and value > 0
    ]
    if not resolutions:
        return 10
    return int(max(1, round(max(resolutions))))


def _metadata_keys(left: pd.DataFrame, right: pd.DataFrame) -> tuple[str, ...]:
    keys = []
    for column in ["site", "turbine_model", "year", "rated_power_kw", "rotor_diameter_m"]:
        if not (_has_data(left, column) and _has_data(right, column)):
            continue
        left_values = set(left[column].dropna().astype(str).str.strip())
        right_values = set(right[column].dropna().astype(str).str.strip())
        if left_values.intersection(right_values):
            keys.append(column)
    return tuple(keys[:2])


def _compatibility_flag(score: int) -> str:
    if score >= 75:
        return "HIGH"
    if score >= 50:
        return "MEDIUM"
    return "LOW"


def _make_plan(
    *,
    left_key: str,
    right_key: str,
    left_label: str,
    right_label: str,
    merge_flag: str,
    key_flag: str,
    strategy_id: str,
    strategy_label: str,
    join_keys: tuple[str, ...],
    score: int,
    preferred_power: str,
    window_minutes: int | None,
    wind_bin_width_ms: float | None,
    temporal_overlap_pct: float | None,
    entity_overlap_pct: float | None,
    resolution_ratio: float | None,
    common_feature_count: int,
    warnings: list[str],
    details: list[str],
    can_execute: bool,
) -> MergePlan:
    score = int(np.clip(score, 0, 100))
    return MergePlan(
        left_key=left_key,
        right_key=right_key,
        left_label=left_label,
        right_label=right_label,
        merge_flag=merge_flag,
        compatibility_flag=_compatibility_flag(score),
        key_flag=key_flag,
        strategy_id=strategy_id,
        strategy_label=strategy_label,
        join_keys=join_keys,
        score=score,
        preferred_power=preferred_power,
        window_minutes=window_minutes,
        wind_bin_width_ms=wind_bin_width_ms,
        temporal_overlap_pct=temporal_overlap_pct,
        entity_overlap_pct=entity_overlap_pct,
        resolution_ratio=resolution_ratio,
        common_feature_count=common_feature_count,
        warnings=tuple(warnings),
        details=tuple(details),
        can_execute=can_execute,
    )


def plan_compatible_merge(
    left_key: str,
    right_key: str,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    specs: dict[str, DatasetSpec],
) -> MergePlan:
    """Choose a deterministic merge strategy for two canonical datasets."""

    left_spec = specs[left_key]
    right_spec = specs[right_key]
    preferred_power = _preferred_power(left_df, right_df)
    common_fields = _canonical_common_fields(left_df, right_df)
    common_count = len(common_fields)
    temporal_overlap = _temporal_overlap_pct(left_df, right_df)
    entity_overlap = _entity_overlap_pct(left_df, right_df)
    resolution_ratio = _resolution_ratio(left_df, right_df)
    has_timestamp = _has_data(left_df, "timestamp") and _has_data(right_df, "timestamp")
    has_wind_power = _has_data(left_df, "wind_speed_ms") and _has_data(right_df, "wind_speed_ms") and preferred_power != "n/a"
    has_reference_curve = "reference_curve" in {left_spec.source_type, right_spec.source_type}
    warnings: list[str] = []
    details = [
        f"Common canonical features: {common_count}",
        f"Preferred power column: {preferred_power}",
    ]

    if has_reference_curve and has_wind_power:
        score = 70 + min(common_count, 5) * 3
        if preferred_power == "power_norm":
            warnings.append("Power is compared in normalized scale.")
        return _make_plan(
            left_key=left_key,
            right_key=right_key,
            left_label=left_spec.label,
            right_label=right_spec.label,
            merge_flag=MERGE_OK if score >= 75 else MERGE_CAUTION,
            key_flag=KEY_WIND_BIN,
            strategy_id="wind_bin_curve",
            strategy_label="Wind-speed bin merge",
            join_keys=("_wind_bin_ms",),
            score=score,
            preferred_power=preferred_power,
            window_minutes=None,
            wind_bin_width_ms=0.5,
            temporal_overlap_pct=temporal_overlap,
            entity_overlap_pct=entity_overlap,
            resolution_ratio=resolution_ratio,
            common_feature_count=common_count,
            warnings=warnings,
            details=details + ["Reference curves are safer to merge by wind-speed bins than by timestamp."],
            can_execute=True,
        )

    if has_timestamp and temporal_overlap and temporal_overlap > 0:
        window = _window_minutes(left_df, right_df)
        if entity_overlap and entity_overlap > 0:
            score = 75 + min(common_count, 5) * 3
            if resolution_ratio and resolution_ratio > 2.0:
                score -= 15
                warnings.append("Temporal resolutions are different; rows will be grouped before merge.")
            return _make_plan(
                left_key=left_key,
                right_key=right_key,
                left_label=left_spec.label,
                right_label=right_spec.label,
                merge_flag=MERGE_OK if score >= 75 else MERGE_CAUTION,
                key_flag=KEY_ENTITY_TIME,
                strategy_id="temporal_entity",
                strategy_label="Entity + time-window merge",
                join_keys=("entity", "_merge_time"),
                score=score,
                preferred_power=preferred_power,
                window_minutes=window,
                wind_bin_width_ms=None,
                temporal_overlap_pct=temporal_overlap,
                entity_overlap_pct=entity_overlap,
                resolution_ratio=resolution_ratio,
                common_feature_count=common_count,
                warnings=warnings,
                details=details + [f"Timestamps will be floored to {window} minute windows."],
                can_execute=True,
            )

        score = 55 + min(common_count, 5) * 3
        warnings.append("No shared entity values were found; merge will aggregate by time window only.")
        if resolution_ratio and resolution_ratio > 2.0:
            score -= 10
            warnings.append("Temporal resolutions are different; aggregation may smooth one dataset more than the other.")
        return _make_plan(
            left_key=left_key,
            right_key=right_key,
            left_label=left_spec.label,
            right_label=right_spec.label,
            merge_flag=MERGE_CAUTION,
            key_flag=KEY_TIME_WINDOW,
            strategy_id="temporal_window",
            strategy_label="Time-window aggregate merge",
            join_keys=("_merge_time",),
            score=score,
            preferred_power=preferred_power,
            window_minutes=window,
            wind_bin_width_ms=None,
            temporal_overlap_pct=temporal_overlap,
            entity_overlap_pct=entity_overlap,
            resolution_ratio=resolution_ratio,
            common_feature_count=common_count,
            warnings=warnings,
            details=details + [f"Timestamps will be floored to {window} minute windows."],
            can_execute=True,
        )

    if has_wind_power:
        score = 60 + min(common_count, 5) * 3
        warnings.append("No usable temporal overlap was found; merge will compare binned power curves.")
        return _make_plan(
            left_key=left_key,
            right_key=right_key,
            left_label=left_spec.label,
            right_label=right_spec.label,
            merge_flag=MERGE_CAUTION,
            key_flag=KEY_WIND_BIN,
            strategy_id="wind_bin_curve",
            strategy_label="Wind-speed bin merge",
            join_keys=("_wind_bin_ms",),
            score=score,
            preferred_power=preferred_power,
            window_minutes=None,
            wind_bin_width_ms=0.5,
            temporal_overlap_pct=temporal_overlap,
            entity_overlap_pct=entity_overlap,
            resolution_ratio=resolution_ratio,
            common_feature_count=common_count,
            warnings=warnings,
            details=details + ["Wind-speed bins use 0.5 m/s intervals."],
            can_execute=True,
        )

    metadata_keys = _metadata_keys(left_df, right_df)
    if metadata_keys:
        score = 45 + 10 * len(metadata_keys) + min(common_count, 4) * 2
        warnings.append("Only metadata keys overlap; this merge should be used for annotation, not operational row matching.")
        return _make_plan(
            left_key=left_key,
            right_key=right_key,
            left_label=left_spec.label,
            right_label=right_spec.label,
            merge_flag=MERGE_CAUTION,
            key_flag=KEY_METADATA,
            strategy_id="metadata_keys",
            strategy_label="Metadata-key merge",
            join_keys=metadata_keys,
            score=score,
            preferred_power=preferred_power,
            window_minutes=None,
            wind_bin_width_ms=None,
            temporal_overlap_pct=temporal_overlap,
            entity_overlap_pct=entity_overlap,
            resolution_ratio=resolution_ratio,
            common_feature_count=common_count,
            warnings=warnings,
            details=details,
            can_execute=True,
        )

    if has_timestamp and (temporal_overlap == 0):
        warnings.append("Datasets have timestamps, but their date ranges do not overlap.")
    if preferred_power == "n/a":
        warnings.append("No compatible power column is available in both datasets.")
    if common_count < 2:
        warnings.append("Too few canonical features overlap.")

    return _make_plan(
        left_key=left_key,
        right_key=right_key,
        left_label=left_spec.label,
        right_label=right_spec.label,
        merge_flag=MERGE_BLOCKED,
        key_flag=KEY_NONE,
        strategy_id="not_recommended",
        strategy_label="Merge not recommended",
        join_keys=(),
        score=min(common_count * 10, 40),
        preferred_power=preferred_power,
        window_minutes=None,
        wind_bin_width_ms=None,
        temporal_overlap_pct=temporal_overlap,
        entity_overlap_pct=entity_overlap,
        resolution_ratio=resolution_ratio,
        common_feature_count=common_count,
        warnings=warnings or ["No deterministic merge key was found."],
        details=details,
        can_execute=False,
    )


def _first_valid(series: pd.Series) -> object:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return valid.iloc[0]


def _floor_timestamp(series: pd.Series, window_minutes: int) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.floor(f"{window_minutes}min")


def _add_wind_bin(series: pd.Series, bin_width: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.floor(values / bin_width) * bin_width


def _aggregate_frame(df: pd.DataFrame, keys: tuple[str, ...], prefix: str) -> pd.DataFrame:
    """Aggregate a frame to merge keys before joining.

    Numeric fields use median because it matches the mean for duplicated pairs and
    is more stable when future key groups have more than two rows.
    """
    clean = df.dropna(subset=list(keys)).copy()
    if clean.empty:
        return pd.DataFrame(columns=list(keys))

    agg_map = {}
    for column in clean.columns:
        if column in keys:
            continue
        if column.startswith("_merge_") and column not in keys:
            continue
        if pd.api.types.is_numeric_dtype(clean[column]):
            agg_map[column] = "median"
        else:
            agg_map[column] = _first_valid

    grouped = clean.groupby(list(keys), observed=True, dropna=False).agg(agg_map).reset_index()
    counts = clean.groupby(list(keys), observed=True, dropna=False).size().rename(f"{prefix}_source_rows").reset_index()
    grouped = counts.merge(grouped, on=list(keys), how="left")

    rename_map = {
        column: f"{prefix}_{column}"
        for column in grouped.columns
        if column not in keys and not column.endswith("_source_rows")
    }
    return grouped.rename(columns=rename_map)


def execute_merge_plan(
    plan: MergePlan,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    join_how: str = "inner",
) -> pd.DataFrame:
    """Run the merge described by a plan and add traceability flags."""

    if not plan.can_execute:
        raise ValueError("Merge plan is blocked and cannot be executed.")
    if join_how not in {"inner", "left", "right", "outer"}:
        raise ValueError("join_how must be one of: inner, left, right, outer.")

    left = left_df.copy()
    right = right_df.copy()

    if plan.strategy_id in {"temporal_entity", "temporal_window"}:
        if plan.window_minutes is None:
            raise ValueError("Temporal merge requires window_minutes.")
        left["_merge_time"] = _floor_timestamp(left["timestamp"], plan.window_minutes)
        right["_merge_time"] = _floor_timestamp(right["timestamp"], plan.window_minutes)
        keys = plan.join_keys
    elif plan.strategy_id == "wind_bin_curve":
        if plan.wind_bin_width_ms is None:
            raise ValueError("Wind-bin merge requires wind_bin_width_ms.")
        left["_wind_bin_ms"] = _add_wind_bin(left["wind_speed_ms"], plan.wind_bin_width_ms)
        right["_wind_bin_ms"] = _add_wind_bin(right["wind_speed_ms"], plan.wind_bin_width_ms)
        keys = plan.join_keys
    elif plan.strategy_id == "metadata_keys":
        keys = plan.join_keys
    else:
        raise ValueError(f"Unsupported merge strategy: {plan.strategy_id}")

    left_agg = _aggregate_frame(left, keys, "left")
    right_agg = _aggregate_frame(right, keys, "right")
    merged = pd.merge(left_agg, right_agg, on=list(keys), how=join_how, indicator="_merge_presence")

    merged.insert(0, "_merge_flag", plan.merge_flag)
    merged.insert(1, "_merge_compatibility_flag", plan.compatibility_flag)
    merged.insert(2, "_merge_key_flag", plan.key_flag)
    merged.insert(3, "_merge_strategy", plan.strategy_id)
    merged.insert(4, "_merge_score", plan.score)
    merged.insert(5, "_merge_left_dataset", plan.left_label)
    merged.insert(6, "_merge_right_dataset", plan.right_label)
    merged.insert(7, "_merge_warnings", " | ".join(plan.warnings))

    return merged
