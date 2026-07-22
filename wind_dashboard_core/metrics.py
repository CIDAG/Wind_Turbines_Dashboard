"""Metrics for dataset coverage, quality, and compatibility."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .loaders import CANONICAL_FEATURES, DatasetSpec


SEVERITY_TO_NUM = {"ok": 0, "alerta": 1, "critico": 2}


def infer_resolution_minutes(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns:
        return np.nan

    timestamped = df.dropna(subset=["timestamp"]).copy()
    if timestamped.empty:
        return np.nan

    resolutions = []
    if "entity" in timestamped.columns:
        groups = timestamped.groupby("entity", observed=True)
    else:
        groups = [("__all__", timestamped)]

    for _, group in groups:
        ts = pd.to_datetime(group["timestamp"], errors="coerce").dropna()
        ts = ts.drop_duplicates().sort_values()
        if len(ts) < 3:
            continue
        diffs = ts.diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if diffs.empty:
            continue
        minutes = diffs.dt.total_seconds().median() / 60.0
        if pd.notna(minutes) and minutes > 0:
            resolutions.append(minutes)

    if not resolutions:
        return np.nan
    return round(float(np.median(resolutions)), 2)


def dataset_profile(df: pd.DataFrame, spec: DatasetSpec) -> dict:
    n_rows = int(len(df))
    canonical_present = [col for col in CANONICAL_FEATURES if col in df.columns]
    coverage = {
        col: bool(col in df.columns and df[col].notna().any())
        for col in CANONICAL_FEATURES
    }

    if n_rows and canonical_present:
        missing_pct = round(float(df[canonical_present].isna().mean().mean() * 100), 2)
        duplicated_pct = round(float(df[canonical_present].duplicated().mean() * 100), 2)
    else:
        missing_pct = np.nan
        duplicated_pct = np.nan

    if "timestamp" in df.columns and df["timestamp"].notna().any():
        ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
        coverage_start = ts.min()
        coverage_end = ts.max()
        resolution_minutes = infer_resolution_minutes(df)
    else:
        coverage_start = pd.NaT
        coverage_end = pd.NaT
        resolution_minutes = np.nan

    n_entities = int(df["entity"].nunique()) if "entity" in df.columns else 0
    power_modes = []
    if coverage.get("power_kw", False):
        power_modes.append("kW")
    if coverage.get("power_norm", False):
        power_modes.append("normalizada")

    return {
        "dataset": spec.key,
        "dataset_label": spec.label,
        "source_type": spec.source_type,
        "rows": n_rows,
        "entities": n_entities,
        "missing_cells_pct": missing_pct,
        "duplicated_rows_pct": duplicated_pct,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "resolution_minutes": resolution_minutes,
        "power_unit": spec.power_unit,
        "power_modes": ", ".join(power_modes) if power_modes else "n/a",
        "coverage": coverage,
        "notes": " | ".join(spec.notes),
    }


def build_profiles(datasets: dict[str, pd.DataFrame], specs: dict[str, DatasetSpec]) -> tuple[dict[str, dict], pd.DataFrame]:
    profiles = {
        key: dataset_profile(df, specs[key])
        for key, df in datasets.items()
    }
    table_rows = []
    for profile in profiles.values():
        row = {
            key: value
            for key, value in profile.items()
            if key not in {"coverage", "notes"}
        }
        table_rows.append(row)
    return profiles, pd.DataFrame(table_rows)


def coverage_matrix(profiles: dict[str, dict]) -> pd.DataFrame:
    matrix = pd.DataFrame(
        {
            profile["dataset_label"]: {
                CANONICAL_FEATURES[field]: int(profile["coverage"].get(field, False))
                for field in CANONICAL_FEATURES
            }
            for profile in profiles.values()
        }
    )
    return matrix


def _max_severity(values: list[str]) -> str:
    if not values:
        return "ok"
    inverse = {value: key for key, value in SEVERITY_TO_NUM.items()}
    return inverse[max(SEVERITY_TO_NUM.get(value, 0) for value in values)]


def pairwise_compatibility(profiles: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for left_key, right_key in combinations(profiles.keys(), 2):
        left = profiles[left_key]
        right = profiles[right_key]
        left_cov = left["coverage"]
        right_cov = right["coverage"]

        common_fields = [
            field for field in CANONICAL_FEATURES
            if left_cov.get(field, False) and right_cov.get(field, False)
        ]

        both_timestamp = left_cov.get("timestamp", False) and right_cov.get("timestamp", False)
        if both_timestamp:
            temporal = "ok"
            if pd.notna(left["resolution_minutes"]) and pd.notna(right["resolution_minutes"]):
                ratio = max(left["resolution_minutes"], right["resolution_minutes"]) / max(
                    min(left["resolution_minutes"], right["resolution_minutes"]),
                    1e-9,
                )
                resolution = "alerta" if ratio > 2.0 else "ok"
                resolution_ratio = round(float(ratio), 2)
            else:
                resolution = "alerta"
                resolution_ratio = np.nan
        else:
            temporal = "alerta" if "wind_speed_ms" in common_fields else "critico"
            resolution = "alerta"
            resolution_ratio = np.nan

        if "power_kw" in common_fields:
            power_scale = "ok"
            preferred_power = "power_kw"
        elif "power_norm" in common_fields:
            power_scale = "ok"
            preferred_power = "power_norm"
        else:
            power_scale = "critico"
            preferred_power = "n/a"

        feature_overlap = "ok" if len(common_fields) >= 4 else "alerta" if len(common_fields) >= 2 else "critico"

        if both_timestamp and {"timestamp", "entity"}.issubset(common_fields):
            strategy = "Comparar por janela temporal e entidade; agregar antes de unir diretamente."
        elif {"wind_speed_ms", preferred_power}.issubset(common_fields):
            strategy = "Comparar por curva vento-potencia."
        elif "wind_speed_ms" in common_fields:
            strategy = "Comparar distribuicao de vento; potencia precisa de normalizacao."
        else:
            strategy = "Merge direto nao recomendado."

        overall = _max_severity([temporal, resolution, power_scale, feature_overlap])

        rows.append(
            {
                "pair": f"{left['dataset_label']} x {right['dataset_label']}",
                "temporal": temporal,
                "resolution": resolution,
                "resolution_ratio": resolution_ratio,
                "power_scale": power_scale,
                "preferred_power": preferred_power,
                "feature_overlap": feature_overlap,
                "common_fields": ", ".join(CANONICAL_FEATURES[field] for field in common_fields),
                "strategy": strategy,
                "overall": overall,
            }
        )

    return pd.DataFrame(rows)


def common_feature_count(profiles: dict[str, dict]) -> int:
    if not profiles:
        return 0
    common = None
    for profile in profiles.values():
        fields = {field for field, present in profile["coverage"].items() if present}
        common = fields if common is None else common.intersection(fields)
    return len(common or set())
