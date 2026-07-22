"""Dataset registry and canonical loaders for wind turbine dashboards.

To add a dataset:
1. Create a loader that returns a dataframe with any known raw columns.
2. Rename/map the raw columns to the canonical names used below.
3. Register it in DATASET_SPECS and LOADERS.

The Streamlit app reads only this registry, so new datasets become available
without changing the dashboard layout.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


CANONICAL_FEATURES: dict[str, str] = {
    "timestamp": "Timestamp",
    "entity": "Turbine/location",
    "wind_speed_ms": "Wind speed (m/s)",
    "power_kw": "Power (kW)",
    "power_norm": "Power normalized",
    "wind_direction_deg": "Wind direction (deg)",
    "nac_direction_deg": "Nacelle direction (deg)",
    "yaw_error": "Yaw error (deg)",
    "wind_speed_max_ms": "Wind speed max (m/s)",
    "density_adjusted_wind_speed_ms": "Density adjusted wind speed (m/s)",
    "temperature_c": "Temperature (C)",
    "nacelle_temperature_c": "Nacelle temperature (C)",
    "cp": "Cp",
    "ct": "Ct",
    "thrust_kn": "Thrust (kN)",
    "rotor_diameter_m": "Rotor diameter (m)",
    "rated_power_kw": "Rated power (kW)",
    "available_capacity_kw": "Available capacity (kW)",
    "potential_power_kw": "Potential power (kW)",
    "energy_export_kwh": "Energy export (kWh)",
    "lost_production_total_kwh": "Lost production total (kWh)",
    "data_availability": "Data availability",
    "time_based_availability": "Time-based availability",
    "production_based_availability": "Production-based availability",
    "capacity_factor": "Capacity factor",
    "rotor_speed_rpm": "Rotor speed (RPM)",
    "generator_rpm": "Generator RPM",
    "pitch_angle_deg": "Pitch angle (deg)",
}

BASE_COLUMNS = ["dataset", "dataset_label", "source_type"]
METADATA_COLUMNS = ["site", "year", "turbine_model", "annual_production_kwh"]
QUALITY_COLUMNS = [
    "outlier_flag",
    "failure_component",
    "failure_remarks",
    "status",
    "status_code",
    "status_message",
    "service_contract_category",
    "iec_category",
]

VARIABLE_DESCRIPTIONS: dict[str, str] = {
    "timestamp": "Datetime of the measurement. Penmanshiel files are exported in UTC.",
    "entity": "Turbine, location, or reference curve represented by the row.",
    "wind_speed_ms": "Mean wind speed for the interval.",
    "power_kw": "Mean active power for the interval.",
    "power_norm": "Power divided by rated power or estimated rated power.",
    "wind_direction_deg": "Mean wind direction in degrees.",
    "nac_direction_deg": "Mean nacelle/yaw position in degrees.",
    "yaw_error": "Nacelle direction minus wind direction, normalized to +/- 180 deg.",
    "wind_speed_max_ms": "Maximum wind speed inside the interval when available.",
    "density_adjusted_wind_speed_ms": "Wind speed adjusted by air density when available.",
    "temperature_c": "Ambient or nacelle ambient temperature in Celsius.",
    "nacelle_temperature_c": "Nacelle internal temperature in Celsius.",
    "cp": "Power coefficient from reference turbine curves.",
    "ct": "Thrust coefficient from reference turbine curves.",
    "thrust_kn": "Thrust in kilonewtons from reference turbine curves.",
    "rotor_diameter_m": "Rotor diameter in meters.",
    "rated_power_kw": "Rated turbine power in kW.",
    "available_capacity_kw": "Capacity available for production in the interval.",
    "potential_power_kw": "Expected/potential power from the default power curve.",
    "energy_export_kwh": "Energy exported during the interval.",
    "lost_production_total_kwh": "Total lost production estimate for the interval.",
    "data_availability": "Greenbyte data availability indicator for the interval.",
    "time_based_availability": "Time-based contractual availability indicator.",
    "production_based_availability": "Production-based system availability indicator.",
    "capacity_factor": "Interval capacity factor.",
    "rotor_speed_rpm": "Rotor speed in RPM.",
    "generator_rpm": "Generator rotational speed in RPM.",
    "pitch_angle_deg": "Average blade pitch angle across available pitch sensors.",
    "site": "Wind farm or site name.",
    "year": "Calendar year inferred from the source file.",
    "turbine_model": "Turbine model/type reported in source metadata.",
    "annual_production_kwh": "Annual production reported in the status file metadata.",
    "status": "Greenbyte status severity/category.",
    "status_code": "Greenbyte status code.",
    "status_message": "Greenbyte status message.",
    "service_contract_category": "Greenbyte service contract category.",
    "iec_category": "IEC status category.",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    source_type: str
    description: str
    loader_name: str
    power_unit: str
    primary_files: tuple[str, ...]
    entity_label: str = "Turbina/local"
    expected_resolution_minutes: float | None = None
    notes: tuple[str, ...] = ()


DATASET_SPECS: dict[str, DatasetSpec] = {
    "edp": DatasetSpec(
        key="edp",
        label="EDP SCADA 2017",
        source_type="operational_scada",
        description="SCADA real de turbinas EDP T01, T06, T07 e T11.",
        loader_name="load_edp",
        power_unit="kW",
        primary_files=(
            "Wind_Turbine_SCADA_tratado_consolidado.csv",
            "Turbina_T01_data.csv",
        ),
        entity_label="Turbina",
        expected_resolution_minutes=10.0,
        notes=(
            "Dataset operacional com timestamp e eventos reais.",
            "power_norm e derivado por turbina usando potencia nominal estimada.",
        ),
    ),
    "kaggle": DatasetSpec(
        key="kaggle",
        label="Kaggle Wind Sites",
        source_type="weather_power_site",
        description="Dados meteorologicos por localizacao com potencia normalizada.",
        loader_name="load_kaggle",
        power_unit="normalized_0_1",
        primary_files=("Location1.csv",),
        entity_label="Localizacao",
        expected_resolution_minutes=60.0,
        notes=("Potencia ja vem normalizada entre 0 e 1.",),
    ),
    "nrel": DatasetSpec(
        key="nrel",
        label="NREL/IEA Reference Turbines",
        source_type="reference_curve",
        description="Curvas de referencia NREL, DTU, LEANWIND e IEA.",
        loader_name="load_nrel",
        power_unit="kW",
        primary_files=("NREL_Reference_5MW_126 (1).csv",),
        entity_label="Turbina de referencia",
        notes=("Dataset sem timestamp: serve como benchmark por curva de potencia.",),
    ),
    "penmanshiel": DatasetSpec(
        key="penmanshiel",
        label="Penmanshiel SCADA 2020+",
        source_type="operational_scada_status",
        description="SCADA Greenbyte por turbina com status operacional e metadados.",
        loader_name="load_penmanshiel",
        power_unit="kW",
        primary_files=("datasets*-penmanshiel",),
        entity_label="Turbina",
        expected_resolution_minutes=10.0,
        notes=(
            "Descobre automaticamente pastas datasets*-penmanshiel para permitir novos anos.",
            "Status Greenbyte e anexado pelo ultimo evento conhecido antes de cada timestamp.",
        ),
    ),
}


def _primary_file_exists(root: Path, file_name: str) -> bool:
    if any(char in file_name for char in "*?[]"):
        return any(root.glob(file_name))
    return (root / file_name).exists()


def available_dataset_keys(root: Path) -> list[str]:
    keys = []
    for key, spec in DATASET_SPECS.items():
        if any(_primary_file_exists(root, file_name) for file_name in spec.primary_files):
            keys.append(key)
    return keys


def load_dataset(key: str, root: Path) -> pd.DataFrame:
    if key not in DATASET_SPECS:
        raise KeyError(f"Dataset nao registrado: {key}")

    spec = DATASET_SPECS[key]
    loader = LOADERS[spec.loader_name]
    df = loader(root)
    return _finalize_dataset(df, spec)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _read_csv_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _parse_timestamp(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce", utc=True)
    return timestamps.dt.tz_convert(None)


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _estimate_power_norm(df: pd.DataFrame) -> pd.Series:
    if "power_kw" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    if "rated_power_kw" in df.columns and df["rated_power_kw"].notna().any():
        denom = df["rated_power_kw"].replace(0, np.nan)
        return df["power_kw"] / denom

    if "entity" not in df.columns:
        denom = df["power_kw"].quantile(0.995)
        return df["power_kw"] / denom if pd.notna(denom) and denom > 0 else pd.Series(np.nan, index=df.index)

    rated = df.groupby("entity")["power_kw"].transform(
        lambda s: s.dropna().clip(lower=0).quantile(0.995)
    )
    rated = rated.replace(0, np.nan)
    return df["power_kw"] / rated


def _finalize_dataset(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    df = _normalize_columns(df)
    df["dataset"] = spec.key
    df["dataset_label"] = spec.label
    df["source_type"] = spec.source_type

    if "entity" not in df.columns:
        df["entity"] = spec.label
    df["entity"] = df["entity"].astype(str).str.strip()

    if "timestamp" in df.columns:
        df["timestamp"] = _parse_timestamp(df["timestamp"])

    numeric_cols = [
        "wind_speed_ms",
        "wind_speed_max_ms",
        "power_kw",
        "power_norm",
        "wind_direction_deg",
        "nac_direction_deg",
        "wind_speed_max_ms",
        "density_adjusted_wind_speed_ms",
        "temperature_c",
        "nacelle_temperature_c",
        "cp",
        "ct",
        "thrust_kn",
        "rotor_diameter_m",
        "rated_power_kw",
        "available_capacity_kw",
        "potential_power_kw",
        "energy_export_kwh",
        "lost_production_total_kwh",
        "data_availability",
        "time_based_availability",
        "production_based_availability",
        "capacity_factor",
        "rotor_speed_rpm",
        "generator_rpm",
        "pitch_angle_deg",
        "annual_production_kwh",
    ]
    df = _coerce_numeric(df, numeric_cols)

    if "yaw_error" not in df.columns and {"nac_direction_deg", "wind_direction_deg"}.issubset(df.columns):
        yaw_delta = df["nac_direction_deg"] - df["wind_direction_deg"]
        df["yaw_error"] = ((yaw_delta + 180) % 360) - 180

    if "power_norm" not in df.columns or df["power_norm"].notna().sum() == 0:
        df["power_norm"] = _estimate_power_norm(df)

    keep_columns = [
        col
        for col in BASE_COLUMNS + list(CANONICAL_FEATURES.keys()) + METADATA_COLUMNS + QUALITY_COLUMNS
        if col in df.columns
    ]
    return df.loc[:, keep_columns].copy()


def load_edp(root: Path) -> pd.DataFrame:
    rename_map = {
        "Turbine_ID": "entity",
        "Timestamp": "timestamp",
        "Amb_WindSpeed_Avg": "wind_speed_ms",
        "Amb_WindSpeed_Max": "wind_speed_max_ms",
        "Amb_Temp_Avg": "temperature_c",
        "Nac_Direction_Avg": "nac_direction_deg",
        "Amb_WindDir_Abs_Avg": "wind_direction_deg",
        "Grd_Prod_Pwr_Avg": "power_kw",
        "Pwr_Avg": "power_kw",
        "Power": "power_kw",
        "ActivePower": "power_kw",
        "outlier_flag": "outlier_flag",
    }

    consolidated = root / "Wind_Turbine_SCADA_tratado_consolidado.csv"
    if consolidated.exists():
        raw_cols = _read_csv_columns(consolidated)
        wanted = set(rename_map.keys())
        usecols = [col for col in raw_cols if col.strip() in wanted]
        df = pd.read_csv(consolidated, usecols=usecols, low_memory=False)
        df = _normalize_columns(df)
        return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    frames = []
    turbine_files = [
        ("Turbina_T01_data.csv", "T01"),
        ("Turbina_T06_data.csv", "T06"),
        ("Turbina_T07_data.csv", "T07"),
        ("Turbina_T11_data.csv", "T11"),
    ]
    for file_name, turbine_id in turbine_files:
        path = root / file_name
        if not path.exists():
            continue
        df = pd.read_csv(path, sep=";", low_memory=False)
        df = _normalize_columns(df)
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        if "entity" not in df.columns:
            df["entity"] = turbine_id
        frames.append(df)

    if not frames:
        raise FileNotFoundError("Nenhum arquivo EDP encontrado.")

    return pd.concat(frames, ignore_index=True, sort=False)


def load_kaggle(root: Path) -> pd.DataFrame:
    rename_map = {
        "Time": "timestamp",
        "Location": "entity",
        "windspeed_100m": "wind_speed_ms",
        "winddirection_100m": "wind_direction_deg",
        "temperature_2m": "temperature_c",
        "Power": "power_norm",
    }
    frames = []
    for idx in range(1, 5):
        location = f"Location{idx}"
        path = root / f"{location}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["Location"] = location
        df = _normalize_columns(df)
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        frames.append(df)

    if not frames:
        raise FileNotFoundError("Nenhum arquivo Kaggle Location*.csv encontrado.")

    return pd.concat(frames, ignore_index=True, sort=False)


def load_nrel(root: Path) -> pd.DataFrame:
    turbines = [
        ("NREL 5MW", "NREL_Reference_5MW_126 (1).csv", 5000, 126),
        ("DTU 10MW", "DTU_Reference_v1_10MW_178.csv", 10000, 178),
        ("LEANWIND 8MW", "LEANWIND_Reference_8MW_164.csv", 8000, 164),
        ("IEA 3.4MW", "IEA_Reference_3.4MW_130.csv", 3400, 130),
        ("IEA 10MW", "IEA_Reference_10MW_198.csv", 10000, 198),
        ("IEA 15MW", "IEA_Reference_15MW_240.csv", 15000, 240),
    ]
    rename_map = {
        "Wind Speed [m/s]": "wind_speed_ms",
        "Power [kW]": "power_kw",
        "Cp [-]": "cp",
        "Ct [-]": "ct",
        "Thrust [kN]": "thrust_kn",
    }

    frames = []
    for entity, file_name, rated_power_kw, rotor_diameter_m in turbines:
        path = root / file_name
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df = _normalize_columns(df)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        df["entity"] = entity
        df["rated_power_kw"] = rated_power_kw
        df["rotor_diameter_m"] = rotor_diameter_m
        frames.append(df)

    if not frames:
        raise FileNotFoundError("Nenhum arquivo de referencia NREL/IEA encontrado.")

    return pd.concat(frames, ignore_index=True, sort=False)


PENMANSHIEL_RENAME_MAP = {
    "Date and time": "timestamp",
    "Wind speed (m/s)": "wind_speed_ms",
    "Wind speed, Maximum (m/s)": "wind_speed_max_ms",
    "Density adjusted wind speed (m/s)": "density_adjusted_wind_speed_ms",
    "Wind direction (deg)": "wind_direction_deg",
    "Nacelle position (deg)": "nac_direction_deg",
    "Energy Export (kWh)": "energy_export_kwh",
    "Lost Production Total (kWh)": "lost_production_total_kwh",
    "Power (kW)": "power_kw",
    "Potential power default PC (kW)": "potential_power_kw",
    "Available Capacity for Production (kW)": "available_capacity_kw",
    "Nacelle ambient temperature (C)": "temperature_c",
    "Nacelle temperature (C)": "nacelle_temperature_c",
    "Rotor speed (RPM)": "rotor_speed_rpm",
    "Generator RPM (RPM)": "generator_rpm",
    "Capacity factor": "capacity_factor",
    "Data Availability": "data_availability",
    "Time-based Contractual Avail.": "time_based_availability",
    "Production-based System Avail.": "production_based_availability",
    "Blade angle (pitch position) A (deg)": "pitch_a_deg",
    "Blade angle (pitch position) B (deg)": "pitch_b_deg",
    "Blade angle (pitch position) C (deg)": "pitch_c_deg",
}


def _clean_greenbyte_column_name(name: str) -> str:
    return (
        str(name)
        .replace("Â°C", "C")
        .replace("°C", "C")
        .replace("Â°", "deg")
        .replace("°", "deg")
        .strip()
    )


def _read_penmanshiel_header(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("# Date and time,"):
                raw_header = line[1:].lstrip().strip()
                columns = next(csv.reader([raw_header]))
                columns = [_clean_greenbyte_column_name(col) for col in columns]
                return columns, line_number
    raise ValueError(f"Cabecalho de dados Penmanshiel nao encontrado em {path.name}.")


def _read_penmanshiel_metadata(path: Path) -> dict[str, object]:
    metadata: dict[str, object] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            text = line[1:].strip()
            if text.startswith("Turbine:"):
                metadata["entity"] = text.split(":", 1)[1].strip()
            elif text.startswith("Turbine type:"):
                model = text.split(":", 1)[1].strip()
                metadata["turbine_model"] = model.split(" (", 1)[0].strip()
            elif text.startswith("Time interval:"):
                metadata["time_interval"] = text.split(":", 1)[1].strip()
                match = re.search(r"(\d{4})-\d{2}-\d{2}", text)
                if match:
                    metadata["year"] = int(match.group(1))
            elif "Sum production:" in text:
                match = re.search(r"Sum production:\s*([0-9.eE+-]+)", text)
                if match:
                    metadata["annual_production_kwh"] = float(match.group(1))
    return metadata


def _penmanshiel_turbine_id(path: Path) -> str | None:
    match = re.search(r"Penmanshiel_(\d{2})_", path.name)
    return match.group(1) if match else None


def _infer_rotor_diameter_m(model: object) -> float | None:
    if not isinstance(model, str):
        return None
    match = re.search(r"MM\s*(\d+)", model, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _discover_penmanshiel_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("datasets*-penmanshiel") if path.is_dir())


def _read_penmanshiel_status(path: Path) -> pd.DataFrame:
    status_cols = [
        "Timestamp start",
        "Timestamp end",
        "Status",
        "Code",
        "Message",
        "Service contract category",
        "IEC category",
    ]
    df = pd.read_csv(
        path,
        comment="#",
        usecols=lambda col: col in status_cols,
        na_values=["-", "NaN", ""],
        low_memory=False,
    )
    df = _normalize_columns(df)
    df = df.rename(
        columns={
            "Timestamp start": "status_timestamp",
            "Timestamp end": "status_end",
            "Status": "status",
            "Code": "status_code",
            "Message": "status_message",
            "Service contract category": "service_contract_category",
            "IEC category": "iec_category",
        }
    )
    df["status_timestamp"] = pd.to_datetime(df["status_timestamp"], errors="coerce")
    df = df.dropna(subset=["status_timestamp"]).sort_values("status_timestamp")
    return df


def _attach_penmanshiel_status(df: pd.DataFrame, status_path: Path | None) -> pd.DataFrame:
    if status_path is None or not status_path.exists() or "timestamp" not in df.columns:
        return df

    status = _read_penmanshiel_status(status_path)
    if status.empty:
        return df

    out = df.copy()
    out["_merge_timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.sort_values("_merge_timestamp")
    merged = pd.merge_asof(
        out,
        status,
        left_on="_merge_timestamp",
        right_on="status_timestamp",
        direction="backward",
    )
    return merged.drop(columns=["_merge_timestamp", "status_timestamp", "status_end"], errors="ignore")


def load_penmanshiel(root: Path) -> pd.DataFrame:
    frames = []
    for folder in _discover_penmanshiel_dirs(root):
        for data_path in sorted(folder.glob("Turbine_Data_Penmanshiel_*.csv")):
            columns, skiprows = _read_penmanshiel_header(data_path)
            usecols = [col for col in PENMANSHIEL_RENAME_MAP if col in columns]
            if not usecols:
                continue

            df = pd.read_csv(
                data_path,
                header=None,
                names=columns,
                skiprows=skiprows,
                usecols=usecols,
                na_values=["NaN"],
                low_memory=False,
            )
            df = _normalize_columns(df)
            df = df.rename(columns={k: v for k, v in PENMANSHIEL_RENAME_MAP.items() if k in df.columns})

            metadata = _read_penmanshiel_metadata(data_path)
            turbine_id = _penmanshiel_turbine_id(data_path)
            entity = metadata.get("entity") or (f"Penmanshiel {turbine_id}" if turbine_id else data_path.stem)
            df["entity"] = str(entity)
            df["site"] = "Penmanshiel"
            df["turbine_model"] = metadata.get("turbine_model", "Senvion MM82")
            if "year" in metadata:
                df["year"] = metadata["year"]
            else:
                match = re.search(r"_(\d{4})-\d{2}-\d{2}_", data_path.name)
                if match:
                    df["year"] = int(match.group(1))
            if "annual_production_kwh" in metadata:
                df["annual_production_kwh"] = metadata["annual_production_kwh"]

            rotor_diameter = _infer_rotor_diameter_m(df["turbine_model"].iloc[0])
            if rotor_diameter is not None:
                df["rotor_diameter_m"] = rotor_diameter

            pitch_cols = [col for col in ["pitch_a_deg", "pitch_b_deg", "pitch_c_deg"] if col in df.columns]
            if pitch_cols:
                df[pitch_cols] = df[pitch_cols].apply(pd.to_numeric, errors="coerce")
                df["pitch_angle_deg"] = df[pitch_cols].mean(axis=1)
                df = df.drop(columns=pitch_cols)

            if "available_capacity_kw" in df.columns:
                available_capacity = pd.to_numeric(df["available_capacity_kw"], errors="coerce")
                rated_power = available_capacity.dropna().quantile(0.99)
                if pd.notna(rated_power) and rated_power > 0:
                    df["rated_power_kw"] = float(rated_power)

            status_path = None
            if turbine_id:
                matches = sorted(folder.glob(f"Status_Penmanshiel_{turbine_id}_*.csv"))
                status_path = matches[0] if matches else None
            if status_path is not None:
                status_metadata = _read_penmanshiel_metadata(status_path)
                if "annual_production_kwh" in status_metadata:
                    df["annual_production_kwh"] = status_metadata["annual_production_kwh"]
                df = _attach_penmanshiel_status(df, status_path)

            frames.append(df)

    if not frames:
        raise FileNotFoundError("Nenhum arquivo Penmanshiel encontrado em datasets*-penmanshiel.")

    return pd.concat(frames, ignore_index=True, sort=False)


LOADERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    "load_edp": load_edp,
    "load_kaggle": load_kaggle,
    "load_nrel": load_nrel,
    "load_penmanshiel": load_penmanshiel,
}
