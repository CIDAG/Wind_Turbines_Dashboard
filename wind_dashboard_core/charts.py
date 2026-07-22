"""Plotly chart builders for the adaptive wind dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .metrics import SEVERITY_TO_NUM


DATASET_COLORS = {
    "EDP SCADA 2017": "#1f77b4",
    "Kaggle Wind Sites": "#2ca02c",
    "NREL/IEA Reference Turbines": "#d62728",
    "Penmanshiel SCADA 2020+": "#9467bd",
}


def style_figure(fig: go.Figure, title: str, height: int | None = None) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        font={"family": "Arial", "size": 12},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 20, "r": 20, "t": 70, "b": 25},
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def balanced_sample(df: pd.DataFrame, limit: int, by: str = "dataset_label") -> pd.DataFrame:
    if len(df) <= limit:
        return df.copy()
    if by not in df.columns or df[by].nunique() <= 1:
        return df.sample(limit, random_state=42).copy()

    per_group = max(1, limit // df[by].nunique())
    samples = []
    for _, group in df.groupby(by, observed=True):
        samples.append(group.sample(min(len(group), per_group), random_state=42))
    sampled = pd.concat(samples, ignore_index=True)
    if len(sampled) > limit:
        sampled = sampled.sample(limit, random_state=42)
    return sampled


def feature_coverage_heatmap(matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        matrix.astype(int),
        text_auto=True,
        color_continuous_scale="Blues",
        zmin=0,
        zmax=1,
        labels={"x": "Dataset", "y": "Feature canonica", "color": "Disponivel"},
        aspect="auto",
    )
    fig.update_xaxes(side="top")
    return style_figure(fig, "Cobertura de features por dataset", height=480)


def pairwise_severity_heatmap(pairwise: pd.DataFrame) -> go.Figure:
    if pairwise.empty:
        return style_figure(go.Figure(), "Compatibilidade par-a-par")

    cols = ["temporal", "resolution", "power_scale", "feature_overlap", "overall"]
    numeric = pairwise.set_index("pair")[cols].apply(lambda col: col.map(SEVERITY_TO_NUM)).astype(float)
    fig = px.imshow(
        numeric,
        text_auto=True,
        color_continuous_scale="YlOrRd",
        zmin=0,
        zmax=2,
        labels={"x": "Cheque", "y": "Par", "color": "Severidade"},
        aspect="auto",
    )
    fig.update_xaxes(side="top")
    return style_figure(fig, "Matriz de severidade da compatibilidade", height=360)


def dataset_correlation_heatmap(
    df: pd.DataFrame,
    dataset_label: str,
    sample_limit: int = 50_000,
) -> go.Figure:
    if df.empty:
        return style_figure(go.Figure(), f"Heatmap de correlacao - {dataset_label}")

    sample_df = df.sample(min(len(df), sample_limit), random_state=42) if len(df) > sample_limit else df.copy()
    numeric = sample_df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.loc[:, numeric.notna().sum() >= 3]
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]

    if numeric.shape[1] < 2:
        return style_figure(go.Figure(), f"Heatmap de correlacao - {dataset_label}")

    corr = numeric.corr(method="pearson").round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels={"x": "Feature", "y": "Feature", "color": "r"},
        aspect="auto",
    )
    fig.update_xaxes(side="top")
    return style_figure(
        fig,
        f"Heatmap de correlacao - {dataset_label} ({len(sample_df):,} linhas)",
        height=max(420, 34 * len(corr.columns)),
    )


def power_scatter(df: pd.DataFrame, y_col: str, sample_limit: int) -> go.Figure:
    needed = ["wind_speed_ms", y_col, "dataset_label", "entity"]
    plot_df = df.dropna(subset=[col for col in needed if col in df.columns]).copy()
    if plot_df.empty:
        return style_figure(go.Figure(), "Curva de potencia")

    plot_df = balanced_sample(plot_df, sample_limit, by="dataset_label")
    hover_cols = [col for col in ["entity", "timestamp", "yaw_error", "temperature_c"] if col in plot_df.columns]

    fig = px.scatter(
        plot_df,
        x="wind_speed_ms",
        y=y_col,
        color="dataset_label",
        symbol="source_type" if "source_type" in plot_df.columns else None,
        color_discrete_map=DATASET_COLORS,
        opacity=0.45,
        hover_data=hover_cols,
        labels={
            "wind_speed_ms": "Velocidade do vento (m/s)",
            "power_kw": "Potencia (kW)",
            "power_norm": "Potencia normalizada",
            "dataset_label": "",
            "source_type": "",
        },
    )
    fig.update_traces(marker={"size": 5})
    title = "Curva de potencia comparada"
    return style_figure(fig, title, height=560)


def empirical_power_curve(df: pd.DataFrame, y_col: str) -> go.Figure:
    required = ["wind_speed_ms", y_col, "dataset_label", "entity"]
    curve_df = df.dropna(subset=[col for col in required if col in df.columns]).copy()
    if curve_df.empty:
        return style_figure(go.Figure(), "Curva empirica")

    max_speed = min(35.0, max(12.0, float(curve_df["wind_speed_ms"].quantile(0.995)) + 1.0))
    bins = np.arange(0, max_speed + 0.5, 0.5)
    curve_df["wind_bin"] = pd.cut(curve_df["wind_speed_ms"], bins=bins, labels=bins[:-1])
    grouped = (
        curve_df.groupby(["dataset_label", "entity", "wind_bin"], observed=True)[y_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped = grouped[grouped["count"] >= 3].copy()
    grouped["wind_bin"] = pd.to_numeric(grouped["wind_bin"], errors="coerce")
    grouped["series"] = grouped["dataset_label"] + " | " + grouped["entity"].astype(str)

    fig = px.line(
        grouped,
        x="wind_bin",
        y="mean",
        color="dataset_label",
        line_group="series",
        hover_name="series",
        color_discrete_map=DATASET_COLORS,
        labels={
            "wind_bin": "Velocidade do vento (m/s)",
            "mean": "Media",
            "dataset_label": "",
            "count": "N",
        },
    )
    fig.update_traces(line={"width": 2})
    return style_figure(fig, "Curva empirica media por bin de vento", height=520)


def temporal_monthly_chart(df: pd.DataFrame, y_col: str) -> go.Figure:
    if "timestamp" not in df.columns:
        return style_figure(go.Figure(), "Serie temporal mensal")

    ts_df = df.dropna(subset=["timestamp", y_col]).copy()
    if ts_df.empty:
        return style_figure(go.Figure(), "Serie temporal mensal")

    ts_df["month"] = ts_df["timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        ts_df.groupby(["dataset_label", "month"], observed=True)[y_col]
        .mean()
        .reset_index()
    )
    fig = px.line(
        monthly,
        x="month",
        y=y_col,
        color="dataset_label",
        markers=True,
        color_discrete_map=DATASET_COLORS,
        labels={
            "month": "Mes",
            "power_kw": "Potencia media (kW)",
            "power_norm": "Potencia media normalizada",
            "dataset_label": "Dataset",
        },
    )
    return style_figure(fig, "Serie temporal mensal por dataset", height=460)


def profile_quality_bar(profile_table: pd.DataFrame) -> go.Figure:
    if profile_table.empty:
        return style_figure(go.Figure(), "Qualidade dos datasets")

    melted = profile_table.melt(
        id_vars=["dataset_label"],
        value_vars=["missing_cells_pct", "duplicated_rows_pct"],
        var_name="metric",
        value_name="percent",
    )
    melted["metric"] = melted["metric"].replace(
        {
            "missing_cells_pct": "Missing cells (%)",
            "duplicated_rows_pct": "Duplicated rows (%)",
        }
    )
    fig = px.bar(
        melted,
        x="dataset_label",
        y="percent",
        color="metric",
        barmode="group",
        labels={"dataset_label": "Dataset", "percent": "%", "metric": "Metrica"},
    )
    return style_figure(fig, "Missing e duplicados por dataset", height=420)


def entity_counts_bar(df: pd.DataFrame) -> go.Figure:
    if df.empty or "entity" not in df.columns:
        return style_figure(go.Figure(), "Registros por entidade")
    counts = (
        df.groupby(["dataset_label", "entity"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    fig = px.bar(
        counts,
        x="entity",
        y="rows",
        color="dataset_label",
        color_discrete_map=DATASET_COLORS,
        labels={"entity": "Entidade", "rows": "Registros", "dataset_label": "Dataset"},
    )
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    fig = style_figure(fig, "Registros por entidade", height=420)
    fig.update_layout(legend_title_text="")
    return fig
