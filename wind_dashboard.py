"""Adaptive Streamlit dashboard for wind turbine open datasets."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from wind_dashboard_core.charts import (
    balanced_sample,
    dataset_correlation_heatmap,
    empirical_power_curve,
    entity_counts_bar,
    feature_coverage_heatmap,
    pairwise_severity_heatmap,
    power_scatter,
    profile_quality_bar,
    temporal_monthly_chart,
)
from wind_dashboard_core.loaders import (
    CANONICAL_FEATURES,
    DATASET_SPECS,
    VARIABLE_DESCRIPTIONS,
    available_dataset_keys,
    load_dataset,
)
from wind_dashboard_core.merge_tools import (
    MERGE_BLOCKED,
    execute_merge_plan,
    plan_compatible_merge,
)
from wind_dashboard_core.metrics import (
    build_profiles,
    common_feature_count,
    coverage_matrix,
    pairwise_compatibility,
)


ROOT = Path(__file__).resolve().parent


st.set_page_config(
    page_title="Wind Dataset Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=True)
def cached_load_dataset(key: str) -> pd.DataFrame:
    return load_dataset(key, ROOT)


def _format_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["coverage_start", "coverage_end"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
            out[col] = out[col].fillna("n/a")
    return out


def _first_valid(series: pd.Series) -> object:
    valid = series.dropna()
    if valid.empty:
        return np.nan
    return valid.iloc[0]


def _variable_description_table(columns: pd.Index | list[str]) -> pd.DataFrame:
    rows = []
    known_columns = set(columns)
    ordered_columns = [
        col
        for col in list(CANONICAL_FEATURES) + ["site", "year", "turbine_model", "annual_production_kwh"]
        + ["status", "status_code", "status_message", "service_contract_category", "iec_category"]
        if col in known_columns
    ]
    for col in ordered_columns:
        rows.append(
            {
                "column": col,
                "label": CANONICAL_FEATURES.get(col, col.replace("_", " ").title()),
                "description": VARIABLE_DESCRIPTIONS.get(col, "Sem descricao cadastrada."),
            }
        )
    return pd.DataFrame(rows)


def _apply_global_filters(df: pd.DataFrame, sample_limit: int) -> tuple[pd.DataFrame, str]:
    with st.sidebar:
        st.divider()
        st.subheader("Filtros globais")

        entity_options = sorted(df["entity_display"].dropna().unique().tolist())
        selected_entities = st.multiselect(
            "Entidades",
            options=entity_options,
            default=entity_options,
        )

        keep_reference = True
        if "timestamp" in df.columns and df["timestamp"].notna().any():
            ts = df["timestamp"].dropna()
            date_range = st.date_input(
                "Periodo",
                value=(ts.min().date(), ts.max().date()),
                min_value=ts.min().date(),
                max_value=ts.max().date(),
            )
            keep_reference = st.checkbox(
                "Manter datasets sem timestamp",
                value=True,
                help="Mantem curvas de referencia, como NREL, mesmo quando ha filtro temporal.",
            )
        else:
            date_range = None

        if "wind_speed_ms" in df.columns and df["wind_speed_ms"].notna().any():
            max_speed = float(min(35.0, max(10.0, df["wind_speed_ms"].quantile(0.995))))
            speed_range = st.slider(
                "Velocidade do vento (m/s)",
                min_value=0.0,
                max_value=max_speed,
                value=(0.0, max_speed),
                step=0.5,
            )
        else:
            speed_range = None

        positive_power = st.checkbox("Graficos com potencia kW >= 0", value=True)
        power_mode = st.radio(
            "Escala de potencia",
            options=["Normalizada", "kW quando disponivel"],
            horizontal=False,
        )

    filtered = df.copy()
    if selected_entities:
        filtered = filtered[filtered["entity_display"].isin(selected_entities)]

    if date_range and len(date_range) == 2 and "timestamp" in filtered.columns:
        start_ts = pd.Timestamp(date_range[0])
        end_ts = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        has_ts = filtered["timestamp"].notna()
        in_range = filtered["timestamp"].between(start_ts, end_ts)
        filtered = filtered[(~has_ts & keep_reference) | (has_ts & in_range)]

    if speed_range and "wind_speed_ms" in filtered.columns:
        filtered = filtered[
            filtered["wind_speed_ms"].isna()
            | filtered["wind_speed_ms"].between(speed_range[0], speed_range[1])
        ]

    if positive_power and "power_kw" in filtered.columns:
        filtered = filtered[filtered["power_kw"].isna() | (filtered["power_kw"] >= 0)]

    y_col = "power_norm" if power_mode == "Normalizada" else "power_kw"
    if y_col == "power_kw" and filtered.get("power_kw", pd.Series(dtype=float)).notna().sum() == 0:
        y_col = "power_norm"

    if len(filtered) > sample_limit * 20:
        st.sidebar.caption(
            f"Dataset filtrado tem {len(filtered):,} linhas; graficos usam amostragem balanceada."
        )

    return filtered, y_col


def _dbscan_feature_sample(df: pd.DataFrame, y_col: str, sample_limit: int) -> pd.DataFrame:
    needed = ["wind_speed_ms", y_col]
    clean = df.dropna(subset=needed).copy()
    return balanced_sample(clean, sample_limit, by="dataset_label")


def _scaled_dbscan_features(df: pd.DataFrame, y_col: str) -> np.ndarray:
    return StandardScaler().fit_transform(df[["wind_speed_ms", y_col]].to_numpy())


def _suggest_dbscan_eps(sample_df: pd.DataFrame, y_col: str, min_samples: int) -> tuple[float, pd.DataFrame]:
    if len(sample_df) < max(20, min_samples + 1):
        return 0.35, pd.DataFrame()

    x_scaled = _scaled_dbscan_features(sample_df, y_col)
    n_neighbors = min(max(2, min_samples), len(sample_df))
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(x_scaled).kneighbors(x_scaled)
    k_distances = np.sort(distances[:, -1])

    x_norm = np.linspace(0.0, 1.0, len(k_distances))
    y_min = float(k_distances.min())
    y_max = float(k_distances.max())
    if np.isclose(y_min, y_max):
        suggested_eps = max(0.05, round(y_max, 3))
    else:
        y_norm = (k_distances - y_min) / (y_max - y_min)
        points = np.column_stack([x_norm, y_norm])
        line_start = points[0]
        line_end = points[-1]
        line_vec = line_end - line_start
        line_norm = np.linalg.norm(line_vec)
        if line_norm == 0:
            elbow_idx = int(0.9 * (len(k_distances) - 1))
        else:
            distances_to_line = np.abs(
                line_vec[0] * (line_start[1] - points[:, 1])
                - line_vec[1] * (line_start[0] - points[:, 0])
            ) / line_norm
            elbow_idx = int(np.argmax(distances_to_line))
        suggested_eps = float(k_distances[elbow_idx])

    suggested_eps = float(np.clip(suggested_eps, 0.05, 2.0))
    kdist_df = pd.DataFrame(
        {
            "ordered_point": np.arange(1, len(k_distances) + 1),
            "k_distance": k_distances,
        }
    )
    return round(suggested_eps, 3), kdist_df


def _run_dbscan_sample(
    df: pd.DataFrame,
    y_col: str,
    sample_limit: int,
    eps: float,
    min_samples: int,
) -> pd.DataFrame:
    clean = _dbscan_feature_sample(df, y_col, sample_limit)
    if len(clean) < 20:
        clean["dbscan_status"] = "Amostra insuficiente"
        return clean

    x_scaled = _scaled_dbscan_features(clean, y_col)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x_scaled)
    clean["dbscan_label"] = labels
    clean["dbscan_status"] = ["Anomalia" if label == -1 else "Normal" for label in labels]
    return clean


def main() -> None:
    st.title("Wind Turbine Dataset Dashboard")
    st.caption("Comparacao adaptativa de features, qualidade e compatibilidade para datasets open source.")

    available_keys = available_dataset_keys(ROOT)
    if not available_keys:
        st.error("Nenhum dataset registrado foi encontrado na pasta do projeto.")
        st.stop()

    with st.sidebar:
        st.header("Datasets")
        selected_keys = st.multiselect(
            "Selecionar datasets",
            options=available_keys,
            default=available_keys,
            format_func=lambda key: DATASET_SPECS[key].label,
        )
        sample_limit = st.slider(
            "Pontos por grafico",
            min_value=2_000,
            max_value=200_000,
            value=20_000,
            step=2_000,
        )
        if sample_limit > 50_000:
            st.caption(
                "Amostras acima de 50.000 pontos podem deixar a renderizacao no navegador mais pesada."
            )

    if not selected_keys:
        st.warning("Selecione ao menos um dataset.")
        st.stop()

    datasets = {}
    load_errors = []
    for key in selected_keys:
        try:
            datasets[key] = cached_load_dataset(key)
        except Exception as exc:  # pragma: no cover - Streamlit surface
            load_errors.append((key, exc))

    for key, exc in load_errors:
        st.error(f"Falha ao carregar {DATASET_SPECS[key].label}: {exc}")

    if not datasets:
        st.stop()

    raw_df = pd.concat(datasets.values(), ignore_index=True, sort=False)
    raw_df["entity_display"] = raw_df["dataset_label"] + " / " + raw_df["entity"].astype(str)

    profiles, profile_table = build_profiles(datasets, DATASET_SPECS)
    filtered_df, y_col = _apply_global_filters(raw_df, sample_limit)
    filtered_profiles, filtered_profile_table = build_profiles(
        {
            key: filtered_df[filtered_df["dataset"] == key].copy()
            for key in datasets
            if key in filtered_df["dataset"].unique()
        },
        DATASET_SPECS,
    )

    tab_summary, tab_compat, tab_merge, tab_power, tab_temporal, tab_status, tab_quality = st.tabs(
        [
            "Resumo",
            "Features & compatibilidade",
            "Merge CSVs",
            "Curvas de potencia",
            "Temporal",
            "Status & caracteristicas",
            "Qualidade",
        ]
    )

    with tab_summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Datasets ativos", len(datasets))
        c2.metric("Linhas filtradas", f"{len(filtered_df):,}")
        c3.metric("Entidades filtradas", filtered_df["entity_display"].nunique())
        c4.metric("Features comuns", common_feature_count(filtered_profiles))

        st.subheader("Perfil dos datasets")
        visible_cols = [
            "dataset_label",
            "source_type",
            "rows",
            "entities",
            "coverage_start",
            "coverage_end",
            "resolution_minutes",
            "power_unit",
            "power_modes",
            "missing_cells_pct",
            "duplicated_rows_pct",
        ]
        st.dataframe(
            _format_datetime_columns(profile_table[visible_cols]),
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(entity_counts_bar(filtered_df), use_container_width=True)

        with st.expander("Como adicionar um novo dataset"):
            st.markdown(
                """
                1. Abra `wind_dashboard_core/loaders.py`.
                2. Crie uma funcao `load_meu_dataset(root)` que leia os arquivos e renomeie colunas para o contrato canonico.
                3. Registre um `DatasetSpec` em `DATASET_SPECS`.
                4. Adicione a funcao no dicionario `LOADERS`.

                Colunas canonicas principais: `entity`, `timestamp`, `wind_speed_ms`, `power_kw`, `power_norm`,
                `wind_direction_deg`, `temperature_c`, `yaw_error`, `cp`, `ct`, `thrust_kn`.
                """
            )

    with tab_compat:
        cov = coverage_matrix(profiles)
        st.plotly_chart(feature_coverage_heatmap(cov), use_container_width=True)
        st.dataframe(cov, use_container_width=True)

        st.subheader("Descrição das variáveis padronizadas")
        variable_table = _variable_description_table(raw_df.columns)
        st.dataframe(variable_table, use_container_width=True, hide_index=True)

        pairwise = pairwise_compatibility(profiles)
        if pairwise.empty:
            st.info("Selecione ao menos dois datasets para calcular compatibilidade par-a-par.")
        else:
            st.plotly_chart(pairwise_severity_heatmap(pairwise), use_container_width=True)
            st.dataframe(pairwise, use_container_width=True, hide_index=True)

        st.subheader("Visualizar head() por dataset")
        head_rows = st.slider(
            "Linhas do head()",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
            key="compat_head_rows",
        )
        head_cols = st.columns(min(len(datasets), 3))
        for idx, key in enumerate(datasets):
            spec = DATASET_SPECS[key]
            with head_cols[idx % len(head_cols)]:
                show_head = st.checkbox(
                    spec.label,
                    value=False,
                    key=f"compat_show_head_{key}",
                )
            if show_head:
                st.markdown(f"**{spec.label} - `head({head_rows})`**")
                st.dataframe(datasets[key].head(head_rows), use_container_width=True, hide_index=True)

        st.subheader("Heatmaps por dataset")
        st.caption(
            "Marque apenas os datasets que deseja calcular. O heatmap usa correlação de Pearson "
            "entre features numéricas do dataset no recorte filtrado atual."
        )
        heatmap_sample_limit = st.slider(
            "Linhas maximas por heatmap",
            min_value=5_000,
            max_value=200_000,
            value=50_000,
            step=5_000,
            key="compat_heatmap_sample_limit",
        )
        if heatmap_sample_limit > 100_000:
            st.caption("Heatmaps grandes costumam demorar mais em datasets futuros com muitas colunas numericas.")
        heatmap_cols = st.columns(min(len(datasets), 3))
        selected_heatmap_keys = []
        for idx, key in enumerate(datasets):
            spec = DATASET_SPECS[key]
            with heatmap_cols[idx % len(heatmap_cols)]:
                if st.checkbox(
                    f"Gerar heatmap - {spec.label}",
                    value=False,
                    key=f"compat_heatmap_{key}",
                ):
                    selected_heatmap_keys.append(key)

        for key in selected_heatmap_keys:
            spec = DATASET_SPECS[key]
            df_heat = filtered_df[filtered_df["dataset"] == key].copy()
            if df_heat.empty:
                st.info(f"{spec.label}: sem linhas no recorte filtrado atual.")
                continue
            st.plotly_chart(
                dataset_correlation_heatmap(df_heat, spec.label, heatmap_sample_limit),
                use_container_width=True,
            )

    with tab_merge:
        st.subheader("Merge deterministico entre datasets")
        st.caption(
            "O merge usa apenas o schema canonico, chaves detectadas e regras fixas. "
            "As flags abaixo explicam se o par e recomendado, exige cuidado ou deve ser bloqueado."
        )

        if len(datasets) < 2:
            st.info("Selecione ao menos dois datasets na barra lateral para planejar um merge.")
        else:
            merge_scope = st.radio(
                "Escopo do merge",
                options=["Recorte filtrado atual", "Datasets completos"],
                horizontal=True,
                help="O recorte filtrado respeita os filtros globais da barra lateral.",
            )
            if merge_scope == "Recorte filtrado atual":
                merge_frames = {
                    key: filtered_df[filtered_df["dataset"] == key].copy()
                    for key in datasets
                }
            else:
                merge_frames = {
                    key: raw_df[raw_df["dataset"] == key].copy()
                    for key in datasets
                }

            merge_frames = {
                key: frame
                for key, frame in merge_frames.items()
                if not frame.empty
            }

            if len(merge_frames) < 2:
                st.warning("O escopo atual deixou menos de dois datasets com linhas disponiveis.")
            else:
                ranked_plans = []
                for left_key, right_key in combinations(merge_frames.keys(), 2):
                    plan = plan_compatible_merge(
                        left_key,
                        right_key,
                        merge_frames[left_key],
                        merge_frames[right_key],
                        DATASET_SPECS,
                    )
                    ranked_plans.append((left_key, right_key, plan))

                ranked_plans = sorted(ranked_plans, key=lambda item: item[2].score, reverse=True)
                ranking_table = pd.DataFrame(
                    [
                        {
                            "pair": f"{plan.left_label} x {plan.right_label}",
                            "merge_flag": plan.merge_flag,
                            "compatibility_flag": plan.compatibility_flag,
                            "key_flag": plan.key_flag,
                            "score": plan.score,
                            "strategy": plan.strategy_label,
                            "join_keys": ", ".join(plan.join_keys) if plan.join_keys else "n/a",
                            "warnings": " | ".join(plan.warnings),
                        }
                        for _, _, plan in ranked_plans
                    ]
                )
                st.dataframe(ranking_table, use_container_width=True, hide_index=True)

                pair_options = [(left_key, right_key) for left_key, right_key, _ in ranked_plans]
                selected_pair = st.selectbox(
                    "Par para gerar merge",
                    options=pair_options,
                    index=0,
                    format_func=lambda pair: (
                        f"{DATASET_SPECS[pair[0]].label} x {DATASET_SPECS[pair[1]].label}"
                    ),
                )
                selected_plan = next(
                    plan
                    for left_key, right_key, plan in ranked_plans
                    if (left_key, right_key) == selected_pair
                )

                m_flag, m_key, m_score, m_strategy = st.columns(4)
                m_flag.metric("Flag de merge", selected_plan.merge_flag)
                m_key.metric("Flag da chave", selected_plan.key_flag)
                m_score.metric("Score", selected_plan.score)
                m_strategy.metric("Compatibilidade", selected_plan.compatibility_flag)

                st.dataframe(pd.DataFrame([selected_plan.to_dict()]), use_container_width=True, hide_index=True)

                if selected_plan.warnings:
                    st.warning("\n".join(f"- {warning}" for warning in selected_plan.warnings))
                if selected_plan.merge_flag == MERGE_BLOCKED:
                    st.error("Este par nao tem uma chave deterministica suficiente para gerar merge.")
                else:
                    c_join, c_preview = st.columns([1, 1])
                    with c_join:
                        join_how = st.selectbox(
                            "Tipo de join",
                            options=["inner", "left", "outer"],
                            index=0,
                            format_func={
                                "inner": "inner - somente chaves nos dois datasets",
                                "left": "left - preserva o primeiro dataset",
                                "outer": "outer - preserva todas as chaves",
                            }.get,
                        )
                    with c_preview:
                        preview_rows = st.slider(
                            "Linhas da previa",
                            min_value=10,
                            max_value=500,
                            value=50,
                            step=10,
                        )

                    if st.button("Gerar merge", type="primary"):
                        merged_df = execute_merge_plan(
                            selected_plan,
                            merge_frames[selected_pair[0]],
                            merge_frames[selected_pair[1]],
                            join_how=join_how,
                        )
                        r1, r2, r3 = st.columns(3)
                        r1.metric("Linhas geradas", f"{len(merged_df):,}")
                        r2.metric("Colunas geradas", len(merged_df.columns))
                        if "_merge_presence" in merged_df.columns:
                            both_rows = int((merged_df["_merge_presence"] == "both").sum())
                            r3.metric("Chaves nos dois", f"{both_rows:,}")
                        else:
                            r3.metric("Chaves nos dois", "n/a")

                        if "_merge_presence" in merged_df.columns:
                            presence_summary = (
                                merged_df["_merge_presence"]
                                .value_counts()
                                .rename_axis("merge_presence")
                                .reset_index(name="rows")
                            )
                            st.dataframe(presence_summary, use_container_width=True, hide_index=True)

                        st.dataframe(merged_df.head(preview_rows), use_container_width=True, hide_index=True)
                        csv_bytes = merged_df.to_csv(index=False).encode("utf-8")
                        file_name = (
                            f"merge_{selected_plan.left_key}_{selected_plan.right_key}_"
                            f"{selected_plan.strategy_id}.csv"
                        )
                        st.download_button(
                            "Baixar CSV gerado",
                            data=csv_bytes,
                            file_name=file_name,
                            mime="text/csv",
                        )

    with tab_power:
        st.subheader("Comparação por curva vento-potência")
        if y_col not in filtered_df.columns or filtered_df[y_col].notna().sum() == 0:
            st.warning("Não há coluna de potência disponível para os filtros atuais.")
        else:
            st.plotly_chart(power_scatter(filtered_df, y_col, sample_limit), use_container_width=True)
            st.plotly_chart(empirical_power_curve(filtered_df, y_col), use_container_width=True)

            counts = (
                filtered_df.dropna(subset=["wind_speed_ms", y_col])
                .groupby(["dataset_label", "entity"], observed=True)
                .size()
                .rename("usable_points")
                .reset_index()
            )
            st.dataframe(counts, use_container_width=True, hide_index=True)

    with tab_temporal:
        if "timestamp" in filtered_df.columns:
            timestamped = filtered_df[filtered_df["timestamp"].notna()]
        else:
            timestamped = filtered_df.iloc[0:0].copy()
        if timestamped.empty:
            st.info("Os datasets filtrados não possuem timestamp disponível.")
        else:
            st.plotly_chart(temporal_monthly_chart(timestamped, y_col), use_container_width=True)
            timeline_table = (
                timestamped.groupby(["dataset_label", "entity"], observed=True)["timestamp"]
                .agg(["min", "max", "count"])
                .reset_index()
                .rename(columns={"min": "inicio", "max": "fim", "count": "registros"})
            )
            st.dataframe(timeline_table, use_container_width=True, hide_index=True)

    with tab_status:
        st.subheader("Caracteristicas das turbinas e datasets")
        characteristic_cols = [
            "site",
            "year",
            "turbine_model",
            "rated_power_kw",
            "rotor_diameter_m",
            "annual_production_kwh",
        ]
        available_characteristics = [col for col in characteristic_cols if col in filtered_df.columns]
        if available_characteristics:
            characteristics = (
                filtered_df.groupby(["dataset_label", "entity"], observed=True)[available_characteristics]
                .agg(_first_valid)
                .reset_index()
            )
            st.dataframe(characteristics, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhuma caracteristica estrutural foi encontrada nos datasets filtrados.")

        availability_cols = [
            "data_availability",
            "time_based_availability",
            "production_based_availability",
            "capacity_factor",
        ]
        availability_cols = [
            col for col in availability_cols if col in filtered_df.columns and filtered_df[col].notna().any()
        ]
        if availability_cols:
            st.subheader("Indicadores operacionais medios")
            availability_summary = (
                filtered_df.groupby(["dataset_label", "entity"], observed=True)[availability_cols]
                .mean(numeric_only=True)
                .round(3)
                .reset_index()
            )
            st.dataframe(availability_summary, use_container_width=True, hide_index=True)

            availability_melted = availability_summary.melt(
                id_vars=["dataset_label", "entity"],
                value_vars=availability_cols,
                var_name="metric",
                value_name="value",
            ).dropna(subset=["value"])
            if not availability_melted.empty:
                fig = px.bar(
                    availability_melted,
                    x="entity",
                    y="value",
                    color="metric",
                    barmode="group",
                    facet_col="dataset_label" if availability_melted["dataset_label"].nunique() > 1 else None,
                    labels={"entity": "Entidade", "value": "Media", "metric": "Indicador"},
                )
                fig.update_layout(template="plotly_white", height=420)
                st.plotly_chart(fig, use_container_width=True)

        status_group_options = [
            col
            for col in ["iec_category", "service_contract_category", "status", "status_message"]
            if col in filtered_df.columns and filtered_df[col].notna().any()
        ]
        if status_group_options:
            st.subheader("Resumo de status operacional")
            status_group_col = st.selectbox(
                "Agrupar status por",
                options=status_group_options,
                format_func=lambda col: {
                    "iec_category": "IEC category",
                    "service_contract_category": "Service contract category",
                    "status": "Status",
                    "status_message": "Mensagem",
                }.get(col, col),
            )
            status_df = filtered_df.dropna(subset=[status_group_col]).copy()
            status_df[status_group_col] = status_df[status_group_col].astype(str)
            status_summary = (
                status_df.groupby(["dataset_label", "entity", status_group_col], observed=True)
                .size()
                .rename("rows")
                .reset_index()
                .sort_values("rows", ascending=False)
            )
            st.dataframe(status_summary, use_container_width=True, hide_index=True)

            top_categories = (
                status_summary.groupby(status_group_col, observed=True)["rows"]
                .sum()
                .nlargest(12)
                .index
            )
            plot_status = status_summary[status_summary[status_group_col].isin(top_categories)].copy()
            if not plot_status.empty:
                fig = px.bar(
                    plot_status,
                    x="entity",
                    y="rows",
                    color=status_group_col,
                    facet_col="dataset_label" if plot_status["dataset_label"].nunique() > 1 else None,
                    labels={
                        "entity": "Entidade",
                        "rows": "Registros anotados",
                        status_group_col: "Status",
                    },
                )
                fig.update_layout(template="plotly_white", height=520)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhum status operacional foi encontrado nos datasets filtrados.")

    with tab_quality:
        st.plotly_chart(profile_quality_bar(profile_table), use_container_width=True)
        st.dataframe(
            _format_datetime_columns(filtered_profile_table),
            use_container_width=True,
            hide_index=True,
        )

        if "outlier_flag" in filtered_df.columns and filtered_df["outlier_flag"].notna().any():
            st.subheader("Outlier flag existente")
            outlier_summary = (
                filtered_df.assign(outlier_flag=filtered_df["outlier_flag"].astype(str))
                .groupby(["dataset_label", "outlier_flag"], observed=True)
                .size()
                .rename("rows")
                .reset_index()
            )
            fig = px.bar(
                outlier_summary,
                x="dataset_label",
                y="rows",
                color="outlier_flag",
                barmode="group",
                labels={"dataset_label": "Dataset", "rows": "Registros", "outlier_flag": "Outlier"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("DBSCAN adaptativo em amostra filtrada", expanded=False):
            st.caption(
                "Executa DBSCAN em uma amostra do dataset escolhido. "
                "A curva k-distance, o palpite de eps e a classificacao nao misturam datasets diferentes."
            )

            dbscan_dataset_options = []
            for label, group in filtered_df.groupby("dataset_label", observed=True):
                has_wind = "wind_speed_ms" in group.columns and group["wind_speed_ms"].notna().any()
                has_power = any(
                    col in group.columns and group[col].notna().any()
                    for col in ["power_norm", "power_kw"]
                )
                if has_wind and has_power:
                    dbscan_dataset_options.append(label)

            if not dbscan_dataset_options:
                st.warning("Nenhum dataset filtrado tem vento e potencia suficientes para DBSCAN.")
                return

            selected_dbscan_dataset = st.selectbox(
                "Dataset para analisar com DBSCAN",
                options=sorted(dbscan_dataset_options),
                help="O DBSCAN e sensivel a densidade. Por isso, ele e ajustado separadamente para cada dataset.",
            )
            dbscan_scope = filtered_df[filtered_df["dataset_label"] == selected_dbscan_dataset].copy()

            power_options = [
                col
                for col in ["power_norm", "power_kw"]
                if col in dbscan_scope.columns and dbscan_scope[col].notna().any()
            ]
            default_power_index = power_options.index(y_col) if y_col in power_options else 0

            c_power, c_sample, c_min_samples = st.columns([1, 1, 1])
            with c_power:
                dbscan_y_col = st.selectbox(
                    "Escala de potencia do DBSCAN",
                    options=power_options,
                    index=default_power_index,
                    format_func=lambda col: "Normalizada" if col == "power_norm" else "kW",
                )
            with c_sample:
                dbscan_limit = st.slider(
                    "Amostra DBSCAN",
                    min_value=1_000,
                    max_value=20_000,
                    value=5_000,
                    step=1_000,
                    help="Quantidade maxima de pontos usados no DBSCAN. Amostras maiores sao mais representativas, mas mais lentas.",
                )
            with c_min_samples:
                min_samples = st.slider(
                    "min_samples",
                    min_value=3,
                    max_value=100,
                    value=10,
                    step=1,
                    help="Numero minimo de vizinhos para formar uma regiao densa.",
                )

            dbscan_sample = _dbscan_feature_sample(dbscan_scope, dbscan_y_col, dbscan_limit)
            suggested_eps, kdist_df = _suggest_dbscan_eps(dbscan_sample, dbscan_y_col, min_samples)

            c_eps, c_hint = st.columns([1, 2])
            with c_eps:
                eps = st.slider(
                    "eps",
                    min_value=0.05,
                    max_value=2.0,
                    value=suggested_eps,
                    step=0.01,
                    help="Raio de vizinhanca no espaco padronizado. Baixo demais marca muitos pontos como anomalia; alto demais suaviza tudo.",
                )
            with c_hint:
                st.metric("Palpite adaptativo de eps", f"{suggested_eps:.3f}")
                st.caption(
                    f"O palpite vem do cotovelo da curva k-distance para {selected_dbscan_dataset}. "
                    "Use como ponto de partida, nao como valor absoluto."
                )

            if not kdist_df.empty:
                show_kdistance = st.checkbox("Mostrar curva k-distance", value=True)
                if show_kdistance:
                    fig_kdist = px.line(
                        kdist_df,
                        x="ordered_point",
                        y="k_distance",
                        labels={
                            "ordered_point": "Pontos ordenados",
                            "k_distance": f"Distancia ao {min_samples}o vizinho",
                        },
                    )
                    fig_kdist.add_hline(
                        y=eps,
                        line_dash="dash",
                        line_color="#d62728",
                        annotation_text=f"eps={eps:.2f}",
                    )
                    fig_kdist.update_layout(
                        title=f"Curva k-distance para escolha de eps - {selected_dbscan_dataset}",
                        template="plotly_white",
                        height=320,
                        margin={"l": 20, "r": 20, "t": 60, "b": 20},
                    )
                    st.plotly_chart(fig_kdist, use_container_width=True)
            else:
                st.info("Amostra insuficiente para estimar eps por k-distance.")

            if st.button("Executar DBSCAN", type="primary"):
                dbscan_df = _run_dbscan_sample(dbscan_scope, dbscan_y_col, dbscan_limit, eps, min_samples)
                if "dbscan_status" in dbscan_df.columns:
                    n_points = len(dbscan_df)
                    n_anomaly = int((dbscan_df["dbscan_status"] == "Anomalia").sum())
                    anomaly_pct = 100 * n_anomaly / n_points if n_points else 0.0
                    n_clusters = (
                        dbscan_df.loc[dbscan_df["dbscan_label"] >= 0, "dbscan_label"].nunique()
                        if "dbscan_label" in dbscan_df.columns
                        else 0
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Pontos analisados", f"{n_points:,}")
                    m2.metric("Clusters", int(n_clusters))
                    m3.metric("Anomalias", f"{n_anomaly:,}")
                    m4.metric("% anomalias", f"{anomaly_pct:.2f}%")

                    status_summary = (
                        dbscan_df.groupby(["entity", "dbscan_status"], observed=True)
                        .size()
                        .rename("rows")
                        .reset_index()
                    )
                    st.dataframe(status_summary, use_container_width=True, hide_index=True)

                    fig = px.scatter(
                        dbscan_df,
                        x="wind_speed_ms",
                        y=dbscan_y_col,
                        color="dbscan_status",
                        symbol="entity" if dbscan_df["entity"].nunique() <= 12 else None,
                        facet_col_wrap=2,
                        opacity=0.55,
                        color_discrete_map={
                            "Normal": "#1f77b4",
                            "Anomalia": "#d62728",
                            "Amostra insuficiente": "#7f7f7f",
                        },
                        category_orders={"dbscan_status": ["Normal", "Anomalia", "Amostra insuficiente"]},
                        hover_data=[
                            col
                            for col in ["entity", "timestamp", "source_type", "dbscan_label"]
                            if col in dbscan_df.columns
                        ],
                        labels={
                            "wind_speed_ms": "Velocidade do vento (m/s)",
                            dbscan_y_col: "Potencia normalizada" if dbscan_y_col == "power_norm" else "Potencia (kW)",
                            "dbscan_status": "DBSCAN",
                            "entity": "Entidade",
                        },
                    )
                    fig.update_traces(marker={"size": 5})
                    fig.update_layout(
                        title=f"DBSCAN - {selected_dbscan_dataset}",
                        template="plotly_white",
                        height=560,
                    )
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
