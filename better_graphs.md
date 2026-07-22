# NREL Dashboard and Visualization Standard

## 1. Objective

This document defines a scalable standard for wind turbine analytics based on the NREL reference workflow.
It standardizes:

- Data model
- Dashboard structure
- Graph style and semantics
- Validation and QA checklist

The same structure can be reused for NREL, EDP, Kaggle, and future projects.

## 2. Standard Data Contract

Minimum columns for a power-curve analytics module:

- wind_speed_ms
- power_kw
- turbine

Recommended columns for deeper analysis:

- power_norm
- cp
- ct
- thrust_kn
- P_nom_kw
- D
- timestamp (for temporal analysis)
- wind_direction_deg (for polar/cylindrical views)

Rule:
All loaders must return a single canonical dataframe using the same column names.

## 3. Scalable Project Layout

Suggested structure:

- nrel_dashboard.py
- better_graphs.md
- data/
- notebooks/
- src/
  - loaders.py
  - metrics.py
  - charts.py
  - theme.py

Operational rule:

- Keep chart logic outside notebooks for reuse.
- Notebooks are for exploration and validation.
- Dashboard is for decision workflow.

## 4. Dashboard Standard (NREL Pattern)

Mandatory tabs:

1. Summary
2. Power Curves
3. 3D Scatter
4. AEP Heatmap

Global filters:

- turbine
- Weibull scenario (k, lambda)
- wind speed range

Mandatory KPIs:

- Cp_max
- V_cut-in
- V_rated
- V_cut-out
- AEP
- Capacity factor
- Number of points

## 5. Graph Standardization

### 5.1 Shared style

- Template: plotly_white
- Font: Arial, size 12
- Marker size (scatter): 4 to 6
- Line width: 2
- Legend title: Turbina
- Always display units in axis labels

### 5.2 Color and identity

- Each turbine has fixed color in all charts.
- 3D scatter marker shape must remain the same for all turbines (circle).
- Do not change color mapping across tabs.

### 5.3 Required chart set

1. 2D power curve by turbine
2. Normalized power curve
3. 3D scatter (wind speed, turbine, power)
4. AEP heatmap (k x lambda)
5. Optional sensitivity chart d(P/P_nom)/dV

### 5.4 Interaction standard

- Hover must include at least: turbine, wind_speed_ms, power_kw, power_norm.
- Zoom and pan enabled in all Plotly charts.
- Data point counts displayed for transparency.

## 6. Why 3D may look like fewer points

In reference datasets, each turbine has a finite discrete set of operating points.
Also, 3D perspective creates overlap.
Standard mitigation:

- Show points per turbine table near the scatter.
- Use opacity around 0.8.
- Keep marker shape fixed and small.

## 7. AEP and Weibull Standard

- Use linear interpolation on power curve.
- Integrate with numpy.trapezoid.
- Default scenarios:
  - Offshore: k=2, lambda=8
  - Onshore: k=2, lambda=6
- Heatmap default turbine: IEA 15MW

## 8. QA Checklist

Before publishing notebook/app updates:

1. All required columns exist.
2. No mixed units in axis labels.
3. Color mapping is consistent.
4. Marker shape in 3D is the same for all turbines.
5. Point counts match input rows per turbine.
6. AEP output is physically plausible.
7. Notebook or app has no lint/runtime errors.

## 9. Scaling to New Datasets

To scale this standard:

1. Add a new loader that maps raw columns to canonical names.
2. Register turbine metadata (P_nom_kw, D, file).
3. Reuse the same chart functions and color map.
4. Keep the same tab structure and KPI cards.

This guarantees comparability between projects.

## 10. Run Instructions (Dashboard)

From workspace root:

- Install dependencies: streamlit, plotly, pandas, numpy, scipy
- Run: streamlit run nrel_dashboard.py

Recommended evolution:

- Move theme, loaders, metrics and charts into src/ modules.
- Keep nrel_dashboard.py as app entry point only.
