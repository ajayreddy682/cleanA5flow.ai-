"""Interactive Plotly dashboards built from the cleaned data.

Only charts that the data can support are suggested automatically. A manual
builder lets the user choose dimensions, measures, aggregation, and filters
without generating meaningless visuals for identifier or text-only columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from modules.data_profiler import (
    ChartSpec,
    ColumnKind,
    ColumnProfile,
    DataProfile,
    format_number,
)

PALETTE = ["#1D4ED8", "#0891B2", "#38BDF8", "#7C3AED", "#F59E0B", "#10B981", "#EF4444", "#64748B"]
CHART_HEIGHT = 380
TEXT_COLOR = "#334155"
GRID_COLOR = "#E6EDF7"
NAVY = "#0B1F3A"

TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Segoe UI, Inter, system-ui, sans-serif", color=TEXT_COLOR, size=12),
        colorway=PALETTE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=58, b=12),
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=12, font_color=NAVY, bordercolor="#D8E0EC"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font_size=11),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, linecolor="#D8E0EC", title_font_size=12),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, linecolor="#D8E0EC", title_font_size=12),
        height=CHART_HEIGHT,
    )
)
pio.templates["cleanflow"] = TEMPLATE
pio.templates.default = "plotly_white+cleanflow"


def apply_theme(fig: go.Figure, *, title: str | None = None, height: int | None = None) -> go.Figure:
    fig.update_layout(template="plotly_white+cleanflow")
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=15, color=NAVY, family="Segoe UI, Inter, system-ui, sans-serif"),
                x=0,
                xanchor="left",
                pad=dict(l=2, t=2),
            )
        )
    if height:
        fig.update_layout(height=height)
    fig.update_layout(margin=dict(l=8, r=8, t=56, b=8))
    return fig


def empty_figure(message: str, *, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=13, color="#64748B"),
        align="center",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_theme(fig, title=title)


# ---------------------------------------------------------------------------
# Automatic charts
# ---------------------------------------------------------------------------
def build_auto_charts(
    df: pd.DataFrame,
    profile: DataProfile,
    specs: list[ChartSpec] | None = None,
    *,
    max_charts: int = 6,
) -> list[tuple[ChartSpec, go.Figure]]:
    """Build figures for the recommended chart specifications."""

    from modules.data_profiler import recommend_charts

    specs = specs if specs is not None else recommend_charts(df, profile, max_charts=max_charts)
    charts: list[tuple[ChartSpec, go.Figure]] = []
    for spec in specs:
        if spec.kind == "heatmap":
            columns = spec.extra.get("columns") or [
                item.name for item in profile.numeric_profiles[:8]
            ]
            fig = correlation_heatmap(df, columns, title=spec.title)
        else:
            fig = build_chart(
                df,
                kind=spec.kind,
                x=spec.x,
                y=spec.y,
                color=spec.color,
                agg=spec.agg,
                limit=spec.limit,
                period=spec.period,
                title=spec.title,
            )
        charts.append((spec, fig))
    return charts


# ---------------------------------------------------------------------------
# Manual chart builder
# ---------------------------------------------------------------------------
CHART_TYPES = {
    "Bar": "bar",
    "Line": "line",
    "Donut": "donut",
    "Pie": "pie",
    "Histogram": "histogram",
    "Scatter": "scatter",
    "Box": "box",
    "Correlation heatmap": "heatmap",
}

AGGREGATIONS = {
    "Sum": "sum",
    "Average": "mean",
    "Count": "count",
    "Minimum": "min",
    "Maximum": "max",
    "Median": "median",
}


def build_chart(
    df: pd.DataFrame,
    *,
    kind: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    agg: str | None = "sum",
    limit: int = 10,
    period: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build one chart, returning a friendly empty state instead of raising."""

    if df.empty:
        return empty_figure("This filtered view has no rows to chart.", title=title)
    try:
        if kind == "heatmap":
            columns = [column for column in (x, y, color) if column] or list(
                df.select_dtypes(include="number").columns
            )
            return correlation_heatmap(df, columns[:8], title=title)
        if kind in {"bar", "donut", "pie", "line"}:
            if not x:
                return empty_figure("Choose a category or date column for this chart.", title=title)
            if x not in df.columns:
                return empty_figure("The selected column is no longer available.", title=title)
            aggregation = agg or "count"
            if kind in {"bar", "line"} and aggregation not in {"count"}:
                if not y or y not in df.columns:
                    return empty_figure("Choose a numeric measure for this chart.", title=title)
            if kind in {"bar", "line"}:
                data = _aggregate(df, x=x, y=y, color=color, agg=aggregation, limit=limit, period=period)
                if data.empty:
                    return empty_figure("There is not enough valid data for this chart.", title=title)
                return _bar_or_line(data, kind=kind, x=x, y=y, color=color, agg=aggregation, title=title)
            if kind in {"donut", "pie"}:
                if aggregation == "count" or not y:
                    data = _category_counts(df, x=x, limit=limit)
                    value_column = "count"
                elif aggregation in {"sum"}:
                    data = _aggregate(df, x=x, y=y, color=None, agg=aggregation, limit=limit, period=None)
                    value_column = "value"
                else:
                    data = _aggregate(df, x=x, y=y, color=None, agg=aggregation, limit=limit, period=None)
                    value_column = "value"
                if data.empty:
                    return empty_figure("There is not enough valid data for this chart.", title=title)
                fig = px.pie(
                    data,
                    names="category",
                    values=value_column,
                    hole=0.55 if kind == "donut" else 0.0,
                    color_discrete_sequence=PALETTE,
                )
                fig.update_traces(
                    textposition="inside",
                    textinfo="percent",
                    hovertemplate="<b>%{label}</b><br>%{value:,.2f}<br>%{percent}<extra></extra>",
                    marker=dict(line=dict(color="#FFFFFF", width=2)),
                )
                fig.update_layout(legend=dict(orientation="v", y=0.5, yanchor="middle", x=1.02, xanchor="left"))
                return apply_theme(fig, title=title or f"Record share by {_humanize(x)}")
        if kind == "histogram":
            if not x or x not in df.columns:
                return empty_figure("Choose a numeric column for the histogram.", title=title)
            numeric = pd.to_numeric(df[x], errors="coerce").dropna()
            if numeric.empty:
                return empty_figure("The selected column has no numeric values.", title=title)
            plot_frame = pd.DataFrame({x: numeric})
            hue = None
            if color and color in df.columns:
                candidates = df.loc[numeric.index, color]
                if candidates.nunique(dropna=True) <= 8:
                    plot_frame[color] = candidates.values
                    hue = color
            fig = px.histogram(
                plot_frame,
                x=x,
                color=hue,
                nbins=min(max(int(limit), 8), 60),
                color_discrete_sequence=PALETTE,
            )
            fig.update_traces(marker_line_color="#FFFFFF", marker_line_width=1)
            fig.update_layout(barmode="overlay", bargap=0.06)
            return apply_theme(fig, title=title or f"Distribution of {_humanize(x)}")
        if kind == "scatter":
            if not x or not y or x not in df.columns or y not in df.columns:
                return empty_figure("Choose two numeric columns for the scatter plot.", title=title)
            plot = df[[x, y] + ([color] if color and color in df.columns else [])].copy()
            plot[x] = pd.to_numeric(plot[x], errors="coerce")
            plot[y] = pd.to_numeric(plot[y], errors="coerce")
            plot = plot.dropna(subset=[x, y])
            if plot.empty:
                return empty_figure("There are no complete numeric pairs to plot.", title=title)
            if len(plot) > limit:
                plot = plot.sample(n=limit, random_state=42)
            hue = color if color and color in plot.columns and plot[color].nunique(dropna=True) <= 8 else None
            fig = px.scatter(
                plot,
                x=x,
                y=y,
                color=hue,
                opacity=0.72,
                color_discrete_sequence=PALETTE,
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#FFFFFF")))
            return apply_theme(fig, title=title or f"{_humanize(y)} vs {_humanize(x)}")
        if kind == "box":
            if not x or not y:
                return empty_figure("Choose a category and a numeric measure.", title=title)
            plot = df[[x, y] + ([color] if color and color in df.columns else [])].copy()
            plot[y] = pd.to_numeric(plot[y], errors="coerce")
            plot = plot.dropna(subset=[y])
            if plot.empty:
                return empty_figure("No numeric values are available for this box plot.", title=title)
            fig = px.box(plot, x=x, y=y, color=color if color in plot.columns else None, color_discrete_sequence=PALETTE)
            return apply_theme(fig, title=title or f"{_humanize(y)} by {_humanize(x)}")
    except Exception:
        return empty_figure(
            "This chart could not be drawn with the current selections. Try a different column or aggregation.",
            title=title,
        )
    return empty_figure("Choose a chart type to begin.", title=title)


def _aggregate(
    df: pd.DataFrame,
    *,
    x: str,
    y: str | None,
    color: str | None,
    agg: str,
    limit: int,
    period: str | None,
) -> pd.DataFrame:
    data = df.copy()
    is_date = pd.api.types.is_datetime64_any_dtype(data[x])
    if is_date:
        data = data.dropna(subset=[x])
        data = _add_period(data, x, period)
        x_plot = "_period_label"
        x_sort = "_period_start"
    else:
        data = data.dropna(subset=[x])
        x_plot = x
        x_sort = None

    if color and color in data.columns and color != x:
        group_columns = [x_plot, color]
    else:
        group_columns = [x_plot]
        color = None

    if agg == "count":
        grouped = data.groupby(group_columns, dropna=False).size().reset_index(name="value")
    else:
        if not y or y not in data.columns:
            return pd.DataFrame()
        data[y] = pd.to_numeric(data[y], errors="coerce")
        data = data.dropna(subset=[y])
        if data.empty:
            return pd.DataFrame()
        if agg == "median":
            grouped = data.groupby(group_columns, dropna=False)[y].median().reset_index(name="value")
        else:
            grouped = data.groupby(group_columns, dropna=False)[y].agg(agg).reset_index(name="value")

    if x_sort:
        sort_lookup = data[[x_plot, x_sort]].drop_duplicates().sort_values(x_sort)
        grouped = sort_lookup.merge(grouped, on=x_plot, how="left").fillna({"value": 0})
        grouped["_sort_value"] = grouped[x_sort]
    else:
        grouped["_sort_value"] = pd.to_numeric(grouped["value"], errors="coerce").fillna(0)

    if not color:
        if x_sort:
            grouped = grouped.sort_values("_sort_value")
        else:
            # For categories, rank by aggregate value and keep the top N.
            grouped = grouped.sort_values("_sort_value", ascending=False)
            if len(grouped) > limit:
                grouped = grouped.head(limit)
            grouped = grouped.sort_values("_sort_value")
    else:
        totals = grouped.groupby(x_plot)["value"].sum().sort_values(ascending=False)
        keep = totals.head(limit).index
        grouped = grouped[grouped[x_plot].isin(keep)]

    grouped[x_plot] = grouped[x_plot].astype(str)
    grouped = grouped.rename(columns={x_plot: "category"})
    return grouped


def _add_period(data: pd.DataFrame, column: str, period: str | None) -> pd.DataFrame:
    series = pd.to_datetime(data[column], errors="coerce")
    data = data.loc[series.notna()].copy()
    series = series.loc[series.notna()]
    span_days = (series.max() - series.min()).days if len(series) else 0
    granularity = period if period in {"D", "M", "Y"} else ("D" if span_days <= 45 else "M" if span_days <= 730 else "Y")

    if granularity == "D":
        starts = series.dt.floor("D")
        labels = starts.dt.strftime("%d %b %Y")
    elif granularity == "Y":
        starts = series.dt.to_period("Y").dt.to_timestamp()
        labels = starts.dt.strftime("%Y")
    else:
        starts = series.dt.to_period("M").dt.to_timestamp()
        labels = starts.dt.strftime("%b %Y")
    data["_period_start"] = starts
    data["_period_label"] = labels
    data["_period_granularity"] = granularity
    return data


def _bar_or_line(
    data: pd.DataFrame,
    *,
    kind: str,
    x: str,
    y: str | None,
    color: str | None,
    agg: str,
    title: str | None,
) -> go.Figure:
    value_label = "Records" if agg == "count" or not y else f"{agg.title()} of {_humanize(y)}"
    if kind == "line":
        fig = px.line(
            data.sort_values("_sort_value"),
            x="category",
            y="value",
            color=color,
            markers=True,
            color_discrete_sequence=PALETTE,
            labels={"category": _humanize(x), "value": value_label},
        )
        fig.update_traces(line=dict(width=2.6), marker=dict(size=6))
    else:
        fig = px.bar(
            data,
            x="value",
            y="category",
            color=color,
            orientation="h",
            color_discrete_sequence=PALETTE,
            labels={"category": _humanize(x), "value": value_label},
            text="value",
        )
        fig.update_traces(
            texttemplate="%{text:,.2s}",
            textposition="outside",
            cliponaxis=False,
            marker_line_width=0,
        )
        fig.update_layout(bargap=0.3)

    fig.update_xaxes(showgrid=kind == "line", title_text=value_label if kind == "bar" else _humanize(x))
    fig.update_yaxes(showgrid=kind == "bar", title_text=_humanize(x) if kind == "bar" else value_label)
    if kind == "bar":
        fig.update_layout(margin=dict(l=8, r=58, t=56, b=8))
    if data["category"].nunique() > 20:
        fig.update_xaxes(tickangle=-35)
    return apply_theme(fig, title=title)


def _category_counts(df: pd.DataFrame, *, x: str, limit: int) -> pd.DataFrame:
    counts = df[x].dropna().astype(str).value_counts()
    if len(counts) > limit:
        top = counts.head(limit - 1)
        other = int(counts.iloc[limit - 1 :].sum())
        frame = pd.DataFrame({"category": top.index.tolist(), "count": top.values.tolist()})
        if other:
            frame.loc[len(frame)] = [f"Other ({len(counts) - (limit - 1)} categories)", other]
        return frame
    return pd.DataFrame({"category": counts.index.tolist(), "count": counts.values.tolist()})


def correlation_heatmap(df: pd.DataFrame, columns: Iterable[str], *, title: str | None = None) -> go.Figure:
    numeric_columns = []
    for column in columns:
        if column not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)
    numeric_columns = numeric_columns[:8]
    if len(numeric_columns) < 2:
        return empty_figure("At least two numeric columns are needed for a correlation heatmap.", title=title)
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] < 2:
        return empty_figure("The numeric columns are constant, so correlation is undefined.", title=title)
    correlation = numeric.corr(numeric_only=True)
    labels = [_humanize(column) for column in correlation.columns]
    text = correlation.round(2).astype(str).values
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, "#0B1F3A"],
                [0.25, "#1D4ED8"],
                [0.5, "#F8FAFC"],
                [0.75, "#0891B2"],
                [1.0, "#38BDF8"],
            ],
            text=text,
            texttemplate="%{text}",
            hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>",
            colorbar=dict(title="r", thickness=12, len=0.75, outlinewidth=0),
        )
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(autorange="reversed")
    return apply_theme(fig, title=title or "Correlation between numeric fields")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
@dataclass
class FilterState:
    categories: dict[str, list[Any]]
    date_column: str | None = None
    start: date | None = None
    end: date | None = None

    @property
    def is_active(self) -> bool:
        return bool(self.categories) or (self.date_column is not None and (self.start or self.end))


def filterable_columns(profile: DataProfile) -> list[ColumnProfile]:
    return [
        item for item in profile.column_profiles
        if item.role == "dimension" and 2 <= item.unique <= 50 and item.kind != ColumnKind.IDENTIFIER
    ]


def apply_filters(df: pd.DataFrame, state: FilterState | None) -> pd.DataFrame:
    if state is None or not state.is_active:
        return df
    filtered = df
    for column, values in state.categories.items():
        if column in filtered.columns and values:
            filtered = filtered[filtered[column].isin(values)]
    if state.date_column and state.date_column in filtered.columns:
        raw_dates = filtered[state.date_column]
        if pd.api.types.is_datetime64_any_dtype(raw_dates):
            date_series = raw_dates
        else:
            # format="mixed" keeps rows with differing time formats instead of
            # silently coercing them to NaT based on the first row's format.
            date_series = pd.to_datetime(raw_dates, errors="coerce", format="mixed")
        mask = date_series.notna()
        if state.start is not None:
            mask &= date_series >= pd.Timestamp(state.start)
        if state.end is not None:
            # Exclusive upper bound keeps the whole end date and avoids the
            # NumPy generic-timedelta deprecation in pandas 2.3+.
            mask &= date_series < pd.Timestamp(state.end) + timedelta(days=1)
        filtered = filtered[mask]
    return filtered


def filtered_summary(df: pd.DataFrame) -> dict[str, Any]:
    rows = int(df.shape[0])
    columns = int(df.shape[1])
    missing = int(df.isna().sum().sum()) if columns else 0
    cells = rows * columns
    return {
        "rows": rows,
        "columns": columns,
        "missing": missing,
        "missing_pct": (missing / cells * 100.0) if cells else 0.0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _humanize(name: str | None) -> str:
    if not name:
        return ""
    text = str(name).replace("_", " ").strip()
    return " ".join(word.capitalize() if word.islower() else word for word in text.split())


def format_axis_number(value: Any) -> str:
    return format_number(value)
