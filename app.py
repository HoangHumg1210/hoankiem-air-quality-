from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inference import forecast_recursive, load_model_bundle, prepare_raw_frame

st.set_page_config(
    page_title="PM2.5 Forecast Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_PATH = Path(__file__).resolve().parent / "data" / "processed" / "data2225_done.csv"
REGISTRY_PATH = Path(__file__).resolve().parent / "model_registry"

MODEL_COLORS = {
    "CNN-GRU + Attention": "#17b26a",
    "Seq2Seq GRU + Attention": "#5b8def",
    "Seq2Seq LSTM + Attention": "#8b5cf6",
    "Seq2Seq TCN": "#60a5fa",
    "Encoder-only Transformer": "#f59e0b",
}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Local Time"] = pd.to_datetime(df["Local Time"])
    return df.sort_values("Local Time").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_model_frame() -> pd.DataFrame:
    model_df = prepare_raw_frame(load_data(), step_hours=3).reset_index()
    model_df["Local Time"] = pd.to_datetime(model_df["Local Time"])
    return model_df.sort_values("Local Time").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_registry_metrics() -> pd.DataFrame:
    rows = []
    for bundle_dir in sorted(REGISTRY_PATH.iterdir()):
        if not bundle_dir.is_dir() or bundle_dir.name.startswith("."):
            continue

        metrics_path = bundle_dir / "metrics.json"
        config_path = bundle_dir / "config.json"
        timeline_path = bundle_dir / "test_timeline.csv"
        if not metrics_path.exists() or not config_path.exists():
            continue

        metrics = pd.read_json(metrics_path, typ="series")
        config = pd.read_json(config_path, typ="series")
        rows.append(
            {
                "Model": str(config["model_name"]),
                "Bundle Key": str(config["bundle_key"]),
                "Bundle Dir": str(bundle_dir),
                "MAE": float(metrics["mae"]),
                "RMSE": float(metrics["rmse"]),
                "MAPE": float(metrics["mape"]),
                "Peak MAE": float(metrics["peak_mae"]),
                "Has Timeline": timeline_path.exists(),
            }
        )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df
    return metrics_df.sort_values(["MAE", "RMSE", "MAPE"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_bundle_cached(bundle_dir: str):
    return load_model_bundle(bundle_dir)


@st.cache_data(show_spinner=False)
def load_backtest_series(bundle_dir: str) -> pd.DataFrame:
    timeline_path = Path(bundle_dir) / "test_timeline.csv"
    if not timeline_path.exists():
        return pd.DataFrame(columns=["Local Time", "PM25", "Predicted"])

    timeline_df = pd.read_csv(timeline_path)
    timeline_df["timestamp"] = pd.to_datetime(timeline_df["timestamp"])
    backtest_df = timeline_df.rename(
        columns={"timestamp": "Local Time", "y_true": "PM25", "y_pred": "Predicted"}
    )
    return backtest_df[["Local Time", "PM25", "Predicted"]].tail(36).reset_index(drop=True)


def clean_model_name(name: str) -> str:
    return name.replace(" (Best Model)", "")


def model_color(name: str) -> str:
    cleaned = clean_model_name(name)
    for key, color in MODEL_COLORS.items():
        if clean_model_name(key) == cleaned:
            return color
    return "#5b8def"


def soft_color(hex_color: str, alpha: float = 0.14) -> str:
    color = hex_color.lstrip("#")
    red, green, blue = (int(color[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def pm25_band(value: float) -> tuple[str, str]:
    bands = [
        (15, "Tot", "#22c55e"),
        (25, "Trung binh", "#f59e0b"),
        (35, "Kem", "#fb923c"),
        (55, "Nhay cam", "#ef4444"),
        (100, "Khong tot", "#dc2626"),
    ]
    for upper, label, color in bands:
        if value <= upper:
            return label, color
    return "Rat xau", "#9333ea"


def forecast_delta_text(delta: float) -> str:
    prefix = "+" if delta >= 0 else "-"
    return f"{prefix}{abs(delta):.1f} ug/m3"


def safe_pct_change(current: float, future: float) -> float:
    if np.isclose(current, 0.0):
        return 0.0
    return (future - current) / current * 100


def make_future_covariates(history: pd.DataFrame, required_cols: list[str], step_hours: int, steps: int) -> pd.DataFrame:
    if history.empty:
        raise ValueError("History is empty, cannot build future covariates.")

    base_row = history.iloc[-1]
    future_index = pd.date_range(
        history["Local Time"].iloc[-1] + pd.Timedelta(hours=step_hours),
        periods=steps,
        freq=f"{step_hours}h",
    )
    future = pd.DataFrame({"Local Time": future_index})
    for col in required_cols:
        if col == "PM25":
            continue
        future[col] = base_row[col]
    return future


def build_forecast(history: pd.DataFrame, horizon_hours: int, bundle_dir: str) -> pd.DataFrame:
    bundle = load_bundle_cached(bundle_dir)
    horizon_steps = max(1, horizon_hours // bundle.step_hours)
    future_covariates = make_future_covariates(
        history=history,
        required_cols=bundle.required_raw_columns,
        step_hours=bundle.step_hours,
        steps=horizon_steps,
    )
    forecast_df = forecast_recursive(
        bundle,
        history_df=history,
        future_covariates_df=future_covariates,
        horizon=horizon_steps,
    )
    forecast = forecast_df.rename(columns={"y_pred": "PM25"})[["Local Time", "PM25"]].copy()
    forecast["Category"] = forecast["PM25"].apply(lambda value: pm25_band(float(value))[0])
    return forecast


def build_forecast_table(forecast: pd.DataFrame, current_pm25: float) -> pd.DataFrame:
    table = forecast.head(6).copy()
    table["Thoi gian"] = table["Local Time"].dt.strftime("%H:%M")
    table["PM2.5 du bao"] = table["PM25"]
    table["Thay doi"] = table["PM25"].apply(lambda value: round(safe_pct_change(current_pm25, float(value)), 1))
    table["Muc chat luong"] = table["PM25"].apply(lambda value: pm25_band(float(value))[0])
    return table[["Thoi gian", "PM2.5 du bao", "Thay doi", "Muc chat luong"]]


def html_metric_card(title: str, value: str, subtitle: str, accent: str, icon_text: str) -> str:
    return f"""
    <div class="metric-card" style="--accent:{accent}; --soft:{soft_color(accent, 0.12)};">
        <div class="metric-top">
            <div class="metric-icon">{icon_text}</div>
            <div class="metric-title">{title}</div>
        </div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """


def html_section_header(title: str, subtitle: str, action: str | None = None) -> str:
    action_html = f'<div class="section-action">{action}</div>' if action else ""
    return f"""
    <div class="section-head">
        <div>
            <div class="section-title">{title}</div>
            <div class="section-subtitle">{subtitle}</div>
        </div>
        {action_html}
    </div>
    """


def html_sidebar_brand(best_model_name: str, latest_time: str, row_count: int) -> str:
    return f"""
    <div class="sidebar-brand">
        <div class="brand-mark">AQ</div>
        <div>
            <div class="brand-title">
                <span>PM2.5</span>
                <span class="brand-accent">FORECAST</span>
            </div>
            <div class="brand-sub">Dashboard du bao chat luong khong khi cho demo Streamlit</div>
        </div>
    </div>
    <div class="sidebar-nav">
        <div class="nav-item active">Trang chu</div>
        <div class="nav-item">Chat luong mo hinh</div>
    </div>
    <div class="mini-card">
        <div class="mini-label">Best model</div>
        <b>{clean_model_name(best_model_name)}</b>
        <div class="mini-text">Cap nhat du lieu luc {latest_time}</div>
    </div>
    <div class="mini-card">
        <div class="mini-label">Dataset</div>
        <b>{row_count:,} ban ghi</b>
        <div class="mini-text">Du lieu lich su san sang de demo ca du bao va so sanh mo hinh.</div>
    </div>
    """


def html_forecast_chips(forecast: pd.DataFrame) -> str:
    cards = []
    preview = forecast.head(8).reset_index(drop=True)
    peak_pos = int(preview["PM25"].idxmax())
    previous_value = None

    for position, row in preview.iterrows():
        _, color = pm25_band(float(row["PM25"]))
        delta = 0.0 if previous_value is None else float(row["PM25"] - previous_value)
        delta_text = f"{delta:+.1f}" if previous_value is not None else "start"
        chip_class = "forecast-chip peak" if position == peak_pos else "forecast-chip"
        cards.append(
            f"""
            <div class="{chip_class}" style="--accent:{color}; --soft:{soft_color(color, 0.12)};">
                <div class="chip-top">
                    <div class="chip-time">{row["Local Time"].strftime("%H:%M")}</div>
                    <div class="chip-diff">{delta_text}</div>
                </div>
                <div class="chip-value">{row["PM25"]:.1f}</div>
                <div class="chip-label">{row["Category"]}</div>
            </div>
            """
        )
        previous_value = float(row["PM25"])

    return '<div class="forecast-chip-grid">' + "".join(cards) + "</div>"


def html_forecast_table(table: pd.DataFrame) -> str:
    rows_html = []
    for row in table.to_dict("records"):
        _, color = pm25_band(float(row["PM2.5 du bao"]))
        change_class = "change-pill up" if row["Thay doi"] >= 0 else "change-pill down"
        rows_html.append(
            f"""
            <tr>
                <td>{row["Thoi gian"]}</td>
                <td><span class="number-pill" style="--accent:{color}; --soft:{soft_color(color, 0.12)};">{row["PM2.5 du bao"]:.1f} ug/m3</span></td>
                <td><span class="{change_class}">{row["Thay doi"]:+.1f}%</span></td>
                <td><span class="quality-pill" style="--accent:{color}; --soft:{soft_color(color, 0.12)};">{row["Muc chat luong"]}</span></td>
            </tr>
            """
        )

    return f"""
    <table class="forecast-table">
        <thead>
            <tr>
                <th>Thoi gian</th>
                <th>PM2.5 du bao</th>
                <th>Thay doi</th>
                <th>Muc chat luong</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows_html)}
        </tbody>
    </table>
    """


def html_alert_box(forecast: pd.DataFrame) -> str:
    peak_row = forecast.loc[forecast["PM25"].idxmax()]
    peak_label, peak_color = pm25_band(float(peak_row["PM25"]))
    early_mean = float(forecast.head(6)["PM25"].mean())
    early_label, _ = pm25_band(early_mean)

    alerts = [
        f"Khung 6 gio dau co xu huong {early_label.lower()}, can theo doi truoc gio cao diem.",
        "Nhom nhay cam nen giam thoi gian hoat dong ngoai troi khi muc canh bao chuyen sang do.",
        "Dong cua vao khung gio dong xe de giam bui min xam nhap trong nha.",
    ]
    list_html = "".join(f"<li>{item}</li>" for item in alerts)

    return f"""
    <div class="alert-card">
        <div class="alert-top">
            <div class="alert-icon">AL</div>
            <div>
                <div class="alert-title">Canh bao va nhan dinh</div>
                <div class="alert-subtitle">Tong hop nhanh de trinh bay trong buoi demo</div>
            </div>
        </div>
        <div class="alert-summary">
            PM2.5 du bao dat dinh <b>{peak_row["PM25"]:.1f} ug/m3</b> vao luc
            <b>{peak_row["Local Time"].strftime("%H:%M")}</b>.
        </div>
        <div class="alert-highlight" style="--accent:{peak_color}; --soft:{soft_color(peak_color, 0.14)};">
            Muc canh bao: {peak_label}
        </div>
        <ul class="alert-list">{list_html}</ul>
    </div>
    """


def html_model_table(metrics: pd.DataFrame, best_model_name: str) -> str:
    rows_html = []
    for row in metrics.to_dict("records"):
        accent = model_color(row["Model"])
        row_class = "model-row best" if row["Model"] == clean_model_name(best_model_name) else "model-row"
        badge = '<span class="model-badge">Best</span>' if row["Model"] == clean_model_name(best_model_name) else ""
        rows_html.append(
            f"""
            <tr class="{row_class}">
                <td>
                    <div class="model-name-cell">
                        <span class="model-dot" style="background:{accent};"></span>
                        <span>{row["Model"]}</span>
                        {badge}
                    </div>
                </td>
                <td>{row["MAE"]:.2f}</td>
                <td>{row["RMSE"]:.2f}</td>
                <td>{row["MAPE"]:.2f}%</td>
                <td>{row["Peak MAE"]:.2f}</td>
            </tr>
            """
        )

    return f"""
    <table class="model-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>MAPE</th>
                <th>Peak MAE</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows_html)}
        </tbody>
    </table>
    """


def make_pm25_chart(actual: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    history = actual.tail(16)
    x_min = min(list(history["Local Time"]) + list(forecast["Local Time"]))
    y_cap = max(100, float(max(history["PM25"].max(), forecast["PM25"].max()) + 18))

    fig = go.Figure()

    zones = [
        (0, 15, "Tot", "rgba(34, 197, 94, 0.10)"),
        (15, 25, "Trung binh", "rgba(245, 158, 11, 0.10)"),
        (25, 35, "Kem", "rgba(251, 146, 60, 0.10)"),
        (35, 55, "Nhay cam", "rgba(239, 68, 68, 0.09)"),
        (55, y_cap, "Khong tot", "rgba(220, 38, 38, 0.08)"),
    ]
    for low, high, label, color in zones:
        fig.add_hrect(y0=low, y1=high, fillcolor=color, line_width=0)
        fig.add_annotation(
            x=x_min,
            y=(low + high) / 2,
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font={"size": 10, "color": "#8b96a8"},
        )

    fig.add_trace(
        go.Scatter(
            x=history["Local Time"],
            y=history["PM25"],
            mode="lines+markers",
            name="Gia tri thuc te",
            line={"color": "#4f8dfd", "width": 3.2, "shape": "spline"},
            marker={"size": 6, "color": "#4f8dfd"},
            hovertemplate="%{x|%H:%M %d/%m}<br>PM2.5: %{y:.1f} ug/m3<extra>Thuc te</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["Local Time"],
            y=forecast["PM25"],
            mode="lines+markers",
            name="Du bao",
            line={"color": "#22c55e", "width": 3.0, "dash": "dot", "shape": "spline"},
            marker={"size": 6, "color": "#22c55e"},
            hovertemplate="%{x|%H:%M %d/%m}<br>PM2.5: %{y:.1f} ug/m3<extra>Du bao</extra>",
        )
    )

    current_x = history["Local Time"].iloc[-1]
    peak_row = forecast.loc[forecast["PM25"].idxmax()]
    fig.add_vline(x=current_x, line_color="#89b4ff", line_dash="dot", line_width=1.5)
    fig.add_annotation(
        x=current_x,
        y=y_cap - 4,
        text="Hien tai",
        showarrow=False,
        bgcolor="#eef5ff",
        bordercolor="#cfe0ff",
        borderpad=5,
        font={"color": "#396fdb", "size": 10},
    )
    fig.add_annotation(
        x=peak_row["Local Time"],
        y=float(peak_row["PM25"]) + 7,
        text=f"{peak_row['PM25']:.1f} ug/m3",
        showarrow=True,
        ax=0,
        ay=-28,
        arrowcolor="#22c55e",
        bgcolor="#edfdf4",
        bordercolor="#b8efca",
        borderpad=6,
        font={"color": "#138a4f", "size": 10},
    )

    fig.update_layout(
        height=355,
        margin={"l": 18, "r": 18, "t": 12, "b": 8},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        hoverlabel={"bgcolor": "white", "font_size": 11, "font_family": "Segoe UI"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11},
        },
        xaxis={
            "showgrid": False,
            "tickformat": "%H:%M",
            "title": "",
            "tickfont": {"color": "#738195", "size": 11},
            "nticks": 10,
        },
        yaxis={
            "title": "PM2.5 (ug/m3)",
            "range": [0, y_cap],
            "gridcolor": "rgba(15, 23, 42, 0.08)",
            "zeroline": False,
            "tickfont": {"color": "#738195", "size": 11},
        },
    )
    return fig


def make_model_compare_chart(metrics: pd.DataFrame) -> go.Figure:
    colors = [model_color(name) for name in metrics["Model"]]
    fig = go.Figure(
        go.Bar(
            x=metrics["Model"],
            y=metrics["MAE"],
            marker={"color": colors, "line": {"width": 0}},
            text=metrics["MAE"].map(lambda value: f"{value:.2f}"),
            textposition="outside",
            hovertemplate="%{x}<br>MAE: %{y:.2f} ug/m3<extra></extra>",
        )
    )
    fig.update_traces(width=0.56)
    fig.update_layout(
        height=315,
        margin={"l": 18, "r": 18, "t": 12, "b": 22},
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis={
            "title": "MAE (ug/m3)",
            "gridcolor": "rgba(15, 23, 42, 0.08)",
            "range": [0, float(metrics["MAE"].max() + 1.2)],
        },
        xaxis={"title": "", "tickfont": {"size": 11}},
        showlegend=False,
    )
    return fig


def make_backtest_chart(backtest_df: pd.DataFrame, model_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=backtest_df["Local Time"],
            y=backtest_df["PM25"],
            mode="lines",
            name="Thuc te",
            line={"color": "#4f8dfd", "width": 3, "shape": "spline"},
            hovertemplate="%{x|%H:%M %d/%m}<br>PM2.5: %{y:.1f} ug/m3<extra>Thuc te</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=backtest_df["Local Time"],
            y=backtest_df["Predicted"],
            mode="lines",
            name=f"Du bao - {clean_model_name(model_name)}",
            line={"color": "#22c55e", "width": 3, "dash": "dot", "shape": "spline"},
            hovertemplate="%{x|%H:%M %d/%m}<br>PM2.5: %{y:.1f} ug/m3<extra>Du bao</extra>",
        )
    )
    fig.update_layout(
        height=300,
        margin={"l": 18, "r": 18, "t": 12, "b": 8},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.05, "x": 0},
        yaxis={"title": "PM2.5 (ug/m3)", "gridcolor": "rgba(15, 23, 42, 0.08)"},
        xaxis={"title": "", "tickformat": "%d/%m %H:%M"},
    )
    return fig


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #eef4fb;
                --panel: #ffffff;
                --line: #dfe8f4;
                --text: #17324d;
                --muted: #718197;
                --navy: #12284b;
                --navy-dark: #0d1f3c;
                --shadow: 0 18px 42px rgba(15, 23, 42, 0.08);
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(91, 141, 239, 0.14), transparent 20%),
                    radial-gradient(circle at left top, rgba(23, 178, 106, 0.10), transparent 26%),
                    linear-gradient(180deg, #f6f9fd 0%, var(--bg) 100%);
                color: var(--text);
                font-family: "Trebuchet MS", "Segoe UI", sans-serif;
            }

            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                padding-left: 1.2rem;
                padding-right: 1.2rem;
                max-width: 1480px;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, var(--navy-dark) 0%, var(--navy) 60%, #102240 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }

            [data-testid="stSidebar"] .block-container {
                padding-top: 1rem;
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }

            [data-testid="stSidebar"] * {
                color: #edf4ff;
            }

            .sidebar-brand {
                display: flex;
                gap: 0.8rem;
                align-items: center;
                margin-bottom: 1rem;
            }

            .brand-mark {
                width: 46px;
                height: 46px;
                border-radius: 16px;
                background: linear-gradient(135deg, #1fe28d, #23b5d3);
                color: #08233b;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 900;
                letter-spacing: 0.04em;
                box-shadow: 0 16px 28px rgba(31, 226, 141, 0.20);
            }

            .brand-title {
                font-size: 1.2rem;
                font-weight: 900;
                line-height: 1.02;
                letter-spacing: 0.02em;
            }

            .brand-title span {
                display: block;
            }

            .brand-accent {
                color: #36d399;
            }

            .brand-sub {
                margin-top: 0.28rem;
                font-size: 0.82rem;
                line-height: 1.45;
                color: rgba(237, 244, 255, 0.74);
            }

            .sidebar-nav {
                display: grid;
                gap: 0.55rem;
                margin-bottom: 1rem;
            }

            .nav-item {
                padding: 0.8rem 0.95rem;
                border-radius: 16px;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.06);
                font-size: 0.88rem;
                font-weight: 800;
            }

            .nav-item.active {
                background: linear-gradient(135deg, rgba(31, 226, 141, 0.20), rgba(35, 181, 211, 0.18));
                border-color: rgba(31, 226, 141, 0.30);
            }

            .sidebar-caption {
                margin-top: 1rem;
                margin-bottom: 0.5rem;
                padding-top: 1rem;
                border-top: 1px solid rgba(255, 255, 255, 0.10);
                font-size: 0.72rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: rgba(237, 244, 255, 0.60);
            }

            .mini-card {
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                margin-top: 0.85rem;
            }

            .mini-label {
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: rgba(237, 244, 255, 0.60);
            }

            .mini-card b {
                display: block;
                margin-top: 0.28rem;
                margin-bottom: 0.22rem;
                font-size: 1rem;
            }

            .mini-text {
                font-size: 0.8rem;
                line-height: 1.45;
                color: rgba(237, 244, 255, 0.74);
            }

            [data-testid="stSidebar"] .stSelectbox label,
            [data-testid="stSidebar"] .stSlider label {
                font-weight: 800;
                color: #dbecff;
            }

            [data-testid="stSidebar"] div[data-baseweb="select"] > div {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.10);
                border-radius: 14px;
                min-height: 46px;
            }

            [data-testid="stSidebar"] .stSlider [role="slider"] {
                background: #1fd28a;
                border: 2px solid white;
                box-shadow: 0 0 0 4px rgba(31, 210, 138, 0.20);
            }

            [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
                background: rgba(255, 255, 255, 0.22);
            }

            [data-testid="stSidebar"] .stButton > button {
                width: 100%;
                background: linear-gradient(135deg, #1fd28a, #15b779);
                color: #07253a;
                border: none;
                border-radius: 16px;
                padding: 0.85rem 1rem;
                font-weight: 900;
                letter-spacing: 0.04em;
                box-shadow: 0 18px 30px rgba(31, 210, 138, 0.26);
            }

            .topbar {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 1rem;
                margin-bottom: 1rem;
            }

            .hero-title {
                margin: 0;
                font-size: 2.15rem;
                font-weight: 900;
                line-height: 1.04;
                color: #102a43;
            }

            .hero-subtitle {
                margin-top: 0.3rem;
                color: var(--muted);
                font-size: 0.96rem;
            }

            .hero-meta {
                display: flex;
                gap: 0.55rem;
                flex-wrap: wrap;
                justify-content: flex-end;
            }

            .meta-pill {
                background: rgba(255, 255, 255, 0.88);
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 0.48rem 0.82rem;
                color: #4b5d71;
                font-size: 0.82rem;
                font-weight: 800;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
            }

            .metric-card {
                background: linear-gradient(180deg, var(--soft), #ffffff 76%);
                border: 1px solid var(--line);
                border-radius: 22px;
                padding: 1rem 1rem 0.95rem;
                box-shadow: var(--shadow);
                min-height: 128px;
            }

            .metric-top {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 0.8rem;
            }

            .metric-icon {
                width: 40px;
                height: 40px;
                border-radius: 14px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(255, 255, 255, 0.88);
                border: 1px solid rgba(255, 255, 255, 0.80);
                color: var(--accent);
                font-size: 0.78rem;
                font-weight: 900;
            }

            .metric-title {
                font-size: 0.9rem;
                font-weight: 800;
                color: #66778c;
                line-height: 1.35;
            }

            .metric-value {
                color: #17324d;
                font-size: 2rem;
                font-weight: 900;
                line-height: 1.04;
                letter-spacing: -0.02em;
            }

            .metric-subtitle {
                margin-top: 0.35rem;
                color: var(--muted);
                font-size: 0.82rem;
                line-height: 1.45;
            }

            .section-shell {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid var(--line);
                border-radius: 24px;
                padding: 1rem 1rem 0.7rem;
                box-shadow: var(--shadow);
            }

            .section-head {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 1rem;
                margin-bottom: 0.9rem;
            }

            .section-title {
                font-size: 1.05rem;
                font-weight: 900;
                color: #17324d;
            }

            .section-subtitle {
                margin-top: 0.22rem;
                font-size: 0.84rem;
                color: var(--muted);
                line-height: 1.45;
            }

            .section-action {
                white-space: nowrap;
                background: #eef5ff;
                border: 1px solid #d6e5ff;
                border-radius: 999px;
                padding: 0.42rem 0.72rem;
                color: #4f78d1;
                font-size: 0.78rem;
                font-weight: 900;
            }

            .forecast-chip-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.72rem;
                margin-bottom: 0.75rem;
            }

            .forecast-chip {
                position: relative;
                overflow: hidden;
                background: linear-gradient(180deg, var(--soft), #ffffff 72%);
                border: 1px solid #e5edf8;
                border-radius: 18px;
                padding: 0.8rem 0.82rem 0.74rem;
                box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
            }

            .forecast-chip::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: var(--accent);
            }

            .forecast-chip.peak {
                transform: translateY(-2px);
                box-shadow: 0 18px 34px rgba(15, 23, 42, 0.08);
            }

            .chip-top {
                display: flex;
                justify-content: space-between;
                gap: 0.6rem;
                align-items: center;
            }

            .chip-time {
                color: #6b7a90;
                font-size: 0.8rem;
                font-weight: 800;
            }

            .chip-diff {
                color: var(--accent);
                font-size: 0.72rem;
                font-weight: 900;
            }

            .chip-value {
                margin-top: 0.42rem;
                color: #17324d;
                font-size: 1.3rem;
                font-weight: 900;
            }

            .chip-label {
                display: inline-flex;
                align-items: center;
                margin-top: 0.3rem;
                border-radius: 999px;
                padding: 0.25rem 0.55rem;
                background: rgba(255, 255, 255, 0.72);
                color: var(--accent);
                font-size: 0.76rem;
                font-weight: 900;
            }

            .forecast-table,
            .model-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0 0.45rem;
            }

            .forecast-table th,
            .model-table th {
                text-align: left;
                padding: 0 0.72rem 0.16rem;
                color: #7d8ca0;
                font-size: 0.75rem;
                font-weight: 900;
            }

            .forecast-table td,
            .model-table td {
                background: #fbfdff;
                border-top: 1px solid #e8eef7;
                border-bottom: 1px solid #e8eef7;
                padding: 0.78rem 0.72rem;
                font-size: 0.85rem;
                color: #304559;
            }

            .forecast-table td:first-child,
            .model-table td:first-child {
                border-left: 1px solid #e8eef7;
                border-top-left-radius: 15px;
                border-bottom-left-radius: 15px;
                font-weight: 800;
            }

            .forecast-table td:last-child,
            .model-table td:last-child {
                border-right: 1px solid #e8eef7;
                border-top-right-radius: 15px;
                border-bottom-right-radius: 15px;
            }

            .number-pill,
            .quality-pill {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 0.28rem 0.56rem;
                background: var(--soft);
                color: var(--accent);
                font-weight: 900;
            }

            .change-pill {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 0.28rem 0.56rem;
                font-weight: 900;
            }

            .change-pill.up {
                background: #eafbf2;
                color: #138a4f;
            }

            .change-pill.down {
                background: #fff1f1;
                color: #dc4444;
            }

            .alert-card {
                background: linear-gradient(180deg, #fff7eb 0%, #fffdf9 100%);
                border: 1px solid #f4d7ae;
                border-radius: 22px;
                padding: 1rem;
                min-height: 100%;
                box-shadow: var(--shadow);
            }

            .alert-top {
                display: flex;
                gap: 0.75rem;
                align-items: center;
                margin-bottom: 0.9rem;
            }

            .alert-icon {
                width: 40px;
                height: 40px;
                border-radius: 14px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #fff1dc;
                color: #ef8c2f;
                font-size: 0.78rem;
                font-weight: 900;
            }

            .alert-title {
                font-size: 1rem;
                font-weight: 900;
                color: #764b23;
            }

            .alert-subtitle {
                margin-top: 0.18rem;
                font-size: 0.8rem;
                color: #9a6a3a;
            }

            .alert-summary {
                color: #6e4723;
                line-height: 1.55;
                margin-bottom: 0.8rem;
            }

            .alert-highlight {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 0.32rem 0.68rem;
                background: var(--soft);
                color: var(--accent);
                font-weight: 900;
                margin-bottom: 0.85rem;
            }

            .alert-list {
                list-style: none;
                padding: 0;
                margin: 0;
                display: grid;
                gap: 0.65rem;
            }

            .alert-list li {
                background: rgba(255, 255, 255, 0.74);
                border: 1px solid rgba(244, 215, 174, 0.78);
                border-radius: 16px;
                padding: 0.75rem 0.8rem;
                color: #7a4d22;
                line-height: 1.5;
            }

            .model-row.best td {
                background: linear-gradient(180deg, #eefcf5 0%, #ffffff 100%);
                border-top-color: #d4f1df;
                border-bottom-color: #d4f1df;
            }

            .model-row.best td:first-child {
                border-left-color: #d4f1df;
            }

            .model-row.best td:last-child {
                border-right-color: #d4f1df;
            }

            .model-name-cell {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-weight: 800;
            }

            .model-dot {
                width: 10px;
                height: 10px;
                border-radius: 999px;
                flex-shrink: 0;
            }

            .model-badge {
                margin-left: auto;
                background: #eafbf2;
                color: #138a4f;
                padding: 0.16rem 0.5rem;
                border-radius: 999px;
                font-size: 0.7rem;
                font-weight: 900;
            }

            .content-divider {
                height: 1px;
                background: linear-gradient(90deg, transparent, #d7e2f0 18%, #d7e2f0 82%, transparent);
                margin: 1.3rem 0;
            }

            .conclusion-card {
                background: linear-gradient(180deg, #effcf5 0%, #ffffff 100%);
                border: 1px solid #c7ecd5;
                border-radius: 22px;
                padding: 1rem;
                color: #1f5c44;
                line-height: 1.62;
                box-shadow: var(--shadow);
            }

            .conclusion-card b {
                color: #11824a;
            }

            @media (max-width: 1200px) {
                .forecast-chip-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            @media (max-width: 980px) {
                .topbar {
                    flex-direction: column;
                }

                .hero-meta {
                    justify-content: flex-start;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    model_df = load_model_frame()
    metrics_df = load_registry_metrics()
    if metrics_df.empty:
        st.error("Khong tim thay artifact hop le trong model_registry.")
        return

    best_model_row = metrics_df.iloc[0]
    best_model_name = str(best_model_row["Model"])
    latest_dataset_time = model_df["Local Time"].iloc[-1].strftime("%H:%M - %d/%m/%Y")

    with st.sidebar:
        st.markdown(
            html_sidebar_brand(best_model_name, latest_dataset_time, len(model_df)),
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-caption">Tuy chinh du bao</div>', unsafe_allow_html=True)
        model_options = metrics_df["Model"].tolist()
        model_name = st.selectbox("Chon mo hinh", model_options, index=model_options.index(best_model_name))
        horizon = st.slider("So gio du bao", min_value=6, max_value=48, value=24, step=6)
        history_hours = st.slider("So gio lich su hien thi", min_value=12, max_value=48, value=24, step=4)
        st.button("Du bao", type="primary")

    selected_metrics = metrics_df.loc[metrics_df["Model"] == model_name].iloc[0]
    bundle_dir = str(selected_metrics["Bundle Dir"])
    bundle = load_bundle_cached(bundle_dir)

    history = model_df.copy()
    history_steps = max(4, history_hours // bundle.step_hours)
    chart_history = history.tail(history_steps)
    current_row = history.iloc[-1]
    current_pm25 = float(current_row["PM25"])
    forecast = build_forecast(history, horizon, bundle_dir)
    next_pm25 = float(forecast["PM25"].iloc[0])
    next_label, _ = pm25_band(next_pm25)
    band_label, band_color = pm25_band(current_pm25)
    delta = next_pm25 - current_pm25
    delta_pct = safe_pct_change(current_pm25, next_pm25)
    latest_time = current_row["Local Time"].strftime("%H:%M - %d/%m/%Y")
    backtest_df = load_backtest_series(bundle_dir)
    forecast_table = build_forecast_table(forecast, current_pm25)

    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="hero-title">He thong du bao PM2.5</div>
                <div class="hero-subtitle">
                    Ung dung Deep Learning cho du bao chat luong khong khi tai Hai Ba Trung, Ha Noi
                </div>
            </div>
            <div class="hero-meta">
                <div class="meta-pill">Best Model: {clean_model_name(best_model_name)}</div>
                <div class="meta-pill">Horizon: {horizon} gio</div>
                <div class="meta-pill">Cap nhat: {latest_time}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown(
            html_metric_card(
                "PM2.5 hien tai",
                f"{current_pm25:.1f} ug/m3",
                f"Muc chat luong: {band_label}",
                "#4f8dfd",
                "PM",
            ),
            unsafe_allow_html=True,
        )
    with metric_cols[1]:
        st.markdown(
            html_metric_card(
                f"Du bao gio toi ({forecast.iloc[0]['Local Time'].strftime('%H:%M')})",
                f"{next_pm25:.1f} ug/m3",
                f"Muc du bao: {next_label}",
                "#8b5cf6",
                "01",
            ),
            unsafe_allow_html=True,
        )
    with metric_cols[2]:
        st.markdown(
            html_metric_card(
                "Thay doi",
                forecast_delta_text(delta),
                f"So voi hien tai: {delta_pct:+.1f}%",
                "#f59e0b",
                "DEL",
            ),
            unsafe_allow_html=True,
        )
    with metric_cols[3]:
        st.markdown(
            html_metric_card(
                "Chat luong khong khi",
                band_label,
                f"Moc hien tai: {current_pm25:.1f} ug/m3",
                band_color,
                "AQ",
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown(
        html_section_header(
            "Dien bien PM2.5 va du bao trong 24 gio toi",
            "Theo doi xu huong thuc te va du bao trong mot khung nhin giong dashboard product.",
        ),
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_pm25_chart(chart_history, forecast), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    top_lower_left, top_lower_right = st.columns([1.65, 1.0], gap="large")

    with top_lower_left:
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        st.markdown(
            html_section_header(
                "Du bao PM2.5 theo gio",
                "8 moc dau tien cua chuoi du bao va bang tom tat de trinh bay nhanh.",
                "Xem chi tiet",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(html_forecast_chips(forecast), unsafe_allow_html=True)
        st.markdown(html_forecast_table(forecast_table), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with top_lower_right:
        st.markdown(html_alert_box(forecast), unsafe_allow_html=True)

    st.markdown('<div class="content-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.markdown(
        html_section_header(
            "Danh gia chat luong mo hinh",
            "So sanh nhanh hieu nang cac mo hinh tren artifact test da luu trong model_registry.",
        ),
        unsafe_allow_html=True,
    )

    score_cols = st.columns(4)
    with score_cols[0]:
        st.markdown(
            html_metric_card(
                "Model tot nhat",
                clean_model_name(best_model_name),
                f"Peak MAE = {best_model_row['Peak MAE']:.2f}",
                "#17b26a",
                "TOP",
            ),
            unsafe_allow_html=True,
        )
    with score_cols[1]:
        st.markdown(
            html_metric_card(
                "MAE",
                f"{selected_metrics['MAE']:.2f} ug/m3",
                clean_model_name(model_name),
                "#5b8def",
                "MAE",
            ),
            unsafe_allow_html=True,
        )
    with score_cols[2]:
        st.markdown(
            html_metric_card(
                "RMSE",
                f"{selected_metrics['RMSE']:.2f} ug/m3",
                clean_model_name(model_name),
                "#8b5cf6",
                "RMS",
            ),
            unsafe_allow_html=True,
        )
    with score_cols[3]:
        st.markdown(
            html_metric_card(
                "MAPE",
                f"{selected_metrics['MAPE']:.2f}%",
                clean_model_name(model_name),
                "#f59e0b",
                "MAP",
            ),
            unsafe_allow_html=True,
        )

    model_col, compare_col = st.columns([1.15, 1.0], gap="large")
    with model_col:
        st.markdown(html_model_table(metrics_df, best_model[0]), unsafe_allow_html=True)
    with compare_col:
        st.plotly_chart(make_model_compare_chart(metrics_df), use_container_width=True)

    bt_col, note_col = st.columns([1.35, 0.8], gap="large")
    with bt_col:
        st.plotly_chart(make_backtest_chart(backtest_df, model_name), use_container_width=True)
    with note_col:
        st.markdown(
            f"""
            <div class="conclusion-card">
                <div class="section-title">Ket luan</div>
                <b>{clean_model_name(best_model_name)}</b> dang cho ket qua on dinh nhat voi
                MAE <b>{best_model_row['MAE']:.2f}</b> va Peak MAE <b>{best_model_row['Peak MAE']:.2f}</b>.
                Dashboard hien da doc metric va backtest that tu artifact trong `model_registry`,
                dong thoi forecast o panel tren duoc sinh boi model da train thay vi du lieu gia lap.
                <br><br>
                Gio app da dung `inference.py`; buoc tiep theo hop ly nhat la bo hardcode cu va
                viet script promote de cap nhat `best_model_bundle` tu dong.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
