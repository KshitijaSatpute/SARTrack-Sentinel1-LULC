"""SARTrack V3 — working Streamlit dashboard adapted to supplied HTML design.

New UI entrypoint only. The existing sartrack_pipeline_v3.py, models, data,
monthly percentage calculations and saved V2 demo files are not modified.
Run from the existing project root:
    streamlit run app_v3_designed.py --server.port 8503
"""

from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
DEMO_MONTHLY = ROOT / "outputs" / "monthly"
DEMO_STABILITY = ROOT / "outputs" / "stability"
CLASSES = ["Water", "Trees", "Grass", "Crops", "Built-up"]
CLASS_COLORS = ["#38BDF8", "#22A06B", "#A3D977", "#E9C65B", "#DF6864"]


def add_months(year: int, month: int, offset: int) -> date:
    position = year * 12 + (month - 1) + offset
    return date(position // 12, position % 12 + 1, 1)


def result_paths(source: str, folder: Path, month_date: date) -> tuple[Path, Path]:
    if source == "demo":
        prefix = f"Esch_{month_date:%B%Y}_LULC"
        return DEMO_MONTHLY / f"{prefix}.png", DEMO_MONTHLY / f"{prefix}.tif"
    prefix = f"LULC_{month_date:%Y_%m}"
    return folder / f"{prefix}.png", folder / f"{prefix}.tif"


def stability_paths(source: str, folder: Path) -> tuple[Path, ...]:
    if source == "demo":
        return (
            DEMO_STABILITY / "Esch_2025_change_count.png",
            DEMO_STABILITY / "Esch_2025_change_count.tif",
            DEMO_STABILITY / "Esch_2025_stable_class.png",
            DEMO_STABILITY / "Esch_2025_stable_class.tif",
            DEMO_STABILITY / "stability_summary.txt",
        )
    return (
        folder / "change_count.png",
        folder / "change_count.tif",
        folder / "stable_class.png",
        folder / "stable_class.tif",
        folder / "stability_summary.txt",
    )


def read_statistics(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Statistics CSV missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12:
        raise ValueError(f"Expected 12 monthly rows, found {len(rows)}")
    for record in rows:
        for class_name in CLASSES:
            float(record[f"{class_name}_pct"])
    return rows


def validate_complete_results(source: str, folder: Path, year: int, month: int) -> Path:
    csv_path = (
        DEMO_MONTHLY / "monthly_land_cover_statistics.csv"
        if source == "demo" else folder / "monthly_land_cover_statistics.csv"
    )
    read_statistics(csv_path)
    required = [csv_path]
    for offset in range(12):
        required.extend(result_paths(source, folder, add_months(year, month, offset)))
    required.extend(stability_paths(source, folder))
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Incomplete results: " + ", ".join(missing[:5]))
    return csv_path


def read_stability(path: Path) -> float | None:
    if not path.is_file():
        return None
    match = re.search(r"Overall stability:\s*([\d.]+)%", path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None


def percent(record: dict[str, str], name: str) -> float:
    return float(record[f"{name}_pct"])


def show_image(path: Path, caption: str | None = None) -> None:
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Image missing: {path.name}")


def download_file(path: Path, label: str, mime: str) -> None:
    if path.is_file():
        st.download_button(
            label, data=path.read_bytes(), file_name=path.name,
            mime=mime, use_container_width=True,
        )
    else:
        st.warning(f"Download not available: {path.name}")


def class_legend() -> None:
    chips = "".join(
        '<span class="legend-chip"><i style="background:{}"></i>{}</span>'.format(color, escape(name))
        for name, color in zip(CLASSES, CLASS_COLORS)
    )
    st.markdown(f'<div class="legend-row">{chips}</div>', unsafe_allow_html=True)


def stats_table(months: list[date], rows: list[dict[str, str]], changes: bool = False) -> None:
    headers = "".join(f"<th>{escape(name)}</th>" for name in CLASSES)
    table_rows = []
    for index, (month, record) in enumerate(zip(months, rows)):
        if changes:
            values = (
                ["—"] * len(CLASSES) if index == 0 else [
                    f"{percent(record, name) - percent(rows[index-1], name):+.2f}"
                    for name in CLASSES
                ]
            )
        else:
            values = [f"{percent(record, name):.2f}%" for name in CLASSES]
        cells = "".join(f"<td>{escape(value)}</td>" for value in values)
        table_rows.append(f"<tr><th scope='row'>{month:%B %Y}</th>{cells}</tr>")
    st.markdown(
        '<div class="data-table-wrap"><table class="sar-table">'
        f'<thead><tr><th>Month</th>{headers}</tr></thead>'
        f'<tbody>{"".join(table_rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def location_map(lat: float, lon: float) -> None:
    """Read-only basemap. Never suggests a classification has been computed."""
    components.html(
        f'''<!DOCTYPE html><html><head><meta charset="utf-8" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>html,body,#map{{height:100%;width:100%;margin:0;background:#eaf2f8;}}
        .leaflet-container{{font-family:system-ui, sans-serif;}}</style></head>
        <body><div id="map"></div>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
        const lat={lat:.6f}, lon={lon:.6f};
        const map=L.map('map',{{scrollWheelZoom:false}}).setView([lat,lon],12);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
          maxZoom:18,attribution:'&copy; OpenStreetMap contributors'
        }}).addTo(map);
        L.circleMarker([lat,lon],{{radius:9,weight:3,color:'#1d4ed8',fillColor:'#06b6d4',fillOpacity:.75}})
          .addTo(map).bindPopup('Selected location');
        </script></body></html>''',
        height=335,
        scrolling=False,
    )


st.set_page_config(
    page_title="SARTrack | Sentinel-1 Temporal Analytics",
    page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed",
)

st.markdown(r"""
<style>
:root { --ink:#132238; --muted:#526477; --teal:#047857; --line:#dce7e5; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background:#ffffff !important; color:var(--ink) !important;
}
header[data-testid="stHeader"], [data-testid="stDecoration"],
[data-testid="stToolbar"], #MainMenu {display:none !important;}
footer {visibility:hidden;}
[data-testid="stSidebar"] {display:none !important;}
.block-container { max-width:1360px; margin:auto; padding:13px 22px 28px; }
[data-testid="stVerticalBlock"] {gap:.85rem;}
[data-testid="stVerticalBlockBorderWrapper"] {
    background:#fff !important; border:1px solid var(--line) !important;
    border-radius:17px !important;box-shadow:0 4px 16px rgba(15,23,42,.045);
}
.sar-header {width:100%; display:flex; align-items:center; justify-content:space-between;
    padding:13px 20px; background:#fff; border:1px solid #e1e8ec;
    border-radius:16px; margin:0 0 13px; gap:15px; }
.sar-brand {display:flex;gap:12px;align-items:center;}
.sar-logo {width:48px;height:48px;display:grid;place-items:center;border-radius:12px;
    background:linear-gradient(130deg,#047857,#0d9488,#06b6d4);font-size:25px;}
.sar-name {font-size:1.87rem;color:#102c36;font-weight:850;letter-spacing:-.045em;line-height:1.15;}
.sar-tagline {color:#64748b;font-size:.7rem;letter-spacing:.08em;font-weight:700;}
.sar-header-nav {display:flex;gap:7px;flex-wrap:wrap;}
.sar-header-nav span {padding:7px 12px;border-radius:9px;border:1px solid #d1e8df;
    background:#f0fdf8;color:#047857;font-weight:700;font-size:.81rem;}
.sar-hero {border-radius:19px;border:1px solid #d8e9e1;
    background:linear-gradient(115deg,#fff 40%,#f0fdfa 100%);
    padding:20px 25px;margin:0 0 15px;}
.sar-kicker {color:#047857;font-size:.73rem;letter-spacing:.09em;font-weight:800;}
.sar-hero h1 {color:#0f2438!important;font-size:clamp(1.7rem,2.65vw,2.45rem)!important;
    font-weight:830;margin:6px 0 7px;line-height:1.18;letter-spacing:-.035em;}
.sar-hero p {color:#516373!important;font-size:.94rem;margin:0;max-width:990px;}
.sar-badges {display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}
.sar-badges span {font-size:.77rem;color:#165a4b;border:1px solid #cee7de;
    background:#fff;border-radius:8px;padding:6px 11px;font-weight:650;}
.sar-kicker-sm {color:#047857;font-size:.72rem;font-weight:830;letter-spacing:.08em;}
.sar-panel-title {color:#132c36;font-size:1.32rem;font-weight:810;margin:3px 0;}
.sar-muted {color:#5a6e7c;font-size:.86rem;line-height:1.45;}
.sar-panel-anchor {height:0;overflow:hidden;}
/* Panels have matching minimum height on setup; do not crop the content. */
[data-testid="stHorizontalBlock"]:has(.sar-setup-panel):has(.sar-preview-panel) {align-items:stretch;}
[data-testid="stVerticalBlockBorderWrapper"]:has(.sar-setup-panel),
[data-testid="stVerticalBlockBorderWrapper"]:has(.sar-preview-panel) {min-height:466px !important;}
[data-testid="stNumberInput"] label p, [data-testid="stDateInput"] label p,
[data-testid="stSelectbox"] label p, [data-testid="stRadio"] label p {
    color:#213547!important;font-weight:720!important;font-size:1rem!important;
}
[data-testid="stNumberInput"] input,[data-testid="stDateInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background:#fff!important;color:#112c40!important;border:1px solid #a2b8c2!important;
    min-height:47px!important;font-size:1rem!important;border-radius:9px!important;
}
[data-testid="stNumberInput"] button {background:#f1f5f9!important;color:#123e4b!important;}
[data-testid="stNumberInput"] button * {color:#123e4b!important;}
div.stButton button {min-height:44px;border-radius:10px!important;font-weight:730!important;}
div.stButton button[kind="primary"] {background:linear-gradient(90deg,#059669,#0d9488)!important;
    color:white!important;border:1px solid #047857!important;box-shadow:0 5px 14px rgba(5,150,105,.15);}
div.stButton button[kind="primary"] * {color:white!important;}
div.stButton button:not([kind="primary"]) {background:#effbf6!important;
    border:1px solid #a5d7c6!important;color:#075e4b!important;}
div.stButton button:not([kind="primary"]) * {color:#075e4b!important;}
[data-testid="stMetric"] {min-height:107px!important;padding:11px 9px!important;
    border-radius:13px!important;border:1px solid #d8e8e3!important;
    border-top:3px solid #0d9488!important;background:#fff!important;
    display:flex!important;flex-direction:column!important;align-items:center!important;
    justify-content:center!important;text-align:center!important;}
[data-testid="stMetricLabel"],[data-testid="stMetricValue"] {
    width:100%!important;display:flex!important;justify-content:center!important;text-align:center!important;}
[data-testid="stMetricLabel"] p {color:#536675!important;font-weight:720!important;font-size:.85rem!important;}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] * {
    color:#102d38!important;font-weight:850!important;font-size:1.78rem!important;}
.sar-class-row {display:flex;flex-wrap:wrap;gap:9px;margin:9px 0 2px;}
.sar-class-chip {border:1px solid #d5e5df;border-radius:10px;background:#fff;
    padding:8px 11px;font-size:.84rem;color:#264352;font-weight:750;}
.sar-class-chip b {display:inline-block;width:11px;height:11px;border-radius:3px;
    margin-right:7px;vertical-align:middle;}
.sar-bar {margin:12px 0 15px;}
.sar-bar-top {display:flex;justify-content:space-between;align-items:center;
    font-size:.91rem;font-weight:710;color:#264353;margin-bottom:5px;}
.sar-track {height:10px;border-radius:20px;background:#eaf2f0;overflow:hidden;}
.sar-track span {display:block;height:100%;border-radius:20px;}
.sar-table-wrap {overflow-x:auto;border:1px solid #b9d2c8;border-radius:11px;margin:8px 0 16px;}
.sar-table {width:100%;border-collapse:collapse;background:#fff;font-size:.88rem;color:#163848;}
.sar-table th,.sar-table td {border:1px solid #b9d2c8!important;padding:10px 12px;text-align:right;}
.sar-table th:first-child {text-align:left;}
.sar-table thead th {background:#dcfce7;color:#124e42;}
.sar-table tbody tr:nth-child(even) {background:#f4fdf8;}
[data-testid="stDownloadButton"] button {min-height:45px!important;border:1px solid #8acdb8!important;
    border-radius:10px!important;background:#e6fbf1!important;color:#075b47!important;font-weight:750!important;}
[data-testid="stDownloadButton"] button * {color:#075b47!important;}
.sar-footer {color:#657887;font-size:.81rem;border-top:1px solid #e1e9e7;padding-top:14px;margin-top:20px;}
@media(max-width:900px) {
    .block-container {padding:11px 12px 25px;}
    .sar-header-nav {display:none;}
    .sar-hero {padding:18px;}
    [data-testid="stVerticalBlockBorderWrapper"]:has(.sar-setup-panel),
    [data-testid="stVerticalBlockBorderWrapper"]:has(.sar-preview-panel) {min-height:0!important;}
}
</style>
""", unsafe_allow_html=True)

# Keep the functional Streamlit app distinct from the standalone HTML mockup.
# The latter contains simulated progress and incorrect classes; this app does not.
PAGES = ["Overview", "Monthly Maps", "Area Changes", "Stability", "Downloads", "Model Details"]


def topbar(active_view: str) -> None:
    first = "Study Area Setup" if active_view == "setup" else "Analytics Dashboard"
    st.markdown(
        '<div class="sar-header"><div class="sar-brand"><div class="sar-logo">🛰️</div>'
        '<div><div class="sar-name">SARTrack</div>'
        '<div class="sar-tagline">SENTINEL-1 RADAR INTELLIGENCE</div></div></div>'
        f'<div class="sar-header-nav"><span>{escape(first)}</span>'
        '<span>12 Monthly Maps</span><span>U-Net V3 Live</span></div></div>',
        unsafe_allow_html=True,
    )


def legend() -> None:
    chips = "".join(
        f'<span class="sar-class-chip"><b style="background:{color}"></b>{escape(name)}</span>'
        for name, color in zip(CLASSES, CLASS_COLORS)
    )
    st.markdown(f'<div class="sar-class-row">{chips}</div>', unsafe_allow_html=True)


def class_bars(record: dict[str, str]) -> None:
    for name, color in zip(CLASSES, CLASS_COLORS):
        value = percent(record, name)
        safe_width = min(100., max(0., value))
        st.markdown(
            '<div class="sar-bar">'
            f'<div class="sar-bar-top"><span>{escape(name)}</span><span>{value:.2f}%</span></div>'
            f'<div class="sar-track"><span style="width:{safe_width:.3f}%;background:{color}"></span>'
            '</div></div>', unsafe_allow_html=True,
        )


def month_table(month_dates: list[date], rows: list[dict[str, str]], mode: str) -> None:
    heads = ''.join(f'<th>{escape(name)}</th>' for name in CLASSES)
    parts = []
    for idx, (month, record) in enumerate(zip(month_dates, rows)):
        if mode == 'distribution':
            cells = [f'{percent(record,name):.2f}%' for name in CLASSES]
        elif mode == 'monthly':
            cells = ['—'] * len(CLASSES) if idx == 0 else [
                f'{percent(record,name)-percent(rows[idx-1],name):+.2f}' for name in CLASSES
            ]
        else:
            raise ValueError('Unrecognized statistics mode')
        values = ''.join(f'<td>{escape(value)}</td>' for value in cells)
        parts.append(f'<tr><th>{month:%B %Y}</th>{values}</tr>')
    st.markdown('<div class="sar-table-wrap"><table class="sar-table">'
                f'<thead><tr><th>Month</th>{heads}</tr></thead><tbody>{"".join(parts)}</tbody>'
                '</table></div>', unsafe_allow_html=True)


def setup_page() -> None:
    topbar('setup')
    st.markdown("""<div class="sar-hero"><div class="sar-kicker">DUAL-POLARIZATION C-BAND SAR</div>
    <h1>Sentinel-1 Land Cover Classification &amp; Temporal Stability</h1>
    <p>Select a study-area center and starting month. Generate twelve consecutive monthly
    predictions with the trained U-Net and explore real land-cover statistics.</p>
    <div class="sar-badges"><span>VV + VH Input</span><span>12 Monthly LULC Maps</span>
    <span>5 Land-Cover Classes</span><span>10 m Nominal Sampling</span></div></div>""", unsafe_allow_html=True)

    left, right = st.columns(2, gap='large')
    with left:
        with st.container(border=True):
            st.markdown('<div class="sar-panel-anchor sar-setup-panel"></div>'
                        '<div class="sar-kicker-sm">CONFIGURE STUDY AREA</div>'
                        '<div class="sar-panel-title">Study Area Parameters</div>'
                        '<div class="sar-muted">Enter coordinates and the first analysis month.</div>',
                        unsafe_allow_html=True)
            a, b = st.columns(2, gap='medium')
            with a:
                lat = st.number_input('Latitude (°N)', min_value=-79.9999,
                                      max_value=83.9999, step=.0001, format='%.4f',
                                      key='sar_lat')
            with b:
                lon = st.number_input('Longitude (°E)', min_value=-180.,
                                      max_value=180., step=.0001, format='%.4f',
                                      key='sar_lon')
            today = date.today()
            latest = add_months(today.year, today.month, -12)
            start_date = st.date_input('Starting Month', min_value=date(2015,1,1),
                                       max_value=latest, key='sar_start_date')
            st.caption('The day is ignored. Processing starts on the 1st of your selected month.')
            generate = st.button('Generate 12 Monthly Maps', type='primary',
                                 use_container_width=True, key='sar_generate')
            saved = st.button('Open saved Esch demo · V2 reference',
                              use_container_width=True, key='sar_saved')
            st.caption('Live inference: U-Net V3  ·  Saved Esch demonstration: U-Net V2')

    with right:
        with st.container(border=True):
            st.markdown('<div class="sar-panel-anchor sar-preview-panel"></div>'
                        '<div class="sar-kicker-sm">GEOGRAPHIC CONTEXT</div>'
                        '<div class="sar-panel-title">Spatial Location Preview</div>'
                        f'<div class="sar-muted">Center: {lat:.4f}° N, {lon:.4f}° E'
                        ' · Basemap preview, not a predicted class map</div>',
                        unsafe_allow_html=True)
            location_map(lat, lon)
            st.caption('Change the coordinates on the left to select another location.')

    legend()
    if saved:
        try:
            validate_complete_results('demo', ROOT, 2025, 1)
            st.session_state.sartrack_active = {
                'source':'demo','folder':str(ROOT),'lat':49.5000,'lon':5.9670,
                'year':2025,'month':1,
            }
            st.session_state.sartrack_view = 'results'
            st.session_state.sar_nav = 'Overview'
            st.rerun()
        except (OSError, ValueError, KeyError):
            st.info(
                "The saved V2 reference is not available in this package. "
                "Use Generate 12 Monthly Maps for live U-Net V3 inference."
            )

    if generate:
        st.session_state.sartrack_active = None
        bar = st.progress(0, text='Starting real Sentinel-1 processing...')
        status = st.empty()
        def on_progress(message: str) -> None:
            status.info(message)
            if message.startswith('SAR '):
                match = re.search(r'SAR (\d+)/12', message)
                if match:
                    bar.progress(int(match.group(1)) / 24, text='Downloading 12 SAR monthly composites')
            elif message.startswith('Prediction '):
                match = re.search(r'Prediction (\d+)/12', message)
                if match:
                    bar.progress((12+int(match.group(1)))/24,
                                 text='Running twelve U-Net predictions')
            elif message.startswith('SUCCESS:'):
                bar.progress(1., text='Output validation in progress')
        try:
            # The computational pipeline and model checkpoint are unchanged.
            from sartrack_pipeline_v3 import run_pipeline
            result_folder = Path(run_pipeline(lat, lon, start_date.year,
                                              start_date.month, progress=on_progress))
            validate_complete_results('live', result_folder, start_date.year, start_date.month)
            st.session_state.sartrack_active = {
                'source':'live','folder':str(result_folder),
                'lat':float(lat),'lon':float(lon),
                'year':start_date.year,'month':start_date.month,
            }
            st.session_state.sar_nav = 'Overview'
            st.session_state.sartrack_view = 'results'
            st.rerun()
        except Exception as exc:
            st.error(f'Live generation failed: {type(exc).__name__}: {exc}')
            st.info('Check Google Earth Engine authentication, Sentinel-1 coverage and V3 model weights.')
        finally:
            bar.empty()


def chart_trends(month_dates: list[date], rows: list[dict[str,str]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for name,color in zip(CLASSES,CLASS_COLORS):
        ax.plot(range(1,13),[percent(r,name) for r in rows],
                label=name,color=color,linewidth=2,marker='o',markersize=4)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels([month.strftime('%b %y') for month in month_dates], rotation=25)
    ax.set_ylabel('Predicted area (%)')
    ax.grid(alpha=.18)
    ax.legend(ncol=5,fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def results_page() -> None:
    active = st.session_state.sartrack_active
    if not active:
        st.session_state.sartrack_view = 'setup'
        st.rerun()
        return
    source = active['source']
    folder = Path(active['folder'])
    year, month = active['year'], active['month']
    month_dates = [add_months(year,month,i) for i in range(12)]
    try:
        csv_path = validate_complete_results(source,folder,year,month)
        rows = read_statistics(csv_path)
        change_png,change_tif,stable_png,stable_tif,summary_path = stability_paths(source,folder)
        stability = read_stability(summary_path)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Cannot display stored results: {exc}')
        if st.button('Back to Study Area'):
            st.session_state.sartrack_view = 'setup'
            st.rerun()
        return

    topbar('results')
    st.markdown('<div class="sar-hero"><div class="sar-kicker">COMPLETED ANALYSIS</div>'
                '<h1>Results &amp; Temporal Analytics</h1>'
                f'<p>Target center: {active["lat"]:.4f}° N, {active["lon"]:.4f}° E'
                f' · {month_dates[0]:%B %Y} – {month_dates[-1]:%B %Y}'
                f' · {"V2 saved Esch demo" if source=="demo" else "V3 live predictions"}</p>'
                '</div>', unsafe_allow_html=True)

    with st.container():
        k1,k2,k3,k4 = st.columns(4,gap='medium')
        k1.metric('Monthly Maps','12 / 12')
        k2.metric(f'{month_dates[-1]:%b %Y} Built-up', f'{percent(rows[-1],"Built-up"):.2f}%')
        k3.metric('Built-up Change',f'{percent(rows[-1],"Built-up")-percent(rows[0],"Built-up"):+.2f} pp')
        k4.metric('Prediction Stability',f'{stability:.2f}%' if stability is not None else 'N/A')

    sidebar, content = st.columns([1.15,3.85],gap='large')
    with sidebar:
        with st.container(border=True):
            st.markdown('<div class="sar-kicker-sm">STUDY AREA</div>'
                        '<div class="sar-panel-title">Analysis Context</div>',
                        unsafe_allow_html=True)
            st.markdown(f'**Latitude:** `{active["lat"]:.4f}`')
            st.markdown(f'**Longitude:** `{active["lon"]:.4f}`')
            st.markdown(f'**Starting month:** {month_dates[0]:%B %Y}')
            st.markdown(f'**Inference:** {"V2 saved demo" if source=="demo" else "U-Net V3 live"}')
            if st.button('← Change Study Area',use_container_width=True):
                st.session_state.sartrack_view = 'setup'
                st.rerun()
        with st.container(border=True):
            st.markdown('<div class="sar-kicker-sm">EXPLORE RESULTS</div>',unsafe_allow_html=True)
            st.radio('Results sections',PAGES,key='sar_nav',label_visibility='collapsed')

    with content:
        section = st.session_state.sar_nav
        if section=='Overview':
            st.markdown('### Predicted Land-Cover Trends')
            chart_trends(month_dates,rows)
            left,right = st.columns([1.1,1],gap='large')
            with left:
                with st.container(border=True):
                    st.markdown('#### Monthly Class Distribution')
                    index = st.select_slider('Select month',options=list(range(12)),
                                             format_func=lambda i:month_dates[i].strftime('%b %Y'),
                                             value=11,key='sar_overview_month')
                    class_bars(rows[index])
            with right:
                with st.container(border=True):
                    st.markdown('#### First-to-Last Change')
                    for name,color in zip(CLASSES,CLASS_COLORS):
                        delta = percent(rows[-1],name) - percent(rows[0],name)
                        st.markdown(f'**{name}:** {delta:+.2f} percentage points')
                    st.caption('Differences of predicted area proportions, not validated ground transitions.')
            st.markdown('#### Monthly Statistics')
            month_table(month_dates,rows,'distribution')

        elif section=='Monthly Maps':
            st.markdown('### 12 Monthly LULC Predictions')
            legend()
            index = st.select_slider('Select month',options=list(range(12)),
                                     format_func=lambda i:month_dates[i].strftime('%b %Y'),
                                     value=5,key='sar_map_month')
            selected_png,_ = result_paths(source,folder,month_dates[index])
            imagecol,statscol = st.columns([2.5,1.2],gap='large')
            with imagecol:
                with st.container(border=True):
                    st.markdown(f'**{month_dates[index]:%B %Y} · Actual saved LULC prediction**')
                    show_image(selected_png)
            with statscol:
                with st.container(border=True):
                    st.markdown('**Selected Month Composition**')
                    class_bars(rows[index])
            st.markdown('#### All 12 Months')
            for start in range(0,12,4):
                cols = st.columns(4,gap='small')
                for offset,col in enumerate(cols):
                    i = start+offset
                    with col:
                        with st.container(border=True):
                            st.markdown(f'**{month_dates[i]:%b %Y}**')
                            png,_=result_paths(source,folder,month_dates[i])
                            show_image(png)

        elif section=='Area Changes':
            st.markdown('### Predicted Area Changes')
            st.caption('Values are differences in percentage points (pp), not relative percent changes.')
            st.markdown('#### Month-to-month changes')
            month_table(month_dates,rows,'monthly')
            st.markdown('#### Start-to-end changes')
            summary_rows = []
            for name in CLASSES:
                before, after = percent(rows[0], name), percent(rows[-1], name)
                summary_rows.append(
                    f'<tr><th>{escape(name)}</th><td>{before:.2f}%</td>'
                    f'<td>{after:.2f}%</td><td>{after-before:+.2f}</td></tr>'
                )
            st.markdown('<div class="sar-table-wrap"><table class="sar-table">'
                        '<thead><tr><th>Class</th><th>Start (%)</th><th>End (%)</th>'
                        '<th>Change (pp)</th></tr></thead><tbody>'
                        + ''.join(summary_rows) + '</tbody></table></div>',
                        unsafe_allow_html=True)
            st.warning('Some apparent monthly change may be driven by SAR seasonality or model errors.')

        elif section=='Stability':
            st.markdown('### Temporal Prediction Stability')
            if stability is not None:
                st.metric('Pixels unchanged across twelve predictions',f'{stability:.2f}%')
            img_left,img_right=st.columns(2,gap='large')
            with img_left:
                with st.container(border=True):
                    show_image(change_png,'Consecutive-month class change count')
            with img_right:
                with st.container(border=True):
                    show_image(stable_png,'Stable predicted classes')
            st.warning('Prediction stability is not proof of unchanged real-world land cover.')

        elif section=='Downloads':
            st.markdown('### Download Actual Run Outputs')
            st.caption('Downloads are files produced by the selected run, not mock exports.')
            download_file(csv_path,'Monthly Statistics · CSV','text/csv')
            index=st.selectbox('Choose month',options=list(range(12)),index=5,
                               format_func=lambda i:month_dates[i].strftime('%B %Y'),
                               key='sar_dl_month')
            png,tif = result_paths(source,folder,month_dates[index])
            p1,p2=st.columns(2)
            with p1:
                download_file(tif,'Monthly LULC · GeoTIFF','image/tiff')
            with p2:
                download_file(png,'Monthly LULC · PNG','image/png')
            p3,p4=st.columns(2)
            with p3:
                download_file(change_tif,'Change Count · GeoTIFF','image/tiff')
            with p4:
                download_file(stable_tif,'Stable Class · GeoTIFF','image/tiff')

        elif section=='Model Details':
            st.markdown('### Model and Method')
            st.markdown("""**Architecture:** Lightweight U-Net · 488,629 parameters  
**Input channels:** Sentinel-1 VV and VH (dual polarization)  
**Predicted classes:** Water, Trees, Grass, Crops, Built-up  
**Training regions:** Dudelange and Bastogne  
**Validation region:** Thionville  
**Independent test region:** Esch/Belval  
**Pseudo-label source:** Dynamic World""")
            a,b,c=st.columns(3)
            a.metric('Test Overall Accuracy','79.90%' if source=='demo' else '83.71%')
            b.metric('Test mean IoU','38.72%' if source=='demo' else '46.37%')
            c.metric('Test Macro F1','49.45%' if source=='demo' else '59.33%')
            st.info('Test scores are against Dynamic World pseudo-labels, not surveyed ground truth. '
                    'They do not give accuracy for the newly selected geographic area.')
            st.markdown('#### Known limitations')
            st.markdown("""- Water and Grass have low IoU on the independent test.
- SAR seasonality and prediction noise can inflate apparent class changes.
- Geographic domain shift can reduce accuracy outside the training regions.
- Live runs need matching Sentinel-1 orbit coverage for each of 12 months.""")

    st.markdown('<div class="sar-footer">SARTrack · Research prototype · Predictions are not '
                'verified ground-truth change maps</div>',unsafe_allow_html=True)


# Initialize session state without silently displaying results for altered inputs.
if 'sartrack_view' not in st.session_state:
    st.session_state.sartrack_view = 'setup'
if 'sartrack_active' not in st.session_state:
    st.session_state.sartrack_active = None
if 'sar_lat' not in st.session_state:
    st.session_state.sar_lat = 49.5
if 'sar_lon' not in st.session_state:
    st.session_state.sar_lon = 5.967
# Reset previously stored incorrect coordinates once per session.
if "sar_coordinate_defaults_fixed" not in st.session_state:
    st.session_state.sar_lat = 49.5000
    st.session_state.sar_lon = 5.9670
    st.session_state.sar_coordinate_defaults_fixed = True
if 'sar_start_date' not in st.session_state:
    st.session_state.sar_start_date = date(2025,1,1)
if 'sar_nav' not in st.session_state:
    st.session_state.sar_nav = 'Overview'

if st.session_state.sartrack_view == 'results':
    results_page()
else:
    setup_page()
