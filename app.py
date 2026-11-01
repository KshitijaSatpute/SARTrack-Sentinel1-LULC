"""SARTrack Streamlit dashboard — live Sentinel-1 U-Net predictions.

Requires the tested sartrack_pipeline.py next to this file.
No pandas/scikit-learn imports (Windows Application Control compatibility).
"""
from pathlib import Path
from datetime import date
import calendar
import csv
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
DEMO_MONTHLY = ROOT / "outputs" / "monthly"
DEMO_STABILITY = ROOT / "outputs" / "stability"
CLASSES = ["Water", "Trees", "Grass", "Crops", "Built-up"]
COLORS = ["#38BDF8", "#22A06B", "#A3D977", "#E9C65B", "#DF6864"]


def add_months(year, month, offset):
    pos = year * 12 + (month - 1) + offset
    return date(pos // 12, pos % 12 + 1, 1)


def result_paths(source, folder, month_date):
    """Return matching PNG/TIFF for original demo or live pipeline output."""
    if source == "demo":
        prefix = f"Esch_{month_date:%B%Y}_LULC"
        return DEMO_MONTHLY / f"{prefix}.png", DEMO_MONTHLY / f"{prefix}.tif"
    prefix = f"LULC_{month_date:%Y_%m}"
    return folder / f"{prefix}.png", folder / f"{prefix}.tif"


def stability_paths(source, folder):
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


def read_statistics(path):
    if not path.is_file():
        raise FileNotFoundError(f"Statistics CSV missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12:
        raise ValueError(f"Expected 12 monthly rows, found {len(rows)} in {path.name}")
    for row in rows:
        for name in CLASSES:
            float(row[f"{name}_pct"])
    return rows


def read_stability(path):
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8")
    match = re.search(r"Overall stability:\s*([\d.]+)%", content)
    return float(match.group(1)) if match else None


def percent(record, class_name):
    return float(record[f"{class_name}_pct"])


def show_image(path, caption=None):
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Missing image: {path.name}")


def download_file(path, label, mime):
    if path.is_file():
        st.download_button(
            label, data=path.read_bytes(), file_name=path.name,
            mime=mime, use_container_width=True,
        )
    else:
        st.warning(f"Missing downloadable file: {path.name}")


def validate_complete_results(source, folder, start_year, start_month):
    if source == "demo":
        csv_path = DEMO_MONTHLY / "monthly_land_cover_statistics.csv"
    else:
        csv_path = folder / "monthly_land_cover_statistics.csv"
    read_statistics(csv_path)
    required = [csv_path]
    for index in range(12):
        png, tif = result_paths(source, folder, add_months(start_year, start_month, index))
        required.extend([png, tif])
    required.extend(stability_paths(source, folder))
    missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Incomplete results: " + ", ".join(missing[:6]))
    return csv_path


st.set_page_config(
    page_title="SARTrack | Land Cover Intelligence",
    page_icon="🛰️", layout="wide", initial_sidebar_state="expanded",
)

# Keep the already-verified dashboard styling; redesign is outside this integration.
st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 1.6rem;}
.hero {
  background: linear-gradient(115deg, #0F172A, #164E63);
  padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 22px;
}
.hero h1 {color: white; margin: 0; font-size: 2.5rem;}
.hero p {color: #CBD5E1; margin-top: 10px;}
div[data-testid="stMetric"] {
  padding: 16px; border: 1px solid rgba(148,163,184,0.25);
  border-radius: 12px; background: rgba(100,116,139,0.07);
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🛰️ SARTrack")
    st.caption("Land Cover Intelligence")
    st.divider()
    st.subheader("Study Area")
    latitude = st.number_input(
        "Latitude", min_value=-79.9999, max_value=83.9999,
        value=49.5000, step=0.0001, format="%.4f",
    )
    longitude = st.number_input(
        "Longitude", min_value=-180.0, max_value=180.0,
        value=5.9670, step=0.0001, format="%.4f",
    )
    # A full set of 12 completed calendar months must be available.
    today = date.today()
    latest_start = add_months(today.year, today.month, -12)
    starting_date = st.date_input(
        "Starting Month", value=date(2025, 1, 1),
        min_value=date(2015, 1, 1), max_value=latest_start,
    )
    st.caption("Day is ignored; analysis begins on the 1st of the chosen month.")
    generate = st.button(
        "Generate 12 Monthly Maps", type="primary", use_container_width=True,
    )
    is_demo_input = (
        abs(latitude - 49.5000) < 0.00005
        and abs(longitude - 5.9670) < 0.00005
        and starting_date.year == 2025 and starting_date.month == 1
    )
    saved_demo = st.button(
        "Open saved Esch demo (instant)",
        disabled=not is_demo_input, use_container_width=True,
    )
    st.divider()
    st.markdown("**Model:** U-Net v2")
    st.caption("Input: Sentinel-1 VV / VH")
    st.caption("Output: 5 predicted land-cover classes")
    st.caption("Data: Google Earth Engine")
    st.divider()
    st.caption("Master's Deep Learning Project")

st.markdown("""
<div class="hero">
  <div style="font-size:12px;letter-spacing:2px;color:#67E8F9;">
    SATELLITE INTELLIGENCE PLATFORM
  </div>
  <h1>🛰️ SARTrack</h1>
  <p>Sentinel-1 Land Cover Classification &amp; Temporal Stability Monitoring</p>
</div>
""", unsafe_allow_html=True)

selection = (
    round(latitude, 4), round(longitude, 4),
    starting_date.year, starting_date.month,
)

if "sartrack_active" not in st.session_state:
    st.session_state.sartrack_active = None

if saved_demo:
    try:
        validate_complete_results("demo", ROOT, 2025, 1)
        st.session_state.sartrack_active = {"selection": selection, "source": "demo", "folder": str(ROOT)}
    except (OSError, ValueError, KeyError) as exc:
        st.session_state.sartrack_active = None
        st.error(f"Saved demo could not be loaded: {exc}")

if generate:
    # Prevent an earlier area being silently displayed when a new run fails.
    st.session_state.sartrack_active = None
    progress_bar = st.progress(0, text="Starting live Sentinel-1 analysis...")
    message_box = st.empty()

    def on_progress(message):
        message_box.info(message)
        if message.startswith("SAR "):
            match = re.search(r"SAR (\d+)/12", message)
            if match:
                progress_bar.progress(int(match.group(1)) / 24, text="Downloading 12 SAR months")
        elif message.startswith("Prediction "):
            match = re.search(r"Prediction (\d+)/12", message)
            if match:
                progress_bar.progress((12 + int(match.group(1))) / 24, text="Running U-Net predictions")
        elif message.startswith("SUCCESS:"):
            progress_bar.progress(1.0, text="All 12 months and stability maps generated")

    try:
        # Import only when running to keep the rest of Streamlit responsive.
        from sartrack_pipeline import run_pipeline

        live_folder = Path(run_pipeline(
            latitude, longitude, starting_date.year,
            starting_date.month, progress=on_progress,
        ))
        validate_complete_results(
            "live", live_folder, starting_date.year, starting_date.month,
        )
        st.session_state.sartrack_active = {
            "selection": selection, "source": "live", "folder": str(live_folder),
        }
        message_box.success("Live processing complete: 12 LULC maps, CSV and stability outputs.")
    except Exception as exc:
        st.error(f"Live generation failed: {type(exc).__name__}: {exc}")
        st.info("Check Google Earth Engine authentication, SAR coverage and the model weights. No previous demo results were changed.")
    finally:
        progress_bar.empty()

active = st.session_state.sartrack_active
if active is None or active["selection"] != selection:
    st.info(
        "Select latitude, longitude and starting month, then click "
        "**Generate 12 Monthly Maps**. For the original Esch/Belval example, "
        "you can instead click **Open saved Esch demo (instant)**."
    )
    st.stop()

source = active["source"]
folder = Path(active["folder"])
start_year, start_month = selection[2], selection[3]
months = [add_months(start_year, start_month, i) for i in range(12)]
try:
    csv_path = validate_complete_results(source, folder, start_year, start_month)
    records = read_statistics(csv_path)
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Result files could not be loaded: {exc}")
    st.stop()

change_png, change_tif, stable_png, stable_tif, summary_txt = stability_paths(source, folder)
stability = read_stability(summary_txt)
first, last = records[0], records[-1]
period = f"{months[0]:%B %Y} – {months[-1]:%B %Y}"
region = "Esch/Belval (saved demo)" if source == "demo" else f"{latitude:.4f}°, {longitude:.4f}° (live)"
st.success(f"Loaded: {region} | {period} | 12 monthly maps")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Monthly Maps", "12 / 12")
k2.metric(f"{months[-1]:%b %Y} Built-up", f"{percent(last, 'Built-up'):.2f}%")
built_change = percent(last, "Built-up") - percent(first, "Built-up")
k3.metric("Start–End Built-up Change", f"{built_change:+.2f} pp")
k4.metric("Prediction Stability", f"{stability:.2f}%" if stability is not None else "N/A")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🗺️ Monthly Maps", "🔁 Stability", "📥 Downloads", "🧠 Model Details",
])

with tab1:
    st.subheader("Monthly Land-Cover Trends")
    fig, ax = plt.subplots(figsize=(11, 4))
    for j, name in enumerate(CLASSES):
        ax.plot(
            range(1, 13), [percent(record, name) for record in records],
            marker="o", linewidth=2, label=name, color=COLORS[j],
        )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([month.strftime("%b %y") for month in months], rotation=25)
    ax.set_ylabel("Predicted Area (%)")
    ax.set_xlabel("Month")
    ax.grid(alpha=0.2)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.subheader(f"{months[-1]:%B %Y} Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(CLASSES, [percent(last, c) for c in CLASSES], color=COLORS)
        ax.set_ylabel("Area (%)")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Start → End Changes")
        changes = [percent(last, c) - percent(first, c) for c in CLASSES]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(CLASSES, changes, color=COLORS)
        ax.axvline(0, color="gray", linewidth=1)
        ax.set_xlabel("Percentage-point change")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Monthly Statistics")
    table = "| Month | " + " | ".join(CLASSES) + " |\n"
    table += "|---|" + "|".join(["---:"] * len(CLASSES)) + "|\n"
    for month, record in zip(months, records):
        values = [f"{percent(record, c):.2f}%" for c in CLASSES]
        table += f"| {month:%B %Y} | " + " | ".join(values) + " |\n"
    st.markdown(table)

    with st.expander("Month-to-month changes (percentage points)"):
        changes_table = "| Month | " + " | ".join(CLASSES) + " |\n"
        changes_table += "|---|" + "|".join(["---:"] * len(CLASSES)) + "|\n"
        for i, (month, record) in enumerate(zip(months, records)):
            if i == 0:
                values = ["—"] * len(CLASSES)
            else:
                values = [f"{percent(record, c) - percent(records[i-1], c):+.2f}" for c in CLASSES]
            changes_table += f"| {month:%B %Y} | " + " | ".join(values) + " |\n"
        st.markdown(changes_table)

with tab2:
    st.subheader("Monthly Land-Cover Maps")
    month_labels = [month.strftime("%B %Y") for month in months]
    selected_index = st.select_slider(
        "Select Month", options=list(range(12)),
        value=5, format_func=lambda i: month_labels[i],
    )
    selected_date = months[selected_index]
    selected_png, _ = result_paths(source, folder, selected_date)
    left, right = st.columns([2, 1])
    with left:
        show_image(selected_png, f"{selected_date:%B %Y}")
    with right:
        st.markdown(f"### {selected_date:%B %Y}")
        for name in CLASSES:
            st.write(f"**{name}:** {percent(records[selected_index], name):.2f}%")
    st.divider()
    st.subheader("All 12 Monthly Maps")
    for start in range(0, 12, 3):
        columns = st.columns(3)
        for offset, column in enumerate(columns):
            month = months[start + offset]
            image_path, _ = result_paths(source, folder, month)
            with column:
                st.markdown(f"**{month:%B %Y}**")
                show_image(image_path)

with tab3:
    st.subheader("Land-Cover Stability Analysis")
    if stability is not None:
        st.metric("Pixels Unchanged Across 12 Predictions", f"{stability:.2f}%")
    col1, col2 = st.columns(2)
    with col1:
        show_image(change_png, "Month-to-month classification change count")
    with col2:
        show_image(stable_png, "Stable predicted land-cover classes")
    st.warning(
        "Stability reflects changes in model predictions, not verified "
        "real-world land-cover transitions."
    )

with tab4:
    st.subheader("Download SARTrack Results")
    download_file(csv_path, "Download Monthly Statistics CSV", "text/csv")
    download_month = st.selectbox(
        "Select Month", options=list(range(12)), index=5,
        format_func=lambda i: month_labels[i],
    )
    png_path, tif_path = result_paths(source, folder, months[download_month])
    download_file(tif_path, "Download LULC GeoTIFF", "image/tiff")
    download_file(png_path, "Download LULC PNG", "image/png")
    download_file(change_tif, "Download Change Count GeoTIFF", "image/tiff")
    download_file(stable_tif, "Download Stable Class GeoTIFF", "image/tiff")

with tab5:
    st.subheader("U-Net v2 — Model Information")
    st.markdown("""
**Architecture:** Lightweight U-Net  
**Parameters:** 488,629  
**Input:** Sentinel-1 VV and VH  
**Output:** Five land-cover classes  

**Training regions:** Dudelange, Bastogne  
**Validation region:** Thionville  
**Independent test:** Esch/Belval  
**Reference labels:** Dynamic World
""")
    a, b, c = st.columns(3)
    a.metric("Test Accuracy", "79.90%")
    b.metric("Test mIoU", "38.72%")
    c.metric("Test Macro F1", "49.45%")
    st.info("Metrics are against Dynamic World pseudo-labels, not manually verified ground truth. They apply to the independent Esch/Belval test, not necessarily to the current selected region.")
    st.subheader("Limitations")
    st.markdown("""
- Water and Crops segmentation need improvement.
- Monthly predicted changes may reflect model uncertainty or seasonal SAR variation.
- Results outside the training region may be affected by domain shift.
- A 12-month run needs Sentinel-1 coverage in every selected month on the same orbit.
- Stability means agreement across model predictions, not verified land-use transitions.
""")

st.divider()
st.caption("SARTrack | Sentinel-1 SAR Deep Learning Prototype | Research and educational use")