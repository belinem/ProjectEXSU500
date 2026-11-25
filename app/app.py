import json
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from skimage.measure import marching_cubes

# Allow importing your team's existing helper functions from src/utils.py
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from utils import RESULTS, SPLITS, la_prior, load_case, normalize, otsu_segment

# Page setup
st.set_page_config(page_title="EXSU 500 - LA Segmentation", layout="wide")

# Mandatory course disclaimer (must always be visible)
st.warning(
    "⚠️ **DISCLAIMER:** This application is a course prototype for research and "
    "educational purposes only. It is NOT a medical device and is NOT for clinical use."
)

st.title("🫀 Left Atrium 3D MRI Segmentation")

# --- SIDEBAR: INPUT & MODEL ---
st.sidebar.header("1. Input MRI Scan")
input_mode = st.sidebar.radio("Choose input source:", ["Sample scan", "Upload .nii.gz"])

sample_dir = ROOT_DIR / "data" / "Task02_Heart" / "imagesTs"
sample_files = sorted(sample_dir.glob("*.nii.gz")) if sample_dir.exists() else []

uploaded_file = None
selected_sample = None

if input_mode == "Sample scan":
    if sample_files:
        selected_sample = st.sidebar.selectbox(
            "Pick a sample cardiac MRI:",
            sample_files,
            format_func=lambda p: p.name,
        )
    else:
        st.sidebar.info("No sample scans found in `data/Task02_Heart/imagesTs`.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a 3D NIfTI file (.nii or .nii.gz)")

st.sidebar.header("2. Model")
model_choice = st.sidebar.radio(
    "Choose a model:",
    ["Otsu Baseline", "Random Forest", "3D U-Net (MONAI)"]
)

run_btn = st.sidebar.button("▶️ Run Segmentation", type="primary", use_container_width=True)
show_overlay = st.sidebar.checkbox("Show segmentation mask overlay", value=True)


# --- CACHED HELPERS ---
@st.cache_data
def load_nifti_from_path(path_str: str):
    img = nib.load(path_str)
    data = img.get_fdata(dtype=np.float32)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return data, spacing


def load_uploaded_nifti(file_obj):
    suffix = ".nii.gz" if file_obj.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_obj.getvalue())
        tmp_path = tmp.name
    try:
        img = nib.load(tmp_path)
        data = img.get_fdata(dtype=np.float32)
        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got {data.ndim}D.")
        return data, spacing
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@st.cache_resource
def get_otsu_setup():
    # Compute the LA spatial prior once from the development set
    labels = [load_case(c)[1] for c in SPLITS["dev"]]
    centroid, box = la_prior(labels)
    cfg_path = RESULTS / "chosen_configs.json"
    cfg = json.loads(cfg_path.read_text())["otsu"] if cfg_path.exists() else {"n_classes": 3, "radius": 9}
    return centroid, box, cfg["n_classes"], cfg["radius"]


def make_3d_mesh(mask: np.ndarray, spacing: tuple):
    # step_size=2 keeps the 3D mesh fast and smooth in the browser
    verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, step_size=2, spacing=spacing)
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="crimson",
                opacity=0.85,
                flatshading=False,
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=450,
    )
    return fig


# --- LOAD THE CHOSEN SCAN ---
volume, spacing, scan_name = None, None, None

if input_mode == "Sample scan" and selected_sample is not None:
    volume, spacing = load_nifti_from_path(str(selected_sample))
    scan_name = selected_sample.name
elif input_mode == "Upload .nii.gz" and uploaded_file is not None:
    try:
        volume, spacing = load_uploaded_nifti(uploaded_file)
        scan_name = uploaded_file.name
    except Exception as e:
        st.error(f"Invalid file: could not load `{uploaded_file.name}` as a 3D NIfTI MRI. ({e})")


# --- RUN MODEL WHEN BUTTON IS CLICKED ---
if volume is not None and run_btn:
    if model_choice == "Otsu Baseline":
        with st.spinner("Running Otsu Baseline segmentation..."):
            centroid, box, n_classes, radius = get_otsu_setup()
            pred_mask = otsu_segment(normalize(volume), centroid, box, n_classes, radius).astype(bool)
            voxel_ml = float(np.prod(spacing)) / 1000.0
            vol_ml = float(pred_mask.sum()) * voxel_ml
            st.session_state["result"] = {
                "scan_name": scan_name,
                "model": model_choice,
                "mask": pred_mask,
                "vol_ml": vol_ml,
                "confidence_note": f"Deterministic Otsu threshold ({n_classes} classes, opening radius {radius}). No calibrated probability.",
            }
    else:
        st.info(f"**{model_choice}** will be plugged in as soon as your teammate saves the trained weights.")


# --- DISPLAY RESULTS ---
if volume is not None:
    st.subheader(f"Loaded Scan: `{scan_name}`")
    st.caption(f"Shape: {volume.shape[0]} × {volume.shape[1]} × {volume.shape[2]} voxels | Spacing: {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm")

    # Check if we have a segmentation result for THIS scan
    res = st.session_state.get("result")
    mask = res["mask"] if (res and res["scan_name"] == scan_name) else None

    if mask is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("Model Used", res["model"])
        m2.metric("Predicted LA Volume", f"{res['vol_ml']:.1f} mL")
        risk_label = "Above 153 mL threshold" if res["vol_ml"] > 153 else "Below 153 mL threshold"
        m3.metric("Clinical Reference (AF Recurrence)", risk_label)
        st.caption(f"**Confidence / Method:** {res['confidence_note']}")

    # Pick smart default slices (center on the LA if we have a mask)
    default_z = int(np.argmax(mask.sum(axis=(0, 1)))) if (mask is not None and mask.sum() > 0) else volume.shape[2] // 2
    default_x = int(np.argmax(mask.sum(axis=(1, 2)))) if (mask is not None and mask.sum() > 0) else volume.shape[0] // 2
    default_y = int(np.argmax(mask.sum(axis=(0, 2)))) if (mask is not None and mask.sum() > 0) else volume.shape[1] // 2

    # Tabs: 2D Slices vs 3D Reconstruction
    tab_2d, tab_3d = st.tabs(["🩻 2D Orthogonal Views", "🧊 Interactive 3D Reconstruction"])

    with tab_2d:
        col1, col2, col3 = st.columns(3)

        with col1:
            z = st.slider("Axial slice (Z)", 0, volume.shape[2] - 1, default_z)
            fig, ax = plt.subplots()
            ax.imshow(np.rot90(volume[:, :, z]), cmap="gray")
            if mask is not None and show_overlay:
                m_slice = mask[:, :, z]
                ax.imshow(np.rot90(np.ma.masked_where(~m_slice, m_slice)), cmap="autumn", alpha=0.45)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            x = st.slider("Sagittal slice (X)", 0, volume.shape[0] - 1, default_x)
            fig, ax = plt.subplots()
            ax.imshow(np.rot90(volume[x, :, :]), cmap="gray")
            if mask is not None and show_overlay:
                m_slice = mask[x, :, :]
                ax.imshow(np.rot90(np.ma.masked_where(~m_slice, m_slice)), cmap="autumn", alpha=0.45)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        with col3:
            y = st.slider("Coronal slice (Y)", 0, volume.shape[1] - 1, default_y)
            fig, ax = plt.subplots()
            ax.imshow(np.rot90(volume[:, y, :]), cmap="gray")
            if mask is not None and show_overlay:
                m_slice = mask[:, y, :]
                ax.imshow(np.rot90(np.ma.masked_where(~m_slice, m_slice)), cmap="autumn", alpha=0.45)
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

    with tab_3d:
        if mask is not None and mask.sum() > 0:
            st.write("Drag to rotate, scroll to zoom:")
            with st.spinner("Building 3D Left Atrium mesh..."):
                st.plotly_chart(make_3d_mesh(mask, spacing), use_container_width=True)
        else:
            st.info("Click **▶️ Run Segmentation** in the sidebar to generate the 3D Left Atrium mesh.")
else:
    st.info("Select a sample scan or upload a `.nii.gz` file from the sidebar to begin.")