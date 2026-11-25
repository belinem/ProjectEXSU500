import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from skimage.measure import marching_cubes

#to import the project
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from utils import keep_largest_component, normalize

MODEL_PATH = ROOT_DIR / "models" / "unet_seed0.pt"
SAMPLE_DIR = ROOT_DIR / "data" / "Task02_Heart" / "imagesTs"

# page configuration
st.set_page_config(
    page_title="EXSU 500 - 3D Left Atrium Segmentation",
    page_icon="🫀",
    layout="wide",
)

# warning about the scope of the project + title
st.warning(
    "This application is a course prototype for research and "
    "educational purposes only. It is not appropriate for clinical use."
)

st.title("Left Atrium 3D MRI Segmentation (3D U-Net)")
st.write("Upload a non-contrast 3D cardiac MRI (`.nii.gz`) or select a sample scan to segment the Left Atrium and estimate its volume.")

# controls (lateral bar)
st.sidebar.header("1. Input Cardiac MRI")
input_mode = st.sidebar.radio("Source:", ["Sample scan", "Upload .nii.gz"])

sample_files = sorted(SAMPLE_DIR.glob("*.nii.gz")) if SAMPLE_DIR.exists() else []
uploaded_file, selected_sample = None, None

if input_mode == "Sample scan":
    if sample_files:
        selected_sample = st.sidebar.selectbox(
            "Choose a sample scan:", sample_files, format_func=lambda p: p.name
        )
    else:
        st.sidebar.info("No sample scans found in `data/Task02_Heart/imagesTs`.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a 3D NIfTI file (.nii or .nii.gz)")

st.sidebar.header("2. Segmentation & Display")
run_btn = st.sidebar.button("▶️ Run 3D U-Net", type="primary", use_container_width=True)
overlay_mode = st.sidebar.radio(
    "2D Overlay:",
    ["Binary Mask", "Probability Heatmap", "Raw MRI"]
)

# store the output of this function in memory so the mri file is not reread every time
@st.cache_data
def load_nifti_from_path(path_str: str):
    img = nib.load(path_str)
    data = img.get_fdata(dtype=np.float32)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return data, spacing


def load_uploaded_nifti(file_obj):
    suffix = ".nii.gz" if file_obj.name.endswith(".gz") else ".nii"
    # We create a temporary file, write the uploaded bytes into it, and keep its path
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

# st.cache_resource is used for heavy objects like ML models
# loads the 3D U-Net weights into RAM only ONCE when the app starts instead of on every click.
@st.cache_resource
def load_unet_model():
    # 3D convolutions, 1 input channel (grayscale MRI), 2 output channels (background + Left Atrium),
    # 5 encoder/decoder levels with 16 to 256 feature maps, downsampling by 2 at each level.
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    # load the trained weights onto the CPU 
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    # Copy the saved weights into the model architecture
    model.load_state_dict(state)
    # Switch the model to evaluation mode 
    model.eval()
    return model


def make_3d_mesh(mask: np.ndarray, spacing: tuple):
    # Convert the 3D binary mask (0s and 1s) into a 3D surface made of triangles (vertices + faces)
    # level=0.5 traces the boundary between 0 (background) and 1 (Left Atrium)
    # step_size=2 skips every second voxel to reduce triangle count 
    # spacing=spacing scales the 3D shape to true physical proportions in millimeters
    verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, step_size=2, spacing=spacing)

    # Create an interactive Plotly 3D mesh figure from the extracted vertices and triangle indices
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],  # 3D coordinates of each vertex
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],  # Which 3 vertices form each triangle
                color="#E63946",      # Medical crimson red color
                opacity=0.90,         # Slightly translucent
                flatshading=False,    # Smooth shading across triangles instead of faceted/blocky look
                lighting=dict(ambient=0.4, diffuse=0.9, specular=0.4, roughness=0.3),  # 3D surface lighting
            )
        ]
    )

    # Configure axis labels, proportions, and sizing
    fig.update_layout(
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",  # Keeps true anatomical proportions (prevents stretching)
        ),
        margin=dict(l=0, r=0, t=10, b=0),  # Remove extra whitespace around the 3D box
        height=480,                        # Fixed height in pixels
    )
    return fig


def plot_slice(img_2d, mask_2d, prob_2d, mode):
    # Create a Matplotlib figure with a dark background matching Streamlit's dark theme
    fig, ax = plt.subplots(facecolor="#0E1117")
    # Display the 2D MRI slice in grayscale; np.rot90 rotates it 90° so anatomy is upright
    ax.imshow(np.rot90(img_2d), cmap="gray")

    # Option 1: Overlay the binary segmentation mask in semi-transparent red/orange
    if mode == "Binary Mask" and mask_2d is not None:
        # np.ma.masked_where hides all False (background) pixels so only the Left Atrium is colored
        ax.imshow(np.rot90(np.ma.masked_where(~mask_2d, mask_2d)), cmap="autumn", alpha=0.45)

    # Option 2: Overlay the continuous probability map (interpretability heatmap)
    elif mode == "Probability Heatmap" and prob_2d is not None:
        # Hide very low probabilities (< 5%) so the background MRI stays clean and visible
        # vmin=0, vmax=1 fixes the color scale consistently from 0% to 100% confidence
        ax.imshow(np.rot90(np.ma.masked_where(prob_2d < 0.05, prob_2d)), cmap="inferno", vmin=0, vmax=1, alpha=0.6)

    # Hide pixel coordinate ticks/axes for a clean radiology viewer look
    ax.axis("off")
    return fig


# loading the selected mri scan:
volume, spacing, scan_name = None, None, None

if input_mode == "Sample scan" and selected_sample is not None:
    # Load the chosen built-in sample scan from disk (cached)
    volume, spacing = load_nifti_from_path(str(selected_sample))
    scan_name = selected_sample.name
elif input_mode == "Upload .nii.gz" and uploaded_file is not None:
    # Wrap upload loading in try/except so invalid files (e.g., PDFs, PNGs) show a clean error
    # message instead of crashing the entire web application
    try:
        volume, spacing = load_uploaded_nifti(uploaded_file)
        scan_name = uploaded_file.name
    except Exception as e:
        st.error(f"Invalid file: could not load `{uploaded_file.name}` as a 3D NIfTI MRI. ({e})")


# running 3D UNET interface
# Only run the neural network when a valid scan is loaded AND the user clicks "Run 3D U-Net"
if volume is not None and run_btn:
    # Verify that the trained weights file actually exists before attempting to load it
    if not MODEL_PATH.exists():
        st.error(f"Model weights not found at `{MODEL_PATH}`. Place `unet_seed0.pt` in the `models/` folder.")
    else:
        # Show an animated loading spinner while CPU inference runs (~20–40s)
        with st.spinner("Running 3D U-Net inference on CPU (~20–40 seconds)..."):
            # Retrieve the cached 3D U-Net model
            model = load_unet_model()
            # Normalize MRI intensities using the exact same preprocessing as during training
            norm_vol = normalize(volume)
            # Convert numpy array (X, Y, Z) to a 5D PyTorch tensor (Batch=1, Channel=1, X, Y, Z)
            x_tensor = torch.from_numpy(norm_vol).unsqueeze(0).unsqueeze(0).float()

            # Disable gradient tracking—saves memory and speeds up inference significantly
            with torch.no_grad():
                # Predict the full 3D volume using overlapping 96x96x96 patches (sliding window)
                # so the full MRI fits in RAM without resizing/losing resolution
                logits = sliding_window_inference(
                    x_tensor, roi_size=(96, 96, 96), sw_batch_size=1, predictor=model, overlap=0.25
                )
                # convert raw model scores (logits) into probabilities [0, 1] using softmax,
                # then select batch 0, channel 1 (the Left Atrium class) and convert back to numpy
                prob_map = torch.softmax(logits, dim=1)[0, 1].numpy()

            # threshold probabilities at 0.5 to get a binary mask, then keep only the largest
            # connected 3D component to remove small stray false-positive noise blobs
            pred_mask = keep_largest_component(prob_map >= 0.5).astype(bool)

            # calculate the physical volume of one voxel in milliliters (1 mL = 1000 mm^3)
            voxel_ml = float(np.prod(spacing)) / 1000.0
            # multiply number of segmented Left Atrium voxels by the single-voxel volume
            vol_ml = float(pred_mask.sum()) * voxel_ml
            # compute average predicted probability inside the segmented mask as a confidence score
            mean_conf = float(prob_map[pred_mask].mean()) if pred_mask.any() else 0.0

            # Save all outputs in st.session_state so they persist when the user moves sliders
            # or switches tabs (without re-running the U-Net)
            st.session_state["result"] = {
                "scan_name": scan_name,
                "mask": pred_mask,
                "prob": prob_map,
                "vol_ml": vol_ml,
                "confidence": mean_conf,
            }


# displaying metrics, 2D slices and 3D reconstruction 
if volume is not None:
    # Show scan filename and technical metadata (dimensions in voxels + physical resolution in mm)
    st.subheader(f"Loaded Scan: `{scan_name}`")
    st.caption(f"Shape: {volume.shape[0]} × {volume.shape[1]} × {volume.shape[2]} voxels | Spacing: {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm")

    # Retrieve saved segmentation results ONLY if they belong to the currently loaded scan
    res = st.session_state.get("result")
    has_res = res is not None and res["scan_name"] == scan_name
    mask = res["mask"] if has_res else None
    prob = res["prob"] if has_res else None

    # Display the 3 top-level KPI metric cards once segmentation has been run
    if has_res:
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted LA Volume", f"{res['vol_ml']:.1f} mL")
        m2.metric("Mean Model Confidence", f"{res['confidence'] * 100:.1f}%")
        # Compare predicted volume against the clinical AF ablation recurrence threshold (153 mL)
        risk_label = "Above 153 mL (Higher AF risk)" if res["vol_ml"] > 153 else "Below 153 mL"
        m3.metric("Clinical Reference", risk_label)

    # Smart slider defaults: if a mask exists, automatically jump to the slice with the largest
    # Left Atrium cross-section along each axis; otherwise, default to the middle of the scan
    default_z = int(np.argmax(mask.sum(axis=(0, 1)))) if (mask is not None and mask.sum() > 0) else volume.shape[2] // 2
    default_x = int(np.argmax(mask.sum(axis=(1, 2)))) if (mask is not None and mask.sum() > 0) else volume.shape[0] // 2
    default_y = int(np.argmax(mask.sum(axis=(0, 2)))) if (mask is not None and mask.sum() > 0) else volume.shape[1] // 2

    # Organize the viewer into two clean tabs: 2D orthogonal slices vs interactive 3D mesh
    tab_2d, tab_3d = st.tabs(["🩻 2D Orthogonal Views", "🧊 Interactive 3D Reconstruction"])

    with tab_2d:
        # Place Axial, Sagittal, and Coronal views side-by-side in 3 equal columns
        col1, col2, col3 = st.columns(3)

        with col1:
            # Slider to scroll through Axial slices (Z axis)
            z = st.slider("Axial slice (Z)", 0, volume.shape[2] - 1, default_z)
            fig = plot_slice(volume[:, :, z], mask[:, :, z] if mask is not None else None, prob[:, :, z] if prob is not None else None, overlay_mode)
            st.pyplot(fig)
            plt.close(fig)  # Free Matplotlib memory immediately to prevent RAM leaks

        with col2:
            # Slider to scroll through Sagittal slices (X axis)
            x = st.slider("Sagittal slice (X)", 0, volume.shape[0] - 1, default_x)
            fig = plot_slice(volume[x, :, :], mask[x, :, :] if mask is not None else None, prob[x, :, :] if prob is not None else None, overlay_mode)
            st.pyplot(fig)
            plt.close(fig)

        with col3:
            # Slider to scroll through Coronal slices (Y axis)
            y = st.slider("Coronal slice (Y)", 0, volume.shape[1] - 1, default_y)
            fig = plot_slice(volume[:, y, :], mask[:, y, :] if mask is not None else None, prob[:, y, :] if prob is not None else None, overlay_mode)
            st.pyplot(fig)
            plt.close(fig)

    with tab_3d:
        # Only build and render the 3D mesh if a non-empty segmentation mask exists
        if mask is not None and mask.sum() > 0:
            st.write("Drag to rotate, scroll to zoom:")
            with st.spinner("Building 3D Left Atrium mesh..."):
                st.plotly_chart(make_3d_mesh(mask, spacing), use_container_width=True)
        else:
            st.info("Click **▶️ Run 3D U-Net** in the sidebar to generate the 3D Left Atrium mesh.")
else:
    # Prompt shown on initial load if no scan is selected or uploaded
    st.info("Select a sample scan or upload a `.nii.gz` file from the sidebar to begin.")