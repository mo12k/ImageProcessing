# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fingerprint Image Enhancement and Quality Assessment System
#
# **BMDS2133 Image Processing - Mode A: Comparative and Enhancement Study**
#
# This notebook implements the proposed classical image-processing pipeline for low-quality fingerprint images. It compares individual techniques, integrates them into a team-level hybrid method, extracts ridge/minutiae features, measures performance, and exports experimental results only when final export is explicitly enabled.
#
# | Contribution | Technique implemented |
# |---|---|
# | Shared team preprocessing | Loading, grayscale conversion, common resize, controlled degradation/reference preparation, dtype/intensity handling and reproducible random seed |
# | Team baseline | Simple Global Histogram Equalisation baseline |
# | Member 1 | M1 - Wiener + CLAHE Enhancement |
# | Member 2 | M2 - Gabor / Modified Gabor Ridge Enhancement |
# | Member 3 | M3 - Morphological Ridge Restoration |
# | Member 4 | M4 - Thinning & Minutiae Extraction |
# | Team | Team Hybrid Pipeline, batch dashboard, optional SVM quality assessment and reporting |
#
# **Important evaluation rule:** full-reference MSE, PSNR and SSIM require aligned clean and degraded images. Therefore, controlled degradation is applied to clean SOCOFing `Real` images. The SOCOFing `Altered` images may be used later for qualitative/application testing, but they are not treated as pixel-perfect ground truth.
#
# **Dataset roles:** SOCOFing `Real` is the reference source for controlled quantitative experiments. `Altered-Easy`, `Altered-Medium` and `Altered-Hard` are reserved for later qualitative/application testing.
#

# %% [markdown]
# ## 1. Environment check
#
# Create the supplied Conda environment before running this notebook. The environment pins NumPy 1.26.4 to avoid the NumPy 1.x/2.x binary-compatibility error that can occur in an existing Anaconda installation.

# %%
from pathlib import Path
from time import perf_counter
from datetime import datetime
import json
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib

if "get_ipython" not in globals():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from scipy import ndimage as ndi, signal

from skimage import exposure, feature, filters, morphology, restoration, transform
from skimage.draw import ellipse
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from skimage.util import img_as_float

from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

try:
    import pywt
except ImportError:
    pywt = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")

try:
    from IPython.display import display, Markdown
except ImportError:
    Markdown = str

    def display(value):
        print(value)

print("Environment loaded successfully")
print(f"NumPy: {np.__version__} | pandas: {pd.__version__}")


# %% [markdown]
# ## 2. Project configuration and dataset
#
# Download SOCOFing from: <https://www.kaggle.com/datasets/ruizgara/socofing>
#
# Extract it so this structure exists once:
#
# ```text
# data/SOCOFing/Real/
# data/SOCOFing/Altered/Altered-Easy/
# data/SOCOFing/Altered/Altered-Medium/
# data/SOCOFing/Altered/Altered-Hard/
# ```
#
# The quantitative experiment uses aligned `Real` references with controlled degradation. The `Altered` folders are kept for later qualitative/application testing and are not used for raw Real-vs-Altered MSE, PSNR or SSIM.
#
# The assignment specification states that an external dataset should be cited rather than included in the submitted ZIP. If the dataset is missing, the notebook uses clearly labelled synthetic ridge images only to verify that the code runs. Synthetic results must not be reported as final experimental evidence.
#

# %%
PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name != "Fingerprint_Enhancement_Project" and (PROJECT_ROOT / "Fingerprint_Enhancement_Project").exists():
    PROJECT_ROOT = PROJECT_ROOT / "Fingerprint_Enhancement_Project"

DATA_ROOT = PROJECT_ROOT / "data" / "SOCOFing"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_EXECUTION_STARTED_AT = datetime.now().astimezone()
NOTEBOOK_EXECUTION_TIMER_START = perf_counter()

IMAGE_SIZE = (256, 256)
DEVELOPMENT_MODE = True
DEVELOPMENT_SAMPLE_IMAGES = 24
FINAL_EVALUATION_MAX_IMAGES = 500  # Formal evaluation uses exactly 500 SOCOFing Real images.
MAX_BATCH_IMAGES = DEVELOPMENT_SAMPLE_IMAGES if DEVELOPMENT_MODE else FINAL_EVALUATION_MAX_IMAGES
RUNTIME_BENCHMARK_SAMPLE_SIZE = 50
REFERENCE_SET_MAX_IMAGES = (
    DEVELOPMENT_SAMPLE_IMAGES
    if DEVELOPMENT_MODE
    else FINAL_EVALUATION_MAX_IMAGES
)
RUN_RUNTIME_BENCHMARK = False
EXPORT_FINAL_RESULTS = False
RUN_ADVANCED_DEVELOPMENT_SEARCH = True
EXPORT_ADVANCED_DEVELOPMENT_RESULTS = True
RUN_OPTIONAL_SVM = False
SAVE_OPTIONAL_SVM_MODEL = False
RANDOM_SEED = 42
SAMPLING_METHOD_DESCRIPTION = "Sorted SOCOFing Real image paths; take the first MAX_BATCH_IMAGES unique files."
LITERATURE_PSNR_BENCHMARK_DB = 28.17
SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

DEGRADED_INPUT_LABEL = "Degraded Input"
GLOBAL_HE_BASELINE_LABEL = "Global HE Baseline"
CLAHE_BASELINE_LABEL = "CLAHE Baseline"
M1_LABEL = "M1 - Wiener + CLAHE Enhancement"
M2_LABEL = "M2 - Gabor / Modified Gabor Ridge Enhancement"
M3_LABEL = "M3 - Morphological Ridge Restoration"
M4_LABEL = "M4 - Thinning & Minutiae Extraction"
TEAM_HYBRID_LABEL = "Team Hybrid Pipeline"
M1_PROPOSED_LABEL = "M1 Proposed - Wavelet + Wiener Restoration"
M2_BASELINE_LABEL = "M2 Baseline - Ordinary Gabor"
M2_PROPOSED_LABEL = "M2 Proposed - Adaptive Orientation-Frequency Gabor"
CANDIDATE_A_LABEL = "Candidate A - Restoration + Adaptive Gabor Fusion"
CANDIDATE_B_LABEL = "Candidate B - STFT Contextual Enhancement"
CANDIDATE_C_LABEL = "Candidate C - Coherence Diffusion + Log-Gabor"
TEAM_PROPOSED_LABEL = "Team Proposed - Quality/Coherence-Guided Hybrid Fusion"

CONTROL_METHODS = [DEGRADED_INPUT_LABEL]
BASELINE_METHODS = [GLOBAL_HE_BASELINE_LABEL, CLAHE_BASELINE_LABEL]
MEMBER_IMAGE_CONTRIBUTIONS = [M1_LABEL, M2_LABEL, M3_LABEL, M4_LABEL]
ENHANCEMENT_QUALITY_METHODS = [
    DEGRADED_INPUT_LABEL,
    GLOBAL_HE_BASELINE_LABEL,
    CLAHE_BASELINE_LABEL,
    M1_LABEL,
    M2_LABEL,
    TEAM_HYBRID_LABEL,
]
TEAM_METHODS = [TEAM_HYBRID_LABEL]
METHOD_ORDER = ENHANCEMENT_QUALITY_METHODS
ADVANCED_DEVELOPMENT_METHOD_ORDER = [
    DEGRADED_INPUT_LABEL,
    CLAHE_BASELINE_LABEL,
    M1_PROPOSED_LABEL,
    M2_BASELINE_LABEL,
    M2_PROPOSED_LABEL,
    CANDIDATE_A_LABEL,
    CANDIDATE_B_LABEL,
    CANDIDATE_C_LABEL,
    TEAM_PROPOSED_LABEL,
]
LEGACY_METHOD_LABELS = {
    "Degraded input": DEGRADED_INPUT_LABEL,
    "Global HE baseline": GLOBAL_HE_BASELINE_LABEL,
    "M1 CLAHE baseline": CLAHE_BASELINE_LABEL,
    "M1 Wiener + CLAHE": M1_LABEL,
    "Gabor": M2_LABEL,
    "Hybrid": TEAM_HYBRID_LABEL,
    "Global HE": GLOBAL_HE_BASELINE_LABEL,
    "CLAHE": CLAHE_BASELINE_LABEL,
}


def find_named_image_folder(root: Path, folder_name: str):
    """Find the first named folder that actually contains supported images."""
    if not root.exists():
        return None
    candidates = [root] if root.name.lower() == folder_name.lower() else []
    candidates.extend(p for p in root.rglob("*") if p.is_dir() and p.name.lower() == folder_name.lower())
    for folder in candidates:
        if any(p.suffix.lower() in SUPPORTED_EXTENSIONS for p in folder.iterdir() if p.is_file()):
            return folder
    return None


def list_image_files(folder):
    folder = None if folder is None else Path(folder)
    if folder is None or not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


REAL_DIR = find_named_image_folder(DATA_ROOT, "Real")
ALTERED_ROOT = DATA_ROOT / "Altered"
ALTERED_DIRS = {
    "Altered-Easy": ALTERED_ROOT / "Altered-Easy",
    "Altered-Medium": ALTERED_ROOT / "Altered-Medium",
    "Altered-Hard": ALTERED_ROOT / "Altered-Hard",
}

real_paths = list_image_files(REAL_DIR)
altered_paths = {name: list_image_files(path) for name, path in ALTERED_DIRS.items()}
USING_DEMO_DATA = len(real_paths) == 0

print(f"Project root : {PROJECT_ROOT}")
print(f"Dataset root : {DATA_ROOT}")
print(f"Real images  : {len(real_paths)}")
for altered_name, paths in altered_paths.items():
    print(f"{altered_name:15s}: {len(paths)}")
print(f"Development mode: {DEVELOPMENT_MODE} | MAX_BATCH_IMAGES: {MAX_BATCH_IMAGES}")
if USING_DEMO_DATA:
    print("WARNING: SOCOFing was not found. Synthetic DEMO images will be used; do not report them as final results.")
else:
    print("SOCOFing detected successfully.")


# %% [markdown]
# ## 3. Image loading and controlled degradation
#
# The synthetic degradation combines blur, sensor noise, reduced contrast, uneven illumination and a partial smudge. A fixed seed makes the experiment reproducible.

# %%
def normalise_image(image, low_percentile=1.0, high_percentile=99.0):
    """Robustly scale a grayscale image to [0, 1]."""
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [low_percentile, high_percentile])
    if high - low < 1e-8:
        return np.clip(image, 0.0, 1.0)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def load_grayscale(path, output_size=IMAGE_SIZE):
    """Load, convert to grayscale and resize without changing the aspect ratio by cropping."""
    with Image.open(path) as pil_image:
        image = np.asarray(pil_image.convert("L"), dtype=np.float32) / 255.0
    image = transform.resize(image, output_size, anti_aliasing=True, preserve_range=True)
    return normalise_image(image)


def synthetic_fingerprint(size=256, seed=0):
    """Generate a loop-like ridge image for code testing only."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    x_shift = x + 0.08 * np.sin(seed)
    y_shift = y + 0.20
    radius = np.sqrt((0.85 * x_shift) ** 2 + y_shift**2)
    angle = np.arctan2(y_shift, x_shift)
    phase = 72 * radius + 5.5 * np.sin(2 * angle) + 2.0 * x
    ridges = 0.5 + 0.45 * np.cos(phase)

    mask = (x / 0.83) ** 2 + ((y + 0.02) / 0.96) ** 2 <= 1
    image = np.where(mask, ridges, 1.0)
    image += rng.normal(0, 0.012, image.shape)
    image = ndi.gaussian_filter(image, sigma=0.55)
    return np.clip(image, 0.0, 1.0)


def make_reference_set(max_images=REFERENCE_SET_MAX_IMAGES):
    """Load real references or create a reproducible demonstration set."""
    if real_paths:
        selected = real_paths if max_images is None else real_paths[:max(0, int(max_images))]
        return [(path.name, load_grayscale(path)) for path in selected]
    demo_count = 12 if max_images is None else min(12, max(1, int(max_images)))
    return [(f"SYNTHETIC_DEMO_{index:02d}", synthetic_fingerprint(seed=index)) for index in range(demo_count)]


def simulate_degradation(reference, severity=0.55, seed=RANDOM_SEED):
    """Create an aligned low-quality input for quantitative evaluation."""
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    image = np.asarray(reference, dtype=np.float32).copy()

    # Optical/sensor blur.
    image = ndi.gaussian_filter(image, sigma=0.35 + 1.8 * severity)

    # Reduced dynamic range plus an illumination gradient.
    image = 0.5 + (image - 0.5) * (1.0 - 0.58 * severity)
    yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    illumination = 1.0 + 0.16 * severity * np.sin(2 * np.pi * xx / image.shape[1])
    image *= illumination

    # Gaussian and sparse impulse noise.
    image += rng.normal(0.0, 0.018 + 0.055 * severity, image.shape)
    impulse_mask = rng.random(image.shape)
    image[impulse_mask < 0.0015 * severity] = 0.0
    image[impulse_mask > 1.0 - 0.0015 * severity] = 1.0

    # Partial smudge/occlusion.
    if severity >= 0.35:
        smudge = np.zeros_like(image, dtype=bool)
        centre_r = int(image.shape[0] * (0.52 + rng.uniform(-0.10, 0.10)))
        centre_c = int(image.shape[1] * (0.52 + rng.uniform(-0.10, 0.10)))
        rr, cc = ellipse(
            centre_r,
            centre_c,
            max(3, int(image.shape[0] * 0.035 * severity)),
            max(5, int(image.shape[1] * 0.10 * severity)),
            shape=image.shape,
            rotation=float(rng.uniform(-0.8, 0.8)),
        )
        smudge[rr, cc] = True
        blurred_patch = ndi.gaussian_filter(image, sigma=3.0 + 2.0 * severity)
        image[smudge] = blurred_patch[smudge]

    return np.clip(image, 0.0, 1.0)


reference_set = make_reference_set()
sample_name, sample_reference = reference_set[0]
sample_degraded = simulate_degradation(sample_reference, severity=0.55, seed=RANDOM_SEED)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(sample_reference, cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"Reference\n{sample_name}")
axes[1].imshow(sample_degraded, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Controlled low-quality input")
for axis in axes:
    axis.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. M1 - Wiener + CLAHE Enhancement
#
# Member 1 final methodology is **Image Restoration and Local Contrast Enhancement using Wiener Filtering and CLAHE**.
#
# The comparison is intentionally limited to:
#
# - **CLAHE Baseline:** baseline local contrast enhancement without restoration.
# - **M1 - Wiener + CLAHE Enhancement:** Member 1 final methodology, where Wiener restoration is applied before CLAHE.
#
# CLAHE-only is not Member 1's final method. It is retained only to show what changes when image restoration is added. Removed variants such as Wiener + CLAHE + Unsharp, Median + CLAHE and CLAHE + Unsharp are not part of the Member 1 experimental comparison.
#
# The result should be interpreted through MSE, PSNR, SSIM and ridge coherence. If the metrics disagree, preserve the trade-off rather than claiming that Wiener + CLAHE is better without support.
#

# %%
def apply_clahe(image, clip_limit=0.025, kernel_size=(32, 32)):
    """Apply CLAHE to a normalised image."""
    return normalise_image(
        exposure.equalize_adapthist(
            normalise_image(image),
            clip_limit=float(clip_limit),
            kernel_size=kernel_size,
        )
    )


def member1_clahe_baseline(image, clip_limit=0.025):
    """Baseline local contrast enhancement without an explicit restoration stage."""
    return apply_clahe(image, clip_limit=clip_limit)


def member1_wiener_clahe(image, clip_limit=0.020, window_size=5):
    """Member 1 final method: Wiener restoration followed by CLAHE."""
    restored = signal.wiener(normalise_image(image), mysize=max(3, int(window_size)))
    return apply_clahe(restored, clip_limit=clip_limit)


member1_sample_outputs = {
    CLAHE_BASELINE_LABEL: member1_clahe_baseline(sample_degraded),
    M1_LABEL: member1_wiener_clahe(sample_degraded),
}
sample_member1 = member1_sample_outputs[M1_LABEL]

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
panels = [(DEGRADED_INPUT_LABEL, sample_degraded), *member1_sample_outputs.items()]
for axis, (title, image) in zip(axes, panels):
    axis.imshow(image, cmap="gray", vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")
fig.suptitle("Member 1 Enhancement Study", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.show()


# %% [markdown]
# ## 5. M2 - Gabor / Modified Gabor Ridge Enhancement
#
# A bank of Gabor filters responds to ridge patterns at several orientations. The strongest orientation response is fused with the CLAHE image. This is more robust than applying a single fixed orientation to the entire fingerprint.

# %%
def multi_orientation_gabor(image, frequency=0.115, orientations=8, blend=0.35):
    """Enhance ridge energy with a multi-orientation Gabor bank."""
    base = normalise_image(image)
    inverted = 1.0 - base
    magnitudes = []
    for theta in np.linspace(0, np.pi, int(orientations), endpoint=False):
        real, imaginary = filters.gabor(inverted, frequency=float(frequency), theta=float(theta))
        magnitudes.append(np.hypot(real, imaginary))
    ridge_energy = normalise_image(np.max(np.stack(magnitudes, axis=0), axis=0))

    # Convert high ridge energy back to dark ridges and preserve part of the original intensity structure.
    ridge_image = 1.0 - ridge_energy
    fused = (1.0 - float(blend)) * base + float(blend) * ridge_image
    return normalise_image(fused), ridge_energy


sample_gabor, sample_ridge_energy = multi_orientation_gabor(sample_member1)


# %% [markdown]
# ## 6. M3 - Morphological Ridge Restoration
#
# Local variance is used to isolate the fingerprint area. Sauvola thresholding extracts dark ridges under uneven illumination, while opening, closing and small-object removal reconnect ridge fragments and suppress isolated noise.

# %%
def fingerprint_mask(image, block_size=17):
    image = normalise_image(image)

    local_mean = ndi.uniform_filter(image, size=block_size, mode="reflect")
    local_mean_sq = ndi.uniform_filter(image**2, size=block_size, mode="reflect")
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean**2, 0.0))

    threshold = max(0.018, float(np.percentile(local_std, 48)))
    mask = local_std > threshold

    mask = morphology.closing(mask, morphology.disk(7))
    mask = morphology.remove_small_holes(mask, 800)
    mask = morphology.remove_small_objects(mask, 700)

    return mask


def raw_sauvola_ridges(image, window_size=25, sauvola_k=0.16):
    image = normalise_image(image)
    mask = fingerprint_mask(image)
    threshold_surface = filters.threshold_sauvola(
        image,
        window_size=int(window_size),
        k=float(sauvola_k)
    )
    return (image < threshold_surface) & mask, mask


def segment_and_restore_ridges(image, window_size=25, sauvola_k=0.16, return_intermediate=False):
    raw_ridges, mask = raw_sauvola_ridges(image, window_size=window_size, sauvola_k=sauvola_k)
    ridges = raw_ridges.copy()

    ridges = morphology.opening(ridges, morphology.disk(1))
    ridges = morphology.closing(ridges, morphology.disk(1))
    ridges = morphology.remove_small_objects(ridges, 12)
    ridges = morphology.remove_small_holes(ridges, 10)

    if return_intermediate:
        return ridges, mask, raw_ridges
    return ridges, mask


sample_binary, sample_mask = segment_and_restore_ridges(sample_gabor)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(sample_gabor, cmap="gray")
plt.title("M2 Gabor enhanced")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(sample_mask, cmap="gray")
plt.title("M3 fingerprint mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sample_binary, cmap="gray")
plt.title("M3 restored binary ridges")
plt.axis("off")

plt.show()

# %% [markdown]
# ## 7. M4 - Thinning & Minutiae Extraction
#
# Ridges are reduced to one-pixel-wide skeletons. For each skeleton pixel, the crossing number around its eight neighbours identifies ridge endings (CN = 1) and bifurcations (CN = 3). Points close to the segmented boundary or to another detected point are removed to reduce false minutiae.

# %%
def thin_ridges(binary_ridges):
    return morphology.thin(np.asarray(binary_ridges, dtype=bool))


def _suppress_nearby_points(points, minimum_distance=7):
    kept = []
    for row, col in points:
        if all((row - kr) ** 2 + (col - kc) ** 2 >= minimum_distance**2 for kr, kc in kept):
            kept.append((int(row), int(col)))
    return np.asarray(kept, dtype=int).reshape(-1, 2)


def extract_minutiae(skeleton, mask=None, border_margin=12, minimum_distance=7):
    """Return arrays of ridge endings and bifurcations using crossing numbers."""
    skeleton = np.asarray(skeleton, dtype=np.uint8)
    padded = np.pad(skeleton, 1)
    neighbours = [
        padded[:-2, 1:-1],      # N
        padded[:-2, 2:],        # NE
        padded[1:-1, 2:],       # E
        padded[2:, 2:],         # SE
        padded[2:, 1:-1],       # S
        padded[2:, :-2],        # SW
        padded[1:-1, :-2],      # W
        padded[:-2, :-2],       # NW
    ]
    transitions = sum(np.abs(neighbours[i].astype(int) - neighbours[(i + 1) % 8].astype(int)) for i in range(8))
    crossing_number = transitions / 2.0
    ending_map = (skeleton == 1) & (crossing_number == 1)
    bifurcation_map = (skeleton == 1) & (crossing_number == 3)

    valid = np.ones_like(skeleton, dtype=bool)
    valid[:border_margin, :] = False
    valid[-border_margin:, :] = False
    valid[:, :border_margin] = False
    valid[:, -border_margin:] = False
    if mask is not None:
        erosion_radius = max(3, border_margin // 2)
        valid &= morphology.binary_erosion(mask, morphology.disk(erosion_radius))

    endings = _suppress_nearby_points(np.argwhere(ending_map & valid), minimum_distance)
    bifurcations = _suppress_nearby_points(np.argwhere(bifurcation_map & valid), minimum_distance)
    return endings, bifurcations


def minutiae_overlay(image, endings, bifurcations, title="Extracted minutiae"):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if len(endings):
        ax.scatter(endings[:, 1], endings[:, 0], s=30, facecolors="none", edgecolors="#00d084", label="Ridge ending")
    if len(bifurcations):
        ax.scatter(bifurcations[:, 1], bifurcations[:, 0], s=34, marker="x", c="#ff3b30", label="Bifurcation")
    ax.set_title(title)
    ax.axis("off")
    if len(endings) or len(bifurcations):
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=True)
    fig.tight_layout()
    return fig, ax


sample_skeleton = thin_ridges(sample_binary)
sample_endings, sample_bifurcations = extract_minutiae(sample_skeleton, sample_mask)
_ = minutiae_overlay(sample_skeleton, sample_endings, sample_bifurcations)
plt.show()
print(f"Ridge endings: {len(sample_endings)} | Bifurcations: {len(sample_bifurcations)}")

# %% [markdown]
# ## 8. Integrated pipeline and quantitative metrics
#
# The integrated comparison keeps experimental controls, baselines, individual member work and team integration separate.
#
# Full-reference MSE, PSNR and SSIM are reported only for aligned grayscale enhancement/restoration outputs:
#
# - **Degraded Input:** controlled low-quality input.
# - **Global HE Baseline:** simple histogram equalisation baseline.
# - **CLAHE Baseline:** local contrast baseline.
# - **M1 - Wiener + CLAHE Enhancement:** Member 1 final methodology.
# - **M2 - Gabor / Modified Gabor Ridge Enhancement:** Member 2 ridge enhancement.
# - **Team Hybrid Pipeline:** integrated team-level output.
#
# M3 and M4 are evaluated with structural outputs and minutiae counts instead of direct greyscale PSNR:
#
# - **M3 - Morphological Ridge Restoration:** binary ridge restoration before skeletonisation.
# - **M4 - Thinning & Minutiae Extraction:** skeleton, ridge endings and bifurcations.
#
# The following measurements are reported:
#
# - **MSE:** lower is better.
# - **PSNR:** higher is better and is stated in decibels (dB).
# - **SSIM:** higher is better; 1 represents identical structure.
# - **Orientation coherence:** higher values indicate more consistent local ridge direction.
# - **Keypoint match score:** percentage of cross-checked ORB matches relative to the smaller detected keypoint set. It is a matching score, not classification accuracy.
# - **Runtime:** lower is faster.
#
# For MSE, PSNR and SSIM, the reference and candidate are only clipped to the same `[0, 1]` range. They are not independently contrast-normalised immediately before metric calculation.
#

# %%
def orientation_coherence(image, mask=None, smoothing_sigma=2.0):
    image = normalise_image(image)
    gx = filters.sobel_h(image)
    gy = filters.sobel_v(image)
    gxx = ndi.gaussian_filter(gx * gx, smoothing_sigma)
    gyy = ndi.gaussian_filter(gy * gy, smoothing_sigma)
    gxy = ndi.gaussian_filter(gx * gy, smoothing_sigma)
    coherence = np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy**2) / (gxx + gyy + 1e-8)
    if mask is None or not np.any(mask):
        return float(np.mean(coherence))
    return float(np.mean(coherence[np.asarray(mask, dtype=bool)]))


def keypoint_match_score(reference, candidate, maximum_keypoints=300):
    """Calculate a reproducible ORB matching score in the range 0-100."""
    try:
        orb_ref = feature.ORB(n_keypoints=maximum_keypoints, fast_threshold=0.06)
        orb_ref.detect_and_extract(normalise_image(reference))
        orb_candidate = feature.ORB(n_keypoints=maximum_keypoints, fast_threshold=0.06)
        orb_candidate.detect_and_extract(normalise_image(candidate))
        matches = feature.match_descriptors(
            orb_ref.descriptors,
            orb_candidate.descriptors,
            cross_check=True,
            max_ratio=0.80,
        )
        denominator = max(1, min(len(orb_ref.keypoints), len(orb_candidate.keypoints)))
        return float(100.0 * len(matches) / denominator)
    except (RuntimeError, ValueError):
        return 0.0


def normalise_method_labels(frame):
    """Map legacy CSV/report method names to the corrected role-aware labels."""
    frame = frame.copy()
    if "Method" in frame.columns:
        frame["Method"] = frame["Method"].replace(LEGACY_METHOD_LABELS)
    return frame


def metric_image(image):
    """Prepare an already aligned [0, 1] image for full-reference metrics without contrast re-normalising it."""
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(image, 0.0, 1.0)


def global_histogram_equalisation(image):
    """Simple team-level baseline using global histogram equalisation only."""
    return metric_image(exposure.equalize_hist(metric_image(image)))


def evaluate_grayscale(reference, candidate, mask, runtime_seconds=np.nan):
    reference = metric_image(reference)
    candidate = metric_image(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"Metric images must be aligned and equal-sized; got {reference.shape} and {candidate.shape}.")
    return {
        "MSE": float(mean_squared_error(reference, candidate)),
        "PSNR (dB)": float(peak_signal_noise_ratio(reference, candidate, data_range=1.0)),
        "SSIM": float(structural_similarity(reference, candidate, data_range=1.0)),
        "Coherence": orientation_coherence(candidate, mask),
        "Match score (%)": keypoint_match_score(reference, candidate),
        "Runtime (s)": float(runtime_seconds),
    }


def _odd_int(value, minimum=3):
    value = max(int(value), int(minimum))
    return value if value % 2 == 1 else value + 1


def _block_starts(length, block_size, step):
    length = int(length)
    block_size = min(int(block_size), length)
    step = max(1, int(step))
    if length <= block_size:
        return [0]
    starts = list(range(0, length - block_size + 1, step))
    if starts[-1] != length - block_size:
        starts.append(length - block_size)
    return starts


def _angle_difference_pi(angle_a, angle_b):
    """Smallest absolute difference between orientations modulo pi."""
    return np.abs(np.angle(np.exp(2j * (angle_a - angle_b)))) / 2.0


def wavelet_denoise(image, wavelet="db2", level=2, threshold_scale=0.12, threshold_strategy="soft"):
    """Denoise with discrete wavelet shrinkage; fall back to TV denoising if PyWavelets is unavailable."""
    base = metric_image(image)
    threshold_strategy = str(threshold_strategy).lower()
    if pywt is None:
        return metric_image(restoration.denoise_tv_chambolle(base, weight=0.025 + 0.05 * float(threshold_scale)))

    try:
        wavelet_object = pywt.Wavelet(wavelet)
        max_level = pywt.dwtn_max_level(base.shape, wavelet_object)
        active_level = max(1, min(int(level), int(max_level)))
        coeffs = pywt.wavedec2(base, wavelet=wavelet_object, level=active_level, mode="symmetric")
    except (TypeError, ValueError):
        return metric_image(restoration.denoise_tv_chambolle(base, weight=0.025 + 0.05 * float(threshold_scale)))

    finest_detail = np.concatenate([np.ravel(detail) for detail in coeffs[-1]])
    sigma = np.median(np.abs(finest_detail - np.median(finest_detail))) / 0.6745
    if not np.isfinite(sigma) or sigma < 1e-8:
        return base.copy()

    universal_threshold = sigma * np.sqrt(2.0 * np.log(base.size))
    threshold_mode = "hard" if threshold_strategy == "hard" else "soft"
    denoised_coeffs = [coeffs[0]]
    for level_index, detail_triplet in enumerate(coeffs[1:], start=1):
        denoised_detail = []
        for detail in detail_triplet:
            if threshold_strategy == "bayes":
                detail_variance = float(np.mean(detail**2))
                signal_sigma = np.sqrt(max(detail_variance - sigma**2, 1e-8))
                threshold_value = float(threshold_scale) * sigma**2 / (signal_sigma + 1e-8)
            elif threshold_strategy == "level_scaled":
                threshold_value = float(threshold_scale) * universal_threshold / np.sqrt(level_index)
            else:
                threshold_value = float(threshold_scale) * universal_threshold
            denoised_detail.append(pywt.threshold(detail, value=threshold_value, mode=threshold_mode))
        denoised_coeffs.append(tuple(denoised_detail))

    reconstructed = pywt.waverec2(denoised_coeffs, wavelet=wavelet_object, mode="symmetric")
    reconstructed = reconstructed[: base.shape[0], : base.shape[1]]
    return metric_image(reconstructed)


def gaussian_psf(sigma=1.1, size=None):
    sigma = max(0.25, float(sigma))
    if size is None:
        size = _odd_int(np.ceil(6 * sigma), minimum=5)
    size = _odd_int(size, minimum=5)
    radius = size // 2
    yy, xx = np.mgrid[-radius: radius + 1, -radius: radius + 1]
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return psf / max(float(psf.sum()), 1e-8)


def frequency_wiener_deconvolution(image, psf_sigma=1.1, balance=0.08, blend=0.05):
    """Mild Wiener deconvolution with a Gaussian PSF and conservative blending."""
    base = metric_image(image)
    psf = gaussian_psf(sigma=psf_sigma)
    otf = np.fft.fft2(np.fft.ifftshift(psf), s=base.shape)
    spectrum = np.fft.fft2(base)
    restored = np.real(np.fft.ifft2((np.conj(otf) / (np.abs(otf) ** 2 + float(balance))) * spectrum))
    restored = metric_image(restored)
    return metric_image((1.0 - float(blend)) * base + float(blend) * restored)


def local_intensity_normalisation(image, window_size=41, contrast=0.12, blend=0.08):
    """Local mean-variance normalisation blended back into the grayscale restoration branch."""
    base = metric_image(image)
    window_size = _odd_int(window_size, minimum=9)
    local_mean = ndi.uniform_filter(base, size=window_size, mode="reflect")
    local_mean_sq = ndi.uniform_filter(base**2, size=window_size, mode="reflect")
    local_std = np.sqrt(np.maximum(local_mean_sq - local_mean**2, 1e-8))
    z_score = (base - local_mean) / (local_std + 1e-4)
    normalised = np.clip(0.5 + float(contrast) * z_score, 0.0, 1.0)
    return metric_image((1.0 - float(blend)) * base + float(blend) * normalised)


def wavelet_wiener_restoration(image, config=None):
    """Member 1 proposed branch: wavelet denoising followed by conservative Wiener restoration."""
    config = {} if config is None else dict(config)
    base = metric_image(image)
    denoised = wavelet_denoise(
        base,
        wavelet=config.get("wavelet", "db2"),
        level=config.get("wavelet_level", 2),
        threshold_scale=config.get("wavelet_threshold_scale", 0.12),
        threshold_strategy=config.get("wavelet_threshold_strategy", "soft"),
    )
    local_wiener = signal.wiener(denoised, mysize=_odd_int(config.get("local_wiener_window", 5), minimum=3))
    local_wiener = metric_image(local_wiener)
    local_wiener_blend = float(config.get("local_wiener_blend", 0.20))
    restored = metric_image((1.0 - local_wiener_blend) * denoised + local_wiener_blend * local_wiener)
    restored = frequency_wiener_deconvolution(
        restored,
        psf_sigma=config.get("psf_sigma", 1.1),
        balance=config.get("deconv_balance", 0.08),
        blend=config.get("deconv_blend", 0.05),
    )
    stages = {
        "Degraded": base,
        "Wavelet denoised": denoised,
        "Deblurred/restored": restored,
    }
    return restored, stages


def estimate_orientation_field(image, smoothing_sigma=3.0, orientation_smoothing_sigma=2.0):
    """Estimate local ridge orientation and orientation coherence from the structure tensor."""
    base = normalise_image(image)
    grad_x = ndi.sobel(base, axis=1, mode="reflect")
    grad_y = ndi.sobel(base, axis=0, mode="reflect")
    gxx_minus_gyy = ndi.gaussian_filter(grad_x**2 - grad_y**2, float(smoothing_sigma), mode="reflect")
    two_gxy = ndi.gaussian_filter(2.0 * grad_x * grad_y, float(smoothing_sigma), mode="reflect")
    energy = ndi.gaussian_filter(grad_x**2 + grad_y**2, float(smoothing_sigma), mode="reflect")

    gradient_orientation = 0.5 * np.arctan2(two_gxy, gxx_minus_gyy)
    ridge_orientation = (gradient_orientation + np.pi / 2.0) % np.pi

    smooth_sin = ndi.gaussian_filter(np.sin(2.0 * ridge_orientation), float(orientation_smoothing_sigma), mode="reflect")
    smooth_cos = ndi.gaussian_filter(np.cos(2.0 * ridge_orientation), float(orientation_smoothing_sigma), mode="reflect")
    ridge_orientation = (0.5 * np.arctan2(smooth_sin, smooth_cos)) % np.pi
    coherence = np.sqrt(gxx_minus_gyy**2 + two_gxy**2) / (energy + 1e-8)
    return ridge_orientation, np.clip(coherence, 0.0, 1.0)


def estimate_block_frequency(block, freq_min=0.045, freq_max=0.18, default_frequency=0.10):
    """Estimate dominant ridge frequency from a local Fourier spectrum."""
    block = metric_image(block)
    centred = block - float(np.mean(block))
    if float(np.std(centred)) < 1e-5:
        return float(default_frequency), 0.0, 0.0

    rows, cols = block.shape
    window = np.outer(np.hanning(rows), np.hanning(cols))
    spectrum = np.abs(np.fft.fft2(centred * window))
    fy = np.fft.fftfreq(rows)
    fx = np.fft.fftfreq(cols)
    grid_y, grid_x = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(grid_x**2 + grid_y**2)
    valid = (radius >= float(freq_min)) & (radius <= float(freq_max))
    if not np.any(valid):
        return float(default_frequency), 0.0, 0.0

    valid_spectrum = np.where(valid, spectrum, 0.0)
    peak_index = np.unravel_index(int(np.argmax(valid_spectrum)), valid_spectrum.shape)
    peak_value = float(valid_spectrum[peak_index])
    band_values = spectrum[valid]
    band_mean = float(np.mean(band_values)) + 1e-8
    confidence = np.clip((peak_value / band_mean - 1.0) / 6.0, 0.0, 1.0)
    normal_orientation = float(np.arctan2(grid_y[peak_index], grid_x[peak_index]) % np.pi)
    ridge_orientation = float((normal_orientation + np.pi / 2.0) % np.pi)
    return float(radius[peak_index]), ridge_orientation, float(confidence)


def estimate_local_frequency_map(
    image,
    mask=None,
    block_size=40,
    freq_min=0.045,
    freq_max=0.18,
    default_frequency=0.10,
):
    """Estimate local ridge frequency on non-overlapping blocks and smooth it into a full image map."""
    base = metric_image(image)
    rows, cols = base.shape
    block_size = min(int(block_size), rows, cols)
    freq_map = np.full_like(base, float(default_frequency), dtype=np.float32)
    confidence_map = np.zeros_like(base, dtype=np.float32)
    fft_orientation_map = np.zeros_like(base, dtype=np.float32)
    mask = np.ones_like(base, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    for row in _block_starts(rows, block_size, block_size):
        for col in _block_starts(cols, block_size, block_size):
            block_mask = mask[row: row + block_size, col: col + block_size]
            if np.mean(block_mask) < 0.18:
                continue
            frequency, fft_orientation, confidence = estimate_block_frequency(
                base[row: row + block_size, col: col + block_size],
                freq_min=freq_min,
                freq_max=freq_max,
                default_frequency=default_frequency,
            )
            freq_map[row: row + block_size, col: col + block_size] = frequency
            confidence_map[row: row + block_size, col: col + block_size] = confidence
            fft_orientation_map[row: row + block_size, col: col + block_size] = fft_orientation

    smooth_size = _odd_int(max(3, block_size // 2), minimum=3)
    freq_map = ndi.median_filter(freq_map, size=smooth_size, mode="nearest")
    confidence_map = np.clip(ndi.gaussian_filter(confidence_map, sigma=max(1.0, block_size / 12.0)), 0.0, 1.0)
    return freq_map, confidence_map, fft_orientation_map


def adaptive_gabor_enhancement(
    image,
    orientation_map,
    frequency_map,
    mask=None,
    orientation_bins=6,
    frequency_bins=(0.07, 0.10, 0.13),
    bandwidth=1.8,
):
    """Select Gabor responses by local orientation and local ridge-frequency bins."""
    base = metric_image(image)
    mask = np.ones_like(base, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    orientation_bins = max(4, int(orientation_bins))
    frequency_bins = np.asarray(frequency_bins, dtype=float)
    theta_values = np.linspace(0.0, np.pi, orientation_bins, endpoint=False)

    theta_index = np.floor(((orientation_map % np.pi) / np.pi) * orientation_bins + 0.5).astype(int) % orientation_bins
    frequency_index = np.argmin(np.abs(frequency_map[..., None] - frequency_bins.reshape(1, 1, -1)), axis=2)
    selected_energy = np.zeros_like(base, dtype=np.float32)
    inverted = 1.0 - base

    for freq_idx, frequency in enumerate(frequency_bins):
        for theta_idx, theta in enumerate(theta_values):
            real_response, imaginary_response = filters.gabor(
                inverted,
                frequency=float(frequency),
                theta=float(theta),
                bandwidth=float(bandwidth),
            )
            response = normalise_image(np.hypot(real_response, imaginary_response))
            selected = (frequency_index == freq_idx) & (theta_index == theta_idx) & mask
            selected_energy[selected] = response[selected]

    ridge_energy = normalise_image(selected_energy)
    ridge_enhanced = np.where(mask, 1.0 - ridge_energy, base)
    return metric_image(ridge_enhanced), ridge_energy


def confidence_guided_fusion(restored, enhanced, mask=None, coherence=None, frequency_confidence=None, strength=0.06, threshold=0.25):
    """Blend ridge enhancement into the restoration branch only where local confidence is sufficient."""
    restored = metric_image(restored)
    enhanced = metric_image(enhanced)
    weight = np.ones_like(restored, dtype=np.float32)
    if coherence is not None:
        weight *= np.clip(coherence, 0.0, 1.0)
    if frequency_confidence is not None:
        weight *= np.clip(frequency_confidence, 0.0, 1.0)
    if mask is not None:
        weight *= np.asarray(mask, dtype=bool)
    threshold = float(threshold)
    if threshold > 0:
        weight = np.clip((weight - threshold) / max(1e-8, 1.0 - threshold), 0.0, 1.0)
    weight = np.clip(ndi.gaussian_filter(weight, sigma=2.0), 0.0, 1.0)
    alpha = float(strength) * weight
    fused = (1.0 - alpha) * restored + alpha * enhanced
    return metric_image(fused), weight


def candidate_a_restoration_adaptive_gabor_fusion(degraded, restored=None, restoration_stages=None, config=None):
    """Candidate A: restoration branch plus local orientation/frequency adaptive Gabor fusion."""
    config = {} if config is None else dict(config)
    if restored is None or restoration_stages is None:
        restored, restoration_stages = wavelet_wiener_restoration(degraded, config)
    mask = fingerprint_mask(restored)
    local_norm = local_intensity_normalisation(
        restored,
        window_size=config.get("local_norm_window", 41),
        contrast=config.get("local_norm_contrast", 0.12),
        blend=config.get("local_norm_blend", 0.08),
    )
    orientation_map, coherence = estimate_orientation_field(
        local_norm,
        smoothing_sigma=config.get("orientation_smoothing_sigma", 3.0),
        orientation_smoothing_sigma=config.get("orientation_vector_smoothing_sigma", 2.0),
    )
    frequency_map, frequency_confidence, _ = estimate_local_frequency_map(
        local_norm,
        mask=mask,
        block_size=config.get("adaptive_block_size", 40),
        freq_min=config.get("frequency_min", 0.045),
        freq_max=config.get("frequency_max", 0.18),
        default_frequency=config.get("default_frequency", 0.10),
    )
    adaptive_gabor, ridge_energy = adaptive_gabor_enhancement(
        local_norm,
        orientation_map=orientation_map,
        frequency_map=frequency_map,
        mask=mask,
        orientation_bins=config.get("adaptive_orientation_bins", 6),
        frequency_bins=config.get("adaptive_frequency_bins", (0.07, 0.10, 0.13)),
        bandwidth=config.get("adaptive_gabor_bandwidth", 1.8),
    )
    fused, fusion_weight = confidence_guided_fusion(
        restored,
        adaptive_gabor,
        mask=mask,
        coherence=coherence,
        frequency_confidence=frequency_confidence,
        strength=config.get("adaptive_fusion_strength", 0.06),
        threshold=config.get("confidence_threshold", 0.25),
    )
    stages = {
        "Degraded": metric_image(degraded),
        "Wavelet denoised": restoration_stages["Wavelet denoised"],
        "Deblurred/restored": restoration_stages["Deblurred/restored"],
        "Local normalised": local_norm,
        "Adaptive Gabor": adaptive_gabor,
        "Fused output": fused,
    }
    aux = {
        "mask": mask,
        "orientation_map": orientation_map,
        "frequency_map": frequency_map,
        "coherence": coherence,
        "frequency_confidence": frequency_confidence,
        "ridge_energy": ridge_energy,
        "fusion_weight": fusion_weight,
    }
    return fused, stages, aux


def stft_contextual_filter(
    image,
    mask=None,
    block_size=48,
    overlap=24,
    freq_min=0.045,
    freq_max=0.18,
    radial_bandwidth=0.026,
    directional_bandwidth=np.pi / 8.0,
    gain=0.35,
    default_frequency=0.10,
):
    """Block STFT-style contextual filtering with overlap-add reconstruction."""
    base = metric_image(image)
    rows, cols = base.shape
    block_size = min(int(block_size), rows, cols)
    overlap = min(int(overlap), block_size - 1)
    step = max(1, block_size - overlap)
    mask = np.ones_like(base, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    window = np.outer(np.hanning(block_size), np.hanning(block_size))
    window = 0.10 + 0.90 * window
    output = np.zeros_like(base, dtype=np.float32)
    weights = np.zeros_like(base, dtype=np.float32)
    frequency_map = np.full_like(base, float(default_frequency), dtype=np.float32)
    confidence_map = np.zeros_like(base, dtype=np.float32)

    fy = np.fft.fftfreq(block_size)
    fx = np.fft.fftfreq(block_size)
    grid_y, grid_x = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(grid_x**2 + grid_y**2)
    spectrum_angle = np.arctan2(grid_y, grid_x) % np.pi
    valid_band = (radius >= float(freq_min)) & (radius <= float(freq_max))

    for row in _block_starts(rows, block_size, step):
        for col in _block_starts(cols, block_size, step):
            block_mask = mask[row: row + block_size, col: col + block_size]
            block = base[row: row + block_size, col: col + block_size]
            if np.mean(block_mask) < 0.18:
                output[row: row + block_size, col: col + block_size] += block * window
                weights[row: row + block_size, col: col + block_size] += window
                continue

            centred = block - float(np.mean(block))
            spectrum = np.fft.fft2(centred * window)
            magnitude = np.abs(spectrum)
            valid_magnitude = np.where(valid_band, magnitude, 0.0)
            peak_index = np.unravel_index(int(np.argmax(valid_magnitude)), valid_magnitude.shape)
            peak_frequency = float(radius[peak_index])
            if peak_frequency <= 0:
                peak_frequency = float(default_frequency)
            band_mean = float(np.mean(magnitude[valid_band])) + 1e-8
            confidence = float(np.clip((valid_magnitude[peak_index] / band_mean - 1.0) / 6.0, 0.0, 1.0))
            normal_orientation = float(np.arctan2(grid_y[peak_index], grid_x[peak_index]) % np.pi)

            radial_filter = np.exp(-((radius - peak_frequency) ** 2) / (2.0 * float(radial_bandwidth) ** 2))
            angular_filter = np.exp(-(_angle_difference_pi(spectrum_angle, normal_orientation) ** 2) / (2.0 * float(directional_bandwidth) ** 2))
            contextual_filter = 1.0 + float(gain) * confidence * radial_filter * angular_filter
            contextual_filter[~valid_band] = 1.0
            filtered = np.real(np.fft.ifft2(spectrum * contextual_filter)) + float(np.mean(block))

            output[row: row + block_size, col: col + block_size] += metric_image(filtered) * window
            weights[row: row + block_size, col: col + block_size] += window
            frequency_map[row: row + block_size, col: col + block_size] = peak_frequency
            confidence_map[row: row + block_size, col: col + block_size] = confidence

    reconstructed = np.divide(output, np.maximum(weights, 1e-8))
    confidence_map = np.clip(ndi.gaussian_filter(confidence_map, sigma=max(1.0, block_size / 12.0)), 0.0, 1.0)
    return metric_image(reconstructed), frequency_map, confidence_map


def candidate_b_stft_contextual_enhancement(degraded, restored=None, restoration_stages=None, config=None):
    """Candidate B: overlapping-block STFT/contextual ridge enhancement blended with restoration."""
    config = {} if config is None else dict(config)
    if restored is None or restoration_stages is None:
        restored, restoration_stages = wavelet_wiener_restoration(degraded, config)
    mask = fingerprint_mask(restored)
    local_norm = local_intensity_normalisation(
        restored,
        window_size=config.get("local_norm_window", 41),
        contrast=config.get("local_norm_contrast", 0.12),
        blend=config.get("stft_local_norm_blend", config.get("local_norm_blend", 0.08)),
    )
    stft_image, frequency_map, frequency_confidence = stft_contextual_filter(
        local_norm,
        mask=mask,
        block_size=config.get("stft_block_size", 48),
        overlap=config.get("stft_overlap", 24),
        freq_min=config.get("frequency_min", 0.045),
        freq_max=config.get("frequency_max", 0.18),
        radial_bandwidth=config.get("stft_radial_bandwidth", 0.026),
        directional_bandwidth=config.get("stft_directional_bandwidth", np.pi / 8.0),
        gain=config.get("stft_gain", 0.35),
        default_frequency=config.get("default_frequency", 0.10),
    )
    orientation_map, coherence = estimate_orientation_field(
        stft_image,
        smoothing_sigma=config.get("orientation_smoothing_sigma", 3.0),
        orientation_smoothing_sigma=config.get("orientation_vector_smoothing_sigma", 2.0),
    )
    fused, fusion_weight = confidence_guided_fusion(
        restored,
        stft_image,
        mask=mask,
        coherence=coherence,
        frequency_confidence=frequency_confidence,
        strength=config.get("stft_fusion_strength", 0.05),
        threshold=config.get("confidence_threshold", 0.25),
    )
    stages = {
        "Degraded": metric_image(degraded),
        "Wavelet denoised": restoration_stages["Wavelet denoised"],
        "Deblurred/restored": restoration_stages["Deblurred/restored"],
        "Local normalised": local_norm,
        "STFT contextual": stft_image,
        "Fused output": fused,
    }
    aux = {
        "mask": mask,
        "orientation_map": orientation_map,
        "frequency_map": frequency_map,
        "coherence": coherence,
        "frequency_confidence": frequency_confidence,
        "fusion_weight": fusion_weight,
    }
    return fused, stages, aux


def oriented_gaussian_kernel(theta, sigma_parallel=1.45, sigma_perpendicular=0.45, radius=None):
    """Small anisotropic Gaussian kernel aligned with the local ridge direction."""
    sigma_parallel = max(0.40, float(sigma_parallel))
    sigma_perpendicular = max(0.25, float(sigma_perpendicular))
    if radius is None:
        radius = int(np.ceil(3.0 * max(sigma_parallel, sigma_perpendicular)))
    radius = max(2, int(radius))
    yy, xx = np.mgrid[-radius: radius + 1, -radius: radius + 1]
    parallel = xx * np.cos(theta) + yy * np.sin(theta)
    perpendicular = -xx * np.sin(theta) + yy * np.cos(theta)
    kernel = np.exp(
        -(parallel**2 / (2.0 * sigma_parallel**2) + perpendicular**2 / (2.0 * sigma_perpendicular**2))
    )
    return kernel / max(float(kernel.sum()), 1e-8)


def orientation_selected_smoothing(image, orientation_map, orientation_bins=6, sigma_parallel=1.45, sigma_perpendicular=0.45):
    base = metric_image(image)
    orientation_bins = max(4, int(orientation_bins))
    theta_values = np.linspace(0.0, np.pi, orientation_bins, endpoint=False)
    theta_index = np.floor(((orientation_map % np.pi) / np.pi) * orientation_bins + 0.5).astype(int) % orientation_bins
    smoothed = np.zeros_like(base, dtype=np.float32)
    for index, theta in enumerate(theta_values):
        kernel = oriented_gaussian_kernel(
            theta,
            sigma_parallel=sigma_parallel,
            sigma_perpendicular=sigma_perpendicular,
        )
        filtered = ndi.convolve(base, kernel, mode="reflect")
        selected = theta_index == index
        smoothed[selected] = filtered[selected]
    return metric_image(smoothed)


def coherence_guided_diffusion(image, mask=None, iterations=2, step=0.18, orientation_bins=6):
    """Approximate coherence-enhancing diffusion using orientation-selected anisotropic smoothing."""
    current = metric_image(image)
    mask = np.ones_like(current, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    orientation_map, coherence = estimate_orientation_field(current)
    for _ in range(max(1, int(iterations))):
        smoothed = orientation_selected_smoothing(
            current,
            orientation_map=orientation_map,
            orientation_bins=orientation_bins,
        )
        weight = np.clip(coherence, 0.0, 1.0) * mask
        current = metric_image((1.0 - float(step) * weight) * current + float(step) * weight * smoothed)
    return current, orientation_map, coherence


def log_gabor_enhancement(
    image,
    orientation_map,
    frequency_map,
    mask=None,
    orientation_bins=6,
    frequency_bins=(0.07, 0.10, 0.13),
    sigma_on_frequency=0.60,
    angular_sigma=np.pi / 7.0,
):
    """Local-selection Log-Gabor bank using estimated orientation and ridge frequency maps."""
    base = metric_image(image)
    mask = np.ones_like(base, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    rows, cols = base.shape
    fy = np.fft.fftfreq(rows)
    fx = np.fft.fftfreq(cols)
    grid_y, grid_x = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(grid_x**2 + grid_y**2)
    radius[0, 0] = 1.0
    frequency_angle = np.arctan2(grid_y, grid_x) % np.pi

    orientation_bins = max(4, int(orientation_bins))
    frequency_bins = np.asarray(frequency_bins, dtype=float)
    normal_orientation_map = (orientation_map - np.pi / 2.0) % np.pi
    theta_values = np.linspace(0.0, np.pi, orientation_bins, endpoint=False)
    theta_index = np.floor(((normal_orientation_map % np.pi) / np.pi) * orientation_bins + 0.5).astype(int) % orientation_bins
    frequency_index = np.argmin(np.abs(frequency_map[..., None] - frequency_bins.reshape(1, 1, -1)), axis=2)

    spectrum = np.fft.fft2((1.0 - base) - float(np.mean(1.0 - base)))
    selected_energy = np.zeros_like(base, dtype=np.float32)
    log_sigma = np.log(float(sigma_on_frequency))
    for freq_idx, frequency in enumerate(frequency_bins):
        radial = np.exp(-(np.log(radius / float(frequency)) ** 2) / (2.0 * log_sigma**2))
        radial[0, 0] = 0.0
        for theta_idx, theta in enumerate(theta_values):
            angular = np.exp(-(_angle_difference_pi(frequency_angle, theta) ** 2) / (2.0 * float(angular_sigma) ** 2))
            response = normalise_image(np.abs(np.fft.ifft2(spectrum * radial * angular)))
            selected = (frequency_index == freq_idx) & (theta_index == theta_idx) & mask
            selected_energy[selected] = response[selected]

    ridge_energy = normalise_image(selected_energy)
    ridge_image = np.where(mask, 1.0 - ridge_energy, base)
    return metric_image(ridge_image), ridge_energy


def candidate_c_coherence_diffusion_log_gabor(degraded, restored=None, restoration_stages=None, config=None):
    """Candidate C: coherence-guided diffusion, local ridge frequency and Log-Gabor fusion."""
    config = {} if config is None else dict(config)
    if restored is None or restoration_stages is None:
        restored, restoration_stages = wavelet_wiener_restoration(degraded, config)
    mask = fingerprint_mask(restored)
    diffused, orientation_map, coherence = coherence_guided_diffusion(
        restored,
        mask=mask,
        iterations=config.get("diffusion_iterations", 2),
        step=config.get("diffusion_step", 0.18),
        orientation_bins=config.get("diffusion_orientation_bins", config.get("adaptive_orientation_bins", 6)),
    )
    frequency_map, frequency_confidence, _ = estimate_local_frequency_map(
        diffused,
        mask=mask,
        block_size=config.get("adaptive_block_size", 40),
        freq_min=config.get("frequency_min", 0.045),
        freq_max=config.get("frequency_max", 0.18),
        default_frequency=config.get("default_frequency", 0.10),
    )
    log_gabor, ridge_energy = log_gabor_enhancement(
        diffused,
        orientation_map=orientation_map,
        frequency_map=frequency_map,
        mask=mask,
        orientation_bins=config.get("log_gabor_orientation_bins", config.get("adaptive_orientation_bins", 6)),
        frequency_bins=config.get("adaptive_frequency_bins", (0.07, 0.10, 0.13)),
        sigma_on_frequency=config.get("log_gabor_sigma_on_frequency", 0.60),
        angular_sigma=config.get("log_gabor_angular_sigma", np.pi / 7.0),
    )
    fused, fusion_weight = confidence_guided_fusion(
        diffused,
        log_gabor,
        mask=mask,
        coherence=coherence,
        frequency_confidence=frequency_confidence,
        strength=config.get("log_gabor_fusion_strength", 0.05),
        threshold=config.get("confidence_threshold", 0.25),
    )
    stages = {
        "Degraded": metric_image(degraded),
        "Wavelet denoised": restoration_stages["Wavelet denoised"],
        "Deblurred/restored": restoration_stages["Deblurred/restored"],
        "Coherence diffusion": diffused,
        "Log-Gabor": log_gabor,
        "Fused output": fused,
    }
    aux = {
        "mask": mask,
        "orientation_map": orientation_map,
        "frequency_map": frequency_map,
        "coherence": coherence,
        "frequency_confidence": frequency_confidence,
        "ridge_energy": ridge_energy,
        "fusion_weight": fusion_weight,
    }
    return fused, stages, aux


def quality_guided_team_fusion(restored, candidate_outputs, mask=None, fusion_strength=0.07):
    """Team candidate: internally weight candidate outputs by ridge coherence and blend with restoration."""
    restored = metric_image(restored)
    mask = np.ones_like(restored, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    valid_outputs = [(label, metric_image(image)) for label, image in candidate_outputs if image is not None]
    if not valid_outputs:
        return restored.copy(), {}

    scores = []
    for _, image in valid_outputs:
        scores.append(max(1e-4, orientation_coherence(image, mask)))
    scores = np.asarray(scores, dtype=float)
    weights = scores / max(float(scores.sum()), 1e-8)
    ridge_stack = np.stack([image for _, image in valid_outputs], axis=0)
    internally_selected = np.tensordot(weights, ridge_stack, axes=(0, 0))
    fused, _ = confidence_guided_fusion(
        restored,
        internally_selected,
        mask=mask,
        coherence=np.ones_like(restored),
        frequency_confidence=np.ones_like(restored),
        strength=fusion_strength,
        threshold=0.0,
    )
    return fused, {label: float(weight) for (label, _), weight in zip(valid_outputs, weights)}


def run_comparison(reference, severity=0.55, seed=RANDOM_SEED):
    """Run comparable grayscale methods and the complete binary-feature pipeline."""
    degraded = simulate_degradation(reference, severity=severity, seed=seed)
    reference_mask = fingerprint_mask(reference)

    outputs = {DEGRADED_INPUT_LABEL: degraded}
    runtimes = {DEGRADED_INPUT_LABEL: 0.0}

    start = perf_counter()
    outputs[GLOBAL_HE_BASELINE_LABEL] = global_histogram_equalisation(degraded)
    runtimes[GLOBAL_HE_BASELINE_LABEL] = perf_counter() - start

    start = perf_counter()
    outputs[CLAHE_BASELINE_LABEL] = member1_clahe_baseline(degraded)
    runtimes[CLAHE_BASELINE_LABEL] = perf_counter() - start

    start = perf_counter()
    member1_final = member1_wiener_clahe(degraded)
    outputs[M1_LABEL] = member1_final
    runtimes[M1_LABEL] = perf_counter() - start

    start = perf_counter()
    gabor_input = apply_clahe(degraded, clip_limit=0.012)
    gabor_only, _ = multi_orientation_gabor(gabor_input, blend=0.58)
    outputs[M2_LABEL] = gabor_only
    runtimes[M2_LABEL] = perf_counter() - start

    start = perf_counter()
    hybrid_gray, ridge_energy = multi_orientation_gabor(member1_final, blend=0.30)
    outputs[TEAM_HYBRID_LABEL] = hybrid_gray
    runtimes[TEAM_HYBRID_LABEL] = perf_counter() - start

    binary, mask, raw_binary = segment_and_restore_ridges(hybrid_gray, return_intermediate=True)
    skeleton = thin_ridges(binary)
    endings, bifurcations = extract_minutiae(skeleton, mask)

    metric_rows = []
    for method, output in outputs.items():
        row = {"Method": method}
        row.update(evaluate_grayscale(reference, output, reference_mask, runtimes[method]))
        metric_rows.append(row)

    return {
        "reference": reference,
        "degraded": degraded,
        "outputs": outputs,
        "ridge_energy": ridge_energy,
        "raw_binary": raw_binary,
        "binary": binary,
        "mask": mask,
        "skeleton": skeleton,
        "endings": endings,
        "bifurcations": bifurcations,
        "metrics": pd.DataFrame(metric_rows),
    }


sample_result = run_comparison(sample_reference, severity=0.55, seed=RANDOM_SEED)
display(sample_result["metrics"].round(4))


# %% [markdown]
# ## 9. Processing dashboard
#
# The dashboard summarises the complete pipeline for one image. In Jupyter, the optional controls permit the image and degradation severity to be changed interactively.

# %%
def plot_individual_contributions(result, sample_title="Fingerprint enhancement result"):
    panels = [
        (result["outputs"][M1_LABEL], "M1 - Wiener + CLAHE", "gray"),
        (result["outputs"][M2_LABEL], "M2 - Gabor / Modified Gabor", "gray"),
        (result["binary"], "M3 - Morphological Ridge Restoration", "gray"),
        (result["skeleton"], "M4 - Thinning & Minutiae Extraction", "gray"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, (image, title, colourmap) in zip(axes.ravel(), panels):
        axis.imshow(image, cmap=colourmap, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"Individual Image Processing Contributions - {sample_title}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_team_hybrid_evaluation(result, sample_title="Fingerprint enhancement result"):
    panels = [
        (result["reference"], "Clean Reference", "gray"),
        (result["degraded"], DEGRADED_INPUT_LABEL, "gray"),
        (result["outputs"][M1_LABEL], "M1 Stage", "gray"),
        (result["outputs"][M2_LABEL], "M2 Stage", "gray"),
        (result["outputs"][TEAM_HYBRID_LABEL], TEAM_HYBRID_LABEL, "gray"),
        (result["skeleton"], "M4 Feature Output", "gray"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for axis, (image, title, colourmap) in zip(axes.ravel(), panels):
        axis.imshow(image, cmap=colourmap, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"Team Hybrid Evaluation - {sample_title}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_pipeline_dashboard(result, sample_title="Fingerprint enhancement result"):
    """Backward-compatible alias for the corrected individual-contribution overview."""
    return plot_individual_contributions(result, sample_title=sample_title)


dashboard_figure = plot_individual_contributions(sample_result, sample_title=sample_name)
plt.show()


try:
    import ipywidgets as widgets
    from IPython.display import clear_output

    def interactive_dashboard(image_index=0, severity=0.55):
        clear_output(wait=True)
        image_index = int(np.clip(image_index, 0, len(reference_set) - 1))
        name, reference = reference_set[image_index]
        result = run_comparison(reference, severity=float(severity), seed=RANDOM_SEED + image_index)
        display(result["metrics"].round(4))
        plot_individual_contributions(result, sample_title=name)
        plt.show()

    widgets.interact(
        interactive_dashboard,
        image_index=widgets.IntSlider(value=0, min=0, max=max(0, len(reference_set) - 1), step=1),
        severity=widgets.FloatSlider(value=0.55, min=0.2, max=0.9, step=0.05),
    )
except Exception:
    print("ipywidgets is optional; run plot_individual_contributions(sample_result) for the static dashboard.")


# %% [markdown]
# ## 10. Batch experiment and critical comparison
#
# Every method is tested on the same reference images and degradation settings. The summary uses the mean and standard deviation across the active batch.
#
# Development mode uses a small configurable sample for debugging and fast experimentation. Final evaluation mode is configured separately and must be intentionally reviewed before any larger run.
#
# No weighted overall score is created here. MSE, PSNR, SSIM and ridge coherence are interpreted separately so that real trade-offs remain visible. If Global HE achieves the best PSNR or MSE, that result should remain in the comparison.
#

# %%
def select_reference_batch(references, max_images=MAX_BATCH_IMAGES):
    if max_images is None:
        return list(references)
    return list(references[:max(0, int(max_images))])


def run_batch_experiment(references, severities=(0.35, 0.55, 0.75), progress=True):
    rows = []
    total = len(references)
    for index, (name, reference) in enumerate(references):
        severity = float(severities[index % len(severities)])
        result = run_comparison(reference, severity=severity, seed=RANDOM_SEED + index)
        frame = result["metrics"].copy()
        frame.insert(0, "Image", name)
        frame.insert(1, "Severity", severity)
        frame["Ridge endings"] = np.nan
        frame["Bifurcations"] = np.nan
        feature_row = frame["Method"] == TEAM_HYBRID_LABEL
        frame.loc[feature_row, "Ridge endings"] = len(result["endings"])
        frame.loc[feature_row, "Bifurcations"] = len(result["bifurcations"])
        rows.append(frame)
        if progress:
            print(f"Processed {index + 1:02d}/{total:02d}: {name}")
    return pd.concat(rows, ignore_index=True)


active_batch_references = select_reference_batch(reference_set, MAX_BATCH_IMAGES)
print(
    f"Active batch: {len(active_batch_references)} image(s) "
    f"({'development' if DEVELOPMENT_MODE else 'final evaluation'} mode)"
)

batch_start = perf_counter()
batch_metrics = normalise_method_labels(run_batch_experiment(active_batch_references))
batch_total_runtime_seconds = perf_counter() - batch_start
print(f"Batch runtime: {batch_total_runtime_seconds:.2f} seconds")
summary_metrics = (
    batch_metrics.groupby("Method", as_index=False)
    .agg(
        MSE_mean=("MSE", "mean"),
        MSE_std=("MSE", "std"),
        PSNR_mean=("PSNR (dB)", "mean"),
        PSNR_std=("PSNR (dB)", "std"),
        SSIM_mean=("SSIM", "mean"),
        SSIM_std=("SSIM", "std"),
        Coherence_mean=("Coherence", "mean"),
        Coherence_std=("Coherence", "std"),
        Match_mean=("Match score (%)", "mean"),
        Match_std=("Match score (%)", "std"),
        Runtime_mean=("Runtime (s)", "mean"),
        Runtime_std=("Runtime (s)", "std"),
    )
)

method_order = METHOD_ORDER
summary_metrics["Method"] = pd.Categorical(summary_metrics["Method"], categories=method_order, ordered=True)
summary_metrics = summary_metrics.sort_values("Method").reset_index(drop=True)
summary_metrics["Method"] = summary_metrics["Method"].astype(str)

enhancement_summary = summary_metrics[~summary_metrics["Method"].isin(CONTROL_METHODS)].copy()
metric_leaders = pd.DataFrame(
    [
        {
            "Metric": "MSE",
            "Direction": "lower is better",
            "Best method": enhancement_summary.loc[enhancement_summary["MSE_mean"].idxmin(), "Method"],
            "Value": enhancement_summary["MSE_mean"].min(),
        },
        {
            "Metric": "PSNR (dB)",
            "Direction": "higher is better",
            "Best method": enhancement_summary.loc[enhancement_summary["PSNR_mean"].idxmax(), "Method"],
            "Value": enhancement_summary["PSNR_mean"].max(),
        },
        {
            "Metric": "SSIM",
            "Direction": "higher is better",
            "Best method": enhancement_summary.loc[enhancement_summary["SSIM_mean"].idxmax(), "Method"],
            "Value": enhancement_summary["SSIM_mean"].max(),
        },
        {
            "Metric": "Coherence",
            "Direction": "higher is better",
            "Best method": enhancement_summary.loc[enhancement_summary["Coherence_mean"].idxmax(), "Method"],
            "Value": enhancement_summary["Coherence_mean"].max(),
        },
    ]
)

print("Batch summary (mean and standard deviation):")
display(summary_metrics.round(4))
print("Metric-specific leaders; no weighted overall score is used:")
display(metric_leaders.round(4))

member4_feature_rows = batch_metrics[batch_metrics["Method"] == TEAM_HYBRID_LABEL]
member4_feature_summary = pd.DataFrame(
    [
        {
            "Metric": "Average detected ridge endings",
            "Value": member4_feature_rows["Ridge endings"].mean(),
        },
        {
            "Metric": "Average detected bifurcations",
            "Value": member4_feature_rows["Bifurcations"].mean(),
        },
    ]
)
print("Member 4 detected minutiae counts from the Team Hybrid Pipeline output:")
display(member4_feature_summary.round(2))


# %% [markdown]
# ## 10. Advanced 24-Image Development Candidates
#
# The following experimental block is deliberately limited to the 24-image development subset. It does not overwrite
# the preserved 500-image `batch_metrics.csv`, `summary_metrics.csv`, PDF report or final metadata.
#
# Implemented candidates:
#
# - **Candidate A:** restoration branch, ROI mask, wavelet denoising, Wiener/deconvolution restoration, local
#   normalisation, local orientation/frequency estimation, adaptive Gabor filtering and confidence-guided fusion.
# - **Candidate B:** restoration branch, local normalisation, overlapping STFT/contextual frequency filtering and
#   confidence-guided fusion.
# - **Candidate C:** restoration branch, orientation field, coherence-guided anisotropic diffusion, local ridge
#   frequency estimation, Log-Gabor filtering and confidence-guided fusion.
#
# The clean reference is used only in the evaluation and parameter-validation tables below, not inside enhancement
# functions.
#

# %%
ADVANCED_PARAMETER_CONFIGS = [
    {
        "config_id": "psnr_preserving",
        "wavelet": "db2",
        "wavelet_level": 2,
        "wavelet_threshold_scale": 0.08,
        "wavelet_threshold_strategy": "soft",
        "local_wiener_window": 5,
        "local_wiener_blend": 0.18,
        "psf_sigma": 1.05,
        "deconv_balance": 0.10,
        "deconv_blend": 0.04,
        "local_norm_window": 45,
        "local_norm_contrast": 0.10,
        "local_norm_blend": 0.05,
        "stft_local_norm_blend": 0.04,
        "orientation_smoothing_sigma": 3.0,
        "orientation_vector_smoothing_sigma": 2.0,
        "adaptive_block_size": 48,
        "adaptive_orientation_bins": 4,
        "adaptive_frequency_bins": (0.07, 0.10, 0.13),
        "adaptive_gabor_bandwidth": 1.9,
        "adaptive_fusion_strength": 0.035,
        "stft_block_size": 56,
        "stft_overlap": 28,
        "stft_radial_bandwidth": 0.028,
        "stft_directional_bandwidth": np.pi / 7.0,
        "stft_gain": 0.22,
        "stft_fusion_strength": 0.035,
        "diffusion_iterations": 1,
        "diffusion_step": 0.14,
        "log_gabor_sigma_on_frequency": 0.62,
        "log_gabor_angular_sigma": np.pi / 6.0,
        "log_gabor_fusion_strength": 0.035,
        "team_fusion_strength": 0.05,
        "confidence_threshold": 0.20,
        "frequency_min": 0.045,
        "frequency_max": 0.18,
        "default_frequency": 0.10,
        "ordinary_gabor_frequency": 0.115,
        "ordinary_gabor_orientations": 8,
        "ordinary_gabor_blend": 0.35,
        "ordinary_gabor_clahe_clip": 0.012,
    },
    {
        "config_id": "balanced",
        "wavelet": "sym4",
        "wavelet_level": 2,
        "wavelet_threshold_scale": 0.14,
        "wavelet_threshold_strategy": "level_scaled",
        "local_wiener_window": 5,
        "local_wiener_blend": 0.22,
        "psf_sigma": 1.15,
        "deconv_balance": 0.08,
        "deconv_blend": 0.07,
        "local_norm_window": 41,
        "local_norm_contrast": 0.12,
        "local_norm_blend": 0.08,
        "stft_local_norm_blend": 0.07,
        "orientation_smoothing_sigma": 3.0,
        "orientation_vector_smoothing_sigma": 2.0,
        "adaptive_block_size": 40,
        "adaptive_orientation_bins": 6,
        "adaptive_frequency_bins": (0.065, 0.095, 0.125, 0.155),
        "adaptive_gabor_bandwidth": 1.7,
        "adaptive_fusion_strength": 0.07,
        "stft_block_size": 48,
        "stft_overlap": 24,
        "stft_radial_bandwidth": 0.026,
        "stft_directional_bandwidth": np.pi / 8.0,
        "stft_gain": 0.35,
        "stft_fusion_strength": 0.06,
        "diffusion_iterations": 2,
        "diffusion_step": 0.18,
        "log_gabor_sigma_on_frequency": 0.60,
        "log_gabor_angular_sigma": np.pi / 7.0,
        "log_gabor_fusion_strength": 0.06,
        "team_fusion_strength": 0.08,
        "confidence_threshold": 0.25,
        "frequency_min": 0.045,
        "frequency_max": 0.18,
        "default_frequency": 0.10,
        "ordinary_gabor_frequency": 0.115,
        "ordinary_gabor_orientations": 8,
        "ordinary_gabor_blend": 0.40,
        "ordinary_gabor_clahe_clip": 0.012,
    },
    {
        "config_id": "ridge_coherence",
        "wavelet": "coif2",
        "wavelet_level": 2,
        "wavelet_threshold_scale": 0.10,
        "wavelet_threshold_strategy": "soft",
        "local_wiener_window": 7,
        "local_wiener_blend": 0.26,
        "psf_sigma": 1.25,
        "deconv_balance": 0.06,
        "deconv_blend": 0.08,
        "local_norm_window": 33,
        "local_norm_contrast": 0.15,
        "local_norm_blend": 0.14,
        "stft_local_norm_blend": 0.12,
        "orientation_smoothing_sigma": 2.5,
        "orientation_vector_smoothing_sigma": 1.8,
        "adaptive_block_size": 32,
        "adaptive_orientation_bins": 6,
        "adaptive_frequency_bins": (0.06, 0.09, 0.12, 0.15),
        "adaptive_gabor_bandwidth": 1.4,
        "adaptive_fusion_strength": 0.12,
        "stft_block_size": 40,
        "stft_overlap": 24,
        "stft_radial_bandwidth": 0.024,
        "stft_directional_bandwidth": np.pi / 9.0,
        "stft_gain": 0.55,
        "stft_fusion_strength": 0.10,
        "diffusion_iterations": 3,
        "diffusion_step": 0.22,
        "log_gabor_sigma_on_frequency": 0.56,
        "log_gabor_angular_sigma": np.pi / 8.0,
        "log_gabor_fusion_strength": 0.10,
        "team_fusion_strength": 0.12,
        "confidence_threshold": 0.18,
        "frequency_min": 0.045,
        "frequency_max": 0.18,
        "default_frequency": 0.10,
        "ordinary_gabor_frequency": 0.115,
        "ordinary_gabor_orientations": 8,
        "ordinary_gabor_blend": 0.45,
        "ordinary_gabor_clahe_clip": 0.012,
    },
]


def _advanced_stage_rows(reference, candidate_label, stages, reference_mask, config_id):
    rows = []
    for stage_order, (stage_name, image) in enumerate(stages.items(), start=1):
        row = {
            "Config ID": config_id,
            "Candidate": candidate_label,
            "Stage order": stage_order,
            "Stage": stage_name,
        }
        row.update(evaluate_grayscale(reference, image, reference_mask))
        rows.append(row)
    return rows


def run_advanced_comparison(reference, severity=0.55, seed=RANDOM_SEED, config=None):
    """Run advanced candidates on one controlled degraded image without using the clean reference for enhancement."""
    config = {} if config is None else dict(config)
    degraded = simulate_degradation(reference, severity=severity, seed=seed)
    reference_mask = fingerprint_mask(reference)

    outputs = {DEGRADED_INPUT_LABEL: degraded}
    runtimes = {DEGRADED_INPUT_LABEL: 0.0}
    stage_rows = []

    start = perf_counter()
    outputs[CLAHE_BASELINE_LABEL] = member1_clahe_baseline(degraded)
    runtimes[CLAHE_BASELINE_LABEL] = perf_counter() - start

    start = perf_counter()
    ordinary_gabor_input = apply_clahe(degraded, clip_limit=config.get("ordinary_gabor_clahe_clip", 0.012))
    ordinary_gabor, _ = multi_orientation_gabor(
        ordinary_gabor_input,
        frequency=config.get("ordinary_gabor_frequency", 0.115),
        orientations=config.get("ordinary_gabor_orientations", 8),
        blend=config.get("ordinary_gabor_blend", 0.35),
    )
    outputs[M2_BASELINE_LABEL] = ordinary_gabor
    runtimes[M2_BASELINE_LABEL] = perf_counter() - start

    start = perf_counter()
    restored, restoration_stages = wavelet_wiener_restoration(degraded, config)
    restoration_runtime = perf_counter() - start
    outputs[M1_PROPOSED_LABEL] = restored
    runtimes[M1_PROPOSED_LABEL] = restoration_runtime

    start = perf_counter()
    candidate_a, candidate_a_stages, candidate_a_aux = candidate_a_restoration_adaptive_gabor_fusion(
        degraded,
        restored=restored,
        restoration_stages=restoration_stages,
        config=config,
    )
    candidate_a_runtime = restoration_runtime + perf_counter() - start
    outputs[M2_PROPOSED_LABEL] = candidate_a_stages["Adaptive Gabor"]
    outputs[CANDIDATE_A_LABEL] = candidate_a
    runtimes[M2_PROPOSED_LABEL] = candidate_a_runtime
    runtimes[CANDIDATE_A_LABEL] = candidate_a_runtime
    stage_rows.extend(_advanced_stage_rows(reference, CANDIDATE_A_LABEL, candidate_a_stages, reference_mask, config["config_id"]))

    start = perf_counter()
    candidate_b, candidate_b_stages, candidate_b_aux = candidate_b_stft_contextual_enhancement(
        degraded,
        restored=restored,
        restoration_stages=restoration_stages,
        config=config,
    )
    candidate_b_runtime = restoration_runtime + perf_counter() - start
    outputs[CANDIDATE_B_LABEL] = candidate_b
    runtimes[CANDIDATE_B_LABEL] = candidate_b_runtime
    stage_rows.extend(_advanced_stage_rows(reference, CANDIDATE_B_LABEL, candidate_b_stages, reference_mask, config["config_id"]))

    start = perf_counter()
    candidate_c, candidate_c_stages, candidate_c_aux = candidate_c_coherence_diffusion_log_gabor(
        degraded,
        restored=restored,
        restoration_stages=restoration_stages,
        config=config,
    )
    candidate_c_runtime = restoration_runtime + perf_counter() - start
    outputs[CANDIDATE_C_LABEL] = candidate_c
    runtimes[CANDIDATE_C_LABEL] = candidate_c_runtime
    stage_rows.extend(_advanced_stage_rows(reference, CANDIDATE_C_LABEL, candidate_c_stages, reference_mask, config["config_id"]))

    start = perf_counter()
    team_output, team_weights = quality_guided_team_fusion(
        restored,
        [
            (CANDIDATE_A_LABEL, candidate_a),
            (CANDIDATE_B_LABEL, candidate_b),
            (CANDIDATE_C_LABEL, candidate_c),
        ],
        mask=candidate_a_aux["mask"],
        fusion_strength=config.get("team_fusion_strength", 0.05),
    )
    team_runtime = restoration_runtime + perf_counter() - start
    outputs[TEAM_PROPOSED_LABEL] = team_output
    runtimes[TEAM_PROPOSED_LABEL] = team_runtime

    binary, mask, raw_binary = segment_and_restore_ridges(team_output, return_intermediate=True)
    skeleton = thin_ridges(binary)
    endings, bifurcations = extract_minutiae(skeleton, mask)

    metric_rows = []
    for method, output in outputs.items():
        row = {"Method": method}
        row.update(evaluate_grayscale(reference, output, reference_mask, runtimes[method]))
        metric_rows.append(row)

    return {
        "reference": reference,
        "degraded": degraded,
        "outputs": outputs,
        "metrics": pd.DataFrame(metric_rows),
        "stage_metrics": pd.DataFrame(stage_rows),
        "team_weights": team_weights,
        "raw_binary": raw_binary,
        "binary": binary,
        "mask": mask,
        "skeleton": skeleton,
        "endings": endings,
        "bifurcations": bifurcations,
    }


def run_advanced_development_search(references, parameter_configs=ADVANCED_PARAMETER_CONFIGS, progress=True):
    """Evaluate each advanced parameter configuration on the same 24-image development subset."""
    rows = []
    stage_rows = []
    feature_rows = []
    active_references = select_reference_batch(references, DEVELOPMENT_SAMPLE_IMAGES)
    for config in parameter_configs:
        config_id = config["config_id"]
        if progress:
            print(f"Advanced config: {config_id}")
        for index, (name, reference) in enumerate(active_references):
            severity = [0.35, 0.55, 0.75][index % 3]
            result = run_advanced_comparison(reference, severity=severity, seed=RANDOM_SEED + index, config=config)
            frame = result["metrics"].copy()
            frame.insert(0, "Config ID", config_id)
            frame.insert(1, "Image", name)
            frame.insert(2, "Severity", severity)
            rows.append(frame)

            stage_frame = result["stage_metrics"].copy()
            stage_frame.insert(1, "Image", name)
            stage_frame.insert(2, "Severity", severity)
            stage_rows.append(stage_frame)

            feature_rows.append(
                {
                    "Config ID": config_id,
                    "Image": name,
                    "Severity": severity,
                    "Ridge endings": len(result["endings"]),
                    "Bifurcations": len(result["bifurcations"]),
                }
            )
            if progress:
                print(f"  Processed {index + 1:02d}/{len(active_references):02d}: {name}")
    return (
        pd.concat(rows, ignore_index=True),
        pd.concat(stage_rows, ignore_index=True),
        pd.DataFrame(feature_rows),
    )


def summarise_advanced_metrics(metrics):
    summary = (
        metrics.groupby(["Config ID", "Method"], as_index=False)
        .agg(
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            PSNR_mean=("PSNR (dB)", "mean"),
            PSNR_std=("PSNR (dB)", "std"),
            SSIM_mean=("SSIM", "mean"),
            SSIM_std=("SSIM", "std"),
            Coherence_mean=("Coherence", "mean"),
            Coherence_std=("Coherence", "std"),
            Match_mean=("Match score (%)", "mean"),
            Match_std=("Match score (%)", "std"),
            Runtime_mean=("Runtime (s)", "mean"),
            Runtime_std=("Runtime (s)", "std"),
        )
    )
    degraded_psnr = summary[summary["Method"] == DEGRADED_INPUT_LABEL].set_index("Config ID")["PSNR_mean"]
    summary["PSNR improvement vs degraded (dB)"] = summary.apply(
        lambda row: row["PSNR_mean"] - float(degraded_psnr.loc[row["Config ID"]]),
        axis=1,
    )
    summary["PSNR gap to 28.17 dB"] = LITERATURE_PSNR_BENCHMARK_DB - summary["PSNR_mean"]
    summary["PSNR delta to 28.17 dB"] = summary["PSNR_mean"] - LITERATURE_PSNR_BENCHMARK_DB
    return summary


def summarise_advanced_stages(stage_metrics):
    stage_summary = (
        stage_metrics.groupby(["Config ID", "Candidate", "Stage order", "Stage"], as_index=False)
        .agg(
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            PSNR_mean=("PSNR (dB)", "mean"),
            PSNR_std=("PSNR (dB)", "std"),
            SSIM_mean=("SSIM", "mean"),
            SSIM_std=("SSIM", "std"),
            Coherence_mean=("Coherence", "mean"),
            Coherence_std=("Coherence", "std"),
        )
    )
    degraded_stage = stage_summary[stage_summary["Stage"] == "Degraded"].set_index(["Config ID", "Candidate"])["PSNR_mean"]
    stage_summary["PSNR improvement vs degraded (dB)"] = stage_summary.apply(
        lambda row: row["PSNR_mean"] - float(degraded_stage.loc[(row["Config ID"], row["Candidate"])]),
        axis=1,
    )
    return stage_summary


def select_advanced_config(search_summary):
    candidate_methods = [
        M1_PROPOSED_LABEL,
        M2_PROPOSED_LABEL,
        CANDIDATE_A_LABEL,
        CANDIDATE_B_LABEL,
        CANDIDATE_C_LABEL,
        TEAM_PROPOSED_LABEL,
    ]
    candidates = search_summary[search_summary["Method"].isin(candidate_methods)].copy()
    best_row = candidates.loc[candidates["PSNR_mean"].idxmax()]
    return str(best_row["Config ID"]), best_row


def build_metric_rankings(summary):
    methods = [
        CLAHE_BASELINE_LABEL,
        M1_PROPOSED_LABEL,
        M2_BASELINE_LABEL,
        M2_PROPOSED_LABEL,
        CANDIDATE_A_LABEL,
        CANDIDATE_B_LABEL,
        CANDIDATE_C_LABEL,
        TEAM_PROPOSED_LABEL,
    ]
    frame = summary[summary["Method"].isin(methods)].copy()
    ranking_frames = []
    for metric, ascending in [("PSNR_mean", False), ("SSIM_mean", False), ("Coherence_mean", False)]:
        ranking = frame.sort_values(metric, ascending=ascending).reset_index(drop=True)
        ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
        ranking.insert(1, "Metric", metric.replace("_mean", ""))
        ranking = ranking[["Metric", "Rank", "Method", metric]]
        ranking = ranking.rename(columns={metric: "Value"})
        ranking_frames.append(ranking)
    return pd.concat(ranking_frames, ignore_index=True)


def _format_key_table(frame, columns, max_rows=None):
    display_frame = frame[columns].copy()
    if max_rows is not None:
        display_frame = display_frame.head(max_rows)
    return display_frame.round(4).to_string(index=False)


def create_advanced_development_report(summary, stage_summary, search_summary, rankings, selected_config_id, selected_config_row):
    candidate_methods = [
        M1_PROPOSED_LABEL,
        M2_PROPOSED_LABEL,
        CANDIDATE_A_LABEL,
        CANDIDATE_B_LABEL,
        CANDIDATE_C_LABEL,
        TEAM_PROPOSED_LABEL,
    ]
    summary_candidates = summary[summary["Method"].isin(candidate_methods + [CLAHE_BASELINE_LABEL, M2_BASELINE_LABEL])]
    best_psnr_row = summary_candidates.loc[summary_candidates["PSNR_mean"].idxmax()]
    best_coherence_row = summary_candidates.loc[summary_candidates["Coherence_mean"].idxmax()]
    candidate_a_row = summary[summary["Method"] == CANDIDATE_A_LABEL].iloc[0]
    candidate_b_row = summary[summary["Method"] == CANDIDATE_B_LABEL].iloc[0]
    candidate_c_row = summary[summary["Method"] == CANDIDATE_C_LABEL].iloc[0]

    lines = [
        "# Advanced 24-Image Development Experiment",
        "",
        "This diagnostic run uses the fixed 24-image development subset only. Existing 500-image evidence files are preserved.",
        "",
        "## A. Candidate architectures implemented",
        "",
        "- Candidate A: ROI segmentation, wavelet denoising, Wiener/deconvolution restoration, local normalisation, local orientation/frequency estimation, adaptive Gabor filtering and confidence-guided fusion.",
        "- Candidate B: overlapping STFT/contextual enhancement with dominant ridge orientation/frequency estimation and restoration-branch fusion.",
        "- Candidate C: orientation field, coherence-guided anisotropic diffusion, local ridge frequency estimation, Log-Gabor filtering and confidence-guided fusion.",
        "- Team proposed: quality/coherence-guided fusion of Candidates A, B and C, blended conservatively with the restoration branch.",
        "",
        "## Selected parameter configuration",
        "",
        f"Selected config: `{selected_config_id}`. Selection was based on mean PSNR over the 24-image development subset, not per-image tuning.",
        "",
        "## B. Stage-by-stage PSNR",
        "",
        "```text",
        _format_key_table(
            stage_summary,
            [
                "Candidate",
                "Stage",
                "PSNR_mean",
                "PSNR_std",
                "SSIM_mean",
                "Coherence_mean",
                "PSNR improvement vs degraded (dB)",
            ],
        ),
        "```",
        "",
        "## C. Member 1 baseline vs proposed",
        "",
        "```text",
        _format_key_table(
            summary[summary["Method"].isin([CLAHE_BASELINE_LABEL, M1_PROPOSED_LABEL])],
            [
                "Method",
                "MSE_mean",
                "MSE_std",
                "PSNR_mean",
                "PSNR_std",
                "SSIM_mean",
                "SSIM_std",
                "Coherence_mean",
                "Coherence_std",
                "PSNR improvement vs degraded (dB)",
                "PSNR gap to 28.17 dB",
                "PSNR delta to 28.17 dB",
            ],
        ),
        "```",
        "",
        "## D. Member 2 baseline vs proposed",
        "",
        "```text",
        _format_key_table(
            summary[summary["Method"].isin([M2_BASELINE_LABEL, M2_PROPOSED_LABEL])],
            [
                "Method",
                "MSE_mean",
                "MSE_std",
                "PSNR_mean",
                "PSNR_std",
                "SSIM_mean",
                "SSIM_std",
                "Coherence_mean",
                "Coherence_std",
                "PSNR improvement vs degraded (dB)",
                "PSNR gap to 28.17 dB",
                "PSNR delta to 28.17 dB",
            ],
        ),
        "```",
        "",
        "## E. Candidate A results",
        "",
        f"Mean PSNR: {candidate_a_row['PSNR_mean']:.4f} dB; signed gap to 28.17 dB: {candidate_a_row['PSNR delta to 28.17 dB']:+.4f} dB.",
        "",
        "## F. Candidate B results",
        "",
        f"Mean PSNR: {candidate_b_row['PSNR_mean']:.4f} dB; signed gap to 28.17 dB: {candidate_b_row['PSNR delta to 28.17 dB']:+.4f} dB.",
        "",
        "## G. Candidate C results",
        "",
        f"Mean PSNR: {candidate_c_row['PSNR_mean']:.4f} dB; signed gap to 28.17 dB: {candidate_c_row['PSNR delta to 28.17 dB']:+.4f} dB.",
        "",
        "## H. Best PSNR candidate",
        "",
        f"{best_psnr_row['Method']} at {best_psnr_row['PSNR_mean']:.4f} dB.",
        "",
        "## I. Best ridge-coherence candidate",
        "",
        f"{best_coherence_row['Method']} at coherence {best_coherence_row['Coherence_mean']:.4f}.",
        "",
        "## J. Current best mean PSNR",
        "",
        f"{best_psnr_row['PSNR_mean']:.4f} dB.",
        "",
        "## K. Gap to 28.17 dB",
        "",
        f"{best_psnr_row['PSNR_mean'] - LITERATURE_PSNR_BENCHMARK_DB:+.4f} dB.",
        "",
        "## Candidate summary",
        "",
        "```text",
        _format_key_table(
            summary_candidates,
            [
                "Method",
                "MSE_mean",
                "MSE_std",
                "PSNR_mean",
                "PSNR_std",
                "SSIM_mean",
                "SSIM_std",
                "Coherence_mean",
                "Coherence_std",
                "PSNR improvement vs degraded (dB)",
                "PSNR gap to 28.17 dB",
                "PSNR delta to 28.17 dB",
            ],
        ),
        "```",
        "",
        "## Metric rankings",
        "",
        "```text",
        rankings.round(4).to_string(index=False),
        "```",
        "",
        "## Parameter search",
        "",
        "```text",
        _format_key_table(
            search_summary.sort_values(["Method", "PSNR_mean"], ascending=[True, False]),
            [
                "Config ID",
                "Method",
                "PSNR_mean",
                "SSIM_mean",
                "Coherence_mean",
                "PSNR improvement vs degraded (dB)",
                "PSNR delta to 28.17 dB",
            ],
        ),
        "```",
        "",
        "## M. Recommended next modification",
        "",
        "The stage diagnostics should decide the next move: if restoration improves PSNR but adaptive ridge stages reduce it, reduce fusion strength further or blend by a stricter confidence mask. If all restoration stages remain far below 28.17 dB, the limiting factor is likely the synthetic degradation severity/noise model rather than the Gabor/STFT stage alone.",
    ]
    return "\n".join(lines)


advanced_batch_metrics = pd.DataFrame()
advanced_stage_metrics = pd.DataFrame()
advanced_feature_metrics = pd.DataFrame()
advanced_search_summary = pd.DataFrame()
advanced_summary_metrics = pd.DataFrame()
advanced_stage_summary = pd.DataFrame()
advanced_rankings = pd.DataFrame()
advanced_selected_config_id = None
advanced_selected_config_row = None
advanced_development_report = ""

if RUN_ADVANCED_DEVELOPMENT_SEARCH:
    advanced_start = perf_counter()
    advanced_batch_metrics, advanced_stage_metrics, advanced_feature_metrics = run_advanced_development_search(
        reference_set,
        parameter_configs=ADVANCED_PARAMETER_CONFIGS,
        progress=True,
    )
    advanced_total_runtime_seconds = perf_counter() - advanced_start
    advanced_search_summary = summarise_advanced_metrics(advanced_batch_metrics)
    advanced_selected_config_id, advanced_selected_config_row = select_advanced_config(advanced_search_summary)

    selected_metrics = advanced_batch_metrics[advanced_batch_metrics["Config ID"] == advanced_selected_config_id].copy()
    advanced_summary_metrics = summarise_advanced_metrics(selected_metrics)
    advanced_summary_metrics["Method"] = pd.Categorical(
        advanced_summary_metrics["Method"],
        categories=ADVANCED_DEVELOPMENT_METHOD_ORDER,
        ordered=True,
    )
    advanced_summary_metrics = advanced_summary_metrics.sort_values("Method").reset_index(drop=True)
    advanced_summary_metrics["Method"] = advanced_summary_metrics["Method"].astype(str)

    selected_stage_metrics = advanced_stage_metrics[advanced_stage_metrics["Config ID"] == advanced_selected_config_id].copy()
    advanced_stage_summary = summarise_advanced_stages(selected_stage_metrics)
    advanced_stage_summary = advanced_stage_summary.sort_values(["Candidate", "Stage order"]).reset_index(drop=True)
    advanced_rankings = build_metric_rankings(advanced_summary_metrics)
    advanced_development_report = create_advanced_development_report(
        advanced_summary_metrics,
        advanced_stage_summary,
        advanced_search_summary,
        advanced_rankings,
        advanced_selected_config_id,
        advanced_selected_config_row,
    )

    print(f"Advanced development runtime: {advanced_total_runtime_seconds:.2f} seconds")
    print(f"Selected advanced config: {advanced_selected_config_id}")
    print("Advanced 24-image candidate summary:")
    display(advanced_summary_metrics.round(4))
    print("Advanced stage-by-stage summary:")
    display(advanced_stage_summary.round(4))
    print("Advanced metric rankings:")
    display(advanced_rankings.round(4))

    if EXPORT_ADVANCED_DEVELOPMENT_RESULTS:
        advanced_batch_csv = OUTPUT_DIR / "advanced_dev_batch_metrics.csv"
        advanced_stage_csv = OUTPUT_DIR / "advanced_dev_stage_metrics.csv"
        advanced_summary_csv = OUTPUT_DIR / "advanced_dev_summary_metrics.csv"
        advanced_search_csv = OUTPUT_DIR / "advanced_dev_parameter_search.csv"
        advanced_feature_csv = OUTPUT_DIR / "advanced_dev_minutiae_counts.csv"
        advanced_metadata_json = OUTPUT_DIR / "advanced_dev_metadata.json"
        advanced_report_md = OUTPUT_DIR / "advanced_development_report.md"

        advanced_batch_metrics.to_csv(advanced_batch_csv, index=False)
        advanced_stage_metrics.to_csv(advanced_stage_csv, index=False)
        advanced_summary_metrics.to_csv(advanced_summary_csv, index=False)
        advanced_search_summary.to_csv(advanced_search_csv, index=False)
        advanced_feature_metrics.to_csv(advanced_feature_csv, index=False)
        advanced_report_md.write_text(advanced_development_report, encoding="utf-8")

        advanced_metadata = {
            "dataset_mode": "synthetic_demo" if USING_DEMO_DATA else "SOCOFing_real_controlled_degradation",
            "development_subset_images": DEVELOPMENT_SAMPLE_IMAGES,
            "active_advanced_images": int(advanced_batch_metrics["Image"].nunique()),
            "parameter_configurations_tested": [config["config_id"] for config in ADVANCED_PARAMETER_CONFIGS],
            "selected_config_id": advanced_selected_config_id,
            "selection_rule": "Highest mean PSNR over the 24-image development subset; no per-image tuning.",
            "literature_psnr_benchmark_db": LITERATURE_PSNR_BENCHMARK_DB,
            "advanced_runtime_seconds": float(advanced_total_runtime_seconds),
            "pywavelets_available": pywt is not None,
            "outputs": [
                advanced_batch_csv.name,
                advanced_stage_csv.name,
                advanced_summary_csv.name,
                advanced_search_csv.name,
                advanced_feature_csv.name,
                advanced_metadata_json.name,
                advanced_report_md.name,
            ],
            "preserved_500_image_files": [
                "batch_metrics.csv",
                "summary_metrics.csv",
                "fingerprint_enhancement_report.pdf",
                "experiment_metadata.json",
                "example_minutiae_overlay.png",
            ],
        }
        advanced_metadata_json.write_text(json.dumps(advanced_metadata, indent=2), encoding="utf-8")
        print("Exported advanced development files without overwriting preserved 500-image evidence.")
else:
    print("Advanced development search is disabled.")


# %%
metric_plot_columns = ["MSE", "PSNR (dB)", "SSIM", "Coherence"]


def _ordered_metric_data(metrics, methods):
    frame = normalise_method_labels(metrics)
    frame = frame[frame["Method"].isin(methods)].copy()
    frame["Method"] = pd.Categorical(frame["Method"], categories=methods, ordered=True)
    return frame.sort_values("Method")


def plot_enhancement_quality_comparison(metrics, methods=ENHANCEMENT_QUALITY_METHODS, title="Enhancement Quality Comparison"):
    frame = _ordered_metric_data(metrics, methods)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, metric in zip(axes.ravel(), metric_plot_columns):
        sns.barplot(
            data=frame,
            x="Method",
            y=metric,
            hue="Method",
            order=methods,
            hue_order=methods,
            legend=False,
            errorbar="sd",
            ax=axis,
        )
        axis.set_title(f"Mean {metric} with standard deviation")
        axis.set_xlabel("")
        wrapped_labels = [textwrap.fill(label.get_text(), 18) for label in axis.get_xticklabels()]
        axis.set_xticks(axis.get_xticks())
        axis.set_xticklabels(wrapped_labels, rotation=0, ha="center")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_member1_enhancement_study(metrics):
    return plot_enhancement_quality_comparison(
        metrics,
        methods=[CLAHE_BASELINE_LABEL, M1_LABEL],
        title="Member 1 Enhancement Study",
    )


def plot_member2_ridge_enhancement_study(metrics):
    return plot_enhancement_quality_comparison(
        metrics,
        methods=[M2_LABEL],
        title="Member 2 Ridge Enhancement Study",
    )


def plot_member3_ridge_restoration_study(result, sample_title="Fingerprint enhancement result"):
    panels = [
        (result["outputs"][TEAM_HYBRID_LABEL], "Input to M3", "gray"),
        (result["raw_binary"], "Before Morphology", "gray"),
        (result["binary"], M3_LABEL, "gray"),
        (result["mask"], "Fingerprint Mask", "gray"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, (image, title, colourmap) in zip(axes.ravel(), panels):
        axis.imshow(image, cmap=colourmap, vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(f"Member 3 Ridge Restoration Study - {sample_title}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_member4_feature_results(result, sample_title="Fingerprint enhancement result"):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(result["skeleton"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Thinning Skeleton")
    axes[0].axis("off")
    axes[1].imshow(result["skeleton"], cmap="gray", vmin=0, vmax=1)
    if len(result["endings"]):
        axes[1].scatter(result["endings"][:, 1], result["endings"][:, 0], s=24, facecolors="none", edgecolors="#00d084")
    if len(result["bifurcations"]):
        axes[1].scatter(result["bifurcations"][:, 1], result["bifurcations"][:, 0], s=28, marker="x", c="#ff3b30")
    axes[1].set_title("Minutiae Overlay")
    axes[1].axis("off")
    axes[2].bar(["Ridge endings", "Bifurcations"], [len(result["endings"]), len(result["bifurcations"])], color=["#00a76f", "#d92d20"])
    axes[2].set_title("Detected Minutiae Counts")
    axes[2].set_ylabel("Count")
    fig.suptitle(f"Member 4 Feature Extraction Results - {sample_title}", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


figure2 = plot_enhancement_quality_comparison(batch_metrics, method_order)
plt.show()

figure3 = plot_member1_enhancement_study(batch_metrics)
plt.show()

figure4 = plot_member2_ridge_enhancement_study(batch_metrics)
plt.show()

figure5 = plot_member3_ridge_restoration_study(sample_result, sample_title=sample_name)
plt.show()

figure6 = plot_member4_feature_results(sample_result, sample_title=sample_name)
plt.show()

figure7 = plot_team_hybrid_evaluation(sample_result, sample_title=sample_name)
plt.show()


# %% [markdown]
# ## 11. Runtime benchmark
#
# The runtime benchmark is a temporary development check used only to estimate practical final batch sizes. It should be run on a moderate sample such as 50 or 100 `Real` images, not on the full dataset.
#
# The benchmark reports total runtime, approximate runtime per image and estimated runtime for 500, 1000 and 6000 images. It does not optimise algorithms and does not export final CSV/PDF results.
#

# %%
def estimate_batch_runtime(total_seconds, sample_size, target_sizes=(500, 1000, 6000)):
    per_image = float(total_seconds) / max(1, int(sample_size))
    rows = [
        {
            "Images": int(target_size),
            "Estimated runtime (s)": per_image * int(target_size),
            "Estimated runtime (min)": per_image * int(target_size) / 60.0,
        }
        for target_size in target_sizes
    ]
    return per_image, pd.DataFrame(rows)


def run_runtime_benchmark(references, sample_size=RUNTIME_BENCHMARK_SAMPLE_SIZE):
    benchmark_references = select_reference_batch(references, min(int(sample_size), len(references)))
    start = perf_counter()
    benchmark_metrics = run_batch_experiment(benchmark_references, progress=False)
    total_seconds = perf_counter() - start
    per_image, estimates = estimate_batch_runtime(total_seconds, len(benchmark_references))
    summary = pd.DataFrame(
        [
            {
                "Benchmark sample size": len(benchmark_references),
                "Measured total runtime (s)": total_seconds,
                "Approx runtime per image (s)": per_image,
            }
        ]
    )
    return benchmark_metrics, summary, estimates


if RUN_RUNTIME_BENCHMARK:
    runtime_benchmark_metrics, runtime_benchmark_summary, runtime_benchmark_estimates = run_runtime_benchmark(
        reference_set,
        sample_size=RUNTIME_BENCHMARK_SAMPLE_SIZE,
    )
    display(runtime_benchmark_summary.round(4))
    display(runtime_benchmark_estimates.round(2))
else:
    print(
        "Runtime benchmark is configured but not run automatically. "
        "Set RUN_RUNTIME_BENCHMARK = True or call run_runtime_benchmark(reference_set, 50)."
    )


# %% [markdown]
# ## 12. Optional Extra Effort - SVM Synthetic Quality Classification
#
# The SVM section is frozen for the main classical image-processing cleanup. It is kept as a clearly separated optional extra-effort component and does not affect Member 1, Member 2, Member 3, Member 4, Team Hybrid Pipeline, PSNR/MSE/SSIM comparison or method interpretation.
#
# When intentionally enabled later, the SVM predicts **Good**, **Fair** or **Poor** quality from six image-quality features. Labels come from controlled degradation levels, so this model assesses degradation quality; it does not recognise fingerprint identity. Accuracy is only reported for the held-out controlled test set.
#

# %%
def image_entropy(image, bins=64):
    histogram, _ = np.histogram(normalise_image(image), bins=bins, range=(0, 1), density=False)
    probabilities = histogram.astype(float) / max(1, histogram.sum())
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def quality_features(image):
    image = normalise_image(image)
    mask = fingerprint_mask(image)
    foreground = image[mask] if np.any(mask) else image.ravel()
    laplacian_variance = float(np.var(filters.laplace(image)))
    gradient = np.hypot(filters.sobel_h(image), filters.sobel_v(image))
    return {
        "Intensity std": float(np.std(foreground)),
        "Entropy": image_entropy(foreground),
        "Laplacian variance": laplacian_variance,
        "Gradient mean": float(np.mean(gradient[mask])) if np.any(mask) else float(np.mean(gradient)),
        "Coherence": orientation_coherence(image, mask),
        "Foreground ratio": float(np.mean(mask)),
    }


def build_quality_dataset(references):
    quality_levels = [(0.20, "Good"), (0.50, "Fair"), (0.80, "Poor")]
    rows = []
    for index, (name, reference) in enumerate(references):
        for severity, label in quality_levels:
            degraded = simulate_degradation(reference, severity=severity, seed=1000 + 10 * index + int(severity * 10))
            row = {"Image": name, "Quality": label, "Severity": severity}
            row.update(quality_features(degraded))
            rows.append(row)
    return pd.DataFrame(rows)


quality_accuracy = np.nan
quality_data = pd.DataFrame()
feature_columns = [
    "Intensity std",
    "Entropy",
    "Laplacian variance",
    "Gradient mean",
    "Coherence",
    "Foreground ratio",
]

if RUN_OPTIONAL_SVM:
    quality_data = build_quality_dataset(reference_set)
    X_train, X_test, y_train, y_test = train_test_split(
        quality_data[feature_columns],
        quality_data["Quality"],
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=quality_data["Quality"],
    )

    quality_model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced"),
    )
    quality_model.fit(X_train, y_train)
    quality_predictions = quality_model.predict(X_test)
    quality_accuracy = accuracy_score(y_test, quality_predictions)

    print(f"Controlled test-set SVM accuracy: {quality_accuracy * 100:.2f}%")
    print(classification_report(y_test, quality_predictions, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        quality_predictions,
        labels=["Good", "Fair", "Poor"],
        cmap="Blues",
    )
    plt.title("SVM fingerprint-quality confusion matrix")
    plt.grid(False)
    plt.show()

    if SAVE_OPTIONAL_SVM_MODEL:
        joblib.dump({"model": quality_model, "features": feature_columns}, OUTPUT_DIR / "svm_quality_model.joblib")
    else:
        print("SVM model export disabled; existing svm_quality_model.joblib is preserved.")
else:
    print("Optional Extra Effort - SVM Synthetic Quality Classification is frozen for this cleanup task.")
    print("Set RUN_OPTIONAL_SVM = True later only when the team is ready to review the SVM separately.")


# %% [markdown]
# ## 13. Result export
#
# Final-looking CSV, PDF and metadata exports are disabled by default in this cleanup task. Set `EXPORT_FINAL_RESULTS = True` only after the team approves a final evaluation run.
#
# Runtime benchmark evidence should remain temporary/development evidence and should not overwrite final-looking output files.
#

# %%
def export_pdf_report(summary, example_result, destination, metrics=None):
    metrics = batch_metrics if metrics is None else normalise_method_labels(metrics)
    summary = normalise_method_labels(summary)
    with PdfPages(destination) as pdf:
        # Page 1: role-aware method summary.
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        ax.set_title("Fingerprint Enhancement - Role-Aware Batch Summary", fontsize=18, fontweight="bold", pad=18)
        table_columns = ["Method", "MSE_mean", "PSNR_mean", "SSIM_mean", "Coherence_mean", "Match_mean", "Runtime_mean"]
        table_frame = summary[table_columns].copy().round(4)
        table_frame["Method"] = table_frame["Method"].str.wrap(32)
        table = ax.table(cellText=table_frame.values, colLabels=table_frame.columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.8)
        ax.text(
            0.5,
            0.12,
            f"Dataset mode: {'SYNTHETIC DEMO - replace with SOCOFing' if USING_DEMO_DATA else 'SOCOFing Real images with controlled degradation'}",
            ha="center",
            fontsize=10,
            color="#b42318" if USING_DEMO_DATA else "#176b3a",
        )
        ax.text(
            0.5,
            0.06,
            "Controls and baselines are not individual member techniques; Team Hybrid Pipeline is reported separately.",
            ha="center",
            fontsize=9,
            color="#344054",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Figure 1: individual member contributions.
        fig = plot_individual_contributions(example_result, sample_title="Example processing pipeline")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Figure 2: enhancement/restoration metrics only where greyscale metrics are meaningful.
        fig = plot_enhancement_quality_comparison(metrics, method_order)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Figure 3: Member 1 baseline-vs-final evidence.
        fig = plot_member1_enhancement_study(metrics)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Figure 4: Member 2 ridge enhancement evidence.
        fig = plot_member2_ridge_enhancement_study(metrics)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Figure 5: Member 3 binary ridge restoration structure.
        fig = plot_member3_ridge_restoration_study(example_result, sample_title="Example processing pipeline")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Figure 6: Member 4 feature extraction outputs and counts.
        fig = plot_member4_feature_results(example_result, sample_title="Example processing pipeline")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Figure 7: team-level integrated hybrid pipeline.
        fig = plot_team_hybrid_evaluation(example_result, sample_title="Example processing pipeline")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


if EXPORT_FINAL_RESULTS:
    batch_csv = OUTPUT_DIR / "batch_metrics.csv"
    summary_csv = OUTPUT_DIR / "summary_metrics.csv"
    report_pdf = OUTPUT_DIR / "fingerprint_enhancement_report.pdf"
    overlay_png = OUTPUT_DIR / "example_minutiae_overlay.png"
    experiment_json = OUTPUT_DIR / "experiment_metadata.json"

    batch_metrics.to_csv(batch_csv, index=False)
    summary_metrics.to_csv(summary_csv, index=False)
    export_pdf_report(summary_metrics, sample_result, report_pdf)

    overlay_figure, _ = minutiae_overlay(
        sample_result["skeleton"],
        sample_result["endings"],
        sample_result["bifurcations"],
        title="Example extracted minutiae",
    )
    overlay_figure.savefig(overlay_png, dpi=180, bbox_inches="tight")
    plt.close(overlay_figure)

    metadata = {
        "dataset_mode": "synthetic_demo" if USING_DEMO_DATA else "SOCOFing_real_controlled_degradation",
        "development_mode": DEVELOPMENT_MODE,
        "experiment_started_at": NOTEBOOK_EXECUTION_STARTED_AT.isoformat(timespec="seconds"),
        "experiment_completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_runtime_seconds": float(perf_counter() - NOTEBOOK_EXECUTION_TIMER_START),
        "batch_runtime_seconds": float(batch_total_runtime_seconds),
        "max_batch_images": MAX_BATCH_IMAGES,
        "number_of_real_images_available": len(real_paths),
        "number_of_reference_images": len(reference_set),
        "active_batch_images": len(active_batch_references),
        "image_size": list(IMAGE_SIZE),
        "random_seed": RANDOM_SEED,
        "sampling_method": "synthetic demo generator" if USING_DEMO_DATA else SAMPLING_METHOD_DESCRIPTION,
        "degradation_model": "simulate_degradation",
        "degradation_severities": [0.35, 0.55, 0.75],
        "degradation_seed_formula": "RANDOM_SEED + image_index",
        "methods_compared": method_order,
        "control_methods": CONTROL_METHODS,
        "baseline_methods": BASELINE_METHODS,
        "member_image_contributions": MEMBER_IMAGE_CONTRIBUTIONS,
        "team_methods": TEAM_METHODS,
        "literature_psnr_benchmark_db": LITERATURE_PSNR_BENCHMARK_DB,
        "member1_final_method": M1_LABEL,
        "member1_baseline": CLAHE_BASELINE_LABEL,
        "team_baseline": GLOBAL_HE_BASELINE_LABEL,
        "member4_feature_summary": member4_feature_summary.to_dict(orient="records"),
        "metric_specific_leaders": metric_leaders.to_dict(orient="records"),
        "svm_controlled_test_accuracy": None if np.isnan(quality_accuracy) else float(quality_accuracy),
        "warning": "Synthetic demo measurements are not final experimental evidence." if USING_DEMO_DATA else "",
    }
    experiment_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Exported files:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f" - {path.name}")
else:
    print("Final exports are disabled. Existing output files were not overwritten by this run.")


# %% [markdown]
# ## 14. Interpretation checklist for the report
#
# 1. Confirm that `USING_DEMO_DATA` is `False` before collecting final results.
# 2. Confirm whether the notebook is in development mode or final evaluation mode, and report the active batch size.
# 3. Report the number of SOCOFing `Real` reference images, controlled degradation levels and fixed random seed.
# 4. Keep `Altered-Easy`, `Altered-Medium` and `Altered-Hard` for later qualitative/application testing; do not report raw Real-vs-Altered full-reference metrics without registration.
# 5. Compare mean **and standard deviation**, not a single best-looking image.
# 6. Interpret MSE, PSNR, SSIM and ridge coherence separately; do not use a weighted overall score.
# 7. If metric leaders disagree, explain the trade-off instead of forcing one method to win.
# 8. Report Degraded Input, Global HE Baseline and CLAHE Baseline as controls/baselines, not as individual member techniques.
# 9. Use the four member labels consistently: M1 - Wiener + CLAHE Enhancement, M2 - Gabor / Modified Gabor Ridge Enhancement, M3 - Morphological Ridge Restoration, and M4 - Thinning & Minutiae Extraction.
# 10. Report Team Hybrid Pipeline separately from the four individual member techniques.
# 11. Do not report M4 skeleton/minutiae outputs as direct greyscale PSNR enhancement scores.
# 12. Explain the trade-off between restoration quality and runtime. Gabor filtering may improve ridge coherence but require more computation.
# 13. Do not rename the ORB match score or minutiae counts as "accuracy". Only a held-out labelled classifier result can be an accuracy value.
# 14. Keep the optional SVM synthetic quality classifier separate from the main classical image-processing comparison.
# 15. Discuss failure cases such as severe smudging, weak ridge contrast, over-segmentation and false minutiae.
# 16. Link the conclusion back to the SMART objectives and state whether the quantitative improvement target was achieved.
#

# %% [markdown]
# ## References
#
# Shehu, Y. I., Ruiz-Garcia, A., Palade, V., & James, A. (2018). *Sokoto Coventry Fingerprint Dataset*. arXiv. <https://arxiv.org/abs/1807.10609>
#
# Shehu, Y. I., Ruiz-Garcia, A., Palade, V., & James, A. (2018). *Sokoto Coventry Fingerprint Dataset (SOCOFing)* [Data set]. Kaggle. <https://www.kaggle.com/datasets/ruizgara/socofing>
#
# Jain, A. K., Hong, L., Pankanti, S., & Bolle, R. (1997). An identity-authentication system using fingerprints. *Proceedings of the IEEE, 85*(9), 1365-1388. <https://doi.org/10.1109/5.628674>
#
# **AI disclosure reminder:** retain the required AI usage disclosure form and describe the prompts used, the code tests performed, the dataset checks completed, and any manual modifications made by the team.
