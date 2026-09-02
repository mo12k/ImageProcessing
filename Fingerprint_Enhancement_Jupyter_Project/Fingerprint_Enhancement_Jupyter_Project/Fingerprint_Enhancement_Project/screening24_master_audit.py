from __future__ import annotations

import json
import math
import re
import warnings
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from PIL import Image
from scipy import ndimage as ndi
from scipy import signal
from skimage import exposure
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import restoration
from skimage import transform
from skimage.draw import ellipse
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "SOCOFing"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (256, 256)
RANDOM_SEED = 42
QUALITY_SAMPLE_SIZE = 500
SCREENING_SAMPLE_SIZE = 24
SEVERITIES = (0.35, 0.55, 0.75)
LITERATURE_PSNR_BENCHMARK_DB = 28.17
SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

REAL_FOLDER = DATA_ROOT / "Real"
ALTERED_FOLDERS = {
    "Altered Easy": DATA_ROOT / "Altered" / "Altered-Easy",
    "Altered Medium": DATA_ROOT / "Altered" / "Altered-Medium",
    "Altered Hard": DATA_ROOT / "Altered" / "Altered-Hard",
}

PROTECTED_OUTPUTS = [
    OUTPUT_DIR / "batch_metrics.csv",
    OUTPUT_DIR / "summary_metrics.csv",
    OUTPUT_DIR / "experiment_metadata.json",
    OUTPUT_DIR / "fingerprint_enhancement_report.pdf",
    OUTPUT_DIR / "example_minutiae_overlay.png",
]

SCREENING_FILES = {
    "dataset_quality": OUTPUT_DIR / "screening24_dataset_quality.csv",
    "preprocessing": OUTPUT_DIR / "screening24_preprocessing.csv",
    "degradation_ablation": OUTPUT_DIR / "screening24_degradation_ablation.csv",
    "single_techniques": OUTPUT_DIR / "screening24_single_techniques.csv",
    "family_summary": OUTPUT_DIR / "screening24_family_summary.csv",
    "member_results": OUTPUT_DIR / "screening24_member_results.csv",
    "combinations": OUTPUT_DIR / "screening24_combinations.csv",
    "stage_metrics": OUTPUT_DIR / "screening24_stage_metrics.csv",
    "minutiae": OUTPUT_DIR / "screening24_minutiae.csv",
    "metadata": OUTPUT_DIR / "screening24_metadata.json",
    "report": OUTPUT_DIR / "screening24_report.md",
}


def metric_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(image, 0.0, 1.0)


def normalise_image(image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, [low_percentile, high_percentile])
    if high - low < 1e-8:
        return metric_image(image)
    return metric_image((image - low) / (high - low))


def load_grayscale_raw(path: Path, output_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    with Image.open(path) as pil_image:
        image = np.asarray(pil_image.convert("L"), dtype=np.float32) / 255.0
    image = transform.resize(image, output_size, anti_aliasing=True, preserve_range=True)
    return metric_image(image)


def load_grayscale_metric(path: Path, output_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    return normalise_image(load_grayscale_raw(path, output_size))


def image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS], key=lambda p: p.name)


def deterministic_even_sample(paths: list[Path], max_count: int) -> list[Path]:
    if max_count is None or len(paths) <= max_count:
        return list(paths)
    indexes = np.linspace(0, len(paths) - 1, max_count, dtype=int)
    return [paths[int(i)] for i in indexes]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_files(paths: list[Path]) -> dict[str, dict[str, object]]:
    snapshot = {}
    for path in paths:
        snapshot[path.name] = {
            "exists": path.exists(),
            "length": path.stat().st_size if path.exists() else None,
            "mtime": path.stat().st_mtime if path.exists() else None,
            "sha256": sha256_file(path),
        }
    return snapshot


def local_std_image(image: np.ndarray, size: int = 17) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    mean = ndi.uniform_filter(image, size=size, mode="reflect")
    mean_sq = ndi.uniform_filter(image * image, size=size, mode="reflect")
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def fingerprint_mask(image: np.ndarray, block_size: int = 17) -> np.ndarray:
    base = metric_image(image)
    local_std = local_std_image(base, block_size)
    std_threshold = max(0.012, float(np.percentile(local_std, 70) * 0.55))
    contrast_mask = local_std > std_threshold
    darkness_mask = base < np.percentile(base, 92)
    mask = contrast_mask | (darkness_mask & (local_std > std_threshold * 0.55))
    mask = morphology.closing(mask, morphology.disk(5))
    mask = morphology.opening(mask, morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=600)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=600)
    return np.asarray(mask, dtype=bool)


def structure_tensor_coherence_map(image: np.ndarray, smoothing_sigma: float = 2.0) -> np.ndarray:
    base = metric_image(image)
    gx = filters.sobel_h(base)
    gy = filters.sobel_v(base)
    gxx = ndi.gaussian_filter(gx * gx, smoothing_sigma)
    gyy = ndi.gaussian_filter(gy * gy, smoothing_sigma)
    gxy = ndi.gaussian_filter(gx * gy, smoothing_sigma)
    return np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy**2) / (gxx + gyy + 1e-8)


def orientation_coherence(image: np.ndarray, mask: np.ndarray | None = None, smoothing_sigma: float = 2.0) -> float:
    coherence = structure_tensor_coherence_map(image, smoothing_sigma)
    if mask is None or not np.any(mask):
        return float(np.mean(coherence))
    return float(np.mean(coherence[np.asarray(mask, dtype=bool)]))


def estimate_orientation_field(
    image: np.ndarray,
    smoothing_sigma: float = 3.0,
    orientation_smoothing_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    base = metric_image(image)
    gx = filters.sobel_h(base)
    gy = filters.sobel_v(base)
    gxx = ndi.gaussian_filter(gx * gx, smoothing_sigma)
    gyy = ndi.gaussian_filter(gy * gy, smoothing_sigma)
    gxy = ndi.gaussian_filter(gx * gy, smoothing_sigma)
    sin2 = ndi.gaussian_filter(2.0 * gxy, orientation_smoothing_sigma)
    cos2 = ndi.gaussian_filter(gxx - gyy, orientation_smoothing_sigma)
    orientation = 0.5 * np.arctan2(sin2, cos2)
    coherence = np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy**2) / (gxx + gyy + 1e-8)
    return orientation, coherence


def block_frequency_stats(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    block_size: int = 32,
    min_frequency: float = 0.035,
    max_frequency: float = 0.22,
) -> tuple[float, float, float]:
    base = metric_image(image)
    mask = np.ones_like(base, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    valid = 0
    eligible = 0
    frequencies: list[float] = []
    window = np.outer(np.hanning(block_size), np.hanning(block_size))
    yy, xx = np.mgrid[-block_size // 2 : block_size // 2, -block_size // 2 : block_size // 2]
    radius = np.sqrt(xx**2 + yy**2)
    center_exclusion = radius <= 2.0
    for row in range(0, base.shape[0] - block_size + 1, block_size):
        for col in range(0, base.shape[1] - block_size + 1, block_size):
            block_mask = mask[row : row + block_size, col : col + block_size]
            if float(np.mean(block_mask)) < 0.35:
                continue
            block = base[row : row + block_size, col : col + block_size]
            if float(np.std(block[block_mask])) < 0.012:
                continue
            eligible += 1
            centred = (block - float(np.mean(block[block_mask]))) * window
            spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centred)))
            spectrum[center_exclusion] = 0.0
            peak_index = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
            radial_frequency = float(radius[peak_index] / block_size)
            if min_frequency <= radial_frequency <= max_frequency:
                valid += 1
                frequencies.append(radial_frequency)
    if eligible == 0:
        return 0.0, 0.0, 0.0
    return float(valid / eligible), float(np.mean(frequencies)) if frequencies else 0.0, float(eligible / 64.0)


def entropy_in_mask(image: np.ndarray, mask: np.ndarray) -> float:
    values = metric_image(image)[mask] if np.any(mask) else metric_image(image).ravel()
    hist, _ = np.histogram(values, bins=64, range=(0.0, 1.0), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-12)) / 64.0)


def quality_indicators(path: Path, split: str) -> dict[str, object]:
    image = load_grayscale_raw(path)
    mask = fingerprint_mask(image)
    foreground = image[mask] if np.any(mask) else image.ravel()
    local_std = local_std_image(image)
    coherence_map = structure_tensor_coherence_map(image)
    frequency_valid, frequency_mean, frequency_eligible = block_frequency_stats(image, mask)
    laplace = cv2.Laplacian((image * 255).astype(np.uint8), cv2.CV_64F)
    dynamic_range = float(np.percentile(foreground, 99) - np.percentile(foreground, 1))
    return {
        "Dataset split": split,
        "Image": path.name,
        "Path": str(path.relative_to(PROJECT_ROOT)),
        "Ridge coherence": float(np.mean(coherence_map[mask])) if np.any(mask) else float(np.mean(coherence_map)),
        "Sharpness Laplacian variance": float(np.var(laplace[mask])) if np.any(mask) else float(np.var(laplace)),
        "Local contrast": float(np.mean(local_std[mask])) if np.any(mask) else float(np.mean(local_std)),
        "Intensity dynamic range": dynamic_range,
        "Foreground usable-area ratio": float(np.mean(mask)),
        "Valid ridge-frequency-block ratio": frequency_valid,
        "Mean valid ridge frequency": frequency_mean,
        "Eligible frequency-block ratio": frequency_eligible,
        "Orientation confidence valid ratio": float(np.mean(coherence_map[mask] > 0.35)) if np.any(mask) else float(np.mean(coherence_map > 0.35)),
        "Entropy": entropy_in_mask(image, mask),
    }


def add_quality_scores(quality_df: pd.DataFrame) -> pd.DataFrame:
    frame = quality_df.copy()
    frame["Log sharpness"] = np.log10(frame["Sharpness Laplacian variance"].clip(lower=1e-8))
    score_columns = [
        "Ridge coherence",
        "Log sharpness",
        "Local contrast",
        "Intensity dynamic range",
        "Foreground usable-area ratio",
        "Valid ridge-frequency-block ratio",
        "Orientation confidence valid ratio",
        "Entropy",
    ]
    real = frame[frame["Dataset split"] == "Real"]
    for column in score_columns:
        low = float(real[column].quantile(0.05))
        high = float(real[column].quantile(0.95))
        if high - low < 1e-10:
            frame[f"{column} score"] = 0.5
        else:
            frame[f"{column} score"] = ((frame[column] - low) / (high - low)).clip(0.0, 1.0)
    frame["Diagnostic quality score"] = frame[[f"{c} score" for c in score_columns]].mean(axis=1)
    real_scores = frame.loc[frame["Dataset split"] == "Real", "Diagnostic quality score"]
    lower_cut = float(real_scores.quantile(1.0 / 3.0))
    upper_cut = float(real_scores.quantile(2.0 / 3.0))
    conditions = [
        frame["Diagnostic quality score"] <= lower_cut,
        frame["Diagnostic quality score"] <= upper_cut,
    ]
    frame["Quality band vs Real"] = np.select(conditions, ["Lower quality", "Medium quality"], default="Higher quality")
    frame.loc[frame["Dataset split"] != "Real", "Quality band vs Real"] = (
        "Relative " + frame.loc[frame["Dataset split"] != "Real", "Quality band vs Real"].astype(str)
    )
    frame["Selected for screening24"] = False
    return frame


def select_screening_paths(quality_df: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    real = quality_df[quality_df["Dataset split"] == "Real"].copy()
    for band in ["Higher quality", "Medium quality", "Lower quality"]:
        candidates = real[real["Quality band vs Real"] == band].sort_values(["Diagnostic quality score", "Image"])
        if band == "Higher quality":
            candidates = candidates.sort_values(["Diagnostic quality score", "Image"], ascending=[False, True])
        sampled = deterministic_even_sample([Path(p) for p in candidates["Path"].tolist()], 8)
        selected.extend(str(p) for p in sampled)
    if len(selected) < SCREENING_SAMPLE_SIZE:
        already = set(selected)
        filler = real[~real["Path"].isin(already)].sort_values(["Image"])
        selected.extend(filler["Path"].head(SCREENING_SAMPLE_SIZE - len(selected)).tolist())
    return selected[:SCREENING_SAMPLE_SIZE]


def evaluate_grayscale(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None, runtime_seconds: float = math.nan) -> dict[str, float]:
    reference = metric_image(reference)
    candidate = metric_image(candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"Metric images must be aligned and equal-sized; got {reference.shape} and {candidate.shape}.")
    return {
        "MSE": float(mean_squared_error(reference, candidate)),
        "PSNR (dB)": float(peak_signal_noise_ratio(reference, candidate, data_range=1.0)),
        "SSIM": float(structural_similarity(reference, candidate, data_range=1.0)),
        "Ridge coherence": orientation_coherence(candidate, mask),
        "Runtime (s)": float(runtime_seconds),
    }


def simulate_degradation_stages(reference: np.ndarray, severity: float = 0.55, seed: int = RANDOM_SEED) -> tuple[np.ndarray, list[dict[str, object]]]:
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    stages: list[dict[str, object]] = []
    image = np.asarray(reference, dtype=np.float32).copy()
    stages.append(
        {
            "Stage": 0,
            "Operation": "Clean Real reference",
            "Parameters": "none",
            "Image": image.copy(),
            "Irreversible risk": "reference only",
        }
    )

    blur_sigma = 0.35 + 1.8 * severity
    image = ndi.gaussian_filter(image, sigma=blur_sigma)
    stages.append(
        {
            "Stage": 1,
            "Operation": "Optical/sensor Gaussian blur",
            "Parameters": f"sigma={blur_sigma:.4f}",
            "Image": image.copy(),
            "Irreversible risk": "blur removes high-frequency ridge detail",
        }
    )

    contrast_scale = 1.0 - 0.58 * severity
    image = 0.5 + (image - 0.5) * contrast_scale
    yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
    illumination = 1.0 + 0.16 * severity * np.sin(2 * np.pi * xx / image.shape[1])
    image *= illumination
    stages.append(
        {
            "Stage": 2,
            "Operation": "Reduced dynamic range + sinusoidal illumination",
            "Parameters": f"contrast_scale={contrast_scale:.4f}; illumination_amplitude={0.16 * severity:.4f}",
            "Image": image.copy(),
            "Irreversible risk": "low contrast can bury weak ridges",
        }
    )

    gaussian_sigma = 0.018 + 0.055 * severity
    image += rng.normal(0.0, gaussian_sigma, image.shape)
    stages.append(
        {
            "Stage": 3,
            "Operation": "Gaussian sensor noise",
            "Parameters": f"sigma={gaussian_sigma:.4f}",
            "Image": image.copy(),
            "Irreversible risk": "noise can obscure ridge/valley boundaries",
        }
    )

    impulse_probability = 0.0015 * severity
    impulse_mask = rng.random(image.shape)
    image[impulse_mask < impulse_probability] = 0.0
    image[impulse_mask > 1.0 - impulse_probability] = 1.0
    stages.append(
        {
            "Stage": 4,
            "Operation": "Sparse salt-and-pepper impulse noise",
            "Parameters": f"pepper_probability={impulse_probability:.6f}; salt_probability={impulse_probability:.6f}",
            "Image": image.copy(),
            "Irreversible risk": "local clipping to 0/1 at impulse pixels",
        }
    )

    if severity >= 0.35:
        smudge = np.zeros_like(image, dtype=bool)
        centre_r = int(image.shape[0] * (0.52 + rng.uniform(-0.10, 0.10)))
        centre_c = int(image.shape[1] * (0.52 + rng.uniform(-0.10, 0.10)))
        radius_r = max(3, int(image.shape[0] * 0.035 * severity))
        radius_c = max(5, int(image.shape[1] * 0.10 * severity))
        rotation = float(rng.uniform(-0.8, 0.8))
        rr, cc = ellipse(centre_r, centre_c, radius_r, radius_c, shape=image.shape, rotation=rotation)
        smudge[rr, cc] = True
        smudge_sigma = 3.0 + 2.0 * severity
        blurred_patch = ndi.gaussian_filter(image, sigma=smudge_sigma)
        image[smudge] = blurred_patch[smudge]
        stages.append(
            {
                "Stage": 5,
                "Operation": "Partial smudge/occlusion",
                "Parameters": (
                    f"centre=({centre_r},{centre_c}); radii=({radius_r},{radius_c}); "
                    f"rotation={rotation:.4f}; patch_blur_sigma={smudge_sigma:.4f}"
                ),
                "Image": image.copy(),
                "Irreversible risk": "localized occlusion/smudge can overwrite ridge detail",
            }
        )

    clipped = np.clip(image, 0.0, 1.0)
    clipped_fraction = float(np.mean((image < 0.0) | (image > 1.0)))
    stages.append(
        {
            "Stage": len(stages),
            "Operation": "Final clipping to [0,1]",
            "Parameters": f"clipped_fraction_before_clip={clipped_fraction:.6f}",
            "Image": clipped.copy(),
            "Irreversible risk": "saturation is irreversible where clipping occurs",
        }
    )
    return clipped, stages


def independent_degradation_component(reference: np.ndarray, operation: str, severity: float, seed: int) -> tuple[np.ndarray, str, str]:
    rng = np.random.default_rng(seed)
    image = np.asarray(reference, dtype=np.float32).copy()
    if operation == "Gaussian noise only":
        sigma = 0.018 + 0.055 * severity
        return metric_image(image + rng.normal(0.0, sigma, image.shape)), f"sigma={sigma:.4f}", "recoverable only statistically"
    if operation == "Salt-and-pepper noise only":
        probability = 0.0015 * severity
        impulse_mask = rng.random(image.shape)
        image[impulse_mask < probability] = 0.0
        image[impulse_mask > 1.0 - probability] = 1.0
        return metric_image(image), f"salt_probability={probability:.6f}; pepper_probability={probability:.6f}", "pixel clipping at impulses"
    if operation == "Blur only":
        sigma = 0.35 + 1.8 * severity
        return metric_image(ndi.gaussian_filter(image, sigma=sigma)), f"sigma={sigma:.4f}", "high-frequency detail loss"
    if operation == "Contrast reduction + illumination only":
        contrast_scale = 1.0 - 0.58 * severity
        yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
        illumination = 1.0 + 0.16 * severity * np.sin(2 * np.pi * xx / image.shape[1])
        image = (0.5 + (image - 0.5) * contrast_scale) * illumination
        return metric_image(image), f"contrast_scale={contrast_scale:.4f}; illumination_amplitude={0.16 * severity:.4f}", "low contrast"
    if operation == "Partial smudge/occlusion only":
        smudge = np.zeros_like(image, dtype=bool)
        centre_r = int(image.shape[0] * (0.52 + rng.uniform(-0.10, 0.10)))
        centre_c = int(image.shape[1] * (0.52 + rng.uniform(-0.10, 0.10)))
        radius_r = max(3, int(image.shape[0] * 0.035 * severity))
        radius_c = max(5, int(image.shape[1] * 0.10 * severity))
        rotation = float(rng.uniform(-0.8, 0.8))
        rr, cc = ellipse(centre_r, centre_c, radius_r, radius_c, shape=image.shape, rotation=rotation)
        smudge[rr, cc] = True
        smudge_sigma = 3.0 + 2.0 * severity
        patch = ndi.gaussian_filter(image, sigma=smudge_sigma)
        image[smudge] = patch[smudge]
        return metric_image(image), f"radii=({radius_r},{radius_c}); patch_blur_sigma={smudge_sigma:.4f}", "localized overwritten ridge detail"
    raise ValueError(operation)


def common_preprocessing_variants(image: np.ndarray) -> dict[str, tuple[str, np.ndarray]]:
    base = metric_image(image)
    return {
        "P0": ("grayscale + resize + float [0,1] only", base),
        "P1": ("P0 + mild 3x3 median filtering", metric_image(ndi.median_filter(base, size=3))),
        "P2": (
            "P0 + mild bilateral filtering",
            metric_image(restoration.denoise_bilateral(base, sigma_color=0.035, sigma_spatial=2.0, channel_axis=None)),
        ),
        "P3": ("P0 + mild Gaussian filtering sigma=0.55", metric_image(ndi.gaussian_filter(base, sigma=0.55))),
    }


def gaussian_psf(size: int = 15, sigma: float = 1.34) -> np.ndarray:
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def frequency_wiener_deconvolution(image: np.ndarray, psf_sigma: float = 1.34, balance: float = 0.06, blend: float = 0.15) -> np.ndarray:
    base = metric_image(image)
    psf = gaussian_psf(size=15, sigma=psf_sigma)
    result = restoration.wiener(base, psf, balance=balance, clip=False)
    result = metric_image(result)
    return metric_image((1.0 - blend) * base + blend * result)


def tikhonov_inverse_restoration(image: np.ndarray, psf_sigma: float = 1.34, regularisation: float = 0.025, blend: float = 0.12) -> np.ndarray:
    base = metric_image(image)
    psf = gaussian_psf(size=15, sigma=psf_sigma)
    padded = np.zeros_like(base)
    kh, kw = psf.shape
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -kh // 2, axis=0)
    padded = np.roll(padded, -kw // 2, axis=1)
    otf = np.fft.fft2(padded)
    restored = np.real(np.fft.ifft2(np.fft.fft2(base) * np.conj(otf) / (np.abs(otf) ** 2 + regularisation)))
    restored = metric_image(restored)
    return metric_image((1.0 - blend) * base + blend * restored)


def richardson_lucy_restoration(image: np.ndarray, psf_sigma: float = 1.34, iterations: int = 8, blend: float = 0.10) -> np.ndarray:
    base = metric_image(image)
    psf = gaussian_psf(size=15, sigma=psf_sigma)
    try:
        restored = restoration.richardson_lucy(base, psf, num_iter=iterations, clip=False)
    except TypeError:
        restored = restoration.richardson_lucy(base, psf, iterations=iterations, clip=False)
    restored = metric_image(restored)
    return metric_image((1.0 - blend) * base + blend * restored)


def wavelet_denoise(image: np.ndarray, wavelet: str = "db2", level: int = 2, threshold_scale: float = 0.10) -> np.ndarray:
    base = metric_image(image)
    wavelet_object = pywt.Wavelet(wavelet)
    active_level = min(level, pywt.dwtn_max_level(base.shape, wavelet_object))
    coeffs = pywt.wavedec2(base, wavelet=wavelet_object, level=active_level, mode="symmetric")
    detail_values = np.concatenate([np.ravel(detail) for level_details in coeffs[1:] for detail in level_details])
    sigma = np.median(np.abs(detail_values)) / 0.6745 if detail_values.size else 0.0
    threshold = threshold_scale * sigma * math.sqrt(2.0 * math.log(base.size))
    denoised = [coeffs[0]]
    for level_details in coeffs[1:]:
        denoised.append(tuple(pywt.threshold(detail, threshold, mode="soft") for detail in level_details))
    reconstructed = pywt.waverec2(denoised, wavelet=wavelet_object, mode="symmetric")
    return metric_image(reconstructed[: base.shape[0], : base.shape[1]])


def anisotropic_diffusion(image: np.ndarray, iterations: int = 8, kappa: float = 0.08, gamma: float = 0.14) -> np.ndarray:
    current = metric_image(image).astype(np.float32)
    for _ in range(iterations):
        north = np.roll(current, -1, axis=0) - current
        south = np.roll(current, 1, axis=0) - current
        east = np.roll(current, -1, axis=1) - current
        west = np.roll(current, 1, axis=1) - current
        update = (
            np.exp(-(north / kappa) ** 2) * north
            + np.exp(-(south / kappa) ** 2) * south
            + np.exp(-(east / kappa) ** 2) * east
            + np.exp(-(west / kappa) ** 2) * west
        )
        current = metric_image(current + gamma * update)
    return current


def apply_clahe(image: np.ndarray, clip_limit: float = 0.025, kernel_size: tuple[int, int] = (32, 32)) -> np.ndarray:
    return metric_image(exposure.equalize_adapthist(metric_image(image), clip_limit=clip_limit, kernel_size=kernel_size))


def adaptive_histogram_equalisation(image: np.ndarray) -> np.ndarray:
    return metric_image(exposure.equalize_adapthist(metric_image(image), clip_limit=1.0, kernel_size=(32, 32)))


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    base = metric_image(image)
    low, high = np.percentile(base, [2.0, 98.0])
    if high - low < 1e-8:
        return base
    return metric_image((base - low) / (high - low))


def gamma_correction(image: np.ndarray, gamma: float = 0.85) -> np.ndarray:
    return metric_image(exposure.adjust_gamma(metric_image(image), gamma=gamma))


def homomorphic_filter(image: np.ndarray, sigma: float = 18.0, gain: float = 0.70) -> np.ndarray:
    base = metric_image(image)
    log_image = np.log1p(base)
    low = ndi.gaussian_filter(log_image, sigma=sigma)
    high = log_image - low
    result = np.expm1(low + gain * high)
    return metric_image(result)


def multi_orientation_gabor(
    image: np.ndarray,
    frequency: float = 0.115,
    orientations: int = 8,
    blend: float = 0.30,
) -> np.ndarray:
    base = metric_image(image)
    inverted = 1.0 - base
    responses = []
    for theta in np.linspace(0.0, np.pi, orientations, endpoint=False):
        real, imaginary = filters.gabor(inverted, frequency=float(frequency), theta=float(theta))
        responses.append(np.sqrt(real**2 + imaginary**2))
    energy = np.max(np.stack(responses, axis=0), axis=0)
    energy = normalise_image(energy)
    ridge_image = 1.0 - energy
    return metric_image((1.0 - blend) * base + blend * ridge_image)


def estimate_local_frequency_map(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    block_size: int = 32,
    frequencies: tuple[float, ...] = (0.075, 0.095, 0.115, 0.140),
) -> tuple[np.ndarray, np.ndarray]:
    base = metric_image(image)
    mask = np.ones_like(base, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    freq_map = np.full_like(base, float(np.median(frequencies)), dtype=np.float32)
    confidence = np.zeros_like(base, dtype=np.float32)
    window = np.outer(np.hanning(block_size), np.hanning(block_size))
    yy, xx = np.mgrid[-block_size // 2 : block_size // 2, -block_size // 2 : block_size // 2]
    radius = np.sqrt(xx**2 + yy**2)
    center_exclusion = radius <= 2.0
    for row in range(0, base.shape[0] - block_size + 1, block_size // 2):
        for col in range(0, base.shape[1] - block_size + 1, block_size // 2):
            block_mask = mask[row : row + block_size, col : col + block_size]
            if float(np.mean(block_mask)) < 0.25:
                continue
            block = base[row : row + block_size, col : col + block_size]
            if float(np.std(block[block_mask])) < 0.012:
                continue
            spectrum = np.abs(np.fft.fftshift(np.fft.fft2((block - np.mean(block[block_mask])) * window)))
            spectrum[center_exclusion] = 0.0
            peak_index = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
            observed = float(radius[peak_index] / block_size)
            nearest = min(frequencies, key=lambda value: abs(value - observed))
            conf = float(np.max(spectrum) / (np.mean(spectrum) + 1e-8))
            freq_map[row : row + block_size, col : col + block_size] = nearest
            confidence[row : row + block_size, col : col + block_size] = max(
                float(np.mean(confidence[row : row + block_size, col : col + block_size])),
                min(conf / 25.0, 1.0),
            )
    return freq_map, confidence


def adaptive_gabor_enhancement(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    frequencies: tuple[float, ...] = (0.075, 0.095, 0.115, 0.140),
    orientation_bins: int = 8,
    use_local_frequency: bool = True,
    blend: float = 0.22,
) -> np.ndarray:
    base = metric_image(image)
    mask = fingerprint_mask(base) if mask is None else np.asarray(mask, dtype=bool)
    orientation, coherence = estimate_orientation_field(base)
    freq_map, freq_confidence = estimate_local_frequency_map(base, mask, frequencies=frequencies)
    theta_bins = np.linspace(0.0, np.pi, orientation_bins, endpoint=False)
    theta_index = np.argmin(np.abs(np.angle(np.exp(1j * (orientation[..., None] - theta_bins[None, None, :])))), axis=2)
    freq_values = frequencies if use_local_frequency else (0.115,)
    freq_index = np.zeros_like(base, dtype=np.int16)
    if use_local_frequency:
        freq_index = np.argmin(np.abs(freq_map[..., None] - np.asarray(frequencies)[None, None, :]), axis=2)
    inverted = 1.0 - base
    selected_energy = np.zeros_like(base, dtype=np.float32)
    for fi, frequency in enumerate(freq_values):
        active_freq_index = fi if use_local_frequency else 0
        for ti, theta in enumerate(theta_bins):
            real, imaginary = filters.gabor(inverted, frequency=float(frequency), theta=float(theta), bandwidth=1.8)
            response = np.sqrt(real**2 + imaginary**2)
            selector = (theta_index == ti) & (freq_index == active_freq_index) & mask
            selected_energy[selector] = response[selector]
    selected_energy = normalise_image(selected_energy)
    ridge_image = np.where(mask, 1.0 - selected_energy, base)
    confidence_weight = np.clip(0.30 + 0.70 * coherence, 0.0, 1.0)
    if use_local_frequency:
        confidence_weight *= np.clip(0.40 + 0.60 * freq_confidence, 0.0, 1.0)
    weight = blend * confidence_weight * mask
    return metric_image((1.0 - weight) * base + weight * ridge_image)


def log_gabor_enhancement(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    frequencies: tuple[float, ...] = (0.075, 0.095, 0.115, 0.140),
    orientation_bins: int = 8,
    blend: float = 0.18,
) -> np.ndarray:
    base = metric_image(image)
    mask = fingerprint_mask(base) if mask is None else np.asarray(mask, dtype=bool)
    orientation, coherence = estimate_orientation_field(base)
    freq_map, _ = estimate_local_frequency_map(base, mask, frequencies=frequencies)
    rows, cols = base.shape
    y = np.fft.fftfreq(rows)
    x = np.fft.fftfreq(cols)
    fx, fy = np.meshgrid(x, y)
    radius = np.sqrt(fx**2 + fy**2)
    radius[0, 0] = 1.0
    angle = np.mod(np.arctan2(fy, fx), np.pi)
    spectrum = np.fft.fft2(1.0 - base)
    theta_bins = np.linspace(0.0, np.pi, orientation_bins, endpoint=False)
    theta_index = np.argmin(np.abs(np.angle(np.exp(1j * (orientation[..., None] - theta_bins[None, None, :])))), axis=2)
    freq_index = np.argmin(np.abs(freq_map[..., None] - np.asarray(frequencies)[None, None, :]), axis=2)
    selected = np.zeros_like(base, dtype=np.float32)
    for fi, frequency in enumerate(frequencies):
        radial = np.exp(-(np.log(radius / frequency) ** 2) / (2.0 * np.log(0.60) ** 2))
        radial[0, 0] = 0.0
        for ti, theta in enumerate(theta_bins):
            angle_delta = np.abs(np.angle(np.exp(1j * (angle - theta))))
            angular = np.exp(-(angle_delta**2) / (2.0 * (np.pi / 7.0) ** 2))
            response = np.abs(np.fft.ifft2(spectrum * radial * angular))
            selector = (theta_index == ti) & (freq_index == fi) & mask
            selected[selector] = response[selector]
    energy = normalise_image(selected)
    ridge_image = np.where(mask, 1.0 - energy, base)
    weight = blend * np.clip(0.25 + 0.75 * coherence, 0.0, 1.0) * mask
    return metric_image((1.0 - weight) * base + weight * ridge_image)


def stft_contextual_enhancement(image: np.ndarray, block_size: int = 32, step: int = 16, blend: float = 0.18) -> np.ndarray:
    base = metric_image(image)
    mask = fingerprint_mask(base)
    accumulator = np.zeros_like(base, dtype=np.float32)
    weights = np.zeros_like(base, dtype=np.float32)
    window = np.outer(np.hanning(block_size), np.hanning(block_size)).astype(np.float32)
    yy, xx = np.mgrid[-block_size // 2 : block_size // 2, -block_size // 2 : block_size // 2]
    radius = np.sqrt(xx**2 + yy**2) / block_size
    bandpass = (radius >= 0.055) & (radius <= 0.18)
    for row in range(0, base.shape[0] - block_size + 1, step):
        for col in range(0, base.shape[1] - block_size + 1, step):
            block_mask = mask[row : row + block_size, col : col + block_size]
            if np.mean(block_mask) < 0.20:
                continue
            block = base[row : row + block_size, col : col + block_size]
            centred = (block - np.mean(block)) * window
            filtered = np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(centred)) * bandpass)))
            accumulator[row : row + block_size, col : col + block_size] += filtered * window
            weights[row : row + block_size, col : col + block_size] += window
    contextual = np.where(weights > 1e-8, accumulator / np.maximum(weights, 1e-8), base)
    contextual = normalise_image(contextual)
    ridge_image = 1.0 - contextual
    return metric_image((1.0 - blend) * base + blend * np.where(mask, ridge_image, base))


def coherence_guided_diffusion(image: np.ndarray, iterations: int = 4, step: float = 0.16) -> np.ndarray:
    current = metric_image(image)
    mask = fingerprint_mask(current)
    for _ in range(iterations):
        orientation, coherence = estimate_orientation_field(current)
        smoothed = np.zeros_like(current)
        bins = np.linspace(0.0, np.pi, 8, endpoint=False)
        indexes = np.argmin(np.abs(np.angle(np.exp(1j * (orientation[..., None] - bins[None, None, :])))), axis=2)
        for index, theta in enumerate(bins):
            sigma_y = 0.50
            sigma_x = 1.35
            kernel = oriented_gaussian_kernel(theta, sigma_x=sigma_x, sigma_y=sigma_y, size=9)
            conv = ndi.convolve(current, kernel, mode="reflect")
            smoothed[indexes == index] = conv[indexes == index]
        weight = np.clip(coherence, 0.0, 1.0) * mask
        current = metric_image((1.0 - step * weight) * current + (step * weight) * smoothed)
    return current


def oriented_gaussian_kernel(theta: float, sigma_x: float = 1.4, sigma_y: float = 0.55, size: int = 9) -> np.ndarray:
    axis = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    xx, yy = np.meshgrid(axis, axis)
    c, s = math.cos(theta), math.sin(theta)
    x_rot = c * xx + s * yy
    y_rot = -s * xx + c * yy
    kernel = np.exp(-(x_rot**2 / (2.0 * sigma_x**2) + y_rot**2 / (2.0 * sigma_y**2)))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def directional_filter_bank(image: np.ndarray, blend: float = 0.18) -> np.ndarray:
    base = metric_image(image)
    mask = fingerprint_mask(base)
    orientation, coherence = estimate_orientation_field(base)
    bins = np.linspace(0.0, np.pi, 8, endpoint=False)
    indexes = np.argmin(np.abs(np.angle(np.exp(1j * (orientation[..., None] - bins[None, None, :])))), axis=2)
    selected = np.zeros_like(base, dtype=np.float32)
    for index, theta in enumerate(bins):
        kernel = oriented_gaussian_kernel(theta, sigma_x=1.6, sigma_y=0.45, size=11)
        response = ndi.convolve(base, kernel, mode="reflect")
        selected[indexes == index] = response[indexes == index]
    weight = blend * np.clip(0.25 + 0.75 * coherence, 0.0, 1.0) * mask
    return metric_image((1.0 - weight) * base + weight * selected)


def ridge_base_binary(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = metric_image(image)
    mask = fingerprint_mask(base)
    threshold_surface = filters.threshold_sauvola(base, window_size=25, k=0.16)
    ridges = (base < threshold_surface) & mask
    ridges = morphology.remove_small_objects(ridges, min_size=18)
    ridges = morphology.remove_small_holes(ridges, area_threshold=18)
    return ridges.astype(bool), mask


def segmentation_outputs(image: np.ndarray) -> dict[str, tuple[str, np.ndarray, str]]:
    base = metric_image(image)
    mask = fingerprint_mask(base)
    coherence = structure_tensor_coherence_map(base)
    local_std = local_std_image(base)
    otsu_threshold = filters.threshold_otsu(base[mask]) if np.any(mask) else filters.threshold_otsu(base)
    sauvola = filters.threshold_sauvola(base, window_size=25, k=0.16)
    return {
        "S1": ("Otsu thresholding", (base < otsu_threshold) & mask, "ridge map"),
        "S2": ("Adaptive/local Sauvola thresholding", (base < sauvola) & mask, "ridge map"),
        "S3": ("Local variance foreground segmentation", local_std > max(0.012, np.percentile(local_std, 70) * 0.55), "foreground mask"),
        "S4": ("Coherence-based foreground segmentation", (coherence > 0.35) & mask, "foreground mask"),
        "S5": ("Orientation-confidence segmentation", (coherence > 0.25) & (local_std > np.percentile(local_std, 55)), "foreground mask"),
    }


def remove_skeleton_spurs(skeleton: np.ndarray, iterations: int = 2) -> np.ndarray:
    cleaned = skeleton.astype(bool).copy()
    for _ in range(iterations):
        endpoints, _ = crossing_number_points(cleaned)
        if len(endpoints) == 0:
            break
        cleaned[endpoints[:, 0], endpoints[:, 1]] = False
    return cleaned


def morphology_outputs(image: np.ndarray) -> dict[str, tuple[str, np.ndarray]]:
    ridges, mask = ridge_base_binary(image)
    seed = morphology.erosion(ridges, morphology.disk(1))
    reconstruction = morphology.reconstruction(seed.astype(np.uint8), ridges.astype(np.uint8), method="dilation") > 0
    bridge = morphology.closing(ridges, morphology.disk(1))
    bridge = morphology.binary_dilation(morphology.skeletonize(bridge), morphology.disk(1)) & mask
    cleaned_skeleton = remove_skeleton_spurs(morphology.skeletonize(ridges), iterations=2)
    spur_cleanup = morphology.binary_dilation(cleaned_skeleton, morphology.disk(1)) & mask
    controlled = morphology.closing(ridges, morphology.disk(1))
    controlled = morphology.remove_small_holes(controlled, area_threshold=28)
    controlled = morphology.remove_small_objects(controlled, min_size=24)
    controlled = morphology.binary_dilation(remove_skeleton_spurs(morphology.skeletonize(controlled), iterations=1), morphology.disk(1)) & mask
    return {
        "M3-1": ("Opening", morphology.opening(ridges, morphology.disk(1))),
        "M3-2": ("Closing", morphology.closing(ridges, morphology.disk(1))),
        "M3-3": ("Opening + Closing", morphology.closing(morphology.opening(ridges, morphology.disk(1)), morphology.disk(1))),
        "M3-4": ("Morphological reconstruction", reconstruction),
        "M3-5": ("Bridge / broken-ridge reconnection", bridge),
        "M3-6": ("Remove small objects / isolated noise", morphology.remove_small_objects(ridges, min_size=24)),
        "M3-7": ("Spur cleanup", spur_cleanup),
        "M3-8": ("Controlled ridge-restoration pipeline", controlled),
    }


def crossing_number_points(skeleton: np.ndarray, mask: np.ndarray | None = None, border_margin: int = 8) -> tuple[np.ndarray, np.ndarray]:
    skel = np.asarray(skeleton, dtype=bool)
    padded = np.pad(skel, 1, mode="constant")
    neighbors = [
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
        padded[:-2, :-2],
    ]
    stack = np.stack(neighbors + [neighbors[0]], axis=0).astype(np.int8)
    crossing = np.sum(np.abs(np.diff(stack, axis=0)), axis=0) / 2.0
    valid = skel.copy()
    valid[:border_margin, :] = False
    valid[-border_margin:, :] = False
    valid[:, :border_margin] = False
    valid[:, -border_margin:] = False
    if mask is not None and np.any(mask):
        valid &= morphology.binary_erosion(mask.astype(bool), morphology.disk(border_margin))
    endpoints = np.argwhere(valid & (crossing == 1))
    bifurcations = np.argwhere(valid & (crossing == 3))
    return endpoints, bifurcations


def filter_close_points(points: np.ndarray, minimum_distance: float = 7.0) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 2)
    kept: list[np.ndarray] = []
    for point in points:
        if all(np.linalg.norm(point - existing) >= minimum_distance for existing in kept):
            kept.append(point)
    return np.asarray(kept, dtype=int).reshape(-1, 2)


def structural_metrics(binary: np.ndarray, grayscale: np.ndarray, reference_mask: np.ndarray | None = None, semantic: str = "ridge map") -> dict[str, float]:
    binary = np.asarray(binary, dtype=bool)
    labels = measure.label(binary, connectivity=2)
    props = measure.regionprops(labels)
    areas = np.array([p.area for p in props], dtype=np.float32)
    ridge_pixels = int(np.sum(binary))
    components = int(len(areas))
    small_components = int(np.sum(areas < 20)) if areas.size else 0
    largest_fraction = float(np.max(areas) / max(ridge_pixels, 1)) if areas.size else 0.0
    skeleton = morphology.skeletonize(binary)
    endpoints, bifurcations = crossing_number_points(skeleton)
    reference_iou = math.nan
    if reference_mask is not None and semantic == "foreground mask":
        ref = np.asarray(reference_mask, dtype=bool)
        union = np.logical_or(binary, ref)
        reference_iou = float(np.sum(np.logical_and(binary, ref)) / max(np.sum(union), 1))
    return {
        "Binary foreground/ridge ratio": float(np.mean(binary)),
        "Connected components": float(components),
        "Small components <20px": float(small_components),
        "Largest component fraction": largest_fraction,
        "Fragmentation per 1000 ridge pixels": float(components / max(ridge_pixels / 1000.0, 1e-8)),
        "Skeleton pixels": float(np.sum(skeleton)),
        "Skeleton endpoints": float(len(endpoints)),
        "Skeleton bifurcations": float(len(bifurcations)),
        "Ridge continuity proxy": float(np.sum(skeleton) / max(len(endpoints), 1)),
        "Structural coherence": orientation_coherence(grayscale, binary) if np.any(binary) else 0.0,
        "Reference ROI IoU proxy": reference_iou,
    }


def zhang_suen_thinning(binary: np.ndarray, max_iterations: int = 80) -> np.ndarray:
    image = np.asarray(binary, dtype=np.uint8).copy()
    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            remove = []
            rows, cols = image.shape
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    if image[r, c] == 0:
                        continue
                    p2, p3, p4 = image[r - 1, c], image[r - 1, c + 1], image[r, c + 1]
                    p5, p6, p7 = image[r + 1, c + 1], image[r + 1, c], image[r + 1, c - 1]
                    p8, p9 = image[r, c - 1], image[r - 1, c - 1]
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    transitions = sum((neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1) for i in range(8))
                    count = sum(neighbors)
                    if not (2 <= count <= 6 and transitions == 1):
                        continue
                    if step == 0:
                        condition = p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
                    else:
                        condition = p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
                    if condition:
                        remove.append((r, c))
            if remove:
                changed = True
                rr, cc = zip(*remove)
                image[np.asarray(rr), np.asarray(cc)] = 0
        if not changed:
            break
    return image.astype(bool)


def guo_hall_thinning(binary: np.ndarray, max_iterations: int = 80) -> np.ndarray:
    image = np.asarray(binary, dtype=np.uint8).copy()
    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            remove = []
            rows, cols = image.shape
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    if image[r, c] == 0:
                        continue
                    p2, p3, p4 = image[r - 1, c], image[r - 1, c + 1], image[r, c + 1]
                    p5, p6, p7 = image[r + 1, c + 1], image[r + 1, c], image[r + 1, c - 1]
                    p8, p9 = image[r, c - 1], image[r - 1, c - 1]
                    c_value = (
                        (1 - p2) * (p3 | p4)
                        + (1 - p4) * (p5 | p6)
                        + (1 - p6) * (p7 | p8)
                        + (1 - p8) * (p9 | p2)
                    )
                    n1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8)
                    n2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9)
                    n_value = min(n1, n2)
                    if step == 0:
                        m_value = (p6 | p7 | (1 - p9)) & p8
                    else:
                        m_value = (p2 | p3 | (1 - p5)) & p4
                    if c_value == 1 and 2 <= n_value <= 3 and m_value == 0:
                        remove.append((r, c))
            if remove:
                changed = True
                rr, cc = zip(*remove)
                image[np.asarray(rr), np.asarray(cc)] = 0
        if not changed:
            break
    return image.astype(bool)


def thinning_outputs(image: np.ndarray) -> dict[str, tuple[str, np.ndarray]]:
    base, mask = ridge_base_binary(image)
    restored = morphology_outputs(image)["M3-8"][1]
    return {
        "T1": ("Standard skeletonisation", morphology.skeletonize(restored)),
        "T2": ("Zhang-Suen thinning", zhang_suen_thinning(restored)),
        "T3": ("Guo-Hall thinning", guo_hall_thinning(restored)),
        "T4": ("Morphological skeleton / medial axis", morphology.medial_axis(restored)),
    }


@dataclass(frozen=True)
class Technique:
    family: str
    technique_id: str
    technique: str
    metric_domain: str
    role_hint: str
    notes: str
    function: object


def grayscale_techniques() -> list[Technique]:
    return [
        Technique("Denoising", "D1", "Gaussian filtering", "grayscale", "M1 candidate", "sigma=0.65", lambda img, ctx: metric_image(ndi.gaussian_filter(img, sigma=0.65))),
        Technique("Denoising", "D2", "Median filtering", "grayscale", "M1 candidate", "3x3 median", lambda img, ctx: metric_image(ndi.median_filter(img, size=3))),
        Technique(
            "Denoising",
            "D3",
            "Bilateral filtering",
            "grayscale",
            "M1 candidate",
            "sigma_color=0.035, sigma_spatial=2.0",
            lambda img, ctx: metric_image(restoration.denoise_bilateral(img, sigma_color=0.035, sigma_spatial=2.0, channel_axis=None)),
        ),
        Technique(
            "Denoising",
            "D4",
            "Non-local means",
            "grayscale",
            "M1 candidate",
            "patch_size=3, patch_distance=5, h=0.055",
            lambda img, ctx: metric_image(restoration.denoise_nl_means(img, patch_size=3, patch_distance=5, h=0.055, fast_mode=True, channel_axis=None)),
        ),
        Technique("Denoising", "D5", "Wiener noise filtering", "grayscale", "M1 candidate", "local window=5", lambda img, ctx: metric_image(signal.wiener(img, mysize=5))),
        Technique("Denoising", "D6", "Wavelet denoising", "grayscale", "M1 candidate", "db2 level=2 shrinkage", lambda img, ctx: wavelet_denoise(img)),
        Technique("Denoising", "D7", "Anisotropic diffusion", "grayscale", "M1 candidate", "Perona-Malik style fixed parameters", lambda img, ctx: anisotropic_diffusion(img)),
        Technique(
            "Restoration/Deblurring",
            "R1",
            "Wiener deconvolution",
            "grayscale",
            "M1 candidate",
            "uses known synthetic Gaussian PSF formula; no clean pixels used",
            lambda img, ctx: frequency_wiener_deconvolution(img, psf_sigma=ctx["blur_sigma"], balance=0.06, blend=0.15),
        ),
        Technique(
            "Restoration/Deblurring",
            "R2",
            "Richardson-Lucy deconvolution",
            "grayscale",
            "M1 candidate",
            "uses known synthetic Gaussian PSF formula; 8 iterations",
            lambda img, ctx: richardson_lucy_restoration(img, psf_sigma=ctx["blur_sigma"], iterations=8, blend=0.10),
        ),
        Technique(
            "Restoration/Deblurring",
            "R3",
            "Total variation restoration",
            "grayscale",
            "M1 candidate",
            "TV Chambolle weight=0.045",
            lambda img, ctx: metric_image(restoration.denoise_tv_chambolle(img, weight=0.045, channel_axis=None)),
        ),
        Technique(
            "Restoration/Deblurring",
            "R4",
            "Tikhonov regularised inverse restoration",
            "grayscale",
            "M1 candidate",
            "uses known synthetic Gaussian PSF formula; lambda=0.025",
            lambda img, ctx: tikhonov_inverse_restoration(img, psf_sigma=ctx["blur_sigma"], regularisation=0.025, blend=0.12),
        ),
        Technique(
            "Restoration/Deblurring",
            "R5",
            "Edge-preserving restoration",
            "grayscale",
            "M1 candidate",
            "TV followed by bilateral filter",
            lambda img, ctx: metric_image(restoration.denoise_bilateral(restoration.denoise_tv_chambolle(img, weight=0.025, channel_axis=None), sigma_color=0.03, sigma_spatial=2.0, channel_axis=None)),
        ),
        Technique("Contrast Enhancement", "C1", "Global histogram equalisation", "grayscale", "M1 baseline candidate", "global equalize_hist", lambda img, ctx: metric_image(exposure.equalize_hist(img))),
        Technique("Contrast Enhancement", "C2", "Adaptive histogram equalisation", "grayscale", "M1 baseline candidate", "AHE-like un-clipped adaptive histogram equalisation", lambda img, ctx: adaptive_histogram_equalisation(img)),
        Technique("Contrast Enhancement", "C3", "CLAHE", "grayscale", "M1 baseline candidate", "clip_limit=0.025", lambda img, ctx: apply_clahe(img, clip_limit=0.025)),
        Technique("Contrast Enhancement", "C4", "Contrast stretching", "grayscale", "M1 candidate", "2nd to 98th percentile", lambda img, ctx: contrast_stretch(img)),
        Technique("Contrast Enhancement", "C5", "Gamma correction", "grayscale", "M1 candidate", "gamma=0.85", lambda img, ctx: gamma_correction(img, gamma=0.85)),
        Technique("Contrast Enhancement", "C6", "Homomorphic filtering", "grayscale", "M1 candidate", "log-domain high-frequency gain=0.70", lambda img, ctx: homomorphic_filter(img)),
        Technique("Fingerprint Ridge Enhancement", "G1", "Ordinary Gabor filter", "grayscale", "M2 baseline", "fixed frequency=0.115, 8 orientations", lambda img, ctx: multi_orientation_gabor(img, frequency=0.115, orientations=8, blend=0.30)),
        Technique(
            "Fingerprint Ridge Enhancement",
            "G2",
            "Literature-inspired Modified Gabor",
            "grayscale",
            "M2 literature technique",
            "local orientation + local ridge frequency; exact article parameters unavailable",
            lambda img, ctx: adaptive_gabor_enhancement(img, use_local_frequency=True, blend=0.18),
        ),
        Technique("Fingerprint Ridge Enhancement", "G3", "Adaptive local-orientation Gabor", "grayscale", "M2 candidate", "local orientation, fixed frequency", lambda img, ctx: adaptive_gabor_enhancement(img, use_local_frequency=False, blend=0.20)),
        Technique("Fingerprint Ridge Enhancement", "G4", "Adaptive orientation-frequency Gabor", "grayscale", "M2 candidate", "local orientation + local frequency", lambda img, ctx: adaptive_gabor_enhancement(img, use_local_frequency=True, blend=0.22)),
        Technique("Fingerprint Ridge Enhancement", "G5", "Log-Gabor", "grayscale", "M2 candidate", "local selection from log-Gabor frequency/orientation bank", lambda img, ctx: log_gabor_enhancement(img, blend=0.18)),
        Technique("Fingerprint Ridge Enhancement", "G6", "STFT contextual enhancement", "grayscale", "M2 candidate", "32px blocks, band-pass contextual filtering", lambda img, ctx: stft_contextual_enhancement(img, blend=0.18)),
        Technique("Fingerprint Ridge Enhancement", "G7", "Coherence-guided directional diffusion", "grayscale", "M2/M3 candidate", "directional smoothing weighted by coherence", lambda img, ctx: coherence_guided_diffusion(img)),
        Technique("Fingerprint Ridge Enhancement", "G8", "Directional oriented filter-bank", "grayscale", "M2 candidate", "orientation-selected elongated Gaussian smoothing", lambda img, ctx: directional_filter_bank(img)),
    ]


def build_screening_items(quality_df: pd.DataFrame, selected_paths: list[str]) -> list[dict[str, object]]:
    path_by_relative = {str(p.relative_to(PROJECT_ROOT)): p for p in image_files(REAL_FOLDER)}
    items = []
    selected_set = set(selected_paths)
    selected_quality = quality_df[quality_df["Path"].isin(selected_set)].set_index("Path")
    for index, rel_path in enumerate(selected_paths):
        path = path_by_relative[rel_path]
        reference = load_grayscale_metric(path)
        severity = float(SEVERITIES[index % len(SEVERITIES)])
        seed = RANDOM_SEED + index
        degraded, stages = simulate_degradation_stages(reference, severity=severity, seed=seed)
        reference_mask = fingerprint_mask(reference)
        common = common_preprocessing_variants(degraded)["P0"][1]
        items.append(
            {
                "index": index,
                "name": path.name,
                "path": path,
                "relative_path": rel_path,
                "reference": reference,
                "reference_mask": reference_mask,
                "severity": severity,
                "seed": seed,
                "blur_sigma": 0.35 + 1.8 * severity,
                "degraded": degraded,
                "degradation_stages": stages,
                "common": common,
                "quality_band": selected_quality.loc[rel_path, "Quality band vs Real"] if rel_path in selected_quality.index else "",
                "quality_score": float(selected_quality.loc[rel_path, "Diagnostic quality score"]) if rel_path in selected_quality.index else math.nan,
            }
        )
    return items


def run_quality_audit() -> pd.DataFrame:
    real_paths = deterministic_even_sample(image_files(REAL_FOLDER), QUALITY_SAMPLE_SIZE)
    rows = [quality_indicators(path, "Real") for path in real_paths]
    for split, folder in ALTERED_FOLDERS.items():
        rows.extend(quality_indicators(path, split) for path in deterministic_even_sample(image_files(folder), QUALITY_SAMPLE_SIZE))
    return add_quality_scores(pd.DataFrame(rows))


def run_preprocessing_audit(items: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for item in items:
        reference = item["reference"]
        reference_mask = item["reference_mask"]
        degraded = item["degraded"]
        degraded_metrics = evaluate_grayscale(reference, degraded, reference_mask, 0.0)
        for pipeline_id, (description, output) in common_preprocessing_variants(degraded).items():
            start = perf_counter()
            runtime = perf_counter() - start
            metrics = evaluate_grayscale(reference, output, reference_mask, runtime)
            rows.append(
                {
                    "Image": item["name"],
                    "Quality band": item["quality_band"],
                    "Severity": item["severity"],
                    "Pipeline ID": pipeline_id,
                    "Pipeline": description,
                    "Selected common preprocessing": pipeline_id == "P0",
                    **metrics,
                    "Delta PSNR vs degraded (dB)": metrics["PSNR (dB)"] - degraded_metrics["PSNR (dB)"],
                    "Delta SSIM vs degraded": metrics["SSIM"] - degraded_metrics["SSIM"],
                    "Delta coherence vs degraded": metrics["Ridge coherence"] - degraded_metrics["Ridge coherence"],
                }
            )
    return pd.DataFrame(rows)


def run_degradation_ablation(items: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for item in items:
        reference = item["reference"]
        reference_mask = item["reference_mask"]
        stages = item["degradation_stages"]
        previous_metrics = None
        for stage in stages:
            stage_metrics = evaluate_grayscale(reference, stage["Image"], reference_mask, 0.0)
            previous_psnr = previous_metrics["PSNR (dB)"] if previous_metrics else math.inf
            delta_psnr = (
                stage_metrics["PSNR (dB)"] - previous_psnr
                if previous_metrics and np.isfinite(previous_psnr)
                else math.nan
            )
            rows.append(
                {
                    "Ablation type": "Sequential current pipeline",
                    "Image": item["name"],
                    "Severity": item["severity"],
                    "Stage": stage["Stage"],
                    "Operation": stage["Operation"],
                    "Parameters": stage["Parameters"],
                    "PSNR before (dB)": previous_psnr,
                    "PSNR after (dB)": stage_metrics["PSNR (dB)"],
                    "Delta PSNR (dB)": delta_psnr,
                    "MSE after": stage_metrics["MSE"],
                    "SSIM after": stage_metrics["SSIM"],
                    "Ridge coherence after": stage_metrics["Ridge coherence"],
                    "Irreversible risk": stage["Irreversible risk"],
                }
            )
            previous_metrics = stage_metrics
        for operation in [
            "Gaussian noise only",
            "Salt-and-pepper noise only",
            "Blur only",
            "Contrast reduction + illumination only",
            "Partial smudge/occlusion only",
        ]:
            output, parameters, risk = independent_degradation_component(reference, operation, item["severity"], item["seed"])
            metrics = evaluate_grayscale(reference, output, reference_mask, 0.0)
            rows.append(
                {
                    "Ablation type": "Independent component",
                    "Image": item["name"],
                    "Severity": item["severity"],
                    "Stage": math.nan,
                    "Operation": operation,
                    "Parameters": parameters,
                    "PSNR before (dB)": math.inf,
                    "PSNR after (dB)": metrics["PSNR (dB)"],
                    "Delta PSNR (dB)": math.nan,
                    "MSE after": metrics["MSE"],
                    "SSIM after": metrics["SSIM"],
                    "Ridge coherence after": metrics["Ridge coherence"],
                    "Irreversible risk": risk,
                }
            )
    rows.append(
        {
            "Ablation type": "Not present in current pipeline",
            "Image": "ALL",
            "Severity": math.nan,
            "Stage": math.nan,
            "Operation": "Brightness/gamma only",
            "Parameters": "not an explicit current degradation stage",
            "PSNR before (dB)": math.nan,
            "PSNR after (dB)": math.nan,
            "Delta PSNR (dB)": math.nan,
            "MSE after": math.nan,
            "SSIM after": math.nan,
            "Ridge coherence after": math.nan,
            "Irreversible risk": "not applicable",
        }
    )
    rows.append(
        {
            "Ablation type": "Not present in current pipeline",
            "Image": "ALL",
            "Severity": math.nan,
            "Stage": math.nan,
            "Operation": "Resizing/downsampling",
            "Parameters": "not in simulate_degradation; loading uses fixed 256x256 alignment",
            "PSNR before (dB)": math.nan,
            "PSNR after (dB)": math.nan,
            "Delta PSNR (dB)": math.nan,
            "MSE after": math.nan,
            "SSIM after": math.nan,
            "Ridge coherence after": math.nan,
            "Irreversible risk": "not applicable",
        }
    )
    return pd.DataFrame(rows)


def run_single_technique_screening(items: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    minutiae_rows = []
    techniques = grayscale_techniques()
    for item in items:
        reference = item["reference"]
        reference_mask = item["reference_mask"]
        degraded = item["degraded"]
        common = item["common"]
        degraded_metrics = evaluate_grayscale(reference, degraded, reference_mask, 0.0)
        common_metrics = evaluate_grayscale(reference, common, reference_mask, 0.0)
        for control_id, control_name, control_role, image in [
            ("CONTROL-DEGRADED", "Degraded Input", "Input condition/control, not a technique", degraded),
            ("CONTROL-P0", "Common Preprocessed Input P0", "Common preprocessing control, not a member technique", common),
        ]:
            metrics = evaluate_grayscale(reference, image, reference_mask, 0.0)
            rows.append(
                {
                    "Image": item["name"],
                    "Quality band": item["quality_band"],
                    "Severity": item["severity"],
                    "Family": "Control",
                    "Technique ID": control_id,
                    "Technique": control_name,
                    "Metric domain": "grayscale",
                    "Role hint": control_role,
                    "Is image-processing technique": False,
                    "Notes": control_role,
                    **metrics,
                    "Delta PSNR vs degraded (dB)": metrics["PSNR (dB)"] - degraded_metrics["PSNR (dB)"],
                    "Delta PSNR vs common preprocessing (dB)": metrics["PSNR (dB)"] - common_metrics["PSNR (dB)"],
                    "PSNR delta to 28.17 dB": metrics["PSNR (dB)"] - LITERATURE_PSNR_BENCHMARK_DB,
                }
            )
        for technique in techniques:
            start = perf_counter()
            output = technique.function(common, item)
            runtime = perf_counter() - start
            metrics = evaluate_grayscale(reference, output, reference_mask, runtime)
            rows.append(
                {
                    "Image": item["name"],
                    "Quality band": item["quality_band"],
                    "Severity": item["severity"],
                    "Family": technique.family,
                    "Technique ID": technique.technique_id,
                    "Technique": technique.technique,
                    "Metric domain": technique.metric_domain,
                    "Role hint": technique.role_hint,
                    "Is image-processing technique": True,
                    "Notes": technique.notes,
                    **metrics,
                    "Delta PSNR vs degraded (dB)": metrics["PSNR (dB)"] - degraded_metrics["PSNR (dB)"],
                    "Delta PSNR vs common preprocessing (dB)": metrics["PSNR (dB)"] - common_metrics["PSNR (dB)"],
                    "PSNR delta to 28.17 dB": metrics["PSNR (dB)"] - LITERATURE_PSNR_BENCHMARK_DB,
                }
            )

        for tech_id, (technique, output, semantic) in segmentation_outputs(common).items():
            metrics = structural_metrics(output, common, reference_mask, semantic)
            rows.append(
                {
                    "Image": item["name"],
                    "Quality band": item["quality_band"],
                    "Severity": item["severity"],
                    "Family": "Segmentation",
                    "Technique ID": tech_id,
                    "Technique": technique,
                    "Metric domain": semantic,
                    "Role hint": "M3 candidate",
                    "Is image-processing technique": True,
                    "Notes": "Binary/structural output; PSNR/MSE/SSIM intentionally not calculated",
                    "MSE": math.nan,
                    "PSNR (dB)": math.nan,
                    "SSIM": math.nan,
                    "Ridge coherence": math.nan,
                    "Runtime (s)": math.nan,
                    "Delta PSNR vs degraded (dB)": math.nan,
                    "Delta PSNR vs common preprocessing (dB)": math.nan,
                    "PSNR delta to 28.17 dB": math.nan,
                    **metrics,
                }
            )
        for tech_id, (technique, output) in morphology_outputs(common).items():
            metrics = structural_metrics(output, common, reference_mask, "ridge map")
            rows.append(
                {
                    "Image": item["name"],
                    "Quality band": item["quality_band"],
                    "Severity": item["severity"],
                    "Family": "Morphological Ridge Restoration",
                    "Technique ID": tech_id,
                    "Technique": technique,
                    "Metric domain": "ridge map",
                    "Role hint": "M3 candidate",
                    "Is image-processing technique": True,
                    "Notes": "Binary ridge restoration; PSNR/MSE/SSIM intentionally not calculated",
                    "MSE": math.nan,
                    "PSNR (dB)": math.nan,
                    "SSIM": math.nan,
                    "Ridge coherence": math.nan,
                    "Runtime (s)": math.nan,
                    "Delta PSNR vs degraded (dB)": math.nan,
                    "Delta PSNR vs common preprocessing (dB)": math.nan,
                    "PSNR delta to 28.17 dB": math.nan,
                    **metrics,
                }
            )
        _, mask = ridge_base_binary(common)
        for tech_id, (technique, skeleton) in thinning_outputs(common).items():
            endpoints, bifurcations = crossing_number_points(skeleton, mask)
            filtered_endings = filter_close_points(endpoints, minimum_distance=7.0)
            filtered_bifurcations = filter_close_points(bifurcations, minimum_distance=7.0)
            metrics = structural_metrics(skeleton, common, reference_mask, "ridge map")
            rows.append(
                {
                    "Image": item["name"],
                    "Quality band": item["quality_band"],
                    "Severity": item["severity"],
                    "Family": "Thinning / Feature Extraction",
                    "Technique ID": tech_id,
                    "Technique": technique,
                    "Metric domain": "skeleton",
                    "Role hint": "M4 candidate",
                    "Is image-processing technique": True,
                    "Notes": "Skeleton/feature output; counts are detections, not ground-truth accuracy",
                    "MSE": math.nan,
                    "PSNR (dB)": math.nan,
                    "SSIM": math.nan,
                    "Ridge coherence": math.nan,
                    "Runtime (s)": math.nan,
                    "Delta PSNR vs degraded (dB)": math.nan,
                    "Delta PSNR vs common preprocessing (dB)": math.nan,
                    "PSNR delta to 28.17 dB": math.nan,
                    **metrics,
                }
            )
            minutiae_rows.append(
                {
                    "Image": item["name"],
                    "Quality band": item["quality_band"],
                    "Severity": item["severity"],
                    "Technique ID": tech_id,
                    "Technique": technique,
                    "Detected ridge endings": int(len(endpoints)),
                    "Detected bifurcations": int(len(bifurcations)),
                    "Filtered ridge endings": int(len(filtered_endings)),
                    "Filtered bifurcations": int(len(filtered_bifurcations)),
                    "Skeleton pixels": int(np.sum(skeleton)),
                    "Connected components": int(measure.label(skeleton, connectivity=2).max()),
                    "Counts are accuracy": False,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(minutiae_rows)


def summarise_single_techniques(single_df: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    grouped = single_df.groupby(["Family", "Technique ID", "Technique", "Metric domain", "Role hint", "Is image-processing technique"], dropna=False)
    for key, group in grouped:
        family, technique_id, technique, metric_domain, role_hint, is_technique = key
        row = {
            "Family": family,
            "Technique ID": technique_id,
            "Technique": technique,
            "Metric domain": metric_domain,
            "Role hint": role_hint,
            "Is image-processing technique": bool(is_technique),
            "Images": int(group["Image"].nunique()),
        }
        for column in [
            "MSE",
            "PSNR (dB)",
            "SSIM",
            "Ridge coherence",
            "Delta PSNR vs degraded (dB)",
            "Delta PSNR vs common preprocessing (dB)",
            "PSNR delta to 28.17 dB",
            "Runtime (s)",
            "Binary foreground/ridge ratio",
            "Connected components",
            "Small components <20px",
            "Largest component fraction",
            "Fragmentation per 1000 ridge pixels",
            "Skeleton pixels",
            "Skeleton endpoints",
            "Skeleton bifurcations",
            "Ridge continuity proxy",
            "Structural coherence",
            "Reference ROI IoU proxy",
        ]:
            if column in group:
                row[f"{column} mean"] = float(group[column].mean(skipna=True)) if group[column].notna().any() else math.nan
                row[f"{column} std"] = float(group[column].std(skipna=True)) if group[column].notna().sum() > 1 else math.nan
        summaries.append(row)
    return pd.DataFrame(summaries)


def safe_idxmax(frame: pd.DataFrame, column: str) -> int | None:
    valid = frame[frame[column].notna()]
    if valid.empty:
        return None
    return int(valid[column].idxmax())


def safe_idxmin(frame: pd.DataFrame, column: str) -> int | None:
    valid = frame[frame[column].notna()]
    if valid.empty:
        return None
    return int(valid[column].idxmin())


def select_family_winners(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = summary_df[summary_df["Is image-processing technique"] == True].copy()
    selection_rules = [
        ("Denoising", "PSNR (dB) mean", "max", "Best PSNR"),
        ("Restoration/Deblurring", "PSNR (dB) mean", "max", "Best PSNR"),
        ("Contrast Enhancement", "PSNR (dB) mean", "max", "Best PSNR with trade-off checked"),
        ("Fingerprint Ridge Enhancement", "Ridge coherence mean", "max", "Best ridge coherence"),
        ("Segmentation", "Reference ROI IoU proxy mean", "max", "Best foreground agreement proxy"),
        ("Morphological Ridge Restoration", "Fragmentation per 1000 ridge pixels mean", "min", "Lowest fragmentation"),
        ("Thinning / Feature Extraction", "Small components <20px mean", "min", "Fewest tiny skeleton components"),
    ]
    for family, metric, direction, label in selection_rules:
        family_frame = candidates[candidates["Family"] == family]
        if family_frame.empty or metric not in family_frame.columns:
            continue
        index = safe_idxmax(family_frame, metric) if direction == "max" else safe_idxmin(family_frame, metric)
        if index is None:
            continue
        record = family_frame.loc[index].to_dict()
        rows.append(
            {
                "Family": family,
                "Best Technique": record["Technique"],
                "Technique ID": record["Technique ID"],
                "Main Metric": label,
                "Result": record.get(metric, math.nan),
                "Metric column": metric,
            }
        )
    return pd.DataFrame(rows)


def choose_combination_candidates(summary_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    m1_frame = summary_df[
        (summary_df["Family"].isin(["Denoising", "Restoration/Deblurring"]))
        & (summary_df["Is image-processing technique"] == True)
    ].sort_values("PSNR (dB) mean", ascending=False)
    m2_by_coherence = summary_df[
        (summary_df["Family"] == "Fingerprint Ridge Enhancement") & (summary_df["Is image-processing technique"] == True)
    ].sort_values("Ridge coherence mean", ascending=False)
    m1_ids = m1_frame["Technique ID"].head(2).tolist()
    m2_ids = m2_by_coherence["Technique ID"].head(2).tolist()
    if "G2" not in m2_ids and "G2" in set(summary_df["Technique ID"]):
        m2_ids = (m2_ids + ["G2"])[:3]
    return m1_ids, m2_ids


def technique_by_id() -> dict[str, Technique]:
    return {tech.technique_id: tech for tech in grayscale_techniques()}


def conservative_fusion(base: np.ndarray, enhanced: np.ndarray, mask: np.ndarray | None = None, strength: float = 0.16) -> np.ndarray:
    base = metric_image(base)
    enhanced = metric_image(enhanced)
    mask = np.ones_like(base, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    coherence = structure_tensor_coherence_map(enhanced)
    weight = strength * np.clip(0.25 + 0.75 * coherence, 0.0, 1.0) * mask
    return metric_image((1.0 - weight) * base + weight * enhanced)


def run_combination_screening(items: list[dict[str, object]], summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = technique_by_id()
    m1_ids, m2_ids = choose_combination_candidates(summary_df)
    combos = []
    for m1_id in m1_ids:
        for m2_id in m2_ids:
            combos.append((f"{m1_id}->{m2_id}", m1_id, m2_id, False))
            combos.append((f"{m1_id}->{m2_id}->fusion", m1_id, m2_id, True))
    combos = combos[:12]
    rows = []
    stage_rows = []
    for combo_id, m1_id, m2_id, use_fusion in combos:
        m1 = registry[m1_id]
        m2 = registry[m2_id]
        for item in items:
            reference = item["reference"]
            reference_mask = item["reference_mask"]
            degraded = item["degraded"]
            common = item["common"]
            m1_start = perf_counter()
            m1_output = m1.function(common, item)
            m1_runtime = perf_counter() - m1_start
            m2_start = perf_counter()
            m2_output = m2.function(m1_output, item)
            m2_runtime = perf_counter() - m2_start
            final_output = conservative_fusion(m1_output, m2_output, fingerprint_mask(m1_output), strength=0.16) if use_fusion else m2_output
            stage_images = [
                ("Degraded", degraded, 0.0),
                ("Common Preprocessing P0", common, 0.0),
                (f"M1 {m1.technique}", m1_output, m1_runtime),
                (f"M2 {m2.technique}", m2_output, m2_runtime),
                ("Team Fusion" if use_fusion else "Team Final Output", final_output, m1_runtime + m2_runtime),
            ]
            previous = None
            for order, (stage_name, image, runtime) in enumerate(stage_images):
                metrics = evaluate_grayscale(reference, image, reference_mask, runtime)
                stage_rows.append(
                    {
                        "Combination ID": combo_id,
                        "Image": item["name"],
                        "Quality band": item["quality_band"],
                        "Severity": item["severity"],
                        "Stage order": order,
                        "Stage": stage_name,
                        **metrics,
                        "Delta PSNR vs previous stage (dB)": metrics["PSNR (dB)"] - previous["PSNR (dB)"] if previous else math.nan,
                        "Delta SSIM vs previous stage": metrics["SSIM"] - previous["SSIM"] if previous else math.nan,
                        "Delta coherence vs previous stage": metrics["Ridge coherence"] - previous["Ridge coherence"] if previous else math.nan,
                    }
                )
                previous = metrics
            final_metrics = evaluate_grayscale(reference, final_output, reference_mask, m1_runtime + m2_runtime)
            degraded_metrics = evaluate_grayscale(reference, degraded, reference_mask, 0.0)
            m3_binary = morphology_outputs(final_output)["M3-8"][1]
            m3_metrics = structural_metrics(m3_binary, final_output, reference_mask, "ridge map")
            rows.append(
                {
                    "Combination ID": combo_id,
                    "Image": item["name"],
                    "Quality band": item["quality_band"],
                    "Severity": item["severity"],
                    "M1 Technique ID": m1_id,
                    "M1 Technique": m1.technique,
                    "M2 Technique ID": m2_id,
                    "M2 Technique": m2.technique,
                    "Uses conservative fusion": use_fusion,
                    **final_metrics,
                    "Delta PSNR vs degraded (dB)": final_metrics["PSNR (dB)"] - degraded_metrics["PSNR (dB)"],
                    "PSNR delta to 28.17 dB": final_metrics["PSNR (dB)"] - LITERATURE_PSNR_BENCHMARK_DB,
                    "M3 controlled fragmentation": m3_metrics["Fragmentation per 1000 ridge pixels"],
                    "M3 controlled continuity proxy": m3_metrics["Ridge continuity proxy"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(stage_rows)


def summarise_combinations(combination_df: pd.DataFrame) -> pd.DataFrame:
    if combination_df.empty:
        return combination_df
    grouped = combination_df.groupby(
        ["Combination ID", "M1 Technique ID", "M1 Technique", "M2 Technique ID", "M2 Technique", "Uses conservative fusion"],
        as_index=False,
    )
    return grouped.agg(
        Images=("Image", "nunique"),
        MSE_mean=("MSE", "mean"),
        MSE_std=("MSE", "std"),
        PSNR_mean=("PSNR (dB)", "mean"),
        PSNR_std=("PSNR (dB)", "std"),
        SSIM_mean=("SSIM", "mean"),
        SSIM_std=("SSIM", "std"),
        Coherence_mean=("Ridge coherence", "mean"),
        Coherence_std=("Ridge coherence", "std"),
        Delta_PSNR_vs_degraded_mean=("Delta PSNR vs degraded (dB)", "mean"),
        PSNR_delta_to_28_17_mean=("PSNR delta to 28.17 dB", "mean"),
        Runtime_mean=("Runtime (s)", "mean"),
        M3_fragmentation_mean=("M3 controlled fragmentation", "mean"),
        M3_continuity_mean=("M3 controlled continuity proxy", "mean"),
    ).sort_values("PSNR_mean", ascending=False)


def extract_literature_context() -> dict[str, object]:
    pdf_path = PROJECT_ROOT / "specification" / "fingerprint enhancement.pdf"
    result = {
        "project_literature_pdf": str(pdf_path.relative_to(PROJECT_ROOT)) if pdf_path.exists() else None,
        "article_url": "https://fjpas.fuoye.edu.ng/index.php/fjpas/article/view/79",
        "article_pdf_url": "https://fjpas.fuoye.edu.ng/index.php/fjpas/article/download/79/71",
        "reported_modified_gabor_psnr_db": 28.17,
        "reported_modified_gabor_mse": 22.27,
        "reported_dataset": "Not confirmed from the locally extractable project-literature context",
        "reported_image_count": "Not confirmed from the locally extractable project-literature context",
        "web_access_note": "The article/PDF URLs returned a captcha/interstitial during this audit, so missing article details are not invented.",
        "exact_reproduction_possible": False,
        "reason_exact_reproduction_not_possible": "The local project literature gives the reported score and broad algorithm family, but not enough exact Modified Gabor parameter details for a verified replication.",
        "extracted_context": "",
    }
    if not pdf_path.exists():
        return result
    data = pdf_path.read_bytes()
    streams = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        try:
            streams.append(zlib.decompress(match.group(1).strip(b"\r\n")))
        except Exception:
            pass
    cmap: dict[int, str] = {}
    for stream in streams:
        text = stream.decode("latin1", errors="ignore")
        for block in re.finditer(r"beginbfchar(.*?)endbfchar", text, re.S):
            for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>", block.group(1)):
                try:
                    cmap[int(src, 16)] = chr(int(dst, 16))
                except Exception:
                    pass
        for block in re.finditer(r"beginbfrange(.*?)endbfrange", text, re.S):
            for src1, src2, dst in re.findall(r"<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>", block.group(1)):
                start, end, dst_start = int(src1, 16), int(src2, 16), int(dst, 16)
                for offset, code in enumerate(range(start, end + 1)):
                    try:
                        cmap[code] = chr(dst_start + offset)
                    except Exception:
                        pass

    def decode_hex(hex_text: str) -> str:
        if len(hex_text) % 4 != 0:
            return ""
        return "".join(cmap.get(int(hex_text[index : index + 4], 16), "") for index in range(0, len(hex_text), 4))

    parts: list[str] = []
    for stream in streams:
        text = stream.decode("latin1", errors="ignore")
        for hex_text in re.findall(r"<([0-9A-Fa-f]+)>\s*Tj", text):
            decoded = decode_hex(hex_text)
            if decoded:
                parts.append(decoded)
        for array_text in re.findall(r"\[(.*?)\]\s*TJ", text, re.S):
            decoded = "".join(decode_hex(hex_text) for hex_text in re.findall(r"<([0-9A-Fa-f]+)>", array_text))
            if decoded:
                parts.append(decoded)
    extracted = re.sub(r"\s+", " ", " ".join(parts))
    index = extracted.find("28.17")
    if index >= 0:
        result["extracted_context"] = extracted[max(0, index - 450) : index + 650]
    return result


def create_figures(
    quality_df: pd.DataFrame,
    preprocessing_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    family_winners_df: pd.DataFrame,
    combination_summary_df: pd.DataFrame,
    minutiae_df: pd.DataFrame,
) -> list[str]:
    figure_paths: list[str] = []

    def savefig(name: str) -> None:
        path = OUTPUT_DIR / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        figure_paths.append(str(path.relative_to(PROJECT_ROOT)))

    plt.figure(figsize=(10, 5))
    order = ["Real", "Altered Easy", "Altered Medium", "Altered Hard"]
    data = [quality_df.loc[quality_df["Dataset split"] == split, "Diagnostic quality score"].dropna().values for split in order]
    plt.boxplot(data, tick_labels=order, showfliers=False)
    plt.ylabel("Diagnostic quality score")
    plt.title("FIGURE A - Dataset/Input Quality")
    savefig("screening24_figure_A_dataset_quality.png")

    pre_summary = preprocessing_df.groupby("Pipeline ID", as_index=False).agg(PSNR=("PSNR (dB)", "mean"), SSIM=("SSIM", "mean"), Coherence=("Ridge coherence", "mean"))
    plt.figure(figsize=(8, 5))
    plt.bar(pre_summary["Pipeline ID"], pre_summary["PSNR"], color="#4C78A8")
    plt.ylabel("Mean PSNR (dB)")
    plt.title("FIGURE B - Common Preprocessing Comparison")
    savefig("screening24_figure_B_preprocessing.png")

    m1 = summary_df[summary_df["Family"].isin(["Denoising", "Restoration/Deblurring"])]
    m1 = m1[m1["Is image-processing technique"] == True].sort_values("PSNR (dB) mean", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(m1["Technique"], m1["PSNR (dB) mean"], color="#59A14F")
    plt.xlabel("Mean PSNR (dB)")
    plt.gca().invert_yaxis()
    plt.title("FIGURE C - M1 Technique Screening")
    savefig("screening24_figure_C_m1_screening.png")

    m2 = summary_df[(summary_df["Family"] == "Fingerprint Ridge Enhancement") & (summary_df["Is image-processing technique"] == True)]
    plt.figure(figsize=(9, 6))
    plt.scatter(m2["PSNR (dB) mean"], m2["Ridge coherence mean"], color="#E15759")
    for _, row in m2.iterrows():
        plt.annotate(row["Technique ID"], (row["PSNR (dB) mean"], row["Ridge coherence mean"]), fontsize=8)
    plt.xlabel("Mean PSNR (dB)")
    plt.ylabel("Mean ridge coherence")
    plt.title("FIGURE D - M2 Ridge Enhancement Screening")
    savefig("screening24_figure_D_m2_ridge_enhancement.png")

    m3 = summary_df[summary_df["Family"].isin(["Segmentation", "Morphological Ridge Restoration"])]
    m3 = m3[m3["Is image-processing technique"] == True].copy()
    plt.figure(figsize=(10, 6))
    plt.barh(m3["Technique ID"] + " " + m3["Technique"], m3["Fragmentation per 1000 ridge pixels mean"], color="#F28E2B")
    plt.xlabel("Mean fragmentation per 1000 ridge pixels")
    plt.gca().invert_yaxis()
    plt.title("FIGURE E - M3 Segmentation/Morphology Results")
    savefig("screening24_figure_E_m3_results.png")

    if not minutiae_df.empty:
        min_summary = minutiae_df.groupby("Technique", as_index=False).agg(Endings=("Filtered ridge endings", "mean"), Bifurcations=("Filtered bifurcations", "mean"))
        x = np.arange(len(min_summary))
        plt.figure(figsize=(9, 5))
        plt.bar(x - 0.18, min_summary["Endings"], width=0.36, label="Filtered endings", color="#B07AA1")
        plt.bar(x + 0.18, min_summary["Bifurcations"], width=0.36, label="Filtered bifurcations", color="#76B7B2")
        plt.xticks(x, min_summary["Technique"], rotation=20, ha="right")
        plt.ylabel("Mean detected count")
        plt.legend()
        plt.title("FIGURE F - M4 Thinning/Minutiae Results")
        savefig("screening24_figure_F_m4_results.png")

    if not combination_summary_df.empty:
        combo_plot = combination_summary_df.sort_values("PSNR_mean", ascending=False).head(12)
        plt.figure(figsize=(10, 6))
        plt.barh(combo_plot["Combination ID"], combo_plot["PSNR_mean"], color="#EDC948")
        plt.xlabel("Mean PSNR (dB)")
        plt.gca().invert_yaxis()
        plt.title("FIGURE G - Team Hybrid Comparison")
        savefig("screening24_figure_G_team_hybrid.png")

        benchmark = pd.concat(
            [
                summary_df[summary_df["Metric domain"] == "grayscale"][["Technique", "PSNR (dB) mean"]].rename(columns={"PSNR (dB) mean": "PSNR"}),
                combo_plot[["Combination ID", "PSNR_mean"]].rename(columns={"Combination ID": "Technique", "PSNR_mean": "PSNR"}),
            ],
            ignore_index=True,
        ).sort_values("PSNR", ascending=False).head(14)
        plt.figure(figsize=(10, 6))
        plt.barh(benchmark["Technique"], benchmark["PSNR"], color="#9C755F")
        plt.axvline(LITERATURE_PSNR_BENCHMARK_DB, color="black", linestyle="--", linewidth=1.5, label="28.17 dB target")
        plt.xlabel("Mean PSNR (dB)")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title("FIGURE H - Tutor Benchmark Comparison")
        savefig("screening24_figure_H_tutor_benchmark.png")
    return figure_paths


def fmt_mean_std(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna()
    if values.empty:
        return "n/a"
    return f"{values.mean():.4f} +/- {values.std():.4f}"


def row_by_metric(frame: pd.DataFrame, metric: str, direction: str = "max") -> pd.Series:
    valid = frame[frame[metric].notna()]
    if valid.empty:
        return pd.Series(dtype=object)
    return valid.loc[valid[metric].idxmax() if direction == "max" else valid[metric].idxmin()]


def create_member_results(
    summary_df: pd.DataFrame,
    family_winners_df: pd.DataFrame,
    combination_summary_df: pd.DataFrame,
    minutiae_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    grayscale = summary_df[summary_df["Metric domain"] == "grayscale"].copy()
    m1_pool = grayscale[grayscale["Family"].isin(["Denoising", "Restoration/Deblurring"])]
    m1_proposed = row_by_metric(m1_pool, "PSNR (dB) mean", "max")
    clahe = grayscale[grayscale["Technique ID"] == "C3"]
    m2_pool = grayscale[grayscale["Family"] == "Fingerprint Ridge Enhancement"]
    m2_proposed = row_by_metric(m2_pool, "Ridge coherence mean", "max")
    modified = grayscale[grayscale["Technique ID"] == "G2"]
    m3_pool = summary_df[summary_df["Family"] == "Morphological Ridge Restoration"]
    m3_proposed = row_by_metric(m3_pool, "Fragmentation per 1000 ridge pixels mean", "min")
    m4_pool = summary_df[summary_df["Family"] == "Thinning / Feature Extraction"]
    m4_proposed = row_by_metric(m4_pool, "Small components <20px mean", "min")
    best_combo = combination_summary_df.iloc[0] if not combination_summary_df.empty else pd.Series(dtype=object)

    rows.append({"Member": "M1", "Role": "Baseline", "Recommended technique": clahe.iloc[0]["Technique"] if not clahe.empty else "CLAHE", "Evidence metric": "contrast baseline; see PSNR/SSIM trade-off"})
    rows.append({"Member": "M1", "Role": "Proposed", "Recommended technique": m1_proposed.get("Technique", ""), "Evidence metric": f"PSNR mean {m1_proposed.get('PSNR (dB) mean', math.nan):.4f} dB"})
    rows.append({"Member": "M2", "Role": "Baseline", "Recommended technique": "Ordinary Gabor filter", "Evidence metric": "fixed-orientation/frequency baseline"})
    rows.append({"Member": "M2", "Role": "Literature technique", "Recommended technique": modified.iloc[0]["Technique"] if not modified.empty else "Literature-inspired Modified Gabor", "Evidence metric": f"our PSNR mean {modified.iloc[0]['PSNR (dB) mean']:.4f} dB; literature reported 28.17 dB" if not modified.empty else ""})
    rows.append({"Member": "M2", "Role": "Proposed", "Recommended technique": m2_proposed.get("Technique", ""), "Evidence metric": f"coherence mean {m2_proposed.get('Ridge coherence mean', math.nan):.4f}"})
    rows.append({"Member": "M3", "Role": "Proposed", "Recommended technique": m3_proposed.get("Technique", ""), "Evidence metric": f"fragmentation {m3_proposed.get('Fragmentation per 1000 ridge pixels mean', math.nan):.4f}"})
    rows.append({"Member": "M4", "Role": "Proposed", "Recommended technique": m4_proposed.get("Technique", ""), "Evidence metric": f"small components {m4_proposed.get('Small components <20px mean', math.nan):.4f}"})
    rows.append({"Member": "Team", "Role": "Hybrid", "Recommended technique": best_combo.get("Combination ID", ""), "Evidence metric": f"PSNR mean {best_combo.get('PSNR_mean', math.nan):.4f} dB"})
    return pd.DataFrame(rows)


def create_report(
    quality_df: pd.DataFrame,
    preprocessing_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    single_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    family_winners_df: pd.DataFrame,
    member_df: pd.DataFrame,
    combination_df: pd.DataFrame,
    combination_summary_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    minutiae_df: pd.DataFrame,
    metadata: dict[str, object],
) -> str:
    literature = metadata["literature_context"]
    real = quality_df[quality_df["Dataset split"] == "Real"]
    altered_summary = quality_df.groupby("Dataset split").agg(
        Images=("Image", "count"),
        Quality=("Diagnostic quality score", "mean"),
        Coherence=("Ridge coherence", "mean"),
        Sharpness=("Sharpness Laplacian variance", "mean"),
        Contrast=("Local contrast", "mean"),
        Foreground=("Foreground usable-area ratio", "mean"),
    )
    pre_summary = preprocessing_df.groupby("Pipeline ID").agg(
        PSNR=("PSNR (dB)", "mean"),
        SSIM=("SSIM", "mean"),
        Coherence=("Ridge coherence", "mean"),
        DeltaPSNR=("Delta PSNR vs degraded (dB)", "mean"),
    )
    seq = degradation_df[degradation_df["Ablation type"] == "Sequential current pipeline"]
    seq = seq[np.isfinite(seq["Delta PSNR (dB)"])]
    seq_loss = seq.groupby("Operation", as_index=False)["Delta PSNR (dB)"].mean().sort_values("Delta PSNR (dB)")
    worst_degradation = seq_loss.iloc[0] if not seq_loss.empty else pd.Series(dtype=object)
    independent = degradation_df[degradation_df["Ablation type"] == "Independent component"]
    independent_loss = independent.groupby("Operation", as_index=False)["PSNR after (dB)"].mean().sort_values("PSNR after (dB)")
    technique_summary = summary_df[summary_df["Is image-processing technique"] == True]
    gray_summary = technique_summary[technique_summary["Metric domain"] == "grayscale"]
    best_psnr = row_by_metric(gray_summary, "PSNR (dB) mean", "max")
    combo_best = combination_summary_df.iloc[0] if not combination_summary_df.empty else pd.Series(dtype=object)
    if not combo_best.empty and combo_best.get("PSNR_mean", -math.inf) > best_psnr.get("PSNR (dB) mean", -math.inf):
        overall_best_name = combo_best["Combination ID"]
        overall_best_psnr = float(combo_best["PSNR_mean"])
        overall_best_std = float(combo_best["PSNR_std"])
    else:
        overall_best_name = best_psnr.get("Technique", "")
        overall_best_psnr = float(best_psnr.get("PSNR (dB) mean", math.nan))
        overall_best_std = float(best_psnr.get("PSNR (dB) std", math.nan))
    best_ssim = row_by_metric(gray_summary, "SSIM mean", "max")
    best_coherence = row_by_metric(gray_summary, "Ridge coherence mean", "max")
    stage_delta_summary = (
        stage_df.groupby("Stage", as_index=False)
        .agg(
            Delta_PSNR=("Delta PSNR vs previous stage (dB)", "mean"),
            Delta_SSIM=("Delta SSIM vs previous stage", "mean"),
            Delta_Coherence=("Delta coherence vs previous stage", "mean"),
        )
        .dropna(subset=["Delta_PSNR"])
    )
    stage_gain = row_by_metric(stage_delta_summary, "Delta_PSNR", "max")
    stage_loss = row_by_metric(stage_delta_summary, "Delta_PSNR", "min")
    selected_names = metadata["screening24_selected_images"]
    modified = gray_summary[gray_summary["Technique ID"] == "G2"]

    lines = [
        "# screening24 Master 24-Image Screening and Experiment-Design Audit",
        "",
        "## A. Original Real Image Quality",
        f"- Real audit sample: {len(real)} images. Diagnostic score bands are mixed by construction: {real['Quality band vs Real'].value_counts().to_dict()}. Real is therefore treated as mixed quality, not automatically high-quality.",
        f"- Real indicators: coherence {fmt_mean_std(real, 'Ridge coherence')}, sharpness {fmt_mean_std(real, 'Sharpness Laplacian variance')}, local contrast {fmt_mean_std(real, 'Local contrast')}.",
        "",
        "## B. Real vs Altered Quality",
        altered_summary.round(4).to_markdown(),
        "- Altered folders are judged using no-reference structure indicators; raw PSNR/MSE/SSIM against Real is not used because Altered images can be geometrically unaligned.",
        "",
        "## C. Is Synthetic Degradation Necessary?",
        "- Only for controlled quantitative evaluation. Real and Altered are useful for structure/quality audits, but controlled degradation keeps the clean Real reference aligned for MSE/PSNR/SSIM.",
        "- Dataset-provided Altered images are valuable for qualitative robustness tests, but they should not be used for full-reference metrics without registration.",
        "",
        "## D. Current Degradation Pipeline",
        "- Clean Real reference -> Gaussian blur -> dynamic-range reduction + sinusoidal illumination -> Gaussian sensor noise -> sparse salt-and-pepper noise -> partial smudge/occlusion -> final clipping.",
        "- Severity schedule is unchanged from the project: 0.35, 0.55, 0.75 repeated over the 24 selected images; seeds are RANDOM_SEED + image_index.",
        "",
        "## E. Degradation Ablation",
        f"- Most destructive sequential stage by mean delta PSNR: {worst_degradation.get('Operation', 'n/a')} ({worst_degradation.get('Delta PSNR (dB)', math.nan):.4f} dB).",
        independent_loss.round(4).to_markdown(index=False),
        "- Potentially irreversible operations are blur, local smudge/occlusion, impulse-pixel clipping, and final saturation.",
        "",
        "## F. Common Preprocessing Candidates",
        pre_summary.round(4).to_markdown(),
        "",
        "## G. Selected Common Preprocessing",
        "- Selected P0: grayscale + resize + float [0,1] only.",
        "- Justification: it is neutral, preserves the current input condition, and does not move tested denoising/contrast/ridge techniques into the shared preprocessing stage.",
        "",
        "## H. Denoising Family Results",
        summary_df[summary_df["Family"] == "Denoising"].sort_values("PSNR (dB) mean", ascending=False).round(4).to_markdown(index=False),
        "",
        "## I. Restoration Family Results",
        summary_df[summary_df["Family"] == "Restoration/Deblurring"].sort_values("PSNR (dB) mean", ascending=False).round(4).to_markdown(index=False),
        "",
        "## J. Contrast Enhancement Family Results",
        summary_df[summary_df["Family"] == "Contrast Enhancement"].sort_values("PSNR (dB) mean", ascending=False).round(4).to_markdown(index=False),
        "",
        "## K. Ridge Enhancement Family Results",
        summary_df[summary_df["Family"] == "Fingerprint Ridge Enhancement"].sort_values("Ridge coherence mean", ascending=False).round(4).to_markdown(index=False),
        "",
        "## L. Modified Gabor Result",
        f"- Reported Literature PSNR = 28.17 dB for Modified Gabor, with MSE 22.27, from Mukaila/Olagunju et al. as cited in the local project literature and article source.",
        f"- Literature source URL: {literature['article_url']} ; PDF URL: {literature['article_pdf_url']}.",
        f"- Dataset used: {literature['reported_dataset']}. Number of images: {literature['reported_image_count']}.",
        f"- Known preprocessing/evaluation from local extractable source: compared Modified Gabor Filter, High Boost Gaussian Filter and Fuzzy-Based Filter, evaluated for noise reduction and ridge clarity; exact reference/evaluation design is not fully available here. Web access note: {literature['web_access_note']}",
        f"- Our implemented Literature-inspired Modified Gabor PSNR = {modified.iloc[0]['PSNR (dB) mean']:.4f} +/- {modified.iloc[0]['PSNR (dB) std']:.4f} dB." if not modified.empty else "- Our implemented Literature-inspired Modified Gabor result was not available.",
        "- Exact reproduction is not claimed because the available source gives broad method/result information but not enough implementation parameters.",
        "",
        "## M. Segmentation Results",
        summary_df[summary_df["Family"] == "Segmentation"].round(4).to_markdown(index=False),
        "",
        "## N. Morphological Ridge Restoration Results",
        summary_df[summary_df["Family"] == "Morphological Ridge Restoration"].round(4).to_markdown(index=False),
        "",
        "## O. Thinning Results",
        summary_df[summary_df["Family"] == "Thinning / Feature Extraction"].round(4).to_markdown(index=False),
        "",
        "## P. Minutiae Extraction Results",
        minutiae_df.groupby("Technique").agg(Images=("Image", "nunique"), Filtered_endings=("Filtered ridge endings", "mean"), Filtered_bifurcations=("Filtered bifurcations", "mean"), Skeleton_pixels=("Skeleton pixels", "mean")).round(4).to_markdown(),
        "- These are detected counts, not accuracy, because no ground-truth minutiae annotations are available.",
        "",
        "## Q. Best Technique From Each Family",
        family_winners_df.round(4).to_markdown(index=False),
        "",
        "## R. M1 Recommended Technique",
        member_df[member_df["Member"] == "M1"].to_markdown(index=False),
        "",
        "## S. M2 Recommended Technique",
        member_df[member_df["Member"] == "M2"].to_markdown(index=False),
        "",
        "## T. M3 Recommended Technique",
        member_df[member_df["Member"] == "M3"].to_markdown(index=False),
        "",
        "## U. M4 Recommended Technique",
        member_df[member_df["Member"] == "M4"].to_markdown(index=False),
        "",
        "## V. Controlled Combination Results",
        combination_summary_df.round(4).to_markdown(index=False),
        "",
        "## W. Best Team Hybrid",
        f"- {combo_best.get('Combination ID', 'n/a')} with mean PSNR {combo_best.get('PSNR_mean', math.nan):.4f} +/- {combo_best.get('PSNR_std', math.nan):.4f} dB and coherence {combo_best.get('Coherence_mean', math.nan):.4f}.",
        "",
        "## X. Best Mean PSNR",
        f"- Highest legitimate mean PSNR in this screening: {overall_best_name} = {overall_best_psnr:.4f} +/- {overall_best_std:.4f} dB.",
        "",
        "## Y. Gap To Tutor Target 28.17 dB",
        f"- Signed gap: {overall_best_psnr - LITERATURE_PSNR_BENCHMARK_DB:.4f} dB. Target exceeded: {overall_best_psnr > LITERATURE_PSNR_BENCHMARK_DB}.",
        "",
        "## Z. Best SSIM",
        f"- {best_ssim.get('Technique', 'n/a')} with SSIM {best_ssim.get('SSIM mean', math.nan):.4f}.",
        "",
        "## AA. Best Ridge Coherence",
        f"- {best_coherence.get('Technique', 'n/a')} with coherence {best_coherence.get('Ridge coherence mean', math.nan):.4f}.",
        "",
        "## AB. Stage Where PSNR Is Most Improved",
        f"- {stage_gain.get('Stage', 'n/a')}: mean delta {stage_gain.get('Delta_PSNR', math.nan):.4f} dB across combination-stage records.",
        "",
        "## AC. Stage Where PSNR Is Most Damaged",
        f"- {stage_loss.get('Stage', 'n/a')}: mean delta {stage_loss.get('Delta_PSNR', math.nan):.4f} dB across combination-stage records.",
        "",
        "## AD. Metric Integrity Verification",
        "- MSE/PSNR/SSIM are calculated only between aligned clean Real references and aligned greyscale outputs clipped to [0,1]. No independent per-output metric normalisation is used.",
        "- Binary masks, morphology outputs and skeletons are evaluated with structural metrics rather than greyscale PSNR/MSE/SSIM.",
        "",
        "## AE. Reference-Leakage Verification",
        "- Clean Real pixels are used only for evaluation and for no-reference quality selection labels after metrics are computed. Enhancement functions receive only degraded/common-preprocessed images and fixed parameters.",
        "- No per-image tuning, SVM training, deep learning or Real-pixel restoration input is used.",
        "",
        "## AF. Existing Evidence Preservation",
        f"- Protected output hashes unchanged during this screening run: {metadata['protected_outputs_unchanged']}.",
        "- Existing 500-image formal evidence and advanced_dev_* files were not regenerated.",
        "",
        "## AG. Exact Files Changed",
        "\n".join(f"- {path}" for path in metadata["created_or_updated_files"]),
        "",
        "## AH. Runtime Summary",
        f"- Total runtime: {metadata['runtime_seconds']:.2f} seconds. Quality audit sample: {metadata['quality_audit_counts']}. Screening set: {len(selected_names)} images.",
        "",
        "## AI. Recommended NEXT experiment only",
        "- Review this screening first. The next experiment should adjust the degradation model only if the team agrees the present synthetic degradation is too destructive or unrealistic; do not scale to 50/100/500 yet.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    quality_df: pd.DataFrame,
    preprocessing_df: pd.DataFrame,
    degradation_df: pd.DataFrame,
    single_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    family_winners_df: pd.DataFrame,
    member_df: pd.DataFrame,
    combination_df: pd.DataFrame,
    combination_summary_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    minutiae_df: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    quality_df.to_csv(SCREENING_FILES["dataset_quality"], index=False)
    preprocessing_df.to_csv(SCREENING_FILES["preprocessing"], index=False)
    degradation_df.to_csv(SCREENING_FILES["degradation_ablation"], index=False)
    single_df.to_csv(SCREENING_FILES["single_techniques"], index=False)
    combined_family = pd.concat(
        [
            summary_df.assign(Table="Technique summary"),
            family_winners_df.rename(columns={"Best Technique": "Technique"}).assign(Table="Family winners"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined_family.to_csv(SCREENING_FILES["family_summary"], index=False)
    member_df.to_csv(SCREENING_FILES["member_results"], index=False)
    combination_summary_df.to_csv(SCREENING_FILES["combinations"], index=False)
    stage_df.to_csv(SCREENING_FILES["stage_metrics"], index=False)
    minutiae_df.to_csv(SCREENING_FILES["minutiae"], index=False)
    SCREENING_FILES["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = create_report(
        quality_df,
        preprocessing_df,
        degradation_df,
        single_df,
        summary_df,
        family_winners_df,
        member_df,
        combination_df,
        combination_summary_df,
        stage_df,
        minutiae_df,
        metadata,
    )
    SCREENING_FILES["report"].write_text(report, encoding="utf-8")


def main() -> None:
    start = perf_counter()
    protected_before = snapshot_files(PROTECTED_OUTPUTS)
    print("screening24 master audit started")
    print(f"Project root: {PROJECT_ROOT}")
    print("Phase A/B: quality audit")
    quality_df = run_quality_audit()
    selected_paths = select_screening_paths(quality_df)
    quality_df.loc[quality_df["Path"].isin(selected_paths), "Selected for screening24"] = True
    print(f"Selected screening images: {len(selected_paths)}")
    items = build_screening_items(quality_df, selected_paths)
    print("Phase C: preprocessing audit")
    preprocessing_df = run_preprocessing_audit(items)
    print("Phase E: degradation ablation")
    degradation_df = run_degradation_ablation(items)
    print("Phase G/M3/M4: single-technique screening")
    single_df, minutiae_df = run_single_technique_screening(items)
    summary_df = summarise_single_techniques(single_df)
    family_winners_df = select_family_winners(summary_df)
    print("Phase K: controlled combination screening")
    combination_df, stage_df = run_combination_screening(items, summary_df)
    combination_summary_df = summarise_combinations(combination_df)
    member_df = create_member_results(summary_df, family_winners_df, combination_summary_df, minutiae_df)
    print("Figures and report")
    figure_paths = create_figures(quality_df, preprocessing_df, summary_df, family_winners_df, combination_summary_df, minutiae_df)
    protected_after = snapshot_files(PROTECTED_OUTPUTS)
    protected_unchanged = protected_before == protected_after
    runtime_seconds = perf_counter() - start
    created_files = [str(path.relative_to(PROJECT_ROOT)) for path in SCREENING_FILES.values()]
    created_files.extend(figure_paths)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "development_mode_required": True,
        "export_final_results_required": False,
        "formal_500_rerun": False,
        "advanced_dev_outputs_overwritten": False,
        "quality_audit_sample_size_per_split": QUALITY_SAMPLE_SIZE,
        "quality_audit_counts": quality_df["Dataset split"].value_counts().to_dict(),
        "screening24_sample_size": SCREENING_SAMPLE_SIZE,
        "screening24_selection_rule": "500-image Real quality audit; select 8 higher, 8 medium, 8 lower diagnostic-quality images by fixed quantile bands; no technique-score selection.",
        "screening24_selected_images": [item["name"] for item in items],
        "screening24_selected_relative_paths": selected_paths,
        "severity_schedule": list(SEVERITIES),
        "seed_rule": "RANDOM_SEED + screening_image_index",
        "selected_common_preprocessing": "P0: grayscale + resize + float [0,1] only",
        "metric_integrity": {
            "mse_psnr_ssim_domain": "aligned greyscale outputs only",
            "data_range": 1.0,
            "candidate_metric_normalisation": "clip only; no independent contrast normalisation before metrics",
            "binary_outputs": "structural metrics only",
        },
        "reference_leakage": {
            "clean_real_used_for_enhancement": False,
            "per_image_parameter_tuning": False,
            "svm_enabled": False,
            "deep_learning_used": False,
        },
        "literature_context": extract_literature_context(),
        "protected_outputs_before": protected_before,
        "protected_outputs_after": protected_after,
        "protected_outputs_unchanged": protected_unchanged,
        "created_or_updated_files": created_files,
        "runtime_seconds": runtime_seconds,
        "stop_gate": "Completed 24-image screening only; did not run 50/100/250/500-image experiments.",
    }
    write_outputs(
        quality_df,
        preprocessing_df,
        degradation_df,
        single_df,
        summary_df,
        family_winners_df,
        member_df,
        combination_df,
        combination_summary_df,
        stage_df,
        minutiae_df,
        metadata,
    )
    print(f"screening24 master audit complete in {runtime_seconds:.2f} seconds")
    print(f"Report: {SCREENING_FILES['report']}")


if __name__ == "__main__":
    main()
