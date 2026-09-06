"""BMDS2133 Fingerprint Enhancement System.

Mode A comparative and enhancement study.

The workflow uses four individual image-processing techniques.  The techniques
are tuned on a subject-disjoint 500-image
development set, frozen, and then evaluated on a separate 1000-image validation
set from unseen subjects.
"""
from __future__ import annotations

import argparse
import os
import hashlib
import importlib.util
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


def _install_missing_runtime_packages() -> None:
    required_packages = {
        "cv2": "opencv-python>=5,<6",
        "numpy": "numpy>=2.4,<2.5",
        "PIL": "pillow>=12,<13",
        "pandas": "pandas>=3,<3.1",
        "scipy": "scipy>=1.16,<2",
        "skimage": "scikit-image>=0.26,<0.27",
        "matplotlib": "matplotlib>=3.11,<4",
        "nbconvert": "nbconvert>=7,<8",
        "ipykernel": "ipykernel>=6,<7",
    }
    missing = [package for import_name, package in required_packages.items() if importlib.util.find_spec(import_name) is None]
    if missing:
        print("Installing missing packages:", ", ".join(missing), flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


_install_missing_runtime_packages()

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, restoration, transform
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

warnings.filterwarnings("ignore", category=FutureWarning)
cv2.setNumThreads(1)

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data" / "SOCOFing"
REAL_DIR = DATASET_ROOT / "Real"
ALTERED_DIR = DATASET_ROOT / "Altered"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "final_mode_a"
MAX_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 1))
SERIAL_MEMBERS = {"M2 Modified Gabor"}

IMAGE_SIZE = (256, 256)
SPLIT_SEED = "BMDS2133-Fingerprint-ModeA-20260902"
NOISE_SEED_BASE = 20260903
ALTERED_SEED = 20260904
DEVELOPMENT_SUBJECTS = 50
VALIDATION_SUBJECTS = 100
EXPECTED_IMAGES_PER_SUBJECT = 10
DEVELOPMENT_COUNT = DEVELOPMENT_SUBJECTS * EXPECTED_IMAGES_PER_SUBJECT
VALIDATION_COUNT = VALIDATION_SUBJECTS * EXPECTED_IMAGES_PER_SUBJECT
ALTERED_PER_SEVERITY_COUNT = 500
CONTROLLED_SEVERITY = 0.55
GAUSSIAN_SIGMA = 0.018 + 0.055 * CONTROLLED_SEVERITY
NLM_TEMPLATE_WINDOW_SIZE = 7
NLM_SEARCH_WINDOW_SIZE = 21
SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

NLM_H_CANDIDATES = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
GABOR_CANDIDATES = [
    {"frequencies": (0.075, 0.095, 0.115, 0.140), "orientation_bins": 8, "blend": 0.12},
    {"frequencies": (0.075, 0.095, 0.115, 0.140), "orientation_bins": 8, "blend": 0.18},
    {"frequencies": (0.075, 0.095, 0.115, 0.140), "orientation_bins": 12, "blend": 0.18},
]
TV_CANDIDATES = [{"weight": 0.01}, {"weight": 0.02}, {"weight": 0.04}, {"weight": 0.06}]
DIFFUSION_CANDIDATES = [
    {"iterations": 2, "step": 0.12},
    {"iterations": 4, "step": 0.16},
    {"iterations": 6, "step": 0.14},
    {"iterations": 4, "step": 0.22},
]


@dataclass(frozen=True)
class FrozenConfiguration:
    selected_method: str
    selected_member: str
    selected_parameters: dict[str, Any]
    all_member_parameters: dict[str, dict[str, Any]]
    ranking_rule: str
    selected_using: str
    frozen_before_validation: bool


def _metric_image(image: np.ndarray) -> np.ndarray:
    base = np.asarray(image, dtype=np.float32)
    return np.clip(np.nan_to_num(base, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def normalise_image(image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    base = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(base, [low_percentile, high_percentile])
    if high - low < 1e-8:
        return _metric_image(base)
    return _metric_image((base - low) / (high - low))


def preprocess_p0(image: np.ndarray, output_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Neutral P0 standardisation: grayscale, resize, dtype conversion, normalisation."""
    source = np.asarray(image)
    array = source
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY if array.shape[2] == 4 else cv2.COLOR_RGB2GRAY)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale or 3-D colour image, got {array.shape}.")
    array = array.astype(np.float32)
    if np.issubdtype(source.dtype, np.integer):
        array /= float(np.iinfo(source.dtype).max)
    elif array.max(initial=0.0) > 1.0:
        array /= 255.0
    if output_size and array.shape != tuple(output_size):
        array = transform.resize(array, output_size, anti_aliasing=True, preserve_range=True)
    return normalise_image(array)


def load_fingerprint(path: str | Path, output_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Fingerprint image not found: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {source.suffix}")
    with Image.open(source) as pil_image:
        return preprocess_p0(np.asarray(pil_image.convert("L")), output_size)


def controlled_degradation(clean_p0: np.ndarray, seed: int) -> np.ndarray:
    """Create a fixed degraded baseline without re-normalising after noise."""
    rng = np.random.default_rng(int(seed))
    return np.clip(_metric_image(clean_p0) + rng.normal(0.0, GAUSSIAN_SIGMA, clean_p0.shape).astype(np.float32), 0.0, 1.0)


def calculate_mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(mean_squared_error(_metric_image(reference), _metric_image(candidate)))


def calculate_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(_metric_image(reference), _metric_image(candidate), data_range=1.0))


def calculate_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(structural_similarity(_metric_image(reference), _metric_image(candidate), data_range=1.0))


def full_reference_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    return {
        "psnr": calculate_psnr(reference, candidate),
        "ssim": calculate_ssim(reference, candidate),
        "mse": calculate_mse(reference, candidate),
    }


def apply_nlm(image: np.ndarray, h: float) -> np.ndarray:
    base = _metric_image(image)
    as_uint8 = np.round(base * 255.0).astype(np.uint8)
    filtered = cv2.fastNlMeansDenoising(
        as_uint8,
        None,
        h=float(h) * 255.0,
        templateWindowSize=NLM_TEMPLATE_WINDOW_SIZE,
        searchWindowSize=NLM_SEARCH_WINDOW_SIZE,
    )
    return _metric_image(filtered.astype(np.float32) / 255.0)


def apply_tv_restoration(image: np.ndarray, weight: float) -> np.ndarray:
    restored = restoration.denoise_tv_chambolle(_metric_image(image), weight=float(weight), channel_axis=None)
    return _metric_image(restored)


def _fingerprint_mask(image: np.ndarray, block_size: int = 17) -> np.ndarray:
    base = _metric_image(image)
    mean = ndi.uniform_filter(base, size=block_size, mode="reflect")
    std = np.sqrt(np.maximum(ndi.uniform_filter(base * base, size=block_size, mode="reflect") - mean * mean, 0))
    threshold = max(0.012, float(np.percentile(std, 70) * 0.55))
    mask = (std > threshold) | ((base < np.percentile(base, 92)) & (std > threshold * 0.55))
    mask = morphology.opening(morphology.closing(mask, morphology.disk(5)), morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_holes(mask.astype(bool), area_threshold=600)
    return morphology.remove_small_objects(mask, min_size=600)


def estimate_orientation_field(
    image: np.ndarray,
    smoothing_sigma: float = 3.0,
    orientation_smoothing_sigma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    base = _metric_image(image)
    gx, gy = filters.sobel_h(base), filters.sobel_v(base)
    gxx = ndi.gaussian_filter(gx * gx, smoothing_sigma)
    gyy = ndi.gaussian_filter(gy * gy, smoothing_sigma)
    gxy = ndi.gaussian_filter(gx * gy, smoothing_sigma)
    sin2 = ndi.gaussian_filter(2 * gxy, orientation_smoothing_sigma)
    cos2 = ndi.gaussian_filter(gxx - gyy, orientation_smoothing_sigma)
    orientation = 0.5 * np.arctan2(sin2, cos2)
    coherence = np.sqrt((gxx - gyy) ** 2 + 4 * gxy**2) / (gxx + gyy + 1e-8)
    return orientation.astype(np.float32), coherence.astype(np.float32)


def estimate_local_frequency_map(
    image: np.ndarray,
    mask: np.ndarray,
    block_size: int = 32,
    frequencies: tuple[float, ...] = (0.075, 0.095, 0.115, 0.140),
) -> tuple[np.ndarray, np.ndarray]:
    base = _metric_image(image)
    mask = np.asarray(mask, bool)
    freq_map = np.full_like(base, float(np.median(frequencies)), dtype=np.float32)
    confidence = np.zeros_like(base, dtype=np.float32)
    window = np.outer(np.hanning(block_size), np.hanning(block_size))
    yy, xx = np.mgrid[-block_size // 2 : block_size // 2, -block_size // 2 : block_size // 2]
    radius = np.sqrt(xx**2 + yy**2)
    center_exclusion = radius <= 2
    for row in range(0, base.shape[0] - block_size + 1, block_size // 2):
        for col in range(0, base.shape[1] - block_size + 1, block_size // 2):
            block_mask = mask[row : row + block_size, col : col + block_size]
            if float(np.mean(block_mask)) < 0.25:
                continue
            block = base[row : row + block_size, col : col + block_size]
            if float(np.std(block[block_mask])) < 0.012:
                continue
            spectrum = np.abs(np.fft.fftshift(np.fft.fft2((block - np.mean(block[block_mask])) * window)))
            spectrum[center_exclusion] = 0
            peak = np.unravel_index(int(np.argmax(spectrum)), spectrum.shape)
            observed = float(radius[peak] / block_size)
            nearest = min(frequencies, key=lambda frequency: abs(frequency - observed))
            conf = float(np.max(spectrum) / (np.mean(spectrum) + 1e-8))
            region = np.s_[row : row + block_size, col : col + block_size]
            freq_map[region] = nearest
            confidence[region] = max(float(np.mean(confidence[region])), min(conf / 25, 1))
    return freq_map, confidence


@lru_cache(maxsize=128)
def _gabor_kernel(frequency: float, theta: float) -> np.ndarray:
    wavelength = 1.0 / float(frequency)
    ksize = max(9, int(round(wavelength * 2.0)) | 1)
    sigma = 0.55 * wavelength
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, float(theta), wavelength, 0.5, 0, ktype=cv2.CV_32F)
    kernel -= float(kernel.mean())
    denom = float(np.sum(np.abs(kernel)))
    return kernel / denom if denom > 1e-8 else kernel


def apply_modified_gabor(
    image: np.ndarray,
    frequencies: tuple[float, ...] = (0.075, 0.095, 0.115, 0.140),
    orientation_bins: int = 8,
    blend: float = 0.18,
) -> np.ndarray:
    """Modified Gabor: orientation-adaptive local ridge filtering."""
    base = _metric_image(image)
    mask = _fingerprint_mask(base)
    orientation, coherence = estimate_orientation_field(base)
    freq_map, freq_conf = estimate_local_frequency_map(base, mask, frequencies=frequencies)
    bins = np.linspace(0, np.pi, int(orientation_bins), endpoint=False)
    oi = np.argmin(np.abs(np.angle(np.exp(1j * (orientation[..., None] - bins[None, None, :])))), axis=2)
    fi = np.argmin(np.abs(freq_map[..., None] - np.asarray(frequencies)[None, None, :]), axis=2)
    selected = np.zeros_like(base, dtype=np.float32)
    inverted = (1.0 - base).astype(np.float32)
    for f_index, frequency in enumerate(frequencies):
        for o_index, theta in enumerate(bins):
            response = np.abs(cv2.filter2D(inverted, cv2.CV_32F, _gabor_kernel(float(frequency), float(theta)), borderType=cv2.BORDER_REFLECT))
            selector = (oi == o_index) & (fi == f_index) & mask
            selected[selector] = response[selector]
    ridge = np.where(mask, 1 - normalise_image(selected), base)
    confidence_weight = np.clip(0.30 + 0.70 * coherence, 0, 1) * np.clip(0.40 + 0.60 * freq_conf, 0, 1)
    weight = float(blend) * confidence_weight * mask
    return _metric_image((1 - weight) * base + weight * ridge)


def _oriented_kernel(theta: float) -> np.ndarray:
    axis = np.arange(9, dtype=np.float32) - 4
    xx, yy = np.meshgrid(axis, axis)
    c, s = math.cos(theta), math.sin(theta)
    xr, yr = c * xx + s * yy, -s * xx + c * yy
    kernel = np.exp(-(xr**2 / (2 * 1.35**2) + yr**2 / (2 * 0.50**2)))
    return (kernel / kernel.sum()).astype(np.float32)


def apply_coherence_guided_directional_diffusion(image: np.ndarray, iterations: int, step: float) -> np.ndarray:
    current = _metric_image(image)
    mask = _fingerprint_mask(current)
    bins = np.linspace(0, np.pi, 8, endpoint=False)
    for _ in range(int(iterations)):
        orientation, coherence = estimate_orientation_field(current)
        indexes = np.argmin(np.abs(np.angle(np.exp(1j * (orientation[..., None] - bins[None, None, :])))), axis=2)
        smoothed = np.zeros_like(current)
        for index, theta in enumerate(bins):
            conv = ndi.convolve(current, _oriented_kernel(float(theta)), mode="reflect")
            smoothed[indexes == index] = conv[indexes == index]
        weight = np.clip(coherence, 0, 1) * mask
        current = _metric_image((1 - float(step) * weight) * current + float(step) * weight * smoothed)
    return current


def segment_fingerprint(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = _metric_image(image)
    mask = _fingerprint_mask(base)
    return (base < filters.threshold_sauvola(base, window_size=25, k=0.16)) & mask, mask


def extract_minutiae_detections(skeleton: np.ndarray, mask: np.ndarray | None = None, border_margin: int = 8) -> dict[str, np.ndarray]:
    skel = np.asarray(skeleton, bool)
    padded = np.pad(skel, 1)
    neighbours = [
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
        padded[:-2, :-2],
    ]
    crossing = np.sum(np.abs(np.diff(np.stack(neighbours + [neighbours[0]]).astype(np.int8), axis=0)), axis=0) / 2
    valid = skel.copy()
    valid[:border_margin] = False
    valid[-border_margin:] = False
    valid[:, :border_margin] = False
    valid[:, -border_margin:] = False
    if mask is not None and np.any(mask):
        valid &= morphology.binary_erosion(np.asarray(mask, bool), morphology.disk(border_margin))
    return {"ridge_endings": np.argwhere(valid & (crossing == 1)), "bifurcations": np.argwhere(valid & (crossing == 3))}


def structural_metrics(image: np.ndarray) -> dict[str, float]:
    base = _metric_image(image)
    mask = _fingerprint_mask(base)
    orientation, coherence = estimate_orientation_field(base)
    del orientation
    local_mean = ndi.uniform_filter(base, 17, mode="reflect")
    local_std = np.sqrt(np.maximum(ndi.uniform_filter(base * base, 17, mode="reflect") - local_mean * local_mean, 0))
    binary, _ = segment_fingerprint(base)
    labels = measure.label(binary, connectivity=2)
    props = measure.regionprops(labels)
    skeleton = morphology.skeletonize(binary)
    points = extract_minutiae_detections(skeleton, mask)
    ridge_pixels = max(int(binary.sum()), 1)
    endpoints = len(points["ridge_endings"])
    return {
        "coherence": float(np.mean(coherence[mask])) if np.any(mask) else float(np.mean(coherence)),
        "contrast": float(local_std[mask].mean()) if mask.any() else float(local_std.mean()),
        "fragmentation": len(props) / (ridge_pixels / 1000.0),
        "continuity": float(skeleton.sum()) / max(endpoints, 1),
        "ridge_coverage": float(binary.sum()) / max(float(mask.sum()), 1.0),
    }


def parse_subject_id(path_or_name: str | Path) -> str:
    name = Path(path_or_name).name
    match = re.match(r"^(\d+)__", name)
    if not match:
        raise ValueError(f"Cannot parse SOCOFing subject ID from filename: {name}")
    return match.group(1)


def deterministic_subject_score(subject_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{subject_id}".encode("utf-8")).hexdigest()


def list_images(folder: str | Path) -> list[Path]:
    source = Path(folder)
    if not source.is_dir():
        return []
    return sorted(
        (path for path in source.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def discover_dataset(root: str | Path = PROJECT_ROOT) -> dict[str, list[Path]]:
    project_root = Path(root).resolve()
    data = project_root / "data" / "SOCOFing"
    if not data.is_dir():
        raise FileNotFoundError("SOCOFing dataset not found under data/SOCOFing.")
    return {
        "Real": list_images(data / "Real"),
        "Altered-Easy": list_images(data / "Altered" / "Altered-Easy"),
        "Altered-Medium": list_images(data / "Altered" / "Altered-Medium"),
        "Altered-Hard": list_images(data / "Altered" / "Altered-Hard"),
    }


def dataset_audit(dataset_paths: dict[str, list[Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = pd.DataFrame(
        [
            {
                "folder": name,
                "image_count": len(paths),
                "representative_file": paths[0].name if paths else "",
            }
            for name, paths in dataset_paths.items()
        ]
    )
    subject_rows = []
    grouped: dict[str, list[Path]] = {}
    for path in dataset_paths["Real"]:
        grouped.setdefault(parse_subject_id(path), []).append(path)
    for subject_id, paths in sorted(grouped.items(), key=lambda item: int(item[0])):
        subject_rows.append({"subject_id": subject_id, "real_image_count": len(paths)})
    return inventory, pd.DataFrame(subject_rows)


def make_subject_disjoint_split(real_paths: list[Path]) -> tuple[dict[str, list[Path]], pd.DataFrame, dict[str, Any]]:
    grouped: dict[str, list[Path]] = {}
    for path in real_paths:
        grouped.setdefault(parse_subject_id(path), []).append(Path(path).resolve())
    counts = {subject_id: len(paths) for subject_id, paths in grouped.items()}
    if set(counts.values()) != {EXPECTED_IMAGES_PER_SUBJECT}:
        raise AssertionError(f"Expected exactly {EXPECTED_IMAGES_PER_SUBJECT} Real images per subject; found {counts}.")
    ranked_subjects = sorted(grouped, key=lambda subject_id: deterministic_subject_score(subject_id))
    required_subjects = DEVELOPMENT_SUBJECTS + VALIDATION_SUBJECTS
    if len(ranked_subjects) < required_subjects:
        raise FileNotFoundError(f"Need at least {required_subjects} subjects; found {len(ranked_subjects)}.")

    dev_subjects = ranked_subjects[:DEVELOPMENT_SUBJECTS]
    val_subjects = ranked_subjects[DEVELOPMENT_SUBJECTS : DEVELOPMENT_SUBJECTS + VALIDATION_SUBJECTS]
    split_subjects = {"development": dev_subjects, "validation": val_subjects}
    split_paths = {
        split: [path for subject_id in subjects for path in sorted(grouped[subject_id], key=lambda item: item.name.lower())]
        for split, subjects in split_subjects.items()
    }
    dev_subject_set = set(dev_subjects)
    val_subject_set = set(val_subjects)
    dev_path_set = set(split_paths["development"])
    val_path_set = set(split_paths["validation"])
    if len(split_paths["development"]) != DEVELOPMENT_COUNT:
        raise AssertionError("Development image count is not 500.")
    if len(split_paths["validation"]) != VALIDATION_COUNT:
        raise AssertionError("Validation image count is not 1000.")
    if dev_subject_set & val_subject_set:
        raise AssertionError("Subject leakage between development and validation.")
    if dev_path_set & val_path_set:
        raise AssertionError("Image leakage between development and validation.")

    rows = []
    rank_by_subject = {subject_id: rank for rank, subject_id in enumerate(ranked_subjects, start=1)}
    for split, paths in split_paths.items():
        for path in paths:
            subject_id = parse_subject_id(path)
            rows.append(
                {
                    "filename": path.name,
                    "subject_id": subject_id,
                    "split": split,
                    "deterministic_subject_score": deterministic_subject_score(subject_id),
                    "subject_rank": rank_by_subject[subject_id],
                    "path": str(path),
                }
            )
    audit = {
        "split_seed": SPLIT_SEED,
        "real_image_count": len(real_paths),
        "total_real_subjects": len(grouped),
        "images_per_subject_distribution": {str(k): list(counts.values()).count(k) for k in sorted(set(counts.values()))},
        "development_images": len(split_paths["development"]),
        "validation_images": len(split_paths["validation"]),
        "development_subjects": len(dev_subject_set),
        "validation_subjects": len(val_subject_set),
        "subject_overlap": len(dev_subject_set & val_subject_set),
        "image_overlap": len(dev_path_set & val_path_set),
    }
    return split_paths, pd.DataFrame(rows), audit


def prepare_pairs(paths: list[Path], seed_base: int) -> list[dict[str, Any]]:
    pairs = []
    for index, path in enumerate(paths):
        clean = load_fingerprint(path)
        degraded = controlled_degradation(clean, seed_base + index)
        baseline = full_reference_metrics(clean, degraded)
        pairs.append({"path": path, "subject_id": parse_subject_id(path), "clean": clean, "degraded": degraded, "seed": seed_base + index, "baseline": baseline})
    return pairs


def baseline_metrics_table(pairs: list[dict[str, Any]], split: str) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        rows.append(
            {
                "split": split,
                "filename": pair["path"].name,
                "subject_id": pair["subject_id"],
                "noise_seed": pair["seed"],
                "psnr": pair["baseline"]["psnr"],
                "ssim": pair["baseline"]["ssim"],
                "mse": pair["baseline"]["mse"],
            }
        )
    return pd.DataFrame(rows)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    return value


def _candidate_label(member: str, params: dict[str, Any]) -> str:
    parts = []
    for key, value in params.items():
        if isinstance(value, tuple):
            value = "/".join(f"{item:g}" for item in value)
        elif isinstance(value, float):
            value = f"{value:g}"
        parts.append(f"{key}={value}")
    return f"{member} ({', '.join(parts)})"


def member_functions() -> dict[str, Callable[[np.ndarray, dict[str, Any]], np.ndarray]]:
    return {
        "M1 NLM": lambda image, params: apply_nlm(image, **params),
        "M2 Modified Gabor": lambda image, params: apply_modified_gabor(image, **params),
        "M3 TV": lambda image, params: apply_tv_restoration(image, **params),
        "M4 Directional Diffusion": lambda image, params: apply_coherence_guided_directional_diffusion(image, **params),
    }


def parameter_grids() -> dict[str, list[dict[str, Any]]]:
    return {
        "M1 NLM": [{"h": h} for h in NLM_H_CANDIDATES],
        "M2 Modified Gabor": GABOR_CANDIDATES,
        "M3 TV": TV_CANDIDATES,
        "M4 Directional Diffusion": DIFFUSION_CANDIDATES,
    }


def summarise_method_metrics(per_image: pd.DataFrame, group_column: str) -> pd.DataFrame:
    summary = (
        per_image.groupby(group_column)
        .agg(
            images=("filename", "count"),
            mean_psnr=("psnr", "mean"),
            std_psnr=("psnr", "std"),
            median_psnr=("psnr", "median"),
            min_psnr=("psnr", "min"),
            max_psnr=("psnr", "max"),
            mean_ssim=("ssim", "mean"),
            std_ssim=("ssim", "std"),
            median_ssim=("ssim", "median"),
            min_ssim=("ssim", "min"),
            max_ssim=("ssim", "max"),
            mean_mse=("mse", "mean"),
            std_mse=("mse", "std"),
            median_mse=("mse", "median"),
            min_mse=("mse", "min"),
            max_mse=("mse", "max"),
            mean_delta_psnr=("delta_psnr", "mean"),
            mean_delta_ssim=("delta_ssim", "mean"),
            mean_delta_mse=("delta_mse", "mean"),
            runtime_seconds_total=("runtime_seconds", "sum"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(["mean_psnr", "mean_ssim", "mean_mse"], ascending=[False, False, True]).reset_index(drop=True)


def run_parameter_search(
    pairs: list[dict[str, Any]],
    member_name: str,
    grid: list[dict[str, Any]],
    runner: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], float]:
    started_search = perf_counter()
    rows = []
    for grid_index, params in enumerate(grid, start=1):
        label = _candidate_label(member_name, params)
        if progress:
            print(f"  {member_name} candidate {grid_index}/{len(grid)}: {params}", flush=True)
        def evaluate_pair(pair: dict[str, Any]) -> dict[str, Any]:
            started = perf_counter()
            enhanced = runner(pair["degraded"], params)
            runtime = perf_counter() - started
            metrics = full_reference_metrics(pair["clean"], enhanced)
            return {
                "filename": pair["path"].name,
                "subject_id": pair["subject_id"],
                "candidate": label,
                "member": member_name,
                "parameters_json": json.dumps(_json_ready(params), sort_keys=True),
                "baseline_psnr": pair["baseline"]["psnr"],
                "baseline_ssim": pair["baseline"]["ssim"],
                "baseline_mse": pair["baseline"]["mse"],
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "mse": metrics["mse"],
                "delta_psnr": metrics["psnr"] - pair["baseline"]["psnr"],
                "delta_ssim": metrics["ssim"] - pair["baseline"]["ssim"],
                "delta_mse": metrics["mse"] - pair["baseline"]["mse"],
                "runtime_seconds": runtime,
            }

        if member_name in SERIAL_MEMBERS:
            for pair_index, pair in enumerate(pairs, start=1):
                if progress and (pair_index == 1 or pair_index % 100 == 0 or pair_index == len(pairs)):
                    print(f"    {member_name}: {pair_index}/{len(pairs)} images", flush=True)
                rows.append(evaluate_pair(pair))
        else:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                rows.extend(executor.map(evaluate_pair, pairs))
    per_image = pd.DataFrame(rows)
    summary = summarise_method_metrics(per_image, "candidate")
    selected_candidate = summary.iloc[0]["candidate"]
    selected_params = json.loads(per_image.loc[per_image.candidate.eq(selected_candidate), "parameters_json"].iloc[0])
    if "frequencies" in selected_params:
        selected_params["frequencies"] = tuple(selected_params["frequencies"])
    return per_image, summary, selected_params, perf_counter() - started_search


def run_all_parameter_searches(pairs: list[dict[str, Any]], progress: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict[str, Any]], pd.DataFrame]:
    funcs = member_functions()
    grids = parameter_grids()
    per_image_tables = {}
    summary_tables = {}
    selected = {}
    runtime_rows = []
    for member_name in ("M1 NLM", "M2 Modified Gabor", "M3 TV", "M4 Directional Diffusion"):
        if progress:
            print(f"Parameter search: {member_name}", flush=True)
        per_image, summary, params, elapsed = run_parameter_search(pairs, member_name, grids[member_name], funcs[member_name], progress)
        per_image_tables[member_name] = per_image
        summary_tables[member_name] = summary
        selected[member_name] = params
        runtime_rows.append({"stage": f"{member_name} parameter search", "seconds": elapsed})
    return per_image_tables, summary_tables, selected, pd.DataFrame(runtime_rows)


def evaluate_members(
    pairs: list[dict[str, Any]],
    selected_parameters: dict[str, dict[str, Any]],
    split: str,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    started_eval = perf_counter()
    funcs = member_functions()
    rows = []
    for member_name in ("M1 NLM", "M2 Modified Gabor", "M3 TV", "M4 Directional Diffusion"):
        if progress:
            print(f"  {split} evaluation: {member_name}", flush=True)
        params = selected_parameters[member_name]
        def evaluate_pair(pair: dict[str, Any]) -> dict[str, Any]:
            started = perf_counter()
            enhanced = funcs[member_name](pair["degraded"], params)
            runtime = perf_counter() - started
            metrics = full_reference_metrics(pair["clean"], enhanced)
            return {
                "split": split,
                "filename": pair["path"].name,
                "subject_id": pair["subject_id"],
                "method": member_name,
                "parameters_json": json.dumps(_json_ready(params), sort_keys=True),
                "baseline_psnr": pair["baseline"]["psnr"],
                "baseline_ssim": pair["baseline"]["ssim"],
                "baseline_mse": pair["baseline"]["mse"],
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "mse": metrics["mse"],
                "delta_psnr": metrics["psnr"] - pair["baseline"]["psnr"],
                "delta_ssim": metrics["ssim"] - pair["baseline"]["ssim"],
                "delta_mse": metrics["mse"] - pair["baseline"]["mse"],
                "runtime_seconds": runtime,
            }

        if member_name in SERIAL_MEMBERS:
            for pair_index, pair in enumerate(pairs, start=1):
                if progress and (pair_index == 1 or pair_index % 100 == 0 or pair_index == len(pairs)):
                    print(f"    {member_name}: {pair_index}/{len(pairs)} images", flush=True)
                rows.append(evaluate_pair(pair))
        else:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                rows.extend(executor.map(evaluate_pair, pairs))
    per_image = pd.DataFrame(rows)
    member_summary = summarise_method_metrics(per_image, "method")
    baseline = baseline_metrics_table(pairs, split)
    baseline_summary = summarise_baseline_for_comparison(baseline)
    comparison = pd.concat([baseline_summary, member_summary.rename(columns={"method": "method"})], ignore_index=True, sort=False)
    return per_image, comparison, perf_counter() - started_eval


def summarise_baseline_for_comparison(baseline: pd.DataFrame) -> pd.DataFrame:
    row = {
        "method": "Degraded Baseline",
        "images": len(baseline),
        "mean_psnr": baseline.psnr.mean(),
        "std_psnr": baseline.psnr.std(),
        "median_psnr": baseline.psnr.median(),
        "min_psnr": baseline.psnr.min(),
        "max_psnr": baseline.psnr.max(),
        "mean_ssim": baseline.ssim.mean(),
        "std_ssim": baseline.ssim.std(),
        "median_ssim": baseline.ssim.median(),
        "min_ssim": baseline.ssim.min(),
        "max_ssim": baseline.ssim.max(),
        "mean_mse": baseline.mse.mean(),
        "std_mse": baseline.mse.std(),
        "median_mse": baseline.mse.median(),
        "min_mse": baseline.mse.min(),
        "max_mse": baseline.mse.max(),
        "mean_delta_psnr": np.nan,
        "mean_delta_ssim": np.nan,
        "mean_delta_mse": np.nan,
        "runtime_seconds_total": 0.0,
        "runtime_seconds_mean": 0.0,
    }
    return pd.DataFrame([row])


def select_development_winner(development_comparison: pd.DataFrame) -> pd.Series:
    candidates = development_comparison[development_comparison.method.ne("Degraded Baseline")].copy()
    candidates = candidates.sort_values(["mean_psnr", "mean_ssim", "mean_mse"], ascending=[False, False, True])
    return candidates.iloc[0]


def freeze_configuration(winner: pd.Series, selected_parameters: dict[str, dict[str, Any]]) -> FrozenConfiguration:
    member = str(winner["method"])
    return FrozenConfiguration(
        selected_method=member,
        selected_member=member,
        selected_parameters=dict(selected_parameters[member]),
        all_member_parameters={key: dict(value) for key, value in selected_parameters.items()},
        ranking_rule="Highest development mean PSNR, then highest development mean SSIM, then lowest development mean MSE.",
        selected_using="Development Set only: 500 Real images from 50 subjects.",
        frozen_before_validation=True,
    )


def run_altered_structural_evaluation(
    dataset_paths: dict[str, list[Path]],
    frozen_config: FrozenConfiguration,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    started_eval = perf_counter()
    funcs = member_functions()
    method = frozen_config.selected_method
    params = frozen_config.selected_parameters
    rows = []
    for offset, severity in enumerate(("Altered-Easy", "Altered-Medium", "Altered-Hard")):
        candidates = sorted(dataset_paths[severity], key=lambda path: path.name.lower())
        if len(candidates) < ALTERED_PER_SEVERITY_COUNT:
            raise FileNotFoundError(f"Need at least {ALTERED_PER_SEVERITY_COUNT} {severity} images.")
        rng = np.random.default_rng(ALTERED_SEED + offset)
        selected_paths = [candidates[int(index)] for index in rng.permutation(len(candidates))[:ALTERED_PER_SEVERITY_COUNT]]
        if progress:
            print(f"  {severity}: processing {len(selected_paths)} images", flush=True)
        def evaluate_path(path: Path) -> dict[str, Any]:
            before = load_fingerprint(path)
            after = funcs[method](before, params)
            before_metrics = structural_metrics(before)
            after_metrics = structural_metrics(after)
            return {
                "severity": severity,
                "filename": path.name,
                "paired_clean_reference_available": False,
                **{f"{key}_before": value for key, value in before_metrics.items()},
                **{f"{key}_after": value for key, value in after_metrics.items()},
                **{f"delta_{key}": after_metrics[key] - before_metrics[key] for key in before_metrics},
            }

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            rows.extend(executor.map(evaluate_path, selected_paths))
    per_image = pd.DataFrame(rows)
    summary_rows = []
    for label, group in list(per_image.groupby("severity")) + [("Overall", per_image)]:
        summary_rows.append(
            {
                "severity": label,
                "count": len(group),
                "paired_clean_reference_available": False,
                **{f"mean_{column}": group[column].mean() for column in group.columns if column.endswith("_before") or column.endswith("_after") or column.startswith("delta_")},
            }
        )
    return per_image, pd.DataFrame(summary_rows), perf_counter() - started_eval


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True), encoding="utf-8")
    return path


def reset_active_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.name == "archive":
            continue
        archive = output_dir / "archive" / child.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            if archive.is_dir():
                shutil.rmtree(archive)
            else:
                archive.unlink()
        shutil.move(str(child), str(archive))


def environment_report() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": __import__("scipy").__version__,
        "scikit_image": __import__("skimage").__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "pillow": Image.__version__,
        "platform": platform.platform(),
        "cv2_import_works": True,
    }


def figure_sample_index(count: int) -> int:
    digest = hashlib.sha256(f"{SPLIT_SEED}:figure".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % count


def plot_parameter_summary(summary: pd.DataFrame, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(summary))
    ax.bar(x, summary["mean_psnr"], yerr=summary["std_psnr"])
    ax.set_xticks(x)
    ax.set_xticklabels(summary.iloc[:, 0], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_comparison(summary: pd.DataFrame, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(summary))
    ax.bar(x, summary["mean_psnr"])
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=25, ha="right")
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_development_example(
    pairs: list[dict[str, Any]],
    selected_parameters: dict[str, dict[str, Any]],
    path: Path,
) -> Path:
    index = figure_sample_index(len(pairs))
    pair = pairs[index]
    funcs = member_functions()
    images = [("Clean P0 reference", pair["clean"]), ("Degraded baseline", pair["degraded"])]
    for member_name in ("M1 NLM", "M2 Modified Gabor", "M3 TV", "M4 Directional Diffusion"):
        images.append((member_name, funcs[member_name](pair["degraded"], selected_parameters[member_name])))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (title, image) in zip(axes.ravel(), images):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"Deterministic demonstration sample: {pair['path'].name}", y=0.98)
    fig.subplots_adjust(top=0.88, hspace=0.28, wspace=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_improvement(summary: pd.DataFrame, title: str, path: Path) -> Path:
    members = summary[summary.method.ne("Degraded Baseline")]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(members))
    ax.bar(x, members["mean_delta_psnr"])
    ax.set_xticks(x)
    ax.set_xticklabels(members["method"], rotation=25, ha="right")
    ax.set_ylabel("Mean Delta PSNR (dB)")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_complete_workflow(
    root: str | Path = PROJECT_ROOT,
    output_subdir: str = "script",
    reset_outputs: bool = False,
    make_plots: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    workflow_start = perf_counter()
    root_path = Path(root).resolve()
    output_dir = root_path / "outputs" / "final_mode_a" / output_subdir
    figures_dir = output_dir / "figures"
    if reset_outputs:
        reset_active_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    runtime_rows = []

    def checkpoint(label: str, started: float) -> None:
        runtime_rows.append({"stage": label, "seconds": perf_counter() - started})

    if progress:
        print("Fingerprint Enhancement System - individual-method Mode A workflow", flush=True)
        print(environment_report(), flush=True)

    started = perf_counter()
    dataset_paths = discover_dataset(root_path)
    inventory, subject_counts = dataset_audit(dataset_paths)
    split_paths, split_manifest, split_audit = make_subject_disjoint_split(dataset_paths["Real"])
    checkpoint("dataset audit and subject-disjoint split", started)
    if progress:
        print("Dataset audit and subject-disjoint split complete", flush=True)

    started = perf_counter()
    if progress:
        print("Preparing P0/degraded caches for 500 development and 1000 validation images", flush=True)
    development_pairs = prepare_pairs(split_paths["development"], NOISE_SEED_BASE)
    validation_pairs = prepare_pairs(split_paths["validation"], NOISE_SEED_BASE)
    development_baseline = baseline_metrics_table(development_pairs, "development")
    validation_baseline = baseline_metrics_table(validation_pairs, "validation")
    checkpoint("P0 loading and controlled degradation cache", started)

    if progress:
        print("[1/6] Development parameter searches on 500 images", flush=True)
    parameter_per_image, parameter_summaries, selected_parameters, parameter_runtime = run_all_parameter_searches(development_pairs, progress)
    runtime_rows.extend(parameter_runtime.to_dict("records"))

    if progress:
        print("[2/6] Development comparison with baseline control", flush=True)
    development_metrics, development_comparison, elapsed = evaluate_members(development_pairs, selected_parameters, "development", progress)
    runtime_rows.append({"stage": "development four-member comparison", "seconds": elapsed})
    winner = select_development_winner(development_comparison)
    frozen_config = freeze_configuration(winner, selected_parameters)

    config_json = {
        **asdict(frozen_config),
        "selected_parameters": _json_ready(frozen_config.selected_parameters),
        "all_member_parameters": _json_ready(frozen_config.all_member_parameters),
        "p0": {
            "purpose": "neutral image standardisation only",
            "steps": ["grayscale conversion", "resize to 256x256", "dtype conversion", "intensity normalisation to [0,1]"],
            "excluded": ["denoising", "CLAHE", "histogram equalisation", "Gabor enhancement", "TV denoising", "directional diffusion", "morphology", "thresholding", "sharpening"],
        },
        "controlled_degradation": {
            "purpose": "paired full-reference Real evaluation",
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "noise_seed_base": NOISE_SEED_BASE,
            "renormalise_after_noise": False,
            "same_degraded_input_for_all_members": True,
        },
        "development_images": DEVELOPMENT_COUNT,
        "validation_images": VALIDATION_COUNT,
        "subject_disjoint": True,
    }

    if progress:
        print("[3/6] Frozen 1000-image validation of all four members", flush=True)
    validation_metrics, validation_comparison, elapsed = evaluate_members(validation_pairs, selected_parameters, "validation", progress)
    runtime_rows.append({"stage": "validation four-member comparison", "seconds": elapsed})
    validation_summary = {
        "frozen_development_winner": frozen_config.selected_method,
        "winner_not_changed_after_validation": True,
        "validation_images": len(validation_pairs),
        "validation_subjects": len({pair["subject_id"] for pair in validation_pairs}),
        "comparison": validation_comparison.to_dict("records"),
    }

    if progress:
        print("[4/6] Supplementary Altered structural evaluation", flush=True)
    altered_metrics, altered_summary, elapsed = run_altered_structural_evaluation(dataset_paths, frozen_config, progress)
    runtime_rows.append({"stage": "supplementary altered structural evaluation", "seconds": elapsed})

    if progress:
        print("[5/6] Saving outputs and figures", flush=True)
    save_csv(inventory, output_dir / "dataset_inventory.csv")
    save_csv(subject_counts, output_dir / "real_subject_counts.csv")
    save_csv(split_manifest, output_dir / "dataset_split_manifest.csv")
    save_json(split_audit, output_dir / "dataset_split_audit.json")
    save_csv(development_baseline, output_dir / "development_baseline_metrics.csv")
    for member_name, summary in parameter_summaries.items():
        number = member_name.split()[0].lower().replace("m", "member")
        save_csv(summary, output_dir / f"{number}_parameter_search.csv")
        save_csv(parameter_per_image[member_name], output_dir / f"{number}_parameter_metrics.csv")
    save_csv(development_metrics, output_dir / "development_member_metrics.csv")
    save_csv(development_comparison, output_dir / "development_member_comparison.csv")
    save_json(config_json, output_dir / "final_selected_configuration.json")
    save_csv(validation_baseline, output_dir / "validation_baseline_metrics.csv")
    save_csv(validation_metrics, output_dir / "validation_member_metrics.csv")
    save_csv(validation_comparison, output_dir / "validation_member_comparison.csv")
    save_json(validation_summary, output_dir / "validation_summary.json")
    save_csv(altered_metrics, output_dir / "altered_structural_metrics.csv")
    save_csv(altered_summary, output_dir / "altered_structural_summary.csv")
    save_json(environment_report(), output_dir / "environment_report.json")

    figure_paths: list[Path] = []
    if make_plots:
        figure_paths.append(plot_development_example(development_pairs, selected_parameters, figures_dir / "example_clean_degraded_members.png"))
        figure_paths.append(plot_parameter_summary(parameter_summaries["M1 NLM"], "Member 1 NLM parameter search", figures_dir / "member1_parameter_search.png"))
        figure_paths.append(plot_parameter_summary(parameter_summaries["M2 Modified Gabor"], "Member 2 Modified Gabor parameter search", figures_dir / "member2_parameter_search.png"))
        figure_paths.append(plot_parameter_summary(parameter_summaries["M3 TV"], "Member 3 TV parameter search", figures_dir / "member3_parameter_search.png"))
        figure_paths.append(plot_parameter_summary(parameter_summaries["M4 Directional Diffusion"], "Member 4 Directional Diffusion parameter search", figures_dir / "member4_parameter_search.png"))
        figure_paths.append(plot_comparison(development_comparison, "500-image development comparison", figures_dir / "development_comparison.png"))
        figure_paths.append(plot_comparison(validation_comparison, "1000-image validation comparison", figures_dir / "validation_comparison.png"))
        figure_paths.append(plot_improvement(validation_comparison, "Validation improvement over degraded baseline", figures_dir / "baseline_vs_enhanced_improvement.png"))

    runtime_rows.append({"stage": "complete workflow", "seconds": perf_counter() - workflow_start})
    runtime = pd.DataFrame(runtime_rows)
    save_csv(runtime, output_dir / "runtime_report.csv")

    if progress:
        print("[6/6] Workflow complete", flush=True)
        print(f"  Development images: {len(development_pairs)}", flush=True)
        print(f"  Validation images: {len(validation_pairs)}", flush=True)
        print(f"  Selected: {frozen_config.selected_method}", flush=True)
        print(f"  Output directory: {output_dir}", flush=True)

    return {
        "dataset_paths": dataset_paths,
        "inventory": inventory,
        "subject_counts": subject_counts,
        "split_paths": split_paths,
        "split_manifest": split_manifest,
        "split_audit": split_audit,
        "development_pairs": development_pairs,
        "validation_pairs": validation_pairs,
        "development_baseline": development_baseline,
        "validation_baseline": validation_baseline,
        "parameter_per_image": parameter_per_image,
        "parameter_summaries": parameter_summaries,
        "selected_parameters": selected_parameters,
        "development_metrics": development_metrics,
        "development_comparison": development_comparison,
        "frozen_config": frozen_config,
        "validation_metrics": validation_metrics,
        "validation_comparison": validation_comparison,
        "validation_summary": validation_summary,
        "altered_metrics": altered_metrics,
        "altered_summary": altered_summary,
        "runtime": runtime,
        "figure_paths": figure_paths,
        "output_dir": output_dir,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the BMDS2133 individual-method Mode A fingerprint enhancement workflow.")
    parser.add_argument("--output-subdir", default="script", help="Subdirectory under outputs/final_mode_a.")
    parser.add_argument("--reset-outputs", action="store_true", help="Archive existing files in the active output subdirectory before running.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console progress output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    return run_complete_workflow(
        PROJECT_ROOT,
        output_subdir=args.output_subdir,
        reset_outputs=args.reset_outputs,
        make_plots=not args.no_plots,
        progress=not args.quiet,
    )


if __name__ == "__main__":
    main()
