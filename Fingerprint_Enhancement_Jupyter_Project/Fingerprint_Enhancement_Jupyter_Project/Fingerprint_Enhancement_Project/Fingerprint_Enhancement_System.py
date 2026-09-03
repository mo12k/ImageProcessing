"""Final reusable core for the BMDS2133 Fingerprint Enhancement System.

The final grayscale recommendation is OpenCV Non-Local Means with h=0.06.
Structural analysis is deliberately separate from grayscale enhancement.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi, signal
from skimage import exposure, filters, morphology, restoration, transform
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

IMAGE_SIZE = (256, 256)
FINAL_NLM_H = 0.06
NLM_TEMPLATE_WINDOW_SIZE = 7
NLM_SEARCH_WINDOW_SIZE = 21
SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _metric_image(image: np.ndarray) -> np.ndarray:
    base = np.asarray(image, dtype=np.float32)
    return np.clip(np.nan_to_num(base, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def normalise_image(image: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    """Percentile-normalize an image to float32 [0, 1]."""
    base = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(base, [low_percentile, high_percentile])
    return _metric_image(base if high - low < 1e-8 else (base - low) / (high - low))


def preprocess_p0(image: np.ndarray, output_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Neutral P0: grayscale, consistent resizing, and intensity normalization."""
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
    """Load a fingerprint using the exact validated P0 file path."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Fingerprint image not found: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {source.suffix}")
    with Image.open(source) as pil_image:
        image = np.asarray(pil_image.convert("L"), dtype=np.float32) / 255.0
    image = transform.resize(image, output_size, anti_aliasing=True, preserve_range=True)
    return normalise_image(image)


def apply_gaussian(image: np.ndarray, sigma: float = 0.65) -> np.ndarray:
    return _metric_image(ndi.gaussian_filter(_metric_image(image), sigma=float(sigma)))


def apply_median(image: np.ndarray, size: int = 3) -> np.ndarray:
    return _metric_image(ndi.median_filter(_metric_image(image), size=int(size)))


def apply_clahe(image: np.ndarray, clip_limit: float = 0.005, kernel_size: int = 32) -> np.ndarray:
    """Apply CLAHE as a conventional baseline."""
    return _metric_image(exposure.equalize_adapthist(_metric_image(image), clip_limit=clip_limit, kernel_size=kernel_size))


def apply_contrast_stretching(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    base = _metric_image(image); lo, hi = np.percentile(base, [low, high])
    return base.copy() if hi - lo < 1e-8 else _metric_image((base - lo) / (hi - lo))


def apply_nlm(image: np.ndarray, h: float = FINAL_NLM_H) -> np.ndarray:
    """Exact validated OpenCV NLM: round-to-uint8, windows 7/21, back to float32."""
    base = _metric_image(image)
    as_uint8 = np.round(base * 255.0).astype(np.uint8)
    filtered = cv2.fastNlMeansDenoising(as_uint8, None, h=float(h) * 255.0,
                                        templateWindowSize=7, searchWindowSize=21)
    return _metric_image(filtered.astype(np.float32) / 255.0)


def apply_tv_restoration(image: np.ndarray, weight: float = 0.02) -> np.ndarray:
    """Frozen Member 3 total-variation restoration."""
    return _metric_image(restoration.denoise_tv_chambolle(_metric_image(image), weight=weight, channel_axis=None))


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


def _orientation_field(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx, gy = filters.sobel_h(image), filters.sobel_v(image)
    gxx, gyy = ndi.gaussian_filter(gx * gx, 2), ndi.gaussian_filter(gy * gy, 2)
    gxy = ndi.gaussian_filter(gx * gy, 2)
    orientation = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
    coherence = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / (gxx + gyy + 1e-8)
    return orientation.astype(np.float32), coherence.astype(np.float32)


def _frequency_map(image: np.ndarray, mask: np.ndarray, frequencies: tuple[float, ...], block_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    freq_map = np.full_like(image, np.median(frequencies), dtype=np.float32)
    confidence = np.zeros_like(image, dtype=np.float32)
    window = np.outer(np.hanning(block_size), np.hanning(block_size))
    for row in range(0, image.shape[0] - block_size + 1, block_size):
        for col in range(0, image.shape[1] - block_size + 1, block_size):
            if np.mean(mask[row:row+block_size, col:col+block_size]) < .2:
                continue
            block = image[row:row+block_size, col:col+block_size]
            spectrum = np.abs(np.fft.fftshift(np.fft.fft2((block - block.mean()) * window)))
            yy, xx = np.mgrid[:block_size, :block_size]
            radius = np.sqrt((yy-block_size/2)**2 + (xx-block_size/2)**2) / block_size
            scores = [float(np.mean(spectrum[np.abs(radius-f) < .012])) for f in frequencies]
            best = int(np.argmax(scores)); region = np.s_[row:row+block_size, col:col+block_size]
            freq_map[region] = frequencies[best]
            confidence[region] = min(scores[best] / (float(np.mean(scores)) + 1e-8) / 25, 1)
    return freq_map, confidence


def apply_modified_gabor(image: np.ndarray, frequencies: tuple[float, ...] = (.075, .095, .115, .140),
                         orientation_bins: int = 8, blend: float = .18) -> np.ndarray:
    """Frozen Member 2 local-orientation/local-frequency adaptive Gabor."""
    base = _metric_image(image); mask = _fingerprint_mask(base)
    orientation, coherence = _orientation_field(base)
    freq_map, freq_confidence = _frequency_map(base, mask, frequencies)
    bins = np.linspace(0, np.pi, orientation_bins, endpoint=False)
    oi = np.argmin(np.abs(np.angle(np.exp(1j*(orientation[..., None]-bins)))), axis=2)
    fi = np.argmin(np.abs(freq_map[..., None]-np.asarray(frequencies)), axis=2)
    selected = np.zeros_like(base); inverted = 1-base
    for f_index, frequency in enumerate(frequencies):
        for o_index, theta in enumerate(bins):
            real, imag = filters.gabor(inverted, frequency=frequency, theta=theta, bandwidth=1.8)
            response = np.sqrt(real**2 + imag**2); selector = (oi == o_index) & (fi == f_index) & mask
            selected[selector] = response[selector]
    ridge = np.where(mask, 1-normalise_image(selected), base)
    confidence = np.clip(coherence * (.5 + .5*freq_confidence), 0, 1) * mask
    return _metric_image((1-blend*confidence)*base + blend*confidence*ridge)


def apply_ordinary_gabor(image: np.ndarray, frequency: float = .115, orientations: int = 8, blend: float = .30) -> np.ndarray:
    base = _metric_image(image); responses = []
    for theta in np.linspace(0, np.pi, orientations, endpoint=False):
        real, imag = filters.gabor(1-base, frequency=frequency, theta=theta)
        responses.append(np.sqrt(real**2 + imag**2))
    return _metric_image((1-blend)*base + blend*(1-normalise_image(np.max(responses, axis=0))))


def apply_wiener(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    return _metric_image(signal.wiener(_metric_image(image), mysize=window_size))


def _oriented_kernel(theta: float) -> np.ndarray:
    axis = np.arange(9, dtype=np.float32)-4; xx, yy = np.meshgrid(axis, axis)
    c, s = math.cos(theta), math.sin(theta); xr, yr = c*xx+s*yy, -s*xx+c*yy
    kernel = np.exp(-(xr**2/(2*1.35**2) + yr**2/(2*.50**2)))
    return (kernel/kernel.sum()).astype(np.float32)


def apply_coherence_guided_directional_diffusion(image: np.ndarray, iterations: int = 4, step: float = .16) -> np.ndarray:
    """Frozen Member 4 coherence-guided directional diffusion."""
    current = _metric_image(image); mask = _fingerprint_mask(current)
    bins = np.linspace(0, np.pi, 8, endpoint=False)
    for _ in range(iterations):
        orientation, coherence = _orientation_field(current)
        indexes = np.argmin(np.abs(np.angle(np.exp(1j*(orientation[..., None]-bins)))), axis=2)
        smoothed = np.zeros_like(current)
        for index, theta in enumerate(bins):
            conv = ndi.convolve(current, _oriented_kernel(theta), mode="reflect")
            smoothed[indexes == index] = conv[indexes == index]
        weight = np.clip(coherence, 0, 1)*mask
        current = _metric_image((1-step*weight)*current + step*weight*smoothed)
    return current


def segment_fingerprint(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = _metric_image(image); mask = _fingerprint_mask(base)
    return (base < filters.threshold_sauvola(base, window_size=25, k=.16)) & mask, mask


def morphological_cleanup(binary: np.ndarray, min_size: int = 24) -> np.ndarray:
    cleaned = morphology.closing(np.asarray(binary, bool), morphology.disk(1))
    return morphology.remove_small_objects(morphology.remove_small_holes(cleaned, area_threshold=28), min_size=min_size)


def thin_ridges(binary: np.ndarray) -> np.ndarray:
    return morphology.skeletonize(np.asarray(binary, bool))


def medial_axis_analysis(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return morphology.medial_axis(np.asarray(binary, bool), return_distance=True)


def extract_minutiae_detections(skeleton: np.ndarray, mask: np.ndarray | None = None, border_margin: int = 8) -> dict[str, np.ndarray]:
    skel = np.asarray(skeleton, bool); padded = np.pad(skel, 1)
    n = [padded[:-2,1:-1], padded[:-2,2:], padded[1:-1,2:], padded[2:,2:],
         padded[2:,1:-1], padded[2:,:-2], padded[1:-1,:-2], padded[:-2,:-2]]
    crossing = np.sum(np.abs(np.diff(np.stack(n+[n[0]]).astype(np.int8), axis=0)), axis=0)/2
    valid = skel.copy(); valid[:border_margin] = valid[-border_margin:] = False
    valid[:, :border_margin] = valid[:, -border_margin:] = False
    if mask is not None and np.any(mask):
        valid &= morphology.binary_erosion(np.asarray(mask, bool), morphology.disk(border_margin))
    return {"ridge_endings": np.argwhere(valid & (crossing == 1)),
            "bifurcations": np.argwhere(valid & (crossing == 3))}


def calculate_mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(mean_squared_error(_metric_image(reference), _metric_image(candidate)))


def calculate_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(_metric_image(reference), _metric_image(candidate), data_range=1.0))


def calculate_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(structural_similarity(_metric_image(reference), _metric_image(candidate), data_range=1.0))


def calculate_ridge_coherence(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    _, coherence = _orientation_field(_metric_image(image)); region = _fingerprint_mask(image) if mask is None else mask
    return float(np.mean(coherence[region])) if np.any(region) else float(np.mean(coherence))


def calculate_local_contrast(image: np.ndarray, mask: np.ndarray | None = None, size: int = 17) -> float:
    base = _metric_image(image); mean = ndi.uniform_filter(base, size=size, mode="reflect")
    std = np.sqrt(np.maximum(ndi.uniform_filter(base*base, size=size, mode="reflect")-mean*mean, 0))
    region = _fingerprint_mask(base) if mask is None else mask
    return float(np.mean(std[region])) if np.any(region) else float(np.mean(std))


def calculate_fragmentation(image: np.ndarray) -> float:
    ridges, _ = segment_fingerprint(image); _, count = ndi.label(ridges)
    return float(count*1000/max(np.count_nonzero(ridges), 1))


def enhance_fingerprint(image: np.ndarray, method: str = "nlm", *, assume_preprocessed: bool = False,
                        **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """GUI-ready enhancement dispatcher returning image and processing metadata."""
    base = _metric_image(image) if assume_preprocessed else preprocess_p0(image)
    methods = {"nlm": apply_nlm, "modified_gabor": apply_modified_gabor, "tv": apply_tv_restoration,
               "directional_diffusion": apply_coherence_guided_directional_diffusion,
               "gaussian": apply_gaussian, "median": apply_median, "clahe": apply_clahe,
               "contrast_stretching": apply_contrast_stretching, "ordinary_gabor": apply_ordinary_gabor,
               "wiener": apply_wiener}
    key = method.lower().strip()
    if key not in methods:
        raise ValueError(f"Unknown method '{method}'. Available: {', '.join(sorted(methods))}")
    enhanced = methods[key](base, **kwargs)
    metadata: dict[str, Any] = {"method": key, "preprocessing": "P0 already supplied" if assume_preprocessed else "P0",
                                "input_shape": tuple(base.shape), "output_shape": tuple(enhanced.shape),
                                "dtype": str(enhanced.dtype)}
    if key == "nlm":
        metadata.update({"recommended": True, "h_normalized": float(kwargs.get("h", FINAL_NLM_H)),
                         "backend": "cv2.fastNlMeansDenoising", "template_window_size": 7,
                         "search_window_size": 21})
    return enhanced, metadata


def apply_final_enhancement(image: np.ndarray, *, assume_preprocessed: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen final recommendation: OpenCV NLM, h=0.06."""
    return enhance_fingerprint(image, "nlm", assume_preprocessed=assume_preprocessed, h=FINAL_NLM_H)

