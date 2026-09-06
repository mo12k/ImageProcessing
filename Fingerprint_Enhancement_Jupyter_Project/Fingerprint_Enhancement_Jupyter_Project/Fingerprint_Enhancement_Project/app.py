"""Streamlit prototype for the BMDS2133 Fingerprint Enhancement System."""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(
    page_title="Fingerprint Enhancement System",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import Fingerprint_Enhancement_System as backend
except Exception as exc:  # pragma: no cover - rendered as a Streamlit setup error.
    backend = None
    BACKEND_IMPORT_ERROR = exc
else:
    BACKEND_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parent
FROZEN_CONFIG_PATH = PROJECT_ROOT / "outputs" / "final_mode_a" / "script" / "final_selected_configuration.json"
VALIDATION_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "final_mode_a" / "script" / "validation_member_comparison.csv"
COMPARE_ALL = "Compare All Techniques"
SUPPORTED_TYPES = ["bmp", "png", "jpg", "jpeg", "tif", "tiff"]

STRUCTURAL_LABELS = {
    "coherence": "Coherence",
    "contrast": "Contrast",
    "fragmentation": "Fragmentation",
    "continuity": "Continuity",
    "ridge_coverage": "Ridge coverage",
}


@dataclass
class UploadedFingerprint:
    name: str
    digest: str
    original_gray: np.ndarray
    p0: np.ndarray
    raw_dimensions: tuple[int, int]
    processed_dimensions: tuple[int, int]


def require_backend() -> Any:
    if backend is None:
        st.error("The image-processing backend could not be imported.")
        with st.expander("Technical detail"):
            st.code(str(BACKEND_IMPORT_ERROR))
        st.stop()
    return backend


def clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON lists back to the tuple-like structures used by the backend."""
    cleaned = dict(params)
    if "frequencies" in cleaned:
        cleaned["frequencies"] = tuple(float(value) for value in cleaned["frequencies"])
    return cleaned


@st.cache_resource(show_spinner=False)
def load_frozen_configuration() -> dict[str, Any]:
    fes = require_backend()
    if not FROZEN_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Frozen configuration file not found: {FROZEN_CONFIG_PATH}")

    config = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    backend_methods = fes.member_functions()
    all_params = {
        method: clean_params(params)
        for method, params in config.get("all_member_parameters", {}).items()
        if method in backend_methods
    }
    ordered_methods = [method for method in backend_methods if method in all_params]
    missing_methods = [method for method in backend_methods if method not in all_params]
    if missing_methods:
        raise ValueError(f"Frozen parameters are missing for: {', '.join(missing_methods)}.")
    if not ordered_methods:
        raise ValueError("No frozen member parameters match the backend method names.")

    selected_method = str(config.get("selected_method", ordered_methods[0]))
    selected_parameters = clean_params(config.get("selected_parameters", all_params.get(selected_method, {})))
    return {
        "raw": config,
        "methods": ordered_methods,
        "all_member_parameters": all_params,
        "selected_method": selected_method,
        "selected_parameters": selected_parameters,
    }


@st.cache_data(show_spinner=False)
def load_validation_comparison() -> pd.DataFrame:
    if not VALIDATION_COMPARISON_PATH.is_file():
        return pd.DataFrame()
    return pd.read_csv(VALIDATION_COMPARISON_PATH)


def file_signature(file: Any | None) -> str:
    if file is None:
        return ""
    data = file.getvalue()
    return hashlib.sha256(data).hexdigest()


def files_signature(files: list[Any] | None) -> tuple[tuple[str, str], ...]:
    return tuple((file.name, file_signature(file)) for file in (files or []))


def decode_uploaded_image(file: Any) -> tuple[np.ndarray, tuple[int, int], str]:
    if file is None:
        raise ValueError("No image file was supplied.")

    suffix = Path(file.name).suffix.lower()
    if suffix and suffix not in require_backend().SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Please upload BMP, PNG, JPG, JPEG or TIFF.")

    try:
        with Image.open(BytesIO(file.getvalue())) as image:
            image.load()
            grayscale = image.convert("L")
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file could not be read as an image.") from exc
    except OSError as exc:
        raise ValueError("The uploaded image appears to be corrupted or incomplete.") from exc

    array = np.asarray(grayscale)
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError("The image must contain a valid two-dimensional fingerprint image.")

    width, height = grayscale.size
    return array, (width, height), grayscale.mode


def preprocess_upload(file: Any) -> UploadedFingerprint:
    fes = require_backend()
    original, raw_dimensions, _ = decode_uploaded_image(file)
    p0 = fes.preprocess_p0(original)
    return UploadedFingerprint(
        name=file.name,
        digest=file_signature(file),
        original_gray=original,
        p0=p0,
        raw_dimensions=raw_dimensions,
        processed_dimensions=(int(p0.shape[1]), int(p0.shape[0])),
    )


def unit_float(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    return np.clip(np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    return np.round(unit_float(image) * 255.0).astype(np.uint8)


def encode_png(image: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image_to_uint8(image), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def method_slug(method: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")


def apply_frozen_method(p0_image: np.ndarray, method: str, configuration: dict[str, Any]) -> tuple[np.ndarray, float]:
    fes = require_backend()
    functions = fes.member_functions()
    if method not in functions:
        raise ValueError(f"Unknown enhancement technique: {method}")
    if method not in configuration["all_member_parameters"]:
        raise ValueError(f"No frozen parameters were found for {method}.")

    started = perf_counter()
    enhanced = functions[method](p0_image, configuration["all_member_parameters"][method])
    return unit_float(enhanced), perf_counter() - started


def calculate_structural_pair(p0_image: np.ndarray, enhanced: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    fes = require_backend()
    return fes.structural_metrics(p0_image), fes.structural_metrics(enhanced)


def structural_table(before: dict[str, float], after: dict[str, float]) -> pd.DataFrame:
    rows = []
    for key, label in STRUCTURAL_LABELS.items():
        before_value = float(before[key])
        after_value = float(after[key])
        rows.append(
            {
                "Metric": label,
                "P0": before_value,
                "Enhanced": after_value,
                "Change": after_value - before_value,
            }
        )
    return pd.DataFrame(rows)


def calculate_reference_metrics(reference_p0: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference_p0.shape != candidate.shape:
        raise ValueError(
            f"Reference dimensions after P0 are {reference_p0.shape}; enhanced dimensions are {candidate.shape}."
        )
    return require_backend().full_reference_metrics(reference_p0, candidate)


def format_reference_metrics(metrics: dict[str, float]) -> dict[str, str]:
    return {
        "PSNR": f"{metrics['psnr']:.3f} dB",
        "SSIM": f"{metrics['ssim']:.4f}",
        "MSE": f"{metrics['mse']:.6f}",
    }


def render_image_triplet(original: np.ndarray, p0: np.ndarray, enhanced: np.ndarray, method: str) -> None:
    original_col, p0_col, enhanced_col = st.columns(3)
    original_col.image(original, caption="Original Input", use_container_width=True, clamp=True)
    p0_col.image(image_to_uint8(p0), caption="Preprocessed / P0", use_container_width=True, clamp=True)
    enhanced_col.image(image_to_uint8(enhanced), caption=f"Enhanced Result - {method}", use_container_width=True, clamp=True)


def render_reference_cards(reference_metrics: dict[str, float] | None, runtime: float, fingerprint: UploadedFingerprint) -> None:
    cols = st.columns(5)
    cols[0].metric("Input Dimensions", f"{fingerprint.raw_dimensions[0]} x {fingerprint.raw_dimensions[1]}")
    cols[1].metric("Processed Dimensions", f"{fingerprint.processed_dimensions[0]} x {fingerprint.processed_dimensions[1]}")
    cols[2].metric("Runtime", f"{runtime:.3f} s")
    if reference_metrics:
        formatted = format_reference_metrics(reference_metrics)
        cols[3].metric("PSNR", formatted["PSNR"])
        cols[4].metric("SSIM", formatted["SSIM"])
        st.metric("MSE", formatted["MSE"])
    else:
        cols[3].metric("PSNR", "Unavailable")
        cols[4].metric("SSIM", "Unavailable")
        st.info("Reference-based PSNR, SSIM and MSE are unavailable because no ground-truth image was supplied.")


def render_structural_metrics(before: dict[str, float], after: dict[str, float]) -> None:
    st.subheader("No-Reference Structural Indicators")
    st.caption("Computed using the existing project structural metric function. These are not PSNR, SSIM or MSE.")
    st.dataframe(structural_table(before, after), use_container_width=True, hide_index=True)


def process_single(
    uploaded_file: Any,
    reference_file: Any | None,
    method: str,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    fingerprint = preprocess_upload(uploaded_file)
    enhanced, runtime = apply_frozen_method(fingerprint.p0, method, configuration)
    before_structural, after_structural = calculate_structural_pair(fingerprint.p0, enhanced)

    reference = preprocess_upload(reference_file) if reference_file is not None else None
    reference_metrics = calculate_reference_metrics(reference.p0, enhanced) if reference else None
    input_reference_metrics = calculate_reference_metrics(reference.p0, fingerprint.p0) if reference else None

    return {
        "fingerprint": fingerprint,
        "reference": reference,
        "method": method,
        "enhanced": enhanced,
        "runtime": runtime,
        "reference_metrics": reference_metrics,
        "input_reference_metrics": input_reference_metrics,
        "before_structural": before_structural,
        "after_structural": after_structural,
    }


def process_comparison(uploaded_file: Any, reference_file: Any | None, configuration: dict[str, Any]) -> dict[str, Any]:
    fingerprint = preprocess_upload(uploaded_file)
    reference = preprocess_upload(reference_file) if reference_file is not None else None
    rows = []
    results = {}

    for method in configuration["methods"]:
        enhanced, runtime = apply_frozen_method(fingerprint.p0, method, configuration)
        before_structural, after_structural = calculate_structural_pair(fingerprint.p0, enhanced)
        row = {"Technique": method, "Runtime (s)": runtime}
        if reference is not None:
            metrics = calculate_reference_metrics(reference.p0, enhanced)
            row.update({"PSNR": metrics["psnr"], "SSIM": metrics["ssim"], "MSE": metrics["mse"]})
        for key in STRUCTURAL_LABELS:
            row[f"{STRUCTURAL_LABELS[key]} Change"] = after_structural[key] - before_structural[key]
        rows.append(row)
        results[method] = {
            "enhanced": enhanced,
            "runtime": runtime,
            "before_structural": before_structural,
            "after_structural": after_structural,
        }

    return {
        "fingerprint": fingerprint,
        "reference": reference,
        "results": results,
        "table": pd.DataFrame(rows),
    }


def process_bulk(files: list[Any], method: str, configuration: dict[str, Any]) -> dict[str, Any]:
    rows = []
    results = {}
    for index, file in enumerate(files, start=1):
        preview_key = f"{index}. {file.name}"
        try:
            fingerprint = preprocess_upload(file)
            enhanced, runtime = apply_frozen_method(fingerprint.p0, method, configuration)
            before_structural, after_structural = calculate_structural_pair(fingerprint.p0, enhanced)
            rows.append(
                {
                    "File": file.name,
                    "Status": "Processed",
                    "Input Dimensions": f"{fingerprint.raw_dimensions[0]} x {fingerprint.raw_dimensions[1]}",
                    "Processed Dimensions": f"{fingerprint.processed_dimensions[0]} x {fingerprint.processed_dimensions[1]}",
                    "Runtime (s)": runtime,
                    "Coherence Change": after_structural["coherence"] - before_structural["coherence"],
                    "Contrast Change": after_structural["contrast"] - before_structural["contrast"],
                    "Continuity Change": after_structural["continuity"] - before_structural["continuity"],
                }
            )
            results[preview_key] = {
                "source_name": file.name,
                "fingerprint": fingerprint,
                "enhanced": enhanced,
                "runtime": runtime,
                "before_structural": before_structural,
                "after_structural": after_structural,
            }
        except Exception as exc:
            rows.append(
                {
                    "File": file.name,
                    "Status": f"Failed: {exc}",
                    "Input Dimensions": "",
                    "Processed Dimensions": "",
                    "Runtime (s)": np.nan,
                    "Coherence Change": np.nan,
                    "Contrast Change": np.nan,
                    "Continuity Change": np.nan,
                }
            )
    return {"method": method, "table": pd.DataFrame(rows), "results": results}


def build_zip_download(results: dict[str, dict[str, Any]], method: str) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (filename, result) in enumerate(results.items(), start=1):
            source_name = result.get("source_name", filename)
            stem = Path(source_name).stem or "fingerprint"
            output_name = f"{index:02d}_{stem}_enhanced_{method_slug(method)}.png"
            archive.writestr(output_name, encode_png(result["enhanced"]))
    return buffer.getvalue()


def build_comparison_zip(results: dict[str, dict[str, Any]]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for method, result in results.items():
            archive.writestr(f"enhanced_{method_slug(method)}.png", encode_png(result["enhanced"]))
    return buffer.getvalue()


def render_project_evidence(configuration: dict[str, Any]) -> None:
    raw = configuration["raw"]
    validation = load_validation_comparison()
    with st.sidebar:
        st.subheader("Frozen Project Evidence")
        st.write(f"Recommended Technique: **{configuration['selected_method']}**")
        st.caption("Selected using the completed development-set benchmark and confirmed with the validation protocol.")
        st.write(f"Development images: **{raw.get('development_images', 'n/a')}**")
        st.write(f"Validation images: **{raw.get('validation_images', 'n/a')}**")
        with st.expander("Frozen Parameters"):
            st.json(raw.get("all_member_parameters", {}))
        if not validation.empty:
            with st.expander("Validation Summary"):
                columns = ["method", "images", "mean_psnr", "mean_ssim", "mean_mse", "runtime_seconds_mean"]
                existing = [column for column in columns if column in validation.columns]
                st.dataframe(validation[existing], use_container_width=True, hide_index=True)


def render_header(configuration: dict[str, Any]) -> None:
    st.title("Fingerprint Enhancement System")
    st.markdown("#### Classical Image Processing for Ridge Enhancement")
    st.write(
        "Enhance low-quality fingerprint images and improve ridge visibility using classical image-processing techniques."
    )
    st.info(
        f"The application uses frozen parameters from the completed experiment. "
        f"Project-wide recommended technique: {configuration['selected_method']}."
    )


def render_single_or_compare(configuration: dict[str, Any]) -> None:
    st.header("Image Enhancement")
    uploaded_file = st.file_uploader(
        "Upload Fingerprint Image",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
        key="single_upload",
    )
    reference_file = st.file_uploader(
        "Upload Reference / Ground-Truth Image",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
        key="reference_upload",
        help="Optional. Reference-based PSNR, SSIM and MSE are calculated only when this image is supplied.",
    )

    method_options = configuration["methods"] + [COMPARE_ALL]
    selected = st.selectbox("Enhancement Technique", method_options, key="selected_method")
    action_label = "Compare techniques" if selected == COMPARE_ALL else "Enhance fingerprint"
    current_signature = (file_signature(uploaded_file), file_signature(reference_file), selected)

    if st.button(action_label, type="primary", disabled=uploaded_file is None):
        try:
            with st.spinner("Processing fingerprint image..."):
                if selected == COMPARE_ALL:
                    st.session_state["comparison_result"] = process_comparison(uploaded_file, reference_file, configuration)
                    st.session_state["comparison_signature"] = current_signature
                    st.session_state.pop("single_result", None)
                else:
                    st.session_state["single_result"] = process_single(uploaded_file, reference_file, selected, configuration)
                    st.session_state["single_signature"] = current_signature
                    st.session_state.pop("comparison_result", None)
        except Exception as exc:
            st.error("The image could not be processed. Please check the file and try again.")
            with st.expander("Technical detail"):
                st.code(str(exc))

    single_result = st.session_state.get("single_result")
    if (
        single_result
        and selected != COMPARE_ALL
        and st.session_state.get("single_signature") == current_signature
    ):
        render_single_result(single_result)

    comparison_result = st.session_state.get("comparison_result")
    if (
        comparison_result
        and selected == COMPARE_ALL
        and st.session_state.get("comparison_signature") == current_signature
    ):
        render_comparison_result(comparison_result)


def render_single_result(result: dict[str, Any]) -> None:
    fingerprint = result["fingerprint"]
    method = result["method"]
    enhanced = result["enhanced"]

    st.subheader("Before / After Visualisation")
    render_image_triplet(fingerprint.original_gray, fingerprint.p0, enhanced, method)

    st.subheader("Data Analysis Dashboard")
    if result["reference"] and result["reference"].raw_dimensions != fingerprint.raw_dimensions:
        st.warning(
            "The reference image dimensions differ from the input image. "
            "Metrics are calculated after both images are standardised with P0."
        )
    render_reference_cards(result["reference_metrics"], result["runtime"], fingerprint)
    if result["reference_metrics"]:
        st.caption("Reference metrics are calculated against the supplied ground-truth image after P0 standardisation.")
        metrics_df = pd.DataFrame(
            [
                {"Image": "P0 Input", **result["input_reference_metrics"]},
                {"Image": f"Enhanced - {method}", **result["reference_metrics"]},
            ]
        )
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    render_structural_metrics(result["before_structural"], result["after_structural"])
    st.download_button(
        "Download Enhanced PNG",
        data=encode_png(enhanced),
        file_name=f"enhanced_{method_slug(method)}.png",
        mime="image/png",
    )


def render_comparison_result(result: dict[str, Any]) -> None:
    fingerprint = result["fingerprint"]
    reference = result["reference"]
    st.subheader("Original and P0")
    original_col, p0_col = st.columns(2)
    original_col.image(fingerprint.original_gray, caption="Original Input", use_container_width=True, clamp=True)
    p0_col.image(image_to_uint8(fingerprint.p0), caption="Preprocessed / P0", use_container_width=True, clamp=True)

    st.subheader("Technique Results")
    result_items = list(result["results"].items())
    for index in range(0, len(result_items), 2):
        cols = st.columns(2)
        for column, (method, method_result) in zip(cols, result_items[index : index + 2]):
            column.image(
                image_to_uint8(method_result["enhanced"]),
                caption=f"{method} ({method_result['runtime']:.3f} s)",
                use_container_width=True,
                clamp=True,
            )

    st.subheader("Comparison Dashboard")
    table = result["table"].copy()
    if reference and reference.raw_dimensions != fingerprint.raw_dimensions:
        st.warning(
            "The reference image dimensions differ from the input image. "
            "Metrics are calculated after both images are standardised with P0."
        )
    if reference is None:
        st.info("Reference-based PSNR, SSIM and MSE are unavailable because no ground-truth image was supplied.")
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.download_button(
        "Download All Enhanced PNGs",
        data=build_comparison_zip(result["results"]),
        file_name="enhanced_all_techniques.zip",
        mime="application/zip",
    )


def render_bulk(configuration: dict[str, Any]) -> None:
    st.header("Bulk Processing")
    st.write("Upload multiple fingerprint images and apply one frozen enhancement technique to each file.")
    uploaded_files = st.file_uploader(
        "Upload Multiple Fingerprint Images",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key="bulk_upload",
    )
    method = st.selectbox("Bulk Enhancement Technique", configuration["methods"], key="bulk_method")
    current_signature = (files_signature(uploaded_files), method)

    if st.button("Process batch", disabled=not uploaded_files):
        try:
            with st.spinner("Processing uploaded fingerprints..."):
                st.session_state["bulk_result"] = process_bulk(uploaded_files, method, configuration)
                st.session_state["bulk_signature"] = current_signature
        except Exception as exc:
            st.error("The batch could not be processed.")
            with st.expander("Technical detail"):
                st.code(str(exc))

    bulk_result = st.session_state.get("bulk_result")
    if not bulk_result or st.session_state.get("bulk_signature") != current_signature:
        return

    st.subheader("Batch Summary")
    st.dataframe(bulk_result["table"], use_container_width=True, hide_index=True)
    if not bulk_result["results"]:
        return

    preview_name = st.selectbox("Preview Processed Result", list(bulk_result["results"]), key="bulk_preview")
    preview = bulk_result["results"][preview_name]
    render_image_triplet(
        preview["fingerprint"].original_gray,
        preview["fingerprint"].p0,
        preview["enhanced"],
        bulk_result["method"],
    )

    st.download_button(
        "Download Enhanced Batch ZIP",
        data=build_zip_download(bulk_result["results"], bulk_result["method"]),
        file_name=f"enhanced_batch_{method_slug(bulk_result['method'])}.zip",
        mime="application/zip",
    )


def main() -> None:
    st.markdown(
        """
<style>
div[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 0.8rem;
    border-radius: 8px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
</style>
""",
        unsafe_allow_html=True,
    )

    require_backend()
    try:
        configuration = load_frozen_configuration()
    except Exception as exc:
        st.error("Frozen experiment configuration could not be loaded.")
        with st.expander("Technical detail"):
            st.code(str(exc))
        st.stop()

    render_project_evidence(configuration)
    render_header(configuration)

    enhancement_tab, bulk_tab = st.tabs(["Single Image / Compare All", "Bulk Processing"])
    with enhancement_tab:
        render_single_or_compare(configuration)
    with bulk_tab:
        render_bulk(configuration)


if __name__ == "__main__":
    main()
