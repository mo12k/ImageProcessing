"""Streamlit prototype for the BMDS2133 Fingerprint Enhancement System."""
from __future__ import annotations

import base64
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image as ReportLabImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except Exception as exc:
    REPORTLAB_AVAILABLE = False
    REPORTLAB_IMPORT_ERROR = exc
    A4 = (595.2755905511812, 841.8897637795277)

    def landscape(pagesize: tuple[float, float]) -> tuple[float, float]:
        return (pagesize[1], pagesize[0])
else:
    REPORTLAB_AVAILABLE = True
    REPORTLAB_IMPORT_ERROR = None

st.set_page_config(
    page_title="Fingerprint Enhancement System",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import Fingerprint_Enhancement_System as backend
except Exception as exc:
    backend = None
    BACKEND_IMPORT_ERROR = exc
else:
    BACKEND_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parent
FROZEN_CONFIG_PATH = PROJECT_ROOT / "outputs" / "final_mode_a" / "script" / "final_selected_configuration.json"
VALIDATION_COMPARISON_PATH = PROJECT_ROOT / "outputs" / "final_mode_a" / "script" / "validation_member_comparison.csv"
COMPARE_ALL = "Compare All Techniques"
PROJECT_TITLE = "BMDS2133 Fingerprint Enhancement System"
SUPPORTED_TYPES = ["bmp", "png", "jpg", "jpeg", "tif", "tiff"]
QUALITY_BENCHMARK_EXPLANATION = (
    "These full-reference metrics are produced from a controlled synthetic degradation of the uploaded fingerprint. "
    "They evaluate how the selected algorithm restores the artificially degraded copy and do not represent "
    "ground-truth quality metrics for the original real-world upload."
)
BULK_QUALITY_BENCHMARK_EXPLANATION = (
    "PSNR, SSIM and MSE shown below are obtained from a controlled synthetic degradation of each uploaded fingerprint. "
    "They evaluate restoration of the artificially degraded copy and are not ground-truth quality measurements of "
    "the original real-world uploaded fingerprints."
)

STRUCTURAL_LABELS = {
    "coherence": "Coherence",
    "contrast": "Contrast",
    "fragmentation": "Fragmentation",
    "continuity": "Continuity",
    "ridge_coverage": "Ridge Coverage",
}

FULL_REFERENCE_LABELS = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "mse": "MSE",
}

BULK_BENCHMARK_COLUMNS = [
    "Synthetic Baseline PSNR",
    "Synthetic Enhanced PSNR",
    "PSNR Improvement",
    "Synthetic Baseline SSIM",
    "Synthetic Enhanced SSIM",
    "SSIM Improvement",
    "Synthetic Baseline MSE",
    "Synthetic Enhanced MSE",
    "MSE Reduction",
]


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


def encode_display_png(image: np.ndarray) -> bytes:
    """Encode uint8 or float image safely for HTML preview."""
    array = np.asarray(image)

    if array.dtype == np.uint8:
        output = array
    else:
        array = np.asarray(image, dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)

        if array.max() > 1.5:
            output = np.clip(array, 0, 255).astype(np.uint8)
        else:
            output = np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)

    buffer = BytesIO()
    Image.fromarray(output, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def image_data_uri(image: np.ndarray) -> str:
    encoded = base64.b64encode(encode_display_png(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def method_slug(method: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", method.lower()).strip("_")


def apply_frozen_method(p0_image: np.ndarray, method: str, configuration: dict[str, Any]) -> np.ndarray:
    fes = require_backend()
    functions = fes.member_functions()

    if method not in functions:
        raise ValueError(f"Unknown enhancement technique: {method}")

    if method not in configuration["all_member_parameters"]:
        raise ValueError(f"No frozen parameters were found for {method}.")

    enhanced = functions[method](p0_image, configuration["all_member_parameters"][method])
    return unit_float(enhanced)


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


def synthetic_benchmark_seed(fingerprint: UploadedFingerprint) -> int:
    """Create a repeatable upload-specific seed based on the project's noise seed base."""
    fes = require_backend()
    seed_base = int(getattr(fes, "NOISE_SEED_BASE", 0))
    split_seed = str(getattr(fes, "SPLIT_SEED", ""))
    p0_bytes = np.ascontiguousarray(fingerprint.p0.astype(np.float32)).tobytes()
    p0_digest = hashlib.sha256(p0_bytes).hexdigest()
    digest_seed = int(hashlib.sha256(f"{split_seed}:synthetic-upload:{p0_digest}".encode("utf-8")).hexdigest()[:8], 16)
    return (seed_base + digest_seed) % (2**32)


def quality_benchmark_table(baseline: dict[str, float], enhanced: dict[str, float]) -> pd.DataFrame:
    rows = []

    for key, label in FULL_REFERENCE_LABELS.items():
        baseline_value = float(baseline[key])
        enhanced_value = float(enhanced[key])
        rows.append(
            {
                "Metric": label,
                "Synthetic Degraded": baseline_value,
                "Enhanced": enhanced_value,
                "Change": enhanced_value - baseline_value,
            }
        )

    return pd.DataFrame(rows)


def calculate_synthetic_benchmark(
    fingerprint: UploadedFingerprint,
    method: str,
    configuration: dict[str, Any],
    synthetic_degraded: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run the controlled synthetic degradation benchmark for one frozen method."""
    fes = require_backend()
    seed = synthetic_benchmark_seed(fingerprint)
    reference = fingerprint.p0
    degraded = synthetic_degraded if synthetic_degraded is not None else fes.controlled_degradation(reference, seed)
    enhanced = apply_frozen_method(degraded, method, configuration)

    baseline_metrics = fes.full_reference_metrics(reference, degraded)
    enhanced_metrics = fes.full_reference_metrics(reference, enhanced)

    return {
        "seed": seed,
        "synthetic_degraded": degraded,
        "synthetic_enhanced": enhanced,
        "baseline_metrics": baseline_metrics,
        "enhanced_metrics": enhanced_metrics,
        "table": quality_benchmark_table(baseline_metrics, enhanced_metrics),
    }


def comparison_quality_benchmark_table(benchmarks: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for method, benchmark in benchmarks.items():
        row = {"Technique": method}
        baseline = benchmark["baseline_metrics"]
        enhanced = benchmark["enhanced_metrics"]
        for key, label in FULL_REFERENCE_LABELS.items():
            row[f"Synthetic Degraded {label}"] = float(baseline[key])
            row[f"Enhanced {label}"] = float(enhanced[key])
            row[f"{label} Change"] = float(enhanced[key]) - float(baseline[key])
        rows.append(row)

    return pd.DataFrame(rows)


def bulk_quality_benchmark_row(
    batch_item: int,
    filename: str,
    status: str,
    benchmark: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "Batch Item": batch_item,
        "File": filename,
        "Status": status,
        **{column: np.nan for column in BULK_BENCHMARK_COLUMNS},
    }

    if benchmark is None:
        return row

    baseline = benchmark["baseline_metrics"]
    enhanced = benchmark["enhanced_metrics"]
    row.update(
        {
            "Synthetic Baseline PSNR": float(baseline["psnr"]),
            "Synthetic Enhanced PSNR": float(enhanced["psnr"]),
            "PSNR Improvement": float(enhanced["psnr"]) - float(baseline["psnr"]),
            "Synthetic Baseline SSIM": float(baseline["ssim"]),
            "Synthetic Enhanced SSIM": float(enhanced["ssim"]),
            "SSIM Improvement": float(enhanced["ssim"]) - float(baseline["ssim"]),
            "Synthetic Baseline MSE": float(baseline["mse"]),
            "Synthetic Enhanced MSE": float(enhanced["mse"]),
            "MSE Reduction": float(baseline["mse"]) - float(enhanced["mse"]),
        }
    )
    return row


def bulk_export_table(result: dict[str, Any]) -> pd.DataFrame:
    structural = result.get("structural_table", result["table"]).copy()
    benchmark = result.get("benchmark_table")

    if benchmark is None:
        return structural

    return benchmark.merge(structural, on=["Batch Item", "File", "Status"], how="outer")


def synthetic_benchmark_winner(table: pd.DataFrame) -> pd.Series | None:
    if table.empty:
        return None

    ranked = table.sort_values(["Enhanced PSNR", "Enhanced SSIM", "Enhanced MSE"], ascending=[False, False, True])
    return ranked.iloc[0]


def report_timestamp() -> str:
    """Return a local timestamp string for generated exports."""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def dimensions_text(dimensions: tuple[int, int]) -> str:
    """Format image dimensions consistently as width x height."""
    return f"{dimensions[0]} x {dimensions[1]}"


def dataframe_to_csv_bytes(table: pd.DataFrame) -> bytes:
    """Encode a result table as UTF-8 CSV bytes for Streamlit downloads."""
    return table.to_csv(index=False).encode("utf-8")


def format_report_value(value: Any, decimals: int = 4) -> str:
    """Format numeric values for compact PDF and CSV-adjacent display."""
    if value is None:
        return "Unavailable"

    try:
        if pd.isna(value):
            return "Unavailable"
    except TypeError:
        pass

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{decimals}f}"

    return str(value)


def structural_change_summary(before: dict[str, float], after: dict[str, float]) -> str:
    """Create a short descriptive summary of no-reference structural indicator changes."""
    changes = {label: float(after[key]) - float(before[key]) for key, label in STRUCTURAL_LABELS.items()}
    ordered = sorted(changes.items(), key=lambda item: abs(item[1]), reverse=True)
    top = ", ".join(f"{label} {change:+.4f}" for label, change in ordered[:3])
    return f"Largest structural indicator changes: {top}. These structural indicators are not PSNR, SSIM or MSE."


def bulk_summary_statistics(result: dict[str, Any]) -> dict[str, Any]:
    """Summarise processed, failed and structural changes for a batch result."""
    table = result.get("structural_table", result["table"])
    processed_mask = table["Status"].eq("Processed") if not table.empty else pd.Series(dtype=bool)
    processed = int(processed_mask.sum())
    failed = int(len(table) - processed)
    processed_rows = table.loc[processed_mask]

    stats = {
        "chosen_technique": result["method"],
        "uploaded_count": int(len(table)),
        "processed_count": processed,
        "failed_count": failed,
        **{
            f"mean_{key}_change": float(processed_rows[f"{label} Change"].mean()) if processed else np.nan
            for key, label in STRUCTURAL_LABELS.items()
        },
    }

    benchmark = result.get("benchmark_table")
    if benchmark is not None and not benchmark.empty:
        benchmark_rows = benchmark.loc[benchmark["Status"].eq("Processed")]
        for column in BULK_BENCHMARK_COLUMNS:
            stats[f"mean_{column}"] = float(benchmark_rows[column].mean()) if not benchmark_rows.empty else np.nan

    return stats


def bulk_summary_text(stats: dict[str, Any]) -> str:
    """Create an automated natural-language summary for batch reporting."""
    if stats["processed_count"] == 0:
        return "No uploaded images were successfully processed, so no aggregate structural changes are available."

    mean_changes = ", ".join(
        f"{label.lower()} {format_report_value(stats[f'mean_{key}_change'])}"
        for key, label in STRUCTURAL_LABELS.items()
    )
    return (
        f"The batch successfully processed {stats['processed_count']} of {stats['uploaded_count']} fingerprint images "
        f"with {stats['chosen_technique']}. Across successfully processed images, the mean structural changes were "
        f"{mean_changes}."
    )


def bulk_benchmark_summary_rows(stats: dict[str, Any]) -> list[list[Any]]:
    labels = [
        ("Mean baseline PSNR", "mean_Synthetic Baseline PSNR"),
        ("Mean enhanced PSNR", "mean_Synthetic Enhanced PSNR"),
        ("Mean PSNR improvement", "mean_PSNR Improvement"),
        ("Mean baseline SSIM", "mean_Synthetic Baseline SSIM"),
        ("Mean enhanced SSIM", "mean_Synthetic Enhanced SSIM"),
        ("Mean SSIM improvement", "mean_SSIM Improvement"),
        ("Mean baseline MSE", "mean_Synthetic Baseline MSE"),
        ("Mean enhanced MSE", "mean_Synthetic Enhanced MSE"),
        ("Mean MSE reduction", "mean_MSE Reduction"),
    ]
    return [["Field", "Value"], *[[label, format_report_value(stats.get(key))] for label, key in labels]]


def require_reportlab() -> None:
    """Raise a readable error when PDF export dependencies are unavailable."""
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(f"ReportLab is unavailable: {REPORTLAB_IMPORT_ERROR}")


def pdf_styles() -> dict[str, ParagraphStyle]:
    """Build the small style set used by generated PDF reports."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="Caption", parent=styles["Small"], textColor=colors.HexColor("#475569")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=12, leading=15, spaceBefore=8))
    return styles


def pdf_paragraph(text: Any, style: ParagraphStyle) -> Paragraph:
    """Create a ReportLab paragraph with escaped text."""
    return Paragraph(escape(str(text)), style)


def pdf_image(image: np.ndarray, max_width: float, max_height: float) -> ReportLabImage:
    """Create a ReportLab image flowable from a NumPy image without writing to disk."""
    png_bytes = encode_display_png(image)

    with Image.open(BytesIO(png_bytes)) as pil_image:
        width, height = pil_image.size

    scale = min(max_width / max(width, 1), max_height / max(height, 1))
    return ReportLabImage(BytesIO(png_bytes), width=width * scale, height=height * scale)


def pdf_table(
    rows: list[list[Any]],
    styles: dict[str, ParagraphStyle],
    column_widths: list[float] | None = None,
    font_size: int = 8,
) -> Table:
    """Create a wrapped, repeat-header table for PDF reports."""
    body_style = styles["Small"]
    table_rows = [[cell if hasattr(cell, "wrap") else pdf_paragraph(cell, body_style) for cell in row] for row in rows]
    table = Table(table_rows, colWidths=column_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def dataframe_report_rows(table: pd.DataFrame, decimals: int = 4) -> list[list[str]]:
    """Convert a dataframe to printable rows with unavailable values made explicit."""
    if table.empty:
        return [["Status"], ["No rows available"]]

    rows = [list(table.columns)]

    for row in table.itertuples(index=False, name=None):
        rows.append([format_report_value(value, decimals) for value in row])

    return rows


def validation_evidence_table() -> pd.DataFrame:
    """Return frozen validation evidence columns that are safe to show as dataset-level metrics."""
    validation = load_validation_comparison()
    columns = ["method", "images", "mean_psnr", "mean_ssim", "mean_mse"]
    existing = [column for column in columns if column in validation.columns]
    return validation[existing] if existing else pd.DataFrame()


def build_report_document(
    story: list[Any],
    pagesize: tuple[float, float] = A4,
    title: str = PROJECT_TITLE,
) -> bytes:
    """Render a ReportLab story into PDF bytes in memory."""
    require_reportlab()
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=pagesize,
        title=title,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )
    document.build(story)
    return buffer.getvalue()


def build_single_analysis_pdf(result: dict[str, Any], configuration: dict[str, Any]) -> bytes:
    """Build a single-image analysis PDF report entirely in memory."""
    require_reportlab()
    styles = pdf_styles()
    fingerprint = result["fingerprint"]
    method = result["method"]
    structural_rows = dataframe_report_rows(structural_table(result["before_structural"], result["after_structural"]))
    validation_evidence = validation_evidence_table()

    story: list[Any] = [
        pdf_paragraph(PROJECT_TITLE, styles["Title"]),
        pdf_paragraph(f"Generated: {report_timestamp()}", styles["Caption"]),
        Spacer(1, 0.12 * inch),
        pdf_paragraph("Analysis Details", styles["Section"]),
        pdf_table(
            [
                ["Field", "Value"],
                ["Uploaded filename", fingerprint.name],
                ["Selected enhancement technique", method],
                ["Project-recommended frozen technique", configuration["selected_method"]],
                ["Input dimensions", dimensions_text(fingerprint.raw_dimensions)],
                ["Processed dimensions", dimensions_text(fingerprint.processed_dimensions)],
            ],
            styles,
            [2.2 * inch, 4.6 * inch],
        ),
        Spacer(1, 0.12 * inch),
        pdf_paragraph("Images", styles["Section"]),
        pdf_table(
            [
                ["Original Input", "P0 Preprocessed", "Enhanced Image"],
                [
                    pdf_image(fingerprint.original_gray, 2.0 * inch, 2.0 * inch),
                    pdf_image(fingerprint.p0, 2.0 * inch, 2.0 * inch),
                    pdf_image(result["enhanced"], 2.0 * inch, 2.0 * inch),
                ],
            ],
            styles,
            [2.15 * inch, 2.15 * inch, 2.15 * inch],
        ),
        Spacer(1, 0.12 * inch),
        pdf_paragraph("No-Reference Structural Indicators", styles["Section"]),
        pdf_table(structural_rows, styles),
        pdf_paragraph(
            "These structural indicators describe changes in fingerprint ridge structure. "
            "They are not PSNR, SSIM, MSE or recognition accuracy.",
            styles["Caption"],
        ),
        Spacer(1, 0.12 * inch),
        pdf_paragraph("Structural Interpretation", styles["Section"]),
        pdf_paragraph(structural_change_summary(result["before_structural"], result["after_structural"]), styles["BodyText"]),
        pdf_paragraph(
            "Full-reference metrics such as PSNR, SSIM and MSE are evaluated separately using the project's "
            "controlled Development and Validation experiments and are not calculated for arbitrary user-uploaded "
            "fingerprints without ground truth.",
            styles["BodyText"],
        ),
    ]

    if result.get("quality_benchmark"):
        benchmark = result["quality_benchmark"]
        story.extend(
            [
                Spacer(1, 0.12 * inch),
                pdf_paragraph("Controlled Quality Benchmark", styles["Section"]),
                pdf_table(dataframe_report_rows(benchmark["table"]), styles),
                pdf_paragraph(QUALITY_BENCHMARK_EXPLANATION, styles["Caption"]),
            ]
        )

    if not validation_evidence.empty:
        story.extend(
            [
                Spacer(1, 0.12 * inch),
                pdf_paragraph("Project Validation Evidence", styles["Section"]),
                pdf_table(dataframe_report_rows(validation_evidence), styles),
                pdf_paragraph(
                    "These are dataset-level frozen validation benchmark results, not metrics for the current uploaded image.",
                    styles["Caption"],
                ),
            ]
        )

    return build_report_document(story, title="Fingerprint Analysis Report")


def build_comparison_report_pdf(result: dict[str, Any], configuration: dict[str, Any]) -> bytes:
    """Build a compare-all PDF report entirely in memory."""
    require_reportlab()
    styles = pdf_styles()
    fingerprint = result["fingerprint"]
    table = result["table"].copy()
    validation_evidence = validation_evidence_table()

    story: list[Any] = [
        pdf_paragraph(PROJECT_TITLE, styles["Title"]),
        pdf_paragraph("Comparison Report", styles["Heading1"]),
        pdf_paragraph(f"Generated: {report_timestamp()}", styles["Caption"]),
        pdf_paragraph(f"Uploaded filename: {fingerprint.name}", styles["BodyText"]),
        pdf_paragraph(f"Project frozen recommended technique: {configuration['selected_method']}", styles["BodyText"]),
        pdf_paragraph(
            "The recommendation comes from the completed controlled Development and Validation benchmark.",
            styles["Caption"],
        ),
        Spacer(1, 0.12 * inch),
        pdf_paragraph("Original and P0", styles["Section"]),
        pdf_table(
            [
                ["Original Input", "P0 Preprocessed"],
                [
                    pdf_image(fingerprint.original_gray, 2.6 * inch, 2.2 * inch),
                    pdf_image(fingerprint.p0, 2.6 * inch, 2.2 * inch),
                ],
            ],
            styles,
            [3.0 * inch, 3.0 * inch],
        ),
        Spacer(1, 0.12 * inch),
        pdf_paragraph("Technique Outputs", styles["Section"]),
    ]

    image_rows: list[list[Any]] = []
    for method, method_result in result["results"].items():
        image_rows.append(
            [
                method,
                pdf_image(method_result["enhanced"], 1.55 * inch, 1.45 * inch),
            ]
        )

    story.extend(
        [
            pdf_table([["Technique", "Enhanced Output"], *image_rows], styles, [2.5 * inch, 1.8 * inch]),
            Spacer(1, 0.12 * inch),
            pdf_paragraph("No-Reference Structural Comparison", styles["Section"]),
            pdf_table(dataframe_report_rows(table), styles),
            pdf_paragraph(
                "Structural changes are no-reference indicators and are not PSNR, SSIM, MSE or recognition accuracy.",
                styles["Caption"],
            ),
        ]
    )

    if result.get("quality_benchmark_table") is not None:
        benchmark_table = result["quality_benchmark_table"]
        winner = synthetic_benchmark_winner(benchmark_table)
        story.extend(
            [
                Spacer(1, 0.12 * inch),
                pdf_paragraph("Controlled Quality Benchmark", styles["Section"]),
                pdf_table(dataframe_report_rows(benchmark_table), styles),
                pdf_paragraph(QUALITY_BENCHMARK_EXPLANATION, styles["Caption"]),
            ]
        )
        if winner is not None:
            story.append(
                pdf_paragraph(
                    f"Synthetic Benchmark Winner: {winner['Technique']}. "
                    f"The project overall recommended method remains {configuration['selected_method']}.",
                    styles["BodyText"],
                )
            )

    if not validation_evidence.empty:
        story.extend(
            [
                Spacer(1, 0.12 * inch),
                pdf_paragraph("Project Validation Evidence", styles["Section"]),
                pdf_table(dataframe_report_rows(validation_evidence), styles),
                pdf_paragraph(
                    "These frozen PSNR, SSIM and MSE values are dataset-level benchmark results, not metrics for this upload.",
                    styles["Caption"],
                ),
            ]
        )

    return build_report_document(story, pagesize=landscape(A4), title="Fingerprint Comparison Report")


def build_bulk_report_pdf(result: dict[str, Any]) -> bytes:
    """Build a batch processing PDF report entirely in memory."""
    require_reportlab()
    styles = pdf_styles()
    stats = bulk_summary_statistics(result)
    structural = result.get("structural_table", result["table"]).copy()
    benchmark = result.get("benchmark_table")

    summary_rows = [
        ["Field", "Value"],
        ["Chosen technique", stats["chosen_technique"]],
        ["Uploaded files", stats["uploaded_count"]],
        ["Processed count", stats["processed_count"]],
        ["Failed count", stats["failed_count"]],
    ]
    summary_rows.extend(
        [f"Mean {label.lower()} change", format_report_value(stats[f"mean_{key}_change"])]
        for key, label in STRUCTURAL_LABELS.items()
    )

    story: list[Any] = [
        pdf_paragraph(PROJECT_TITLE, styles["Title"]),
        pdf_paragraph("Batch Report", styles["Heading1"]),
        pdf_paragraph(f"Generated: {report_timestamp()}", styles["Caption"]),
        Spacer(1, 0.12 * inch),
        pdf_paragraph("Batch Summary", styles["Section"]),
        pdf_table(summary_rows, styles, [2.2 * inch, 4.8 * inch]),
        Spacer(1, 0.12 * inch),
        pdf_paragraph("Automated Summary", styles["Section"]),
        pdf_paragraph(bulk_summary_text(stats), styles["BodyText"]),
        pdf_paragraph(
            "Structural indicators describe changes in ridge structure. They are not PSNR, SSIM, MSE or recognition accuracy.",
            styles["Caption"],
        ),
        Spacer(1, 0.12 * inch),
    ]

    if benchmark is not None:
        story.extend(
            [
                pdf_paragraph("Controlled Quality Benchmark", styles["Section"]),
                pdf_paragraph(BULK_QUALITY_BENCHMARK_EXPLANATION, styles["BodyText"]),
                pdf_paragraph("Synthetic Benchmark Means", styles["Section"]),
                pdf_table(bulk_benchmark_summary_rows(stats), styles, [2.4 * inch, 1.4 * inch]),
                Spacer(1, 0.12 * inch),
                pdf_paragraph("Per-File Benchmark Metrics", styles["Section"]),
                pdf_table(dataframe_report_rows(benchmark), styles, font_size=7),
                Spacer(1, 0.12 * inch),
            ]
        )

    story.extend(
        [
            pdf_paragraph("Actual Image Structural Indicators", styles["Section"]),
            pdf_table(dataframe_report_rows(structural), styles),
        ]
    )

    return build_report_document(story, pagesize=landscape(A4), title="Fingerprint Batch Report")


def render_pdf_download(label: str, data_factory: Any, file_name: str, key: str) -> None:
    """Render a PDF download button and keep ReportLab failures graceful."""
    try:
        pdf_bytes = data_factory()
    except RuntimeError as exc:
        st.warning("PDF export is unavailable because ReportLab could not be imported.")
        with st.expander("PDF export technical detail"):
            st.code(str(exc))
        return
    except Exception as exc:
        st.error("The PDF report could not be generated.")
        with st.expander("PDF export technical detail"):
            st.code(str(exc))
        return

    st.download_button(label, data=pdf_bytes, file_name=file_name, mime="application/pdf", key=key)


def render_image_triplet(
    original: np.ndarray,
    p0: np.ndarray,
    enhanced: np.ndarray,
    method: str,
) -> None:
    cards = [
        {
            "badge": "01",
            "title": "Original Input",
            "subtitle": "Uploaded fingerprint image",
            "image": original,
        },
        {
            "badge": "02",
            "title": "Preprocessed / P0",
            "subtitle": "Grayscale, resized and normalised",
            "image": p0,
        },
        {
            "badge": "03",
            "title": "Enhanced Result",
            "subtitle": method,
            "image": enhanced,
        },
    ]

    card_html_parts = []

    for card in cards:
        card_html_parts.append(
            dedent(
                f"""
                <div class="fp-image-card">
                    <div class="fp-image-card-header">
                        <span class="fp-step-badge">{escape(card["badge"])}</span>
                        <div>
                            <div class="fp-card-title">{escape(card["title"])}</div>
                            <div class="fp-card-subtitle">{escape(card["subtitle"])}</div>
                        </div>
                    </div>

                    <div class="fp-image-frame">
                        <img src="{image_data_uri(card["image"])}" alt="{escape(card["title"])}">
                    </div>
                </div>
                """
            ).strip()
        )

    card_html = "\n".join(card_html_parts)

    html = dedent(
        f"""
        <div class="fp-visual-section">
            <div class="fp-section-header">
                <div>
                    <div class="fp-section-kicker">Visual Output</div>
                    <div class="fp-section-title">Before / After Visualisation</div>
                </div>
                <div class="fp-section-note">P0 &gt; Enhancement Pipeline</div>
            </div>

            <div class="fp-image-grid">
        {card_html}
            </div>
        </div>
        """
    ).strip()
    html = "\n".join(line.lstrip() for line in html.splitlines())

    st.markdown(html, unsafe_allow_html=True)


def render_image_detail_cards(fingerprint: UploadedFingerprint) -> None:
    cols = st.columns(2)

    cols[0].metric("Input Dimensions", f"{fingerprint.raw_dimensions[0]} x {fingerprint.raw_dimensions[1]}")
    cols[1].metric("Processed Dimensions", f"{fingerprint.processed_dimensions[0]} x {fingerprint.processed_dimensions[1]}")


def render_structural_metrics(before: dict[str, float], after: dict[str, float]) -> None:
    st.subheader("Structural Indicators")
    st.caption(
        "These structural indicators describe changes in fingerprint ridge structure. "
        "They are not PSNR, SSIM, MSE or recognition accuracy."
    )
    st.dataframe(structural_table(before, after), width="stretch", hide_index=True)


def render_controlled_quality_benchmark(benchmark: dict[str, Any]) -> None:
    st.subheader("Controlled Quality Benchmark")
    st.dataframe(benchmark["table"], width="stretch", hide_index=True)
    st.caption(QUALITY_BENCHMARK_EXPLANATION)


def render_comparison_quality_benchmark(result: dict[str, Any], configuration: dict[str, Any]) -> None:
    benchmark_table = result.get("quality_benchmark_table")
    if benchmark_table is None:
        return

    st.subheader("Controlled Quality Benchmark")
    st.dataframe(benchmark_table, width="stretch", hide_index=True)
    st.caption(QUALITY_BENCHMARK_EXPLANATION)

    winner = synthetic_benchmark_winner(benchmark_table)
    if winner is not None:
        st.success(
            f"Synthetic Benchmark Winner: {winner['Technique']}. "
            f"The project overall recommended method remains {configuration['selected_method']}."
        )


def process_single(
    uploaded_file: Any,
    method: str,
    configuration: dict[str, Any],
    include_quality_benchmark: bool = False,
) -> dict[str, Any]:
    fingerprint = preprocess_upload(uploaded_file)
    enhanced = apply_frozen_method(fingerprint.p0, method, configuration)
    before_structural, after_structural = calculate_structural_pair(fingerprint.p0, enhanced)
    benchmark = calculate_synthetic_benchmark(fingerprint, method, configuration) if include_quality_benchmark else None

    return {
        "fingerprint": fingerprint,
        "method": method,
        "enhanced": enhanced,
        "before_structural": before_structural,
        "after_structural": after_structural,
        "quality_benchmark": benchmark,
    }


def process_comparison(
    uploaded_file: Any,
    configuration: dict[str, Any],
    include_quality_benchmark: bool = False,
) -> dict[str, Any]:
    fingerprint = preprocess_upload(uploaded_file)
    rows = []
    results = {}
    benchmark_results = {}
    synthetic_degraded = None

    if include_quality_benchmark:
        synthetic_degraded = require_backend().controlled_degradation(fingerprint.p0, synthetic_benchmark_seed(fingerprint))

    for method in configuration["methods"]:
        enhanced = apply_frozen_method(fingerprint.p0, method, configuration)
        before_structural, after_structural = calculate_structural_pair(fingerprint.p0, enhanced)

        row = {
            "Technique": method,
        }

        for key in STRUCTURAL_LABELS:
            row[f"{STRUCTURAL_LABELS[key]} Change"] = after_structural[key] - before_structural[key]

        rows.append(row)

        results[method] = {
            "enhanced": enhanced,
            "before_structural": before_structural,
            "after_structural": after_structural,
        }

        if include_quality_benchmark:
            benchmark_results[method] = calculate_synthetic_benchmark(
                fingerprint,
                method,
                configuration,
                synthetic_degraded=synthetic_degraded,
            )

    comparison_result = {
        "fingerprint": fingerprint,
        "results": results,
        "table": pd.DataFrame(rows),
        "quality_benchmarks": benchmark_results,
    }
    if include_quality_benchmark:
        comparison_result["quality_benchmark_table"] = comparison_quality_benchmark_table(benchmark_results)

    return comparison_result


def process_bulk(
    files: list[Any],
    method: str,
    configuration: dict[str, Any],
    include_quality_benchmark: bool = False,
) -> dict[str, Any]:
    structural_rows = []
    benchmark_rows = []
    results = {}

    for index, file in enumerate(files, start=1):
        preview_key = f"{index}. {file.name}"

        try:
            fingerprint = preprocess_upload(file)
            enhanced = apply_frozen_method(fingerprint.p0, method, configuration)
            before_structural, after_structural = calculate_structural_pair(fingerprint.p0, enhanced)

            row = {
                "Batch Item": index,
                "File": file.name,
                "Status": "Processed",
                "Input Dimensions": f"{fingerprint.raw_dimensions[0]} x {fingerprint.raw_dimensions[1]}",
                "Processed Dimensions": f"{fingerprint.processed_dimensions[0]} x {fingerprint.processed_dimensions[1]}",
            }
            for key, label in STRUCTURAL_LABELS.items():
                row[f"{label} Change"] = after_structural[key] - before_structural[key]
            structural_rows.append(row)

            benchmark = None
            if include_quality_benchmark:
                benchmark = calculate_synthetic_benchmark(fingerprint, method, configuration)
                benchmark_rows.append(bulk_quality_benchmark_row(index, file.name, "Processed", benchmark))

            results[preview_key] = {
                "source_name": file.name,
                "fingerprint": fingerprint,
                "enhanced": enhanced,
                "before_structural": before_structural,
                "after_structural": after_structural,
                "quality_benchmark": benchmark,
            }

        except Exception as exc:
            structural_rows.append(
                {
                    "Batch Item": index,
                    "File": file.name,
                    "Status": f"Failed: {exc}",
                    "Input Dimensions": "",
                    "Processed Dimensions": "",
                    **{f"{label} Change": np.nan for label in STRUCTURAL_LABELS.values()},
                }
            )
            if include_quality_benchmark:
                benchmark_rows.append(bulk_quality_benchmark_row(index, file.name, f"Failed: {exc}", None))

    structural_table_df = pd.DataFrame(structural_rows)
    result = {
        "method": method,
        "table": structural_table_df,
        "structural_table": structural_table_df,
        "results": results,
        "include_quality_benchmark": include_quality_benchmark,
    }

    if include_quality_benchmark:
        benchmark_table_df = pd.DataFrame(benchmark_rows)
        result["benchmark_table"] = benchmark_table_df
        result["table"] = bulk_export_table(result)

    return result


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
                columns = [
                    "method",
                    "images",
                    "mean_psnr",
                    "mean_ssim",
                    "mean_mse",
                ]
                existing = [column for column in columns if column in validation.columns]
                st.dataframe(validation[existing], width="stretch", hide_index=True)


def render_header(configuration: dict[str, Any]) -> None:
    html = dedent(
        f"""
        <div class="fp-hero">
            <div>
                <div class="fp-hero-kicker">BMDS2133 Image Processing</div>
                <h1>Fingerprint Enhancement System</h1>
                <p>
                    Enhance low-quality fingerprint images and improve ridge visibility using classical
                    image-processing techniques.
                </p>
            </div>
            <div class="fp-hero-badge">
                <span>Recommended Technique</span>
                <strong>{escape(configuration["selected_method"])}</strong>
            </div>
        </div>
        """
    ).strip()

    st.markdown(html, unsafe_allow_html=True)


def render_single_or_compare(configuration: dict[str, Any]) -> None:
    st.header("Image Enhancement")

    uploaded_file = st.file_uploader(
        "Upload Fingerprint Image",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
        key="single_upload",
    )

    method_options = configuration["methods"] + [COMPARE_ALL]
    selected = st.selectbox("Enhancement Technique", method_options, key="selected_method")
    include_quality_benchmark = st.checkbox(
        "Include Controlled Quality Benchmark",
        value=False,
        key="include_quality_benchmark",
        help=(
            "Runs a deterministic synthetic degradation benchmark using the uploaded fingerprint's P0 image "
            "as a temporary reference."
        ),
    )
    action_label = "Compare techniques" if selected == COMPARE_ALL else "Enhance fingerprint"
    current_signature = (file_signature(uploaded_file), selected, include_quality_benchmark)

    if st.button(action_label, type="primary", disabled=uploaded_file is None):
        try:
            with st.spinner("Processing fingerprint image..."):
                if selected == COMPARE_ALL:
                    st.session_state["comparison_result"] = process_comparison(
                        uploaded_file,
                        configuration,
                        include_quality_benchmark=include_quality_benchmark,
                    )
                    st.session_state["comparison_signature"] = current_signature
                    st.session_state.pop("single_result", None)
                else:
                    st.session_state["single_result"] = process_single(
                        uploaded_file,
                        selected,
                        configuration,
                        include_quality_benchmark=include_quality_benchmark,
                    )
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
        render_single_result(single_result, configuration)

    comparison_result = st.session_state.get("comparison_result")

    if (
        comparison_result
        and selected == COMPARE_ALL
        and st.session_state.get("comparison_signature") == current_signature
    ):
        render_comparison_result(comparison_result, configuration)


def render_single_result(result: dict[str, Any], configuration: dict[str, Any]) -> None:
    fingerprint = result["fingerprint"]
    method = result["method"]
    enhanced = result["enhanced"]

    render_image_triplet(fingerprint.original_gray, fingerprint.p0, enhanced, method)

    st.subheader("Data Analysis Dashboard")

    render_image_detail_cards(fingerprint)
    render_structural_metrics(result["before_structural"], result["after_structural"])
    if result.get("quality_benchmark"):
        render_controlled_quality_benchmark(result["quality_benchmark"])

    st.download_button(
        "Download Enhanced PNG",
        data=encode_png(enhanced),
        file_name=f"enhanced_{method_slug(method)}.png",
        mime="image/png",
        key="single_download_png",
    )

    render_pdf_download(
        "Export Analysis Report (PDF)",
        lambda: build_single_analysis_pdf(result, configuration),
        file_name=f"analysis_report_{method_slug(method)}.pdf",
        key="single_export_pdf",
    )


def render_comparison_result(result: dict[str, Any], configuration: dict[str, Any]) -> None:
    fingerprint = result["fingerprint"]

    st.subheader("Original and P0")

    original_col, p0_col = st.columns(2)

    original_col.image(
        fingerprint.original_gray,
        caption="Original Input",
        width="stretch",
        clamp=True,
    )

    p0_col.image(
        image_to_uint8(fingerprint.p0),
        caption="Preprocessed / P0",
        width="stretch",
        clamp=True,
    )

    st.subheader("Technique Results")

    result_items = list(result["results"].items())

    for index in range(0, len(result_items), 2):
        cols = st.columns(2)

        for column, (method, method_result) in zip(cols, result_items[index: index + 2]):
            column.image(
                image_to_uint8(method_result["enhanced"]),
                caption=method,
                width="stretch",
                clamp=True,
            )

    st.subheader("Comparison Dashboard")

    table = result["table"].copy()

    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "These structural indicators describe changes in fingerprint ridge structure. "
        "They are not PSNR, SSIM, MSE or recognition accuracy."
    )
    st.info(
        f"Project Recommended Technique: {configuration['selected_method']}. "
        "This recommendation comes from the completed frozen Development/Validation benchmark."
    )
    render_comparison_quality_benchmark(result, configuration)

    st.download_button(
        "Export Comparison Table (CSV)",
        data=dataframe_to_csv_bytes(table),
        file_name="comparison_table.csv",
        mime="text/csv",
        key="comparison_export_csv",
    )

    st.download_button(
        "Download All Enhanced PNGs",
        data=build_comparison_zip(result["results"]),
        file_name="enhanced_all_techniques.zip",
        mime="application/zip",
        key="comparison_download_zip",
    )

    render_pdf_download(
        "Export Comparison Report (PDF)",
        lambda: build_comparison_report_pdf(result, configuration),
        file_name="comparison_report.pdf",
        key="comparison_export_pdf",
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

    if uploaded_files:
        st.success(f"Total uploaded images: {len(uploaded_files)}")

    method = st.selectbox("Bulk Enhancement Technique", configuration["methods"], key="bulk_method")
    include_quality_benchmark = st.checkbox(
        "Include Controlled Quality Benchmark",
        value=False,
        key="bulk_include_quality_benchmark",
        help=(
            "Runs a deterministic synthetic degradation benchmark for each uploaded fingerprint using its own P0 image "
            "as a temporary reference."
        ),
    )
    current_signature = (files_signature(uploaded_files), method, include_quality_benchmark)

    if st.button("Process batch", disabled=not uploaded_files):
        try:
            with st.spinner("Processing uploaded fingerprints..."):
                st.session_state["bulk_result"] = process_bulk(
                    uploaded_files,
                    method,
                    configuration,
                    include_quality_benchmark=include_quality_benchmark,
                )
                st.session_state["bulk_signature"] = current_signature

        except Exception as exc:
            st.error("The batch could not be processed.")
            with st.expander("Technical detail"):
                st.code(str(exc))

    bulk_result = st.session_state.get("bulk_result")

    if not bulk_result or st.session_state.get("bulk_signature") != current_signature:
        return

    st.subheader("Batch Summary")
    if bulk_result.get("benchmark_table") is not None:
        st.subheader("Controlled Quality Benchmark")
        st.write(BULK_QUALITY_BENCHMARK_EXPLANATION)
        st.dataframe(bulk_result["benchmark_table"], width="stretch", hide_index=True)
        st.subheader("Actual Image Structural Indicators")
        st.dataframe(bulk_result["structural_table"], width="stretch", hide_index=True)
    else:
        st.dataframe(bulk_result["table"], width="stretch", hide_index=True)

    st.download_button(
        "Export Batch Summary (CSV)",
        data=dataframe_to_csv_bytes(bulk_export_table(bulk_result)),
        file_name=f"batch_summary_{method_slug(bulk_result['method'])}.csv",
        mime="text/csv",
        key="bulk_export_csv",
    )

    render_pdf_download(
        "Export Batch Report (PDF)",
        lambda: build_bulk_report_pdf(bulk_result),
        file_name=f"batch_report_{method_slug(bulk_result['method'])}.pdf",
        key="bulk_export_pdf",
    )

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
        key="bulk_download_zip",
    )


def main() -> None:
    st.markdown(
        dedent(
            """
            <style>
            div[data-testid="stMetric"] {
                background: #f8fafc;
                border: 1px solid #e5e7eb;
                padding: 0.9rem;
                border-radius: 14px;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }

            .fp-hero {
                background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
                border: 1px solid #e2e8f0;
                border-radius: 22px;
                padding: 1.4rem 1.5rem;
                margin-bottom: 1.4rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1.2rem;
            }

            .fp-hero-kicker {
                color: #64748b;
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.3rem;
            }

            .fp-hero h1 {
                color: #0f172a;
                font-size: 2.1rem;
                line-height: 1.1;
                margin: 0;
                font-weight: 850;
            }

            .fp-hero p {
                margin: 0.65rem 0 0 0;
                color: #475569;
                font-size: 1rem;
                max-width: 760px;
            }

            .fp-hero-badge {
                background: #ffffff;
                border: 1px solid #dbe4f0;
                border-radius: 18px;
                padding: 0.9rem 1rem;
                min-width: 230px;
                box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
            }

            .fp-hero-badge span {
                display: block;
                color: #64748b;
                font-size: 0.76rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            .fp-hero-badge strong {
                color: #0f172a;
                font-size: 0.98rem;
                line-height: 1.2;
            }

            .fp-visual-section {
                margin-top: 0.8rem;
                margin-bottom: 1.6rem;
            }

            .fp-section-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-end;
                gap: 1rem;
                margin-bottom: 1rem;
            }

            .fp-section-kicker {
                font-size: 0.78rem;
                font-weight: 800;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #64748b;
                margin-bottom: 0.15rem;
            }

            .fp-section-title {
                font-size: 1.45rem;
                font-weight: 850;
                color: #0f172a;
            }

            .fp-section-note {
                font-size: 0.85rem;
                color: #475569;
                background: #f1f5f9;
                border: 1px solid #e2e8f0;
                padding: 0.45rem 0.75rem;
                border-radius: 999px;
                white-space: nowrap;
            }

            .fp-image-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1rem;
            }

            .fp-image-card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 18px;
                padding: 0.9rem;
                box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
            }

            .fp-image-card-header {
                display: flex;
                align-items: center;
                gap: 0.7rem;
                margin-bottom: 0.8rem;
            }

            .fp-step-badge {
                width: 2rem;
                height: 2rem;
                border-radius: 999px;
                background: #0f172a;
                color: #ffffff;
                font-size: 0.8rem;
                font-weight: 850;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            }

            .fp-card-title {
                font-size: 0.95rem;
                font-weight: 850;
                color: #0f172a;
                line-height: 1.2;
            }

            .fp-card-subtitle {
                font-size: 0.78rem;
                color: #64748b;
                margin-top: 0.15rem;
                line-height: 1.2;
            }

            .fp-image-frame {
                background:
                    linear-gradient(45deg, #f8fafc 25%, transparent 25%),
                    linear-gradient(-45deg, #f8fafc 25%, transparent 25%),
                    linear-gradient(45deg, transparent 75%, #f8fafc 75%),
                    linear-gradient(-45deg, transparent 75%, #f8fafc 75%);
                background-size: 18px 18px;
                background-position: 0 0, 0 9px, 9px -9px, -9px 0px;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 0.6rem;
                min-height: 230px;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }

            .fp-image-frame img {
                width: 100%;
                max-height: 260px;
                object-fit: contain;
                border-radius: 10px;
                image-rendering: auto;
            }

            @media (max-width: 900px) {
                .fp-hero {
                    flex-direction: column;
                    align-items: stretch;
                }

                .fp-hero h1 {
                    font-size: 1.7rem;
                }

                .fp-hero-badge {
                    min-width: 0;
                }

                .fp-image-grid {
                    grid-template-columns: 1fr;
                }

                .fp-section-header {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .fp-section-note {
                    white-space: normal;
                }
            }
            </style>
            """
        ).strip(),
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
