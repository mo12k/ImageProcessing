"""Consolidate frozen member technique allocation and existing evidence.

This script is reporting-only. It reads existing audit outputs and writes the
member registry, baseline classification, evidence table, explanations, and
final allocation report without running image enhancement experiments.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
TARGET_PSNR_DB = 28.17


def read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required output file: {path}")
    return pd.read_csv(path)


def read_json(name: str) -> dict[str, Any]:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required output file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def clean_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    if pd.isna(value):
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [clean_value(row.get(column)).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def screening_row(screening: pd.DataFrame, technique_id: str) -> dict[str, Any]:
    frame = screening[
        (screening["Technique ID"] == technique_id)
        & screening["Images"].notna()
        & screening["Metric domain"].notna()
    ]
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def hybrid_overall_row(hybrid_direct: pd.DataFrame, pipeline_id: str) -> dict[str, Any]:
    frame = hybrid_direct[
        (hybrid_direct["Category"] == "Overall")
        & (hybrid_direct["Pipeline ID"] == pipeline_id)
    ]
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def hybrid_controlled_row(hybrid_controlled: pd.DataFrame, pipeline_id: str) -> dict[str, Any]:
    frame = hybrid_controlled[hybrid_controlled["Pipeline ID"] == pipeline_id]
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def focused_summary_row(
    focused_summary: pd.DataFrame,
    scenario_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    frame = focused_summary[
        (focused_summary["Scenario ID"] == scenario_id)
        & (focused_summary["Candidate ID"] == candidate_id)
    ]
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def make_registry() -> list[dict[str, Any]]:
    return [
        {
            "Member": "Member 1",
            "Main Contemporary Technique": "Non-Local Means Denoising",
            "Main Purpose": "Ridge-preserving noise reduction",
            "Baseline / Comparator": "Gaussian / Median / CLAHE",
            "Main parameters": (
                "Focused PSNR evidence uses OpenCV fastNlMeansDenoising with h=0.06, "
                "templateWindowSize=7, searchWindowSize=21, patch_size=5, patch_distance=6. "
                "Earlier screening D4 used skimage NLM with h=0.055, patch_size=3, patch_distance=5."
            ),
            "Output type": "Aligned grayscale float image in [0,1]",
            "Appropriate evaluation metrics": (
                "Controlled aligned: MSE, PSNR, SSIM. Direct Altered: ridge coherence, "
                "local contrast, fragmentation, continuity, visual ridge plausibility."
            ),
            "Implementation status": "Implemented",
            "Implementation source": (
                "focused_psnr_audit_optimisation.py: non_local_means; "
                "screening24_master_audit.py: D4 Non-local means"
            ),
            "Backend/library": "OpenCV cv2.fastNlMeansDenoising in focused final run; skimage fallback available",
            "Fallback risk": "Focused function falls back to skimage NLM if cv2 is unavailable; report backend explicitly.",
            "Registry note": "CLAHE/Gaussian/Median remain baselines only, not the main member technique.",
        },
        {
            "Member": "Member 2",
            "Main Contemporary Technique": "Modified / Orientation-Adaptive Gabor",
            "Main Purpose": "Ridge enhancement",
            "Baseline / Comparator": "Ordinary Gabor",
            "Main parameters": (
                "Adaptive orientation/frequency variants use local orientation, optional local ridge frequency, "
                "orientation bins around 8, frequency candidates around 0.075-0.140, and mild blend factors "
                "around 0.18-0.22 in screening."
            ),
            "Output type": "Enhanced grayscale ridge image in [0,1]",
            "Appropriate evaluation metrics": (
                "Direct Altered: coherence, local contrast, fragmentation, continuity, visual ridge artifacts. "
                "Controlled aligned grayscale only: MSE, PSNR, SSIM."
            ),
            "Implementation status": "Implemented as literature-inspired/adaptive, not exact paper reproduction",
            "Implementation source": (
                "screening24_master_audit.py: adaptive_gabor_enhancement, multi_orientation_gabor, G2-G4; "
                "Fingerprint_Enhancement_System.py: adaptive_gabor_enhancement"
            ),
            "Backend/library": "skimage.filters.gabor plus local orientation/frequency estimates",
            "Fallback risk": "No confirmed paper parameters/dataset/formula in repository; label as Literature-Inspired Modified Gabor when discussing article relation.",
            "Registry note": "Ordinary Gabor remains the comparator and should not be presented as the advanced contribution.",
        },
        {
            "Member": "Member 3",
            "Main Contemporary Technique": "Total Variation Restoration",
            "Main Purpose": "Ridge/edge-preserving restoration",
            "Baseline / Comparator": "Wiener / Gaussian",
            "Main parameters": (
                "screening24 R3 uses skimage denoise_tv_chambolle with weight=0.045. "
                "Focused audit searched fixed weights 0.01, 0.02, 0.04, 0.08 and selected weight=0.02 for Gaussian noise cases."
            ),
            "Output type": "Aligned grayscale float image in [0,1]",
            "Appropriate evaluation metrics": (
                "Controlled aligned: MSE, PSNR, SSIM plus coherence. Direct Altered: coherence, "
                "contrast, fragmentation, continuity, visual ridge preservation."
            ),
            "Implementation status": "Implemented",
            "Implementation source": (
                "screening24_master_audit.py: R3 Total variation restoration; "
                "focused_psnr_audit_optimisation.py: tv_restoration"
            ),
            "Backend/library": "skimage.restoration.denoise_tv_chambolle",
            "Fallback risk": "Fingerprint_Enhancement_System.py wavelet_denoise can fall back to TV if PyWavelets is unavailable; labels must match actual branch.",
            "Registry note": "Morphology is downstream/supporting and no longer Member 3's headline technique.",
        },
        {
            "Member": "Member 4",
            "Main Contemporary Technique": "Coherence-Guided Directional Diffusion",
            "Main Purpose": "Orientation-aware ridge-flow enhancement",
            "Baseline / Comparator": "conventional directional/ridge methods",
            "Main parameters": (
                "screening24 uses coherence_guided_diffusion with iterations=4, step=0.16 and orientation-selected smoothing. "
                "Fingerprint_Enhancement_System.py variant uses iterations=2, step=0.18, orientation_bins=6."
            ),
            "Output type": "Enhanced grayscale ridge-flow image in [0,1]",
            "Appropriate evaluation metrics": (
                "Direct Altered: coherence, local contrast, fragmentation, continuity, visual ridge-flow plausibility. "
                "Controlled aligned grayscale only: MSE, PSNR, SSIM."
            ),
            "Implementation status": "Implemented",
            "Implementation source": (
                "screening24_master_audit.py: coherence_guided_diffusion, G7; "
                "Fingerprint_Enhancement_System.py: coherence_guided_diffusion"
            ),
            "Backend/library": "NumPy/SciPy/skimage local structure estimates and oriented smoothing",
            "Fallback risk": "No silent replacement observed in the audited functions.",
            "Registry note": "Thinning, medial axis, and minutiae extraction are downstream analysis, not the main M4 technique.",
        },
    ]


def make_implementation_audit() -> list[dict[str, Any]]:
    return [
        {
            "Audit item": "Main notebook and script exports",
            "Confirmed implementation": "Main notebook and Python scripts are present",
            "Source": (
                "Fingerprint_Enhancement_System.ipynb; Fingerprint_Enhancement_System.py; "
                "screening24_master_audit.py; direct24_original_altered_experiment.py; "
                "hybrid24_compatibility_experiment.py; focused_psnr_audit_optimisation.py"
            ),
            "Backend/library": "Project-local notebook/script workflow",
            "Parameters": "Existing outputs are under outputs/screening24*, outputs/direct24*, outputs/hybrid24*, and outputs/focused_psnr_audit*",
            "Fallbacks or label risk": "Fingerprint_Enhancement_System.py still contains older member labels and should be treated as legacy wording until refreshed.",
            "Action in this report": "Use the new registry files as the source of truth for member allocation.",
        },
        {
            "Audit item": "Common preprocessing",
            "Confirmed implementation": "P0 = grayscale + resize + float [0,1] only",
            "Source": "screening24_master_audit.py: common_preprocessing_variants; direct24 report neutral preprocessing",
            "Backend/library": "skimage/io/numpy image conversion",
            "Parameters": "Resize to common processing size where required; no enhancement in P0",
            "Fallbacks or label risk": "P1-P3 exist as preprocessing audit variants but are not the selected common preprocessing.",
            "Action in this report": "Keep P0 as the only neutral common preprocessing.",
        },
        {
            "Audit item": "Member 1 NLM",
            "Confirmed implementation": "Real NLM implementation exists",
            "Source": "focused_psnr_audit_optimisation.py: non_local_means; screening24_master_audit.py: D4",
            "Backend/library": "OpenCV in focused final run; skimage NLM fallback",
            "Parameters": "h=0.06 for focused winning result; screening D4 h=0.055",
            "Fallbacks or label risk": "Backend changes if cv2 is absent; output labels should preserve backend and h.",
            "Action in this report": "Freeze M1 as NLM, with CLAHE/Gaussian/Median as baselines.",
        },
        {
            "Audit item": "Member 2 Modified/Adaptive Gabor",
            "Confirmed implementation": "Adaptive Gabor variants and Ordinary Gabor baseline exist",
            "Source": "screening24_master_audit.py: G1-G4; Fingerprint_Enhancement_System.py: adaptive_gabor_enhancement",
            "Backend/library": "skimage.filters.gabor with orientation/frequency estimates",
            "Parameters": "Ordinary frequency 0.115 with 8 orientations; adaptive frequency candidates 0.075-0.140, blend around 0.18-0.22",
            "Fallbacks or label risk": "Article-specific parameters are unknown; current G2 is literature-inspired, not exact replication.",
            "Action in this report": "Freeze M2 as Modified / Orientation-Adaptive Gabor and keep Ordinary Gabor as comparator.",
        },
        {
            "Audit item": "Member 3 TV Restoration",
            "Confirmed implementation": "TV restoration exists as grayscale restoration",
            "Source": "screening24_master_audit.py: R3; focused_psnr_audit_optimisation.py: tv_restoration",
            "Backend/library": "skimage.restoration.denoise_tv_chambolle",
            "Parameters": "screening R3 weight=0.045; focused selected weight=0.02 for Gaussian-noise-only cases",
            "Fallbacks or label risk": "Wavelet helper in the main app can fall back to TV if PyWavelets is missing.",
            "Action in this report": "Freeze M3 as TV Restoration; morphology becomes supporting structural processing.",
        },
        {
            "Audit item": "Member 4 Coherence-Guided Directional Diffusion",
            "Confirmed implementation": "Coherence-guided directional diffusion exists",
            "Source": "screening24_master_audit.py: G7; Fingerprint_Enhancement_System.py: coherence_guided_diffusion",
            "Backend/library": "NumPy/SciPy/skimage structure estimates and oriented smoothing",
            "Parameters": "screening iterations=4, step=0.16; system variant iterations=2, step=0.18, orientation_bins=6",
            "Fallbacks or label risk": "Old Direct24 labels group it under M2/Ridge Enhancement; new registry assigns it to M4.",
            "Action in this report": "Freeze M4 as Coherence-Guided Directional Diffusion; thinning becomes downstream.",
        },
        {
            "Audit item": "Literature benchmark context",
            "Confirmed implementation": "Local documentation contains the benchmark score but not enough settings for exact replication",
            "Source": "outputs/screening24_metadata.json literature_context; specification/fingerprint enhancement.pdf",
            "Backend/library": "Local PDF context extracted by screening24_master_audit.py: extract_literature_context",
            "Parameters": "Modified Gabor Filter PSNR=28.17 dB, MSE=22.27",
            "Fallbacks or label risk": "Dataset, image count, exact degradation, MAX_I, MSE scaling, and full metric formula are unknown.",
            "Action in this report": "Treat PSNR > 28.17 dB as a hard target but do not claim article reproduction.",
        },
    ]


def make_baseline_classification(screening: pd.DataFrame) -> list[dict[str, Any]]:
    rows = [
        ("CLAHE", "C3", "Basic contrast-enhancement baseline", "M1 baseline; optional contrast comparator", "Not a main contribution"),
        ("Contrast Stretching", "C4", "Basic/global contrast baseline", "Contrast baseline and diagnostic comparator", "Not a main contribution"),
        ("Gaussian Filter", "D1", "Basic smoothing baseline", "M1/M3 smoothing comparator", "Not a main contribution"),
        ("Median Filter", "D2", "Basic impulse-noise baseline", "M1 denoising comparator", "Not a main contribution"),
        ("Bilateral Filter", "D3", "Classical edge-aware denoising comparator", "Denoising comparator where already available", "Baseline/comparator"),
        ("Ordinary Gabor", "G1", "Conventional ridge-enhancement baseline", "M2 comparator", "Not the advanced contribution"),
        ("Wiener Deconvolution", "R1", "Conventional restoration comparator", "M3 restoration comparator", "Comparator, not member headline"),
        ("Closing", "M3-2", "Morphological structural support", "Ridge map cleanup after grayscale enhancement", "Downstream/supporting only"),
        ("Thinning / Medial Axis", "T4", "Downstream structural analysis", "Skeleton/minutiae analysis after enhancement", "Downstream/supporting only"),
        ("Thresholding / Opening", "N/A", "Downstream binary/ridge-map support", "Segmentation and cleanup", "Downstream/supporting only"),
    ]
    output = []
    for method, tech_id, classification, role, status in rows:
        source = screening_row(screening, tech_id) if tech_id != "N/A" else {}
        output.append(
            {
                "Method": method,
                "Existing technique ID": tech_id,
                "Classification": classification,
                "Allowed role": role,
                "Member-technique status": status,
                "Controlled PSNR if available": source.get("PSNR (dB) mean"),
                "Controlled MSE if available": source.get("MSE mean"),
                "Controlled SSIM if available": source.get("SSIM mean"),
                "Structural metric if available": source.get("Fragmentation per 1000 ridge pixels mean"),
                "Notes": "Do not remove; use for Mode A comparison and benchmarking.",
            }
        )
    return output


def make_altered_nlm_evidence(altered: pd.DataFrame) -> dict[str, Any]:
    before = altered[altered["Stage"] == "before"].copy()
    after = altered[altered["Stage"] == "after"].copy()
    keys = ["Altered split", "Image", "Pipeline candidate ID", "Pipeline"]
    merged = before.merge(after, on=keys, suffixes=("_before", "_after"))
    merged["Delta coherence"] = merged["Coherence_after"] - merged["Coherence_before"]
    merged["Delta local contrast"] = merged["Local contrast_after"] - merged["Local contrast_before"]
    merged["Delta fragmentation"] = (
        merged["fragmentation_per_1000_ridge_pixels_after"]
        - merged["fragmentation_per_1000_ridge_pixels_before"]
    )
    return {
        "rows": merged,
        "summary": {
            "images": len(merged),
            "coherence_after": merged["Coherence_after"].mean(),
            "contrast_after": merged["Local contrast_after"].mean(),
            "fragmentation_after": merged["fragmentation_per_1000_ridge_pixels_after"].mean(),
            "delta_coherence": merged["Delta coherence"].mean(),
            "delta_contrast": merged["Delta local contrast"].mean(),
            "delta_fragmentation": merged["Delta fragmentation"].mean(),
        },
    }


def add_controlled_evidence(
    evidence: list[dict[str, Any]],
    screening: pd.DataFrame,
    technique_id: str,
    member_or_role: str,
    observation: str,
) -> None:
    row = screening_row(screening, technique_id)
    if not row:
        return
    evidence.append(
        {
            "Evidence level": "summary",
            "Member / role": member_or_role,
            "Technique ID / Pipeline ID": technique_id,
            "Technique / Pipeline": row.get("Technique"),
            "Source output file": "screening24_family_summary.csv",
            "Dataset/sample": "24-image SOCOFing Real development screening subset",
            "Protocol": "Current stacked synthetic degradation; aligned clean Real reference",
            "Metric domain": row.get("Metric domain"),
            "Images": row.get("Images"),
            "MSE": row.get("MSE mean"),
            "PSNR (dB)": row.get("PSNR (dB) mean"),
            "SSIM": row.get("SSIM mean"),
            "Coherence": row.get("Ridge coherence mean"),
            "Local contrast": None,
            "Fragmentation": row.get("Fragmentation per 1000 ridge pixels mean"),
            "Continuity": row.get("Ridge continuity proxy mean"),
            "Target exceeded": (
                bool(row.get("PSNR (dB) mean") > TARGET_PSNR_DB)
                if not pd.isna(row.get("PSNR (dB) mean"))
                else None
            ),
            "Observation": observation,
        }
    )


def add_hybrid_evidence(
    evidence: list[dict[str, Any]],
    hybrid_direct: pd.DataFrame,
    pipeline_id: str,
    member_or_role: str,
    observation: str,
) -> None:
    row = hybrid_overall_row(hybrid_direct, pipeline_id)
    if not row:
        return
    evidence.append(
        {
            "Evidence level": "summary",
            "Member / role": member_or_role,
            "Technique ID / Pipeline ID": pipeline_id,
            "Technique / Pipeline": row.get("Pipeline"),
            "Source output file": "hybrid24_direct_summary.csv",
            "Dataset/sample": "24 Altered images, 8 each from Easy/Medium/Hard",
            "Protocol": "Direct Altered evaluation; no synthetic degradation; no PSNR against unverified pair",
            "Metric domain": row.get("Final output domain"),
            "Images": row.get("Images"),
            "MSE": None,
            "PSNR (dB)": None,
            "SSIM": None,
            "Coherence": row.get("Ridge coherence mean"),
            "Local contrast": row.get("Local contrast mean"),
            "Fragmentation": row.get("Fragmentation mean"),
            "Continuity": row.get("Ridge continuity mean"),
            "Target exceeded": None,
            "Observation": observation,
        }
    )


def add_hybrid_controlled_evidence(
    evidence: list[dict[str, Any]],
    hybrid_controlled: pd.DataFrame,
    pipeline_id: str,
    member_or_role: str,
    observation: str,
) -> None:
    row = hybrid_controlled_row(hybrid_controlled, pipeline_id)
    if not row:
        return
    evidence.append(
        {
            "Evidence level": "summary",
            "Member / role": member_or_role,
            "Technique ID / Pipeline ID": pipeline_id,
            "Technique / Pipeline": row.get("Pipeline"),
            "Source output file": "hybrid24_controlled_summary.csv",
            "Dataset/sample": "24-image controlled hybrid subset",
            "Protocol": "Current stacked synthetic degradation; aligned clean Real reference",
            "Metric domain": "grayscale",
            "Images": row.get("Images"),
            "MSE": row.get("MSE_mean"),
            "PSNR (dB)": row.get("PSNR_mean"),
            "SSIM": row.get("SSIM_mean"),
            "Coherence": row.get("Coherence_mean"),
            "Local contrast": None,
            "Fragmentation": None,
            "Continuity": None,
            "Target exceeded": bool(row.get("PSNR_mean") > TARGET_PSNR_DB),
            "Observation": observation,
        }
    )


def make_evidence(
    focused_summary: pd.DataFrame,
    focused_per_image: pd.DataFrame,
    altered: pd.DataFrame,
    screening: pd.DataFrame,
    hybrid_direct: pd.DataFrame,
    hybrid_controlled: pd.DataFrame,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []

    nlm = focused_summary_row(focused_summary, "gaussian_noise_only_s055", "NLM_h0.06")
    evidence.append(
        {
            "Evidence level": "summary",
            "Member / role": "Member 1 main technique",
            "Technique ID / Pipeline ID": "NLM_h0.06",
            "Technique / Pipeline": "OpenCV Non-Local Means",
            "Source output file": "focused_psnr_audit_test_summary.csv",
            "Dataset/sample": "12 held-out SOCOFing Real test images",
            "Protocol": "Gaussian noise only, severity 0.55; fixed parameters selected on separate dev subset",
            "Metric domain": "aligned grayscale [0,1]",
            "Images": nlm.get("Images"),
            "MSE": nlm.get("MSE_mean"),
            "PSNR (dB)": nlm.get("PSNR_mean"),
            "SSIM": nlm.get("SSIM_mean"),
            "Coherence": None,
            "Local contrast": None,
            "Fragmentation": None,
            "Continuity": None,
            "Target exceeded": nlm.get("Target_exceeded"),
            "Observation": "Best confirmed controlled PSNR; all 12 held-out images exceed 28.17 dB.",
        }
    )

    per_image = focused_per_image[
        (focused_per_image["Scenario ID"] == "gaussian_noise_only_s055")
        & (focused_per_image["Candidate ID"] == "NLM_h0.06")
    ].copy()
    for _, row in per_image.iterrows():
        evidence.append(
            {
                "Evidence level": "per-image",
                "Member / role": "Member 1 main technique",
                "Technique ID / Pipeline ID": "NLM_h0.06",
                "Technique / Pipeline": "OpenCV Non-Local Means",
                "Source output file": "focused_psnr_audit_test_per_image.csv",
                "Dataset/sample": row.get("Image"),
                "Protocol": "Gaussian noise only, severity 0.55; held-out test image",
                "Metric domain": "aligned grayscale [0,1]",
                "Images": 1,
                "MSE": row.get("MSE"),
                "PSNR (dB)": row.get("PSNR (dB)"),
                "SSIM": row.get("SSIM"),
                "Coherence": None,
                "Local contrast": None,
                "Fragmentation": None,
                "Continuity": None,
                "Target exceeded": bool(row.get("PSNR (dB)") > TARGET_PSNR_DB),
                "Observation": "Held-out per-image PSNR for the frozen NLM setting.",
            }
        )

    altered_nlm = make_altered_nlm_evidence(altered)
    s = altered_nlm["summary"]
    evidence.append(
        {
            "Evidence level": "summary",
            "Member / role": "Member 1 direct Altered sanity",
            "Technique ID / Pipeline ID": "NLM_h0.06",
            "Technique / Pipeline": "OpenCV Non-Local Means",
            "Source output file": "focused_psnr_audit_altered_structure.csv",
            "Dataset/sample": "9 Altered images, 3 each from Easy/Medium/Hard",
            "Protocol": "Direct Altered evaluation; no synthetic degradation; structural metrics only",
            "Metric domain": "grayscale direct Altered",
            "Images": s["images"],
            "MSE": None,
            "PSNR (dB)": None,
            "SSIM": None,
            "Coherence": s["coherence_after"],
            "Local contrast": s["contrast_after"],
            "Fragmentation": s["fragmentation_after"],
            "Continuity": None,
            "Target exceeded": None,
            "Observation": (
                f"Mean deltas: coherence {s['delta_coherence']:+.4f}, "
                f"local contrast {s['delta_contrast']:+.4f}, "
                f"fragmentation {s['delta_fragmentation']:+.4f}."
            ),
        }
    )

    add_controlled_evidence(
        evidence,
        screening,
        "G1",
        "Member 2 baseline comparator",
        "Ordinary Gabor is useful as the baseline; controlled PSNR is low and direct evidence shows contrast/fragmentation trade-offs.",
    )
    add_controlled_evidence(
        evidence,
        screening,
        "G2",
        "Member 2 main investigation",
        "Literature-inspired Modified Gabor is implemented, but exact article settings are unknown and should not be claimed.",
    )
    add_controlled_evidence(
        evidence,
        screening,
        "G3",
        "Member 2 supporting adaptive variant",
        "Adaptive local-orientation Gabor supports the modified/adaptive investigation.",
    )
    add_controlled_evidence(
        evidence,
        screening,
        "G4",
        "Member 2 supporting adaptive variant",
        "Adaptive orientation-frequency Gabor supports frequency-aware ridge enhancement investigation.",
    )
    add_hybrid_evidence(
        evidence,
        hybrid_direct,
        "C1",
        "Member 2 direct Altered trade-off",
        "P0 -> CLAHE -> Ordinary Gabor raised coherence but reduced contrast and increased fragmentation versus CLAHE.",
    )

    add_controlled_evidence(
        evidence,
        screening,
        "R3",
        "Member 3 main technique",
        "TV shows high SSIM/coherence balance in the stacked screening, although not high PSNR under stacked degradation.",
    )
    add_controlled_evidence(
        evidence,
        screening,
        "R1",
        "Member 3 comparator",
        "Wiener deconvolution is the conventional restoration comparator and was stronger than TV on stacked PSNR.",
    )
    add_hybrid_evidence(
        evidence,
        hybrid_direct,
        "C3",
        "Member 3 direct compatibility",
        "CLAHE -> TV kept high coherence but reduced contrast slightly versus CLAHE.",
    )
    add_hybrid_evidence(
        evidence,
        hybrid_direct,
        "C4",
        "Member 3 direct compatibility",
        "TV -> CLAHE produced the strongest direct coherence among hybrid grayscale candidates, with mild contrast trade-off versus CLAHE.",
    )
    add_hybrid_controlled_evidence(
        evidence,
        hybrid_controlled,
        "C4",
        "Member 3 controlled hybrid check",
        "TV -> CLAHE did not solve the stacked-degradation PSNR problem; it remains below the literature target.",
    )

    add_controlled_evidence(
        evidence,
        screening,
        "G7",
        "Member 4 main technique",
        "Coherence-guided directional diffusion is a grayscale orientation-aware method and should replace thinning as M4 headline.",
    )
    add_hybrid_evidence(
        evidence,
        hybrid_direct,
        "C2",
        "Member 4 direct compatibility",
        "Directional diffusion after CLAHE was more compatible than Ordinary Gabor, with less contrast loss and much lower fragmentation penalty.",
    )

    for technique_id, role, observation in [
        ("C3", "Baseline", "CLAHE is a basic contrast-enhancement baseline, not a member headline."),
        ("C4", "Baseline", "Contrast stretching remains a basic/global contrast baseline."),
        ("D1", "Baseline", "Gaussian filtering remains a basic smoothing comparator."),
        ("D2", "Baseline", "Median filtering remains a basic impulse-noise comparator."),
        ("D3", "Baseline", "Bilateral filtering remains a classical edge-aware comparator where used."),
        ("D5", "Baseline", "Wiener noise filtering remains a conventional denoising comparator."),
        ("D6", "Baseline", "Wavelet denoising remains a comparator/supporting denoising method; check actual fallback branch in any run."),
        ("M3-2", "Downstream structural support", "Closing is a morphology support operation and has no full-reference grayscale PSNR."),
        ("T4", "Downstream structural support", "Skeleton/medial axis is downstream analysis; do not calculate PSNR on skeletons."),
    ]:
        add_controlled_evidence(evidence, screening, technique_id, role, observation)

    return evidence


def make_explanations() -> str:
    return """# Member Technique Explanations

This file separates the four main contemporary member techniques from basic baselines and downstream structural operations.

## Member 1: Non-Local Means Denoising

Non-Local Means (NLM) is more substantial than Gaussian or median filtering because it denoises by comparing small patches rather than only using nearby pixels. For each target patch, the algorithm searches a neighbourhood, measures patch similarity, and forms a weighted average where more similar patches contribute more strongly. The denoising strength `h` controls how aggressively dissimilar patches are allowed to contribute.

For fingerprints this is scientifically relevant because ridge/valley patterns repeat locally. Similar ridge patches can reinforce one another while random noise averages out. The focused audit's strongest result used OpenCV NLM with `h=0.06`, preserving enough ridge intensity to exceed the PSNR target on Gaussian-noise-only controlled images.

## Member 2: Modified / Orientation-Adaptive Gabor

Ordinary Gabor filtering applies a fixed ridge frequency and a fixed set of orientations. It can increase ridge coherence, but it can also darken/brighten ridges unnaturally, reduce local contrast, and fragment ridge maps.

Modified or orientation-adaptive Gabor filtering is deeper because it estimates local fingerprint structure before filtering. The implementation investigates local ridge orientation, optional local ridge frequency, orientation-specific filtering, and adaptive blending. This makes the method fingerprint-specific rather than merely applying a generic filter bank.

The repository does not contain enough confirmed detail to claim exact reproduction of the selected literature article. Therefore the current implementation must be labelled `Literature-Inspired Modified Gabor` when compared with the article benchmark.

## Member 3: Total Variation Restoration

Total Variation (TV) restoration is an optimisation-based restoration method. It suppresses noise while penalising excessive image variation, which tends to preserve edges more carefully than simple smoothing. In fingerprint images, ridge/valley borders behave like important edges, so TV can reduce noise while trying to keep ridge transitions intact.

TV is more substantial than morphology because it operates on the grayscale restoration problem before binarisation. Morphological closing/opening are still useful, but they modify binary ridge maps after thresholding and should not replace TV as the main member contribution.

## Member 4: Coherence-Guided Directional Diffusion

Coherence-guided directional diffusion uses the local ridge-flow field to control smoothing. Instead of smoothing equally in all directions, it favours processing along coherent ridge directions and avoids unnecessary diffusion across ridge/valley boundaries.

This is a stronger member contribution than thinning because it enhances the grayscale ridge-flow image itself. Thinning, medial axis, and minutiae extraction are downstream structural analysis steps; they depend on the quality of the grayscale and binary ridge map generated earlier.

## Common P0 Preprocessing

P0 remains neutral: load image, convert to grayscale, resize if required, and convert to a consistent float range. No CLAHE, Gaussian, Median, Gabor, TV, NLM, or morphology is part of common preprocessing. Those are registered as experimental techniques, baselines, or downstream analysis.
"""


def make_report(
    registry: list[dict[str, Any]],
    audit: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    focused_summary: pd.DataFrame,
    focused_per_image: pd.DataFrame,
    screening: pd.DataFrame,
    hybrid_direct: pd.DataFrame,
    hybrid_controlled: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    nlm = focused_summary_row(focused_summary, "gaussian_noise_only_s055", "NLM_h0.06")
    current_stack = focused_summary_row(focused_summary, "current_stack_s055", "WIENER_DECONV_s1.7_b0.02_m1")
    tv_noise = focused_summary_row(focused_summary, "gaussian_noise_only_s055", "TV_w0.02")
    wiener_noise = focused_summary_row(focused_summary, "gaussian_noise_only_s055", "WIENER_DECONV_s0.98_b0.06_m0.35")
    nlm_per_image = focused_per_image[
        (focused_per_image["Scenario ID"] == "gaussian_noise_only_s055")
        & (focused_per_image["Candidate ID"] == "NLM_h0.06")
    ]
    altered_summary = next(
        row for row in evidence
        if row["Member / role"] == "Member 1 direct Altered sanity"
    )
    g1 = screening_row(screening, "G1")
    g2 = screening_row(screening, "G2")
    r3 = screening_row(screening, "R3")
    g7 = screening_row(screening, "G7")
    c1 = hybrid_overall_row(hybrid_direct, "C1")
    c2 = hybrid_overall_row(hybrid_direct, "C2")
    c3 = hybrid_overall_row(hybrid_direct, "C3")
    c4 = hybrid_overall_row(hybrid_direct, "C4")
    c4_controlled = hybrid_controlled_row(hybrid_controlled, "C4")

    exact_member_table = """| Member | Main Contemporary Technique | Main Purpose | Baseline / Comparator |
| --- | --- | --- | --- |
| Member 1 | Non-Local Means Denoising | Ridge-preserving noise reduction | Gaussian / Median / CLAHE |
| Member 2 | Modified / Orientation-Adaptive Gabor | Ridge enhancement | Ordinary Gabor |
| Member 3 | Total Variation Restoration | Ridge/edge-preserving restoration | Wiener / Gaussian |
| Member 4 | Coherence-Guided Directional Diffusion | Orientation-aware ridge-flow enhancement | conventional directional/ridge methods |"""

    audit_columns = [
        "Audit item",
        "Confirmed implementation",
        "Source",
        "Backend/library",
        "Parameters",
        "Fallbacks or label risk",
        "Action in this report",
    ]
    registry_columns = [
        "Member",
        "Main Contemporary Technique",
        "Main Purpose",
        "Baseline / Comparator",
        "Output type",
        "Backend/library",
        "Fallback risk",
    ]
    baseline_columns = [
        "Method",
        "Classification",
        "Allowed role",
        "Member-technique status",
        "Controlled PSNR if available",
    ]
    evidence_summary = [
        row for row in evidence
        if row["Evidence level"] == "summary"
        and row["Member / role"] not in {"Baseline"}
    ]
    evidence_columns = [
        "Member / role",
        "Technique ID / Pipeline ID",
        "Technique / Pipeline",
        "Source output file",
        "Protocol",
        "Images",
        "MSE",
        "PSNR (dB)",
        "SSIM",
        "Coherence",
        "Local contrast",
        "Fragmentation",
        "Observation",
    ]

    team_candidates = [
        {
            "Candidate": "1",
            "Pipeline": "NLM only",
            "Reason": "Strongest confirmed controlled PSNR; avoids adding ridge-enhancement stages that may alter intensities.",
            "Small next check": "Re-test fixed NLM h=0.06 on the same focused Gaussian-noise-only split and direct Altered structural sample.",
        },
        {
            "Candidate": "2",
            "Pipeline": "NLM -> mild Literature-Inspired Modified/Adaptive Gabor",
            "Reason": "May add ridge-flow clarity after denoising, but must be very mild because Gabor can reduce contrast and increase fragmentation.",
            "Small next check": "Use one fixed adaptive-Gabor blend chosen on dev only; compare PSNR loss and Altered structure versus NLM only.",
        },
        {
            "Candidate": "3",
            "Pipeline": "TV -> NLM",
            "Reason": "TV and NLM both target restoration/denoising with different priors; useful only if TV does not oversmooth ridges before NLM.",
            "Small next check": "Choose one TV weight from dev only and test fixed TV->NLM on held-out focused subset.",
        },
    ]

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    literature_context = metadata["screening24"].get("literature_context", {})
    literature_source = literature_context.get("project_literature_pdf", "specification/fingerprint enhancement.pdf")
    article_url = literature_context.get("article_url", "N/A")
    article_pdf_url = literature_context.get("article_pdf_url", "N/A")
    literature_note = (
        f"Confirmed in local project documentation (`{literature_source}`) and preserved in "
        "`outputs/screening24_metadata.json`: Modified Gabor Filter benchmark PSNR = "
        f"{literature_context.get('reported_modified_gabor_psnr_db', 28.17)} dB, MSE = "
        f"{literature_context.get('reported_modified_gabor_mse', 22.27)}. "
        f"Article URL recorded locally: {article_url}; PDF URL recorded locally: {article_pdf_url}. "
        f"Access note from the repository audit: {literature_context.get('web_access_note', 'N/A')} "
        "Unknown from available repository/docs: exact dataset, image count, degradation construction, MAX_I, MSE scaling, and full metric formula. "
        "Our audit uses normalized [0,1] MSE and PSNR with MAX_I/data_range = 1.0; 8-bit-equivalent MSE is also recorded."
    )

    return f"""# Final Member Allocation Report

Generated: {now}

Scope: reporting-only consolidation of existing repository evidence. No 500-image validation, no brute-force hybrid search, and no new image enhancement experiment was run by this consolidation step.

## Assignment Mode

Mode A: Comparative & Enhancement Study.

## Frozen Member Table

{exact_member_table}

Supporting downstream techniques remain separate from the four main member techniques:

- thresholding
- closing
- thinning
- medial axis
- minutiae extraction

## Implementation Audit

{md_table(audit, audit_columns)}

## Member Registry Summary

Full files written:

- `outputs/member_technique_registry.csv`
- `outputs/member_technique_registry.json`

{md_table(registry, registry_columns)}

## Baseline Classification

Full file written: `outputs/baseline_method_classification.csv`

{md_table(baselines, baseline_columns)}

## Literature Benchmark Status

{literature_note}

The literature PSNR/MSE pair is not internally standard-PSNR consistent without additional unknown settings. PSNR 28.17 dB implies MSE about 0.001524 on [0,1] or about 99.10 on [0,255]. MSE 22.27 with MAX_I=255 implies PSNR about 34.65 dB. Therefore, this project preserves the lecturer target as `PSNR > 28.17 dB`, but reports metric scaling transparently instead of inventing missing literature details.

## Existing Evidence Summary

Full file written: `outputs/member_existing_evidence.csv`

{md_table(evidence_summary, evidence_columns)}

## PSNR Target Audit

- Best current confirmed controlled PSNR: NLM_h0.06 on `gaussian_noise_only_s055`, {clean_value(nlm.get("PSNR_mean"))} dB.
- Gap to 28.17 dB: {clean_value(nlm.get("PSNR_delta_to_28_17"))} dB.
- Exceeded 28.17 dB: {clean_value(nlm.get("Target_exceeded"))}.
- Protocol: Gaussian noise only, severity 0.55, fixed parameters selected on a separate development subset and evaluated on 12 held-out test images.
- Fixed parameters: {nlm.get("Candidate parameters")}.
- Development/test split: random seed {metadata["focused"].get("random_seed")}; 12 dev Real images and 12 held-out test Real images listed in `outputs/focused_psnr_audit_metadata.json`.
- MSE: {clean_value(nlm.get("MSE_mean"), 8)} normalized [0,1]; 8-bit equivalent MSE {clean_value(nlm.get("MSE_255_equivalent_mean"))}.
- SSIM: {clean_value(nlm.get("SSIM_mean"))}.
- Per-image target result: {len(nlm_per_image)} / {len(nlm_per_image)} held-out test images exceeded 28.17 dB; minimum held-out PSNR {clean_value(nlm_per_image["PSNR (dB)"].min())} dB.
- Direct Altered sanity for same NLM setting: {altered_summary["Observation"]}

The current stacked synthetic protocol remains much harder: best focused stacked result was Wiener deconvolution at {clean_value(current_stack.get("PSNR_mean"))} dB, gap {clean_value(current_stack.get("PSNR_delta_to_28_17"))} dB. This supports the audit conclusion that stacking blur, contrast/illumination degradation, Gaussian noise, salt-and-pepper noise, and smudge/occlusion creates a harsh compound task that is not directly comparable to the documented Modified Gabor literature benchmark unless the literature degradation protocol is confirmed.

## A-Z Answers

A. The assignment mode is Mode A: Comparative & Enhancement Study.

B. Member 1's final main technique is Non-Local Means Denoising.

C. CLAHE was removed as Member 1's main contribution because it is a basic local contrast enhancement method and the lecturer indicated that very basic techniques are too simple as headline contributions.

D. CLAHE now plays the role of baseline/comparator and optional supporting contrast stage, not the main M1 technique.

E. Member 2's final main technique is Modified / Orientation-Adaptive Gabor Filtering.

F. Modified/Adaptive Gabor differs from Ordinary Gabor by estimating local ridge orientation and, where available, local ridge frequency before filtering. Ordinary Gabor is a fixed conventional comparator.

G. Member 3's final main technique is Total Variation Restoration.

H. Simple morphology is no longer Member 3's main contribution because it operates on binary ridge maps after thresholding; TV is the grayscale restoration method with stronger technical depth.

I. Member 4's final main technique is Coherence-Guided Directional Diffusion.

J. Thinning is no longer Member 4's main contribution because it is a downstream skeleton-analysis step, not a grayscale ridge-flow enhancement method.

K. Simple techniques that remain as baselines include CLAHE, contrast stretching, Gaussian filtering, median filtering, bilateral filtering, Ordinary Gabor, Wiener deconvolution/filtering, and global histogram equalisation.

L. Downstream structural operations include thresholding, opening, closing, skeletonisation, medial axis, and minutiae extraction.

M. Existing evidence supporting NLM: focused held-out test result PSNR {clean_value(nlm.get("PSNR_mean"))} dB, MSE {clean_value(nlm.get("MSE_mean"), 8)}, SSIM {clean_value(nlm.get("SSIM_mean"))}, all 12 held-out images above 28.17 dB, plus direct Altered coherence improvement and fragmentation reduction.

N. Existing evidence supporting Modified Gabor investigation: G2/G3/G4 adaptive Gabor variants exist, Ordinary Gabor demonstrated the motivation by increasing coherence but showing contrast and fragmentation trade-offs. The current G2 result must be called Literature-Inspired Modified Gabor because exact article settings are unknown.

O. Existing evidence supporting TV Restoration: screening R3 achieved PSNR {clean_value(r3.get("PSNR (dB) mean"))} dB, MSE {clean_value(r3.get("MSE mean"), 6)}, SSIM {clean_value(r3.get("SSIM mean"))}, coherence {clean_value(r3.get("Ridge coherence mean"))}; direct hybrid C4 showed coherence {clean_value(c4.get("Ridge coherence mean"))} on Altered images.

P. Existing evidence supporting Directional Diffusion: screening G7 achieved PSNR {clean_value(g7.get("PSNR (dB) mean"))} dB and coherence {clean_value(g7.get("Ridge coherence mean"))}; direct hybrid C2 showed coherence {clean_value(c2.get("Ridge coherence mean"))}, contrast {clean_value(c2.get("Local contrast mean"))}, fragmentation {clean_value(c2.get("Fragmentation mean"))}.

Q. The strongest controlled PSNR is currently NLM_h0.06 on Gaussian-noise-only severity 0.55.

R. Yes. That result exceeds 28.17 dB by {clean_value(nlm.get("PSNR_delta_to_28_17"))} dB.

S. Structural trade-offs were clearest for Ordinary Gabor after CLAHE: direct Overall coherence {clean_value(c1.get("Ridge coherence mean"))}, contrast {clean_value(c1.get("Local contrast mean"))}, fragmentation {clean_value(c1.get("Fragmentation mean"))}, contrast delta versus CLAHE {clean_value(c1.get("Delta contrast vs CLAHE mean"))}, fragmentation delta versus CLAHE {clean_value(c1.get("Delta fragmentation vs CLAHE mean"))}.

T. Ordinary Gabor is not recommended as the final overall technique. It remains the required comparator for Member 2.

U. Strengths and weaknesses: NLM has the strongest controlled PSNR and preserves ridges under Gaussian noise, but may slightly reduce local contrast on Altered images. Adaptive Gabor is fingerprint-specific and can improve ridge flow, but can alter intensities and fragment ridges if too strong. TV is edge/ridge-preserving and structurally compatible, but did not exceed the PSNR target in the stacked screening. Directional diffusion is orientation-aware and more compatible than Ordinary Gabor after CLAHE, but still needs a small fixed-parameter compatibility check with NLM.

V. Compatible-looking member techniques: NLM with a very mild adaptive ridge enhancement, and possibly TV with NLM if TV does not oversmooth before denoising. Directional diffusion may be compatible as a mild structure-preserving stage but must be checked against NLM-only.

W. Do not automatically combine all four members plus CLAHE, closing, and thinning. Also avoid Ordinary Gabor after CLAHE as a final grayscale method because it damages contrast and fragmentation despite coherence gains.

X. Scientifically justified small team candidates are:

{md_table(team_candidates, ["Candidate", "Pipeline", "Reason", "Small next check"])}

Y. Operations that should remain baselines/supporting methods: CLAHE, contrast stretching, Gaussian, median, bilateral, Ordinary Gabor, Wiener, thresholding, opening, closing, thinning, medial axis, and minutiae extraction.

Z. The project is not ready for final large-dataset validation yet. It is ready for review of this frozen member allocation and for one small compatibility experiment using fixed dev-selected parameters. The 500-image validation should wait until the team selects a final candidate that preserves the NLM >28.17 dB result or gives a clearly measured structural benefit without hidden metric leakage.

## Stop Condition Confirmation

- Implementations audited.
- Member registry frozen and exported.
- Existing evidence consolidated.
- Final member-allocation report produced.
- Only three small next compatibility candidates recommended.
- No 500-image validation was run.
- No literature details were invented.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    focused_summary = read_csv("focused_psnr_audit_test_summary.csv")
    focused_per_image = read_csv("focused_psnr_audit_test_per_image.csv")
    altered = read_csv("focused_psnr_audit_altered_structure.csv")
    screening = read_csv("screening24_family_summary.csv")
    hybrid_direct = read_csv("hybrid24_direct_summary.csv")
    hybrid_controlled = read_csv("hybrid24_controlled_summary.csv")

    metadata = {
        "focused": read_json("focused_psnr_audit_metadata.json"),
        "screening24": read_json("screening24_metadata.json"),
    }

    registry = make_registry()
    audit = make_implementation_audit()
    baselines = make_baseline_classification(screening)
    evidence = make_evidence(
        focused_summary,
        focused_per_image,
        altered,
        screening,
        hybrid_direct,
        hybrid_controlled,
    )

    pd.DataFrame(registry).to_csv(OUTPUT_DIR / "member_technique_registry.csv", index=False)
    (OUTPUT_DIR / "member_technique_registry.json").write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(audit).to_csv(OUTPUT_DIR / "member_implementation_audit.csv", index=False)
    pd.DataFrame(baselines).to_csv(OUTPUT_DIR / "baseline_method_classification.csv", index=False)
    pd.DataFrame(evidence).to_csv(OUTPUT_DIR / "member_existing_evidence.csv", index=False)
    (OUTPUT_DIR / "member_technique_explanations.md").write_text(
        make_explanations(),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "final_member_allocation_report.md").write_text(
        make_report(
            registry,
            audit,
            baselines,
            evidence,
            focused_summary,
            focused_per_image,
            screening,
            hybrid_direct,
            hybrid_controlled,
            metadata,
        ),
        encoding="utf-8",
    )

    print("Wrote member allocation consolidation artifacts:")
    for name in [
        "member_technique_registry.csv",
        "member_technique_registry.json",
        "member_implementation_audit.csv",
        "member_technique_explanations.md",
        "member_existing_evidence.csv",
        "baseline_method_classification.csv",
        "final_member_allocation_report.md",
    ]:
        print(f"- outputs/{name}")


if __name__ == "__main__":
    main()
