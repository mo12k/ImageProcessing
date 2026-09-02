from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import exposure
from skimage import measure
from skimage import morphology

import screening24_master_audit as base


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIRECT_SAMPLE_PER_CATEGORY = 8
RANDOM_SEED = 42

ALTERED_CATEGORIES = {
    "Easy": PROJECT_ROOT / "data" / "SOCOFing" / "Altered" / "Altered-Easy",
    "Medium": PROJECT_ROOT / "data" / "SOCOFing" / "Altered" / "Altered-Medium",
    "Hard": PROJECT_ROOT / "data" / "SOCOFing" / "Altered" / "Altered-Hard",
}
REAL_FOLDER = PROJECT_ROOT / "data" / "SOCOFing" / "Real"

DIRECT_FILES = {
    "per_image": OUTPUT_DIR / "direct24_per_image_metrics.csv",
    "per_severity": OUTPUT_DIR / "direct24_per_severity_summary.csv",
    "family": OUTPUT_DIR / "direct24_family_summary.csv",
    "hybrid_stage": OUTPUT_DIR / "direct24_hybrid_stage_metrics.csv",
    "runtime": OUTPUT_DIR / "direct24_runtime.json",
    "report": OUTPUT_DIR / "direct24_report.md",
    "metadata": OUTPUT_DIR / "direct24_metadata.json",
}


def existing_output_snapshot() -> dict[str, dict[str, object]]:
    protected = sorted([p for p in OUTPUT_DIR.iterdir() if p.is_file() and not p.name.startswith("direct24_")], key=lambda p: p.name)
    return base.snapshot_files(protected)


def direct_image_files(folder: Path) -> list[Path]:
    return base.image_files(folder)


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return list(paths)
    indexes = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[int(i)] for i in indexes]


def real_pair_candidate(altered_path: Path) -> tuple[str, bool]:
    stem = altered_path.stem
    for suffix in ["_CR", "_Obl", "_Zcut"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    candidate = REAL_FOLDER / f"{stem}{altered_path.suffix}"
    return str(candidate.relative_to(PROJECT_ROOT)), candidate.exists()


def load_p0(path: Path) -> np.ndarray:
    return base.load_grayscale_raw(path)


def ridge_binary_from_gray(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return base.ridge_base_binary(image)


def connected_component_metrics(binary: np.ndarray) -> dict[str, float]:
    binary = np.asarray(binary, dtype=bool)
    labels = measure.label(binary, connectivity=2)
    props = measure.regionprops(labels)
    areas = np.asarray([p.area for p in props], dtype=np.float32)
    ridge_pixels = int(np.sum(binary))
    skeleton = morphology.skeletonize(binary)
    endpoints, bifurcations = base.crossing_number_points(skeleton)
    return {
        "Ridge pixels": float(ridge_pixels),
        "Connected components": float(len(areas)),
        "Small connected components <20px": float(np.sum(areas < 20)) if areas.size else 0.0,
        "Largest component fraction": float(np.max(areas) / max(ridge_pixels, 1)) if areas.size else 0.0,
        "Fragmentation per 1000 ridge pixels": float(len(areas) / max(ridge_pixels / 1000.0, 1e-8)),
        "Skeleton pixels": float(np.sum(skeleton)),
        "Skeleton endpoints": float(len(endpoints)),
        "Skeleton bifurcations": float(len(bifurcations)),
        "Ridge continuity proxy": float(np.sum(skeleton) / max(len(endpoints), 1)),
    }


def direct_quality_metrics(image: np.ndarray, binary_override: np.ndarray | None = None, metric_image: np.ndarray | None = None) -> dict[str, float]:
    grayscale = base.metric_image(image)
    mask = base.fingerprint_mask(grayscale)
    local_std = base.local_std_image(grayscale)
    foreground = grayscale[mask] if np.any(mask) else grayscale.ravel()
    background = grayscale[~mask] if np.any(~mask) else grayscale.ravel()
    ridge_binary, _ = ridge_binary_from_gray(grayscale)
    if binary_override is not None:
        ridge_binary = np.asarray(binary_override, dtype=bool)
    ridge_area = float(np.sum(ridge_binary))
    mask_area = float(np.sum(mask))
    contrast_inside = float(np.mean(local_std[mask])) if np.any(mask) else float(np.mean(local_std))
    contrast_outside = float(np.mean(local_std[~mask])) if np.any(~mask) else 0.0
    intensity_gap = abs(float(np.mean(foreground)) - float(np.mean(background)))
    separation_proxy = float(intensity_gap + max(0.0, contrast_inside - contrast_outside))
    source_for_coherence = grayscale if metric_image is None else base.metric_image(metric_image)
    component_metrics = connected_component_metrics(ridge_binary)
    endpoints = component_metrics["Skeleton endpoints"]
    bifurcations = component_metrics["Skeleton bifurcations"]
    return {
        "Ridge coherence": base.orientation_coherence(source_for_coherence, mask),
        "Local contrast": contrast_inside,
        "Foreground/background separation proxy": separation_proxy,
        "Foreground usable-area ratio": float(np.mean(mask)),
        "Ridge coverage in foreground": ridge_area / max(mask_area, 1.0),
        "Detected ridge endings": endpoints,
        "Detected bifurcations": bifurcations,
        **component_metrics,
    }


def normalize_metric_deltas(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    baseline = frame[frame["Technique ID"] == "P0"].set_index(["Image", "Category"])
    for index, row in frame.iterrows():
        key = (row["Image"], row["Category"])
        if key not in baseline.index:
            continue
        base_row = baseline.loc[key]
        for column in [
            "Ridge coherence",
            "Local contrast",
            "Foreground/background separation proxy",
            "Ridge coverage in foreground",
            "Fragmentation per 1000 ridge pixels",
            "Small connected components <20px",
            "Ridge continuity proxy",
        ]:
            if column in frame.columns:
                frame.loc[index, f"Delta {column} vs P0"] = row[column] - base_row[column]
    return frame


def m1_techniques() -> list[dict[str, object]]:
    return [
        {
            "Family": "M1 / Contrast or General Enhancement",
            "Technique ID": "M1-C1",
            "Technique": "CLAHE",
            "Output domain": "grayscale",
            "Purpose": "local contrast enhancement baseline",
            "Function": lambda img: base.apply_clahe(img, clip_limit=0.025),
        },
        {
            "Family": "M1 / Contrast or General Enhancement",
            "Technique ID": "M1-C2",
            "Technique": "Contrast Stretching",
            "Output domain": "grayscale",
            "Purpose": "global contrast expansion",
            "Function": base.contrast_stretch,
        },
        {
            "Family": "M1 / Contrast or General Enhancement",
            "Technique ID": "M1-C3",
            "Technique": "Global Histogram Equalization",
            "Output domain": "grayscale",
            "Purpose": "global histogram redistribution baseline",
            "Function": lambda img: base.metric_image(exposure.equalize_hist(img)),
        },
    ]


def m2_techniques() -> list[dict[str, object]]:
    return [
        {
            "Family": "M2 / Ridge Enhancement",
            "Technique ID": "M2-R1",
            "Technique": "Ordinary Gabor",
            "Output domain": "grayscale",
            "Purpose": "ordinary fixed-frequency multi-orientation ridge filter",
            "Function": lambda img: base.multi_orientation_gabor(img, frequency=0.115, orientations=8, blend=0.30),
        },
        {
            "Family": "M2 / Ridge Enhancement",
            "Technique ID": "M2-R2",
            "Technique": "Literature-Inspired Modified Gabor",
            "Output domain": "grayscale",
            "Purpose": "local orientation and local frequency Gabor; not exact paper replication",
            "Function": lambda img: base.adaptive_gabor_enhancement(img, use_local_frequency=True, blend=0.18),
        },
        {
            "Family": "M2 / Ridge Enhancement",
            "Technique ID": "M2-R3",
            "Technique": "Coherence-Guided Directional Diffusion",
            "Output domain": "grayscale",
            "Purpose": "ridge-flow-preserving directional smoothing",
            "Function": base.coherence_guided_diffusion,
        },
        {
            "Family": "M2 / Ridge Enhancement",
            "Technique ID": "M2-R4",
            "Technique": "Directional Oriented Filter-Bank",
            "Output domain": "grayscale",
            "Purpose": "orientation-selected ridge-flow filtering",
            "Function": base.directional_filter_bank,
        },
    ]


def supporting_restoration_techniques() -> list[dict[str, object]]:
    return [
        {
            "Family": "Restoration / Denoising Supporting Candidate",
            "Technique ID": "SUP-D1",
            "Technique": "Wiener Deconvolution",
            "Output domain": "grayscale",
            "Purpose": "supporting restoration; assumes mild Gaussian PSF because direct Altered blur is not known",
            "Function": lambda img: base.frequency_wiener_deconvolution(img, psf_sigma=1.25, balance=0.08, blend=0.12),
        },
        {
            "Family": "Restoration / Denoising Supporting Candidate",
            "Technique ID": "SUP-D2",
            "Technique": "Wavelet Denoising",
            "Output domain": "grayscale",
            "Purpose": "supporting denoising without clean-reference use",
            "Function": base.wavelet_denoise,
        },
        {
            "Family": "Restoration / Denoising Supporting Candidate",
            "Technique ID": "SUP-D3",
            "Technique": "Total Variation Restoration",
            "Output domain": "grayscale",
            "Purpose": "supporting edge-preserving smoothing",
            "Function": lambda img: base.metric_image(base.restoration.denoise_tv_chambolle(img, weight=0.045, channel_axis=None)),
        },
    ]


def morphology_techniques(image: np.ndarray) -> list[dict[str, object]]:
    outputs = base.morphology_outputs(image)
    selected = ["M3-2", "M3-3", "M3-4", "M3-6", "M3-8"]
    return [
        {
            "Family": "M3 / Morphological Ridge Restoration",
            "Technique ID": technique_id,
            "Technique": name,
            "Output domain": "binary ridge map",
            "Purpose": "binary ridge-structure restoration",
            "Output": output,
        }
        for technique_id, (name, output) in outputs.items()
        if technique_id in selected
    ]


def thinning_techniques(image: np.ndarray) -> list[dict[str, object]]:
    restored = base.morphology_outputs(image)["M3-8"][1]
    return [
        {
            "Family": "M4 / Thinning",
            "Technique ID": "M4-T1",
            "Technique": "Morphological Skeleton",
            "Output domain": "skeleton",
            "Purpose": "skeletonised ridge analysis output",
            "Output": morphology.skeletonize(restored),
        },
        {
            "Family": "M4 / Thinning",
            "Technique ID": "M4-T2",
            "Technique": "Medial Axis",
            "Output domain": "skeleton",
            "Purpose": "medial-axis skeleton analysis output",
            "Output": morphology.medial_axis(restored),
        },
    ]


def select_direct_images() -> pd.DataFrame:
    rows = []
    for category, folder in ALTERED_CATEGORIES.items():
        selected = evenly_spaced(direct_image_files(folder), DIRECT_SAMPLE_PER_CATEGORY)
        for order, path in enumerate(selected):
            pair_path, pair_exists = real_pair_candidate(path)
            rows.append(
                {
                    "Category": category,
                    "Category order": order,
                    "Image": path.name,
                    "Path": str(path.relative_to(PROJECT_ROOT)),
                    "Real pair candidate path": pair_path,
                    "Real pair candidate exists": pair_exists,
                    "Pairing used for metrics": False,
                    "Selection rule": "sorted category file list, 8 evenly spaced indexes",
                }
            )
    return pd.DataFrame(rows)


def run_direct_single_techniques(selection_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, float]]:
    rows = []
    examples: dict[str, dict[str, np.ndarray]] = {}
    runtimes: dict[str, float] = {}
    techniques = m1_techniques() + m2_techniques() + supporting_restoration_techniques()
    for _, selected in selection_df.iterrows():
        category = selected["Category"]
        image_name = selected["Image"]
        image_path = PROJECT_ROOT / selected["Path"]
        p0 = load_p0(image_path)
        baseline_metrics = direct_quality_metrics(p0)
        rows.append(
            {
                **selected.to_dict(),
                "Family": "Input / Common Preprocessing Control",
                "Technique ID": "P0",
                "Technique": "Original Altered Input after P0",
                "Output domain": "grayscale",
                "Purpose": "input condition and neutral preprocessing control, not enhancement",
                "Is enhancement technique": False,
                "Runtime (s)": 0.0,
                **baseline_metrics,
            }
        )
        outputs_for_visual = {"Original Altered input": p0, "P0 neutral input": p0}

        for technique in techniques:
            start = perf_counter()
            output = technique["Function"](p0)
            runtime = perf_counter() - start
            runtimes.setdefault(str(technique["Technique"]), 0.0)
            runtimes[str(technique["Technique"])] += runtime
            metrics = direct_quality_metrics(output)
            rows.append(
                {
                    **selected.to_dict(),
                    "Family": technique["Family"],
                    "Technique ID": technique["Technique ID"],
                    "Technique": technique["Technique"],
                    "Output domain": technique["Output domain"],
                    "Purpose": technique["Purpose"],
                    "Is enhancement technique": True,
                    "Runtime (s)": runtime,
                    **metrics,
                }
            )
            if technique["Technique"] in [
                "CLAHE",
                "Contrast Stretching",
                "Ordinary Gabor",
                "Literature-Inspired Modified Gabor",
                "Coherence-Guided Directional Diffusion",
                "Wiener Deconvolution",
                "Wavelet Denoising",
                "Total Variation Restoration",
            ]:
                outputs_for_visual[technique["Technique"]] = output

        m3_start = perf_counter()
        m3_items = morphology_techniques(p0)
        m3_runtime = perf_counter() - m3_start
        for item in m3_items:
            metrics = direct_quality_metrics(p0, binary_override=item["Output"], metric_image=p0)
            rows.append(
                {
                    **selected.to_dict(),
                    "Family": item["Family"],
                    "Technique ID": item["Technique ID"],
                    "Technique": item["Technique"],
                    "Output domain": item["Output domain"],
                    "Purpose": item["Purpose"],
                    "Is enhancement technique": True,
                    "Runtime (s)": m3_runtime / max(len(m3_items), 1),
                    **metrics,
                }
            )
            if item["Technique"] == "Closing":
                outputs_for_visual["M3 Closing"] = item["Output"].astype(float)

        m4_start = perf_counter()
        m4_items = thinning_techniques(p0)
        m4_runtime = perf_counter() - m4_start
        for item in m4_items:
            metrics = direct_quality_metrics(p0, binary_override=item["Output"], metric_image=p0)
            rows.append(
                {
                    **selected.to_dict(),
                    "Family": item["Family"],
                    "Technique ID": item["Technique ID"],
                    "Technique": item["Technique"],
                    "Output domain": item["Output domain"],
                    "Purpose": item["Purpose"],
                    "Is enhancement technique": True,
                    "Runtime (s)": m4_runtime / max(len(m4_items), 1),
                    **metrics,
                }
            )
            if item["Technique"] in {"Morphological Skeleton", "Medial Axis"}:
                outputs_for_visual[item["Technique"]] = item["Output"].astype(float)

        if category not in examples:
            examples[category] = {
                "image_name": image_name,
                "outputs": outputs_for_visual,
            }
    return normalize_metric_deltas(pd.DataFrame(rows)), examples, runtimes


def summarise_by_category(metrics_df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "Images": ("Image", "nunique"),
        "Ridge coherence mean": ("Ridge coherence", "mean"),
        "Ridge coherence std": ("Ridge coherence", "std"),
        "Local contrast mean": ("Local contrast", "mean"),
        "Local contrast std": ("Local contrast", "std"),
        "Foreground/background separation proxy mean": ("Foreground/background separation proxy", "mean"),
        "Ridge coverage mean": ("Ridge coverage in foreground", "mean"),
        "Fragmentation mean": ("Fragmentation per 1000 ridge pixels", "mean"),
        "Small components mean": ("Small connected components <20px", "mean"),
        "Ridge continuity mean": ("Ridge continuity proxy", "mean"),
        "Detected endings mean": ("Detected ridge endings", "mean"),
        "Detected bifurcations mean": ("Detected bifurcations", "mean"),
        "Runtime mean": ("Runtime (s)", "mean"),
        "Delta coherence vs P0 mean": ("Delta Ridge coherence vs P0", "mean"),
        "Delta contrast vs P0 mean": ("Delta Local contrast vs P0", "mean"),
        "Delta fragmentation vs P0 mean": ("Delta Fragmentation per 1000 ridge pixels vs P0", "mean"),
        "Delta continuity vs P0 mean": ("Delta Ridge continuity proxy vs P0", "mean"),
    }
    by_category = (
        metrics_df.groupby(
            ["Category", "Family", "Technique ID", "Technique", "Output domain", "Purpose", "Is enhancement technique"],
            as_index=False,
        )
        .agg(**aggregations)
        .sort_values(["Category", "Family", "Technique ID"])
    )
    overall = (
        metrics_df.groupby(["Family", "Technique ID", "Technique", "Output domain", "Purpose", "Is enhancement technique"], as_index=False)
        .agg(**aggregations)
        .assign(Category="Overall")
    )
    return pd.concat([by_category, overall], ignore_index=True, sort=False)


def score_m1(frame: pd.DataFrame) -> pd.Series:
    # M1's purpose is useful contrast enhancement while preserving ridge structure.
    return (
        frame["Delta contrast vs P0 mean"].fillna(0.0) * 3.0
        + frame["Delta coherence vs P0 mean"].fillna(0.0) * 1.5
        + frame["Delta continuity vs P0 mean"].fillna(0.0) / 80.0
        - frame["Delta fragmentation vs P0 mean"].fillna(0.0).clip(lower=0.0) / 35.0
    )


def score_m2(frame: pd.DataFrame) -> pd.Series:
    # M2 prioritises coherent ridge flow and less fragmentation.
    return (
        frame["Delta coherence vs P0 mean"].fillna(0.0) * 4.0
        + frame["Delta continuity vs P0 mean"].fillna(0.0) / 70.0
        - frame["Delta fragmentation vs P0 mean"].fillna(0.0).clip(lower=0.0) / 25.0
    )


def score_m3(frame: pd.DataFrame) -> pd.Series:
    return -frame["Fragmentation mean"].fillna(1e6) + frame["Ridge continuity mean"].fillna(0.0) / 20.0


def score_m4(frame: pd.DataFrame) -> pd.Series:
    return -frame["Small components mean"].fillna(1e6) + frame["Ridge continuity mean"].fillna(0.0) / 20.0


def score_restoration(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["Delta coherence vs P0 mean"].fillna(0.0) * 2.5
        + frame["Delta continuity vs P0 mean"].fillna(0.0) / 80.0
        + frame["Delta contrast vs P0 mean"].fillna(0.0)
        - frame["Delta fragmentation vs P0 mean"].fillna(0.0).clip(lower=0.0) / 30.0
    )


def family_winner_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rules = [
        ("M1 / Contrast or General Enhancement", score_m1, "contrast gain with coherence/fragmentation guard"),
        ("M2 / Ridge Enhancement", score_m2, "ridge coherence and continuity with fragmentation guard"),
        ("Restoration / Denoising Supporting Candidate", score_restoration, "ridge preservation and denoising support"),
        ("M3 / Morphological Ridge Restoration", score_m3, "lowest fragmentation with continuity"),
        ("M4 / Thinning", score_m4, "fewest tiny skeleton components with continuity"),
    ]
    for family, scorer, metric_label in rules:
        for category in ["Easy", "Medium", "Hard", "Overall"]:
            frame = summary_df[
                (summary_df["Family"] == family)
                & (summary_df["Category"] == category)
                & (summary_df["Is enhancement technique"] == True)
            ].copy()
            if frame.empty:
                continue
            frame["Family score"] = scorer(frame)
            winner = frame.loc[frame["Family score"].idxmax()]
            rows.append(
                {
                    "Family": family,
                    "Category": category,
                    "Technique ID": winner["Technique ID"],
                    "Technique": winner["Technique"],
                    "Family score": float(winner["Family score"]),
                    "Main metric logic": metric_label,
                    "Ridge coherence": float(winner["Ridge coherence mean"]),
                    "Local contrast": float(winner["Local contrast mean"]),
                    "Fragmentation": float(winner["Fragmentation mean"]),
                    "Ridge continuity": float(winner["Ridge continuity mean"]),
                    "Main strength": describe_strength(winner),
                    "Main weakness": describe_weakness(winner),
                }
            )
    return pd.DataFrame(rows)


def describe_strength(row: pd.Series) -> str:
    strengths = [
        ("coherence", row.get("Delta coherence vs P0 mean", 0.0)),
        ("contrast", row.get("Delta contrast vs P0 mean", 0.0)),
        ("continuity", row.get("Delta continuity vs P0 mean", 0.0)),
    ]
    name, value = max(strengths, key=lambda item: item[1])
    if name == "coherence":
        return f"coherence change {value:+.4f}"
    if name == "contrast":
        return f"local contrast change {value:+.4f}"
    return f"continuity change {value:+.4f}"


def describe_weakness(row: pd.Series) -> str:
    frag = row.get("Delta fragmentation vs P0 mean", 0.0)
    coh = row.get("Delta coherence vs P0 mean", 0.0)
    contrast = row.get("Delta contrast vs P0 mean", 0.0)
    if frag > 0:
        return f"fragmentation worsens {frag:+.4f}"
    if coh < 0:
        return f"coherence decreases {coh:+.4f}"
    if contrast < 0:
        return f"contrast decreases {contrast:+.4f}"
    return "no major structural penalty in selected score"


def pick_overall_winner(family_df: pd.DataFrame, family: str) -> pd.Series:
    frame = family_df[(family_df["Family"] == family) & (family_df["Category"] == "Overall")]
    if frame.empty:
        return pd.Series(dtype=object)
    return frame.iloc[0]


def run_one_hybrid(selection_df: pd.DataFrame, family_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    m1 = pick_overall_winner(family_df, "M1 / Contrast or General Enhancement")
    m2 = pick_overall_winner(family_df, "M2 / Ridge Enhancement")
    m3 = pick_overall_winner(family_df, "M3 / Morphological Ridge Restoration")
    m4 = pick_overall_winner(family_df, "M4 / Thinning")
    m1_registry = {item["Technique"]: item["Function"] for item in m1_techniques()}
    m2_registry = {item["Technique"]: item["Function"] for item in m2_techniques()}
    hybrid_definition = {
        "Hybrid ID": "DIRECT24-H1",
        "M1": m1.get("Technique", ""),
        "M2": m2.get("Technique", ""),
        "M3": m3.get("Technique", ""),
        "M4": m4.get("Technique", ""),
        "M4 role": "downstream analysis/output branch, not part of greyscale enhancement",
    }
    stage_rows = []
    per_image_rows = []
    for _, selected in selection_df.iterrows():
        image_path = PROJECT_ROOT / selected["Path"]
        p0 = load_p0(image_path)
        stages: list[tuple[str, str, str, np.ndarray, str]] = [
            ("Original Altered", "input", "grayscale", p0, "Original Altered image after P0 neutral handling"),
        ]
        m1_start = perf_counter()
        m1_output = m1_registry[str(m1["Technique"])](p0)
        m1_runtime = perf_counter() - m1_start
        stages.append((f"M1 - {m1['Technique']}", "M1", "grayscale", m1_output, "selected from single-technique M1 evidence"))

        m2_start = perf_counter()
        m2_output = m2_registry[str(m2["Technique"])](m1_output)
        m2_runtime = perf_counter() - m2_start
        stages.append((f"M2 - {m2['Technique']}", "M2", "grayscale", m2_output, "selected from single-technique M2 evidence"))

        m3_outputs = base.morphology_outputs(m2_output)
        m3_id_by_name = {name: output for _, (name, output) in m3_outputs.items()}
        m3_output = m3_id_by_name.get(str(m3["Technique"]), m3_outputs["M3-8"][1])
        stages.append((f"M3 - {m3['Technique']}", "M3", "binary ridge map", m3_output.astype(float), "selected binary ridge restoration branch"))

        if str(m4["Technique"]) == "Medial Axis":
            m4_output = morphology.medial_axis(m3_output)
        else:
            m4_output = morphology.skeletonize(m3_output)
        stages.append((f"M4 - {m4['Technique']}", "M4", "skeleton", m4_output.astype(float), "downstream thinning/feature extraction branch"))

        previous = None
        for order, (stage, family, domain, image, note) in enumerate(stages):
            binary_override = image > 0.5 if domain != "grayscale" else None
            metrics = direct_quality_metrics(image if domain == "grayscale" else m2_output, binary_override=binary_override, metric_image=m2_output if domain != "grayscale" else image)
            row = {
                **selected.to_dict(),
                "Hybrid ID": hybrid_definition["Hybrid ID"],
                "Stage order": order,
                "Stage": stage,
                "Stage family": family,
                "Output domain": domain,
                "Stage note": note,
                "Runtime (s)": 0.0 if order == 0 else (m1_runtime if family == "M1" else m2_runtime if family == "M2" else math.nan),
                **metrics,
            }
            if previous is not None:
                for column in [
                    "Ridge coherence",
                    "Local contrast",
                    "Foreground/background separation proxy",
                    "Ridge coverage in foreground",
                    "Fragmentation per 1000 ridge pixels",
                    "Small connected components <20px",
                    "Ridge continuity proxy",
                ]:
                    row[f"Delta {column} vs previous stage"] = row[column] - previous[column]
            stage_rows.append(row)
            previous = row

        final_stage = stage_rows[-1]
        per_image_rows.append({**selected.to_dict(), **hybrid_definition, **{k: final_stage[k] for k in final_stage if k not in selected.index}})
    return pd.DataFrame(per_image_rows), pd.DataFrame(stage_rows), hybrid_definition


def create_direct_visuals(examples: dict[str, dict[str, np.ndarray]], summary_df: pd.DataFrame) -> list[str]:
    restoration = pick_overall_winner(
        family_winner_table(summary_df),
        "Restoration / Denoising Supporting Candidate",
    )
    restoration_name = str(restoration.get("Technique", "Wiener Deconvolution"))
    output_files = []
    panel_order = [
        "Original Altered input",
        "P0 neutral input",
        "CLAHE",
        "Contrast Stretching",
        "Ordinary Gabor",
        "Literature-Inspired Modified Gabor",
        "Coherence-Guided Directional Diffusion",
        restoration_name,
        "M3 Closing",
        "Morphological Skeleton",
        "Medial Axis",
    ]
    for category, example in examples.items():
        outputs = example["outputs"]
        panels = [(name, outputs[name]) for name in panel_order if name in outputs]
        columns = 4
        rows = int(math.ceil(len(panels) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.1, rows * 3.0))
        axes = np.asarray(axes).reshape(-1)
        for axis, (name, image) in zip(axes, panels):
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_title(name, fontsize=8)
            axis.axis("off")
        for axis in axes[len(panels) :]:
            axis.axis("off")
        fig.suptitle(f"direct24 {category} visual comparison - {example['image_name']}", fontsize=10)
        fig.tight_layout()
        path = OUTPUT_DIR / f"direct24_visual_{category.lower()}_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        output_files.append(str(path.relative_to(PROJECT_ROOT)))
    return output_files


def family_pivot(family_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in family_df["Family"].dropna().unique():
        frame = family_df[family_df["Family"] == family]
        overall = frame[frame["Category"] == "Overall"]
        rows.append(
            {
                "Family": family,
                "Technique": overall.iloc[0]["Technique"] if not overall.empty else "",
                "Easy": frame[frame["Category"] == "Easy"]["Technique"].iloc[0] if not frame[frame["Category"] == "Easy"].empty else "",
                "Medium": frame[frame["Category"] == "Medium"]["Technique"].iloc[0] if not frame[frame["Category"] == "Medium"].empty else "",
                "Hard": frame[frame["Category"] == "Hard"]["Technique"].iloc[0] if not frame[frame["Category"] == "Hard"].empty else "",
                "Overall": overall.iloc[0]["Technique"] if not overall.empty else "",
                "Main strength": overall.iloc[0]["Main strength"] if not overall.empty else "",
                "Main weakness": overall.iloc[0]["Main weakness"] if not overall.empty else "",
            }
        )
    return pd.DataFrame(rows)


def best_by(summary_df: pd.DataFrame, category: str, column: str, ascending: bool = False, family: str | None = None) -> pd.Series:
    frame = summary_df[(summary_df["Category"] == category) & (summary_df["Is enhancement technique"] == True)].copy()
    if family is not None:
        frame = frame[frame["Family"] == family]
    frame = frame[frame[column].notna()]
    if frame.empty:
        return pd.Series(dtype=object)
    return frame.sort_values(column, ascending=ascending).iloc[0]


def damage_flags(summary_df: pd.DataFrame) -> pd.DataFrame:
    frame = summary_df[(summary_df["Category"] == "Overall") & (summary_df["Is enhancement technique"] == True)].copy()
    damaged = frame[
        (frame["Delta coherence vs P0 mean"] < -0.02)
        | (frame["Delta fragmentation vs P0 mean"] > 3.0)
        | (frame["Delta continuity vs P0 mean"] < -5.0)
    ][
        [
            "Family",
            "Technique ID",
            "Technique",
            "Delta coherence vs P0 mean",
            "Delta contrast vs P0 mean",
            "Delta fragmentation vs P0 mean",
            "Delta continuity vs P0 mean",
        ]
    ].sort_values(["Delta fragmentation vs P0 mean", "Delta coherence vs P0 mean"], ascending=[False, True])
    return damaged


def create_report(
    selection_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    severity_summary_df: pd.DataFrame,
    family_df: pd.DataFrame,
    hybrid_stage_df: pd.DataFrame,
    runtime: dict[str, object],
    visual_files: list[str],
) -> str:
    family_table = family_pivot(family_df)
    m1 = family_df[family_df["Family"] == "M1 / Contrast or General Enhancement"]
    m2 = family_df[family_df["Family"] == "M2 / Ridge Enhancement"]
    m3 = family_df[family_df["Family"] == "M3 / Morphological Ridge Restoration"]
    m4 = family_df[family_df["Family"] == "M4 / Thinning"]
    overall_summary = severity_summary_df[severity_summary_df["Category"] == "Overall"]
    strongest_single = best_by(overall_summary, "Overall", "Ridge coherence mean")
    best_coherence = best_by(overall_summary, "Overall", "Ridge coherence mean")
    best_continuity = best_by(overall_summary, "Overall", "Fragmentation mean", ascending=True)
    best_contrast = best_by(overall_summary, "Overall", "Delta contrast vs P0 mean")
    hybrid_final = hybrid_stage_df[hybrid_stage_df["Stage family"] == "M4"].groupby("Category", as_index=False).agg(
        Coherence=("Ridge coherence", "mean"),
        Contrast=("Local contrast", "mean"),
        Fragmentation=("Fragmentation per 1000 ridge pixels", "mean"),
        Continuity=("Ridge continuity proxy", "mean"),
    )
    hybrid_gray = hybrid_stage_df[hybrid_stage_df["Stage family"] == "M2"].groupby("Category", as_index=False).agg(
        Coherence=("Ridge coherence", "mean"),
        Contrast=("Local contrast", "mean"),
        Fragmentation=("Fragmentation per 1000 ridge pixels", "mean"),
        Continuity=("Ridge continuity proxy", "mean"),
    )
    p0 = metrics_df[metrics_df["Technique ID"] == "P0"].groupby("Category", as_index=False).agg(
        Coherence=("Ridge coherence", "mean"),
        Contrast=("Local contrast", "mean"),
        Fragmentation=("Fragmentation per 1000 ridge pixels", "mean"),
        Continuity=("Ridge continuity proxy", "mean"),
    )
    hybrid_vs_input = hybrid_gray.merge(p0, on="Category", suffixes=("_hybrid", "_p0"))
    hybrid_vs_input["Coherence delta"] = hybrid_vs_input["Coherence_hybrid"] - hybrid_vs_input["Coherence_p0"]
    hybrid_vs_input["Contrast delta"] = hybrid_vs_input["Contrast_hybrid"] - hybrid_vs_input["Contrast_p0"]
    hybrid_vs_input["Fragmentation delta"] = hybrid_vs_input["Fragmentation_hybrid"] - hybrid_vs_input["Fragmentation_p0"]
    stage_help = best_by(
        hybrid_stage_df.groupby(["Stage"], as_index=False).agg(
            Delta_coherence=("Delta Ridge coherence vs previous stage", "mean"),
            Delta_contrast=("Delta Local contrast vs previous stage", "mean"),
            Delta_fragmentation=("Delta Fragmentation per 1000 ridge pixels vs previous stage", "mean"),
        ).assign(Category="Overall", Family="Hybrid Stage", Technique="stage", **{"Is enhancement technique": True}),
        "Overall",
        "Delta_coherence",
    )
    stage_damage = best_by(
        hybrid_stage_df.groupby(["Stage"], as_index=False).agg(
            Delta_coherence=("Delta Ridge coherence vs previous stage", "mean"),
            Delta_contrast=("Delta Local contrast vs previous stage", "mean"),
            Delta_fragmentation=("Delta Fragmentation per 1000 ridge pixels vs previous stage", "mean"),
        ).assign(Category="Overall", Family="Hybrid Stage", Technique="stage", **{"Is enhancement technique": True}),
        "Overall",
        "Delta_fragmentation",
    )
    damaged = damage_flags(overall_summary)
    direct_counts = selection_df.groupby("Category")["Image"].count().to_dict()
    pair_counts = selection_df.groupby("Category")["Real pair candidate exists"].sum().astype(int).to_dict()
    hybrid_overall = hybrid_vs_input.mean(numeric_only=True).to_dict()
    hybrid_better_than_input = (
        hybrid_overall.get("Coherence delta", 0.0) > 0
        and hybrid_overall.get("Fragmentation delta", 0.0) <= 0
    )
    hybrid_best_single_gap = float(hybrid_gray["Coherence"].mean() - strongest_single.get("Ridge coherence mean", math.nan))
    hybrid_outperforms_single = hybrid_best_single_gap > 0
    hybrid_single_verdict = (
        "Yes by ridge coherence only; no as a universal structural winner because contrast and fragmentation must also be checked."
        if hybrid_outperforms_single
        else "No by the selected ridge-coherence comparison."
    )
    lines = [
        "# direct24 Direct Original Altered Fingerprint Enhancement Report",
        "",
        "## A. What original dataset images were directly tested?",
        f"- 24 original SOCOFing Altered images were tested directly: {direct_counts}. Selection was deterministic: sorted file lists with 8 evenly spaced paths per category.",
        f"- Real filename pair candidates existed as follows: {pair_counts}. These pairs were documented but not used for metrics because geometric alignment was not verified.",
        "",
        "## B. Was synthetic degradation used in the main experiment?",
        "- No. The main experiment used original Altered Easy/Medium/Hard images directly.",
        "",
        "## C. What neutral preprocessing was used?",
        "- P0 only: read image, convert to grayscale if needed, resize to 256x256 for consistent processing, convert to float [0,1]. No filtering, CLAHE, histogram equalisation, denoising, sharpening, or Gabor filtering was applied as common preprocessing.",
        "",
        "## D. Which M1 techniques were tested?",
        "- CLAHE, Contrast Stretching, Global Histogram Equalization.",
        "",
        "## E. Which M2 techniques were tested?",
        "- Ordinary Gabor, Literature-Inspired Modified Gabor, Coherence-Guided Directional Diffusion, Directional Oriented Filter-Bank.",
        "",
        "## F. Which restoration/denoising supporting techniques were tested?",
        "- Wiener Deconvolution, Wavelet Denoising, Total Variation Restoration. These are labelled as supporting restoration/denoising candidates, not as CLAHE/Gabor equivalents.",
        "",
        "## G. Which M3 techniques were tested?",
        "- Closing, Opening + Closing, Morphological Reconstruction, Remove Small Objects, Controlled Ridge-Restoration Pipeline.",
        "",
        "## H. Which M4 techniques were tested?",
        "- Morphological Skeleton and Medial Axis.",
        "",
        "## I. Which metrics were used for direct real-image evaluation, and why?",
        "- Direct metrics: ridge coherence, local contrast, foreground/background separation proxy, ridge coverage, connected-component fragmentation, small connected-component count, ridge-continuity proxy, detected ridge endings/bifurcations, and runtime.",
        "- These are no-reference or structural indicators suited to unregistered Altered fingerprints. PSNR/MSE/SSIM are not primary direct rankings because clean pixel-aligned ground truth is not verified.",
        "",
        "## J. What was the best M1 technique for Easy, Medium, Hard and overall?",
        m1[["Category", "Technique", "Main strength", "Main weakness", "Ridge coherence", "Local contrast", "Fragmentation"]].round(4).to_markdown(index=False),
        "",
        "## K. What was the best M2 technique for Easy, Medium, Hard and overall?",
        m2[["Category", "Technique", "Main strength", "Main weakness", "Ridge coherence", "Local contrast", "Fragmentation"]].round(4).to_markdown(index=False),
        "",
        "## L. What was the best M3 technique?",
        m3[["Category", "Technique", "Main strength", "Main weakness", "Fragmentation", "Ridge continuity"]].round(4).to_markdown(index=False),
        "",
        "## M. What was the best M4 technique?",
        m4[["Category", "Technique", "Main strength", "Main weakness", "Fragmentation", "Ridge continuity"]].round(4).to_markdown(index=False),
        "",
        "## N. Did any technique visibly damage ridge information despite improving a metric?",
        damaged.round(4).to_markdown(index=False) if not damaged.empty else "- No severe structural damage flag was triggered by the defined direct metrics.",
        "- Visual comparison PNGs were exported for one Easy, one Medium and one Hard example; use these to confirm whether metric gains match ridge visibility.",
        "",
        "## O. Which technique produced the best ridge coherence?",
        f"- {best_coherence.get('Technique', 'n/a')} ({best_coherence.get('Family', 'n/a')}), overall ridge coherence {best_coherence.get('Ridge coherence mean', math.nan):.4f}.",
        "",
        "## P. Which technique produced the best ridge continuity / lowest fragmentation?",
        f"- {best_continuity.get('Technique', 'n/a')} ({best_continuity.get('Family', 'n/a')}), fragmentation {best_continuity.get('Fragmentation mean', math.nan):.4f}.",
        "",
        "## Q. Which technique produced the best contrast improvement?",
        f"- {best_contrast.get('Technique', 'n/a')} ({best_contrast.get('Family', 'n/a')}), delta local contrast vs P0 {best_contrast.get('Delta contrast vs P0 mean', math.nan):+.4f}.",
        "",
        "## R. What technique is recommended for each team member and why?",
        family_table.to_markdown(index=False),
        "",
        "## S. What one evidence-based team hybrid was constructed?",
        f"- DIRECT24-H1: P0 -> {pick_overall_winner(family_df, 'M1 / Contrast or General Enhancement').get('Technique')} -> {pick_overall_winner(family_df, 'M2 / Ridge Enhancement').get('Technique')} -> {pick_overall_winner(family_df, 'M3 / Morphological Ridge Restoration').get('Technique')} -> {pick_overall_winner(family_df, 'M4 / Thinning').get('Technique')} as downstream analysis.",
        "",
        "## T. Did the hybrid improve over the original Altered inputs?",
        f"- Hybrid greyscale branch vs P0 overall: coherence delta {hybrid_overall.get('Coherence delta', math.nan):+.4f}, contrast delta {hybrid_overall.get('Contrast delta', math.nan):+.4f}, fragmentation delta {hybrid_overall.get('Fragmentation delta', math.nan):+.4f}. Genuinely better by the strict coherence-plus-fragmentation rule: {hybrid_better_than_input}.",
        "",
        "## U. Did the hybrid outperform the strongest single technique?",
        f"- Strongest single technique by coherence: {strongest_single.get('Technique', 'n/a')} at {strongest_single.get('Ridge coherence mean', math.nan):.4f}. Hybrid greyscale branch mean coherence gap: {hybrid_best_single_gap:+.4f}. Verdict: {hybrid_single_verdict}",
        "",
        "## V. Which hybrid stage helped most?",
        f"- {stage_help.get('Stage', 'n/a')} gave the largest mean coherence gain: {stage_help.get('Delta_coherence', math.nan):+.4f}.",
        "",
        "## W. Which hybrid stage damaged quality most?",
        f"- {stage_damage.get('Stage', 'n/a')} gave the largest mean fragmentation increase: {stage_damage.get('Delta_fragmentation', math.nan):+.4f}.",
        "",
        "## X. How do results differ between Easy, Medium and Hard?",
        severity_summary_df[
            (severity_summary_df["Technique ID"] == "P0")
            & (severity_summary_df["Category"].isin(["Easy", "Medium", "Hard"]))
        ][["Category", "Ridge coherence mean", "Local contrast mean", "Fragmentation mean", "Ridge continuity mean"]].round(4).to_markdown(index=False),
        "- The per-category tables above show family winners can differ by difficulty; do not force one universal winner where the category evidence differs.",
        "",
        "## Y. Why are PSNR/MSE/SSIM not the main ranking metrics in this direct experiment?",
        "- The direct inputs are Altered images whose clean Real counterparts may be transformed or cropped. Without verified pixel alignment, full-reference metrics would measure registration differences rather than enhancement quality.",
        "",
        "## Z. What role does the previous synthetic-degradation experiment still serve?",
        "- It remains a secondary controlled quantitative experiment: Real reference -> synthetic degradation -> enhancement -> PSNR/MSE/SSIM against Real. It answers recoverability under known degradation, not direct real Altered-image enhancement.",
        "",
        "## AA. Is the literature 28.17 dB Modified Gabor result directly comparable to this project?",
        "- No. The reported 28.17 dB is contextual literature evidence only. It is not directly comparable unless dataset, image selection, degradation, preprocessing, parameters, PSNR formula and reference design are all the same, and those details remain unavailable here.",
        "",
        "## AB. Based on the direct 24-image evidence, is the experiment now ready for a larger dataset run?",
        "- Not yet. Review the direct visual panels and per-category winners first, then run one larger direct Altered-image validation only for the selected M1/M2/M3/M4 pipeline and strongest single-technique baseline.",
        "- Visual evidence files:",
        "\n".join(f"- {path}" for path in visual_files),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    timer = perf_counter()
    protected_before = existing_output_snapshot()
    print("direct24 original Altered-image experiment started")
    selection_df = select_direct_images()
    print(f"Selected {len(selection_df)} images: {selection_df.groupby('Category')['Image'].count().to_dict()}")
    metrics_df, examples, technique_runtimes = run_direct_single_techniques(selection_df)
    severity_summary_df = summarise_by_category(metrics_df)
    family_df = family_winner_table(severity_summary_df)
    hybrid_per_image_df, hybrid_stage_df, hybrid_definition = run_one_hybrid(selection_df, family_df)
    visual_files = create_direct_visuals(examples, severity_summary_df)
    protected_after = existing_output_snapshot()
    runtime = {
        "created_at": datetime.now().astimezone().isoformat(),
        "runtime_seconds": perf_counter() - timer,
        "main_experiment_used_synthetic_degradation": False,
        "full_500_image_experiment_run": False,
        "bruteforce_hybrids_run": False,
        "selection_counts": selection_df.groupby("Category")["Image"].count().to_dict(),
        "selection_rule": "For each Altered category, sort filenames and choose 8 evenly spaced images.",
        "neutral_preprocessing": "P0: grayscale, resize to 256x256, float [0,1] only",
        "real_pair_candidates_verified_by_filename": selection_df.groupby("Category")["Real pair candidate exists"].sum().astype(int).to_dict(),
        "pairing_used_for_full_reference_metrics": False,
        "psnr_mse_ssim_primary_metrics": False,
        "technique_runtime_totals_seconds": technique_runtimes,
        "hybrid_definition": hybrid_definition,
        "protected_existing_outputs_unchanged": protected_before == protected_after,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "visual_files": visual_files,
    }

    metrics_df.to_csv(DIRECT_FILES["per_image"], index=False)
    severity_summary_df.to_csv(DIRECT_FILES["per_severity"], index=False)
    family_df.to_csv(DIRECT_FILES["family"], index=False)
    hybrid_stage_df.to_csv(DIRECT_FILES["hybrid_stage"], index=False)
    DIRECT_FILES["runtime"].write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    DIRECT_FILES["metadata"].write_text(
        json.dumps(
            {
                "selection": selection_df.to_dict(orient="records"),
                "outputs": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in DIRECT_FILES.items()},
                "visual_files": visual_files,
                "hybrid_per_image_rows": hybrid_per_image_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = create_report(selection_df, metrics_df, severity_summary_df, family_df, hybrid_stage_df, runtime, visual_files)
    DIRECT_FILES["report"].write_text(report, encoding="utf-8")
    print(f"direct24 complete in {runtime['runtime_seconds']:.2f} seconds")
    print(f"Report: {DIRECT_FILES['report']}")


if __name__ == "__main__":
    main()
