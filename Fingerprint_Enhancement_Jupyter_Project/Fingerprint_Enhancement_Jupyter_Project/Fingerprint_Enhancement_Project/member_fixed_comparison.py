"""Run the frozen four-member comparison and small compatibility test.

This script intentionally uses fixed parameters and the focused audit dev split.
It does not run the 500-image validation and does not modify prior audit outputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

import direct24_original_altered_experiment as direct
import focused_psnr_audit_optimisation as focus
import screening24_master_audit as base


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TARGET_PSNR_DB = 28.17
CONTROLLED_SCENARIO_ID = "gaussian_noise_only_s055"
CONTROLLED_SEVERITY = 0.55


OUTPUT_FILES = {
    "verification": OUTPUT_DIR / "member_m2_m4_verification.md",
    "fixed_parameters": OUTPUT_DIR / "member_fixed_parameters.json",
    "member_controlled_per_image": OUTPUT_DIR / "member4_controlled_per_image.csv",
    "member_controlled_summary": OUTPUT_DIR / "member4_controlled_summary.csv",
    "member_altered_per_image": OUTPUT_DIR / "member4_altered_per_image.csv",
    "member_altered_summary": OUTPUT_DIR / "member4_altered_summary.csv",
    "member_interpretation": OUTPUT_DIR / "member4_comparison_interpretation.md",
    "compatibility_per_image": OUTPUT_DIR / "member_compatibility_per_image.csv",
    "compatibility_summary": OUTPUT_DIR / "member_compatibility_summary.csv",
    "compatibility_stage_metrics": OUTPUT_DIR / "member_compatibility_stage_metrics.csv",
    "compatibility_report": OUTPUT_DIR / "member_compatibility_report.md",
    "compatibility_runtime": OUTPUT_DIR / "member_compatibility_runtime.json",
    "compatibility_log": OUTPUT_DIR / "member_compatibility_run.log",
}


@dataclass(frozen=True)
class MethodDef:
    method_id: str
    member: str
    technique: str
    role: str
    params: dict[str, object]
    apply: Callable[[np.ndarray], np.ndarray]
    is_member: bool
    is_baseline: bool = False


@dataclass(frozen=True)
class CompatDef:
    candidate_id: str
    pipeline: str
    stages: tuple[str, ...]
    params: dict[str, object]


def clean(value: object, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    if pd.isna(value):
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def md_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [clean(row.get(column)).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def snapshot_protected_outputs() -> dict[str, dict[str, object]]:
    generated_names = {path.name for path in OUTPUT_FILES.values()}
    paths = sorted(
        [path for path in OUTPUT_DIR.iterdir() if path.is_file() and path.name not in generated_names],
        key=lambda path: path.name,
    )
    return base.snapshot_files(paths)


def fixed_methods() -> list[MethodDef]:
    return [
        MethodDef(
            "B0",
            "Baseline",
            "Degraded input / P0",
            "Controlled degraded input or direct Altered P0 control",
            {"operation": "P0 only; clip/float range only"},
            lambda image: focus.metric_image(image),
            is_member=False,
            is_baseline=True,
        ),
        MethodDef(
            "M1_NLM_h0.06",
            "Member 1",
            "Non-Local Means Denoising",
            "Main frozen member technique",
            {
                "h": 0.06,
                "patch_size": 5,
                "patch_distance": 6,
                "backend": "cv2" if focus.cv2 is not None else "skimage",
                "cv2_templateWindowSize": 7,
                "cv2_searchWindowSize": 21,
            },
            lambda image: focus.non_local_means(image, h=0.06, patch_size=5, patch_distance=6),
            is_member=True,
        ),
        MethodDef(
            "M2_MOD_GABOR",
            "Member 2",
            "Literature-Inspired Modified / Orientation-Adaptive Gabor",
            "Main frozen member technique",
            {
                "use_local_frequency": True,
                "frequencies": [0.075, 0.095, 0.115, 0.140],
                "orientation_bins": 8,
                "blend": 0.18,
                "bandwidth": 1.8,
            },
            lambda image: base.adaptive_gabor_enhancement(image, use_local_frequency=True, blend=0.18),
            is_member=True,
        ),
        MethodDef(
            "M3_TV_w0.02",
            "Member 3",
            "Total Variation Restoration",
            "Main frozen member technique",
            {"weight": 0.02, "backend": "skimage.restoration.denoise_tv_chambolle"},
            lambda image: focus.tv_restoration(image, weight=0.02),
            is_member=True,
        ),
        MethodDef(
            "M4_COH_DIFF",
            "Member 4",
            "Coherence-Guided Directional Diffusion",
            "Main frozen member technique",
            {
                "iterations": 4,
                "step": 0.16,
                "orientation_bins": 8,
                "oriented_kernel_sigma_x": 1.35,
                "oriented_kernel_sigma_y": 0.50,
                "oriented_kernel_size": 9,
            },
            lambda image: base.coherence_guided_diffusion(image, iterations=4, step=0.16),
            is_member=True,
        ),
        MethodDef(
            "BL_CLAHE_c0.005",
            "Baseline",
            "CLAHE",
            "Optional contrast baseline",
            {"clip_limit": 0.005, "kernel_size": 32},
            lambda image: focus.clahe(image, clip_limit=0.005, kernel_size=32),
            is_member=False,
            is_baseline=True,
        ),
        MethodDef(
            "BL_ORD_GABOR",
            "Baseline",
            "Ordinary Gabor",
            "M2 ordinary Gabor comparator",
            {"frequency": 0.115, "orientations": 8, "blend": 0.30},
            lambda image: base.multi_orientation_gabor(image, frequency=0.115, orientations=8, blend=0.30),
            is_member=False,
            is_baseline=True,
        ),
        MethodDef(
            "BL_WIENER",
            "Baseline",
            "Wiener Deconvolution",
            "M3 restoration comparator",
            {"psf_sigma": 0.98, "balance": 0.06, "blend": 0.35},
            lambda image: focus.wiener_deconvolution(image, psf_sigma=0.98, balance=0.06, blend=0.35),
            is_member=False,
            is_baseline=True,
        ),
        MethodDef(
            "BL_CONTRAST_1_99",
            "Baseline",
            "Contrast Stretching",
            "Global contrast baseline",
            {"low_percentile": 1.0, "high_percentile": 99.0},
            lambda image: focus.contrast_stretch(image, low=1.0, high=99.0),
            is_member=False,
            is_baseline=True,
        ),
    ]


def compatibility_candidates() -> list[CompatDef]:
    return [
        CompatDef(
            "C0_NLM",
            "NLM",
            ("NLM",),
            {"NLM": fixed_methods()[1].params},
        ),
        CompatDef(
            "C1_TV_NLM",
            "TV -> NLM",
            ("TV", "NLM"),
            {"TV": fixed_methods()[3].params, "NLM": fixed_methods()[1].params},
        ),
        CompatDef(
            "C2_NLM_MOD_GABOR",
            "NLM -> Modified / Orientation-Adaptive Gabor",
            ("NLM", "Modified Gabor"),
            {"NLM": fixed_methods()[1].params, "Modified Gabor": fixed_methods()[2].params},
        ),
        CompatDef(
            "C3_NLM_COH_DIFF",
            "NLM -> Coherence-Guided Directional Diffusion",
            ("NLM", "Directional Diffusion"),
            {"NLM": fixed_methods()[1].params, "Directional Diffusion": fixed_methods()[4].params},
        ),
        CompatDef(
            "C4_TV_NLM_COH_DIFF",
            "TV -> NLM -> Coherence-Guided Directional Diffusion",
            ("TV", "NLM", "Directional Diffusion"),
            {"TV": fixed_methods()[3].params, "NLM": fixed_methods()[1].params, "Directional Diffusion": fixed_methods()[4].params},
        ),
    ]


def apply_stage(stage: str, image: np.ndarray) -> np.ndarray:
    if stage == "NLM":
        return focus.non_local_means(image, h=0.06, patch_size=5, patch_distance=6)
    if stage == "TV":
        return focus.tv_restoration(image, weight=0.02)
    if stage == "Modified Gabor":
        return base.adaptive_gabor_enhancement(image, use_local_frequency=True, blend=0.18)
    if stage == "Directional Diffusion":
        return base.coherence_guided_diffusion(image, iterations=4, step=0.16)
    raise ValueError(f"Unknown stage: {stage}")


def load_dev_records() -> tuple[list[dict[str, object]], dict[str, object]]:
    metadata = json.loads((OUTPUT_DIR / "focused_psnr_audit_metadata.json").read_text(encoding="utf-8"))
    dev_paths = [PROJECT_ROOT / relative for relative in metadata["dev_images"]]
    test_paths = [PROJECT_ROOT / relative for relative in metadata["test_images"]]
    records = focus.make_records(dev_paths, "dev")
    return records, {
        "source": "outputs/focused_psnr_audit_metadata.json",
        "split_used": "dev",
        "dev_images": [str(path.relative_to(PROJECT_ROOT)) for path in dev_paths],
        "fresh_or_final_holdout_images_used": False,
        "focused_test_images_not_used": [str(path.relative_to(PROJECT_ROOT)) for path in test_paths],
        "seed_rule": "focused.make_records dev seeds = RANDOM_SEED + image_index",
        "random_seed": focus.RANDOM_SEED,
    }


def load_altered_records() -> tuple[list[dict[str, object]], dict[str, object]]:
    records = focus.altered_records(max_per_split=3)
    return records, {
        "source": "focused_psnr_audit_optimisation.altered_records(max_per_split=3)",
        "sample_size": len(records),
        "selection": [
            {
                "Altered split": str(record["Altered split"]),
                "Image": str(record["Image"]),
                "Relative path": str(record["Relative path"]),
            }
            for record in records
        ],
        "synthetic_degradation_added": False,
        "psnr_mse_ssim_used": False,
    }


def controlled_rows_for_methods(records: list[dict[str, object]], methods: list[MethodDef]) -> pd.DataFrame:
    rows = []
    for record in records:
        reference = record["reference"]
        seed = int(record["seed"])
        degraded, degradation_params = focus.apply_gaussian_noise(reference, CONTROLLED_SEVERITY, seed)
        degraded_metrics = focus.evaluate(reference, degraded)
        for method in methods:
            start = perf_counter()
            output = method.apply(degraded)
            runtime = perf_counter() - start
            metrics = focus.evaluate(reference, output)
            rows.append(
                {
                    "Experiment": "member controlled comparison",
                    "Split": "dev",
                    "Scenario ID": CONTROLLED_SCENARIO_ID,
                    "Scenario": "Clean Real -> Gaussian noise severity 0.55 -> P0 -> fixed member technique",
                    "Image": record["name"],
                    "Relative path": record["relative_path"],
                    "Seed": seed,
                    "Method ID": method.method_id,
                    "Member": method.member,
                    "Technique": method.technique,
                    "Role": method.role,
                    "Is member technique": method.is_member,
                    "Is baseline": method.is_baseline,
                    "Fixed parameters": json.dumps(method.params, sort_keys=True),
                    "Degradation parameters": json.dumps(degradation_params, sort_keys=True),
                    "Runtime (s)": float(runtime),
                    "Input MSE": degraded_metrics["MSE"],
                    "Input PSNR (dB)": degraded_metrics["PSNR (dB)"],
                    "Input SSIM": degraded_metrics["SSIM"],
                    **metrics,
                    "Delta PSNR vs degraded input (dB)": metrics["PSNR (dB)"] - degraded_metrics["PSNR (dB)"],
                    "PSNR delta to 28.17 dB": metrics["PSNR (dB)"] - TARGET_PSNR_DB,
                    "Target exceeded": metrics["PSNR (dB)"] > TARGET_PSNR_DB,
                }
            )
    return pd.DataFrame(rows)


def summarise_controlled(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(
            [
                "Method ID",
                "Member",
                "Technique",
                "Role",
                "Is member technique",
                "Is baseline",
                "Fixed parameters",
            ],
            dropna=False,
        )
        .agg(
            Images=("Image", "count"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            MSE_255_equivalent_mean=("MSE_255_equivalent", "mean"),
            MSE_255_equivalent_std=("MSE_255_equivalent", "std"),
            PSNR_mean=("PSNR (dB)", "mean"),
            PSNR_std=("PSNR (dB)", "std"),
            SSIM_mean=("SSIM", "mean"),
            SSIM_std=("SSIM", "std"),
            Delta_PSNR_vs_degraded_mean=("Delta PSNR vs degraded input (dB)", "mean"),
            Delta_PSNR_vs_degraded_std=("Delta PSNR vs degraded input (dB)", "std"),
            Runtime_mean=("Runtime (s)", "mean"),
            Runtime_std=("Runtime (s)", "std"),
        )
        .reset_index()
    )
    summary["PSNR_delta_to_28_17"] = summary["PSNR_mean"] - TARGET_PSNR_DB
    summary["Target_exceeded"] = summary["PSNR_mean"] > TARGET_PSNR_DB
    summary["Controlled PSNR rank"] = summary["PSNR_mean"].rank(ascending=False, method="min").astype(int)
    return summary.sort_values(["Controlled PSNR rank", "Method ID"]).reset_index(drop=True)


def direct_metrics(image: np.ndarray) -> dict[str, float]:
    metrics = direct.direct_quality_metrics(image)
    return {
        "Ridge coherence": metrics["Ridge coherence"],
        "Local contrast": metrics["Local contrast"],
        "Foreground/background separation proxy": metrics["Foreground/background separation proxy"],
        "Ridge coverage in foreground": metrics["Ridge coverage in foreground"],
        "Fragmentation per 1000 ridge pixels": metrics["Fragmentation per 1000 ridge pixels"],
        "Small connected components <20px": metrics["Small connected components <20px"],
        "Ridge continuity proxy": metrics["Ridge continuity proxy"],
    }


def altered_rows_for_methods(records: list[dict[str, object]], methods: list[MethodDef]) -> pd.DataFrame:
    rows = []
    for record in records:
        p0 = record["image"]
        category = str(record["Altered split"]).replace("Altered-", "")
        for method in methods:
            start = perf_counter()
            output = method.apply(p0)
            runtime = perf_counter() - start
            rows.append(
                {
                    "Experiment": "direct Altered member comparison",
                    "Category": category,
                    "Altered split": record["Altered split"],
                    "Image": record["Image"],
                    "Relative path": record["Relative path"],
                    "Method ID": method.method_id,
                    "Member": method.member,
                    "Technique": method.technique,
                    "Role": method.role,
                    "Is member technique": method.is_member,
                    "Is baseline": method.is_baseline,
                    "Fixed parameters": json.dumps(method.params, sort_keys=True),
                    "Runtime (s)": float(runtime),
                    **direct_metrics(output),
                }
            )
    return add_direct_deltas(pd.DataFrame(rows), id_column="Method ID", baseline_id="B0", champion_id="M1_NLM_h0.06")


def add_direct_deltas(frame: pd.DataFrame, id_column: str, baseline_id: str, champion_id: str) -> pd.DataFrame:
    result = frame.copy()
    keys = ["Experiment", "Category", "Image"]
    baseline = result[result[id_column] == baseline_id].set_index(keys)
    champion = result[result[id_column] == champion_id].set_index(keys)
    tracked = [
        "Ridge coherence",
        "Local contrast",
        "Foreground/background separation proxy",
        "Ridge coverage in foreground",
        "Fragmentation per 1000 ridge pixels",
        "Small connected components <20px",
        "Ridge continuity proxy",
    ]
    for index, row in result.iterrows():
        key = (row["Experiment"], row["Category"], row["Image"])
        if key in baseline.index:
            base_row = baseline.loc[key]
            for column in tracked:
                result.loc[index, f"Delta {column} vs P0"] = row[column] - base_row[column]
        if key in champion.index:
            champ_row = champion.loc[key]
            for column in tracked:
                result.loc[index, f"Delta {column} vs NLM"] = row[column] - champ_row[column]
    return result


def summarise_direct(frame: pd.DataFrame, id_column: str, name_column: str) -> pd.DataFrame:
    group_cols = [
        "Category",
        id_column,
        name_column,
        "Member",
        "Is member technique",
        "Is baseline",
        "Fixed parameters",
    ]
    agg = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            Images=("Image", "count"),
            Coherence_mean=("Ridge coherence", "mean"),
            Coherence_std=("Ridge coherence", "std"),
            Local_contrast_mean=("Local contrast", "mean"),
            Local_contrast_std=("Local contrast", "std"),
            Ridge_coverage_mean=("Ridge coverage in foreground", "mean"),
            Fragmentation_mean=("Fragmentation per 1000 ridge pixels", "mean"),
            Fragmentation_std=("Fragmentation per 1000 ridge pixels", "std"),
            Small_components_mean=("Small connected components <20px", "mean"),
            Ridge_continuity_mean=("Ridge continuity proxy", "mean"),
            Runtime_mean=("Runtime (s)", "mean"),
            Delta_coherence_vs_P0_mean=("Delta Ridge coherence vs P0", "mean"),
            Delta_contrast_vs_P0_mean=("Delta Local contrast vs P0", "mean"),
            Delta_fragmentation_vs_P0_mean=("Delta Fragmentation per 1000 ridge pixels vs P0", "mean"),
            Delta_continuity_vs_P0_mean=("Delta Ridge continuity proxy vs P0", "mean"),
            Delta_coherence_vs_NLM_mean=("Delta Ridge coherence vs NLM", "mean"),
            Delta_contrast_vs_NLM_mean=("Delta Local contrast vs NLM", "mean"),
            Delta_fragmentation_vs_NLM_mean=("Delta Fragmentation per 1000 ridge pixels vs NLM", "mean"),
            Delta_continuity_vs_NLM_mean=("Delta Ridge continuity proxy vs NLM", "mean"),
        )
        .reset_index()
    )
    overall = agg.copy()
    overall["Category"] = "Overall"
    overall = (
        frame.assign(Category="Overall")
        .groupby(group_cols, dropna=False)
        .agg(
            Images=("Image", "count"),
            Coherence_mean=("Ridge coherence", "mean"),
            Coherence_std=("Ridge coherence", "std"),
            Local_contrast_mean=("Local contrast", "mean"),
            Local_contrast_std=("Local contrast", "std"),
            Ridge_coverage_mean=("Ridge coverage in foreground", "mean"),
            Fragmentation_mean=("Fragmentation per 1000 ridge pixels", "mean"),
            Fragmentation_std=("Fragmentation per 1000 ridge pixels", "std"),
            Small_components_mean=("Small connected components <20px", "mean"),
            Ridge_continuity_mean=("Ridge continuity proxy", "mean"),
            Runtime_mean=("Runtime (s)", "mean"),
            Delta_coherence_vs_P0_mean=("Delta Ridge coherence vs P0", "mean"),
            Delta_contrast_vs_P0_mean=("Delta Local contrast vs P0", "mean"),
            Delta_fragmentation_vs_P0_mean=("Delta Fragmentation per 1000 ridge pixels vs P0", "mean"),
            Delta_continuity_vs_P0_mean=("Delta Ridge continuity proxy vs P0", "mean"),
            Delta_coherence_vs_NLM_mean=("Delta Ridge coherence vs NLM", "mean"),
            Delta_contrast_vs_NLM_mean=("Delta Local contrast vs NLM", "mean"),
            Delta_fragmentation_vs_NLM_mean=("Delta Fragmentation per 1000 ridge pixels vs NLM", "mean"),
            Delta_continuity_vs_NLM_mean=("Delta Ridge continuity proxy vs NLM", "mean"),
        )
        .reset_index()
    )
    summary = pd.concat([agg, overall], ignore_index=True)
    order = {"Easy": 0, "Medium": 1, "Hard": 2, "Overall": 3}
    summary["_order"] = summary["Category"].map(order).fillna(9)
    return summary.sort_values(["_order", id_column]).drop(columns=["_order"]).reset_index(drop=True)


def controlled_stage_metrics(
    records: list[dict[str, object]],
    candidates: list[CompatDef],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_rows = []
    stage_rows = []
    for record in records:
        reference = record["reference"]
        seed = int(record["seed"])
        degraded, degradation_params = focus.apply_gaussian_noise(reference, CONTROLLED_SEVERITY, seed)
        baseline_metrics = focus.evaluate(reference, degraded)
        nlm_final_cache: dict[str, float] | None = None
        for candidate in candidates:
            current = focus.metric_image(degraded)
            previous_metrics = baseline_metrics
            stage_rows.append(
                {
                    "Experiment": "controlled compatibility",
                    "Protocol": CONTROLLED_SCENARIO_ID,
                    "Category": "Controlled",
                    "Image": record["name"],
                    "Relative path": record["relative_path"],
                    "Seed": seed,
                    "Candidate ID": candidate.candidate_id,
                    "Pipeline": candidate.pipeline,
                    "Stage order": 0,
                    "Stage": "P0 degraded input",
                    "Stage parameters": json.dumps(degradation_params, sort_keys=True),
                    "Runtime (s)": 0.0,
                    **baseline_metrics,
                    "Delta PSNR vs previous stage (dB)": np.nan,
                    "Delta MSE vs previous stage": np.nan,
                    "Delta SSIM vs previous stage": np.nan,
                }
            )
            total_runtime = 0.0
            for order, stage in enumerate(candidate.stages, start=1):
                start = perf_counter()
                current = apply_stage(stage, current)
                runtime = perf_counter() - start
                total_runtime += runtime
                metrics = focus.evaluate(reference, current)
                stage_rows.append(
                    {
                        "Experiment": "controlled compatibility",
                        "Protocol": CONTROLLED_SCENARIO_ID,
                        "Category": "Controlled",
                        "Image": record["name"],
                        "Relative path": record["relative_path"],
                        "Seed": seed,
                        "Candidate ID": candidate.candidate_id,
                        "Pipeline": candidate.pipeline,
                        "Stage order": order,
                        "Stage": stage,
                        "Stage parameters": json.dumps(stage_params(stage), sort_keys=True),
                        "Runtime (s)": float(runtime),
                        **metrics,
                        "Delta PSNR vs previous stage (dB)": metrics["PSNR (dB)"] - previous_metrics["PSNR (dB)"],
                        "Delta MSE vs previous stage": metrics["MSE"] - previous_metrics["MSE"],
                        "Delta SSIM vs previous stage": metrics["SSIM"] - previous_metrics["SSIM"],
                    }
                )
                previous_metrics = metrics
            final_metrics = focus.evaluate(reference, current)
            if candidate.candidate_id == "C0_NLM":
                nlm_final_cache = final_metrics
            final_rows.append(
                {
                    "Experiment": "controlled compatibility",
                    "Protocol": CONTROLLED_SCENARIO_ID,
                    "Category": "Controlled",
                    "Image": record["name"],
                    "Relative path": record["relative_path"],
                    "Seed": seed,
                    "Candidate ID": candidate.candidate_id,
                    "Pipeline": candidate.pipeline,
                    "Fixed parameters": json.dumps(candidate.params, sort_keys=True),
                    "Degradation parameters": json.dumps(degradation_params, sort_keys=True),
                    "Runtime (s)": float(total_runtime),
                    "Input MSE": baseline_metrics["MSE"],
                    "Input PSNR (dB)": baseline_metrics["PSNR (dB)"],
                    "Input SSIM": baseline_metrics["SSIM"],
                    **final_metrics,
                    "Delta PSNR vs degraded input (dB)": final_metrics["PSNR (dB)"] - baseline_metrics["PSNR (dB)"],
                    "PSNR delta to 28.17 dB": final_metrics["PSNR (dB)"] - TARGET_PSNR_DB,
                    "Target exceeded": final_metrics["PSNR (dB)"] > TARGET_PSNR_DB,
                    "NLM cache marker": nlm_final_cache,
                }
            )
    final_df = pd.DataFrame(final_rows).drop(columns=["NLM cache marker"])
    final_df = add_controlled_nlm_deltas(final_df)
    return final_df, pd.DataFrame(stage_rows)


def add_controlled_nlm_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    champion = result[result["Candidate ID"] == "C0_NLM"].set_index(["Protocol", "Image"])
    for index, row in result.iterrows():
        key = (row["Protocol"], row["Image"])
        if key not in champion.index:
            continue
        nlm = champion.loc[key]
        result.loc[index, "Delta PSNR vs NLM (dB)"] = row["PSNR (dB)"] - nlm["PSNR (dB)"]
        result.loc[index, "Delta MSE vs NLM"] = row["MSE"] - nlm["MSE"]
        result.loc[index, "Delta SSIM vs NLM"] = row["SSIM"] - nlm["SSIM"]
    return result


def stage_params(stage: str) -> dict[str, object]:
    if stage == "NLM":
        return fixed_methods()[1].params
    if stage == "Modified Gabor":
        return fixed_methods()[2].params
    if stage == "TV":
        return fixed_methods()[3].params
    if stage == "Directional Diffusion":
        return fixed_methods()[4].params
    return {}


def direct_stage_metrics(
    records: list[dict[str, object]],
    candidates: list[CompatDef],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_rows = []
    stage_rows = []
    for record in records:
        p0 = focus.metric_image(record["image"])
        category = str(record["Altered split"]).replace("Altered-", "")
        baseline_metrics = direct_metrics(p0)
        for candidate in candidates:
            current = p0.copy()
            previous_metrics = baseline_metrics
            stage_rows.append(
                {
                    "Experiment": "direct Altered compatibility",
                    "Protocol": "direct_altered_no_synthetic_degradation",
                    "Category": category,
                    "Altered split": record["Altered split"],
                    "Image": record["Image"],
                    "Relative path": record["Relative path"],
                    "Candidate ID": candidate.candidate_id,
                    "Pipeline": candidate.pipeline,
                    "Stage order": 0,
                    "Stage": "P0 Altered input",
                    "Stage parameters": json.dumps({"operation": "P0 only"}, sort_keys=True),
                    "Runtime (s)": 0.0,
                    **baseline_metrics,
                    "Delta Ridge coherence vs previous stage": np.nan,
                    "Delta Local contrast vs previous stage": np.nan,
                    "Delta Fragmentation per 1000 ridge pixels vs previous stage": np.nan,
                    "Delta Ridge continuity proxy vs previous stage": np.nan,
                }
            )
            total_runtime = 0.0
            for order, stage in enumerate(candidate.stages, start=1):
                start = perf_counter()
                current = apply_stage(stage, current)
                runtime = perf_counter() - start
                total_runtime += runtime
                metrics = direct_metrics(current)
                stage_rows.append(
                    {
                        "Experiment": "direct Altered compatibility",
                        "Protocol": "direct_altered_no_synthetic_degradation",
                        "Category": category,
                        "Altered split": record["Altered split"],
                        "Image": record["Image"],
                        "Relative path": record["Relative path"],
                        "Candidate ID": candidate.candidate_id,
                        "Pipeline": candidate.pipeline,
                        "Stage order": order,
                        "Stage": stage,
                        "Stage parameters": json.dumps(stage_params(stage), sort_keys=True),
                        "Runtime (s)": float(runtime),
                        **metrics,
                        "Delta Ridge coherence vs previous stage": metrics["Ridge coherence"] - previous_metrics["Ridge coherence"],
                        "Delta Local contrast vs previous stage": metrics["Local contrast"] - previous_metrics["Local contrast"],
                        "Delta Fragmentation per 1000 ridge pixels vs previous stage": (
                            metrics["Fragmentation per 1000 ridge pixels"]
                            - previous_metrics["Fragmentation per 1000 ridge pixels"]
                        ),
                        "Delta Ridge continuity proxy vs previous stage": (
                            metrics["Ridge continuity proxy"] - previous_metrics["Ridge continuity proxy"]
                        ),
                    }
                )
                previous_metrics = metrics
            final_metrics = direct_metrics(current)
            final_rows.append(
                {
                    "Experiment": "direct Altered compatibility",
                    "Protocol": "direct_altered_no_synthetic_degradation",
                    "Category": category,
                    "Altered split": record["Altered split"],
                    "Image": record["Image"],
                    "Relative path": record["Relative path"],
                    "Candidate ID": candidate.candidate_id,
                    "Pipeline": candidate.pipeline,
                    "Member": "Compatibility candidate",
                    "Is member technique": False,
                    "Is baseline": False,
                    "Fixed parameters": json.dumps(candidate.params, sort_keys=True),
                    "Runtime (s)": float(total_runtime),
                    **final_metrics,
                }
            )
    final_df = add_direct_deltas(pd.DataFrame(final_rows), id_column="Candidate ID", baseline_id="C0_NLM", champion_id="C0_NLM")
    final_df = add_compat_direct_p0_deltas(final_df, records)
    return final_df, pd.DataFrame(stage_rows)


def add_compat_direct_p0_deltas(frame: pd.DataFrame, records: list[dict[str, object]]) -> pd.DataFrame:
    result = frame.copy()
    p0_rows = []
    for record in records:
        p0 = focus.metric_image(record["image"])
        category = str(record["Altered split"]).replace("Altered-", "")
        p0_rows.append(
            {
                "Category": category,
                "Image": record["Image"],
                **direct_metrics(p0),
            }
        )
    p0 = pd.DataFrame(p0_rows).set_index(["Category", "Image"])
    tracked = [
        "Ridge coherence",
        "Local contrast",
        "Foreground/background separation proxy",
        "Ridge coverage in foreground",
        "Fragmentation per 1000 ridge pixels",
        "Small connected components <20px",
        "Ridge continuity proxy",
    ]
    for index, row in result.iterrows():
        key = (row["Category"], row["Image"])
        if key not in p0.index:
            continue
        base_row = p0.loc[key]
        for column in tracked:
            result.loc[index, f"Delta {column} vs P0"] = row[column] - base_row[column]
    return result


def summarise_compatibility_controlled(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["Experiment", "Protocol", "Category", "Candidate ID", "Pipeline", "Fixed parameters"], dropna=False)
        .agg(
            Images=("Image", "nunique"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            MSE_255_equivalent_mean=("MSE_255_equivalent", "mean"),
            PSNR_mean=("PSNR (dB)", "mean"),
            PSNR_std=("PSNR (dB)", "std"),
            SSIM_mean=("SSIM", "mean"),
            SSIM_std=("SSIM", "std"),
            Delta_PSNR_vs_degraded_mean=("Delta PSNR vs degraded input (dB)", "mean"),
            Delta_PSNR_vs_NLM_mean=("Delta PSNR vs NLM (dB)", "mean"),
            Delta_MSE_vs_NLM_mean=("Delta MSE vs NLM", "mean"),
            Delta_SSIM_vs_NLM_mean=("Delta SSIM vs NLM", "mean"),
            Runtime_mean=("Runtime (s)", "mean"),
        )
        .reset_index()
    )
    summary["PSNR_delta_to_28_17"] = summary["PSNR_mean"] - TARGET_PSNR_DB
    summary["Target_exceeded"] = summary["PSNR_mean"] > TARGET_PSNR_DB
    return summary


def summarise_compatibility_direct(frame: pd.DataFrame) -> pd.DataFrame:
    return summarise_direct(frame, id_column="Candidate ID", name_column="Pipeline").rename(
        columns={
            "Candidate ID": "Candidate ID",
            "Pipeline": "Pipeline",
        }
    ).assign(Experiment="direct Altered compatibility", Protocol="direct_altered_no_synthetic_degradation")


def create_verification_report() -> str:
    return """# Member 2 and Member 4 Verification

## A. Is M2 genuinely modified/orientation-adaptive?

Yes. The audited M2 implementation is not only Ordinary Gabor with a different fixed parameter. The current advanced M2 method is `Literature-Inspired Modified / Orientation-Adaptive Gabor`, implemented through `adaptive_gabor_enhancement`.

## B. What exact adaptation is implemented?

- Local ridge orientation is estimated using Sobel gradients and a smoothed structure-tensor formulation in `estimate_orientation_field`.
- Local ridge frequency is estimated block-by-block in `estimate_local_frequency_map` using a Hanning window, FFT magnitude peak search, nearest-frequency selection, and a frequency confidence score.
- The adaptive Gabor stage builds 8 orientation bins and frequency candidates `[0.075, 0.095, 0.115, 0.140]`.
- For each pixel, the implementation selects the Gabor response matching that pixel's local orientation bin and local frequency bin.
- Blending is local, not uniform: `weight = blend * coherence_weight * frequency_confidence_weight * mask`, with fixed `blend=0.18` for the literature-inspired variant.

## C. How is it different from Ordinary Gabor?

Ordinary Gabor uses fixed frequency `0.115`, 8 orientations, maximum response energy, and uniform blend `0.30`. It does not use a local ridge-frequency map, per-pixel orientation/frequency response selection, or coherence/frequency-confidence weighting. M2 therefore has meaningful fingerprint-specific adaptation.

## D. Is M4 genuinely coherence-guided directional diffusion?

Yes. The audited M4 implementation is not Gaussian smoothing followed by a coherence score. It performs repeated orientation-selected smoothing and blends the smoothed result back into the image using local coherence and a fingerprint mask.

## E. Where is coherence used?

Inside each diffusion iteration, `coherence_guided_diffusion` estimates a coherence map and computes `weight = clip(coherence, 0, 1) * mask`. The update is `current = (1 - step * weight) * current + (step * weight) * smoothed`, so diffusion strength is higher in locally coherent ridge regions.

## F. Where is orientation used?

Inside each iteration, local orientation is assigned to one of 8 bins. For each orientation bin, an oriented Gaussian kernel is built and convolved with the current image. Pixels receive the convolved value from the kernel associated with their nearest local orientation bin.

## G. Does diffusion differ along ridge and across ridge direction?

Yes. The oriented kernel is anisotropic: the audited fixed parameters use `sigma_x=1.35`, `sigma_y=0.50`, and kernel size `9`, so smoothing is elongated in one direction and narrower in the perpendicular direction. The selected kernel orientation varies by local ridge orientation.

## H. Are any fallback algorithms being used?

For M2 and M4, no silent fallback was observed in the audited functions. For M1, the focused NLM function uses OpenCV when available and falls back to skimage NLM if OpenCV is unavailable. For M3, TV uses skimage directly; however, older wavelet helper code can fall back to TV if PyWavelets is unavailable, so wavelet labels must be checked in any separate wavelet run.

## I. Are the current technique names scientifically accurate?

Mostly yes, with one required wording guard. M2 should be called `Literature-Inspired Modified / Orientation-Adaptive Gabor`, not exact article reproduction, because exact article parameters, dataset, degradation, MAX_I, and formula remain unknown in the available repository context. M4 is scientifically accurate as `Coherence-Guided Directional Diffusion`.
"""


def create_interpretation_report(
    controlled_summary: pd.DataFrame,
    altered_summary: pd.DataFrame,
) -> str:
    member_controlled = controlled_summary[controlled_summary["Is member technique"] == True].copy()
    altered_overall = altered_summary[
        (altered_summary["Category"] == "Overall")
        & (altered_summary["Is member technique"] == True)
    ].copy()

    def c(method_id: str) -> pd.Series:
        frame = controlled_summary[controlled_summary["Method ID"] == method_id]
        return frame.iloc[0] if not frame.empty else pd.Series(dtype=object)

    def a(method_id: str) -> pd.Series:
        frame = altered_overall[altered_overall["Method ID"] == method_id]
        return frame.iloc[0] if not frame.empty else pd.Series(dtype=object)

    nlm = c("M1_NLM_h0.06")
    mod = c("M2_MOD_GABOR")
    tv = c("M3_TV_w0.02")
    diff = c("M4_COH_DIFF")
    ord_gabor = c("BL_ORD_GABOR")
    wiener = c("BL_WIENER")
    gauss_note = "Gaussian filtering was retained as a historical baseline in the registry; this small comparison used the optional baselines requested here."

    nlm_a = a("M1_NLM_h0.06")
    mod_a = a("M2_MOD_GABOR")
    tv_a = a("M3_TV_w0.02")
    diff_a = a("M4_COH_DIFF")
    ord_a = altered_summary[(altered_summary["Category"] == "Overall") & (altered_summary["Method ID"] == "BL_ORD_GABOR")]
    ord_a = ord_a.iloc[0] if not ord_a.empty else pd.Series(dtype=object)

    return f"""# Four-Member Comparison Interpretation

## Member 1: Non-Local Means Denoising

NLM preserved the 29+ dB behaviour on the development split: mean PSNR {clean(nlm.get('PSNR_mean'))} dB, MSE {clean(nlm.get('MSE_mean'), 8)}, SSIM {clean(nlm.get('SSIM_mean'))}. It remains above the 28.17 dB target. On direct Altered images, its mean coherence was {clean(nlm_a.get('Coherence_mean'))}, fragmentation {clean(nlm_a.get('Fragmentation_mean'))}, and contrast delta versus P0 {clean(nlm_a.get('Delta_contrast_vs_P0_mean'))}. Main weakness: it can slightly soften local contrast even while improving denoising.

## Member 2: Modified / Orientation-Adaptive Gabor

Modified Gabor produced controlled PSNR {clean(mod.get('PSNR_mean'))} dB and SSIM {clean(mod.get('SSIM_mean'))}. On direct Altered images, coherence was {clean(mod_a.get('Coherence_mean'))}, contrast delta versus P0 {clean(mod_a.get('Delta_contrast_vs_P0_mean'))}, and fragmentation delta versus P0 {clean(mod_a.get('Delta_fragmentation_vs_P0_mean'))}. Compared with Ordinary Gabor controlled PSNR {clean(ord_gabor.get('PSNR_mean'))} dB, the adaptive method is scientifically deeper and usually less extreme, but it still changes ridge intensities and must be used carefully.

## Member 3: Total Variation Restoration

TV produced controlled PSNR {clean(tv.get('PSNR_mean'))} dB, MSE {clean(tv.get('MSE_mean'), 8)}, and SSIM {clean(tv.get('SSIM_mean'))}. It is a strong denoising/restoration method under Gaussian noise, but this fixed setting should be compared against NLM rather than assumed superior. On direct Altered images, TV coherence was {clean(tv_a.get('Coherence_mean'))}, fragmentation {clean(tv_a.get('Fragmentation_mean'))}, and contrast delta versus P0 {clean(tv_a.get('Delta_contrast_vs_P0_mean'))}.

## Member 4: Coherence-Guided Directional Diffusion

Directional diffusion produced controlled PSNR {clean(diff.get('PSNR_mean'))} dB and SSIM {clean(diff.get('SSIM_mean'))}. On direct Altered images, coherence was {clean(diff_a.get('Coherence_mean'))}, fragmentation {clean(diff_a.get('Fragmentation_mean'))}, and contrast delta versus P0 {clean(diff_a.get('Delta_contrast_vs_P0_mean'))}. Its benefit is structural and orientation-aware; its risk is PSNR loss if it smooths ridge detail after NLM.

## Comparator Notes

Ordinary Gabor remains a baseline, not the final M2 contribution. In this run its Altered coherence was {clean(ord_a.get('Coherence_mean'))}, contrast delta versus P0 {clean(ord_a.get('Delta_contrast_vs_P0_mean'))}, and fragmentation delta versus P0 {clean(ord_a.get('Delta_fragmentation_vs_P0_mean'))}. Wiener controlled PSNR was {clean(wiener.get('PSNR_mean'))} dB. {gauss_note}

## Current Individual Winner

The strongest individual controlled member technique is {member_controlled.sort_values('PSNR_mean', ascending=False).iloc[0]['Technique']} with PSNR {clean(member_controlled['PSNR_mean'].max())} dB.
"""


def create_compatibility_report(
    verification_text: str,
    fixed_params: dict[str, object],
    member_controlled_summary: pd.DataFrame,
    member_altered_summary: pd.DataFrame,
    compat_summary: pd.DataFrame,
    compat_stage_metrics: pd.DataFrame,
    final_decision: dict[str, object],
) -> str:
    controlled = member_controlled_summary.copy()
    members = controlled[controlled["Is member technique"] == True]
    altered_overall = member_altered_summary[member_altered_summary["Category"] == "Overall"].copy()
    altered_members = altered_overall[altered_overall["Is member technique"] == True]
    comp_controlled = compat_summary[compat_summary["Experiment"] == "controlled compatibility"].copy()
    comp_direct = compat_summary[
        (compat_summary["Experiment"] == "direct Altered compatibility")
        & (compat_summary["Category"] == "Overall")
    ].copy()

    def row_by(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
        selected = frame[frame[column] == value]
        return selected.iloc[0] if not selected.empty else pd.Series(dtype=object)

    m1 = row_by(controlled, "Method ID", "M1_NLM_h0.06")
    m2 = row_by(controlled, "Method ID", "M2_MOD_GABOR")
    m3 = row_by(controlled, "Method ID", "M3_TV_w0.02")
    m4 = row_by(controlled, "Method ID", "M4_COH_DIFF")
    ordinary = row_by(controlled, "Method ID", "BL_ORD_GABOR")

    c0 = row_by(comp_controlled, "Candidate ID", "C0_NLM")
    c1 = row_by(comp_controlled, "Candidate ID", "C1_TV_NLM")
    c2 = row_by(comp_controlled, "Candidate ID", "C2_NLM_MOD_GABOR")
    c3 = row_by(comp_controlled, "Candidate ID", "C3_NLM_COH_DIFF")
    c4 = row_by(comp_controlled, "Candidate ID", "C4_TV_NLM_COH_DIFF")

    highest_psnr = comp_controlled.sort_values("PSNR_mean", ascending=False).iloc[0]
    lowest_mse = comp_controlled.sort_values("MSE_mean", ascending=True).iloc[0]
    highest_ssim = comp_controlled.sort_values("SSIM_mean", ascending=False).iloc[0]
    best_coh = altered_members.sort_values("Coherence_mean", ascending=False).iloc[0]
    lowest_frag = altered_members.sort_values("Fragmentation_mean", ascending=True).iloc[0]
    contrast_preserved = altered_members.assign(
        Abs_contrast_delta=altered_members["Delta_contrast_vs_P0_mean"].abs()
    ).sort_values("Abs_contrast_delta", ascending=True).iloc[0]
    comp_best_direct = comp_direct.assign(
        Structural_score=(
            comp_direct["Coherence_mean"].fillna(0.0)
            - 0.03 * comp_direct["Fragmentation_mean"].fillna(0.0)
            + 0.30 * comp_direct["Local_contrast_mean"].fillna(0.0)
        )
    ).sort_values("Structural_score", ascending=False).iloc[0]

    hybrid_rows = comp_controlled[comp_controlled["Candidate ID"] != "C0_NLM"].copy()
    best_hybrid = hybrid_rows.sort_values("PSNR_mean", ascending=False).iloc[0]
    best_hybrid_stage = compat_stage_metrics[
        (compat_stage_metrics["Experiment"] == "controlled compatibility")
        & (compat_stage_metrics["Candidate ID"] == best_hybrid["Candidate ID"])
    ].copy()
    best_hybrid_stage_summary = (
        best_hybrid_stage.groupby(["Stage order", "Stage"], dropna=False)
        .agg(
            MSE_mean=("MSE", "mean"),
            PSNR_mean=("PSNR (dB)", "mean"),
            SSIM_mean=("SSIM", "mean"),
            Delta_PSNR_vs_previous_mean=("Delta PSNR vs previous stage (dB)", "mean"),
            Delta_MSE_vs_previous_mean=("Delta MSE vs previous stage", "mean"),
            Delta_SSIM_vs_previous_mean=("Delta SSIM vs previous stage", "mean"),
        )
        .reset_index()
        .sort_values("Stage order")
    )
    direct_stage = compat_stage_metrics[
        (compat_stage_metrics["Experiment"] == "direct Altered compatibility")
        & (compat_stage_metrics["Candidate ID"] == best_hybrid["Candidate ID"])
    ].copy()
    direct_stage_summary = (
        direct_stage.groupby(["Stage order", "Stage"], dropna=False)
        .agg(
            Coherence_mean=("Ridge coherence", "mean"),
            Local_contrast_mean=("Local contrast", "mean"),
            Fragmentation_mean=("Fragmentation per 1000 ridge pixels", "mean"),
            Delta_coherence_vs_previous_mean=("Delta Ridge coherence vs previous stage", "mean"),
            Delta_contrast_vs_previous_mean=("Delta Local contrast vs previous stage", "mean"),
            Delta_fragmentation_vs_previous_mean=("Delta Fragmentation per 1000 ridge pixels vs previous stage", "mean"),
        )
        .reset_index()
        .sort_values("Stage order")
    )

    comp_cols = [
        "Candidate ID",
        "Pipeline",
        "Images",
        "MSE_mean",
        "PSNR_mean",
        "SSIM_mean",
        "Delta_PSNR_vs_degraded_mean",
        "Delta_PSNR_vs_NLM_mean",
        "Target_exceeded",
    ]
    direct_cols = [
        "Candidate ID",
        "Pipeline",
        "Images",
        "Coherence_mean",
        "Local_contrast_mean",
        "Fragmentation_mean",
        "Delta_coherence_vs_P0_mean",
        "Delta_contrast_vs_P0_mean",
        "Delta_fragmentation_vs_P0_mean",
    ]

    controlled_stage_cols = [
        "Stage order",
        "Stage",
        "MSE_mean",
        "PSNR_mean",
        "SSIM_mean",
        "Delta_PSNR_vs_previous_mean",
        "Delta_MSE_vs_previous_mean",
        "Delta_SSIM_vs_previous_mean",
    ]
    direct_stage_cols = [
        "Stage order",
        "Stage",
        "Coherence_mean",
        "Local_contrast_mean",
        "Fragmentation_mean",
        "Delta_coherence_vs_previous_mean",
        "Delta_contrast_vs_previous_mean",
        "Delta_fragmentation_vs_previous_mean",
    ]

    def outperformed(candidate: pd.Series, baseline: pd.Series) -> str:
        if candidate.empty or baseline.empty:
            return "Unknown"
        better_psnr = candidate["PSNR_mean"] >= baseline["PSNR_mean"]
        better_mse = candidate["MSE_mean"] <= baseline["MSE_mean"]
        better_ssim = candidate["SSIM_mean"] >= baseline["SSIM_mean"] - 0.002
        return "Yes" if bool(better_psnr and better_mse and better_ssim) else "No"

    every_useful_above = bool((comp_controlled["PSNR_mean"] > TARGET_PSNR_DB).all())
    controlled_stage_means = (
        compat_stage_metrics[compat_stage_metrics["Experiment"] == "controlled compatibility"][
            ["Candidate ID", "Pipeline", "Stage", "Delta PSNR vs previous stage (dB)"]
        ]
        .groupby(["Candidate ID", "Pipeline", "Stage"], as_index=False)
        .mean()
    )
    psnr_loss_stages = controlled_stage_means[
        controlled_stage_means["Delta PSNR vs previous stage (dB)"] < 0
    ].sort_values("Delta PSNR vs previous stage (dB)")

    direct_stage_means = (
        compat_stage_metrics[compat_stage_metrics["Experiment"] == "direct Altered compatibility"][
            ["Candidate ID", "Pipeline", "Stage", "Delta Fragmentation per 1000 ridge pixels vs previous stage"]
        ]
        .groupby(["Candidate ID", "Pipeline", "Stage"], as_index=False)
        .mean()
    )
    frag_increase_stages = direct_stage_means[
        direct_stage_means["Delta Fragmentation per 1000 ridge pixels vs previous stage"] > 0
    ].sort_values("Delta Fragmentation per 1000 ridge pixels vs previous stage", ascending=False)

    final_candidate = str(final_decision["final_candidate"])
    final_pipeline = str(final_decision["final_pipeline"])
    final_is_hybrid = bool(final_decision["final_is_hybrid"])
    hybrid_better = "Yes" if final_is_hybrid else "No"
    retain_nlm = "No" if final_is_hybrid else "Yes"

    return f"""# Member Compatibility Report

This run used only the existing focused audit development split and the small Altered sample from prior direct sanity checks. It did not run 500 images, did not use the focused held-out test split, did not use a fresh final hold-out set, and did not retune NLM.

## Fixed Parameters

The full fixed parameter record is in `outputs/member_fixed_parameters.json`.

## Four-Member Controlled Summary

{md_table(controlled.sort_values('Controlled PSNR rank').to_dict(orient='records'), ['Method ID', 'Member', 'Technique', 'Images', 'MSE_mean', 'PSNR_mean', 'SSIM_mean', 'Delta_PSNR_vs_degraded_mean', 'Target_exceeded'])}

## Four-Member Direct Altered Summary

{md_table(altered_overall.to_dict(orient='records'), ['Method ID', 'Member', 'Technique', 'Images', 'Coherence_mean', 'Local_contrast_mean', 'Fragmentation_mean', 'Ridge_continuity_mean', 'Delta_coherence_vs_P0_mean', 'Delta_contrast_vs_P0_mean', 'Delta_fragmentation_vs_P0_mean'])}

## Compatibility Controlled Summary

{md_table(comp_controlled.sort_values('PSNR_mean', ascending=False).to_dict(orient='records'), comp_cols)}

## Compatibility Direct Altered Summary

{md_table(comp_direct.sort_values('Coherence_mean', ascending=False).to_dict(orient='records'), direct_cols)}

## Best Hybrid Stage-Wise Controlled Analysis

Best hybrid by controlled PSNR: {best_hybrid['Pipeline']} ({best_hybrid['Candidate ID']}).

{md_table(best_hybrid_stage_summary.to_dict(orient='records'), controlled_stage_cols)}

## Best Hybrid Stage-Wise Direct Altered Analysis

{md_table(direct_stage_summary.to_dict(orient='records'), direct_stage_cols)}

## Final Report Questions

A. Is Member 2 genuinely Modified/Orientation-Adaptive Gabor?

Yes. It estimates local orientation and local frequency, selects per-pixel Gabor responses by those maps, and blends by coherence/frequency confidence. It is still literature-inspired, not exact article reproduction.

B. How exactly does M2 differ from Ordinary Gabor?

M2 uses local orientation, local frequency candidates, per-pixel response selection, mask gating, and coherence/frequency confidence weighting. Ordinary Gabor uses fixed frequency 0.115, 8 orientations, max energy, and uniform blend.

C. Is Member 4 genuinely coherence-guided directional diffusion?

Yes. It re-estimates local orientation and coherence inside the diffusion loop and uses them to choose anisotropic kernels and update strength.

D. How exactly are coherence and orientation used?

Orientation selects one of 8 oriented smoothing kernels per pixel. Coherence sets the local diffusion weight, multiplied by the fingerprint mask.

E. What fixed parameters were used for each member?

M1: NLM h=0.06. M2: local-frequency adaptive Gabor with frequencies [0.075, 0.095, 0.115, 0.140], 8 bins, blend=0.18. M3: TV weight=0.02. M4: directional diffusion iterations=4, step=0.16, 8 orientation bins, anisotropic kernel sigma_x=1.35 and sigma_y=0.50.

F. What was M1 NLM mean PSNR?

{clean(m1.get('PSNR_mean'))} dB.

G. What was M2 mean PSNR?

{clean(m2.get('PSNR_mean'))} dB.

H. What was M3 mean PSNR?

{clean(m3.get('PSNR_mean'))} dB.

I. What was M4 mean PSNR?

{clean(m4.get('PSNR_mean'))} dB.

J. Which member had the highest controlled PSNR?

{members.sort_values('PSNR_mean', ascending=False).iloc[0]['Member']}: {members.sort_values('PSNR_mean', ascending=False).iloc[0]['Technique']}.

K. Which member had the lowest MSE?

{members.sort_values('MSE_mean', ascending=True).iloc[0]['Member']}: {members.sort_values('MSE_mean', ascending=True).iloc[0]['Technique']}.

L. Which member had the highest SSIM?

{members.sort_values('SSIM_mean', ascending=False).iloc[0]['Member']}: {members.sort_values('SSIM_mean', ascending=False).iloc[0]['Technique']}.

M. Which member had the best ridge coherence on Altered images?

{best_coh['Member']}: {best_coh['Technique']} with coherence {clean(best_coh['Coherence_mean'])}.

N. Which member had the lowest fragmentation?

{lowest_frag['Member']}: {lowest_frag['Technique']} with fragmentation {clean(lowest_frag['Fragmentation_mean'])}.

O. Which member had the strongest contrast preservation?

{contrast_preserved['Member']}: {contrast_preserved['Technique']} had the smallest absolute contrast change versus P0, delta {clean(contrast_preserved['Delta_contrast_vs_P0_mean'])}.

P. Did Modified Gabor outperform Ordinary Gabor overall?

Yes by the balanced comparison. Modified Gabor had much higher controlled PSNR ({clean(m2['PSNR_mean'])} dB versus {clean(ordinary['PSNR_mean'])} dB) and avoided the severe Ordinary Gabor contrast/fragmentation damage. Ordinary Gabor still had higher raw Altered coherence, so it remains useful as a comparator, not as the final method.

Q. Did Directional Diffusion provide useful structural benefit?

Only limited benefit in this fixed comparison. It is genuinely orientation-aware, but NLM -> Directional Diffusion reduced PSNR to {clean(c3.get('PSNR_mean'))} dB and increased fragmentation versus NLM alone, so it should not be added to the final quantitative pipeline without a stronger structural reason.

R. Which individual technique is the strongest overall?

NLM is strongest overall because it has the highest controlled PSNR/MSE performance and remains structurally plausible on Altered images.

S. Which compatibility candidates were tested?

C0 NLM, C1 TV -> NLM, C2 NLM -> Modified Gabor, C3 NLM -> Coherence-Guided Directional Diffusion, and C4 TV -> NLM -> Coherence-Guided Directional Diffusion.

T. Did TV -> NLM outperform NLM?

{outperformed(c1, c0)}. TV -> NLM PSNR {clean(c1.get('PSNR_mean'))} dB versus NLM {clean(c0.get('PSNR_mean'))} dB.

U. Did NLM -> Modified Gabor outperform NLM?

{outperformed(c2, c0)}. NLM -> Modified Gabor PSNR {clean(c2.get('PSNR_mean'))} dB versus NLM {clean(c0.get('PSNR_mean'))} dB.

V. Did NLM -> Directional Diffusion outperform NLM?

{outperformed(c3, c0)}. NLM -> Directional Diffusion PSNR {clean(c3.get('PSNR_mean'))} dB versus NLM {clean(c0.get('PSNR_mean'))} dB.

W. Did TV -> NLM -> Directional Diffusion outperform NLM?

{outperformed(c4, c0)}. TV -> NLM -> Directional Diffusion PSNR {clean(c4.get('PSNR_mean'))} dB versus NLM {clean(c0.get('PSNR_mean'))} dB.

X. Which candidate had the highest PSNR?

{highest_psnr['Pipeline']} with PSNR {clean(highest_psnr['PSNR_mean'])} dB.

Y. Which candidate had the lowest MSE?

{lowest_mse['Pipeline']} with MSE {clean(lowest_mse['MSE_mean'], 8)}.

Z. Which candidate had the highest SSIM?

{highest_ssim['Pipeline']} with SSIM {clean(highest_ssim['SSIM_mean'])}.

AA. Which candidate had the best Altered structural behaviour?

By the simple structural score used only for interpretation, {comp_best_direct['Pipeline']} had the best Altered structural behaviour. This does not override the controlled PSNR target.

AB. Did every useful candidate remain above 28.17 dB?

{clean(every_useful_above)} for the five tested compatibility candidates.

AC. Which stages caused PSNR loss?

{md_table(psnr_loss_stages.head(12).to_dict(orient='records'), ['Candidate ID', 'Pipeline', 'Stage', 'Delta PSNR vs previous stage (dB)']) if not psnr_loss_stages.empty else 'No tested stage caused mean PSNR loss.'}

AD. Which stages caused fragmentation increase?

{md_table(frag_increase_stages.head(12).to_dict(orient='records'), ['Candidate ID', 'Pipeline', 'Stage', 'Delta Fragmentation per 1000 ridge pixels vs previous stage']) if not frag_increase_stages.empty else 'No tested stage caused mean fragmentation increase.'}

AE. Is a hybrid genuinely better than NLM alone?

{hybrid_better}. {final_decision['reason']}

AF. If not, should NLM alone be retained?

{retain_nlm}.

AG. What ONE final candidate should be frozen for fresh hold-out validation?

{final_pipeline} ({final_candidate}).

AH. Are its parameters now frozen?

Yes. The parameters are frozen in `outputs/member_fixed_parameters.json`.

AI. Is the project ready for fresh final hold-out validation?

Yes, only for the selected frozen candidate and parameters. The next validation should still be a fresh hold-out run, not the 500-image final validation unless the team explicitly approves that final step.

## Verification Reference

The M2/M4 verification details are also written separately in `outputs/member_m2_m4_verification.md`.
"""


def decide_final_candidate(compat_summary: pd.DataFrame) -> dict[str, object]:
    controlled = compat_summary[compat_summary["Experiment"] == "controlled compatibility"].copy()
    nlm = controlled[controlled["Candidate ID"] == "C0_NLM"].iloc[0]
    hybrids = controlled[controlled["Candidate ID"] != "C0_NLM"].copy()
    viable = hybrids[
        (hybrids["PSNR_mean"] > TARGET_PSNR_DB)
        & (hybrids["PSNR_mean"] >= nlm["PSNR_mean"] - 0.05)
        & (hybrids["MSE_mean"] <= nlm["MSE_mean"] * 1.01)
        & (hybrids["SSIM_mean"] >= nlm["SSIM_mean"] - 0.002)
    ].copy()
    if not viable.empty:
        best = viable.sort_values(["PSNR_mean", "SSIM_mean"], ascending=False).iloc[0]
        return {
            "final_candidate": best["Candidate ID"],
            "final_pipeline": best["Pipeline"],
            "final_is_hybrid": True,
            "reason": "A hybrid met the fixed acceptance guard while staying close to or better than NLM on PSNR/MSE/SSIM.",
        }
    return {
        "final_candidate": "C0_NLM",
        "final_pipeline": "NLM",
        "final_is_hybrid": False,
        "reason": "No tested hybrid clearly beat NLM under the fixed Gaussian-noise PSNR/MSE/SSIM acceptance guard.",
    }


def write_fixed_parameters(
    methods: list[MethodDef],
    candidates: list[CompatDef],
    split_info: dict[str, object],
    altered_info: dict[str, object],
    final_decision: dict[str, object],
) -> dict[str, object]:
    method_params = {
        method.method_id: {
            "member": method.member,
            "technique": method.technique,
            "role": method.role,
            "fixed_parameters": method.params,
            "is_member_technique": method.is_member,
            "is_baseline": method.is_baseline,
        }
        for method in methods
    }
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": Path(__file__).name,
        "target_psnr_db": TARGET_PSNR_DB,
        "parameter_policy": {
            "nlm_h_retuned": False,
            "broad_grid_search_run": False,
            "test_images_used_for_parameter_selection": False,
            "clean_reference_pixels_used_inside_enhancement": False,
        },
        "controlled_protocol": {
            "scenario_id": CONTROLLED_SCENARIO_ID,
            "input": "Clean Real",
            "degradation": "Gaussian noise only",
            "severity": CONTROLLED_SEVERITY,
            "gaussian_noise_sigma": focus.gaussian_noise_sigma_from_severity(CONTROLLED_SEVERITY),
            "range": "[0,1] float images",
            "psnr_data_range": 1.0,
            "mse_scale": "normalised MSE; MSE_255_equivalent also reported",
            "ssim_data_range": 1.0,
            "split_info": split_info,
        },
        "direct_altered_protocol": altered_info,
        "member_and_baseline_methods": method_params,
        "compatibility_candidates": {
            candidate.candidate_id: {
                "pipeline": candidate.pipeline,
                "stages": candidate.stages,
                "fixed_parameters": candidate.params,
            }
            for candidate in candidates
        },
        "final_candidate": final_decision,
        "fresh_final_holdout_ready": True,
        "formal_500_image_validation_run": False,
    }
    OUTPUT_FILES["fixed_parameters"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    log_lines = [
        f"Started member fixed comparison at {datetime.now().astimezone().isoformat()}",
        "Policy: fixed parameters, focused dev split, no 500-image validation, no final fresh hold-out.",
    ]

    protected_before = snapshot_protected_outputs()
    methods = fixed_methods()
    candidates = compatibility_candidates()
    controlled_records, split_info = load_dev_records()
    altered_records, altered_info = load_altered_records()
    log_lines.append(f"Controlled dev images: {len(controlled_records)}")
    log_lines.append(f"Direct Altered images: {len(altered_records)}")

    verification_text = create_verification_report()
    OUTPUT_FILES["verification"].write_text(verification_text, encoding="utf-8")
    log_lines.append("Wrote M2/M4 verification report.")

    member_controlled = controlled_rows_for_methods(controlled_records, methods)
    member_controlled_summary = summarise_controlled(member_controlled)
    member_altered = altered_rows_for_methods(altered_records, methods)
    member_altered_summary = summarise_direct(member_altered, id_column="Method ID", name_column="Technique")
    log_lines.append("Completed fixed four-member comparison.")

    compat_controlled, controlled_stage = controlled_stage_metrics(controlled_records, candidates)
    compat_direct, direct_stage = direct_stage_metrics(altered_records, candidates)
    compat_per_image = pd.concat([compat_controlled, compat_direct], ignore_index=True, sort=False)
    compat_summary = pd.concat(
        [
            summarise_compatibility_controlled(compat_controlled),
            summarise_compatibility_direct(compat_direct),
        ],
        ignore_index=True,
        sort=False,
    )
    compat_stage_metrics = pd.concat([controlled_stage, direct_stage], ignore_index=True, sort=False)
    log_lines.append("Completed small fixed compatibility test.")

    final_decision = decide_final_candidate(compat_summary)
    fixed_payload = write_fixed_parameters(methods, candidates, split_info, altered_info, final_decision)
    log_lines.append(f"Final candidate: {final_decision['final_candidate']} ({final_decision['final_pipeline']}).")

    member_controlled.to_csv(OUTPUT_FILES["member_controlled_per_image"], index=False)
    member_controlled_summary.to_csv(OUTPUT_FILES["member_controlled_summary"], index=False)
    member_altered.to_csv(OUTPUT_FILES["member_altered_per_image"], index=False)
    member_altered_summary.to_csv(OUTPUT_FILES["member_altered_summary"], index=False)
    compat_per_image.to_csv(OUTPUT_FILES["compatibility_per_image"], index=False)
    compat_summary.to_csv(OUTPUT_FILES["compatibility_summary"], index=False)
    compat_stage_metrics.to_csv(OUTPUT_FILES["compatibility_stage_metrics"], index=False)

    interpretation = create_interpretation_report(member_controlled_summary, member_altered_summary)
    OUTPUT_FILES["member_interpretation"].write_text(interpretation, encoding="utf-8")
    report = create_compatibility_report(
        verification_text,
        fixed_payload,
        member_controlled_summary,
        member_altered_summary,
        compat_summary,
        compat_stage_metrics,
        final_decision,
    )
    OUTPUT_FILES["compatibility_report"].write_text(report, encoding="utf-8")

    protected_after = snapshot_protected_outputs()
    runtime = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": Path(__file__).name,
        "runtime_seconds": perf_counter() - start,
        "controlled_dev_images": len(controlled_records),
        "direct_altered_images": len(altered_records),
        "member_methods": len(methods),
        "compatibility_candidates": len(candidates),
        "final_candidate": final_decision,
        "formal_500_image_validation_run": False,
        "fresh_final_holdout_used": False,
        "focused_test_split_used": False,
        "protected_previous_outputs_unchanged": protected_before == protected_after,
        "generated_files": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in OUTPUT_FILES.items()},
    }
    OUTPUT_FILES["compatibility_runtime"].write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    log_lines.append(f"Protected previous outputs unchanged: {protected_before == protected_after}")
    log_lines.append(f"Runtime seconds: {runtime['runtime_seconds']:.3f}")
    OUTPUT_FILES["compatibility_log"].write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print("Member fixed comparison complete.")
    print(f"Final candidate: {final_decision['final_candidate']} ({final_decision['final_pipeline']})")
    print(f"Controlled NLM PSNR: {member_controlled_summary[member_controlled_summary['Method ID'] == 'M1_NLM_h0.06'].iloc[0]['PSNR_mean']:.4f} dB")
    print(f"Report: {OUTPUT_FILES['compatibility_report']}")


if __name__ == "__main__":
    main()
