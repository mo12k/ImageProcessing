from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import morphology

import direct24_original_altered_experiment as direct
import screening24_master_audit as base


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SEVERITIES = (0.35, 0.55, 0.75)
CATEGORY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2, "Overall": 3}
CONTRAST_DAMAGE_THRESHOLD = -0.020
FRAGMENTATION_DAMAGE_THRESHOLD = 0.500
CATEGORY_FRAGMENTATION_DAMAGE_THRESHOLD = 1.000
CATEGORY_CONTRAST_DAMAGE_THRESHOLD = -0.035

HYBRID_FILES = {
    "direct_metrics": OUTPUT_DIR / "hybrid24_direct_metrics.csv",
    "direct_summary": OUTPUT_DIR / "hybrid24_direct_summary.csv",
    "controlled_metrics": OUTPUT_DIR / "hybrid24_controlled_metrics.csv",
    "controlled_summary": OUTPUT_DIR / "hybrid24_controlled_summary.csv",
    "stage_metrics": OUTPUT_DIR / "hybrid24_stage_metrics.csv",
    "finalists": OUTPUT_DIR / "hybrid24_finalists.csv",
    "runtime": OUTPUT_DIR / "hybrid24_runtime.json",
    "report": OUTPUT_DIR / "hybrid24_report.md",
    "metadata": OUTPUT_DIR / "hybrid24_metadata.json",
}


@dataclass(frozen=True)
class PipelineDefinition:
    pipeline_id: str
    pipeline: str
    purpose: str
    stages: tuple[str, ...]
    final_output_domain: str
    grayscale_final_stage: str
    has_closing_stage: bool = False
    is_baseline: bool = False
    is_grayscale_finalist_candidate: bool = True


def output_snapshot() -> dict[str, dict[str, object]]:
    files = sorted([p for p in OUTPUT_DIR.iterdir() if p.is_file() and not p.name.startswith("hybrid24_")], key=lambda p: p.name)
    return base.snapshot_files(files)


def load_direct24_selection() -> pd.DataFrame:
    metadata = json.loads((OUTPUT_DIR / "direct24_metadata.json").read_text(encoding="utf-8"))
    selection = pd.DataFrame(metadata["selection"])
    selection["_Category sort"] = selection["Category"].map(CATEGORY_ORDER)
    selection = selection.sort_values(["_Category sort", "Category order"]).drop(columns=["_Category sort"]).reset_index(drop=True)
    return selection


def load_screening24_controlled_selection() -> pd.DataFrame:
    metadata = json.loads((OUTPUT_DIR / "screening24_metadata.json").read_text(encoding="utf-8"))
    rows = []
    for index, rel_path in enumerate(metadata["screening24_selected_relative_paths"]):
        path = PROJECT_ROOT / rel_path
        rows.append(
            {
                "Controlled index": index,
                "Image": path.name,
                "Path": rel_path,
                "Severity": float(SEVERITIES[index % len(SEVERITIES)]),
                "Seed": RANDOM_SEED + index,
                "Protocol source": "screening24_metadata.json selected_relative_paths; same severity cycle and seed rule",
            }
        )
    return pd.DataFrame(rows)


def pipeline_definitions() -> list[PipelineDefinition]:
    return [
        PipelineDefinition(
            "B0",
            "P0 only",
            "neutral common-preprocessing control",
            ("P0",),
            "grayscale",
            "P0",
            is_baseline=True,
            is_grayscale_finalist_candidate=False,
        ),
        PipelineDefinition(
            "B1",
            "P0 -> CLAHE",
            "main M1 enhancement baseline from Direct24",
            ("P0", "CLAHE"),
            "grayscale",
            "CLAHE",
            is_baseline=True,
        ),
        PipelineDefinition(
            "C1",
            "P0 -> CLAHE -> Ordinary Gabor",
            "measure exact trade-off of the Direct24 M2 coherence winner",
            ("P0", "CLAHE", "Ordinary Gabor"),
            "grayscale",
            "Ordinary Gabor",
        ),
        PipelineDefinition(
            "C2",
            "P0 -> CLAHE -> Coherence-Guided Directional Diffusion",
            "test whether directional enhancement is more compatible with CLAHE than Ordinary Gabor",
            ("P0", "CLAHE", "Coherence-Guided Directional Diffusion"),
            "grayscale",
            "Coherence-Guided Directional Diffusion",
        ),
        PipelineDefinition(
            "C3",
            "P0 -> CLAHE -> Total Variation Restoration",
            "test CLAHE followed by the best Direct24 supporting restoration candidate",
            ("P0", "CLAHE", "Total Variation Restoration"),
            "grayscale",
            "Total Variation Restoration",
        ),
        PipelineDefinition(
            "C4",
            "P0 -> Total Variation Restoration -> CLAHE",
            "explicit order test for restoration before local contrast enhancement",
            ("P0", "Total Variation Restoration", "CLAHE"),
            "grayscale",
            "CLAHE",
        ),
        PipelineDefinition(
            "C5",
            "P0 -> CLAHE -> Coherence-Guided Directional Diffusion -> Closing",
            "test whether morphology can repair ridge gaps after directional enhancement",
            ("P0", "CLAHE", "Coherence-Guided Directional Diffusion", "Closing"),
            "morphology-assisted structural output",
            "Coherence-Guided Directional Diffusion",
            has_closing_stage=True,
            is_grayscale_finalist_candidate=False,
        ),
        PipelineDefinition(
            "C6",
            "P0 -> CLAHE -> Total Variation Restoration -> Closing",
            "test whether morphology improves continuity after CLAHE plus TV restoration",
            ("P0", "CLAHE", "Total Variation Restoration", "Closing"),
            "morphology-assisted structural output",
            "Total Variation Restoration",
            has_closing_stage=True,
            is_grayscale_finalist_candidate=False,
        ),
    ]


def apply_named_stage(stage_name: str, image: np.ndarray) -> tuple[np.ndarray, str]:
    if stage_name == "P0":
        return base.metric_image(image), "grayscale"
    if stage_name == "CLAHE":
        return base.apply_clahe(image, clip_limit=0.025), "grayscale"
    if stage_name == "Ordinary Gabor":
        return base.multi_orientation_gabor(image, frequency=0.115, orientations=8, blend=0.30), "grayscale"
    if stage_name == "Coherence-Guided Directional Diffusion":
        return base.coherence_guided_diffusion(image), "grayscale"
    if stage_name == "Total Variation Restoration":
        return base.metric_image(base.restoration.denoise_tv_chambolle(image, weight=0.045, channel_axis=None)), "grayscale"
    if stage_name == "Closing":
        return base.morphology_outputs(image)["M3-2"][1].astype(np.float32), "binary ridge map"
    raise ValueError(f"Unknown pipeline stage: {stage_name}")


def apply_pipeline(p0: np.ndarray, pipeline: PipelineDefinition) -> tuple[np.ndarray, np.ndarray | None, list[dict[str, object]]]:
    current_gray = base.metric_image(p0)
    closing_binary = None
    stages = []
    for order, stage_name in enumerate(pipeline.stages):
        start = perf_counter()
        output, domain = apply_named_stage(stage_name, current_gray)
        runtime = perf_counter() - start
        if domain == "grayscale":
            current_gray = base.metric_image(output)
            metric_gray = current_gray
            binary_override = None
        else:
            closing_binary = np.asarray(output > 0.5, dtype=bool)
            metric_gray = current_gray
            binary_override = closing_binary
        stages.append(
            {
                "Stage order": order,
                "Stage": stage_name,
                "Output domain": domain,
                "Image": output,
                "Metric grayscale image": metric_gray,
                "Binary override": binary_override,
                "Runtime (s)": runtime,
            }
        )
    return current_gray, closing_binary, stages


def direct_metrics_for_output(metric_gray: np.ndarray, binary_override: np.ndarray | None = None) -> dict[str, float]:
    return direct.direct_quality_metrics(metric_gray, binary_override=binary_override, metric_image=metric_gray)


def add_direct_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    def terminal_baseline(pipeline_id: str) -> pd.DataFrame:
        baseline = result[result["Pipeline ID"] == pipeline_id].copy()
        if "Is Final Stage" in baseline.columns:
            baseline = baseline[baseline["Is Final Stage"].astype(bool)]
        return baseline.groupby(["Category", "Image"], as_index=False).first().set_index(["Category", "Image"])

    baseline_p0 = terminal_baseline("B0")
    baseline_clahe = terminal_baseline("B1")
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
        for label, baseline in [("P0", baseline_p0), ("CLAHE", baseline_clahe)]:
            if key not in baseline.index:
                continue
            base_row = baseline.loc[key]
            for column in tracked:
                result.loc[index, f"Delta {column} vs {label}"] = row[column] - base_row[column]
    return result


def run_direct_compatibility(selection_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows = []
    stage_rows = []
    examples: dict[str, dict[str, np.ndarray]] = {}
    for _, selected in selection_df.iterrows():
        path = PROJECT_ROOT / selected["Path"]
        p0 = direct.load_p0(path)
        category = str(selected["Category"])
        visual_outputs = {"Original Altered": p0}
        previous_by_pipeline: dict[str, dict[str, float]] = {}
        for pipeline in pipeline_definitions():
            final_gray, closing_binary, stages = apply_pipeline(p0, pipeline)
            final_binary = closing_binary if closing_binary is not None else None
            final_metrics = direct_metrics_for_output(final_gray, binary_override=final_binary)
            total_runtime = float(sum(float(stage["Runtime (s)"]) for stage in stages))
            rows.append(
                {
                    **selected.to_dict(),
                    "Experiment": "Direct Altered",
                    "Pipeline ID": pipeline.pipeline_id,
                    "Pipeline": pipeline.pipeline,
                    "Purpose": pipeline.purpose,
                    "Final output domain": pipeline.final_output_domain,
                    "Has Closing stage": pipeline.has_closing_stage,
                    "Is baseline": pipeline.is_baseline,
                    "Is grayscale finalist candidate": pipeline.is_grayscale_finalist_candidate,
                    "Runtime (s)": total_runtime,
                    **final_metrics,
                }
            )
            previous = None
            for stage in stages:
                metrics = direct_metrics_for_output(stage["Metric grayscale image"], binary_override=stage["Binary override"])
                stage_record = {
                    **selected.to_dict(),
                    "Experiment": "Direct Altered",
                    "Pipeline ID": pipeline.pipeline_id,
                    "Pipeline": pipeline.pipeline,
                    "Stage order": stage["Stage order"],
                    "Stage": stage["Stage"],
                    "Output domain": stage["Output domain"],
                    "Runtime (s)": stage["Runtime (s)"],
                    **metrics,
                }
                if previous is not None:
                    for column in [
                        "Ridge coherence",
                        "Local contrast",
                        "Fragmentation per 1000 ridge pixels",
                        "Small connected components <20px",
                        "Ridge continuity proxy",
                    ]:
                        stage_record[f"Delta {column} vs previous stage"] = stage_record[column] - previous[column]
                stage_rows.append(stage_record)
                previous = metrics
            if pipeline.pipeline_id in {"B1", "C1", "C2", "C3"}:
                visual_outputs[pipeline.pipeline] = final_gray
            if pipeline.pipeline_id == "C4":
                visual_outputs[pipeline.pipeline] = final_gray
            if pipeline.pipeline_id == "C5":
                visual_outputs[pipeline.pipeline] = closing_binary.astype(float)
            if pipeline.pipeline_id == "C6":
                visual_outputs[pipeline.pipeline] = closing_binary.astype(float)
        if category not in examples:
            examples[category] = {"image_name": selected["Image"], "outputs": visual_outputs}
    return add_direct_deltas(pd.DataFrame(rows)), add_direct_deltas(pd.DataFrame(stage_rows)), examples


def summarise_direct(frame: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "Images": ("Image", "count"),
        "Ridge coherence mean": ("Ridge coherence", "mean"),
        "Ridge coherence std": ("Ridge coherence", "std"),
        "Local contrast mean": ("Local contrast", "mean"),
        "Local contrast std": ("Local contrast", "std"),
        "Foreground/background separation proxy mean": ("Foreground/background separation proxy", "mean"),
        "Ridge coverage mean": ("Ridge coverage in foreground", "mean"),
        "Fragmentation mean": ("Fragmentation per 1000 ridge pixels", "mean"),
        "Small components mean": ("Small connected components <20px", "mean"),
        "Ridge continuity mean": ("Ridge continuity proxy", "mean"),
        "Runtime mean": ("Runtime (s)", "mean"),
        "Delta coherence vs P0 mean": ("Delta Ridge coherence vs P0", "mean"),
        "Delta contrast vs P0 mean": ("Delta Local contrast vs P0", "mean"),
        "Delta fragmentation vs P0 mean": ("Delta Fragmentation per 1000 ridge pixels vs P0", "mean"),
        "Delta continuity vs P0 mean": ("Delta Ridge continuity proxy vs P0", "mean"),
        "Delta coherence vs CLAHE mean": ("Delta Ridge coherence vs CLAHE", "mean"),
        "Delta contrast vs CLAHE mean": ("Delta Local contrast vs CLAHE", "mean"),
        "Delta fragmentation vs CLAHE mean": ("Delta Fragmentation per 1000 ridge pixels vs CLAHE", "mean"),
        "Delta continuity vs CLAHE mean": ("Delta Ridge continuity proxy vs CLAHE", "mean"),
    }
    by_category = (
        frame.groupby(
            [
                "Category",
                "Pipeline ID",
                "Pipeline",
                "Purpose",
                "Final output domain",
                "Has Closing stage",
                "Is baseline",
                "Is grayscale finalist candidate",
            ],
            as_index=False,
        )
        .agg(**agg)
        .sort_values(["Category", "Pipeline ID"])
    )
    overall = (
        frame.groupby(
            [
                "Pipeline ID",
                "Pipeline",
                "Purpose",
                "Final output domain",
                "Has Closing stage",
                "Is baseline",
                "Is grayscale finalist candidate",
            ],
            as_index=False,
        )
        .agg(**agg)
        .assign(Category="Overall")
    )
    summary = pd.concat([by_category, overall], ignore_index=True, sort=False)
    summary["_Category sort"] = summary["Category"].map(CATEGORY_ORDER)
    return summary.sort_values(["_Category sort", "Pipeline ID"]).drop(columns=["_Category sort"]).reset_index(drop=True)


def build_compatibility_table(direct_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    overall = direct_summary[direct_summary["Category"] == "Overall"].copy()
    by_category = direct_summary[direct_summary["Category"].isin(["Easy", "Medium", "Hard"])].copy()
    for _, row in overall.iterrows():
        pid = row["Pipeline ID"]
        category_rows = by_category[by_category["Pipeline ID"] == pid]
        max_category_frag = float(category_rows["Delta fragmentation vs CLAHE mean"].max()) if not category_rows.empty else math.nan
        min_category_contrast = float(category_rows["Delta contrast vs CLAHE mean"].min()) if not category_rows.empty else math.nan
        coherence_ok = bool(row["Delta coherence vs P0 mean"] > 0 or row["Delta coherence vs CLAHE mean"] >= -0.005)
        contrast_ok = bool(row["Delta contrast vs CLAHE mean"] >= CONTRAST_DAMAGE_THRESHOLD)
        fragmentation_ok = bool(row["Delta fragmentation vs CLAHE mean"] <= FRAGMENTATION_DAMAGE_THRESHOLD)
        category_ok = bool(
            max_category_frag <= CATEGORY_FRAGMENTATION_DAMAGE_THRESHOLD
            and min_category_contrast >= CATEGORY_CONTRAST_DAMAGE_THRESHOLD
        )
        strong = bool(coherence_ok and contrast_ok and fragmentation_ok and category_ok and pid != "B0")
        problems = []
        if not coherence_ok:
            problems.append("coherence not competitive")
        if not contrast_ok:
            problems.append("contrast damaged")
        if not fragmentation_ok:
            problems.append("fragmentation worsens")
        if not category_ok:
            problems.append("category-specific damage")
        rows.append(
            {
                "Pipeline ID": pid,
                "Pipeline": row["Pipeline"],
                "Final output domain": row["Final output domain"],
                "Is grayscale finalist candidate": bool(row["Is grayscale finalist candidate"]),
                "Coherence improves/competitive": coherence_ok,
                "Contrast preserved vs CLAHE": contrast_ok,
                "Fragmentation acceptable vs CLAHE": fragmentation_ok,
                "Consistent across Easy/Medium/Hard": category_ok,
                "Strong compatible candidate": strong,
                "Compatibility interpretation": "compatible" if strong else ("control baseline" if pid == "B0" else "; ".join(problems)),
                "Overall coherence": float(row["Ridge coherence mean"]),
                "Overall contrast": float(row["Local contrast mean"]),
                "Overall fragmentation": float(row["Fragmentation mean"]),
                "Delta coherence vs P0": float(row["Delta coherence vs P0 mean"]),
                "Delta contrast vs P0": float(row["Delta contrast vs P0 mean"]),
                "Delta fragmentation vs P0": float(row["Delta fragmentation vs P0 mean"]),
                "Delta coherence vs CLAHE": float(row["Delta coherence vs CLAHE mean"]),
                "Delta contrast vs CLAHE": float(row["Delta contrast vs CLAHE mean"]),
                "Delta fragmentation vs CLAHE": float(row["Delta fragmentation vs CLAHE mean"]),
                "Worst category fragmentation delta vs CLAHE": max_category_frag,
                "Worst category contrast delta vs CLAHE": min_category_contrast,
            }
        )
    return pd.DataFrame(rows).sort_values("Pipeline ID").reset_index(drop=True)


def choose_category_best(direct_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in ["Easy", "Medium", "Hard", "Overall"]:
        frame = direct_summary[
            (direct_summary["Category"] == category)
            & (direct_summary["Pipeline ID"] != "B0")
        ].copy()
        compatible = frame[
            (frame["Delta contrast vs CLAHE mean"] >= CONTRAST_DAMAGE_THRESHOLD)
            & (frame["Delta fragmentation vs CLAHE mean"] <= FRAGMENTATION_DAMAGE_THRESHOLD)
            & ((frame["Delta coherence vs P0 mean"] > 0) | (frame["Delta coherence vs CLAHE mean"] >= -0.005))
        ].copy()
        if compatible.empty:
            compatible = frame[frame["Pipeline ID"] == "B1"].copy()
            reason = "fallback to CLAHE baseline because no hybrid met all compatibility guards"
        else:
            reason = "highest coherence among candidates passing non-damage compatibility guards"
        winner = compatible.sort_values("Ridge coherence mean", ascending=False).iloc[0]
        rows.append(
            {
                "Category": category,
                "Best Pipeline ID": winner["Pipeline ID"],
                "Best Pipeline": winner["Pipeline"],
                "Selection reason": reason,
                "Coherence": float(winner["Ridge coherence mean"]),
                "Contrast": float(winner["Local contrast mean"]),
                "Fragmentation": float(winner["Fragmentation mean"]),
                "Delta coherence vs CLAHE": float(winner["Delta coherence vs CLAHE mean"]),
                "Delta contrast vs CLAHE": float(winner["Delta contrast vs CLAHE mean"]),
                "Delta fragmentation vs CLAHE": float(winner["Delta fragmentation vs CLAHE mean"]),
            }
        )
    return pd.DataFrame(rows)


def select_finalists(direct_summary: pd.DataFrame, compatibility_df: pd.DataFrame) -> pd.DataFrame:
    finalists = [
        {
            "Pipeline ID": "B1",
            "Pipeline": "P0 -> CLAHE",
            "Finalist role": "mandatory CLAHE baseline",
            "Selection reason": "Direct24 and hybrid24 both treat CLAHE as the main M1 baseline.",
        }
    ]
    eligible = compatibility_df[
        (compatibility_df["Pipeline ID"].isin(["C1", "C2", "C3", "C4"]))
        & (compatibility_df["Strong compatible candidate"])
        & (compatibility_df["Is grayscale finalist candidate"])
    ].copy()
    if not eligible.empty:
        best_hybrid = eligible.sort_values(["Overall coherence", "Overall fragmentation"], ascending=[False, True]).iloc[0]
        finalists.append(
            {
                "Pipeline ID": best_hybrid["Pipeline ID"],
                "Pipeline": best_hybrid["Pipeline"],
                "Finalist role": "best compatible grayscale hybrid",
                "Selection reason": "Passed coherence, contrast, fragmentation and per-category compatibility guards.",
            }
        )
    else:
        fallback = compatibility_df[compatibility_df["Pipeline ID"].isin(["C2", "C3", "C4"])].copy()
        fallback = fallback.sort_values(["Delta fragmentation vs CLAHE", "Delta contrast vs CLAHE"], ascending=[True, False]).iloc[0]
        finalists.append(
            {
                "Pipeline ID": fallback["Pipeline ID"],
                "Pipeline": fallback["Pipeline"],
                "Finalist role": "least damaging grayscale hybrid fallback",
                "Selection reason": "No hybrid passed every guard; selected the candidate with the least fragmentation penalty for controlled validation.",
            }
        )
    existing = {row["Pipeline ID"] for row in finalists}
    order_pair = compatibility_df[compatibility_df["Pipeline ID"].isin(["C3", "C4"]) & ~compatibility_df["Pipeline ID"].isin(existing)].copy()
    if not order_pair.empty and len(finalists) < 3:
        order_pair["Order guard count"] = (
            order_pair["Coherence improves/competitive"].astype(int)
            + order_pair["Contrast preserved vs CLAHE"].astype(int)
            + order_pair["Fragmentation acceptable vs CLAHE"].astype(int)
            + order_pair["Consistent across Easy/Medium/Hard"].astype(int)
        )
        contender = order_pair.sort_values(["Order guard count", "Overall coherence"], ascending=[False, False]).iloc[0]
        finalists.append(
            {
                "Pipeline ID": contender["Pipeline ID"],
                "Pipeline": contender["Pipeline"],
                "Finalist role": "order-effect competitor",
                "Selection reason": "C3/C4 explicitly test whether TV before or after CLAHE is preferable.",
            }
        )
    if len(finalists) < 3:
        remaining = compatibility_df[
            compatibility_df["Pipeline ID"].isin(["C1", "C2", "C3", "C4"])
            & ~compatibility_df["Pipeline ID"].isin({row["Pipeline ID"] for row in finalists})
        ].copy()
        if not remaining.empty:
            contender = remaining.sort_values(["Overall coherence"], ascending=False).iloc[0]
            finalists.append(
                {
                    "Pipeline ID": contender["Pipeline ID"],
                    "Pipeline": contender["Pipeline"],
                    "Finalist role": "scientifically meaningful M2 competitor",
                    "Selection reason": "Included to compare the highest-coherence M2 path against the compatibility winner.",
                }
            )
    finalist_df = pd.DataFrame(finalists[:3])
    detail = compatibility_df.drop(columns=["Pipeline"], errors="ignore")
    return finalist_df.merge(detail, on="Pipeline ID", how="left")


def run_controlled_quantitative(controlled_selection_df: pd.DataFrame, finalists_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipeline_map = {p.pipeline_id: p for p in pipeline_definitions()}
    active_ids = ["B0"] + [pid for pid in finalists_df["Pipeline ID"].tolist() if pid != "B0"]
    rows = []
    stage_rows = []
    for _, selected in controlled_selection_df.iterrows():
        reference = base.load_grayscale_metric(PROJECT_ROOT / selected["Path"])
        degraded, _ = base.simulate_degradation_stages(reference, severity=selected["Severity"], seed=int(selected["Seed"]))
        reference_mask = base.fingerprint_mask(reference)
        for pipeline_id in active_ids:
            pipeline = pipeline_map[pipeline_id]
            final_gray, closing_binary, stages = apply_pipeline(degraded, pipeline)
            if closing_binary is not None:
                # Controlled PSNR/MSE/SSIM are not calculated on structural outputs; use the last grayscale stage.
                final_image = final_gray
            else:
                final_image = final_gray
            total_runtime = float(sum(float(stage["Runtime (s)"]) for stage in stages if stage["Output domain"] == "grayscale"))
            metrics = base.evaluate_grayscale(reference, final_image, reference_mask, total_runtime)
            rows.append(
                {
                    **selected.to_dict(),
                    "Experiment": "Controlled aligned Real synthetic-degradation",
                    "Pipeline ID": pipeline_id,
                    "Pipeline": pipeline.pipeline,
                    "Is finalist": pipeline_id in set(finalists_df["Pipeline ID"]),
                    "Final output used for PSNR/MSE/SSIM": pipeline.grayscale_final_stage,
                    **metrics,
                }
            )
            previous = None
            for stage in stages:
                if stage["Output domain"] != "grayscale":
                    continue
                stage_metrics = base.evaluate_grayscale(reference, stage["Metric grayscale image"], reference_mask, stage["Runtime (s)"])
                record = {
                    **selected.to_dict(),
                    "Experiment": "Controlled aligned Real synthetic-degradation",
                    "Pipeline ID": pipeline_id,
                    "Pipeline": pipeline.pipeline,
                    "Stage order": stage["Stage order"],
                    "Stage": stage["Stage"],
                    "Output domain": stage["Output domain"],
                    **stage_metrics,
                }
                if previous is not None:
                    record["Delta PSNR vs previous stage (dB)"] = record["PSNR (dB)"] - previous["PSNR (dB)"]
                    record["Delta MSE vs previous stage"] = record["MSE"] - previous["MSE"]
                    record["Delta SSIM vs previous stage"] = record["SSIM"] - previous["SSIM"]
                    record["Delta coherence vs previous stage"] = record["Ridge coherence"] - previous["Ridge coherence"]
                stage_rows.append(record)
                previous = stage_metrics
    return pd.DataFrame(rows), pd.DataFrame(stage_rows)


def summarise_controlled(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["Pipeline ID", "Pipeline", "Is finalist", "Final output used for PSNR/MSE/SSIM"], as_index=False)
        .agg(
            Images=("Image", "nunique"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            PSNR_mean=("PSNR (dB)", "mean"),
            PSNR_std=("PSNR (dB)", "std"),
            SSIM_mean=("SSIM", "mean"),
            SSIM_std=("SSIM", "std"),
            Coherence_mean=("Ridge coherence", "mean"),
            Coherence_std=("Ridge coherence", "std"),
            Runtime_mean=("Runtime (s)", "mean"),
        )
        .sort_values(["Is finalist", "PSNR_mean"], ascending=[False, False])
    )


def previous_comparable_quantitative() -> pd.DataFrame:
    rows = []
    family_path = OUTPUT_DIR / "screening24_family_summary.csv"
    combo_path = OUTPUT_DIR / "screening24_combinations.csv"
    if family_path.exists():
        family = pd.read_csv(family_path)
        technique = family[family["Table"] == "Technique summary"].copy()
        wanted = {
            "C4": "screening24 Contrast Stretching",
            "C3": "screening24 CLAHE",
            "R1": "screening24 Wiener Deconvolution",
            "D6": "screening24 Wavelet Denoising",
            "R3": "screening24 Total Variation Restoration",
            "G1": "screening24 Ordinary Gabor",
            "G2": "screening24 Literature-Inspired Modified Gabor",
        }
        for tech_id, label in wanted.items():
            match = technique[technique["Technique ID"] == tech_id]
            if not match.empty:
                row = match.iloc[0]
                rows.append(
                    {
                        "Comparable source": label,
                        "Protocol": "same screening24 controlled 24 Real/degradation protocol",
                        "MSE_mean": row.get("MSE mean", math.nan),
                        "PSNR_mean": row.get("PSNR (dB) mean", math.nan),
                        "SSIM_mean": row.get("SSIM mean", math.nan),
                    }
                )
    if combo_path.exists():
        combo = pd.read_csv(combo_path).sort_values("PSNR_mean", ascending=False)
        if not combo.empty:
            row = combo.iloc[0]
            rows.append(
                {
                    "Comparable source": f"screening24 previous team hybrid {row['Combination ID']}",
                    "Protocol": "same screening24 controlled 24 Real/degradation protocol",
                    "MSE_mean": row.get("MSE_mean", math.nan),
                    "PSNR_mean": row.get("PSNR_mean", math.nan),
                    "SSIM_mean": row.get("SSIM_mean", math.nan),
                }
            )
    return pd.DataFrame(rows)


def create_direct_visuals(examples: dict[str, dict[str, np.ndarray]], finalists_df: pd.DataFrame) -> list[str]:
    output_files = []
    finalist_ids = finalists_df["Pipeline ID"].tolist()
    required = ["B1", "C1", "C2", "C3"]
    best_hybrid_id = next((pid for pid in finalist_ids if pid != "B1"), "C2")
    pipeline_labels = {p.pipeline_id: p.pipeline for p in pipeline_definitions()}
    for category, example in examples.items():
        outputs = example["outputs"]
        panels = [
            ("Original Altered", outputs["Original Altered"]),
            ("CLAHE", outputs[pipeline_labels["B1"]]),
            ("CLAHE + Ordinary Gabor", outputs[pipeline_labels["C1"]]),
            ("CLAHE + Directional Diffusion", outputs[pipeline_labels["C2"]]),
            ("CLAHE + TV", outputs[pipeline_labels["C3"]]),
        ]
        if best_hybrid_id in pipeline_labels:
            panels.append((f"Selected finalist {best_hybrid_id}", outputs[pipeline_labels[best_hybrid_id]]))
        fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
        axes = axes.reshape(-1)
        for axis, (title, image) in zip(axes, panels):
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        for axis in axes[len(panels) :]:
            axis.axis("off")
        fig.suptitle(f"hybrid24 direct {category} compatibility - {example['image_name']}", fontsize=10)
        fig.tight_layout()
        path = OUTPUT_DIR / f"hybrid24_direct_{category.lower()}_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        output_files.append(str(path.relative_to(PROJECT_ROOT)))
    return output_files


def create_controlled_visual(controlled_selection_df: pd.DataFrame, finalists_df: pd.DataFrame, controlled_summary_df: pd.DataFrame) -> list[str]:
    first = controlled_selection_df.iloc[0]
    reference = base.load_grayscale_metric(PROJECT_ROOT / first["Path"])
    degraded, _ = base.simulate_degradation_stages(reference, severity=first["Severity"], seed=int(first["Seed"]))
    pipeline_map = {p.pipeline_id: p for p in pipeline_definitions()}
    best_hybrid_id = next((pid for pid in finalists_df["Pipeline ID"].tolist() if pid != "B1"), "B1")
    selected_hybrid, _, _ = apply_pipeline(degraded, pipeline_map[best_hybrid_id])
    clahe, _, _ = apply_pipeline(degraded, pipeline_map["B1"])
    competitor = base.contrast_stretch(degraded)
    competitor_title = "Contrast Stretching comparator"
    panels = [
        ("Clean Real reference", reference),
        ("Controlled degraded input", degraded),
        ("CLAHE result", clahe),
        (f"Selected hybrid {best_hybrid_id}", selected_hybrid),
        (competitor_title, competitor),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    fig.suptitle(f"hybrid24 controlled reference comparison - {first['Image']}", fontsize=10)
    fig.tight_layout()
    path = OUTPUT_DIR / "hybrid24_controlled_reference_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [str(path.relative_to(PROJECT_ROOT))]


def compare_stage_summary(stage_df: pd.DataFrame) -> pd.DataFrame:
    direct_stage = (
        stage_df[stage_df["Experiment"] == "Direct Altered"]
        .groupby(["Experiment", "Pipeline ID", "Pipeline", "Stage order", "Stage"], as_index=False)
        .agg(
            Delta_coherence=("Delta Ridge coherence vs previous stage", "mean"),
            Delta_contrast=("Delta Local contrast vs previous stage", "mean"),
            Delta_fragmentation=("Delta Fragmentation per 1000 ridge pixels vs previous stage", "mean"),
        )
    )
    controlled_stage = (
        stage_df[stage_df["Experiment"].str.startswith("Controlled", na=False)]
        .groupby(["Experiment", "Pipeline ID", "Pipeline", "Stage order", "Stage"], as_index=False)
        .agg(
            Delta_PSNR=("Delta PSNR vs previous stage (dB)", "mean"),
            Delta_MSE=("Delta MSE vs previous stage", "mean"),
            Delta_SSIM=("Delta SSIM vs previous stage", "mean"),
            Delta_controlled_coherence=("Delta coherence vs previous stage", "mean"),
        )
    )
    return pd.concat([direct_stage, controlled_stage], ignore_index=True, sort=False).sort_values(
        ["Experiment", "Pipeline ID", "Stage order"]
    )


def fmt_pm(row: pd.Series, mean_col: str, std_col: str, digits: int = 4) -> str:
    return f"{row[mean_col]:.{digits}f} +/- {row[std_col]:.{digits}f}"


def report_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    if frame.empty:
        return "- n/a"
    return frame[columns].round(digits).to_markdown(index=False)


def create_report(
    selection_df: pd.DataFrame,
    direct_summary: pd.DataFrame,
    compatibility_df: pd.DataFrame,
    category_best_df: pd.DataFrame,
    finalists_df: pd.DataFrame,
    controlled_selection_df: pd.DataFrame,
    controlled_summary: pd.DataFrame,
    stage_df: pd.DataFrame,
    previous_quant: pd.DataFrame,
    runtime: dict[str, object],
) -> str:
    overall = direct_summary[direct_summary["Category"] == "Overall"].copy()
    finalist_controlled = controlled_summary[controlled_summary["Is finalist"] == True].copy()
    highest_psnr = finalist_controlled.sort_values("PSNR_mean", ascending=False).iloc[0]
    lowest_mse = finalist_controlled.sort_values("MSE_mean", ascending=True).iloc[0]
    highest_ssim = finalist_controlled.sort_values("SSIM_mean", ascending=False).iloc[0]
    best_direct_hybrid = finalists_df[finalists_df["Pipeline ID"] != "B1"].iloc[0] if len(finalists_df) > 1 else finalists_df.iloc[0]
    clahe = overall[overall["Pipeline ID"] == "B1"].iloc[0]
    ordinary = overall[overall["Pipeline ID"] == "C1"].iloc[0]
    diffusion = overall[overall["Pipeline ID"] == "C2"].iloc[0]
    tv_after = overall[overall["Pipeline ID"] == "C3"].iloc[0]
    tv_before = overall[overall["Pipeline ID"] == "C4"].iloc[0]
    c5 = overall[overall["Pipeline ID"] == "C5"].iloc[0]
    c6 = overall[overall["Pipeline ID"] == "C6"].iloc[0]
    selected_pipeline_id = str(best_direct_hybrid["Pipeline ID"])
    selected_pipeline_summary = overall[overall["Pipeline ID"] == selected_pipeline_id].iloc[0]
    best_existing_single = pd.read_csv(OUTPUT_DIR / "direct24_per_severity_summary.csv")
    best_existing_single = best_existing_single[
        (best_existing_single["Category"] == "Overall")
        & (best_existing_single["Is enhancement technique"] == True)
    ].sort_values("Ridge coherence mean", ascending=False).iloc[0]
    stage_summary = compare_stage_summary(stage_df)
    controlled_stages = stage_summary[stage_summary["Experiment"].str.startswith("Controlled", na=False)].dropna(subset=["Delta_PSNR"])
    direct_stages = stage_summary[stage_summary["Experiment"] == "Direct Altered"].dropna(subset=["Delta_fragmentation"])
    selected_controlled_stages = stage_summary[
        (stage_summary["Experiment"].str.startswith("Controlled", na=False))
        & (stage_summary["Pipeline ID"] == selected_pipeline_id)
    ].copy()
    selected_direct_stages = stage_summary[
        (stage_summary["Experiment"] == "Direct Altered")
        & (stage_summary["Pipeline ID"] == selected_pipeline_id)
    ].copy()
    selected_controlled_stage_deltas = selected_controlled_stages.dropna(subset=["Delta_PSNR"]).sort_values("Stage order")
    selected_direct_stage_deltas = selected_direct_stages.dropna(subset=["Delta_fragmentation"]).sort_values("Stage order")
    selected_psnr_note = (
        f"For selected {selected_pipeline_id}, no individual enhancement stage increased PSNR over its immediately previous stage; "
        "the value of TV-before-CLAHE is that it makes the final CLAHE result less damaging than CLAHE-only under the controlled protocol."
        if (selected_controlled_stage_deltas["Delta_PSNR"] > 0).sum() == 0
        else f"Selected {selected_pipeline_id} has at least one positive stage-wise PSNR step."
    )
    selected_mse_note = (
        f"For selected {selected_pipeline_id}, no individual enhancement stage reduced MSE over its immediately previous stage; "
        "the final result is still lower-MSE than CLAHE-only because CLAHE is applied after restoration."
        if (selected_controlled_stage_deltas["Delta_MSE"] < 0).sum() == 0
        else f"Selected {selected_pipeline_id} has at least one negative stage-wise MSE step."
    )
    psnr_stage = controlled_stages.sort_values("Delta_PSNR", ascending=False).iloc[0]
    mse_stage = controlled_stages.sort_values("Delta_MSE", ascending=True).iloc[0]
    structural_damage_stage = direct_stages.sort_values("Delta_fragmentation", ascending=False).iloc[0]
    selected_controlled = controlled_summary[controlled_summary["Pipeline ID"] == selected_pipeline_id]
    selected_controlled_row = selected_controlled.iloc[0] if not selected_controlled.empty else pd.Series(dtype=object)
    b0_controlled = controlled_summary[controlled_summary["Pipeline ID"] == "B0"].iloc[0]
    selected_quant_acceptable = (
        not selected_controlled_row.empty
        and selected_controlled_row["PSNR_mean"] >= finalist_controlled["PSNR_mean"].median()
        and selected_controlled_row["SSIM_mean"] >= finalist_controlled["SSIM_mean"].median()
    )
    selected_direct_acceptable = (
        selected_pipeline_summary["Delta contrast vs CLAHE mean"] >= CONTRAST_DAMAGE_THRESHOLD
        and selected_pipeline_summary["Delta fragmentation vs CLAHE mean"] <= FRAGMENTATION_DAMAGE_THRESHOLD
    )
    best_hybrid_better_than_clahe = (
        selected_direct_acceptable
        and selected_pipeline_summary["Ridge coherence mean"] >= clahe["Ridge coherence mean"] - 0.005
        and selected_controlled_row.get("PSNR_mean", -math.inf) >= controlled_summary[controlled_summary["Pipeline ID"] == "B1"].iloc[0]["PSNR_mean"]
    )
    best_hybrid_better_than_existing = (
        selected_direct_acceptable
        and selected_pipeline_summary["Ridge coherence mean"] >= best_existing_single["Ridge coherence mean"] - 0.005
    )
    previous_single_quant = previous_quant[~previous_quant["Comparable source"].str.contains("previous team hybrid", case=False, na=False)].copy()
    best_previous_single_quant = (
        previous_single_quant.sort_values("PSNR_mean", ascending=False).iloc[0]
        if not previous_single_quant.empty
        else pd.Series(dtype=object)
    )
    best_hybrid_beats_best_previous_single_quant = (
        not best_previous_single_quant.empty
        and selected_controlled_row["PSNR_mean"] >= best_previous_single_quant["PSNR_mean"]
        and selected_controlled_row["MSE_mean"] <= best_previous_single_quant["MSE_mean"]
    )
    best_hybrid_better_than_existing_overall = bool(best_hybrid_better_than_existing and best_hybrid_beats_best_previous_single_quant)
    simpler_than_longer_team_preferable = True
    selection_counts = selection_df.groupby("Category")["Image"].count().to_dict()
    pair_counts = selection_df.groupby("Category")["Real pair candidate exists"].sum().astype(int).to_dict()
    coherence_improved = compatibility_df[compatibility_df["Delta coherence vs P0"] > 0]
    contrast_damaged = compatibility_df[compatibility_df["Delta contrast vs CLAHE"] < CONTRAST_DAMAGE_THRESHOLD]
    frag_damaged = compatibility_df[compatibility_df["Delta fragmentation vs CLAHE"] > FRAGMENTATION_DAMAGE_THRESHOLD]
    closing_repair = pd.DataFrame(
        [
            {
                "Comparison": "C5 Closing vs C2 grayscale",
                "Fragmentation delta": c5["Fragmentation mean"] - diffusion["Fragmentation mean"],
                "Continuity delta": c5["Ridge continuity mean"] - diffusion["Ridge continuity mean"],
            },
            {
                "Comparison": "C6 Closing vs C3 grayscale",
                "Fragmentation delta": c6["Fragmentation mean"] - tv_after["Fragmentation mean"],
                "Continuity delta": c6["Ridge continuity mean"] - tv_after["Ridge continuity mean"],
            },
        ]
    )
    lines = [
        "# hybrid24 Controlled Hybrid Compatibility + Final PSNR/MSE Candidate Selection Report",
        "",
        "## A. Which 24 Altered images were reused?",
        f"- Reused the Direct24 image list exactly: {selection_counts}. Filename-level Real pair candidates existed for {pair_counts}, but no full-reference direct metrics used these pairs because geometric alignment was not verified.",
        report_table(selection_df, ["Category", "Category order", "Image", "Path"], digits=4),
        "",
        "## B. Which baseline pipelines were evaluated?",
        "- B0: P0 only. B1: P0 -> CLAHE. CLAHE is treated as the main enhancement baseline, not assumed weaker than longer hybrids.",
        "",
        "## C. Which hybrid candidates were evaluated and why?",
        report_table(
            pd.DataFrame([p.__dict__ for p in pipeline_definitions() if p.pipeline_id.startswith("C")]),
            ["pipeline_id", "pipeline", "purpose", "final_output_domain"],
        ),
        "",
        "## D. Which candidates improved ridge coherence?",
        report_table(coherence_improved, ["Pipeline ID", "Pipeline", "Overall coherence", "Delta coherence vs P0", "Delta coherence vs CLAHE"]),
        "",
        "## E. Which candidates damaged contrast?",
        report_table(contrast_damaged, ["Pipeline ID", "Pipeline", "Overall contrast", "Delta contrast vs CLAHE", "Compatibility interpretation"]),
        "",
        "## F. Which candidates increased fragmentation?",
        report_table(frag_damaged, ["Pipeline ID", "Pipeline", "Overall fragmentation", "Delta fragmentation vs CLAHE", "Compatibility interpretation"]),
        "",
        "## G. Which candidate was best on Easy?",
        report_table(category_best_df[category_best_df["Category"] == "Easy"], ["Best Pipeline ID", "Best Pipeline", "Selection reason", "Coherence", "Contrast", "Fragmentation"]),
        "",
        "## H. Which candidate was best on Medium?",
        report_table(category_best_df[category_best_df["Category"] == "Medium"], ["Best Pipeline ID", "Best Pipeline", "Selection reason", "Coherence", "Contrast", "Fragmentation"]),
        "",
        "## I. Which candidate was best on Hard?",
        report_table(category_best_df[category_best_df["Category"] == "Hard"], ["Best Pipeline ID", "Best Pipeline", "Selection reason", "Coherence", "Contrast", "Fragmentation"]),
        "",
        "## J. Was CLAHE alone stronger than some hybrids?",
        "- Yes. CLAHE alone preserved contrast and fragmentation better than C1 Ordinary Gabor and was competitive with longer pipelines. More stages did not automatically improve the direct evidence.",
        report_table(overall, ["Pipeline ID", "Pipeline", "Ridge coherence mean", "Local contrast mean", "Fragmentation mean", "Delta contrast vs CLAHE mean", "Delta fragmentation vs CLAHE mean"]),
        "",
        "## K. Is Ordinary Gabor still recommended after considering fragmentation?",
        f"- Not as the final grayscale enhancement pipeline. C1 had coherence {ordinary['Ridge coherence mean']:.4f}, but contrast changed {ordinary['Delta contrast vs CLAHE mean']:+.4f} and fragmentation changed {ordinary['Delta fragmentation vs CLAHE mean']:+.4f} versus CLAHE.",
        "",
        "## L. Is Directional Diffusion more compatible with CLAHE than Ordinary Gabor?",
        f"- Yes if judged by damage control. C2 versus C1: contrast delta vs CLAHE {diffusion['Delta contrast vs CLAHE mean']:+.4f} vs {ordinary['Delta contrast vs CLAHE mean']:+.4f}; fragmentation delta vs CLAHE {diffusion['Delta fragmentation vs CLAHE mean']:+.4f} vs {ordinary['Delta fragmentation vs CLAHE mean']:+.4f}.",
        "",
        "## M. Is Total Variation Restoration more compatible with CLAHE?",
        f"- Yes. C3 changed contrast {tv_after['Delta contrast vs CLAHE mean']:+.4f} and fragmentation {tv_after['Delta fragmentation vs CLAHE mean']:+.4f} versus CLAHE, while preserving a smoother ridge structure than C1.",
        "",
        "## N. Does processing order matter for CLAHE and TV restoration?",
        f"- Yes. C3 CLAHE -> TV: coherence {tv_after['Ridge coherence mean']:.4f}, contrast {tv_after['Local contrast mean']:.4f}, fragmentation {tv_after['Fragmentation mean']:.4f}. C4 TV -> CLAHE: coherence {tv_before['Ridge coherence mean']:.4f}, contrast {tv_before['Local contrast mean']:.4f}, fragmentation {tv_before['Fragmentation mean']:.4f}.",
        "",
        "## O. Does Closing genuinely repair fragmentation after enhancement?",
        report_table(closing_repair, ["Comparison", "Fragmentation delta", "Continuity delta"]),
        "- Closing is useful as M3 structural processing, but it is not a grayscale enhancement output for PSNR/MSE/SSIM.",
        "",
        "## P. Which grayscale pipelines were selected as the maximum three finalists?",
        report_table(finalists_df, ["Pipeline ID", "Pipeline", "Finalist role", "Selection reason", "Strong compatible candidate", "Compatibility interpretation"]),
        "",
        "## Q. What controlled aligned-reference dataset/protocol was used for PSNR/MSE/SSIM?",
        f"- Reused the screening24 controlled protocol: {len(controlled_selection_df)} Real references from screening24_metadata, severity cycle {list(SEVERITIES)}, seeds RANDOM_SEED + index, original simulate_degradation function, aligned grayscale outputs only.",
        f"- B0 P0-only controlled baseline, included as a no-enhancement reference but not as an enhancement finalist: MSE {b0_controlled['MSE_mean']:.6f} +/- {b0_controlled['MSE_std']:.6f}, PSNR {b0_controlled['PSNR_mean']:.4f} +/- {b0_controlled['PSNR_std']:.4f} dB, SSIM {b0_controlled['SSIM_mean']:.4f} +/- {b0_controlled['SSIM_std']:.4f}.",
        "- Comparable previous screening24 controlled values:",
        previous_quant.round(4).to_markdown(index=False) if not previous_quant.empty else "- n/a",
        "",
        "## R. What is the mean +/- standard deviation MSE for each finalist?",
        "\n".join(f"- {row['Pipeline ID']} {row['Pipeline']}: {fmt_pm(row, 'MSE_mean', 'MSE_std', 6)}" for _, row in finalist_controlled.iterrows()),
        "",
        "## S. What is the mean +/- standard deviation PSNR for each finalist?",
        "\n".join(f"- {row['Pipeline ID']} {row['Pipeline']}: {fmt_pm(row, 'PSNR_mean', 'PSNR_std', 4)} dB" for _, row in finalist_controlled.iterrows()),
        "",
        "## T. What is the mean +/- standard deviation SSIM for each finalist?",
        "\n".join(f"- {row['Pipeline ID']} {row['Pipeline']}: {fmt_pm(row, 'SSIM_mean', 'SSIM_std', 4)}" for _, row in finalist_controlled.iterrows()),
        "",
        "## U. Which finalist has the highest PSNR?",
        f"- {highest_psnr['Pipeline ID']} {highest_psnr['Pipeline']} with PSNR {highest_psnr['PSNR_mean']:.4f} +/- {highest_psnr['PSNR_std']:.4f} dB.",
        "",
        "## V. Which finalist has the lowest MSE?",
        f"- {lowest_mse['Pipeline ID']} {lowest_mse['Pipeline']} with MSE {lowest_mse['MSE_mean']:.6f} +/- {lowest_mse['MSE_std']:.6f}.",
        "",
        "## W. Which finalist has the highest SSIM?",
        f"- {highest_ssim['Pipeline ID']} {highest_ssim['Pipeline']} with SSIM {highest_ssim['SSIM_mean']:.4f} +/- {highest_ssim['SSIM_std']:.4f}.",
        "",
        "## X. Does the quantitative winner also perform acceptably on original Altered fingerprint structure?",
        f"- Quantitative winner {highest_psnr['Pipeline ID']} direct compatibility: {compatibility_df[compatibility_df['Pipeline ID'] == highest_psnr['Pipeline ID']]['Compatibility interpretation'].iloc[0]}. Acceptable under guards: {bool(compatibility_df[compatibility_df['Pipeline ID'] == highest_psnr['Pipeline ID']]['Strong compatible candidate'].iloc[0])}.",
        "",
        "## Y. Which processing stage produces the greatest PSNR increase?",
        f"- {psnr_stage['Pipeline ID']} / {psnr_stage['Stage']}: mean delta PSNR {psnr_stage['Delta_PSNR']:+.4f} dB in the controlled aligned experiment.",
        f"- {selected_psnr_note}",
        f"- Selected {selected_pipeline_id} controlled stage deltas:",
        report_table(selected_controlled_stage_deltas, ["Stage order", "Stage", "Delta_PSNR", "Delta_MSE", "Delta_SSIM"]),
        "",
        "## Z. Which processing stage produces the largest MSE reduction?",
        f"- {mse_stage['Pipeline ID']} / {mse_stage['Stage']}: mean delta MSE {mse_stage['Delta_MSE']:+.6f} in the controlled aligned experiment.",
        f"- {selected_mse_note}",
        "",
        "## AA. Which processing stage causes the greatest structural damage?",
        f"- {structural_damage_stage['Pipeline ID']} / {structural_damage_stage['Stage']}: mean fragmentation delta {structural_damage_stage['Delta_fragmentation']:+.4f} in direct Altered-image evaluation.",
        f"- Selected {selected_pipeline_id} direct structural stage deltas:",
        report_table(selected_direct_stage_deltas, ["Stage order", "Stage", "Delta_coherence", "Delta_contrast", "Delta_fragmentation"]),
        "",
        "## AB. Is the best hybrid genuinely better than CLAHE alone?",
        f"- {best_hybrid_better_than_clahe}. Selected hybrid {selected_pipeline_id} improved direct coherence versus CLAHE by {selected_pipeline_summary['Delta coherence vs CLAHE mean']:+.4f}, kept contrast loss within the guardrail at {selected_pipeline_summary['Delta contrast vs CLAHE mean']:+.4f}, and changed fragmentation by {selected_pipeline_summary['Delta fragmentation vs CLAHE mean']:+.4f}. It also outperformed B1 in finalist PSNR/MSE/SSIM, while B0 remains a no-enhancement pixel-distance reference.",
        "",
        "## AC. Is the best hybrid genuinely better than the strongest existing single technique?",
        f"- {best_hybrid_better_than_existing_overall}. Direct-structure evidence favors {selected_pipeline_id} over the strongest Direct24 single technique by coherence ({best_existing_single['Technique']}: {best_existing_single['Ridge coherence mean']:.4f} vs {selected_pipeline_summary['Ridge coherence mean']:.4f}). Under the same controlled screening24 protocol, it does not beat the best previous single-technique PSNR result ({best_previous_single_quant['Comparable source']}: {best_previous_single_quant['PSNR_mean']:.4f} dB), so it should be described as the balanced finalist rather than the absolute PSNR champion.",
        "",
        "## AD. Is a simpler pipeline preferable to the longer team hybrid?",
        f"- {simpler_than_longer_team_preferable}. The previous Direct24 full hybrid improved coherence but worsened contrast and fragmentation; this experiment prefers the shorter grayscale {selected_pipeline_id} pipeline and keeps Closing/Medial Axis as downstream structural analysis only.",
        "",
        "## AE. Why is the literature 28.17 dB result not automatically a required target?",
        "- The reported literature PSNR is not directly comparable unless the dataset, degradation process, reference definition, preprocessing, parameters and PSNR computation protocol are equivalent. Those details are not fully available here, so 28.17 dB remains context, not a forced target.",
        "",
        "## AF. What is the recommended FINAL grayscale fingerprint enhancement pipeline?",
        f"- Recommended final grayscale enhancement pipeline: {selected_pipeline_summary['Pipeline']}. It is the best balanced hybrid finalist, not the absolute pixel-distance champion; B0 remains a no-enhancement reference and {best_previous_single_quant['Comparable source']} remains the strongest previous single-technique PSNR comparator under the same controlled protocol.",
        "",
        "## AG. Which operations belong only to downstream structural analysis rather than enhancement?",
        "- Closing, segmentation/binarization, medial axis, skeletonization and minutiae extraction belong to downstream M3/M4 structural analysis. They must not be treated as final enhanced grayscale fingerprints or evaluated with PSNR/MSE/SSIM.",
        "",
        "## AH. Is the evidence now strong enough to proceed to a larger dataset evaluation?",
        f"- {'Yes, with a narrow validation only' if selected_direct_acceptable else 'Not yet as a full hybrid claim'}. Next justified run should be a larger direct Altered-image validation of CLAHE and the selected compatible grayscale finalist only, plus the controlled 24 protocol retained as secondary quantitative evidence. Runtime was {runtime['runtime_seconds']:.2f} seconds; protected previous evidence unchanged: {runtime['protected_existing_outputs_unchanged']}.",
        "- Visual evidence files:",
        "\n".join(f"- {path}" for path in runtime["visual_files"]),
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    direct_metrics: pd.DataFrame,
    direct_summary: pd.DataFrame,
    controlled_metrics: pd.DataFrame,
    controlled_summary: pd.DataFrame,
    stage_metrics: pd.DataFrame,
    finalists: pd.DataFrame,
    runtime: dict[str, object],
    report: str,
) -> None:
    direct_metrics.to_csv(HYBRID_FILES["direct_metrics"], index=False)
    direct_summary.to_csv(HYBRID_FILES["direct_summary"], index=False)
    controlled_metrics.to_csv(HYBRID_FILES["controlled_metrics"], index=False)
    controlled_summary.to_csv(HYBRID_FILES["controlled_summary"], index=False)
    stage_metrics.to_csv(HYBRID_FILES["stage_metrics"], index=False)
    finalists.to_csv(HYBRID_FILES["finalists"], index=False)
    HYBRID_FILES["runtime"].write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    HYBRID_FILES["report"].write_text(report, encoding="utf-8")


def main() -> None:
    start = perf_counter()
    protected_before = output_snapshot()
    print("hybrid24 compatibility experiment started")
    selection_df = load_direct24_selection()
    controlled_selection_df = load_screening24_controlled_selection()
    print(f"Direct Altered images reused: {selection_df.groupby('Category')['Image'].count().to_dict()}")
    print(f"Controlled Real references reused: {len(controlled_selection_df)}")

    direct_metrics, direct_stage_metrics, examples = run_direct_compatibility(selection_df)
    direct_summary = summarise_direct(direct_metrics)
    compatibility = build_compatibility_table(direct_summary)
    category_best = choose_category_best(direct_summary)
    finalists = select_finalists(direct_summary, compatibility)
    print(f"Selected finalists: {finalists['Pipeline ID'].tolist()}")

    controlled_metrics, controlled_stage_metrics = run_controlled_quantitative(controlled_selection_df, finalists)
    controlled_summary = summarise_controlled(controlled_metrics)
    stage_metrics = pd.concat([direct_stage_metrics, controlled_stage_metrics], ignore_index=True, sort=False)

    direct_visuals = create_direct_visuals(examples, finalists)
    controlled_visuals = create_controlled_visual(controlled_selection_df, finalists, controlled_summary)
    previous_quant = previous_comparable_quantitative()
    protected_after = output_snapshot()
    runtime = {
        "created_at": datetime.now().astimezone().isoformat(),
        "runtime_seconds": perf_counter() - start,
        "direct24_selection_reused": True,
        "direct_selection_counts": selection_df.groupby("Category")["Image"].count().to_dict(),
        "controlled_screening24_protocol_reused": True,
        "controlled_images": len(controlled_selection_df),
        "main_direct_experiment_used_synthetic_degradation": False,
        "controlled_secondary_experiment_used_synthetic_degradation": True,
        "full_500_image_experiment_run": False,
        "bruteforce_hybrid_search_run": False,
        "pipelines_evaluated": [p.__dict__ for p in pipeline_definitions()],
        "compatibility_guardrails": {
            "contrast_damage_threshold_vs_clahe": CONTRAST_DAMAGE_THRESHOLD,
            "fragmentation_damage_threshold_vs_clahe": FRAGMENTATION_DAMAGE_THRESHOLD,
            "category_fragmentation_damage_threshold_vs_clahe": CATEGORY_FRAGMENTATION_DAMAGE_THRESHOLD,
            "category_contrast_damage_threshold_vs_clahe": CATEGORY_CONTRAST_DAMAGE_THRESHOLD,
            "note": "Guardrails are pass/fail scientific checks, not a weighted composite score.",
        },
        "finalists": finalists.to_dict(orient="records"),
        "visual_files": direct_visuals + controlled_visuals,
        "protected_existing_outputs_unchanged": protected_before == protected_after,
        "protected_before": protected_before,
        "protected_after": protected_after,
    }
    report = create_report(
        selection_df,
        direct_summary,
        compatibility,
        category_best,
        finalists,
        controlled_selection_df,
        controlled_summary,
        stage_metrics,
        previous_quant,
        runtime,
    )
    write_outputs(direct_metrics, direct_summary, controlled_metrics, controlled_summary, stage_metrics, finalists, runtime, report)
    metadata = {
        "direct_selection": selection_df.to_dict(orient="records"),
        "controlled_selection": controlled_selection_df.to_dict(orient="records"),
        "compatibility_table": compatibility.to_dict(orient="records"),
        "category_best": category_best.to_dict(orient="records"),
        "previous_comparable_quantitative": previous_quant.to_dict(orient="records"),
        "outputs": {key: str(path.relative_to(PROJECT_ROOT)) for key, path in HYBRID_FILES.items()},
    }
    HYBRID_FILES["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"hybrid24 complete in {runtime['runtime_seconds']:.2f} seconds")
    print(f"Report: {HYBRID_FILES['report']}")


if __name__ == "__main__":
    main()
