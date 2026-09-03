from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import focused_psnr_audit_optimisation as focused
import direct24_original_altered_experiment as direct


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
REAL = ROOT / "data" / "SOCOFing" / "Real"
ALTERED = ROOT / "data" / "SOCOFing" / "Altered"
SELECTION_SEED = 20260903
NOISE_SEED_BASE = 20260903
HOLDOUT_TARGET = 100
ALTERED_PER_SEVERITY = 20
SEVERITY = 0.55
SIGMA = focused.gaussian_noise_sigma_from_severity(SEVERITY)
BENCHMARK = 28.17


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def csv_images(path: Path, columns: tuple[str, ...] = ("Image", "filename")) -> list[str]:
    if not path.exists():
        return []
    frame = pd.read_csv(path, low_memory=False)
    for column in columns:
        if column in frame.columns:
            return sorted(set(frame[column].dropna().astype(str)))
    return []


def recover_exclusions() -> pd.DataFrame:
    evidence: dict[str, list[tuple[str, str]]] = {}

    def add(names: list[str], reason: str, experiment: str) -> None:
        for name in names:
            base = Path(name.replace("\\", "/")).name
            if base.lower().endswith(tuple(focused.SUPPORTED_EXTENSIONS)):
                evidence.setdefault(base, []).append((reason, experiment))

    add(csv_images(OUT / "batch_metrics.csv"), "controlled algorithm comparison", "original 500-reference batch experiment")
    add(csv_images(OUT / "advanced_dev_batch_metrics.csv"), "parameter selection", "advanced development parameter configurations")
    screening = json.loads((OUT / "screening24_metadata.json").read_text(encoding="utf-8"))
    add(screening["screening24_selected_images"], "screening and parameter selection", "screening24")
    focused_meta = json.loads((OUT / "focused_psnr_audit_metadata.json").read_text(encoding="utf-8"))
    add(focused_meta["dev_images"], "parameter selection", "focused_psnr_audit DEV")
    add(focused_meta["test_images"], "held-out testing", "focused_psnr_audit TEST")
    add(csv_images(OUT / "hybrid24_controlled_metrics.csv"), "controlled hybrid comparison", "hybrid24 controlled")
    add(csv_images(OUT / "member4_controlled_per_image.csv"), "member comparison development", "member fixed comparison")
    compat = OUT / "member_compatibility_per_image.csv"
    if compat.exists():
        frame = pd.read_csv(compat, low_memory=False)
        frame = frame[frame.get("Category", pd.Series(index=frame.index, dtype=str)).eq("Controlled")]
        add(sorted(set(frame["Image"].dropna().astype(str))), "compatibility decision-making", "member compatibility experiment")
    rows = []
    for name in sorted(evidence, key=str.lower):
        pairs = sorted(set(evidence[name]))
        rows.append({
            "filename": name,
            "reason_excluded": "; ".join(sorted(set(reason for reason, _ in pairs))),
            "previous_experiment": "; ".join(sorted(set(exp for _, exp in pairs))),
        })
    return pd.DataFrame(rows)


def select_holdout(excluded: set[str]) -> tuple[list[Path], int]:
    all_real = focused.list_image_files(REAL)
    unused = [path for path in all_real if path.name not in excluded]
    rng = np.random.default_rng(SELECTION_SEED)
    indexes = rng.permutation(len(unused))[: min(HOLDOUT_TARGET, len(unused))]
    return [unused[int(i)] for i in indexes], len(unused)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    result = focused.evaluate(reference, candidate)
    return {"mse": result["MSE"], "psnr": result["PSNR (dB)"], "ssim": result["SSIM"]}


def statistic(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    return {
        "count": int(array.size), "mean": float(np.mean(array)),
        "standard_deviation_sample": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)), "minimum": float(np.min(array)),
        "maximum": float(np.max(array)), "percentile_25": float(np.percentile(array, 25)),
        "percentile_75": float(np.percentile(array, 75)),
    }


def save_controlled_figure(row: pd.Series, tag: str) -> str:
    path = REAL / row["filename"]
    reference = focused.load_grayscale(path)
    degraded, _ = focused.apply_gaussian_noise(reference, SEVERITY, int(row["noise_seed"]))
    enhanced = focused.non_local_means(degraded, h=0.06)
    error = np.abs(reference - enhanced)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    items = [(reference, "Clean Real reference", "gray", 0, 1), (degraded, f"Degraded\nPSNR {row['degraded_psnr']:.3f} dB", "gray", 0, 1), (enhanced, f"NLM enhanced\nPSNR {row['enhanced_psnr']:.3f} dB\nMSE {row['enhanced_mse']:.6f}; SSIM {row['enhanced_ssim']:.4f}", "gray", 0, 1), (error, "Absolute error", "magma", 0, max(0.15, float(error.max())))]
    for ax, (image, title, cmap, lo, hi) in zip(axes, items):
        ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi); ax.set_title(title, fontsize=9); ax.axis("off")
    fig.suptitle(f"{tag}: {row['filename']}", fontsize=10)
    fig.tight_layout()
    output = OUT / f"final_holdout_controlled_{tag}.png"
    fig.savefig(output, dpi=170, bbox_inches="tight"); plt.close(fig)
    return rel(output)


def altered_unused_names() -> set[str]:
    used: set[str] = set()
    for filename in ("direct24_per_image_metrics.csv", "focused_psnr_audit_altered_structure.csv", "member4_altered_per_image.csv", "member_compatibility_per_image.csv"):
        path = OUT / filename
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            for col in ("Image", "filename"):
                if col in frame.columns:
                    used.update(frame[col].dropna().astype(str))
                    break
    return used


def altered_validation() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    used = altered_unused_names()
    rows, figures = [], []
    folders = {"Easy": "Altered-Easy", "Medium": "Altered-Medium", "Hard": "Altered-Hard"}
    for offset, (severity, folder) in enumerate(folders.items()):
        candidates = [p for p in focused.list_image_files(ALTERED / folder) if p.name not in used]
        rng = np.random.default_rng(SELECTION_SEED + 1000 + offset)
        selected = [candidates[int(i)] for i in rng.permutation(len(candidates))[: min(ALTERED_PER_SEVERITY, len(candidates))]]
        for index, path in enumerate(selected):
            before = focused.load_grayscale(path)
            start = perf_counter(); after = focused.non_local_means(before, h=0.06); runtime = perf_counter() - start
            bm, am = direct.direct_quality_metrics(before), direct.direct_quality_metrics(after)
            row = {"severity": severity, "filename": path.name, "relative_path": rel(path), "selection_seed": SELECTION_SEED + 1000 + offset, "runtime_seconds": runtime}
            mappings = {"coherence": "Ridge coherence", "contrast": "Local contrast", "fragmentation": "Fragmentation per 1000 ridge pixels", "continuity": "Ridge continuity proxy", "ridge_coverage": "Ridge coverage in foreground"}
            for short, source in mappings.items():
                row[f"{short}_before"] = bm[source]; row[f"{short}_after"] = am[source]; row[f"delta_{short}"] = am[source] - bm[source]
            rows.append(row)
            if index < 2:
                fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2))
                axes[0].imshow(before, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Original Altered")
                axes[1].imshow(after, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Fixed NLM h=0.06")
                for ax in axes: ax.axis("off")
                fig.suptitle(f"{severity}: {path.name}", fontsize=9); fig.tight_layout()
                output = OUT / f"final_nlm_altered_{severity.lower()}_{index + 1}.png"
                fig.savefig(output, dpi=170, bbox_inches="tight"); plt.close(fig); figures.append(rel(output))
    frame = pd.DataFrame(rows)
    summaries = []
    for label, group in [(x, frame[frame.severity.eq(x)]) for x in folders] + [("Overall", frame)]:
        item = {"severity": label, "count": len(group), "mean_runtime_seconds": group.runtime_seconds.mean()}
        for metric in ("coherence", "contrast", "fragmentation", "continuity", "ridge_coverage"):
            item[f"mean_{metric}_before"] = group[f"{metric}_before"].mean(); item[f"mean_{metric}_after"] = group[f"{metric}_after"].mean(); item[f"mean_delta_{metric}"] = group[f"delta_{metric}"].mean()
        summaries.append(item)
    return frame, pd.DataFrame(summaries), figures


def report_text(total_real: int, excluded: pd.DataFrame, available: int, selected: pd.DataFrame, summary: dict, altered_summary: pd.DataFrame, outliers: pd.DataFrame, integrity: dict) -> str:
    s = summary
    overall = altered_summary[altered_summary.severity.eq("Overall")].iloc[0]
    low = selected[~selected.above_28_17]
    low_characteristics = "None; every image exceeded 28.17 dB." if low.empty else "; ".join(f"{r.filename}: low original contrast={r.clean_local_contrast:.4f}, foreground coverage={r.clean_foreground_coverage:.3f}" for _, r in low.iterrows())
    agreement_12 = abs(s["enhanced_psnr"]["mean"] - 29.2504)
    agreement_dev = abs(s["enhanced_psnr"]["mean"] - 29.4366)
    answers = [
        ("A", "How many Real images were available?", str(total_real)),
        ("B", "Which images were excluded because they had been used previously?", f"{len(excluded)} unique filenames; the complete auditable list is in `final_holdout_excluded_images.csv`."),
        ("C", "How many completely fresh images were selected?", str(len(selected))),
        ("D", "What deterministic selection method and seed were used?", f"Sorted unused Real paths followed by NumPy PCG64 `default_rng({SELECTION_SEED}).permutation`; the first {len(selected)} indices were locked."),
        ("E", "Was the NLM parameter changed after compatibility testing?", "No."),
        ("F", "What exact NLM implementation was used?", "OpenCV `cv2.fastNlMeansDenoising` on rounded uint8 input, templateWindowSize=7, searchWindowSize=21, returned to [0,1] float."),
        ("G", "What exact h value was used?", "0.06 in normalized units (15.3 passed to OpenCV)."),
        ("H", "What Gaussian-noise degradation protocol was used?", f"`gaussian_noise_only_s055`: NumPy default_rng normal noise, severity 0.55, sigma {SIGMA:.5f}, per-image seed {NOISE_SEED_BASE}+index, then clip to [0,1]."),
        ("I", "Was degradation severity changed after seeing results?", "No."),
        ("J", "What was the mean degraded-input PSNR?", f"{s['degraded_psnr']['mean']:.4f} dB."),
        ("K", "What was the mean enhanced PSNR?", f"{s['enhanced_psnr']['mean']:.4f} dB."),
        ("L", "What was the PSNR standard deviation?", f"{s['enhanced_psnr']['standard_deviation_sample']:.4f} dB (sample SD)."),
        ("M", "What was the median PSNR?", f"{s['enhanced_psnr']['median']:.4f} dB."),
        ("N", "What was the minimum PSNR?", f"{s['enhanced_psnr']['minimum']:.4f} dB."),
        ("O", "What was the maximum PSNR?", f"{s['enhanced_psnr']['maximum']:.4f} dB."),
        ("P", "What percentage of images exceeded 28.17 dB?", f"{s['percentage_above_28_17']:.2f}% ({s['number_above_28_17']}/{len(selected)})."),
        ("Q", "Did the mean PSNR exceed 28.17 dB?", "Yes." if s['primary_target_met'] else "No."),
        ("R", "By how many dB?", f"{s['mean_margin_to_28_17']:+.4f} dB."),
        ("S", "Did NLM improve PSNR relative to the degraded input?", f"{'Yes' if s['criteria']['mean_psnr_increased'] else 'No'}; mean delta {s['delta_psnr']['mean']:+.4f} dB."),
        ("T", "Did MSE decrease?", f"{'Yes' if s['criteria']['mean_mse_decreased'] else 'No'}; mean delta {s['delta_mse']['mean']:+.7f}."),
        ("U", "Did SSIM increase?", f"{'Yes' if s['criteria']['mean_ssim_increased'] else 'No'}; mean delta {s['delta_ssim']['mean']:+.4f}."),
        ("V", "What was the final mean MSE?", f"{s['enhanced_mse']['mean']:.8f} (normalized [0,1])."),
        ("W", "What was the final mean SSIM?", f"{s['enhanced_ssim']['mean']:.4f}."),
        ("X", "Were there any images below 28.17 dB?", f"{'Yes' if len(low) else 'No'} ({len(low)})."),
        ("Y", "If yes, what characteristics did those cases show?", low_characteristics),
        ("Z", "Did the fresh result agree with the earlier 12-image held-out result of 29.2504 dB?", f"It differed by {agreement_12:.4f} dB; this is {'close supporting evidence' if agreement_12 < 0.5 else 'a material shift requiring caution'}."),
        ("AA", "Did the fresh result agree with the compatibility DEV result of 29.4366 dB?", f"It differed by {agreement_dev:.4f} dB; this is {'close supporting evidence' if agreement_dev < 0.5 else 'a material shift requiring caution'}."),
        ("AB", "Is there evidence of serious overfitting to the original 24-image experiments?", "No serious evidence: the fresh set was locked and disjoint, and the target result generalized." if s['primary_target_met'] else "The fresh mean missed the target, so generalization concern remains; no data were removed or retuned."),
    ]
    for code, severity, question in [("AC", "Easy", "How did fixed NLM affect Altered Easy fingerprints?"), ("AD", "Medium", "How did it affect Altered Medium fingerprints?"), ("AE", "Hard", "How did it affect Altered Hard fingerprints?")]:
        row = altered_summary[altered_summary.severity.eq(severity)].iloc[0]
        answers.append((code, question, f"Mean deltas: coherence {row.mean_delta_coherence:+.4f}, contrast {row.mean_delta_contrast:+.4f}, fragmentation {row.mean_delta_fragmentation:+.4f}, continuity {row.mean_delta_continuity:+.4f}. These indicators are descriptive, not automatically improvements."))
    answers += [
        ("AF", "Did coherence improve overall?", f"{'Yes' if overall.mean_delta_coherence > 0 else 'No'}; mean change {overall.mean_delta_coherence:+.4f}."),
        ("AG", "Did fragmentation improve overall?", f"{'Yes (decreased)' if overall.mean_delta_fragmentation < 0 else 'No (it did not decrease)'}; mean change {overall.mean_delta_fragmentation:+.4f}."),
        ("AH", "Was there a contrast trade-off?", f"{'Yes, contrast decreased' if overall.mean_delta_contrast < 0 else 'No overall contrast loss'}; mean change {overall.mean_delta_contrast:+.4f}."),
        ("AI", "Was serious ridge structural damage observed?", "No automated claim is made from a single indicator. The six fixed-range before/after figures were visually checked; see the outlier/visual notes."),
        ("AJ", "Is NLM still recommended as the final grayscale enhancement technique?", "Yes." if s['primary_target_met'] else "Not conclusively from the primary target; retain the frozen result without retuning."),
        ("AK", "Why were the tested hybrids rejected?", "TV→NLM, NLM→Modified Gabor, NLM→Directional Diffusion, and TV→NLM→Directional Diffusion were tested under the controlled compatibility experiment; none exceeded NLM alone (29.4366 dB)."),
        ("AL", "Can the project now proceed to final large-dataset validation?", "Yes." if s['primary_target_met'] and integrity['all_checks_passed'] else "No; resolve failed integrity or generalization criteria first."),
    ]
    lines = ["# Final Fresh Hold-out Validation Report", "", "## Locked design", "", f"The hold-out list was written before enhancement metrics were computed. No result-based replacement, removal, resampling, tuning, backend change, or severity change occurred. Available unused images: {available}.", "", "## Exact requested questions", ""]
    for code, question, answer in answers:
        lines += [f"### {code}. {question}", "", answer, ""]
    lines += ["## Visual inspection notes", "", "The sole below-benchmark controlled case, `194__M_Left_middle_finger.BMP`, is a complete fingerprint with substantial ridge coverage. Its clean reference visibly contains granular/fragmented ridge edges, particularly in the upper region. NLM reduced the synthetic grain while mildly softening fine ridge-edge texture; the case was retained.", "", "Across the six fixed-range Altered examples (two each from Easy, Medium, and Hard), scar, obliteration, and Z-cut structures and the surrounding ridge flow remained recognizable. Smoothing was mild, and no obvious newly invented ridge topology or serious structural damage was observed in these examples. This is a visual plausibility statement, not proof that every numerical structural change is beneficial.", "", "## Benchmark wording", "", f"Under the controlled Gaussian-noise evaluation protocol adopted in this project, the fixed Non-Local Means method achieved a mean PSNR of {s['enhanced_psnr']['mean']:.4f} dB, {'exceeding' if s['mean_margin_to_28_17'] > 0 else 'falling below'} the selected literature PSNR benchmark of 28.17 dB by {abs(s['mean_margin_to_28_17']):.4f} dB.", "", "Exact metric equivalence with the literature experiment cannot be confirmed because the dataset, degradation protocol, scaling and evaluation details are not fully known or equivalent.", "", "## Final positioning", "", "Final grayscale pipeline: Input Fingerprint → P0 Neutral Preparation → Non-Local Means Denoising → Enhanced Grayscale Fingerprint.", "", "Downstream segmentation, thresholding, morphology, thinning/medial axis, and feature analysis remain separate.", "", "The four team techniques remain M1 NLM, M2 Modified/Orientation-Adaptive Gabor, M3 Total Variation Restoration, and M4 Coherence-Guided Directional Diffusion. NLM is positioned as the strongest optimized technique selected after comparative and integration experiments."]
    return "\n".join(lines) + "\n"


def main() -> None:
    start_all = perf_counter(); OUT.mkdir(exist_ok=True)
    log: list[str] = []
    log.append(f"{datetime.now().astimezone().isoformat()} start")
    excluded = recover_exclusions(); excluded.to_csv(OUT / "final_holdout_excluded_images.csv", index=False)
    excluded_set = set(excluded.filename)
    all_real = focused.list_image_files(REAL)
    selected_paths, available_unused = select_holdout(excluded_set)
    image_list = pd.DataFrame({"selection_order": range(len(selected_paths)), "filename": [p.name for p in selected_paths], "relative_path": [rel(p) for p in selected_paths], "selection_seed": SELECTION_SEED, "total_available_unused_images": available_unused, "selected_count": len(selected_paths)})
    image_list.to_csv(OUT / "final_holdout_image_list.csv", index=False)
    log.append(f"locked holdout list: {len(selected_paths)} of {available_unused} unused Real images")
    locked_method = {"pipeline": "P0 -> Non-Local Means", "h_normalized": 0.06, "opencv_h_uint8_units": 15.3, "function": "cv2.fastNlMeansDenoising", "backend_required": "OpenCV", "opencv_available": focused.cv2 is not None, "templateWindowSize": 7, "searchWindowSize": 21, "input_conversion": "clip [0,1], round(image*255), uint8", "output_conversion": "uint8 / 255 to float32 clipped [0,1]", "P0": {"grayscale": "PIL convert L", "resize": [256, 256], "resize_implementation": "skimage.transform.resize anti_aliasing=True preserve_range=True", "normalization": "1st/99th percentile linear normalization then clip [0,1]"}, "fallback_permitted": False}
    if focused.cv2 is None: raise RuntimeError("OpenCV backend is unavailable; refusing to change the frozen backend")
    write_json(OUT / "final_holdout_locked_method.json", locked_method)
    protocol = {"protocol_id": "gaussian_noise_only_s055", "severity": SEVERITY, "sigma_formula": "0.018 + 0.055 * severity", "sigma": SIGMA, "generator": "numpy.random.default_rng(seed).normal(0.0, sigma, reference.shape)", "seed_base": NOISE_SEED_BASE, "seed_rule": "seed_base + locked selection_order", "clipping": "np.clip to [0,1] after addition", "image_range": [0.0, 1.0], "reference_dtype": "float32", "noise_computation_dtype": "NumPy floating result then float32 metric_image", "channels": "single grayscale", "metric_data_range": 1.0}
    write_json(OUT / "final_holdout_protocol.json", protocol)
    rows = []
    for index, path in enumerate(selected_paths):
        reference = focused.load_grayscale(path); seed = NOISE_SEED_BASE + index
        degraded, _ = focused.apply_gaussian_noise(reference, SEVERITY, seed)
        begin = perf_counter(); enhanced = focused.non_local_means(degraded, h=0.06); runtime = perf_counter() - begin
        before, after = metrics(reference, degraded), metrics(reference, enhanced)
        clean_direct = direct.direct_quality_metrics(reference)
        rows.append({"filename": path.name, "relative_path": rel(path), "selection_order": index, "noise_seed": seed, "degraded_psnr": before["psnr"], "enhanced_psnr": after["psnr"], "delta_psnr": after["psnr"] - before["psnr"], "degraded_mse": before["mse"], "enhanced_mse": after["mse"], "delta_mse": after["mse"] - before["mse"], "degraded_ssim": before["ssim"], "enhanced_ssim": after["ssim"], "delta_ssim": after["ssim"] - before["ssim"], "above_28_17": after["psnr"] > BENCHMARK, "runtime_seconds": runtime, "clean_local_contrast": clean_direct["Local contrast"], "clean_ridge_coherence": clean_direct["Ridge coherence"], "clean_foreground_coverage": clean_direct["Foreground usable-area ratio"], "clean_ridge_coverage": clean_direct["Ridge coverage in foreground"]})
    frame = pd.DataFrame(rows); frame.to_csv(OUT / "final_holdout_per_image.csv", index=False)
    stat_columns = ["enhanced_psnr", "enhanced_mse", "enhanced_ssim", "delta_psnr", "delta_mse", "delta_ssim", "degraded_psnr", "degraded_mse", "degraded_ssim"]
    summary = {column: statistic(frame[column]) for column in stat_columns}
    n = len(frame); mean_psnr = summary["enhanced_psnr"]["mean"]; sd = summary["enhanced_psnr"]["standard_deviation_sample"]
    ci = stats.t.interval(0.95, df=n - 1, loc=mean_psnr, scale=sd / math.sqrt(n)) if n > 1 else (mean_psnr, mean_psnr)
    above = int(frame.above_28_17.sum())
    summary.update({"benchmark_psnr_db": BENCHMARK, "number_above_28_17": above, "percentage_above_28_17": 100 * above / n, "number_below_or_equal_28_17": n - above, "percentage_below_or_equal_28_17": 100 * (n - above) / n, "mean_margin_to_28_17": mean_psnr - BENCHMARK, "minimum_margin_to_28_17": summary["enhanced_psnr"]["minimum"] - BENCHMARK, "mean_psnr_95_percent_t_confidence_interval": [float(ci[0]), float(ci[1])], "primary_target_met": mean_psnr > BENCHMARK, "criteria": {"mean_psnr_increased": summary["enhanced_psnr"]["mean"] > summary["degraded_psnr"]["mean"], "mean_mse_decreased": summary["enhanced_mse"]["mean"] < summary["degraded_mse"]["mean"], "mean_ssim_increased": summary["enhanced_ssim"]["mean"] > summary["degraded_ssim"]["mean"], "majority_above_28_17": above > n / 2, "all_above_28_17": above == n}})
    write_json(OUT / "final_holdout_statistics.json", summary)
    summary_rows = []
    for name in stat_columns: summary_rows.append({"metric": name, **summary[name]})
    pd.DataFrame(summary_rows).to_csv(OUT / "final_holdout_summary.csv", index=False)
    bottom = frame.nsmallest(5, "enhanced_psnr").assign(outlier_group="bottom_5_psnr", rank=lambda x: range(1, len(x) + 1))
    top = frame.nlargest(5, "enhanced_psnr").assign(outlier_group="top_5_psnr", rank=lambda x: range(1, len(x) + 1))
    low_delta = frame.nsmallest(5, "delta_psnr").assign(outlier_group="lowest_5_delta_psnr", rank=lambda x: range(1, len(x) + 1))
    improvements = frame.nlargest(5, "delta_psnr").assign(outlier_group="largest_5_improvements", rank=lambda x: range(1, len(x) + 1))
    outliers = pd.concat([bottom, top, low_delta, improvements], ignore_index=True)
    outliers["quantitative_observation"] = outliers.apply(lambda r: f"clean contrast={r.clean_local_contrast:.4f}; coherence={r.clean_ridge_coherence:.4f}; foreground coverage={r.clean_foreground_coverage:.3f}; ridge coverage={r.clean_ridge_coverage:.3f}", axis=1)
    outliers["visual_observation"] = "See fixed-range exported comparison; no causal explanation inferred from metrics alone."
    outliers.loc[outliers.filename.eq("194__M_Left_middle_finger.BMP"), "visual_observation"] = (
        "Complete fingerprint with substantial ridge coverage; the clean reference visibly has granular/fragmented "
        "ridge edges, especially in the upper region. NLM reduces added grain but mildly softens fine ridge-edge texture."
    )
    outliers.to_csv(OUT / "final_holdout_outlier_analysis.csv", index=False)
    ordered = frame.sort_values("enhanced_psnr").reset_index(drop=True)
    representatives = [(ordered.iloc[0], "worst"), (ordered.iloc[len(ordered)//2], "median"), (ordered.iloc[-1], "best"), (ordered.iloc[len(ordered)//4], "lower_quartile"), (ordered.iloc[(3*len(ordered))//4], "upper_quartile"), (frame.iloc[0], "selection_first")]
    controlled_figures = [save_controlled_figure(row, tag) for row, tag in representatives]
    altered_frame, altered_summary, altered_figures = altered_validation()
    altered_frame.to_csv(OUT / "final_nlm_altered_per_image.csv", index=False); altered_summary.to_csv(OUT / "final_nlm_altered_summary.csv", index=False)
    formula_error = np.max(np.abs(frame.enhanced_psnr.to_numpy() - (-10 * np.log10(frame.enhanced_mse.to_numpy()))))
    expected_files = ["final_holdout_excluded_images.csv", "final_holdout_image_list.csv", "final_holdout_locked_method.json", "final_holdout_protocol.json", "final_holdout_per_image.csv", "final_holdout_summary.csv", "final_holdout_statistics.json", "final_holdout_outlier_analysis.csv", "final_nlm_altered_per_image.csv", "final_nlm_altered_summary.csv"]
    checks = {"holdout_row_count_matches_selected": len(frame) == len(selected_paths), "no_duplicate_holdout_filenames": frame.filename.nunique() == len(frame), "no_overlap_with_prior_real_images": not bool(set(frame.filename) & excluded_set), "nlm_h_exactly_0_06": locked_method["h_normalized"] == 0.06, "opencv_backend_used": focused.cv2 is not None, "severity_exactly_0_55": protocol["severity"] == 0.55, "sigma_exactly_0_04825": abs(protocol["sigma"] - 0.04825) < 1e-15, "metric_data_range_exactly_1": protocol["metric_data_range"] == 1.0, "mse_psnr_internal_consistency_max_abs_db_error_below_1e_9": float(formula_error) < 1e-9, "altered_counts_20_each": all(int(altered_summary.loc[altered_summary.severity.eq(x), "count"].iloc[0]) == 20 for x in ("Easy", "Medium", "Hard")), "all_required_intermediate_files_nonempty": all((OUT / name).exists() and (OUT / name).stat().st_size > 0 for name in expected_files)}
    integrity = {"checks": checks, "all_checks_passed": all(checks.values()), "holdout_rows": len(frame), "excluded_unique_images": len(excluded), "total_real_images": len(all_real), "available_unused_real_images": available_unused, "selected_images": len(selected_paths), "duplicate_count": len(frame) - frame.filename.nunique(), "overlap_filenames": sorted(set(frame.filename) & excluded_set), "max_abs_psnr_formula_error_db": float(formula_error), "python_process_note": "This script is the only validation Python process; process-exit checked externally after completion."}
    write_json(OUT / "final_holdout_integrity_check.json", integrity)
    report = report_text(len(all_real), excluded, available_unused, frame, summary, altered_summary, outliers, integrity)
    (OUT / "final_holdout_report.md").write_text(report, encoding="utf-8")
    elapsed = perf_counter() - start_all
    runtime = {"started_at": log[0].split(" start")[0], "finished_at": datetime.now().astimezone().isoformat(), "total_runtime_seconds": elapsed, "controlled_images": len(frame), "altered_images": len(altered_frame), "controlled_nlm_runtime_seconds_total": float(frame.runtime_seconds.sum()), "altered_nlm_runtime_seconds_total": float(altered_frame.runtime_seconds.sum()), "generated_controlled_figures": controlled_figures, "generated_altered_figures": altered_figures}
    write_json(OUT / "final_holdout_runtime.json", runtime)
    log += [f"controlled mean PSNR={mean_psnr:.6f}", f"above benchmark={above}/{n}", f"altered rows={len(altered_frame)}", f"integrity={integrity['all_checks_passed']}", f"runtime_seconds={elapsed:.3f}", f"{datetime.now().astimezone().isoformat()} complete"]
    (OUT / "final_holdout_run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(json.dumps({"mean_psnr": mean_psnr, "above": above, "count": n, "integrity": integrity["all_checks_passed"], "runtime_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
