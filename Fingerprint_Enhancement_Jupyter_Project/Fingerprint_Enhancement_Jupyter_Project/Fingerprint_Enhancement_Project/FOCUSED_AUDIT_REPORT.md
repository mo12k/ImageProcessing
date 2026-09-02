# Focused Audit Report - Fingerprint Enhancement Project

Date: 2026-09-01

This audit did not rerun the 500-image experiment. The existing output evidence was preserved in `backups/audit_preserve_2026-09-01_184155/` before source edits.

## A. Current 500-Image Verification

The latest formal experiment evidence is the saved notebook output, `outputs/batch_metrics.csv`, `outputs/summary_metrics.csv`, and `outputs/experiment_metadata.json`.

- `development_mode`: `false` for the saved formal experiment.
- `max_batch_images`: `500`.
- `active_batch_images`: `500`.
- `number_of_real_images_available`: `6000`.
- `number_of_reference_images`: `500`.
- CSV rows: `3000`, equal to 500 images x 6 method rows.
- Unique evaluated images: `500`.
- First image: `100__M_Left_index_finger.BMP`.
- Last image: `145__M_Right_thumb_finger.BMP`.
- Sampling method: sorted SOCOFing `Real` image paths, taking the first `MAX_BATCH_IMAGES` unique files.
- Degradation severities: `0.35`, `0.55`, `0.75`, cycled by image index.
- Degradation seed formula: `RANDOM_SEED + image_index`, with `RANDOM_SEED = 42`.
- Formal batch runtime: `2125.45` seconds; total notebook runtime: `2152.32` seconds.

Important source-state note: before correction, the saved notebook source/output reflected the 500-image final run, while the paired `Fingerprint_Enhancement_System.py` file was still in development-safe mode. After correction, both source files are development-safe by default and final export is disabled unless explicitly approved.

## B. Cumulative PSNR From Existing CSV Only

These values were calculated from the existing `outputs/batch_metrics.csv` without reprocessing images.

| Images | Degraded Input | Global HE Baseline | CLAHE Baseline | M1 - Wiener + CLAHE Enhancement | M2 - Gabor / Modified Gabor Ridge Enhancement | Team Hybrid Pipeline |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 15.5893 | 15.5981 | 10.7384 | 13.2037 | 11.3246 | 13.4306 |
| 50 | 15.7255 | 15.2980 | 10.5983 | 13.1492 | 11.1568 | 13.2807 |
| 100 | 15.7285 | 15.2987 | 10.4552 | 13.1067 | 11.1893 | 13.3418 |
| 250 | 15.8343 | 15.1439 | 10.3830 | 13.2898 | 11.1949 | 13.4240 |
| 500 | 15.8575 | 14.9824 | 10.3089 | 13.3338 | 11.2151 | 13.4970 |

The current first 24 images are already low. The low 500-image mean is not caused by a progressive collapse as more images are added.

## C. Why The Old ~28 dB Result Disappeared

The available saved evidence does not reproduce or contain a 28 dB run.

- The current 500-image first-24 subset already has low PSNR: Team Hybrid Pipeline `13.4306 dB`, M1 `13.2037 dB`, M2 `11.3246 dB`.
- The available 24-image backup checkpoint also had low PSNR: Degraded input `16.4481 dB`, Global HE `16.0137 dB`, CLAHE `11.7415 dB`, Gabor `11.7119 dB`, Hybrid `12.3307 dB`.
- The available 24-image backup used the same legitimate full-reference metric pattern: `mean_squared_error`, `peak_signal_noise_ratio(..., data_range=1.0)`, `structural_similarity(..., data_range=1.0)`.
- The current degradation model is strong: blur, reduced dynamic range, illumination variation, Gaussian noise, impulse noise, and smudge/occlusion.

Based on available files, the old ~28 dB result most likely came from an unavailable or different experiment setup: weaker/no degradation, synthetic/easier data, a different reference/candidate pairing, independent contrast normalisation before metrics, comparing processed output to degraded input instead of clean reference, or a single-image/result-display misunderstanding. It is not supported by the current CSVs, notebook outputs, metadata, or available backup outputs.

## D. Current Metric Calculation Verification

- PSNR reference image: clean SOCOFing `Real` fingerprint loaded as grayscale, resized to `256 x 256`, and normalised during loading.
- PSNR processed image: each aligned grayscale candidate output from `run_comparison`.
- M3 binary ridges and M4 skeleton/minutiae are not direct greyscale PSNR enhancement outputs.
- MSE implementation: `skimage.metrics.mean_squared_error(reference, candidate)`.
- PSNR implementation: `skimage.metrics.peak_signal_noise_ratio(reference, candidate, data_range=1.0)`.
- SSIM implementation: `skimage.metrics.structural_similarity(reference, candidate, data_range=1.0)`.
- Metric image range: `[0, 1]`.
- Metric preprocessing: `nan_to_num` and clipping to `[0, 1]`.
- No independent contrast renormalisation is applied immediately before MSE/PSNR/SSIM.
- No metric scaling, hardcoding, multiplication, or fabricated benchmark logic was added.

## E. Current Four Member Techniques

- M1 - Wiener + CLAHE Enhancement.
- M2 - Gabor / Modified Gabor Ridge Enhancement.
- M3 - Morphological Ridge Restoration.
- M4 - Thinning & Minutiae Extraction.

## F. Baselines Separated From Member Techniques

The corrected source separates:

- Degraded Input: experimental control.
- Global HE Baseline: baseline.
- CLAHE Baseline: baseline and M1 comparison reference.

These are not listed as individual member techniques.

## G. Team Hybrid Separated

`Team Hybrid Pipeline` is now a separate team-level method label, not Member 3 or Member 4.

## H. Figures And Labels Corrected

The source now defines the requested figure structure:

- Figure 1: `plot_individual_contributions`.
- Figure 2: `plot_enhancement_quality_comparison`.
- Figure 3: `plot_member1_enhancement_study`.
- Figure 4: `plot_member2_ridge_enhancement_study`.
- Figure 5: `plot_member3_ridge_restoration_study`.
- Figure 6: `plot_member4_feature_results`.
- Figure 7: `plot_team_hybrid_evaluation`.

The generated PDF export function now uses these role-aware figures when a future approved export is run. Existing saved output files were not overwritten.

## I. Member 1 Evidence

On the current 500-image CSV:

- CLAHE Baseline PSNR mean: `10.3089 dB`.
- M1 - Wiener + CLAHE Enhancement PSNR mean: `13.3338 dB`.
- Mean PSNR gain over CLAHE Baseline: `+3.0249 dB`.
- M1 improves over CLAHE Baseline on PSNR for `500/500` images.
- CLAHE Baseline SSIM mean: `0.4489`; M1 SSIM mean: `0.5670`.
- CLAHE Baseline coherence mean: `0.5480`; M1 coherence mean: `0.6929`.

M1 is evidence-backed versus the CLAHE baseline, but it still does not exceed the 28.17 dB literature benchmark and does not beat Degraded Input or Global HE on PSNR.

## J. Member 2 Evidence

The current Gabor implementation is not a true adaptive Modified Gabor filter. It uses:

- Fixed global frequency: `0.115`.
- Fixed orientation bank count: `8`.
- Max response fusion across orientations.
- Fixed blend values: `0.58` for M2 output and `0.30` inside Team Hybrid Pipeline.

Current 500-image M2 results:

- PSNR mean: `11.2151 dB`.
- SSIM mean: `0.4014`.
- Coherence mean: `0.5728`.

No separate improved/adaptive Modified Gabor result exists yet in the current saved evidence.

## K. Member 3 Evidence

Member 3 currently produces segmentation, Sauvola thresholding, and morphology-based binary ridge restoration. The corrected source now preserves a `raw_binary` intermediate and provides a before/after morphology figure. No direct greyscale PSNR claim should be made for M3 binary ridges.

## L. Member 4 Evidence

The current CSV contains Team Hybrid Pipeline minutiae counts:

- Average detected ridge endings: `121.22`.
- Average detected bifurcations: `45.37`.

These are counts, not accuracy. The current saved CSV repeats the same counts on every method row, so future source has been corrected to attach those counts only to `Team Hybrid Pipeline` rows and summarize them separately as Member 4 feature output.

## M. Technical Options To Legitimately Exceed 28.17 dB

- First verify the benchmark protocol: dataset, degradation level, whether PSNR is full image or ROI/masked, whether the reference is clean/original or a restored target, and whether intensity normalisation is allowed by the paper.
- Implement adaptive Modified Gabor: local orientation estimation from the structure tensor, local ridge frequency estimation by block projection/FFT, segmentation-mask gating, and per-block Gabor filtering.
- Tune Gabor strength legitimately: frequency bounds, orientation smoothing, sigma/bandwidth, number of orientations, and blend with the denoised image.
- Tune M1 restoration: Wiener window size, noise estimate, CLAHE clip limit, CLAHE tile size, and optional mild denoising before contrast enhancement.
- Test sequence variants: denoise -> CLAHE -> adaptive Gabor; denoise -> adaptive Gabor -> CLAHE; adaptive Gabor on M1 output versus degraded input.
- Avoid over-enhancement: high CLAHE or high Gabor blend can improve visible ridges while hurting PSNR/SSIM.
- Consider reporting both full-image and ROI metrics only if the literature uses ROI metrics; do not replace the main metric silently.
- If the current degradation is harsher than the literature benchmark, align the experimental degradation protocol to the paper and disclose it. Do not weaken degradation merely to inflate PSNR.

## N. Exact Files Changed

- `Fingerprint_Enhancement_System.py`: corrected labels, safe development/export defaults, Figure 1-7 plotting functions, role-aware PDF export, M3 intermediate output, Member 4 feature summary, and future minutiae-count placement.
- `Fingerprint_Enhancement_System.ipynb`: synced from the corrected Python source and cleared stale outputs/widget state.
- `README_FIRST.md`: updated running/export instructions, member mapping, SVM status, and minutiae wording.
- `FOCUSED_AUDIT_REPORT.md`: this audit report.
- `backups/audit_preserve_2026-09-01_184155/`: preservation copy of the pre-edit notebook, Python source, CSVs, metadata, generated PDF, and overlay PNG.

Pre-existing modified output files in `outputs/` were not overwritten during this correction.

## O. Exact Proposed Next Experiment

Do not run 500 images next.

1. Stage 1: run the corrected notebook in development mode on the fixed first 24-image subset.
2. Add one clearly named candidate, such as `M2 - Adaptive Modified Gabor Ridge Enhancement` or `Team Hybrid Pipeline v2`, without removing the current baseline methods.
3. Grid-search only legitimate image-processing parameters: Wiener window/noise estimate, CLAHE clip/tile size, adaptive Gabor orientation/frequency parameters, Gabor blend, and pipeline order.
4. Report mean PSNR, standard deviation, minimum, maximum, and evaluated image count.
5. Move to 50 images only if the 24-image mean PSNR exceeds `28.17 dB`.
6. Move to 100 images only if the 50-image result remains stable.
7. Request explicit approval before any new 500-image final run or final-looking output overwrite.
