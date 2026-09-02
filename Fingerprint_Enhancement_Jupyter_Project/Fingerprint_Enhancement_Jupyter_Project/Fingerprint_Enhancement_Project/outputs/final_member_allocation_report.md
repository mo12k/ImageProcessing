# Final Member Allocation Report

Generated: 2026-09-02T13:07:37+00:00

Scope: reporting-only consolidation of existing repository evidence. No 500-image validation, no brute-force hybrid search, and no new image enhancement experiment was run by this consolidation step.

## Assignment Mode

Mode A: Comparative & Enhancement Study.

## Frozen Member Table

| Member | Main Contemporary Technique | Main Purpose | Baseline / Comparator |
| --- | --- | --- | --- |
| Member 1 | Non-Local Means Denoising | Ridge-preserving noise reduction | Gaussian / Median / CLAHE |
| Member 2 | Modified / Orientation-Adaptive Gabor | Ridge enhancement | Ordinary Gabor |
| Member 3 | Total Variation Restoration | Ridge/edge-preserving restoration | Wiener / Gaussian |
| Member 4 | Coherence-Guided Directional Diffusion | Orientation-aware ridge-flow enhancement | conventional directional/ridge methods |

Supporting downstream techniques remain separate from the four main member techniques:

- thresholding
- closing
- thinning
- medial axis
- minutiae extraction

## Implementation Audit

| Audit item | Confirmed implementation | Source | Backend/library | Parameters | Fallbacks or label risk | Action in this report |
| --- | --- | --- | --- | --- | --- | --- |
| Main notebook and script exports | Main notebook and Python scripts are present | Fingerprint_Enhancement_System.ipynb; Fingerprint_Enhancement_System.py; screening24_master_audit.py; direct24_original_altered_experiment.py; hybrid24_compatibility_experiment.py; focused_psnr_audit_optimisation.py | Project-local notebook/script workflow | Existing outputs are under outputs/screening24*, outputs/direct24*, outputs/hybrid24*, and outputs/focused_psnr_audit* | Fingerprint_Enhancement_System.py still contains older member labels and should be treated as legacy wording until refreshed. | Use the new registry files as the source of truth for member allocation. |
| Common preprocessing | P0 = grayscale + resize + float [0,1] only | screening24_master_audit.py: common_preprocessing_variants; direct24 report neutral preprocessing | skimage/io/numpy image conversion | Resize to common processing size where required; no enhancement in P0 | P1-P3 exist as preprocessing audit variants but are not the selected common preprocessing. | Keep P0 as the only neutral common preprocessing. |
| Member 1 NLM | Real NLM implementation exists | focused_psnr_audit_optimisation.py: non_local_means; screening24_master_audit.py: D4 | OpenCV in focused final run; skimage NLM fallback | h=0.06 for focused winning result; screening D4 h=0.055 | Backend changes if cv2 is absent; output labels should preserve backend and h. | Freeze M1 as NLM, with CLAHE/Gaussian/Median as baselines. |
| Member 2 Modified/Adaptive Gabor | Adaptive Gabor variants and Ordinary Gabor baseline exist | screening24_master_audit.py: G1-G4; Fingerprint_Enhancement_System.py: adaptive_gabor_enhancement | skimage.filters.gabor with orientation/frequency estimates | Ordinary frequency 0.115 with 8 orientations; adaptive frequency candidates 0.075-0.140, blend around 0.18-0.22 | Article-specific parameters are unknown; current G2 is literature-inspired, not exact replication. | Freeze M2 as Modified / Orientation-Adaptive Gabor and keep Ordinary Gabor as comparator. |
| Member 3 TV Restoration | TV restoration exists as grayscale restoration | screening24_master_audit.py: R3; focused_psnr_audit_optimisation.py: tv_restoration | skimage.restoration.denoise_tv_chambolle | screening R3 weight=0.045; focused selected weight=0.02 for Gaussian-noise-only cases | Wavelet helper in the main app can fall back to TV if PyWavelets is missing. | Freeze M3 as TV Restoration; morphology becomes supporting structural processing. |
| Member 4 Coherence-Guided Directional Diffusion | Coherence-guided directional diffusion exists | screening24_master_audit.py: G7; Fingerprint_Enhancement_System.py: coherence_guided_diffusion | NumPy/SciPy/skimage structure estimates and oriented smoothing | screening iterations=4, step=0.16; system variant iterations=2, step=0.18, orientation_bins=6 | Old Direct24 labels group it under M2/Ridge Enhancement; new registry assigns it to M4. | Freeze M4 as Coherence-Guided Directional Diffusion; thinning becomes downstream. |
| Literature benchmark context | Local documentation contains the benchmark score but not enough settings for exact replication | outputs/screening24_metadata.json literature_context; specification/fingerprint enhancement.pdf | Local PDF context extracted by screening24_master_audit.py: extract_literature_context | Modified Gabor Filter PSNR=28.17 dB, MSE=22.27 | Dataset, image count, exact degradation, MAX_I, MSE scaling, and full metric formula are unknown. | Treat PSNR > 28.17 dB as a hard target but do not claim article reproduction. |

## Member Registry Summary

Full files written:

- `outputs/member_technique_registry.csv`
- `outputs/member_technique_registry.json`

| Member | Main Contemporary Technique | Main Purpose | Baseline / Comparator | Output type | Backend/library | Fallback risk |
| --- | --- | --- | --- | --- | --- | --- |
| Member 1 | Non-Local Means Denoising | Ridge-preserving noise reduction | Gaussian / Median / CLAHE | Aligned grayscale float image in [0,1] | OpenCV cv2.fastNlMeansDenoising in focused final run; skimage fallback available | Focused function falls back to skimage NLM if cv2 is unavailable; report backend explicitly. |
| Member 2 | Modified / Orientation-Adaptive Gabor | Ridge enhancement | Ordinary Gabor | Enhanced grayscale ridge image in [0,1] | skimage.filters.gabor plus local orientation/frequency estimates | No confirmed paper parameters/dataset/formula in repository; label as Literature-Inspired Modified Gabor when discussing article relation. |
| Member 3 | Total Variation Restoration | Ridge/edge-preserving restoration | Wiener / Gaussian | Aligned grayscale float image in [0,1] | skimage.restoration.denoise_tv_chambolle | Fingerprint_Enhancement_System.py wavelet_denoise can fall back to TV if PyWavelets is unavailable; labels must match actual branch. |
| Member 4 | Coherence-Guided Directional Diffusion | Orientation-aware ridge-flow enhancement | conventional directional/ridge methods | Enhanced grayscale ridge-flow image in [0,1] | NumPy/SciPy/skimage local structure estimates and oriented smoothing | No silent replacement observed in the audited functions. |

## Baseline Classification

Full file written: `outputs/baseline_method_classification.csv`

| Method | Classification | Allowed role | Member-technique status | Controlled PSNR if available |
| --- | --- | --- | --- | --- |
| CLAHE | Basic contrast-enhancement baseline | M1 baseline; optional contrast comparator | Not a main contribution | 10.4546 |
| Contrast Stretching | Basic/global contrast baseline | Contrast baseline and diagnostic comparator | Not a main contribution | 17.1119 |
| Gaussian Filter | Basic smoothing baseline | M1/M3 smoothing comparator | Not a main contribution | 15.9146 |
| Median Filter | Basic impulse-noise baseline | M1 denoising comparator | Not a main contribution | 15.8746 |
| Bilateral Filter | Classical edge-aware denoising comparator | Denoising comparator where already available | Baseline/comparator | 15.7772 |
| Ordinary Gabor | Conventional ridge-enhancement baseline | M2 comparator | Not the advanced contribution | 13.6777 |
| Wiener Deconvolution | Conventional restoration comparator | M3 restoration comparator | Comparator, not member headline | 16.1707 |
| Closing | Morphological structural support | Ridge map cleanup after grayscale enhancement | Downstream/supporting only | N/A |
| Thinning / Medial Axis | Downstream structural analysis | Skeleton/minutiae analysis after enhancement | Downstream/supporting only | N/A |
| Thresholding / Opening | Downstream binary/ridge-map support | Segmentation and cleanup | Downstream/supporting only | N/A |

## Literature Benchmark Status

Confirmed in local project documentation (`specification\fingerprint enhancement.pdf`) and preserved in `outputs/screening24_metadata.json`: Modified Gabor Filter benchmark PSNR = 28.17 dB, MSE = 22.27. Article URL recorded locally: https://fjpas.fuoye.edu.ng/index.php/fjpas/article/view/79; PDF URL recorded locally: https://fjpas.fuoye.edu.ng/index.php/fjpas/article/download/79/71. Access note from the repository audit: The article/PDF URLs returned a captcha/interstitial during this audit, so missing article details are not invented. Unknown from available repository/docs: exact dataset, image count, degradation construction, MAX_I, MSE scaling, and full metric formula. Our audit uses normalized [0,1] MSE and PSNR with MAX_I/data_range = 1.0; 8-bit-equivalent MSE is also recorded.

The literature PSNR/MSE pair is not internally standard-PSNR consistent without additional unknown settings. PSNR 28.17 dB implies MSE about 0.001524 on [0,1] or about 99.10 on [0,255]. MSE 22.27 with MAX_I=255 implies PSNR about 34.65 dB. Therefore, this project preserves the lecturer target as `PSNR > 28.17 dB`, but reports metric scaling transparently instead of inventing missing literature details.

## Existing Evidence Summary

Full file written: `outputs/member_existing_evidence.csv`

| Member / role | Technique ID / Pipeline ID | Technique / Pipeline | Source output file | Protocol | Images | MSE | PSNR (dB) | SSIM | Coherence | Local contrast | Fragmentation | Observation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Member 1 main technique | NLM_h0.06 | OpenCV Non-Local Means | focused_psnr_audit_test_summary.csv | Gaussian noise only, severity 0.55; fixed parameters selected on separate dev subset | 12.0000 | 0.0012 | 29.2504 | 0.9440 | N/A | N/A | N/A | Best confirmed controlled PSNR; all 12 held-out images exceed 28.17 dB. |
| Member 1 direct Altered sanity | NLM_h0.06 | OpenCV Non-Local Means | focused_psnr_audit_altered_structure.csv | Direct Altered evaluation; no synthetic degradation; structural metrics only | 9.0000 | N/A | N/A | N/A | 0.7180 | 0.1871 | 4.5370 | Mean deltas: coherence +0.0475, local contrast -0.0050, fragmentation -0.5182. |
| Member 2 baseline comparator | G1 | Ordinary Gabor filter | screening24_family_summary.csv | Current stacked synthetic degradation; aligned clean Real reference | 24.0000 | 0.0448 | 13.6777 | 0.4409 | 0.4969 | N/A | N/A | Ordinary Gabor is useful as the baseline; controlled PSNR is low and direct evidence shows contrast/fragmentation trade-offs. |
| Member 2 main investigation | G2 | Literature-inspired Modified Gabor | screening24_family_summary.csv | Current stacked synthetic degradation; aligned clean Real reference | 24.0000 | 0.0298 | 15.7256 | 0.5097 | 0.4761 | N/A | N/A | Literature-inspired Modified Gabor is implemented, but exact article settings are unknown and should not be claimed. |
| Member 2 supporting adaptive variant | G3 | Adaptive local-orientation Gabor | screening24_family_summary.csv | Current stacked synthetic degradation; aligned clean Real reference | 24.0000 | 0.0293 | 15.7711 | 0.5173 | 0.4812 | N/A | N/A | Adaptive local-orientation Gabor supports the modified/adaptive investigation. |
| Member 2 supporting adaptive variant | G4 | Adaptive orientation-frequency Gabor | screening24_family_summary.csv | Current stacked synthetic degradation; aligned clean Real reference | 24.0000 | 0.0299 | 15.6715 | 0.5095 | 0.4770 | N/A | N/A | Adaptive orientation-frequency Gabor supports frequency-aware ridge enhancement investigation. |
| Member 2 direct Altered trade-off | C1 | P0 -> CLAHE -> Ordinary Gabor | hybrid24_direct_summary.csv | Direct Altered evaluation; no synthetic degradation; no PSNR against unverified pair | 24.0000 | N/A | N/A | N/A | 0.7261 | 0.1644 | 3.9991 | P0 -> CLAHE -> Ordinary Gabor raised coherence but reduced contrast and increased fragmentation versus CLAHE. |
| Member 3 main technique | R3 | Total variation restoration | screening24_family_summary.csv | Current stacked synthetic degradation; aligned clean Real reference | 24.0000 | 0.0308 | 15.5634 | 0.5719 | 0.6212 | N/A | N/A | TV shows high SSIM/coherence balance in the stacked screening, although not high PSNR under stacked degradation. |
| Member 3 comparator | R1 | Wiener deconvolution | screening24_family_summary.csv | Current stacked synthetic degradation; aligned clean Real reference | 24.0000 | 0.0276 | 16.1707 | 0.5436 | 0.4982 | N/A | N/A | Wiener deconvolution is the conventional restoration comparator and was stronger than TV on stacked PSNR. |
| Member 3 direct compatibility | C3 | P0 -> CLAHE -> Total Variation Restoration | hybrid24_direct_summary.csv | Direct Altered evaluation; no synthetic degradation; no PSNR against unverified pair | 24.0000 | N/A | N/A | N/A | 0.7255 | 0.2037 | 2.9321 | CLAHE -> TV kept high coherence but reduced contrast slightly versus CLAHE. |
| Member 3 direct compatibility | C4 | P0 -> Total Variation Restoration -> CLAHE | hybrid24_direct_summary.csv | Direct Altered evaluation; no synthetic degradation; no PSNR against unverified pair | 24.0000 | N/A | N/A | N/A | 0.7475 | 0.2136 | 2.8435 | TV -> CLAHE produced the strongest direct coherence among hybrid grayscale candidates, with mild contrast trade-off versus CLAHE. |
| Member 3 controlled hybrid check | C4 | P0 -> Total Variation Restoration -> CLAHE | hybrid24_controlled_summary.csv | Current stacked synthetic degradation; aligned clean Real reference | 24.0000 | 0.0493 | 13.2241 | 0.6092 | 0.6129 | N/A | N/A | TV -> CLAHE did not solve the stacked-degradation PSNR problem; it remains below the literature target. |
| Member 4 main technique | G7 | Coherence-guided directional diffusion | screening24_family_summary.csv | Current stacked synthetic degradation; aligned clean Real reference | 24.0000 | 0.0297 | 15.7757 | 0.5044 | 0.4978 | N/A | N/A | Coherence-guided directional diffusion is a grayscale orientation-aware method and should replace thinning as M4 headline. |
| Member 4 direct compatibility | C2 | P0 -> CLAHE -> Coherence-Guided Directional Diffusion | hybrid24_direct_summary.csv | Direct Altered evaluation; no synthetic degradation; no PSNR against unverified pair | 24.0000 | N/A | N/A | N/A | 0.7062 | 0.2076 | 2.8949 | Directional diffusion after CLAHE was more compatible than Ordinary Gabor, with less contrast loss and much lower fragmentation penalty. |
| Downstream structural support | M3-2 | Closing | screening24_family_summary.csv | Current stacked synthetic degradation; aligned clean Real reference | 24.0000 | N/A | N/A | N/A | N/A | N/A | 4.6558 | Closing is a morphology support operation and has no full-reference grayscale PSNR. |
| Downstream structural support | T4 | Morphological skeleton / medial axis | screening24_family_summary.csv | Current stacked synthetic degradation; aligned clean Real reference | 24.0000 | N/A | N/A | N/A | N/A | N/A | 19.9913 | Skeleton/medial axis is downstream analysis; do not calculate PSNR on skeletons. |

## PSNR Target Audit

- Best current confirmed controlled PSNR: NLM_h0.06 on `gaussian_noise_only_s055`, 29.2504 dB.
- Gap to 28.17 dB: 1.0804 dB.
- Exceeded 28.17 dB: Yes.
- Protocol: Gaussian noise only, severity 0.55, fixed parameters selected on a separate development subset and evaluated on 12 held-out test images.
- Fixed parameters: {"backend": "cv2", "h": 0.06, "patch_distance": 6, "patch_size": 5}.
- Development/test split: random seed 20260902; 12 dev Real images and 12 held-out test Real images listed in `outputs/focused_psnr_audit_metadata.json`.
- MSE: 0.00119664 normalized [0,1]; 8-bit equivalent MSE 77.8115.
- SSIM: 0.9440.
- Per-image target result: 12 / 12 held-out test images exceeded 28.17 dB; minimum held-out PSNR 28.6759 dB.
- Direct Altered sanity for same NLM setting: Mean deltas: coherence +0.0475, local contrast -0.0050, fragmentation -0.5182.

The current stacked synthetic protocol remains much harder: best focused stacked result was Wiener deconvolution at 17.8842 dB, gap -10.2858 dB. This supports the audit conclusion that stacking blur, contrast/illumination degradation, Gaussian noise, salt-and-pepper noise, and smudge/occlusion creates a harsh compound task that is not directly comparable to the documented Modified Gabor literature benchmark unless the literature degradation protocol is confirmed.

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

M. Existing evidence supporting NLM: focused held-out test result PSNR 29.2504 dB, MSE 0.00119664, SSIM 0.9440, all 12 held-out images above 28.17 dB, plus direct Altered coherence improvement and fragmentation reduction.

N. Existing evidence supporting Modified Gabor investigation: G2/G3/G4 adaptive Gabor variants exist, Ordinary Gabor demonstrated the motivation by increasing coherence but showing contrast and fragmentation trade-offs. The current G2 result must be called Literature-Inspired Modified Gabor because exact article settings are unknown.

O. Existing evidence supporting TV Restoration: screening R3 achieved PSNR 15.5634 dB, MSE 0.030833, SSIM 0.5719, coherence 0.6212; direct hybrid C4 showed coherence 0.7475 on Altered images.

P. Existing evidence supporting Directional Diffusion: screening G7 achieved PSNR 15.7757 dB and coherence 0.4978; direct hybrid C2 showed coherence 0.7062, contrast 0.2076, fragmentation 2.8949.

Q. The strongest controlled PSNR is currently NLM_h0.06 on Gaussian-noise-only severity 0.55.

R. Yes. That result exceeds 28.17 dB by 1.0804 dB.

S. Structural trade-offs were clearest for Ordinary Gabor after CLAHE: direct Overall coherence 0.7261, contrast 0.1644, fragmentation 3.9991, contrast delta versus CLAHE -0.0589, fragmentation delta versus CLAHE 1.1859.

T. Ordinary Gabor is not recommended as the final overall technique. It remains the required comparator for Member 2.

U. Strengths and weaknesses: NLM has the strongest controlled PSNR and preserves ridges under Gaussian noise, but may slightly reduce local contrast on Altered images. Adaptive Gabor is fingerprint-specific and can improve ridge flow, but can alter intensities and fragment ridges if too strong. TV is edge/ridge-preserving and structurally compatible, but did not exceed the PSNR target in the stacked screening. Directional diffusion is orientation-aware and more compatible than Ordinary Gabor after CLAHE, but still needs a small fixed-parameter compatibility check with NLM.

V. Compatible-looking member techniques: NLM with a very mild adaptive ridge enhancement, and possibly TV with NLM if TV does not oversmooth before denoising. Directional diffusion may be compatible as a mild structure-preserving stage but must be checked against NLM-only.

W. Do not automatically combine all four members plus CLAHE, closing, and thinning. Also avoid Ordinary Gabor after CLAHE as a final grayscale method because it damages contrast and fragmentation despite coherence gains.

X. Scientifically justified small team candidates are:

| Candidate | Pipeline | Reason | Small next check |
| --- | --- | --- | --- |
| 1 | NLM only | Strongest confirmed controlled PSNR; avoids adding ridge-enhancement stages that may alter intensities. | Re-test fixed NLM h=0.06 on the same focused Gaussian-noise-only split and direct Altered structural sample. |
| 2 | NLM -> mild Literature-Inspired Modified/Adaptive Gabor | May add ridge-flow clarity after denoising, but must be very mild because Gabor can reduce contrast and increase fragmentation. | Use one fixed adaptive-Gabor blend chosen on dev only; compare PSNR loss and Altered structure versus NLM only. |
| 3 | TV -> NLM | TV and NLM both target restoration/denoising with different priors; useful only if TV does not oversmooth ridges before NLM. | Choose one TV weight from dev only and test fixed TV->NLM on held-out focused subset. |

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
