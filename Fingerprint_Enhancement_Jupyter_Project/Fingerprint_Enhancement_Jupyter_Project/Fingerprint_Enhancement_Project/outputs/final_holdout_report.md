# Final Fresh Hold-out Validation Report

## Locked design

The hold-out list was written before enhancement metrics were computed. No result-based replacement, removal, resampling, tuning, backend change, or severity change occurred. Available unused images: 5455.

## Exact requested questions

### A. How many Real images were available?

6000

### B. Which images were excluded because they had been used previously?

545 unique filenames; the complete auditable list is in `final_holdout_excluded_images.csv`.

### C. How many completely fresh images were selected?

100

### D. What deterministic selection method and seed were used?

Sorted unused Real paths followed by NumPy PCG64 `default_rng(20260903).permutation`; the first 100 indices were locked.

### E. Was the NLM parameter changed after compatibility testing?

No.

### F. What exact NLM implementation was used?

OpenCV `cv2.fastNlMeansDenoising` on rounded uint8 input, templateWindowSize=7, searchWindowSize=21, returned to [0,1] float.

### G. What exact h value was used?

0.06 in normalized units (15.3 passed to OpenCV).

### H. What Gaussian-noise degradation protocol was used?

`gaussian_noise_only_s055`: NumPy default_rng normal noise, severity 0.55, sigma 0.04825, per-image seed 20260903+index, then clip to [0,1].

### I. Was degradation severity changed after seeing results?

No.

### J. What was the mean degraded-input PSNR?

27.3876 dB.

### K. What was the mean enhanced PSNR?

29.6443 dB.

### L. What was the PSNR standard deviation?

0.6757 dB (sample SD).

### M. What was the median PSNR?

29.6160 dB.

### N. What was the minimum PSNR?

28.0519 dB.

### O. What was the maximum PSNR?

31.8516 dB.

### P. What percentage of images exceeded 28.17 dB?

99.00% (99/100).

### Q. Did the mean PSNR exceed 28.17 dB?

Yes.

### R. By how many dB?

+1.4743 dB.

### S. Did NLM improve PSNR relative to the degraded input?

Yes; mean delta +2.2568 dB.

### T. Did MSE decrease?

Yes; mean delta -0.0007300.

### U. Did SSIM increase?

Yes; mean delta +0.1035.

### V. What was the final mean MSE?

0.00109816 (normalized [0,1]).

### W. What was the final mean SSIM?

0.9383.

### X. Were there any images below 28.17 dB?

Yes (1).

### Y. If yes, what characteristics did those cases show?

194__M_Left_middle_finger.BMP: low original contrast=0.2082, foreground coverage=1.000

### Z. Did the fresh result agree with the earlier 12-image held-out result of 29.2504 dB?

It differed by 0.3939 dB; this is close supporting evidence.

### AA. Did the fresh result agree with the compatibility DEV result of 29.4366 dB?

It differed by 0.2077 dB; this is close supporting evidence.

### AB. Is there evidence of serious overfitting to the original 24-image experiments?

No serious evidence: the fresh set was locked and disjoint, and the target result generalized.

### AC. How did fixed NLM affect Altered Easy fingerprints?

Mean deltas: coherence +0.0459, contrast -0.0043, fragmentation -0.0714, continuity +1.5415. These indicators are descriptive, not automatically improvements.

### AD. How did it affect Altered Medium fingerprints?

Mean deltas: coherence +0.0565, contrast -0.0060, fragmentation -0.0989, continuity +2.8178. These indicators are descriptive, not automatically improvements.

### AE. How did it affect Altered Hard fingerprints?

Mean deltas: coherence +0.0627, contrast -0.0057, fragmentation -0.1181, continuity +2.7009. These indicators are descriptive, not automatically improvements.

### AF. Did coherence improve overall?

Yes; mean change +0.0550.

### AG. Did fragmentation improve overall?

Yes (decreased); mean change -0.0961.

### AH. Was there a contrast trade-off?

Yes, contrast decreased; mean change -0.0053.

### AI. Was serious ridge structural damage observed?

No automated claim is made from a single indicator. The six fixed-range before/after figures were visually checked; see the outlier/visual notes.

### AJ. Is NLM still recommended as the final grayscale enhancement technique?

Yes.

### AK. Why were the tested hybrids rejected?

TV→NLM, NLM→Modified Gabor, NLM→Directional Diffusion, and TV→NLM→Directional Diffusion were tested under the controlled compatibility experiment; none exceeded NLM alone (29.4366 dB).

### AL. Can the project now proceed to final large-dataset validation?

Yes.

## Visual inspection notes

The sole below-benchmark controlled case, `194__M_Left_middle_finger.BMP`, is a complete fingerprint with substantial ridge coverage. Its clean reference visibly contains granular/fragmented ridge edges, particularly in the upper region. NLM reduced the synthetic grain while mildly softening fine ridge-edge texture; the case was retained.

Across the six fixed-range Altered examples (two each from Easy, Medium, and Hard), scar, obliteration, and Z-cut structures and the surrounding ridge flow remained recognizable. Smoothing was mild, and no obvious newly invented ridge topology or serious structural damage was observed in these examples. This is a visual plausibility statement, not proof that every numerical structural change is beneficial.

## Benchmark wording

Under the controlled Gaussian-noise evaluation protocol adopted in this project, the fixed Non-Local Means method achieved a mean PSNR of 29.6443 dB, exceeding the selected literature PSNR benchmark of 28.17 dB by 1.4743 dB.

Exact metric equivalence with the literature experiment cannot be confirmed because the dataset, degradation protocol, scaling and evaluation details are not fully known or equivalent.

## Final positioning

Final grayscale pipeline: Input Fingerprint → P0 Neutral Preparation → Non-Local Means Denoising → Enhanced Grayscale Fingerprint.

Downstream segmentation, thresholding, morphology, thinning/medial axis, and feature analysis remain separate.

The four team techniques remain M1 NLM, M2 Modified/Orientation-Adaptive Gabor, M3 Total Variation Restoration, and M4 Coherence-Guided Directional Diffusion. NLM is positioned as the strongest optimized technique selected after comparative and integration experiments.
