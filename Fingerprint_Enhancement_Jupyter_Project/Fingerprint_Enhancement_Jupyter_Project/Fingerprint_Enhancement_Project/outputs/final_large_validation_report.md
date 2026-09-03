# Final Large-Dataset Robustness Validation

This is a **500-image large-dataset robustness evaluation**, not a fresh unseen hold-out. The separate fresh 100-image evidence was preserved.

## Requested questions

### A. What final method was frozen?

P0 neutral preparation → OpenCV Non-Local Means denoising.

### B. What exact NLM parameters were used?

`cv2.fastNlMeansDenoising`; normalized h=0.06 (OpenCV uint8 h=15.3), template window 7, search window 21; rounded [0,1] float to uint8 and converted output back to [0,1]. P0 uses grayscale, 256×256 resize, and 1st/99th-percentile normalization.

### C. Were parameters changed after the fresh hold-out?

No.

### D. What controlled degradation protocol was used?

`gaussian_noise_only_s055`, severity 0.55, Gaussian sigma 0.04825, deterministic per-image NumPy RNG, clipping to [0,1].

### E. Was degradation severity changed?

No.

### F. How many Real images were used?

500

### G. Is this 500-image run a fresh unseen hold-out?

No. It is a 500-image large-dataset robustness evaluation using the canonical prior list.

### H. What was mean degraded PSNR?

27.3851 dB.

### I. What was mean enhanced PSNR?

29.6303 dB.

### J. What was PSNR standard deviation?

0.6246 dB (sample SD).

### K. What was median enhanced PSNR?

29.5711 dB.

### L. What was minimum enhanced PSNR?

28.4178 dB.

### M. What was maximum enhanced PSNR?

31.7460 dB.

### N. What percentage exceeded 28.17 dB?

100.00% (500/500).

### O. How many images were below 28.17 dB?

0. All are listed in `final_large_real_outliers.csv`.

### P. What was the mean margin above 28.17 dB?

+1.4603 dB.

### Q. Did enhanced PSNR exceed degraded PSNR?

Yes; mean delta +2.2452 dB.

### R. What was mean enhanced MSE?

0.00109989 in normalized [0,1].

### S. Did MSE decrease?

Yes; mean delta -0.00072917.

### T. What was mean enhanced SSIM?

0.9357.

### U. Did SSIM improve?

Yes; mean delta +0.1018.

### V. How does the 500-image result compare with the fresh 100-image result of 29.6443 dB?

Large minus fresh mean = -0.0140 dB; SD difference = -0.0511 dB; threshold-rate difference = +1.00 percentage points. The datasets remain separate.

### W. Is the difference small enough to support robustness?

Yes under the prespecified descriptive ±0.5 dB closeness interpretation.

### X. What were the worst controlled cases?

100__M_Left_thumb_finger.BMP (28.418 dB), 115__F_Right_thumb_finger.BMP (28.428 dB), 124__M_Right_thumb_finger.BMP (28.465 dB), 115__F_Left_thumb_finger.BMP (28.473 dB), 13__F_Right_thumb_finger.BMP (28.497 dB), 104__M_Left_thumb_finger.BMP (28.511 dB), 100__M_Right_thumb_finger.BMP (28.520 dB), 115__F_Right_middle_finger.BMP (28.531 dB), 110__F_Right_index_finger.BMP (28.557 dB), 100__M_Right_middle_finger.BMP (28.572 dB)

### Y. Were any cases excluded after seeing results?

No.

### Z. How many Altered Easy images were processed?

500

### AA. How many Altered Medium images were processed?

500

### AB. How many Altered Hard images were processed?

500

### AC. What was the coherence change for Easy?

+0.0544.

### AD. What was the coherence change for Medium?

+0.0571.

### AE. What was the coherence change for Hard?

+0.0602.

### AF. What was the fragmentation change for Easy?

-0.0984.

### AG. What was the fragmentation change for Medium?

-0.1220.

### AH. What was the fragmentation change for Hard?

-0.1628.

### AI. What was the contrast trade-off?

Easy -0.0052; Medium -0.0057; Hard -0.0061. Negative values indicate contrast loss.

### AJ. Was serious ridge structural damage observed?

No serious damage was evident in the exported representative and weakest-coherence examples. Alteration regions and surrounding ridge paths remained recognizable, with no obvious newly created ridge paths or ridge merging. Mild smoothing/contrast loss was visible; the weakest Easy example also had a small measured fragmentation increase (+0.0393). This conclusion is limited to the inspected figures.

### AK. Did NLM remain the final recommended grayscale enhancement method?

Yes.

### AL. Why were hybrid approaches not retained?

Prior compatibility results were: NLM 29.4366 dB; TV→NLM 28.9436; NLM→Directional Diffusion 28.2968; NLM→Modified Gabor 25.2277; TV→NLM→Directional Diffusion 26.6099. No tested hybrid exceeded NLM alone.

### AM. Did the large-scale result still exceed 28.17 dB?

Yes.

### AN. Is the experimental phase now complete?

Yes.

### AO. What should be done next?

Consolidate the final code/notebook, integrate the prototype/dashboard, write the final report, finalize figures/tables, and prepare the presentation.

## Scientific positioning

Under the controlled Gaussian-noise evaluation protocol used in this project, the fixed Non-Local Means method achieved a mean PSNR of 29.6303 dB across 500 Real fingerprint images, exceeding the selected literature PSNR benchmark of 28.17 dB by 1.4603 dB.

Exact numerical equivalence with the literature experiment cannot be confirmed because the datasets, degradation protocol, scaling and metric-reporting conventions are not fully equivalent.

Previous member comparison: M1 NLM 29.4366 dB, M3 TV 29.1850 dB, M4 Directional Diffusion 27.8028 dB, and M2 Modified Gabor 25.0002 dB. This run validates the selected winner and does not restart the competition.

Final grayscale pipeline: Input → P0 neutral preparation → NLM → enhanced grayscale fingerprint. Segmentation, morphology, thinning/medial axis, and feature analysis remain downstream.
