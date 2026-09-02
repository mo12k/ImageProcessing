# Focused PSNR Benchmark Audit and Optimisation Experiment

Generated: 2026-09-02T18:52:34.068006+08:00

This run did not execute the 500-image validation. It used a fixed 12-image development split for parameter selection and a disjoint fixed 12-image test split for reporting.

## A. Best current PSNR

- Best actual restoration/enhancement test PSNR: Contrast stretching on `smudge_only_s055` = 37.9248 +/- 1.7762 dB.
- Best overall test PSNR including the no-enhancement control: Contrast stretching on `smudge_only_s055` = 37.9248 +/- 1.7762 dB.
- Best current stacked-protocol PSNR: Wiener deconvolution = 17.8842 +/- 0.4368 dB.

## B. Gap to 28.17 dB

- Best actual restoration/enhancement signed gap: +9.7548 dB.
- Current stacked-protocol signed gap: -10.2858 dB.

## C. Whether 28.17 dB was exceeded

- Exceeded by best actual restoration/enhancement candidate: True.
- Exceeded under the current stacked degradation protocol: False.

## D. Degradation protocol producing the best result

- `smudge_only_s055`: Smudge only, severity 0.55 component.
- Degradation parameters: {"smudge_centre": [127, 146], "smudge_radii": [4, 14], "smudge_rotation": 0.35313280294156413, "smudge_sigma": 4.1}.

## E. Scientific justification of that protocol

- It is scientifically justified as an isolated controlled degradation component because it is derived from the existing project degradation formula and tests the recoverability of one named physical/statistical corruption.
- It is not confirmed to be directly comparable with the selected literature article because the article's dataset, degradation process, reference definition, ROI/full-image rule, preprocessing, and PSNR/MSE scaling are unknown in the local repository evidence.
- Therefore, exceeding 28.17 dB under this isolated protocol should be reported as controlled component evidence, not as a verified reproduction of the literature evaluation.

## F. Algorithm/pipeline producing the result

- Candidate: Contrast stretching (`CONTRAST_1_99`).
- Family: Contrast Enhancement.

## G. Exact fixed parameters

- {"high_percentile": 99.0, "low_percentile": 1.0}

## H. Development/test split

- Selection seed: 20260902.
- Development images: 12.
- Test images: 12.
- Development files:
  - 34__M_Right_ring_finger.BMP
  - 12__M_Left_middle_finger.BMP
  - 379__F_Left_ring_finger.BMP
  - 216__M_Right_ring_finger.BMP
  - 598__M_Right_little_finger.BMP
  - 241__M_Right_thumb_finger.BMP
  - 318__F_Left_little_finger.BMP
  - 533__M_Left_thumb_finger.BMP
  - 352__M_Left_ring_finger.BMP
  - 399__M_Right_ring_finger.BMP
  - 576__M_Left_middle_finger.BMP
  - 506__M_Right_ring_finger.BMP
- Test files:
  - 559__M_Right_index_finger.BMP
  - 52__M_Right_little_finger.BMP
  - 384__F_Left_middle_finger.BMP
  - 410__M_Right_ring_finger.BMP
  - 337__F_Right_index_finger.BMP
  - 74__F_Right_ring_finger.BMP
  - 296__M_Left_index_finger.BMP
  - 103__F_Left_middle_finger.BMP
  - 355__M_Right_middle_finger.BMP
  - 86__M_Right_index_finger.BMP
  - 576__M_Left_little_finger.BMP
  - 537__M_Right_thumb_finger.BMP

## I. MSE

- Best actual candidate normalised MSE: 0.00017252 +/- 0.00005900.
- Best actual candidate 8-bit-equivalent MSE: 11.2179 +/- 3.8366.

## J. SSIM

- Best actual candidate SSIM: 0.9980 +/- 0.0003.

## K. Same pipeline on original Altered images

The selected pipeline is structurally cautious on the sampled Altered images if judged by no-reference indicators: mean coherence delta +0.0000, local-contrast delta +0.0000, fragmentation delta +0.0000. These are not full-reference accuracy metrics.
- Altered montage: `outputs\focused_psnr_audit_altered_montage.png`.

## Metric protocol and dynamic range audit

- Our metric domain: [0,1] float images clipped only before MSE/PSNR/SSIM.
- Our PSNR `MAX_I`: 1.0.
- Our MSE scale: normalised MSE; multiply by 65025 for 8-bit equivalent.
- A standard PSNR of 28.17 dB implies MSE 0.00152405 on [0,1], or 99.1015 on [0,255].
- The literature's recorded MSE 22.27 would imply PSNR 34.6536 dB if MAX_I=255, or -13.4772 dB if MAX_I=1.
- Audit conclusion: The repo-confirmed literature PSNR/MSE pair is not internally consistent under standard PSNR without an unknown scaling/reporting convention.

## Literature settings: confirmed vs unknown from repository

Confirmed from local repository/documentation:
- Modified Gabor Filter is the selected literature benchmark method.
- The project documentation records PSNR = 28.17 dB and MSE = 22.27.
- Existing repo reports state that exact dataset, image count, degradation process, preprocessing, parameters, reference definition, and PSNR formula are not confirmed from locally extractable information.

Unknown from local repository/documentation:
- Dataset used by the article experiment.
- Number and identity of evaluated images.
- Whether images were clean/degraded, naturally noisy, or synthetically degraded.
- Whether PSNR used full image, ROI, or a mask.
- Whether image intensities were [0,1], [0,255], or separately normalised before metrics.
- The exact Modified Gabor parameters and preprocessing chain.

## Why the current controlled PSNR is low

- The current synthetic pipeline stacks several degradations. The audit stage losses below show that blur and contrast/illumination dominate the drop, while noise, impulse pixels, and smudge add smaller but still irreversible changes.

| Stage | Delta_PSNR_mean | PSNR_after_mean |
| --- | --- | --- |
| low_contrast_illumination | -5.2779 | 15.9403 |
| gaussian_noise | -0.3793 | 15.561 |
| salt_pepper_noise | -0.0875 | 15.4735 |
| smudge_occlusion | -0.0114 | 15.4621 |

Input difficulty by controlled scenario on the held-out test split:

| Scenario ID | Scenario | MSE_mean | MSE_255_equivalent_mean | PSNR_mean | SSIM_mean |
| --- | --- | --- | --- | --- | --- |
| smudge_only_s055 | Smudge only, severity 0.55 component | 0.0002 | 11.2179 | 37.9248 | 0.998 |
| salt_pepper_only_s055 | Salt-and-pepper only, severity 0.55 component | 0.0006 | 41.4938 | 32.0139 | 0.9857 |
| gaussian_noise_only_s035 | Gaussian noise only, severity 0.35 component | 0.0012 | 75.2109 | 29.3706 | 0.9014 |
| gaussian_noise_only_s055 | Gaussian noise only, severity 0.55 component | 0.0019 | 124.728 | 27.174 | 0.8664 |
| blur_only_s055 | Blur only, severity 0.55 component | 0.0083 | 538.943 | 21.012 | 0.8445 |
| blur_plus_gaussian_noise_s055 | Blur + Gaussian noise, severity 0.55 components | 0.0103 | 671.9266 | 19.9704 | 0.7187 |
| low_contrast_illumination_s055 | Low contrast + illumination only, severity 0.55 component | 0.0144 | 936.3975 | 18.4483 | 0.8898 |
| current_stack_s055 | Current stacked degradation, severity 0.55 | 0.0287 | 1866.3863 | 15.4606 | 0.5308 |

## Selected fixed candidates from development search

| Scenario ID | Candidate ID | Family | Candidate | Candidate parameters | PSNR_mean | SSIM_mean |
| --- | --- | --- | --- | --- | --- | --- |
| blur_only_s055 | WIENER_DECONV_s1.34_b0.02_m1 | Restoration/Deblurring | Wiener deconvolution | {"balance": 0.02, "blend": 1.0, "psf_sigma": 1.34} | 26.9976 | 0.9454 |
| blur_plus_gaussian_noise_s055 | WIENER_DECONV_s1.34_b0.02_m1 | Restoration/Deblurring | Wiener deconvolution | {"balance": 0.02, "blend": 1.0, "psf_sigma": 1.34} | 24.6842 | 0.8527 |
| current_stack_s055 | WIENER_DECONV_s1.7_b0.02_m1 | Restoration/Deblurring | Wiener deconvolution | {"balance": 0.02, "blend": 1.0, "psf_sigma": 1.7} | 17.5808 | 0.7135 |
| gaussian_noise_only_s035 | NLM_h0.04 | Denoising | Non-local means | {"backend": "skimage", "h": 0.04, "patch_distance": 6, "patch_size": 5} | 31.6611 | 0.9604 |
| gaussian_noise_only_s055 | NLM_h0.06 | Denoising | Non-local means | {"backend": "skimage", "h": 0.06, "patch_distance": 6, "patch_size": 5} | 30.0625 | 0.9434 |
| low_contrast_illumination_s055 | CLAHE_c0.005 | Contrast Enhancement | CLAHE | {"clip_limit": 0.005, "kernel_size": 32} | 26.7668 | 0.9671 |
| salt_pepper_only_s055 | GAUSS_s0.35 | Denoising | Gaussian filtering | {"sigma": 0.35} | 32.5669 | 0.9849 |
| smudge_only_s055 | CONTRAST_1_99 | Contrast Enhancement | Contrast stretching | {"high_percentile": 99.0, "low_percentile": 1.0} | 40.6544 | 0.9981 |

## Held-out test summary for selected candidates

| Scenario ID | Candidate ID | Family | Candidate | MSE_mean | MSE_255_equivalent_mean | PSNR_mean | PSNR_std | SSIM_mean | SSIM_std | PSNR_delta_to_28_17 | Target_exceeded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur_only_s055 | WIENER_DECONV_s1.34_b0.02_m1 | Restoration/Deblurring | Wiener deconvolution | 0.0022 | 142.8169 | 26.7228 | 1.1976 | 0.9455 | 0.0069 | -1.4472 | False |
| blur_only_s055 | NLM040_THEN_WIENER_DECONV | Combination | Non-local means -> Wiener deconvolution | 0.0062 | 400.5323 | 22.2784 | 1.3631 | 0.8574 | 0.0327 | -5.8916 | False |
| blur_only_s055 | CLAHE_c0.005 | Contrast Enhancement | CLAHE | 0.0071 | 461.4084 | 21.6296 | 1.2392 | 0.8808 | 0.0251 | -6.5404 | False |
| blur_only_s055 | INPUT | Control | Degraded input | 0.0083 | 538.943 | 21.012 | 1.5178 | 0.8445 | 0.0338 | -7.158 | False |
| blur_only_s055 | GAUSS_s0.35 | Denoising | Gaussian filtering | 0.0085 | 550.0199 | 20.9233 | 1.5168 | 0.841 | 0.0345 | -7.2467 | False |
| blur_only_s055 | TV_w0.01 | Restoration/Denoising | Total variation restoration | 0.0092 | 597.6162 | 20.5567 | 1.4833 | 0.8209 | 0.0389 | -7.6133 | False |
| blur_only_s055 | HOMOMORPHIC_s12_g0.95 | Illumination/Contrast | Homomorphic filtering | 0.0093 | 602.8647 | 20.5116 | 1.4711 | 0.8196 | 0.035 | -7.6584 | False |
| blur_plus_gaussian_noise_s055 | WIENER_DECONV_s1.34_b0.02_m1 | Restoration/Deblurring | Wiener deconvolution | 0.0036 | 234.9291 | 24.4722 | 0.7038 | 0.8646 | 0.0252 | -3.6978 | False |
| blur_plus_gaussian_noise_s055 | NLM040_THEN_WIENER_DECONV | Combination | Non-local means -> Wiener deconvolution | 0.0062 | 406.1067 | 22.196 | 1.2755 | 0.8517 | 0.0284 | -5.974 | False |
| blur_plus_gaussian_noise_s055 | NLM_h0.04 | Denoising | Non-local means | 0.0097 | 631.9458 | 20.2827 | 1.334 | 0.7897 | 0.0396 | -7.8873 | False |
| blur_plus_gaussian_noise_s055 | CLAHE_c0.005 | Contrast Enhancement | CLAHE | 0.0099 | 641.1888 | 20.1384 | 0.8997 | 0.7448 | 0.0401 | -8.0316 | False |
| blur_plus_gaussian_noise_s055 | TV_w0.01 | Restoration/Denoising | Total variation restoration | 0.0103 | 667.8227 | 20.029 | 1.2762 | 0.7597 | 0.0361 | -8.141 | False |
| blur_plus_gaussian_noise_s055 | INPUT | Control | Degraded input | 0.0103 | 671.9266 | 19.9704 | 1.1082 | 0.7187 | 0.04 | -8.1996 | False |
| blur_plus_gaussian_noise_s055 | HOMOMORPHIC_s12_g0.95 | Illumination/Contrast | Homomorphic filtering | 0.0112 | 727.4025 | 19.6264 | 1.1206 | 0.7088 | 0.0386 | -8.5436 | False |
| current_stack_s055 | WIENER_DECONV_s1.7_b0.02_m1 | Restoration/Deblurring | Wiener deconvolution | 0.0164 | 1063.2158 | 17.8842 | 0.4368 | 0.7482 | 0.0413 | -10.2858 | False |
| current_stack_s055 | CONTRAST_5_95 | Contrast Enhancement | Contrast stretching | 0.0185 | 1202.3513 | 17.3701 | 0.6235 | 0.6249 | 0.0582 | -10.7999 | False |
| current_stack_s055 | TV020_THEN_CLAHE005 | Combination | Total variation restoration -> CLAHE | 0.0212 | 1380.4145 | 16.7844 | 0.7578 | 0.6489 | 0.0426 | -11.3856 | False |
| current_stack_s055 | GAUSS_s0.55 | Denoising | Gaussian filtering | 0.0283 | 1837.1677 | 15.54 | 0.74 | 0.572 | 0.0363 | -12.63 | False |
| current_stack_s055 | TV_w0.01 | Restoration/Denoising | Total variation restoration | 0.0285 | 1853.4317 | 15.4972 | 0.704 | 0.5573 | 0.037 | -12.6728 | False |
| current_stack_s055 | INPUT | Control | Degraded input | 0.0287 | 1866.3863 | 15.4606 | 0.6496 | 0.5308 | 0.0408 | -12.7094 | False |
| current_stack_s055 | HOMOMORPHIC_s12_g0.95 | Illumination/Contrast | Homomorphic filtering | 0.0299 | 1943.2258 | 15.2881 | 0.6729 | 0.5204 | 0.0385 | -12.8819 | False |
| gaussian_noise_only_s035 | NLM_h0.04 | Denoising | Non-local means | 0.0007 | 46.8514 | 31.4539 | 0.5579 | 0.9627 | 0.0063 | 3.2839 | True |
| gaussian_noise_only_s035 | TV_w0.02 | Restoration/Denoising | Total variation restoration | 0.0009 | 55.5295 | 30.7163 | 0.5577 | 0.9597 | 0.0055 | 2.5463 | True |
| gaussian_noise_only_s035 | WIENER_DECONV_s0.98_b0.06_m0.35 | Restoration/Deblurring | Wiener deconvolution | 0.001 | 66.737 | 29.9023 | 0.3822 | 0.9239 | 0.016 | 1.7323 | True |
| gaussian_noise_only_s035 | WAVELET025_THEN_WIENER_DECONV | Combination | Wavelet denoising -> Wiener deconvolution | 0.0011 | 72.8887 | 29.5961 | 0.985 | 0.9537 | 0.0037 | 1.4261 | True |
| gaussian_noise_only_s035 | CONTRAST_1_99 | Contrast Enhancement | Contrast stretching | 0.0012 | 75.2109 | 29.3706 | 0.1596 | 0.9014 | 0.0236 | 1.2006 | True |
| gaussian_noise_only_s035 | INPUT | Control | Degraded input | 0.0012 | 75.2109 | 29.3706 | 0.1596 | 0.9014 | 0.0236 | 1.2006 | True |
| gaussian_noise_only_s035 | HOMOMORPHIC_s12_g0.95 | Illumination/Contrast | Homomorphic filtering | 0.0013 | 83.0706 | 28.939 | 0.1614 | 0.9006 | 0.0221 | 0.769 | True |
| gaussian_noise_only_s055 | NLM_h0.06 | Denoising | Non-local means | 0.001 | 65.8155 | 29.9586 | 0.3281 | 0.9469 | 0.0132 | 1.7886 | True |
| gaussian_noise_only_s055 | TV_w0.02 | Restoration/Denoising | Total variation restoration | 0.0013 | 82.4612 | 28.9926 | 0.4936 | 0.9427 | 0.0099 | 0.8226 | True |
| gaussian_noise_only_s055 | WAVELET025_THEN_WIENER_DECONV | Combination | Wavelet denoising -> Wiener deconvolution | 0.0014 | 88.8268 | 28.7151 | 0.8515 | 0.9463 | 0.0058 | 0.5451 | True |
| gaussian_noise_only_s055 | WIENER_DECONV_s0.98_b0.06_m0.35 | Restoration/Deblurring | Wiener deconvolution | 0.0015 | 96.6883 | 28.2852 | 0.2807 | 0.8973 | 0.0232 | 0.1152 | True |
| gaussian_noise_only_s055 | CONTRAST_1_99 | Contrast Enhancement | Contrast stretching | 0.0019 | 124.728 | 27.174 | 0.1656 | 0.8664 | 0.0326 | -0.996 | False |
| gaussian_noise_only_s055 | INPUT | Control | Degraded input | 0.0019 | 124.728 | 27.174 | 0.1656 | 0.8664 | 0.0326 | -0.996 | False |
| gaussian_noise_only_s055 | HOMOMORPHIC_s12_g0.95 | Illumination/Contrast | Homomorphic filtering | 0.002 | 129.8577 | 26.9981 | 0.1386 | 0.8679 | 0.0309 | -1.1719 | False |
| low_contrast_illumination_s055 | CLAHE_c0.005 | Contrast Enhancement | CLAHE | 0.0025 | 164.9612 | 26.1384 | 1.4291 | 0.9703 | 0.0031 | -2.0316 | False |
| low_contrast_illumination_s055 | TV020_THEN_CLAHE005 | Combination | Total variation restoration -> CLAHE | 0.0028 | 178.8414 | 25.7348 | 1.1655 | 0.9596 | 0.006 | -2.4352 | False |
| low_contrast_illumination_s055 | WIENER_DECONV_s0.98_b0.02_m1 | Restoration/Deblurring | Wiener deconvolution | 0.0113 | 737.1682 | 19.4961 | 0.6299 | 0.9213 | 0.0077 | -8.6739 | False |
| low_contrast_illumination_s055 | INPUT | Control | Degraded input | 0.0144 | 936.3975 | 18.4483 | 0.5565 | 0.8898 | 0.0081 | -9.7217 | False |
| low_contrast_illumination_s055 | NLM_h0.025 | Denoising | Non-local means | 0.0145 | 942.9934 | 18.4162 | 0.5419 | 0.8767 | 0.0088 | -9.7538 | False |
| low_contrast_illumination_s055 | TV_w0.01 | Restoration/Denoising | Total variation restoration | 0.0152 | 989.6217 | 18.2065 | 0.5432 | 0.8723 | 0.0068 | -9.9635 | False |
| low_contrast_illumination_s055 | HOMOMORPHIC_s12_g0.95 | Illumination/Contrast | Homomorphic filtering | 0.0158 | 1029.2501 | 18.0395 | 0.5742 | 0.8752 | 0.0082 | -10.1305 | False |
| salt_pepper_only_s055 | GAUSS_s0.35 | Denoising | Gaussian filtering | 0.0006 | 36.9498 | 32.5159 | 0.7651 | 0.9862 | 0.0052 | 4.3459 | True |
| salt_pepper_only_s055 | CONTRAST_1_99 | Contrast Enhancement | Contrast stretching | 0.0006 | 41.4938 | 32.0139 | 0.7748 | 0.9857 | 0.0053 | 3.8439 | True |
| salt_pepper_only_s055 | INPUT | Control | Degraded input | 0.0006 | 41.4938 | 32.0139 | 0.7748 | 0.9857 | 0.0053 | 3.8439 | True |
| salt_pepper_only_s055 | TV_w0.01 | Restoration/Denoising | Total variation restoration | 0.0007 | 42.4939 | 31.8989 | 0.7013 | 0.9839 | 0.0057 | 3.7289 | True |
| salt_pepper_only_s055 | WIENER_DECONV_s0.98_b0.06_m0.35 | Restoration/Deblurring | Wiener deconvolution | 0.0007 | 45.773 | 31.5662 | 0.6331 | 0.98 | 0.005 | 3.3962 | True |
| salt_pepper_only_s055 | HOMOMORPHIC_s12_g0.95 | Illumination/Contrast | Homomorphic filtering | 0.0007 | 47.9113 | 31.3597 | 0.561 | 0.9723 | 0.0049 | 3.1897 | True |
| salt_pepper_only_s055 | WAVELET025_THEN_WIENER_DECONV | Combination | Wavelet denoising -> Wiener deconvolution | 0.0011 | 69.6575 | 29.7651 | 0.7994 | 0.9581 | 0.0073 | 1.5951 | True |
| smudge_only_s055 | CONTRAST_1_99 | Contrast Enhancement | Contrast stretching | 0.0002 | 11.2179 | 37.9248 | 1.7762 | 0.998 | 0.0003 | 9.7548 | True |
| smudge_only_s055 | INPUT | Control | Degraded input | 0.0002 | 11.2179 | 37.9248 | 1.7762 | 0.998 | 0.0003 | 9.7548 | True |
| smudge_only_s055 | GAUSS_s0.35 | Denoising | Gaussian filtering | 0.0002 | 11.5607 | 37.7755 | 1.7237 | 0.9979 | 0.0003 | 9.6055 | True |
| smudge_only_s055 | TV_w0.01 | Restoration/Denoising | Total variation restoration | 0.0002 | 15.6547 | 36.3233 | 1.2062 | 0.9958 | 0.0008 | 8.1533 | True |
| smudge_only_s055 | HOMOMORPHIC_s12_g0.95 | Illumination/Contrast | Homomorphic filtering | 0.0003 | 20.9749 | 35.0905 | 1.3855 | 0.9839 | 0.0027 | 6.9205 | True |
| smudge_only_s055 | WIENER_DECONV_s0.98_b0.06_m0.35 | Restoration/Deblurring | Wiener deconvolution | 0.0005 | 31.5968 | 33.2611 | 1.2049 | 0.9903 | 0.0009 | 5.0911 | True |
| smudge_only_s055 | WAVELET025_THEN_WIENER_DECONV | Combination | Wavelet denoising -> Wiener deconvolution | 0.0009 | 59.879 | 30.4765 | 1.151 | 0.9674 | 0.0036 | 2.3065 | True |

## Per-image result location

- Full held-out per-image metrics: `outputs/focused_psnr_audit_test_per_image.csv`.
- Full development grid per-image metrics: `outputs/focused_psnr_audit_dev_per_image.csv`.

## Final interpretation

- A scientifically valid, non-leaking classical pipeline exceeded 28.17 dB in this audit: True.
- A tested method exceeded 28.17 dB under the current stacked degradation protocol: False.
- Because the selected literature article's evaluation protocol is not fully available in the repository and its PSNR/MSE pair is not standard-formula-consistent, the safest claim is conditional: the target is exceeded for the named controlled component scenario, but direct article outperformance remains unverified unless the lecturer accepts that protocol as comparable.

Runtime: 129.68 seconds.

## Local extracted literature context near 28.17

rint Images Using Gabor Filter . https://iopscience.iop.or g/article/10.1088/1742-6596/1 196/1/012045/pdf S t u d y 2 – P e r f o r m a n c e E v a l u a t i o n o f F i n g e r p r i n t E n h a n c e m e n t A l g o r i t h m s ● Compared Modified Gabor Filter , High Boost Gaussian Filter , and Fuzzy-Based Filter . ● Evaluated the techniques based on noise reduction and fingerprint ridge clarity . ● The Modified Gabor Filter achieved a PSNR of 28.17 dB and MSE of 22.27 . ● The results show that the selection of enhancement filters af fects fingerprint image quality . ● Suitable as a reference for selecting an appropriate enhancement technique. ● Limitation: enhancement filtering alone does not perform segmentation, thinning, or fingerprint feature extraction. Article: Mukaila et al. (2019), Performance Evaluation of Fingerprint Image Enhancement Algorithms . https://fjpas.fuoye.edu.ng/index.php/fjpas/article/view/79/71 P l a n n i n g Member Pr epr ocessing Image Pr ocessing T echnique Member 1: Image Pr epr ocessing Module Convert images to grayscale, resize them, normalise intens
