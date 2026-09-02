# Member Compatibility Report

This run used only the existing focused audit development split and the small Altered sample from prior direct sanity checks. It did not run 500 images, did not use the focused held-out test split, did not use a fresh final hold-out set, and did not retune NLM.

## Fixed Parameters

The full fixed parameter record is in `outputs/member_fixed_parameters.json`.

## Four-Member Controlled Summary

| Method ID | Member | Technique | Images | MSE_mean | PSNR_mean | SSIM_mean | Delta_PSNR_vs_degraded_mean | Target_exceeded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1_NLM_h0.06 | Member 1 | Non-Local Means Denoising | 12.0000 | 0.0011 | 29.4366 | 0.9408 | 2.1204 | Yes |
| M3_TV_w0.02 | Member 3 | Total Variation Restoration | 12.0000 | 0.0012 | 29.1850 | 0.9402 | 1.8688 | Yes |
| BL_WIENER | Baseline | Wiener Deconvolution | 12.0000 | 0.0014 | 28.4338 | 0.8852 | 1.1176 | Yes |
| M4_COH_DIFF | Member 4 | Coherence-Guided Directional Diffusion | 12.0000 | 0.0017 | 27.8028 | 0.8684 | 0.4865 | No |
| B0 | Baseline | Degraded input / P0 | 12.0000 | 0.0019 | 27.3162 | 0.8502 | 0.0000 | No |
| BL_CONTRAST_1_99 | Baseline | Contrast Stretching | 12.0000 | 0.0019 | 27.3162 | 0.8502 | 0.0000 | No |
| M2_MOD_GABOR | Member 2 | Literature-Inspired Modified / Orientation-Adaptive Gabor | 12.0000 | 0.0032 | 25.0002 | 0.8373 | -2.3160 | No |
| BL_CLAHE_c0.005 | Baseline | CLAHE | 12.0000 | 0.0035 | 24.5462 | 0.8346 | -2.7700 | No |
| BL_ORD_GABOR | Baseline | Ordinary Gabor | 12.0000 | 0.0162 | 18.0003 | 0.7783 | -9.3159 | No |

## Four-Member Direct Altered Summary

| Method ID | Member | Technique | Images | Coherence_mean | Local_contrast_mean | Fragmentation_mean | Ridge_continuity_mean | Delta_coherence_vs_P0_mean | Delta_contrast_vs_P0_mean | Delta_fragmentation_vs_P0_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | Baseline | Degraded input / P0 | 9.0000 | 0.6705 | 0.1921 | 2.6367 | 26.6380 | 0.0000 | 0.0000 | 0.0000 |
| BL_CLAHE_c0.005 | Baseline | CLAHE | 9.0000 | 0.6848 | 0.2035 | 2.5362 | 27.7802 | 0.0142 | 0.0115 | -0.1005 |
| BL_CONTRAST_1_99 | Baseline | Contrast Stretching | 9.0000 | 0.6705 | 0.1921 | 2.6367 | 26.6380 | 0.0000 | 0.0000 | 0.0000 |
| BL_ORD_GABOR | Baseline | Ordinary Gabor | 9.0000 | 0.7353 | 0.1426 | 3.9944 | 22.7437 | 0.0648 | -0.0494 | 1.3578 |
| BL_WIENER | Baseline | Wiener Deconvolution | 9.0000 | 0.6935 | 0.2017 | 2.5533 | 28.4756 | 0.0230 | 0.0096 | -0.0833 |
| M1_NLM_h0.06 | Member 1 | Non-Local Means Denoising | 9.0000 | 0.7180 | 0.1871 | 2.5610 | 27.8600 | 0.0475 | -0.0050 | -0.0757 |
| M2_MOD_GABOR | Member 2 | Literature-Inspired Modified / Orientation-Adaptive Gabor | 9.0000 | 0.6863 | 0.1768 | 2.9320 | 25.3496 | 0.0158 | -0.0153 | 0.2954 |
| M3_TV_w0.02 | Member 3 | Total Variation Restoration | 9.0000 | 0.6927 | 0.1832 | 2.7552 | 26.0341 | 0.0221 | -0.0089 | 0.1185 |
| M4_COH_DIFF | Member 4 | Coherence-Guided Directional Diffusion | 9.0000 | 0.6780 | 0.1774 | 2.7440 | 25.4700 | 0.0075 | -0.0147 | 0.1074 |

## Compatibility Controlled Summary

| Candidate ID | Pipeline | Images | MSE_mean | PSNR_mean | SSIM_mean | Delta_PSNR_vs_degraded_mean | Delta_PSNR_vs_NLM_mean | Target_exceeded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0_NLM | NLM | 12.0000 | 0.0011 | 29.4366 | 0.9408 | 2.1204 | 0.0000 | Yes |
| C1_TV_NLM | TV -> NLM | 12.0000 | 0.0013 | 28.9436 | 0.9286 | 1.6274 | -0.4930 | Yes |
| C3_NLM_COH_DIFF | NLM -> Coherence-Guided Directional Diffusion | 12.0000 | 0.0015 | 28.2968 | 0.9320 | 0.9806 | -1.1398 | Yes |
| C4_TV_NLM_COH_DIFF | TV -> NLM -> Coherence-Guided Directional Diffusion | 12.0000 | 0.0022 | 26.6099 | 0.9106 | -0.7064 | -2.8267 | No |
| C2_NLM_MOD_GABOR | NLM -> Modified / Orientation-Adaptive Gabor | 12.0000 | 0.0030 | 25.2277 | 0.9091 | -2.0885 | -4.2089 | No |

## Compatibility Direct Altered Summary

| Candidate ID | Pipeline | Images | Coherence_mean | Local_contrast_mean | Fragmentation_mean | Delta_coherence_vs_P0_mean | Delta_contrast_vs_P0_mean | Delta_fragmentation_vs_P0_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C4_TV_NLM_COH_DIFF | TV -> NLM -> Coherence-Guided Directional Diffusion | 9.0000 | 0.7372 | 0.1644 | 2.8214 | 0.0667 | -0.0277 | 0.1848 |
| C1_TV_NLM | TV -> NLM | 9.0000 | 0.7360 | 0.1780 | 2.6376 | 0.0655 | -0.0140 | 0.0009 |
| C2_NLM_MOD_GABOR | NLM -> Modified / Orientation-Adaptive Gabor | 9.0000 | 0.7323 | 0.1721 | 2.7889 | 0.0618 | -0.0199 | 0.1522 |
| C3_NLM_COH_DIFF | NLM -> Coherence-Guided Directional Diffusion | 9.0000 | 0.7219 | 0.1722 | 2.7171 | 0.0514 | -0.0198 | 0.0805 |
| C0_NLM | NLM | 9.0000 | 0.7180 | 0.1871 | 2.5610 | 0.0475 | -0.0050 | -0.0757 |

## Best Hybrid Stage-Wise Controlled Analysis

Best hybrid by controlled PSNR: TV -> NLM (C1_TV_NLM).

| Stage order | Stage | MSE_mean | PSNR_mean | SSIM_mean | Delta_PSNR_vs_previous_mean | Delta_MSE_vs_previous_mean | Delta_SSIM_vs_previous_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | P0 degraded input | 0.0019 | 27.3162 | 0.8502 | N/A | N/A | N/A |
| 1.0000 | TV | 0.0012 | 29.1850 | 0.9402 | 1.8688 | -0.0006 | 0.0900 |
| 2.0000 | NLM | 0.0013 | 28.9436 | 0.9286 | -0.2414 | 0.0001 | -0.0116 |

## Best Hybrid Stage-Wise Direct Altered Analysis

| Stage order | Stage | Coherence_mean | Local_contrast_mean | Fragmentation_mean | Delta_coherence_vs_previous_mean | Delta_contrast_vs_previous_mean | Delta_fragmentation_vs_previous_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | P0 Altered input | 0.6705 | 0.1921 | 2.6367 | N/A | N/A | N/A |
| 1.0000 | TV | 0.6927 | 0.1832 | 2.7552 | 0.0221 | -0.0089 | 0.1185 |
| 2.0000 | NLM | 0.7360 | 0.1780 | 2.6376 | 0.0433 | -0.0052 | -0.1176 |

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

29.4366 dB.

G. What was M2 mean PSNR?

25.0002 dB.

H. What was M3 mean PSNR?

29.1850 dB.

I. What was M4 mean PSNR?

27.8028 dB.

J. Which member had the highest controlled PSNR?

Member 1: Non-Local Means Denoising.

K. Which member had the lowest MSE?

Member 1: Non-Local Means Denoising.

L. Which member had the highest SSIM?

Member 1: Non-Local Means Denoising.

M. Which member had the best ridge coherence on Altered images?

Member 1: Non-Local Means Denoising with coherence 0.7180.

N. Which member had the lowest fragmentation?

Member 1: Non-Local Means Denoising with fragmentation 2.5610.

O. Which member had the strongest contrast preservation?

Member 1: Non-Local Means Denoising had the smallest absolute contrast change versus P0, delta -0.0050.

P. Did Modified Gabor outperform Ordinary Gabor overall?

Yes by the balanced comparison. Modified Gabor had much higher controlled PSNR (25.0002 dB versus 18.0003 dB) and avoided the severe Ordinary Gabor contrast/fragmentation damage. Ordinary Gabor still had higher raw Altered coherence, so it remains useful as a comparator, not as the final method.

Q. Did Directional Diffusion provide useful structural benefit?

Only limited benefit in this fixed comparison. It is genuinely orientation-aware, but NLM -> Directional Diffusion reduced PSNR to 28.2968 dB and increased fragmentation versus NLM alone, so it should not be added to the final quantitative pipeline without a stronger structural reason.

R. Which individual technique is the strongest overall?

NLM is strongest overall because it has the highest controlled PSNR/MSE performance and remains structurally plausible on Altered images.

S. Which compatibility candidates were tested?

C0 NLM, C1 TV -> NLM, C2 NLM -> Modified Gabor, C3 NLM -> Coherence-Guided Directional Diffusion, and C4 TV -> NLM -> Coherence-Guided Directional Diffusion.

T. Did TV -> NLM outperform NLM?

No. TV -> NLM PSNR 28.9436 dB versus NLM 29.4366 dB.

U. Did NLM -> Modified Gabor outperform NLM?

No. NLM -> Modified Gabor PSNR 25.2277 dB versus NLM 29.4366 dB.

V. Did NLM -> Directional Diffusion outperform NLM?

No. NLM -> Directional Diffusion PSNR 28.2968 dB versus NLM 29.4366 dB.

W. Did TV -> NLM -> Directional Diffusion outperform NLM?

No. TV -> NLM -> Directional Diffusion PSNR 26.6099 dB versus NLM 29.4366 dB.

X. Which candidate had the highest PSNR?

NLM with PSNR 29.4366 dB.

Y. Which candidate had the lowest MSE?

NLM with MSE 0.00114977.

Z. Which candidate had the highest SSIM?

NLM with SSIM 0.9408.

AA. Which candidate had the best Altered structural behaviour?

By the simple structural score used only for interpretation, TV -> NLM had the best Altered structural behaviour. This does not override the controlled PSNR target.

AB. Did every useful candidate remain above 28.17 dB?

No for the five tested compatibility candidates.

AC. Which stages caused PSNR loss?

| Candidate ID | Pipeline | Stage | Delta PSNR vs previous stage (dB) |
| --- | --- | --- | --- |
| C2_NLM_MOD_GABOR | NLM -> Modified / Orientation-Adaptive Gabor | Modified Gabor | -4.2089 |
| C4_TV_NLM_COH_DIFF | TV -> NLM -> Coherence-Guided Directional Diffusion | Directional Diffusion | -2.3338 |
| C3_NLM_COH_DIFF | NLM -> Coherence-Guided Directional Diffusion | Directional Diffusion | -1.1398 |
| C1_TV_NLM | TV -> NLM | NLM | -0.2414 |
| C4_TV_NLM_COH_DIFF | TV -> NLM -> Coherence-Guided Directional Diffusion | NLM | -0.2414 |

AD. Which stages caused fragmentation increase?

| Candidate ID | Pipeline | Stage | Delta Fragmentation per 1000 ridge pixels vs previous stage |
| --- | --- | --- | --- |
| C2_NLM_MOD_GABOR | NLM -> Modified / Orientation-Adaptive Gabor | Modified Gabor | 0.2279 |
| C4_TV_NLM_COH_DIFF | TV -> NLM -> Coherence-Guided Directional Diffusion | Directional Diffusion | 0.1839 |
| C3_NLM_COH_DIFF | NLM -> Coherence-Guided Directional Diffusion | Directional Diffusion | 0.1562 |
| C1_TV_NLM | TV -> NLM | TV | 0.1185 |
| C4_TV_NLM_COH_DIFF | TV -> NLM -> Coherence-Guided Directional Diffusion | TV | 0.1185 |

AE. Is a hybrid genuinely better than NLM alone?

No. No tested hybrid clearly beat NLM under the fixed Gaussian-noise PSNR/MSE/SSIM acceptance guard.

AF. If not, should NLM alone be retained?

Yes.

AG. What ONE final candidate should be frozen for fresh hold-out validation?

NLM (C0_NLM).

AH. Are its parameters now frozen?

Yes. The parameters are frozen in `outputs/member_fixed_parameters.json`.

AI. Is the project ready for fresh final hold-out validation?

Yes, only for the selected frozen candidate and parameters. The next validation should still be a fresh hold-out run, not the 500-image final validation unless the team explicitly approves that final step.

## Verification Reference

The M2/M4 verification details are also written separately in `outputs/member_m2_m4_verification.md`.
