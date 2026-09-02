# direct24 Direct Original Altered Fingerprint Enhancement Report

## A. What original dataset images were directly tested?
- 24 original SOCOFing Altered images were tested directly: {'Easy': 8, 'Hard': 8, 'Medium': 8}. Selection was deterministic: sorted file lists with 8 evenly spaced paths per category.
- Real filename pair candidates existed as follows: {'Easy': 8, 'Hard': 8, 'Medium': 8}. These pairs were documented but not used for metrics because geometric alignment was not verified.

## B. Was synthetic degradation used in the main experiment?
- No. The main experiment used original Altered Easy/Medium/Hard images directly.

## C. What neutral preprocessing was used?
- P0 only: read image, convert to grayscale if needed, resize to 256x256 for consistent processing, convert to float [0,1]. No filtering, CLAHE, histogram equalisation, denoising, sharpening, or Gabor filtering was applied as common preprocessing.

## D. Which M1 techniques were tested?
- CLAHE, Contrast Stretching, Global Histogram Equalization.

## E. Which M2 techniques were tested?
- Ordinary Gabor, Literature-Inspired Modified Gabor, Coherence-Guided Directional Diffusion, Directional Oriented Filter-Bank.

## F. Which restoration/denoising supporting techniques were tested?
- Wiener Deconvolution, Wavelet Denoising, Total Variation Restoration. These are labelled as supporting restoration/denoising candidates, not as CLAHE/Gabor equivalents.

## G. Which M3 techniques were tested?
- Closing, Opening + Closing, Morphological Reconstruction, Remove Small Objects, Controlled Ridge-Restoration Pipeline.

## H. Which M4 techniques were tested?
- Morphological Skeleton and Medial Axis.

## I. Which metrics were used for direct real-image evaluation, and why?
- Direct metrics: ridge coherence, local contrast, foreground/background separation proxy, ridge coverage, connected-component fragmentation, small connected-component count, ridge-continuity proxy, detected ridge endings/bifurcations, and runtime.
- These are no-reference or structural indicators suited to unregistered Altered fingerprints. PSNR/MSE/SSIM are not primary direct rankings because clean pixel-aligned ground truth is not verified.

## J. What was the best M1 technique for Easy, Medium, Hard and overall?
| Category   | Technique   | Main strength             | Main weakness                                 |   Ridge coherence |   Local contrast |   Fragmentation |
|:-----------|:------------|:--------------------------|:----------------------------------------------|------------------:|-----------------:|----------------:|
| Easy       | CLAHE       | coherence change +0.0420  | no major structural penalty in selected score |            0.6988 |           0.2309 |          3.2334 |
| Medium     | CLAHE       | coherence change +0.0478  | fragmentation worsens +0.0074                 |            0.7026 |           0.2198 |          2.0599 |
| Hard       | CLAHE       | continuity change +0.4755 | no major structural penalty in selected score |            0.6941 |           0.2192 |          3.1463 |
| Overall    | CLAHE       | coherence change +0.0452  | no major structural penalty in selected score |            0.6985 |           0.2233 |          2.8132 |

## K. What was the best M2 technique for Easy, Medium, Hard and overall?
| Category   | Technique      | Main strength            | Main weakness                 |   Ridge coherence |   Local contrast |   Fragmentation |
|:-----------|:---------------|:-------------------------|:------------------------------|------------------:|-----------------:|----------------:|
| Easy       | Ordinary Gabor | coherence change +0.0607 | fragmentation worsens +0.9548 |            0.7175 |           0.1501 |          4.7414 |
| Medium     | Ordinary Gabor | coherence change +0.0661 | fragmentation worsens +1.5523 |            0.7209 |           0.1398 |          3.6048 |
| Hard       | Ordinary Gabor | coherence change +0.0664 | fragmentation worsens +1.7449 |            0.7149 |           0.1386 |          5.5519 |
| Overall    | Ordinary Gabor | coherence change +0.0644 | fragmentation worsens +1.4173 |            0.7178 |           0.1428 |          4.6327 |

## L. What was the best M3 technique?
| Category   | Technique   | Main strength             | Main weakness                                 |   Fragmentation |   Ridge continuity |
|:-----------|:------------|:--------------------------|:----------------------------------------------|----------------:|-------------------:|
| Easy       | Closing     | continuity change +6.5123 | no major structural penalty in selected score |          2.7955 |            30.4624 |
| Medium     | Closing     | continuity change +8.3497 | no major structural penalty in selected score |          1.4024 |            41.352  |
| Hard       | Closing     | continuity change +3.3269 | no major structural penalty in selected score |          2.691  |            22.7585 |
| Overall    | Closing     | continuity change +6.0630 | no major structural penalty in selected score |          2.2963 |            31.5243 |

## M. What was the best M4 technique?
| Category   | Technique   | Main strength             | Main weakness                 |   Fragmentation |   Ridge continuity |
|:-----------|:------------|:--------------------------|:------------------------------|----------------:|-------------------:|
| Easy       | Medial Axis | continuity change +5.8856 | fragmentation worsens +9.3150 |         13.1017 |            29.8357 |
| Medium     | Medial Axis | continuity change +7.8570 | fragmentation worsens +4.8717 |          6.9243 |            40.8592 |
| Hard       | Medial Axis | continuity change +3.4015 | fragmentation worsens +8.3832 |         12.1902 |            22.8331 |
| Overall    | Medial Axis | continuity change +5.7147 | fragmentation worsens +7.5233 |         10.7387 |            31.176  |

## N. Did any technique visibly damage ridge information despite improving a metric?
| Family                               | Technique ID   | Technique              |   Delta coherence vs P0 mean |   Delta contrast vs P0 mean |   Delta fragmentation vs P0 mean |   Delta continuity vs P0 mean |
|:-------------------------------------|:---------------|:-----------------------|-----------------------------:|----------------------------:|---------------------------------:|------------------------------:|
| M4 / Thinning                        | M4-T1          | Morphological Skeleton |                       0      |                      0      |                           8.0943 |                        8.0078 |
| M4 / Thinning                        | M4-T2          | Medial Axis            |                       0      |                      0      |                           7.5233 |                        5.7147 |
| M2 / Ridge Enhancement               | M2-R1          | Ordinary Gabor         |                       0.0644 |                     -0.0487 |                           1.4173 |                       -5.5378 |
| M3 / Morphological Ridge Restoration | M3-3           | Opening + Closing      |                       0      |                      0      |                           1.2217 |                       -6.2283 |
- Visual comparison PNGs were exported for one Easy, one Medium and one Hard example; use these to confirm whether metric gains match ridge visibility.

## O. Which technique produced the best ridge coherence?
- Ordinary Gabor (M2 / Ridge Enhancement), overall ridge coherence 0.7178.

## P. Which technique produced the best ridge continuity / lowest fragmentation?
- Closing (M3 / Morphological Ridge Restoration), fragmentation 2.2963.

## Q. Which technique produced the best contrast improvement?
- CLAHE (M1 / Contrast or General Enhancement), delta local contrast vs P0 +0.0318.

## R. What technique is recommended for each team member and why?
| Family                                       | Technique                   | Easy                        | Medium               | Hard                        | Overall                     | Main strength             | Main weakness                                 |
|:---------------------------------------------|:----------------------------|:----------------------------|:---------------------|:----------------------------|:----------------------------|:--------------------------|:----------------------------------------------|
| M1 / Contrast or General Enhancement         | CLAHE                       | CLAHE                       | CLAHE                | CLAHE                       | CLAHE                       | coherence change +0.0452  | no major structural penalty in selected score |
| M2 / Ridge Enhancement                       | Ordinary Gabor              | Ordinary Gabor              | Ordinary Gabor       | Ordinary Gabor              | Ordinary Gabor              | coherence change +0.0644  | fragmentation worsens +1.4173                 |
| Restoration / Denoising Supporting Candidate | Total Variation Restoration | Total Variation Restoration | Wiener Deconvolution | Total Variation Restoration | Total Variation Restoration | coherence change +0.0465  | fragmentation worsens +0.0849                 |
| M3 / Morphological Ridge Restoration         | Closing                     | Closing                     | Closing              | Closing                     | Closing                     | continuity change +6.0630 | no major structural penalty in selected score |
| M4 / Thinning                                | Medial Axis                 | Medial Axis                 | Medial Axis          | Medial Axis                 | Medial Axis                 | continuity change +5.7147 | fragmentation worsens +7.5233                 |

## S. What one evidence-based team hybrid was constructed?
- DIRECT24-H1: P0 -> CLAHE -> Ordinary Gabor -> Closing -> Medial Axis as downstream analysis.

## T. Did the hybrid improve over the original Altered inputs?
- Hybrid greyscale branch vs P0 overall: coherence delta +0.0727, contrast delta -0.0271, fragmentation delta +0.7837. Genuinely better by the strict coherence-plus-fragmentation rule: False.

## U. Did the hybrid outperform the strongest single technique?
- Strongest single technique by coherence: Ordinary Gabor at 0.7178. Hybrid greyscale branch mean coherence gap: +0.0083. Verdict: Yes by ridge coherence only; no as a universal structural winner because contrast and fragmentation must also be checked.

## V. Which hybrid stage helped most?
- M1 - CLAHE gave the largest mean coherence gain: +0.0452.

## W. Which hybrid stage damaged quality most?
- M4 - Medial Axis gave the largest mean fragmentation increase: +8.9406.

## X. How do results differ between Easy, Medium and Hard?
| Category   |   Ridge coherence mean |   Local contrast mean |   Fragmentation mean |   Ridge continuity mean |
|:-----------|-----------------------:|----------------------:|---------------------:|------------------------:|
| Easy       |                 0.6568 |                0.2019 |               3.7866 |                 23.9501 |
| Hard       |                 0.6485 |                0.1845 |               3.8071 |                 19.4316 |
| Medium     |                 0.6548 |                0.1881 |               2.0526 |                 33.0023 |
- The per-category tables above show family winners can differ by difficulty; do not force one universal winner where the category evidence differs.

## Y. Why are PSNR/MSE/SSIM not the main ranking metrics in this direct experiment?
- The direct inputs are Altered images whose clean Real counterparts may be transformed or cropped. Without verified pixel alignment, full-reference metrics would measure registration differences rather than enhancement quality.

## Z. What role does the previous synthetic-degradation experiment still serve?
- It remains a secondary controlled quantitative experiment: Real reference -> synthetic degradation -> enhancement -> PSNR/MSE/SSIM against Real. It answers recoverability under known degradation, not direct real Altered-image enhancement.

## AA. Is the literature 28.17 dB Modified Gabor result directly comparable to this project?
- No. The reported 28.17 dB is contextual literature evidence only. It is not directly comparable unless dataset, image selection, degradation, preprocessing, parameters, PSNR formula and reference design are all the same, and those details remain unavailable here.

## AB. Based on the direct 24-image evidence, is the experiment now ready for a larger dataset run?
- Not yet. Review the direct visual panels and per-category winners first, then run one larger direct Altered-image validation only for the selected M1/M2/M3/M4 pipeline and strongest single-technique baseline.
- Visual evidence files:
- outputs\direct24_visual_easy_comparison.png
- outputs\direct24_visual_medium_comparison.png
- outputs\direct24_visual_hard_comparison.png
