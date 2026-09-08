# Chapter 4 Experimental Results Revision Pack

This revision pack is based on the generated experiment outputs in
`Fingerprint_Enhancement_Project/outputs/final_mode_a/script/`. The editable
report source was not available in the workspace; only the compiled PDF was
found. Therefore, the following text is prepared as direct replacement and
insertion content for Chapter 4.

## 1. Revised Configuration Paragraph

Replace the existing paragraph beginning with "Table 4.1 summarises the selected
configuration..." and ending with "...were not modified using Validation Set
results." with:

Table 4.1 summarises the selected configuration for each enhancement technique
after Development Set parameter searching. NLM selected a denoising strength of
h = 0.07, TV Restoration selected a regularisation weight of 0.02, and
Coherence-Guided Directional Diffusion selected 2 iterations with a step size of
0.12. The final Modified Gabor configuration used detail_clip = 0.10,
frequencies = [0.075, 0.095, 0.115, 0.140], gamma = 0.50, orientation_bins = 12,
orientation_offset = 1.5707963267948966 radians, sigma_scale = 0.55, and
strength = 0.12. All selected configurations were finalised before validation
and were not modified using Validation Set results.

## 2. Corrected Table 4.1

**Table 4.1: Development-Selected Configuration for Each Enhancement Technique**

| Member | Technique | Development-Selected Configuration | Selection Status |
| --- | --- | --- | --- |
| M1 | NLM | h = 0.07 | Selected |
| M2 | Modified Gabor | detail_clip = 0.10; frequencies = [0.075, 0.095, 0.115, 0.140]; gamma = 0.50; orientation_bins = 12; orientation_offset = 1.5707963267948966 radians; sigma_scale = 0.55; strength = 0.12 | Selected |
| M3 | TV Restoration | weight = 0.02 | Selected |
| M4 | Coherence-Guided Directional Diffusion | iterations = 2; step = 0.12 | Selected |

## 3. Corrected Development Results Table

Replace the existing Table 4.2 with:

**Table 4.2: Quantitative Performance Comparison on the Development Dataset**

| Method | Images | Mean PSNR | Mean SSIM | Mean MSE | Delta PSNR | Delta SSIM | Mean Runtime (s/image) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Degraded Baseline | 500 | 27.357 | 0.8415 | 0.001842 | – | – | – |
| M1 NLM | 500 | 29.568 | 0.9336 | 0.001114 | 2.2110 | 0.0921 | 0.259 |
| M2 Modified Gabor | 500 | 27.367 | 0.8419 | 0.001838 | 0.0102 | 0.0004 | 0.468 |
| M3 TV | 500 | 29.287 | 0.9379 | 0.001188 | 1.9303 | 0.0964 | 0.239 |
| M4 Directional Diffusion | 500 | 27.986 | 0.8511 | 0.001593 | 0.6296 | 0.0096 | 0.528 |

## 4. Revised Development Results Discussion

Replace the existing development-results discussion beginning with "Table 4.2
presents the quantitative comparison..." and ending with "...used in the
experiment." with:

Table 4.2 presents the quantitative comparison on the 500-image Development Set.
NLM achieved the highest mean PSNR of 29.568 dB, improving PSNR by approximately
2.2110 dB over the degraded baseline. It also achieved a strong mean SSIM of
0.9336 and the lowest mean MSE of approximately 0.001114 among the four
enhancement techniques. TV Restoration achieved the highest mean SSIM of 0.9379
and recorded the fastest mean runtime among the enhancement methods at
approximately 0.239 seconds per image. Modified Gabor produced only a very small
quantitative improvement over the degraded baseline, while Coherence-Guided
Directional Diffusion produced a moderate improvement but remained below NLM and
TV in overall quantitative performance.

## 5. New Validation Results Table

Insert the following table immediately after the revised development-results
discussion. Since no later tables were present in Chapter 4, this should be
numbered Table 4.3.

**Table 4.3: Quantitative Performance Comparison on the Validation Dataset**

| Method | Images | Mean PSNR | Mean SSIM | Mean MSE | Delta PSNR | Delta SSIM | Mean Runtime (s/image) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Degraded Baseline | 1000 | 27.363 | 0.8385 | 0.001838 | – | – | – |
| M1 NLM | 1000 | 29.559 | 0.9300 | 0.001114 | 2.1962 | 0.0915 | 0.272 |
| M2 Modified Gabor | 1000 | 27.373 | 0.8388 | 0.001834 | 0.0096 | 0.0004 | 0.451 |
| M3 TV | 1000 | 29.321 | 0.9358 | 0.001177 | 1.9582 | 0.0973 | 0.230 |
| M4 Directional Diffusion | 1000 | 27.992 | 0.8481 | 0.001590 | 0.6293 | 0.0096 | 0.508 |

## 6. New Validation Results Discussion

Insert immediately after Table 4.3:

The validation results are highly consistent with the Development Set findings.
NLM achieved the highest validation PSNR of 29.559 dB, corresponding to an
improvement of 2.1962 dB compared with the degraded validation baseline, and
maintained a low mean MSE of 0.001114. TV Restoration achieved the highest
validation SSIM of 0.9358, produced the largest SSIM improvement of approximately
0.0973, and was also the fastest enhancement method on the Validation Set at
approximately 0.230 seconds per image. Directional Diffusion required
approximately 0.508 seconds per image, while Modified Gabor again produced only
minimal quantitative improvement. The similarity between the Development and
Validation Set results indicates that the relative performance of the methods
remained stable on unseen fingerprint images.

## 7. Revised Overall Comparison / Winner Paragraph

Replace the first two paragraphs of Section 4.2, from "Based on the predefined
Development Set ranking rule..." through "...in terms of SSIM and runtime." with:

Based on the predefined PSNR-first ranking criterion, Non-Local Means (NLM) was
selected as the overall best-performing enhancement method. It achieved the
highest PSNR and lowest MSE on both the Development and Validation Sets while
maintaining high SSIM values. On the Development Set, NLM achieved a mean PSNR
of 29.568 dB and the lowest mean MSE of 0.001114. Its performance remained stable
on the independent Validation Set, where it achieved a mean PSNR of 29.559 dB
and the same rounded mean MSE of 0.001114. This indicates that the selected NLM
configuration generalised well to fingerprint images from unseen subjects rather
than only performing well on the Development Set.

Nevertheless, Total Variation (TV) Restoration achieved the highest SSIM and
the fastest runtime on both datasets. On the Validation Set, TV achieved a mean
SSIM of 0.9358 compared with 0.9300 for NLM, and required approximately 0.230
seconds per image. Therefore, TV is particularly strong in structural
preservation and computational efficiency, while NLM provides the strongest
overall reconstruction quality according to the selected PSNR-first evaluation
criterion. The selection of NLM therefore reflects the weighting of the adopted
evaluation criteria rather than superiority across every individual metric.

## 8. Corrected Directional Diffusion Runtime Paragraph

Replace the paragraph beginning with "Coherence-Guided Directional Diffusion
produced a moderate improvement..." and ending with "...used in the current
experiment." with:

Coherence-Guided Directional Diffusion produced a moderate improvement over the
degraded baseline but required the highest processing time among the four
methods. On the Validation Set, it improved mean PSNR from 27.363 dB to
27.992 dB but required approximately 0.508 seconds per image. This suggests that
its structure-aware iterative processing introduces additional computational
cost while providing less benefit than NLM and TV under the Gaussian-noise
degradation used in the current experiment.

## 9. New Structural Evaluation Subsection

Insert the following subsection after the paragraph that explains why altered
fingerprints are kept separate from the main paired PSNR, SSIM, and MSE
experiment. Number it as Table 4.4.

### Structural Evaluation on Altered Fingerprint Images

Since the altered fingerprint subsets are not treated as pixel-aligned clean
reference pairs, their evaluation focuses on structural ridge measurements
rather than full-reference PSNR, SSIM, and MSE. Table 4.4 summarises the overall
structural results across 1,500 altered fingerprint images from the
Altered-Easy, Altered-Medium, and Altered-Hard subsets.

**Table 4.4: Structural Evaluation on Altered Fingerprint Images**

| Method | Altered Images | Mean Coherence | Mean Contrast | Mean Fragmentation | Mean Continuity | Mean Ridge Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Altered Baseline | 1500 | 0.6508 | 0.1826 | 5.1562 | 22.3965 | 0.3693 |
| M1 NLM | 1500 | 0.7234 | 0.1740 | 3.7403 | 28.2292 | 0.3617 |
| M2 Modified Gabor | 1500 | 0.6516 | 0.1828 | 5.1535 | 22.4020 | 0.3693 |
| M3 TV | 1500 | 0.6676 | 0.1742 | 4.6801 | 23.1128 | 0.3657 |
| M4 Directional Diffusion | 1500 | 0.6541 | 0.1777 | 5.1788 | 22.3086 | 0.3668 |

NLM demonstrated the strongest overall structural improvement on the altered
fingerprint evaluation. It increased mean coherence from 0.6508 to 0.7234 and
mean continuity from 22.3965 to 28.2292, while also reducing mean fragmentation
from 5.1562 to 3.7403. TV Restoration also improved coherence, continuity, and
fragmentation, but the magnitude of its structural improvement was smaller than
that of NLM. Modified Gabor produced only minimal structural change, while
Directional Diffusion showed a small coherence increase but did not improve
continuity or fragmentation overall. This structural evaluation is useful because
traditional full-reference metrics such as PSNR, SSIM, and MSE measure
similarity to a reference image, whereas altered-image structural evaluation
provides additional evidence about how well fingerprint ridge patterns are
preserved or restored under more difficult image degradation.

## 10. Replacement / Insertion Checklist

- Replace the Table 4.1 introductory configuration paragraph with the revised
  configuration paragraph in Section 1.
- Replace the existing Table 4.1 row for Modified Gabor, or replace the full
  Table 4.1 with the corrected version in Section 2.
- Replace the existing Table 4.2 with the corrected Development Dataset table
  in Section 3.
- Replace the outdated development-results discussion that reports NLM mean MSE
  as approximately 0.001121 with the revised paragraph in Section 4.
- Insert the new Validation Dataset table as Table 4.3 after the development
  results discussion.
- Insert the validation-results discussion in Section 6 immediately after
  Table 4.3.
- Replace the Section 4.2 overall winner discussion with the revised two
  paragraphs in Section 7.
- Replace the outdated TV validation runtime sentence reporting approximately
  0.254 seconds per image with the corrected value of approximately 0.230
  seconds per image, as included in Section 7.
- Replace the outdated Directional Diffusion validation runtime sentence
  reporting approximately 0.772 seconds per image with the corrected paragraph
  in Section 8.
- Insert the structural evaluation subsection and Table 4.4 in Section 9 after
  the paragraph explaining that altered fingerprints are not used for paired
  full-reference metrics.
- No figure renumbering is required because only tables are added. The newly
  added validation and structural tables should be numbered Table 4.3 and
  Table 4.4 respectively.
