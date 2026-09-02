# hybrid24 Controlled Hybrid Compatibility + Final PSNR/MSE Candidate Selection Report

## A. Which 24 Altered images were reused?
- Reused the Direct24 image list exactly: {'Easy': 8, 'Hard': 8, 'Medium': 8}. Filename-level Real pair candidates existed for {'Easy': 8, 'Hard': 8, 'Medium': 8}, but no full-reference direct metrics used these pairs because geometric alignment was not verified.
| Category   |   Category order | Image                              | Path                                                                    |
|:-----------|-----------------:|:-----------------------------------|:------------------------------------------------------------------------|
| Easy       |                0 | 100__M_Left_index_finger_CR.BMP    | data\SOCOFing\Altered\Altered-Easy\100__M_Left_index_finger_CR.BMP      |
| Easy       |                1 | 179__M_Left_little_finger_Obl.BMP  | data\SOCOFing\Altered\Altered-Easy\179__M_Left_little_finger_Obl.BMP    |
| Easy       |                2 | 255__M_Right_middle_finger_Obl.BMP | data\SOCOFing\Altered\Altered-Easy\255__M_Right_middle_finger_Obl.BMP   |
| Easy       |                3 | 332__M_Left_thumb_finger_CR.BMP    | data\SOCOFing\Altered\Altered-Easy\332__M_Left_thumb_finger_CR.BMP      |
| Easy       |                4 | 40__F_Left_index_finger_CR.BMP     | data\SOCOFing\Altered\Altered-Easy\40__F_Left_index_finger_CR.BMP       |
| Easy       |                5 | 487__M_Right_index_finger_Zcut.BMP | data\SOCOFing\Altered\Altered-Easy\487__M_Right_index_finger_Zcut.BMP   |
| Easy       |                6 | 564__M_Left_ring_finger_Obl.BMP    | data\SOCOFing\Altered\Altered-Easy\564__M_Left_ring_finger_Obl.BMP      |
| Easy       |                7 | 9__M_Right_thumb_finger_Zcut.BMP   | data\SOCOFing\Altered\Altered-Easy\9__M_Right_thumb_finger_Zcut.BMP     |
| Medium     |                0 | 100__M_Left_index_finger_CR.BMP    | data\SOCOFing\Altered\Altered-Medium\100__M_Left_index_finger_CR.BMP    |
| Medium     |                1 | 178__M_Right_thumb_finger_Zcut.BMP | data\SOCOFing\Altered\Altered-Medium\178__M_Right_thumb_finger_Zcut.BMP |
| Medium     |                2 | 256__M_Right_little_finger_Obl.BMP | data\SOCOFing\Altered\Altered-Medium\256__M_Right_little_finger_Obl.BMP |
| Medium     |                3 | 333__M_Left_little_finger_CR.BMP   | data\SOCOFing\Altered\Altered-Medium\333__M_Left_little_finger_CR.BMP   |
| Medium     |                4 | 40__F_Left_little_finger_CR.BMP    | data\SOCOFing\Altered\Altered-Medium\40__F_Left_little_finger_CR.BMP    |
| Medium     |                5 | 489__M_Left_index_finger_Zcut.BMP  | data\SOCOFing\Altered\Altered-Medium\489__M_Left_index_finger_Zcut.BMP  |
| Medium     |                6 | 565__M_Right_middle_finger_Obl.BMP | data\SOCOFing\Altered\Altered-Medium\565__M_Right_middle_finger_Obl.BMP |
| Medium     |                7 | 9__M_Right_thumb_finger_Zcut.BMP   | data\SOCOFing\Altered\Altered-Medium\9__M_Right_thumb_finger_Zcut.BMP   |
| Hard       |                0 | 100__M_Left_index_finger_CR.BMP    | data\SOCOFing\Altered\Altered-Hard\100__M_Left_index_finger_CR.BMP      |
| Hard       |                1 | 179__M_Right_little_finger_CR.BMP  | data\SOCOFing\Altered\Altered-Hard\179__M_Right_little_finger_CR.BMP    |
| Hard       |                2 | 25__F_Left_ring_finger_Obl.BMP     | data\SOCOFing\Altered\Altered-Hard\25__F_Left_ring_finger_Obl.BMP       |
| Hard       |                3 | 335__M_Left_middle_finger_CR.BMP   | data\SOCOFing\Altered\Altered-Hard\335__M_Left_middle_finger_CR.BMP     |
| Hard       |                4 | 40__F_Right_little_finger_Obl.BMP  | data\SOCOFing\Altered\Altered-Hard\40__F_Right_little_finger_Obl.BMP    |
| Hard       |                5 | 48__F_Left_index_finger_Obl.BMP    | data\SOCOFing\Altered\Altered-Hard\48__F_Left_index_finger_Obl.BMP      |
| Hard       |                6 | 566__M_Left_middle_finger_Zcut.BMP | data\SOCOFing\Altered\Altered-Hard\566__M_Left_middle_finger_Zcut.BMP   |
| Hard       |                7 | 9__M_Right_thumb_finger_Zcut.BMP   | data\SOCOFing\Altered\Altered-Hard\9__M_Right_thumb_finger_Zcut.BMP     |

## B. Which baseline pipelines were evaluated?
- B0: P0 only. B1: P0 -> CLAHE. CLAHE is treated as the main enhancement baseline, not assumed weaker than longer hybrids.

## C. Which hybrid candidates were evaluated and why?
| pipeline_id   | pipeline                                                         | purpose                                                                                | final_output_domain                   |
|:--------------|:-----------------------------------------------------------------|:---------------------------------------------------------------------------------------|:--------------------------------------|
| C1            | P0 -> CLAHE -> Ordinary Gabor                                    | measure exact trade-off of the Direct24 M2 coherence winner                            | grayscale                             |
| C2            | P0 -> CLAHE -> Coherence-Guided Directional Diffusion            | test whether directional enhancement is more compatible with CLAHE than Ordinary Gabor | grayscale                             |
| C3            | P0 -> CLAHE -> Total Variation Restoration                       | test CLAHE followed by the best Direct24 supporting restoration candidate              | grayscale                             |
| C4            | P0 -> Total Variation Restoration -> CLAHE                       | explicit order test for restoration before local contrast enhancement                  | grayscale                             |
| C5            | P0 -> CLAHE -> Coherence-Guided Directional Diffusion -> Closing | test whether morphology can repair ridge gaps after directional enhancement            | morphology-assisted structural output |
| C6            | P0 -> CLAHE -> Total Variation Restoration -> Closing            | test whether morphology improves continuity after CLAHE plus TV restoration            | morphology-assisted structural output |

## D. Which candidates improved ridge coherence?
| Pipeline ID   | Pipeline                                                         |   Overall coherence |   Delta coherence vs P0 |   Delta coherence vs CLAHE |
|:--------------|:-----------------------------------------------------------------|--------------------:|------------------------:|---------------------------:|
| B1            | P0 -> CLAHE                                                      |              0.6985 |                  0.0452 |                     0      |
| C1            | P0 -> CLAHE -> Ordinary Gabor                                    |              0.7261 |                  0.0727 |                     0.0276 |
| C2            | P0 -> CLAHE -> Coherence-Guided Directional Diffusion            |              0.7062 |                  0.0528 |                     0.0077 |
| C3            | P0 -> CLAHE -> Total Variation Restoration                       |              0.7255 |                  0.0722 |                     0.027  |
| C4            | P0 -> Total Variation Restoration -> CLAHE                       |              0.7475 |                  0.0942 |                     0.049  |
| C5            | P0 -> CLAHE -> Coherence-Guided Directional Diffusion -> Closing |              0.7062 |                  0.0528 |                     0.0077 |
| C6            | P0 -> CLAHE -> Total Variation Restoration -> Closing            |              0.7255 |                  0.0722 |                     0.027  |

## E. Which candidates damaged contrast?
| Pipeline ID   | Pipeline                      |   Overall contrast |   Delta contrast vs CLAHE | Compatibility interpretation                                      |
|:--------------|:------------------------------|-------------------:|--------------------------:|:------------------------------------------------------------------|
| B0            | P0 only                       |             0.1915 |                   -0.0318 | control baseline                                                  |
| C1            | P0 -> CLAHE -> Ordinary Gabor |             0.1644 |                   -0.0589 | contrast damaged; fragmentation worsens; category-specific damage |

## F. Which candidates increased fragmentation?
| Pipeline ID   | Pipeline                      |   Overall fragmentation |   Delta fragmentation vs CLAHE | Compatibility interpretation                                      |
|:--------------|:------------------------------|------------------------:|-------------------------------:|:------------------------------------------------------------------|
| C1            | P0 -> CLAHE -> Ordinary Gabor |                  3.9991 |                         1.1859 | contrast damaged; fragmentation worsens; category-specific damage |

## G. Which candidate was best on Easy?
| Best Pipeline ID   | Best Pipeline                              | Selection reason                                                           |   Coherence |   Contrast |   Fragmentation |
|:-------------------|:-------------------------------------------|:---------------------------------------------------------------------------|------------:|-----------:|----------------:|
| C4                 | P0 -> Total Variation Restoration -> CLAHE | highest coherence among candidates passing non-damage compatibility guards |      0.7452 |     0.2214 |          3.2107 |

## H. Which candidate was best on Medium?
| Best Pipeline ID   | Best Pipeline                              | Selection reason                                                           |   Coherence |   Contrast |   Fragmentation |
|:-------------------|:-------------------------------------------|:---------------------------------------------------------------------------|------------:|-----------:|----------------:|
| C4                 | P0 -> Total Variation Restoration -> CLAHE | highest coherence among candidates passing non-damage compatibility guards |      0.7583 |     0.2098 |          2.1348 |

## I. Which candidate was best on Hard?
| Best Pipeline ID   | Best Pipeline                              | Selection reason                                                           |   Coherence |   Contrast |   Fragmentation |
|:-------------------|:-------------------------------------------|:---------------------------------------------------------------------------|------------:|-----------:|----------------:|
| C4                 | P0 -> Total Variation Restoration -> CLAHE | highest coherence among candidates passing non-damage compatibility guards |       0.739 |     0.2095 |          3.1851 |

## J. Was CLAHE alone stronger than some hybrids?
- Yes. CLAHE alone preserved contrast and fragmentation better than C1 Ordinary Gabor and was competitive with longer pipelines. More stages did not automatically improve the direct evidence.
| Pipeline ID   | Pipeline                                                         |   Ridge coherence mean |   Local contrast mean |   Fragmentation mean |   Delta contrast vs CLAHE mean |   Delta fragmentation vs CLAHE mean |
|:--------------|:-----------------------------------------------------------------|-----------------------:|----------------------:|---------------------:|-------------------------------:|------------------------------------:|
| B0            | P0 only                                                          |                 0.6533 |                0.1915 |               3.2154 |                        -0.0318 |                              0.4022 |
| B1            | P0 -> CLAHE                                                      |                 0.6985 |                0.2233 |               2.8132 |                         0      |                              0      |
| C1            | P0 -> CLAHE -> Ordinary Gabor                                    |                 0.7261 |                0.1644 |               3.9991 |                        -0.0589 |                              1.1859 |
| C2            | P0 -> CLAHE -> Coherence-Guided Directional Diffusion            |                 0.7062 |                0.2076 |               2.8949 |                        -0.0157 |                              0.0817 |
| C3            | P0 -> CLAHE -> Total Variation Restoration                       |                 0.7255 |                0.2037 |               2.9321 |                        -0.0196 |                              0.1189 |
| C4            | P0 -> Total Variation Restoration -> CLAHE                       |                 0.7475 |                0.2136 |               2.8435 |                        -0.0097 |                              0.0303 |
| C5            | P0 -> CLAHE -> Coherence-Guided Directional Diffusion -> Closing |                 0.7062 |                0.2076 |               1.718  |                        -0.0157 |                             -1.0952 |
| C6            | P0 -> CLAHE -> Total Variation Restoration -> Closing            |                 0.7255 |                0.2037 |               1.9165 |                        -0.0196 |                             -0.8967 |

## K. Is Ordinary Gabor still recommended after considering fragmentation?
- Not as the final grayscale enhancement pipeline. C1 had coherence 0.7261, but contrast changed -0.0589 and fragmentation changed +1.1859 versus CLAHE.

## L. Is Directional Diffusion more compatible with CLAHE than Ordinary Gabor?
- Yes if judged by damage control. C2 versus C1: contrast delta vs CLAHE -0.0157 vs -0.0589; fragmentation delta vs CLAHE +0.0817 vs +1.1859.

## M. Is Total Variation Restoration more compatible with CLAHE?
- Yes. C3 changed contrast -0.0196 and fragmentation +0.1189 versus CLAHE, while preserving a smoother ridge structure than C1.

## N. Does processing order matter for CLAHE and TV restoration?
- Yes. C3 CLAHE -> TV: coherence 0.7255, contrast 0.2037, fragmentation 2.9321. C4 TV -> CLAHE: coherence 0.7475, contrast 0.2136, fragmentation 2.8435.

## O. Does Closing genuinely repair fragmentation after enhancement?
| Comparison                 |   Fragmentation delta |   Continuity delta |
|:---------------------------|----------------------:|-------------------:|
| C5 Closing vs C2 grayscale |               -1.1769 |             5.55   |
| C6 Closing vs C3 grayscale |               -1.0156 |             4.1493 |
- Closing is useful as M3 structural processing, but it is not a grayscale enhancement output for PSNR/MSE/SSIM.

## P. Which grayscale pipelines were selected as the maximum three finalists?
| Pipeline ID   | Pipeline                                   | Finalist role                    | Selection reason                                                                 | Strong compatible candidate   | Compatibility interpretation   |
|:--------------|:-------------------------------------------|:---------------------------------|:---------------------------------------------------------------------------------|:------------------------------|:-------------------------------|
| B1            | P0 -> CLAHE                                | mandatory CLAHE baseline         | Direct24 and hybrid24 both treat CLAHE as the main M1 baseline.                  | True                          | compatible                     |
| C4            | P0 -> Total Variation Restoration -> CLAHE | best compatible grayscale hybrid | Passed coherence, contrast, fragmentation and per-category compatibility guards. | True                          | compatible                     |
| C3            | P0 -> CLAHE -> Total Variation Restoration | order-effect competitor          | C3/C4 explicitly test whether TV before or after CLAHE is preferable.            | True                          | compatible                     |

## Q. What controlled aligned-reference dataset/protocol was used for PSNR/MSE/SSIM?
- Reused the screening24 controlled protocol: 24 Real references from screening24_metadata, severity cycle [0.35, 0.55, 0.75], seeds RANDOM_SEED + index, original simulate_degradation function, aligned grayscale outputs only.
- B0 P0-only controlled baseline, included as a no-enhancement reference but not as an enhancement finalist: MSE 0.029308 +/- 0.014158, PSNR 15.8970 +/- 2.3719 dB, SSIM 0.5081 +/- 0.1574.
- Comparable previous screening24 controlled values:
| Comparable source                               | Protocol                                                 |   MSE_mean |   PSNR_mean |   SSIM_mean |
|:------------------------------------------------|:---------------------------------------------------------|-----------:|------------:|------------:|
| screening24 Contrast Stretching                 | same screening24 controlled 24 Real/degradation protocol |     0.023  |     17.1119 |      0.5541 |
| screening24 CLAHE                               | same screening24 controlled 24 Real/degradation protocol |     0.0936 |     10.4546 |      0.4603 |
| screening24 Wiener Deconvolution                | same screening24 controlled 24 Real/degradation protocol |     0.0276 |     16.1707 |      0.5436 |
| screening24 Wavelet Denoising                   | same screening24 controlled 24 Real/degradation protocol |     0.029  |     15.9232 |      0.5394 |
| screening24 Total Variation Restoration         | same screening24 controlled 24 Real/degradation protocol |     0.0308 |     15.5634 |      0.5719 |
| screening24 Ordinary Gabor                      | same screening24 controlled 24 Real/degradation protocol |     0.0448 |     13.6777 |      0.4409 |
| screening24 Literature-Inspired Modified Gabor  | same screening24 controlled 24 Real/degradation protocol |     0.0298 |     15.7256 |      0.5097 |
| screening24 previous team hybrid R1->G7->fusion | same screening24 controlled 24 Real/degradation protocol |     0.0277 |     16.1493 |      0.5427 |

## R. What is the mean +/- standard deviation MSE for each finalist?
- C4 P0 -> Total Variation Restoration -> CLAHE: 0.049310 +/- 0.012402
- C3 P0 -> CLAHE -> Total Variation Restoration: 0.081852 +/- 0.022267
- B1 P0 -> CLAHE: 0.093614 +/- 0.025823

## S. What is the mean +/- standard deviation PSNR for each finalist?
- C4 P0 -> Total Variation Restoration -> CLAHE: 13.2241 +/- 1.2280 dB
- C3 P0 -> CLAHE -> Total Variation Restoration: 11.0294 +/- 1.2246 dB
- B1 P0 -> CLAHE: 10.4546 +/- 1.2645 dB

## T. What is the mean +/- standard deviation SSIM for each finalist?
- C4 P0 -> Total Variation Restoration -> CLAHE: 0.6092 +/- 0.1102
- C3 P0 -> CLAHE -> Total Variation Restoration: 0.5086 +/- 0.1270
- B1 P0 -> CLAHE: 0.4603 +/- 0.1331

## U. Which finalist has the highest PSNR?
- C4 P0 -> Total Variation Restoration -> CLAHE with PSNR 13.2241 +/- 1.2280 dB.

## V. Which finalist has the lowest MSE?
- C4 P0 -> Total Variation Restoration -> CLAHE with MSE 0.049310 +/- 0.012402.

## W. Which finalist has the highest SSIM?
- C4 P0 -> Total Variation Restoration -> CLAHE with SSIM 0.6092 +/- 0.1102.

## X. Does the quantitative winner also perform acceptably on original Altered fingerprint structure?
- Quantitative winner C4 direct compatibility: compatible. Acceptable under guards: True.

## Y. Which processing stage produces the greatest PSNR increase?
- C3 / Total Variation Restoration: mean delta PSNR +0.5747 dB in the controlled aligned experiment.
- For selected C4, no individual enhancement stage increased PSNR over its immediately previous stage; the value of TV-before-CLAHE is that it makes the final CLAHE result less damaging than CLAHE-only under the controlled protocol.
- Selected C4 controlled stage deltas:
|   Stage order | Stage                       |   Delta_PSNR |   Delta_MSE |   Delta_SSIM |
|--------------:|:----------------------------|-------------:|------------:|-------------:|
|             1 | Total Variation Restoration |      -0.3336 |      0.0015 |       0.0638 |
|             2 | CLAHE                       |      -2.3394 |      0.0185 |       0.0374 |

## Z. Which processing stage produces the largest MSE reduction?
- C3 / Total Variation Restoration: mean delta MSE -0.011763 in the controlled aligned experiment.
- For selected C4, no individual enhancement stage reduced MSE over its immediately previous stage; the final result is still lower-MSE than CLAHE-only because CLAHE is applied after restoration.

## AA. Which processing stage causes the greatest structural damage?
- C1 / Ordinary Gabor: mean fragmentation delta +1.1859 in direct Altered-image evaluation.
- Selected C4 direct structural stage deltas:
|   Stage order | Stage                       |   Delta_coherence |   Delta_contrast |   Delta_fragmentation |
|--------------:|:----------------------------|------------------:|-----------------:|----------------------:|
|             1 | Total Variation Restoration |            0.0465 |          -0.0181 |                0.0849 |
|             2 | CLAHE                       |            0.0477 |           0.0402 |               -0.4568 |

## AB. Is the best hybrid genuinely better than CLAHE alone?
- True. Selected hybrid C4 improved direct coherence versus CLAHE by +0.0490, kept contrast loss within the guardrail at -0.0097, and changed fragmentation by +0.0303. It also outperformed B1 in finalist PSNR/MSE/SSIM, while B0 remains a no-enhancement pixel-distance reference.

## AC. Is the best hybrid genuinely better than the strongest existing single technique?
- False. Direct-structure evidence favors C4 over the strongest Direct24 single technique by coherence (Ordinary Gabor: 0.7178 vs 0.7475). Under the same controlled screening24 protocol, it does not beat the best previous single-technique PSNR result (screening24 Contrast Stretching: 17.1119 dB), so it should be described as the balanced finalist rather than the absolute PSNR champion.

## AD. Is a simpler pipeline preferable to the longer team hybrid?
- True. The previous Direct24 full hybrid improved coherence but worsened contrast and fragmentation; this experiment prefers the shorter grayscale C4 pipeline and keeps Closing/Medial Axis as downstream structural analysis only.

## AE. Why is the literature 28.17 dB result not automatically a required target?
- The reported literature PSNR is not directly comparable unless the dataset, degradation process, reference definition, preprocessing, parameters and PSNR computation protocol are equivalent. Those details are not fully available here, so 28.17 dB remains context, not a forced target.

## AF. What is the recommended FINAL grayscale fingerprint enhancement pipeline?
- Recommended final grayscale enhancement pipeline: P0 -> Total Variation Restoration -> CLAHE. It is the best balanced hybrid finalist, not the absolute pixel-distance champion; B0 remains a no-enhancement reference and screening24 Contrast Stretching remains the strongest previous single-technique PSNR comparator under the same controlled protocol.

## AG. Which operations belong only to downstream structural analysis rather than enhancement?
- Closing, segmentation/binarization, medial axis, skeletonization and minutiae extraction belong to downstream M3/M4 structural analysis. They must not be treated as final enhanced grayscale fingerprints or evaluated with PSNR/MSE/SSIM.

## AH. Is the evidence now strong enough to proceed to a larger dataset evaluation?
- Yes, with a narrow validation only. Next justified run should be a larger direct Altered-image validation of CLAHE and the selected compatible grayscale finalist only, plus the controlled 24 protocol retained as secondary quantitative evidence. Runtime was 84.05 seconds; protected previous evidence unchanged: True.
- Visual evidence files:
- outputs\hybrid24_direct_easy_comparison.png
- outputs\hybrid24_direct_medium_comparison.png
- outputs\hybrid24_direct_hard_comparison.png
- outputs\hybrid24_controlled_reference_comparison.png
