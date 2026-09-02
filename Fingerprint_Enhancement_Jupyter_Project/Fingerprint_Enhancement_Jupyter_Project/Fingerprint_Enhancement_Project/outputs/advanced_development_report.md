# Advanced 24-Image Development Experiment

This diagnostic run uses the fixed 24-image development subset only. Existing 500-image evidence files are preserved.

## A. Candidate architectures implemented

- Candidate A: ROI segmentation, wavelet denoising, Wiener/deconvolution restoration, local normalisation, local orientation/frequency estimation, adaptive Gabor filtering and confidence-guided fusion.
- Candidate B: overlapping STFT/contextual enhancement with dominant ridge orientation/frequency estimation and restoration-branch fusion.
- Candidate C: orientation field, coherence-guided anisotropic diffusion, local ridge frequency estimation, Log-Gabor filtering and confidence-guided fusion.
- Team proposed: quality/coherence-guided fusion of Candidates A, B and C, blended conservatively with the restoration branch.

## Selected parameter configuration

Selected config: `psnr_preserving`. Selection was based on mean PSNR over the 24-image development subset, not per-image tuning.

## Stage-by-stage metrics

```text
                                        Candidate               Stage  MSE_mean  MSE_std  PSNR_mean  PSNR_std  SSIM_mean  SSIM_std  Coherence_mean  Coherence_std  PSNR improvement vs degraded (dB)
Candidate A - Restoration + Adaptive Gabor Fusion            Degraded    0.0312   0.0147    15.5893    2.2940     0.5196    0.1474          0.5953         0.1231                             0.0000
Candidate A - Restoration + Adaptive Gabor Fusion    Wavelet denoised    0.0311   0.0144    15.5953    2.2571     0.5393    0.1512          0.6337         0.1096                             0.0060
Candidate A - Restoration + Adaptive Gabor Fusion  Deblurred/restored    0.0320   0.0142    15.4197    2.1357     0.5387    0.1476          0.6483         0.1024                            -0.1696
Candidate A - Restoration + Adaptive Gabor Fusion    Local normalised    0.0338   0.0141    15.1124    1.9800     0.5339    0.1445          0.6486         0.1022                            -0.4769
Candidate A - Restoration + Adaptive Gabor Fusion      Adaptive Gabor    0.0480   0.0082    13.2562    0.7884     0.3671    0.0812          0.5931         0.0276                            -2.3331
Candidate A - Restoration + Adaptive Gabor Fusion        Fused output    0.0321   0.0141    15.3933    2.1110     0.5369    0.1468          0.6447         0.1013                            -0.1960
        Candidate B - STFT Contextual Enhancement            Degraded    0.0312   0.0147    15.5893    2.2940     0.5196    0.1474          0.5953         0.1231                             0.0000
        Candidate B - STFT Contextual Enhancement    Wavelet denoised    0.0311   0.0144    15.5953    2.2571     0.5393    0.1512          0.6337         0.1096                             0.0060
        Candidate B - STFT Contextual Enhancement  Deblurred/restored    0.0320   0.0142    15.4197    2.1357     0.5387    0.1476          0.6483         0.1024                            -0.1696
        Candidate B - STFT Contextual Enhancement    Local normalised    0.0334   0.0141    15.1745    2.0112     0.5349    0.1451          0.6485         0.1022                            -0.4148
        Candidate B - STFT Contextual Enhancement     STFT contextual    0.0588   0.0119    12.3963    0.9240     0.4064    0.1075          0.6792         0.0984                            -3.1930
        Candidate B - STFT Contextual Enhancement        Fused output    0.0322   0.0141    15.3773    2.1099     0.5369    0.1471          0.6483         0.1023                            -0.2120
    Candidate C - Coherence Diffusion + Log-Gabor            Degraded    0.0312   0.0147    15.5893    2.2940     0.5196    0.1474          0.5953         0.1231                             0.0000
    Candidate C - Coherence Diffusion + Log-Gabor    Wavelet denoised    0.0311   0.0144    15.5953    2.2571     0.5393    0.1512          0.6337         0.1096                             0.0060
    Candidate C - Coherence Diffusion + Log-Gabor  Deblurred/restored    0.0320   0.0142    15.4197    2.1357     0.5387    0.1476          0.6483         0.1024                            -0.1696
    Candidate C - Coherence Diffusion + Log-Gabor Coherence diffusion    0.0320   0.0141    15.4102    2.1227     0.5385    0.1470          0.6598         0.1012                            -0.1791
    Candidate C - Coherence Diffusion + Log-Gabor           Log-Gabor    0.1323   0.0112     8.8004    0.3703     0.1689    0.0553          0.8202         0.0521                            -6.7889
    Candidate C - Coherence Diffusion + Log-Gabor        Fused output    0.0324   0.0140    15.3295    2.0679     0.5339    0.1456          0.6601         0.1012                            -0.2598
```

## Member 1 baseline vs proposed

```text
                                    Method  MSE_mean  MSE_std  PSNR_mean  PSNR_std  SSIM_mean  SSIM_std  Coherence_mean  Coherence_std  PSNR improvement vs degraded (dB)  PSNR delta to 28.17 dB
                            CLAHE Baseline    0.0871   0.0219    10.7384    1.1344     0.5028    0.1088          0.5845         0.1097                            -4.8509                -17.4316
M1 Proposed - Wavelet + Wiener Restoration    0.0320   0.0142    15.4197    2.1357     0.5387    0.1476          0.6483         0.1024                            -0.1696                -12.7503
```

## Member 2 baseline vs proposed

```text
                                            Method  MSE_mean  MSE_std  PSNR_mean  PSNR_std  SSIM_mean  SSIM_std  Coherence_mean  Coherence_std  PSNR improvement vs degraded (dB)  PSNR delta to 28.17 dB
                      M2 Baseline - Ordinary Gabor    0.0557   0.0129    12.6504    0.9863     0.5165    0.1044          0.5928         0.1123                            -2.9390                -15.5196
M2 Proposed - Adaptive Orientation-Frequency Gabor    0.0480   0.0082    13.2562    0.7884     0.3671    0.0812          0.5931         0.0276                            -2.3331                -14.9138
```

## Candidate results

- Candidate A - Restoration + Adaptive Gabor Fusion: PSNR 15.3933 +/- 2.1110 dB; MSE 0.0321; SSIM 0.5369; coherence 0.6447; signed target gap -12.7767 dB.
- Candidate B - STFT Contextual Enhancement: PSNR 15.3773 +/- 2.1099 dB; MSE 0.0322; SSIM 0.5369; coherence 0.6483; signed target gap -12.7927 dB.
- Candidate C - Coherence Diffusion + Log-Gabor: PSNR 15.3295 +/- 2.0679 dB; MSE 0.0324; SSIM 0.5339; coherence 0.6601; signed target gap -12.8405 dB.

## Best candidates

Best PSNR: M1 Proposed - Wavelet + Wiener Restoration at 15.4197 dB (-12.7503 dB vs 28.17).
Best ridge coherence: Candidate C - Coherence Diffusion + Log-Gabor at 0.6601.

## Candidate summary

```text
                                                Method  MSE_mean  MSE_std  PSNR_mean  PSNR_std  SSIM_mean  SSIM_std  Coherence_mean  Coherence_std  PSNR improvement vs degraded (dB)  PSNR delta to 28.17 dB
                                        CLAHE Baseline    0.0871   0.0219    10.7384    1.1344     0.5028    0.1088          0.5845         0.1097                            -4.8509                -17.4316
            M1 Proposed - Wavelet + Wiener Restoration    0.0320   0.0142    15.4197    2.1357     0.5387    0.1476          0.6483         0.1024                            -0.1696                -12.7503
                          M2 Baseline - Ordinary Gabor    0.0557   0.0129    12.6504    0.9863     0.5165    0.1044          0.5928         0.1123                            -2.9390                -15.5196
    M2 Proposed - Adaptive Orientation-Frequency Gabor    0.0480   0.0082    13.2562    0.7884     0.3671    0.0812          0.5931         0.0276                            -2.3331                -14.9138
     Candidate A - Restoration + Adaptive Gabor Fusion    0.0321   0.0141    15.3933    2.1110     0.5369    0.1468          0.6447         0.1013                            -0.1960                -12.7767
             Candidate B - STFT Contextual Enhancement    0.0322   0.0141    15.3773    2.1099     0.5369    0.1471          0.6483         0.1023                            -0.2120                -12.7927
         Candidate C - Coherence Diffusion + Log-Gabor    0.0324   0.0140    15.3295    2.0679     0.5339    0.1456          0.6601         0.1012                            -0.2598                -12.8405
Team Proposed - Quality/Coherence-Guided Hybrid Fusion    0.0320   0.0141    15.4172    2.1339     0.5386    0.1476          0.6484         0.1023                            -0.1721                -12.7528
```

## Metric rankings

### PSNR
```text
 Rank                                                 Method  PSNR_mean
    1             M1 Proposed - Wavelet + Wiener Restoration    15.4197
    2 Team Proposed - Quality/Coherence-Guided Hybrid Fusion    15.4172
    3      Candidate A - Restoration + Adaptive Gabor Fusion    15.3933
    4              Candidate B - STFT Contextual Enhancement    15.3773
    5          Candidate C - Coherence Diffusion + Log-Gabor    15.3295
    6     M2 Proposed - Adaptive Orientation-Frequency Gabor    13.2562
    7                           M2 Baseline - Ordinary Gabor    12.6504
    8                                         CLAHE Baseline    10.7384
```
### SSIM
```text
 Rank                                                 Method  SSIM_mean
    1             M1 Proposed - Wavelet + Wiener Restoration     0.5387
    2 Team Proposed - Quality/Coherence-Guided Hybrid Fusion     0.5386
    3              Candidate B - STFT Contextual Enhancement     0.5369
    4      Candidate A - Restoration + Adaptive Gabor Fusion     0.5369
    5          Candidate C - Coherence Diffusion + Log-Gabor     0.5339
    6                           M2 Baseline - Ordinary Gabor     0.5165
    7                                         CLAHE Baseline     0.5028
    8     M2 Proposed - Adaptive Orientation-Frequency Gabor     0.3671
```
### Coherence
```text
 Rank                                                 Method  Coherence_mean
    1          Candidate C - Coherence Diffusion + Log-Gabor          0.6601
    2 Team Proposed - Quality/Coherence-Guided Hybrid Fusion          0.6484
    3              Candidate B - STFT Contextual Enhancement          0.6483
    4             M1 Proposed - Wavelet + Wiener Restoration          0.6483
    5      Candidate A - Restoration + Adaptive Gabor Fusion          0.6447
    6     M2 Proposed - Adaptive Orientation-Frequency Gabor          0.5931
    7                           M2 Baseline - Ordinary Gabor          0.5928
    8                                         CLAHE Baseline          0.5845
```

## Parameter search

```text
      Config ID                                                 Method  PSNR_mean  SSIM_mean  Coherence_mean  PSNR improvement vs degraded (dB)  PSNR delta to 28.17 dB
       balanced                                         CLAHE Baseline    10.7384     0.5028          0.5845                            -4.8509                -17.4316
psnr_preserving                                         CLAHE Baseline    10.7384     0.5028          0.5845                            -4.8509                -17.4316
ridge_coherence                                         CLAHE Baseline    10.7384     0.5028          0.5845                            -4.8509                -17.4316
psnr_preserving      Candidate A - Restoration + Adaptive Gabor Fusion    15.3933     0.5369          0.6447                            -0.1960                -12.7767
       balanced      Candidate A - Restoration + Adaptive Gabor Fusion    15.3222     0.5411          0.6590                            -0.2672                -12.8478
ridge_coherence      Candidate A - Restoration + Adaptive Gabor Fusion    15.1239     0.5266          0.6439                            -0.4655                -13.0461
psnr_preserving              Candidate B - STFT Contextual Enhancement    15.3773     0.5369          0.6483                            -0.2120                -12.7927
       balanced              Candidate B - STFT Contextual Enhancement    15.3170     0.5420          0.6678                            -0.2723                -12.8530
ridge_coherence              Candidate B - STFT Contextual Enhancement    15.1616     0.5292          0.6637                            -0.4277                -13.0084
psnr_preserving          Candidate C - Coherence Diffusion + Log-Gabor    15.3295     0.5339          0.6601                            -0.2598                -12.8405
       balanced          Candidate C - Coherence Diffusion + Log-Gabor    15.2372     0.5364          0.6954                            -0.3521                -12.9328
ridge_coherence          Candidate C - Coherence Diffusion + Log-Gabor    15.0534     0.5193          0.7115                            -0.5359                -13.1166
       balanced                                         Degraded Input    15.5893     0.5196          0.5953                             0.0000                -12.5807
psnr_preserving                                         Degraded Input    15.5893     0.5196          0.5953                             0.0000                -12.5807
ridge_coherence                                         Degraded Input    15.5893     0.5196          0.5953                             0.0000                -12.5807
psnr_preserving             M1 Proposed - Wavelet + Wiener Restoration    15.4197     0.5387          0.6483                            -0.1696                -12.7503
       balanced             M1 Proposed - Wavelet + Wiener Restoration    15.3835     0.5447          0.6675                            -0.2059                -12.7865
ridge_coherence             M1 Proposed - Wavelet + Wiener Restoration    15.2755     0.5336          0.6625                            -0.3138                -12.8945
psnr_preserving                           M2 Baseline - Ordinary Gabor    12.6504     0.5165          0.5928                            -2.9390                -15.5196
       balanced                           M2 Baseline - Ordinary Gabor    12.4520     0.5047          0.5934                            -3.1373                -15.7180
ridge_coherence                           M2 Baseline - Ordinary Gabor    12.1954     0.4895          0.5942                            -3.3939                -15.9746
psnr_preserving     M2 Proposed - Adaptive Orientation-Frequency Gabor    13.2562     0.3671          0.5931                            -2.3331                -14.9138
       balanced     M2 Proposed - Adaptive Orientation-Frequency Gabor    12.5271     0.3382          0.6321                            -3.0622                -15.6429
ridge_coherence     M2 Proposed - Adaptive Orientation-Frequency Gabor    10.4892     0.2651          0.6328                            -5.1001                -17.6808
psnr_preserving Team Proposed - Quality/Coherence-Guided Hybrid Fusion    15.4172     0.5386          0.6484                            -0.1721                -12.7528
       balanced Team Proposed - Quality/Coherence-Guided Hybrid Fusion    15.3767     0.5444          0.6680                            -0.2127                -12.7933
ridge_coherence Team Proposed - Quality/Coherence-Guided Hybrid Fusion    15.2580     0.5327          0.6639                            -0.3313                -12.9120
```

## Recommended next modification

Focus on restoration and degradation diagnosis. The ridge-enhancement stages improve coherence in places but lower PSNR, so the next change should target a less destructive restoration branch or stricter confidence fusion rather than stronger Gabor filtering.