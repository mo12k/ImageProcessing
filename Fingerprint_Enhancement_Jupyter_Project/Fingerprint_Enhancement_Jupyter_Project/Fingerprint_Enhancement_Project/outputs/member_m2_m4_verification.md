# Member 2 and Member 4 Verification

## A. Is M2 genuinely modified/orientation-adaptive?

Yes. The audited M2 implementation is not only Ordinary Gabor with a different fixed parameter. The current advanced M2 method is `Literature-Inspired Modified / Orientation-Adaptive Gabor`, implemented through `adaptive_gabor_enhancement`.

## B. What exact adaptation is implemented?

- Local ridge orientation is estimated using Sobel gradients and a smoothed structure-tensor formulation in `estimate_orientation_field`.
- Local ridge frequency is estimated block-by-block in `estimate_local_frequency_map` using a Hanning window, FFT magnitude peak search, nearest-frequency selection, and a frequency confidence score.
- The adaptive Gabor stage builds 8 orientation bins and frequency candidates `[0.075, 0.095, 0.115, 0.140]`.
- For each pixel, the implementation selects the Gabor response matching that pixel's local orientation bin and local frequency bin.
- Blending is local, not uniform: `weight = blend * coherence_weight * frequency_confidence_weight * mask`, with fixed `blend=0.18` for the literature-inspired variant.

## C. How is it different from Ordinary Gabor?

Ordinary Gabor uses fixed frequency `0.115`, 8 orientations, maximum response energy, and uniform blend `0.30`. It does not use a local ridge-frequency map, per-pixel orientation/frequency response selection, or coherence/frequency-confidence weighting. M2 therefore has meaningful fingerprint-specific adaptation.

## D. Is M4 genuinely coherence-guided directional diffusion?

Yes. The audited M4 implementation is not Gaussian smoothing followed by a coherence score. It performs repeated orientation-selected smoothing and blends the smoothed result back into the image using local coherence and a fingerprint mask.

## E. Where is coherence used?

Inside each diffusion iteration, `coherence_guided_diffusion` estimates a coherence map and computes `weight = clip(coherence, 0, 1) * mask`. The update is `current = (1 - step * weight) * current + (step * weight) * smoothed`, so diffusion strength is higher in locally coherent ridge regions.

## F. Where is orientation used?

Inside each iteration, local orientation is assigned to one of 8 bins. For each orientation bin, an oriented Gaussian kernel is built and convolved with the current image. Pixels receive the convolved value from the kernel associated with their nearest local orientation bin.

## G. Does diffusion differ along ridge and across ridge direction?

Yes. The oriented kernel is anisotropic: the audited fixed parameters use `sigma_x=1.35`, `sigma_y=0.50`, and kernel size `9`, so smoothing is elongated in one direction and narrower in the perpendicular direction. The selected kernel orientation varies by local ridge orientation.

## H. Are any fallback algorithms being used?

For M2 and M4, no silent fallback was observed in the audited functions. For M1, the focused NLM function uses OpenCV when available and falls back to skimage NLM if OpenCV is unavailable. For M3, TV uses skimage directly; however, older wavelet helper code can fall back to TV if PyWavelets is unavailable, so wavelet labels must be checked in any separate wavelet run.

## I. Are the current technique names scientifically accurate?

Mostly yes, with one required wording guard. M2 should be called `Literature-Inspired Modified / Orientation-Adaptive Gabor`, not exact article reproduction, because exact article parameters, dataset, degradation, MAX_I, and formula remain unknown in the available repository context. M4 is scientifically accurate as `Coherence-Guided Directional Diffusion`.
