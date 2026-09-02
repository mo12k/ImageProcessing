# Member Technique Explanations

This file separates the four main contemporary member techniques from basic baselines and downstream structural operations.

## Member 1: Non-Local Means Denoising

Non-Local Means (NLM) is more substantial than Gaussian or median filtering because it denoises by comparing small patches rather than only using nearby pixels. For each target patch, the algorithm searches a neighbourhood, measures patch similarity, and forms a weighted average where more similar patches contribute more strongly. The denoising strength `h` controls how aggressively dissimilar patches are allowed to contribute.

For fingerprints this is scientifically relevant because ridge/valley patterns repeat locally. Similar ridge patches can reinforce one another while random noise averages out. The focused audit's strongest result used OpenCV NLM with `h=0.06`, preserving enough ridge intensity to exceed the PSNR target on Gaussian-noise-only controlled images.

## Member 2: Modified / Orientation-Adaptive Gabor

Ordinary Gabor filtering applies a fixed ridge frequency and a fixed set of orientations. It can increase ridge coherence, but it can also darken/brighten ridges unnaturally, reduce local contrast, and fragment ridge maps.

Modified or orientation-adaptive Gabor filtering is deeper because it estimates local fingerprint structure before filtering. The implementation investigates local ridge orientation, optional local ridge frequency, orientation-specific filtering, and adaptive blending. This makes the method fingerprint-specific rather than merely applying a generic filter bank.

The repository does not contain enough confirmed detail to claim exact reproduction of the selected literature article. Therefore the current implementation must be labelled `Literature-Inspired Modified Gabor` when compared with the article benchmark.

## Member 3: Total Variation Restoration

Total Variation (TV) restoration is an optimisation-based restoration method. It suppresses noise while penalising excessive image variation, which tends to preserve edges more carefully than simple smoothing. In fingerprint images, ridge/valley borders behave like important edges, so TV can reduce noise while trying to keep ridge transitions intact.

TV is more substantial than morphology because it operates on the grayscale restoration problem before binarisation. Morphological closing/opening are still useful, but they modify binary ridge maps after thresholding and should not replace TV as the main member contribution.

## Member 4: Coherence-Guided Directional Diffusion

Coherence-guided directional diffusion uses the local ridge-flow field to control smoothing. Instead of smoothing equally in all directions, it favours processing along coherent ridge directions and avoids unnecessary diffusion across ridge/valley boundaries.

This is a stronger member contribution than thinning because it enhances the grayscale ridge-flow image itself. Thinning, medial axis, and minutiae extraction are downstream structural analysis steps; they depend on the quality of the grayscale and binary ridge map generated earlier.

## Common P0 Preprocessing

P0 remains neutral: load image, convert to grayscale, resize if required, and convert to a consistent float range. No CLAHE, Gaussian, Median, Gabor, TV, NLM, or morphology is part of common preprocessing. Those are registered as experimental techniques, baselines, or downstream analysis.
