# Final GUI Readiness Check

**Status: ready for GUI/prototype integration.** The GUI can import one core module without copying algorithm code.

Recommended calls:

- Load/P0: \`load_fingerprint\`, \`preprocess_p0\`, \`normalise_image\`
- Default final enhancement: \`apply_final_enhancement\` or \`enhance_fingerprint(image, method="nlm")\`
- Members: \`apply_nlm\`, \`apply_modified_gabor\`, \`apply_tv_restoration\`, \`apply_coherence_guided_directional_diffusion\`
- Baselines: \`apply_gaussian\`, \`apply_median\`, \`apply_clahe\`, \`apply_contrast_stretching\`, \`apply_ordinary_gabor\`, \`apply_wiener\`
- Structure: \`segment_fingerprint\`, \`morphological_cleanup\`, \`thin_ridges\`, \`medial_axis_analysis\`, \`extract_minutiae_detections\`
- Metrics: \`calculate_mse\`, \`calculate_psnr\`, \`calculate_ssim\`, \`calculate_ridge_coherence\`, \`calculate_local_contrast\`, \`calculate_fragmentation\`

\`enhance_fingerprint\` returns the enhanced float32 image and metadata suitable for GUI status/details panels. The default is OpenCV NLM with h=0.06. The GUI should keep grayscale enhancement and downstream binary/skeleton views as separate stages.
