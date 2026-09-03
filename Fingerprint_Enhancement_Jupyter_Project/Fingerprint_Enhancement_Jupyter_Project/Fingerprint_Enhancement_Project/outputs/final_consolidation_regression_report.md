# Final NLM Consolidation Regression Report

Ten deterministic images were selected from \`outputs/final_large_real_image_list.csv\`. For each image, the saved final-large Gaussian-noise protocol and seed were applied once, then the old validated \`focused_psnr_audit_optimisation.non_local_means\` and new \`Fingerprint_Enhancement_System.apply_nlm\` were compared.

- Outputs numerically equivalent: **Yes**
- Maximum pixel difference: **0**
- Maximum PSNR difference: **0 dB**
- Maximum MSE difference: **0**
- Maximum SSIM difference: **0**
- Output dimensions equal for all samples: **Yes**
- Validated final algorithm preserved: **Yes**

This was the specifically requested small regression check. It did not rerun the 500-image or 1,500-image experiments.
