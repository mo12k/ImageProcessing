# Fingerprint Enhancement Project — Start Here

## Final system files

- \`Fingerprint_Enhancement_System.ipynb\` — final presentation/demonstration notebook; loads saved large-validation evidence.
- \`Fingerprint_Enhancement_System.py\` — single reusable processing module and future GUI API.
- \`environment.yml\` — Conda environment definition.

## Final methodology

| Member | Final technique |
|---|---|
| Member 1 | Non-Local Means (NLM) Denoising |
| Member 2 | Modified / Orientation-Adaptive Gabor Filtering |
| Member 3 | Total Variation Restoration |
| Member 4 | Coherence-Guided Directional Diffusion |

The final recommended grayscale method is **OpenCV Non-Local Means, h = 0.06**. Its uint8 conversion and 7/21 windows are frozen. CLAHE remains a conventional baseline; it is not Member 1's final technique.

## Final validation

- Fresh hold-out: 100 Real images, mean PSNR 29.6443 dB; 99/100 above 28.17 dB.
- Large robustness evaluation: 500 Real images, mean PSNR **29.6303 dB**; **500/500** above the **28.17 dB** benchmark.
- Original Altered evaluation: 1,500 images; coherence and fragmentation improved overall, with mild contrast reduction.
- No tested hybrid outperformed NLM alone.

## Experimental / evidence scripts

- \`screening24_master_audit.py\`
- \`direct24_original_altered_experiment.py\`
- \`hybrid24_compatibility_experiment.py\`
- \`focused_psnr_audit_optimisation.py\`
- \`member_allocation_consolidation.py\`
- \`member_fixed_comparison.py\`
- \`final_holdout_validation.py\`
- \`final_large_validation.py\`

These remain at the project root because they use sibling imports and project-root-relative data/output paths. They are preserved evidence, not the final application API. Do not rerun large experiments from the presentation notebook.

## Running the final notebook

Create/import the environment from \`environment.yml\`, open \`Fingerprint_Enhancement_System.ipynb\`, and use **Restart Kernel and Run All Cells**. The notebook reads saved CSV/JSON evidence and performs only one optional representative-image demonstration.

## GUI integration

A future GUI should import \`enhance_fingerprint\` or \`apply_final_enhancement\` from \`Fingerprint_Enhancement_System.py\`. The next phase is GUI/dashboard prototype development—not another enhancement experiment.
