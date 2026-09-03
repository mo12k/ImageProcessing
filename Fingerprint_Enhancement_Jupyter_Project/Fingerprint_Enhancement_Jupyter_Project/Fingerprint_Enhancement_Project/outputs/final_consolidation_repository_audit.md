# Final Consolidation Repository Audit

## Scope and structure

The project root contains the final notebook/core, the complete SOCOFing data tree, saved outputs, backups, specifications, and eight experiment/evidence scripts. \`README.md\` is not present; \`README_FIRST.md\` is the entry point. Backups were inspected as preserved snapshots and were not changed.

## Canonical implementations

- **A — NLM:** \`focused_psnr_audit_optimisation.non_local_means\`; OpenCV uint8 conversion, normalized \`h=0.06\`, template window 7, search window 21.
- **B — Modified Gabor:** \`screening24_master_audit.adaptive_gabor_enhancement\`, frozen through \`member_fixed_comparison.py\` with local orientation/frequency, eight bins, bandwidth 1.8, blend 0.18.
- **C — TV:** \`focused_psnr_audit_optimisation.tv_restoration\`, frozen weight 0.02.
- **D — Directional diffusion:** \`screening24_master_audit.coherence_guided_diffusion\`, four iterations and step 0.16.

## Duplication and dependencies

Common duplicated families include float clipping/normalization, P0 loading/resizing, masks, coherence, local contrast, fragmentation, MSE/PSNR/SSIM, and technique wrappers. The final reusable copies are now in \`Fingerprint_Enhancement_System.py\`; evidence scripts remain unchanged.

Experiment-script imports:

- \`direct24_original_altered_experiment.py\` imports \`screening24_master_audit.py\`.
- \`hybrid24_compatibility_experiment.py\` imports \`direct24_original_altered_experiment.py\`.
- \`hybrid24_compatibility_experiment.py\` imports \`screening24_master_audit.py\`.
- \`member_fixed_comparison.py\` imports \`direct24_original_altered_experiment.py\`.
- \`member_fixed_comparison.py\` imports \`focused_psnr_audit_optimisation.py\`.
- \`member_fixed_comparison.py\` imports \`screening24_master_audit.py\`.
- \`final_holdout_validation.py\` imports \`direct24_original_altered_experiment.py\`.
- \`final_holdout_validation.py\` imports \`focused_psnr_audit_optimisation.py\`.
- \`final_large_validation.py\` imports \`direct24_original_altered_experiment.py\`.
- \`final_large_validation.py\` imports \`final_holdout_validation.py\`.
- \`final_large_validation.py\` imports \`focused_psnr_audit_optimisation.py\`.

All experiment scripts define their project/output locations relative to \`Path(__file__).resolve().parent\`; several also import sibling experiment modules. Moving only some scripts would change both import resolution and the meaning of their \`ROOT/PROJECT_ROOT\` paths. README/notebook references also identify root-level filenames.

## Organisation decision

The eight experiment scripts were **left at the project root**. This is the safest reproducibility-preserving choice: moving them would require coordinated import and path rewrites and would make historical scripts differ from the code that generated the preserved evidence. They are now documented as Experimental / Evidence Scripts. No data, outputs, backups, specifications, or evidence scripts were moved or edited.
