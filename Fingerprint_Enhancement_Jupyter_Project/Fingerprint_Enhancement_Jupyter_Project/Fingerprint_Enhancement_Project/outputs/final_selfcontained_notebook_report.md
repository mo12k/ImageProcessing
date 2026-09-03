# Final self-contained notebook report

**A.** Yes. The previous consolidated notebook was presentation-oriented.  
**B.** Yes. It loaded old member, compatibility, hold-out, 500-image and Altered CSV/JSON outputs.  
**C.** Yes. Multiple 3.3–3.6 MB pre-consolidation versions were recovered and protected.  
**D.** Dataset loading, P0, baseline filters, structural processing, metrics, controlled degradation, batch execution and visualisation were restored.  
**E.** Yes; M1 OpenCV NLM is visible, with normalised h=0.06 and windows 7/21.  
**F.** Yes; M2 visibly estimates local orientation/frequency and applies adaptive Gabor selection.  
**G.** Yes; M3 TV restoration with fixed weight 0.02 is visible.  
**H.** Yes; M4 coherence-guided directional diffusion is visible.  
**I.** Yes; Gaussian, median, CLAHE, contrast stretching, ordinary Gabor and Wiener baselines are present.  
**J.** Yes; segmentation, thresholding, opening/closing, cleanup, skeletonisation, medial axis and minutiae detections are present.  
**K.** Yes; MSE, PSNR, SSIM, coherence, contrast, fragmentation, continuity and ridge coverage functions are present.  
**L.** Yes; the four-member comparison is recomputed from 12 deterministically selected Real images.  
**M.** Yes; all five compatibility candidates are recomputed.  
**N.** Yes; the frozen fresh 100-image hold-out is recomputed.  
**O.** Yes; the canonical 500-image validation is recomputed.  
**P.** Yes; 500 Easy, 500 Medium and 500 Hard originals are recomputed (1,500 total).  
**Q.** The regenerated 500-image mean PSNR is **29.6303 dB**.  
**R.** **500/500** exceeded 28.17 dB.  
**S.** Difference from 29.6303 dB is **0.0000065 dB**.  
**T.** Yes. Overall coherence changed by **+0.05724**, fragmentation by **-0.12772**, and contrast by **-0.00566**: coherence and fragmentation improved with mild contrast loss.  
**U.** No key final result is loaded from historical CSV/JSON.  
**V.** Yes. A clean ordered Run All completed and the notebook was saved with fresh outputs.  
**W.** Yes. `Fingerprint_Enhancement_System.py` remains available and was not redesigned.  
**X.** No experimental script was deleted.  
**Y.** Later archive candidates are `direct24_original_altered_experiment.py`, `focused_psnr_audit_optimisation.py`, `hybrid24_compatibility_experiment.py`, `member_allocation_consolidation.py`, `member_fixed_comparison.py`, `final_holdout_validation.py`, `final_large_validation.py`, `screening24_master_audit.py`, `build_selfcontained_notebook.py`, `finalise_selfcontained_notebook.py`. See `final_file_cleanup_recommendation.csv`; deletion is not recommended in this task.  
**Z.** Yes. The `.ipynb` is now suitable as the primary assignment submission source: it is readable, executable and evidence-producing.

## Recomputed headline results

- Members: NLM 29.4366, TV 29.1850, directional diffusion 27.8028, modified Gabor 25.0002 dB.
- Fresh 100: 29.6443 dB; 99/100 above 28.17.
- Final 500: 29.6303 dB; 500/500 above 28.17.
- Original Altered: 1,500 images; conclusions reproduced without paired-reference claims.
