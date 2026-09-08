# Submission Inspection Report

This inspection checks the current PDF report, project code, generated experiment
outputs, citations, and packaging readiness. The current PDF still contains the
old Chapter 4 values; the Chapter 4 revision pack must be applied before final
submission.

## Overall Verdict

Conditionally ready after revision.

The project code and generated experiment outputs support the revised Chapter 4
results. The submitted report will better fulfil the experimental-result
requirements once the outdated Chapter 4 tables, validation values, runtime
values, and altered-image structural results are updated using the revision
pack. The main remaining submission risks are formatting/package-related rather
than experimental-validity problems.

## Experiment and Code Consistency

- Python syntax check passed for `Fingerprint_Enhancement_System.py`, `app.py`,
  and `mode_a_equivalence.py`.
- `mode_a_equivalence.py` executed successfully with exit code 0.
- The generated split audit confirms 500 development images from 50 subjects,
  1000 validation images from 100 subjects, zero subject overlap, and zero image
  overlap.
- `final_selected_configuration.json` confirms that validation results were not
  used for tuning and that the selected method is M1 NLM.
- The final Modified Gabor parameters were found in the experiment output:
  detail_clip = 0.10; frequencies = [0.075, 0.095, 0.115, 0.140]; gamma = 0.50;
  orientation_bins = 12; orientation_offset = 1.5707963267948966 radians;
  sigma_scale = 0.55; strength = 0.12.
- The development and validation values in the revision pack match
  `development_member_comparison.csv` and `validation_member_comparison.csv`.
- The altered-image structural table uses actual values from
  `altered_structural_summary.csv`, not fabricated values.

## Chapter 4 Readiness

Chapter 4 in the current PDF does not yet fully fulfil the updated experiment
outputs because it still contains outdated values:

- Table 4.2 reports old mean MSE and runtime values.
- The development discussion reports NLM mean MSE as approximately 0.001121
  instead of approximately 0.001114.
- The validation table is missing.
- The altered-image structural results are discussed but not numerically
  presented.
- TV validation runtime is still stated as approximately 0.254 seconds per image
  instead of approximately 0.230 seconds per image.
- Directional Diffusion validation runtime is still stated as approximately
  0.772 seconds per image instead of approximately 0.508 seconds per image.

After applying `chapter4_experimental_results_revisions.md`, the result section
will align with the final experiment outputs and will more clearly distinguish
NLM as the PSNR-first winner from TV as the strongest SSIM/runtime method.

## Citation and Reference Check

The report includes references for the main techniques, dataset, metrics, and
contextual frameworks:

- NLM: Buades, Coll, and Morel (2005).
- Modified Gabor/fingerprint enhancement: Hong, Wan, and Jain (1998).
- TV restoration: Rudin, Osher, and Fatemi (1992), and Chambolle (2004).
- Directional diffusion: Weickert (1999).
- SSIM/image quality: Wang et al. (2004).
- SOCOFing dataset: Shehu et al. (2018) and the Kaggle dataset entry.
- Contextual motivation: United Nations SDG 9 and Malaysia MADANI.

The citation coverage is acceptable: each listed technical reference is relevant
to the implemented project, and the dataset citation is present. However, I
recommend the following clean-up before final submission:

- Check the year for Gonzalez and Woods, *Digital Image Processing* (4th ed.).
  The report uses 2017, while the ISBN 9781292223070 is commonly catalogued as
  the Pearson Global Edition published in 2018.
- Keep both SOCOFing references only if the report intentionally cites both the
  original arXiv paper and the Kaggle dataset page. Otherwise, combine or remove
  one to avoid redundancy.
- The United Nations SDG 9 page and Malaysia MADANI page are web pages; if your
  required citation style needs access dates, add retrieval dates.
- Regenerate the Table of Contents after inserting the new validation and
  structural-evaluation material.

## Submission Package Check

Files expected for a reproducible technical submission are present:

- Main implementation: `Fingerprint_Enhancement_System.py`.
- Streamlit prototype: `app.py`.
- Notebook evidence: `Fingerprint_Enhancement_System.ipynb`.
- Generated experiment outputs: `outputs/final_mode_a/script/`.
- Requirements file: `requirements.txt`.
- README: `README.md`.
- Compiled report PDF.

Items to verify before upload:

- The workspace contains the SOCOFing image data locally. The report states that
  the dataset is cited rather than bundled with the submitted archive. If the
  submission rules prohibit bundling datasets, exclude the actual SOCOFing image
  folders and keep only the placeholder/instructions.
- `environment.yml` is stricter than the tested local setup because it requires
  Python >=3.14,<3.15. The code was inspected using Python 3.12 with the
  installed package set. If graders use the conda environment file, consider
  loosening this to a version consistent with the tested runtime.
- The current PDF is not the final version until Chapter 4 is updated and the
  Table of Contents is refreshed.

## Final Readiness Answer

The project can fulfil the submission requirements after the Chapter 4 revision
pack is applied and the final report formatting is refreshed. The experimental
evidence is strong and traceable to code-generated outputs. The current PDF
alone is not yet fully ready because it contains outdated Chapter 4 numerical
results and missing validation/structural tables.

## External Reference Verification Sources

- Buades et al. NLM paper: https://ieeexplore.ieee.org/document/1467423/
- Chambolle TV paper: https://link.springer.com/article/10.1023/B:JMIV.0000011325.36760.1e
- Hong, Wan, and Jain fingerprint enhancement paper: https://doi.org/10.1109/34.709565
- SOCOFing arXiv paper: https://arxiv.org/abs/1807.10609
- SOCOFing Kaggle dataset: https://www.kaggle.com/datasets/ruizgara/socofing
- Wang et al. SSIM paper: https://ieeexplore.ieee.org/document/1284395
- Weickert diffusion paper: https://link.springer.com/article/10.1023/A:1008009714131
- United Nations SDG 9 page: https://sdgs.un.org/goals/goal9
- Malaysia MADANI page: https://malaysiamadani.gov.my/pengenalan/
