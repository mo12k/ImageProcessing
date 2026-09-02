# Fingerprint Enhancement Project - Start Here

This package contains a complete Jupyter Notebook implementation of the proposed fingerprint image enhancement study. It is designed for Anaconda Navigator and uses the SOCOFing dataset.

## Files

- `Fingerprint_Enhancement_System.ipynb` - main Jupyter Notebook to open and run.
- `Fingerprint_Enhancement_System.py` - the same notebook in readable Python cell format; keep it as a backup.
- `environment.yml` - creates a compatible Conda environment and avoids common NumPy binary-version errors.
- `data/SOCOFing/` - place the extracted dataset here; the folder is created automatically when the notebook is first run.
- `outputs/` - generated CSV, PDF, image, JSON and SVM model files.

## Step 1 - Download the dataset

Download SOCOFing from:

https://www.kaggle.com/datasets/ruizgara/socofing

Extract the download. Copy the dataset into this project so that this path exists:

```text
Fingerprint_Enhancement_Project/
  data/
    SOCOFing/
      Real/
        1__M_Left_index_finger.BMP
        ...
      Altered/
        Altered-Easy/
        Altered-Medium/
        Altered-Hard/
```

The notebook also accepts an extra nested `SOCOFing/SOCOFing/Real` folder. Do not place the full external dataset in the submitted assignment ZIP because the specification requests a citation instead.

## Step 2 - Create the environment in Anaconda Navigator

1. Open **Anaconda Navigator**.
2. Select **Environments** on the left.
3. Select **Import** at the bottom.
4. Select `environment.yml` from this folder.
5. Keep the name `fingerprint-enhancement` and select **Import**.
6. Wait until the installation finishes.

The supplied environment uses NumPy 1.26.4, so it avoids the error saying that a module compiled with NumPy 1.x cannot run with NumPy 2.x.

## Step 3 - Launch and run Jupyter

1. Return to **Home** in Anaconda Navigator.
2. Choose `fingerprint-enhancement` from the environment menu.
3. Launch **JupyterLab** or **Notebook**.
4. Browse to this folder and open `Fingerprint_Enhancement_System.ipynb`.
5. Select **Kernel > Restart Kernel and Run All Cells** for the current development subset only.

Near the beginning, confirm that the notebook prints:

```text
SOCOFing detected successfully.
```

If it prints `Synthetic DEMO images will be used`, the dataset path is wrong. The synthetic fallback only proves that the code works and must not be used as the final assignment result.

## Step 4 - Collect reviewed results

The notebook runs in development mode by default and does not overwrite final-looking result files unless `EXPORT_FINAL_RESULTS = True` is deliberately set for an approved run. Review the dashboard, method table, graphs and extracted minutiae before exporting. The `outputs` folder may contain preserved diagnostic/final evidence:

- `batch_metrics.csv`
- `summary_metrics.csv`
- `fingerprint_enhancement_report.pdf`
- `example_minutiae_overlay.png`
- `experiment_metadata.json`

Use the real SOCOFing batch outputs in the report. Report matching score as a **matching score**, not as classification accuracy. Minutiae counts should be described as detected ridge endings and detected bifurcations unless ground-truth minutiae annotations are added. The optional SVM quality classifier is separate from the main classical image-processing comparison and is disabled by default.

## Contribution mapping

| Member | Main notebook sections |
|---|---|
| Member 1 | Section 4: M1 - Wiener + CLAHE Enhancement |
| Member 2 | Section 5: M2 - Gabor / Modified Gabor Ridge Enhancement |
| Member 3 | Section 6: M3 - Morphological Ridge Restoration |
| Member 4 | Section 7: M4 - Thinning & Minutiae Extraction |
| Whole team | Sections 8-13: controls, baselines, Team Hybrid Pipeline, batch evaluation, optional SVM and reporting |

Experimental controls and baselines are reported separately from individual member techniques:

- Degraded Input
- Global HE Baseline
- CLAHE Baseline
- Team Hybrid Pipeline

Each member should understand and be able to explain their functions, parameters, inputs, outputs and limitations during the demonstration.
