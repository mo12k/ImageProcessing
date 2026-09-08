# Fingerprint Enhancement System GUI

This Streamlit prototype provides a professional application layer for the completed BMDS2133 Fingerprint Enhancement System experiment.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Architecture

```text
Streamlit Frontend
|
v
Fingerprint_Enhancement_System.py
|
v
Classical Image Processing Techniques
```

The notebook remains the experimental evidence. The development experiment selected the final parameters, those parameters were frozen before validation, and the GUI applies the validated implementation to user-supplied fingerprint images.

The GUI does not retrain, re-optimise, rerun parameter search, or rerun the development/validation experiments. Reference-based PSNR, SSIM and MSE are shown only when a valid ground-truth image is supplied.
