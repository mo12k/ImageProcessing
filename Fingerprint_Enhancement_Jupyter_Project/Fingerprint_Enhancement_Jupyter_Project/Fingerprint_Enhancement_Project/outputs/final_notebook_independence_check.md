# Final notebook independence check

- Static audit covered all 23 code cells.
- Historical result-read calls found: **none**.
- Dataset images are the only required experimental inputs. Frozen image-name lists are embedded as protocol configuration; no measurements are embedded.
- The notebook writes newly computed CSV outputs under `outputs/`, but does not read them.
- A clean, ordered nbconvert execution completed with 0 error outputs and saved all generated cell outputs.
- Key results can therefore be recomputed when historical CSV/JSON result artifacts are absent.
