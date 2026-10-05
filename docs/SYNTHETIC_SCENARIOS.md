# Synthetic 50 / 75 RPM scenarios

These scenarios make the software demonstrable at three selectable speed labels. **They are not recovered measurements, a calibrated physical model, or evidence of diagnosis accuracy at 50/75 RPM.** The historical feature tables remain unchanged and unverified.

## Method and assumptions

1. Fit the original decision tree only on the same 780 real 25 RPM rows (seed 42). Keep the original 195-row holdout. Select the lowest held-out row per category: 39 parents, without selecting for successful predictions.
2. For each parent and target label, take the first `4096 × factor` full-resolution samples on all nine channels, where factor is 2 for the 50 RPM scenario and 3 for 75 RPM.
3. Apply SciPy `resample_poly` with up=1, down=factor and its default anti-alias filter to produce 4,096 samples. This compresses the sample-index pattern; there is no known sampling-rate calibration. It cannot establish a physical shaft speed.
4. Preserve each channel mean and multiply its deviations by 1.15 or 1.30. Add Gaussian noise with standard deviation equal to 1% of that transformed channel's standard deviation. Gain/noise are explicit illustrative choices, not fitted machine physics. Use NumPy SeedSequence `[42, target_rpm, parent_row]`.
5. Write a named nine-channel CSV with ten significant digits, reload those exact bytes, and recompute all 72 time features. The exported tree prediction must match scikit-learn. Hash both the original source and the new CSV.

The resulting 78 synthetic CSVs cover 39 inherited conditions at each scenario label. The transformations share parents, so they are neither new independent experiments nor new confirmed faults. Tachometer and other channels are treated as numeric signals; their synthetic values do not validate RPM. Aliasing suppression, cropping, gain and noise can change class information. This is useful for observing distribution shift, not for drawing physical conclusions.

## Interpretation

The 25 RPM real row-level holdout metric remains 187/195 (95.9%), subject to the existing acquisition-independence and filename-label limits. It is never assigned to the synthetic datasets. Synthetic JSON has `accuracy: null`; the UI reports only agreements with **inherited** labels. The model is not retrained on descendants. The browser resets the result whenever speed or condition changes.

The healthy comparison is always from the same selected dataset and transformation. Its amplitude is not an alarm threshold. Charts are peak-preserving overviews; model features use the full 64,000-sample real signal or the full 4,096-sample synthetic CSV, respectively.

## Files and reproducibility

- `demo/data.json`: real 25 RPM examples and model.
- `demo/synthetic-50.json`, `demo/synthetic-75.json`: synthetic examples and the unchanged real-trained model.
- `demo/csv/synthetic_<speed>rpm_<condition>_source<row>.csv`: readable, full synthetic signals.
- `demo/manifest.json`: origin, label basis, parent row, kind, speed, filenames and hashes for all 117 examples.
- Downloaded feature comparisons and analysis JSON include the kind and scenario speed in both filenames and content. Original source filenames remain available for audit.

```sh
python demo/export_demo.py '/path/to/Fault data split 25' demo
python -m pytest -q
node demo/check_browser_model.cjs demo/data.json
node demo/check_browser_model.cjs demo/synthetic-50.json
node demo/check_browser_model.cjs demo/synthetic-75.json
python -m http.server 8000 --directory demo
```

Original 25 RPM files are required only for regeneration, not to run the bundled demo. Never replace the legacy feature tables with these scenarios or present the new CSVs as measured evidence. To validate real higher-speed performance, acquire new labeled signals with verified instrumentation and independent acquisition-run identifiers.
