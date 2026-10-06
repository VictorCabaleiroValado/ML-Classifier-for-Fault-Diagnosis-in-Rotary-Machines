> Historical methodology. These scenarios were retired from the active demo after real 50/75 RPM originals were recovered on 5 October 2026. Files now live in `demo/archive/`. Current data and evaluation: [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

# Synthetic 50 / 75 RPM scenarios

These scenarios test the sensitivity of a 25 RPM classifier to artificial signal changes. The numbers 50/75 RPM are nominal scenario names, not operating points demonstrated by this model. **They are not recovered measurements, a calibrated physical model, or evidence of diagnosis accuracy at 50/75 RPM.** At the time of this experiment, historical 50/75 RPM tables were unverified. They have since been superseded by reconstructed real-data tables.

## Method and assumptions

1. Fit the original decision tree only on the same 780 real 25 RPM rows (seed 42). Keep the original 195-row holdout. Select the lowest held-out row per category: 39 parents, without selecting for successful predictions.
2. For each parent and target label, take the first `4096 × factor` full-resolution samples on all nine channels, where factor is 2 for the 50 RPM scenario and 3 for 75 RPM.
3. Apply SciPy `resample_poly` with up=1, down=factor and its default anti-alias filter to produce 4,096 samples. This compresses the sample-index pattern; there is no known sampling-rate calibration. It cannot establish a physical shaft speed.
4. Preserve each channel mean and multiply its deviations by 1.15 or 1.30. Add Gaussian noise with standard deviation equal to 1% of that transformed channel's standard deviation. Gain/noise are explicit illustrative choices, not fitted machine physics. Use NumPy SeedSequence `[42, target_rpm, parent_row]`.
5. Write a named nine-channel CSV with ten significant digits, reload those exact bytes, and recompute all 72 time features. The exported tree prediction must match scikit-learn. Hash both the original source and the new CSV.

The resulting 78 synthetic CSVs cover 39 inherited conditions at each scenario label. The transformations share parents, so they are neither new independent experiments nor new confirmed faults. Tachometer and other channels are treated as numeric signals; their synthetic values do not validate RPM. Aliasing suppression, cropping, gain and noise can change class information. This is useful for observing distribution shift, not for drawing physical conclusions.

## Interpretation

The bundled results show poor transfer to these perturbations: only **5/39** outputs at nominal 50 RPM and **6/39** at nominal 75 RPM match the inherited labels. Only **4/39** and **6/39**, respectively, retain the same prediction as their real 25 RPM parent. These are different comparisons: a stable prediction may already disagree with its source label, and a changed prediction may happen to agree.

The earlier interface showed these counts before analysis and treats synthetic outputs as sensitivity results, not correct/incorrect diagnoses at higher speeds. Both changed and unchanged examples are accessible; all 39 cases remain available. Labels, signals and the fitted model are unchanged. We do not tune perturbations, relabel outputs or train on test descendants to manufacture agreement. Software parity tests passing means the intended calculation executes, not that the classifier generalizes to these inputs.

Cropping, resampling, gain and noise all change together; this is not an ablation that identifies a causal speed effect. Original measurements have now been recovered; independent acquisition-level evaluation remains necessary.

The 25 RPM real row-level holdout metric remains 187/195 (95.9%), subject to the existing acquisition-independence and filename-label limits. It is never assigned to the synthetic datasets. Synthetic JSON has `accuracy: null`; the UI reports only agreements with **inherited** labels. The model is not retrained on descendants. The browser resets the result whenever speed or condition changes.

The healthy comparison is always from the same selected dataset and transformation. Its amplitude is not an alarm threshold. Charts are peak-preserving overviews; model features use the full 64,000-sample real signal or the full 4,096-sample synthetic CSV, respectively.

## Files and reproducibility

- `demo/archive/data.json`: the exact historical real 25 RPM parent examples and model.
- `demo/archive/synthetic-50.json`, `demo/archive/synthetic-75.json`: synthetic examples and the unchanged real-trained model.
- `demo/archive/csv/synthetic_<speed>rpm_<condition>_source<row>.csv`: readable, full synthetic signals.
- `demo/archive/manifest.json`: origin, label basis, parent row, kind, speed, filenames and hashes for all 117 examples.
- Downloaded feature comparisons and analysis JSON include the kind and scenario speed in both filenames and content. Original source filenames remain available for audit.

The commands below reproduce the **historical version**, not the current exporter. Use a separate checkout of commit `3dabdc72cf5226bcfb28daec488aa2306645789f`; do not run them over the active real-data demo.

```sh
python demo/export_demo.py '/path/to/Fault data split 25' demo
python -m pytest -q
node demo/check_browser_model.cjs demo/data.json
node demo/check_browser_model.cjs demo/synthetic-50.json
node demo/check_browser_model.cjs demo/synthetic-75.json
python -m http.server 8000 --directory demo
```

Original 25 RPM files are required only for regeneration, not to run the bundled demo. Never replace the legacy feature tables with these scenarios or present the new CSVs as measured evidence. For the current recovered measurements and models, follow DATA_PROVENANCE.md and demo/README.md. Independent acquisition-run identifiers and an external test are still needed.
