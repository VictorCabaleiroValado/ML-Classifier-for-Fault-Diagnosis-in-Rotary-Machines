# Interactive vibration demo

[Try the browser demo](https://victorcabaleirovalado.github.io/demo/)

Select a real 25 RPM measurement, inspect a peak-preserving motor signal overview, and run a decision tree locally in the browser. The UI compares the predicted category with the filename-derived recorded label and shows the first three threshold decisions. It does not claim calibrated confidence or physical causal explanations.

## Reproduce the export

From the repository root, in the Python environment described in the main README:

```sh
python demo/export_demo.py "/path/to/Fault data split 25" /tmp/vibration-demo
cp demo/index.html demo/style.css demo/demo.js /tmp/vibration-demo/
python -m http.server 8000 --directory /tmp/vibration-demo
```

Open http://localhost:8000. The original measurement folder is required for regeneration; it is not bundled in this repository. The published JSON includes the tree, 39 derived signal overviews, full feature vectors, source filenames and hashes, training/test row indices, feature-table hash and package versions.

## Evaluation and limits

- Uses only the reconstructed 25 RPM time-domain table: 72 features, 39 categories.
- A single decision tree is fitted on 780 training rows; 195 rows are held out, stratified by category, seed 42. No fitting on demo rows.
- One example per category is selected by the lowest held-out row index, without conditioning selection on the prediction.
- Each example's full-resolution extracted features must numerically match the reconstructed table before export.
- Exported tree traversal is checked against scikit-learn on all 195 held-out rows. Browser traversal uses float32 feature comparisons to match scikit-learn.
- The chart retains minimum and maximum points within 240 index bins; this is a display summary, not the model input. The horizontal axis is sample index; unverified physical units or sampling rates are not invented.
- Labels are traceable to filenames, not independent physical certification. Related acquisition runs may occur in both partitions. The reported holdout accuracy is exploratory, not validated generalization to independent equipment.
- No upload support, live sensor integration, calibrated probability or maintenance recommendation is provided.

See [data provenance](../docs/DATA_PROVENANCE.md).
