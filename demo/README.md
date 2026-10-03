# Interactive fault diagnosis demo

[Explore the interactive demo](https://victorcabaleirovalado.github.io/demo/)

A beginner-friendly engineering demo using real 25 RPM measurements, styled to match the portfolio: Geist typography, dark surfaces and light-blue accents. The main view guides visitors through three starter examples, a simplified machine diagram, a motor-signal comparison and the model result. The diagram highlights components from the recorded source description, never from the prediction. The complete 39-example selector is optional. Everyday washing-machine and bicycle analogies explain the idea without claiming those devices supplied the data. Features, the complete decision path, evaluation and provenance are available in optional disclosure panels. Plain-language guidance explains what sensors measure, how to read the plot, what bearings and shafts do, and how a correct or incorrect prediction is checked against the source label.

## Interactive workflow

1. Choose any of 39 held-out measurements. Suggested cases include No Fault, a bearing fault and a model disagreement; shaft conditions remain in the selector.
2. Compare the motor overview with one No Fault reference on a shared vertical scale. Inspect the full recording, first/last quarter or middle half.
3. Run the tree on the full feature vector. Compare a readable predicted condition with the readable source condition. The original dataset labels remain available under the data details and in JSON exports.
4. Explore all eight statistics for each of nine channels. Features used by this prediction are highlighted after analysis.
5. Follow every threshold check. Download the analysis as JSON or all 72 selected/reference features as CSV.

The interface includes responsive layouts, keyboard focus, reduced-motion support, descriptive chart text, data validation, retry state and a no-JavaScript explanation. It reuses the portfolio's same-origin Geist font, with an Arial fallback when previewed separately. It uses no analytics library or third-party JavaScript. No data uploads are performed.

## Reproduce the export

From the repository root, in the Python environment described in the main README:

```sh
python demo/export_demo.py "/path/to/Fault data split 25" /tmp/vibration-demo
cp demo/index.html demo/style.css demo/demo.js /tmp/vibration-demo/
python -m http.server 8000 --directory /tmp/vibration-demo
```

Open http://localhost:8000. Original measurements are required for regeneration and are not bundled here. The published JSON includes the tree, 39 derived signal overviews, full feature vectors, filenames and hashes, split indices, feature-table hash and package versions. The 2026 interface redesign preserves that export unchanged.

## Evaluation and limits

- Uses only the reconstructed 25 RPM time-domain table: 72 features, 39 categories.
- A single decision tree is fitted on 780 rows; 195 are held out, stratified by category, seed 42. Training excludes the demo examples.
- One example per category is selected by the lowest held-out row index, without conditioning selection on the prediction.
- Each example's full-resolution extracted features must numerically match the reconstructed table before export.
- Exported tree traversal is checked against scikit-learn on all 195 held-out rows. Browser traversal uses float32 comparisons.
- Full-holdout category accuracy is 95.9%; the separate 39-example display subset has 36 matches and 3 disagreements. Neither is a confidence score for an individual prediction.
- The chart retains min/max points within 240 index bins. Lines connect those retained points, not a full waveform. Statistics and model inputs come from full measurements. The horizontal axis uses sample index; sampling rate and physical units are not inferred.
- The reference is one No Fault measurement, not a healthy population. A larger feature value does not automatically imply a worse fault.
- Source labels derive from filenames, not independent physical certification. A disagreement is a model error relative to the label, not proof that the label is wrong.
- Related acquisitions may occur in both partitions. Row-level holdout evaluation does not establish performance on independent machines or operating conditions.
- No uploads, live sensors, calibrated probability, severity, remaining-life estimate or maintenance recommendation.
- The Python project supports frequency-domain extraction, but the browser does not compute FFT from reduced display points.

See [data provenance](../docs/DATA_PROVENANCE.md).

## Verify browser logic

```sh
node demo/check_browser_model.cjs /path/to/exported/data.json
```

Checks cover all 39 expected predictions, split separation, 156 signal ranges, feature names, float32 threshold boundaries and invalid/training-contaminated examples. UI checks additionally cover case selection, stale-result reset, sensor selection, reference toggling and JSON/CSV downloads. The portfolio stores identical UI files under `public/demo/` and published `docs/demo/`.
