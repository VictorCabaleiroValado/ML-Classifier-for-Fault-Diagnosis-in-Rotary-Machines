# Real-measurement browser demo

117 real held-out recordings: 39 each at 25, 50 and 75 RPM. Each speed uses a separate time-domain decision tree trained on 780 recordings and tested on 195. Labels come from original source filenames. No synthetic data enter the active UI, training or holdout evaluation.

```bash
python -m http.server 8000 --directory demo
python demo/export_demo.py "/path/to/Fault data split 25" "/path/to/Original Archives" demo
node demo/check_browser_model.cjs demo/data.json
node demo/check_browser_model.cjs demo/real-50.json
node demo/check_browser_model.cjs demo/real-75.json
```

The archive directory must contain the checksum-verified `Fault data split 50.zip` and `Fault data split 75.zip`. Reconstruct their tables first using `recover_measurements.py`; install `requirements-lock.txt` for reproducibility.

The exporter verifies full-signal features against the corresponding table, exported tree predictions against scikit-learn and selected source hashes against the recovered manifests. Category selection uses the smallest held-out index, not the most favorable example. JavaScript tests cover predictions, finite values, same-speed provenance, no training-row examples, float32 threshold boundaries, cycle rejection and chart ranges.

A feature CSV is a 72-row comparison with a no-fault reference, not a waveform. The original source filename stays visible; full waveform files are in the linked Figshare archives. Analysis JSON records RPM, model training speed, source hash, holdout score and decision path. Original data: David Jensen (2023), DOI 10.6084/m9.figshare.22693120.v1, CC BY 4.0. Display overviews and features are derived from those signals.

`archive/` retains the earlier synthetic experiments solely for reproducibility. They are not loaded by `index.html`. Recorded labels and row holdout do not establish independent acquisition/machine generalization. See [data provenance](../docs/DATA_PROVENANCE.md).
