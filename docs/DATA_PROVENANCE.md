# Data provenance and reconstruction

## Authoritative source

David Jensen (2023), **Single and Double Fault Scenarios for a Rotary Machine**, Figshare, DOI [10.6084/m9.figshare.22693120.v1](https://doi.org/10.6084/m9.figshare.22693120.v1), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset describes 38 fault conditions plus one healthy condition at 25, 50 and 75 RPM. Recovery on 5 October 2026 resolved the previous missing-originals limitation for 50/75 RPM.

`data/publisher.json` pins original download URLs, byte counts and publisher MD5 values. `recover_measurements.py` verifies those values, records archive SHA-256, validates each ZIP member's CRC and calculates SHA-256 for each CSV. It reads directly from the ZIP, without duplicating 36 GB of decompressed data. Original archives are retained in the owner's local project under `Main/Data/Original Archives` and distributed publicly by Figshare, not GitHub Pages.

## What was rebuilt

Both time and frequency tables at 50/75 RPM were recomputed from all 975 recordings per speed. The existing 25 RPM tables had already been recomputed from 975 local original recordings. The maintained project therefore contains features from 2,925 real recordings, 39 conditions per speed and 25 files per condition.

Each source has 18 alternating time/sensor columns, with interval, channel-name and unit metadata after the header. Recovery verifies nine named channels in the correct order, the 1/6400-second interval, tachometer units V and other channels g, then retains every numeric sensor sample. Time features use eight statistics per channel (72 total); frequency features use seven (63 total). The latter use magnitudes of the unnormalized full FFT, with correctly named skewness/kurtosis columns. These summaries are not a calibrated power spectral density.

Manifests record original filenames, archive members, explicit labels, trial numbers, sample counts, source hashes and header acquisition timestamps for the recovered speeds. Category names use the established shared 25 RPM map. Naming aliases are explicit: `Bearing (1/2) Ball`, `Inner`, `Outer` and `Combination` at 50/75 RPM map to the corresponding `Fault (...)` names at 25 RPM; `inner`/`inner race`, `Combintion`/`Combination`, case and spaces after an opening parenthesis are normalized; other unknown conditions fail rather than silently receiving a label. Original filename spelling is preserved. No position-based label ranges are used.

## Recording-length finding

At 50 RPM, `Bearing (2) Ball 50 Trial 17.csv` contains **34,044 samples** (5.319375 seconds); the other 974 recordings have 64,000 samples. All 975 recordings at 75 RPM have 64,000 samples. This shorter source is preserved without padding, resampling or invented values, and is also selected naturally among the held-out demo examples. Its sample count and duration remain visible. The source does not establish why the recording is shorter.

Features describe each complete available recording. Duration affects the statistical precision of time summaries and can affect unnormalized FFT summaries; the frequency results therefore also carry this length-comparability limitation. A future equal-window or normalized-spectral comparison should be a separately specified evaluation, not a silent change to the source.

The recovered archives contain 975 distinct file hashes and 975 distinct header timestamps per speed. Distinct timestamps do not establish independent experimental setups or machines.

## What the labels and evaluations establish

Labels are traceable to the publisher's filenames, not independently certified physical diagnoses. A disagreement is an error relative to the recorded label; it does not justify changing that label to match a model. Numeric category IDs describe nominal classes, not severity.

Evaluation uses a reproducible 80/20 split stratified by category, seed 42: 780 training and 195 test files **per speed**. Models are fitted separately for each speed and each target. Predictors exclude labels, filenames, timestamps and all metadata. Scalers are fitted only on training data.

**These are exploratory row-level results.** Trial numbers do not prove independent acquisition runs. Header timestamps are preserved in manifests and feature-table metadata as source evidence, and are excluded from predictors. They do not certify independent machines or experiments; related recordings may cross the split. Accuracy is not a calibrated confidence score, industrial validation, a remaining-life estimate or maintenance advice. Authentic experimental groups and an external test are still needed.

## Legacy evidence and synthetic history

Historical feature hashes remain `unverified_legacy` in the provenance catalog so old copies cannot silently acquire reconstructed status. The active root-level 50/75 RPM tables have new hashes and `reconstructed_from_raw` records. Previous versions remain available in Git history; original academic PDFs and root-level `*_RPM_Results.txt` logs are historical, not current evaluation evidence.

The previous 78 synthetic scenarios are retained under `demo/archive/` with their original inherited labels. They are retired from the active UI. They were useful software sensitivity tests, but their 5/39 and 6/39 label agreements never measured real 50/75 RPM diagnostic accuracy. See [historical method](SYNTHETIC_SCENARIOS.md).

## Reproducible browser model

The active demo exports one separate tree and 39 real held-out examples at each speed. The lowest held-out row per category is selected before considering correctness. Full-recording features are checked against the raw CSV; browser display overviews keep min/max points from 240 bins and are not used for feature extraction. Python and JavaScript compare tree thresholds using float32 input semantics. Feature CSVs, analysis JSON, original filenames, hashes and publisher links distinguish derived exports from full raw measurements.
