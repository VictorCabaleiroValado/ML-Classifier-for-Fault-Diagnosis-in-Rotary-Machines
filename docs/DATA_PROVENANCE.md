# Data provenance and maintenance notes

## Reconstructed 25 RPM tables

Both domains were recomputed from the owner's 975 original measurement CSVs using the shared extractor. The instrument files contain 18 alternating time/sensor columns; the three metadata rows following the CSV header are excluded. All numeric sensor samples are retained.

Labels come from the fault names in the source filenames and are recorded explicitly in `data/manifests/25_rpm.csv`. Categories are listed in `25_rpm_categories.json`. The two source names `Bearing (2) Fault (inner)` (22 files) and `Bearing (2) Fault (inner race)` (3 files) are treated as aliases of the same inner-race condition, yielding 39 categories with 25 observations each. This naming normalization is explicit and should be checked against experimental records if a different interpretation is intended. `No Fault` is the healthy class. No positional label ranges are used.

Original spelling in source filenames is retained. The manifest does not establish independence between acquisition runs: trial filenames alone do not prove independent acquisitions. These labels are traceable to filenames, not independently certified physical diagnoses.

The old frequency extractor placed skewness values under kurtosis columns and vice versa. A raw 25 RPM measurement matched a legacy frequency row to numerical precision only after exchanging those values. Its historical category also differed from the filename-derived category, demonstrating the danger of positional labeling. The rebuilt tables correct both issues. The old time table also differed slightly from features recomputed using only numeric sample rows; it was regenerated rather than selectively patched.

## Historical 50 and 75 RPM tables

The owner supplied a valid local `time_domain_feature_extraction_75.csv` containing 975 observations, replacing the error message committed in the original repository. The other supplied local feature tables matched their existing GitHub versions byte for byte.

The original 50 and 75 RPM measurements are unavailable. Their tables pass structural/numeric checks, but label correctness and frequency-column semantics are unverified. They are preserved without invented corrections. Their SHA-256 hashes are marked `unverified_legacy` in `data/provenance.json`; evaluation requires explicit `--allow-unverified-data`. Restoring a readable table is not verification of its scientific validity.

## What changed in the software

- Eliminated target leakage in time-domain models and removed duplicated model code.
- Added training-only scaling, repeatable splits and independently fitted target models.
- Fixed cross-speed extractor state accumulation and output filename mismatches.
- Replaced filename-order labeling with explicit manifests; subsets retain their labels.
- Added validation and useful errors for missing/corrupt data, insufficient class coverage, nonnumeric samples and accidental overwrites.
- Captured input hashes, provenance, split indices, software versions and warnings in evaluation reports.

Root-level historical result logs and the original PDF were not recomputed or rewritten. They are not evidence for the corrected implementation. New evaluation files are exploratory software validation, not a deployment or independent scientific validation.
