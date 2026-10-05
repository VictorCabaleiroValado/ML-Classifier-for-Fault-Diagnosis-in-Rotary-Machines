"""Reproduce the real demo and isolated, explicitly synthetic speed scenarios.

Synthetic descendants are generated only AFTER the real train/test split and
never enter training or the real holdout metric. See docs/SYNTHETIC_SCENARIOS.md.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fault_diagnosis as fd

SCENARIO_SAMPLES = 4096


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def filename(label, rpm, row, kind, suffix='csv'):
    condition = re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')
    return f'{kind}_{rpm}rpm_{condition}_source{row:04d}.{suffix}'


def synthesize(samples, rpm, source_row, seed=42):
    """A deterministic signal perturbation, NOT a physical RPM simulator."""
    if rpm not in (50, 75):
        raise ValueError('Synthetic scenario RPM must be 50 or 75.')
    factor = rpm // 25
    values = samples.to_numpy(dtype=float)
    if list(samples.columns) != list(fd.SENSORS) or len(values) < SCENARIO_SAMPLES * factor or not np.isfinite(values).all():
        raise ValueError('Scenario requires finite source samples on all nine channels.')
    # Anti-alias before decimation. The output index is illustrative, not time.
    signal = resample_poly(values[:SCENARIO_SAMPLES * factor], 1, factor, axis=0)
    mean = signal.mean(axis=0)
    gain = 1 + .15 * (factor - 1)
    signal = mean + gain * (signal - mean)
    rng = np.random.default_rng(np.random.SeedSequence([seed, rpm, int(source_row)]))
    signal += rng.normal(size=signal.shape) * signal.std(axis=0) * .01
    return pd.DataFrame(signal, columns=fd.SENSORS)


def overview(samples):
    signal = samples.Motor.to_numpy()
    envelope = []
    for block in np.array_split(np.arange(len(signal)), min(240, len(signal))):
        ids = sorted({int(block[np.argmin(signal[block])]), int(block[np.argmax(signal[block])])})
        envelope.extend([[j, round(float(signal[j]), 6)] for j in ids])
    return envelope


def export(raw_dir, out):
    raw_dir, out = Path(raw_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'csv').mkdir(exist_ok=True)
    path = fd.feature_path(25, 'time')
    df, X, y = fd.load_features(path, 'time')
    train, test = fd.split_indices(y)
    model = fd.build_model('tree')
    model.fit(X.iloc[train], y.iloc[train].fault_category)
    t = model.tree_
    tree = {'left': t.children_left.tolist(), 'right': t.children_right.tolist(),
            'feature': t.feature.tolist(), 'threshold': t.threshold.tolist(),
            'category': [int(model.classes_[np.argmax(v)]) for v in t.value]}

    def predict(x):
        n = 0
        while tree['left'][n] != -1:
            n = tree['left'][n] if np.float32(x[tree['feature'][n]]) <= tree['threshold'][n] else tree['right'][n]
        return tree['category'][n]

    pred = model.predict(X.iloc[test])
    assert [predict(x) for x in X.iloc[test].to_numpy()] == pred.tolist()
    categories = json.loads((fd.ROOT / 'data/manifests/25_rpm_categories.json').read_text())
    examples = {rpm: [] for rpm in (25, 50, 75)}
    manifests = []
    for c in sorted(y.fault_category.unique()):
        i = min(int(i) for i in test if y.iloc[i].fault_category == c)
        source = raw_dir / df.iloc[i].source_file
        samples = pd.read_csv(source, usecols=list(range(1, 18, 2)), skiprows=range(1, 4), dtype=float)
        samples.columns = fd.SENSORS
        features = fd.signal_features(samples, 'time')
        np.testing.assert_allclose(list(features.values()), X.iloc[i].values, rtol=1e-8, atol=1e-10)
        parent_hash = sha256(source)
        for rpm in (25, 50, 75):
            kind = 'real' if rpm == 25 else 'synthetic'
            label = categories[str(c)]
            name = filename(label, rpm, i, kind)
            if rpm == 25:
                signal, vector, digest = samples, X.iloc[i].tolist(), parent_hash
            else:
                csv_path = out / 'csv' / name
                synthesize(samples, rpm, i).to_csv(csv_path, index=False, float_format='%.10g')
                # Features always describe the EXACT downloadable CSV, not pre-rounding data.
                signal = pd.read_csv(csv_path)
                vector = list(fd.signal_features(signal, 'time').values())
                digest = sha256(csv_path)
            prediction = predict(vector)
            assert prediction == int(model.predict(pd.DataFrame([vector], columns=X.columns))[0])
            example = {'id': i, 'label': label, 'category': int(c), 'source': source.name,
                       'sha256': digest, 'sourceSha256': parent_hash, 'features': vector,
                       'signal': overview(signal), 'samples': len(signal), 'expectedPrediction': prediction,
                       'rpm': rpm, 'kind': kind, 'downloadName': name,
                       'labelBasis': 'source_filename' if rpm == 25 else 'inherited_from_25rpm_parent',
                       'parentPrediction': predict(X.iloc[i].values), 'parentMotorRms': float(X.iloc[i]['rms_Motor']),
                       'parentRow': i, 'rawCsv': None if rpm == 25 else 'csv/' + name}
            examples[rpm].append(example)
            manifests.append({k: v for k, v in example.items() if k not in ('features', 'signal', 'expectedPrediction')})
    for rpm in (25, 50, 75):
        payload = {'schemaVersion': 2, 'rpm': rpm, 'kind': 'real' if rpm == 25 else 'synthetic',
                   'tree': tree, 'categories': categories, 'examples': examples[rpm], 'features': list(X.columns),
                   'trainCount': len(train), 'testCount': len(test),
                   'accuracy': float(np.mean(pred == y.iloc[test].fault_category)) if rpm == 25 else None,
                   'seed': 42, 'trainIndices': train.tolist(), 'testIndices': test.tolist(),
                   'featureTableSha256': sha256(path), 'modelTrainingRpm': 25,
                   'versions': {'sklearn': fd.sklearn.__version__, 'numpy': np.__version__},
                   'scenario': None if rpm == 25 else {'method': 'resample_poly_then_gain_and_noise_v1',
                       'parentRpm': 25, 'decimationFactor': rpm // 25, 'outputSamples': SCENARIO_SAMPLES,
                       'gain': 1 + .15 * (rpm // 25 - 1), 'noiseStdFraction': .01,
                       'claim': 'Illustrative stress test, not a measured or physically validated RPM condition.'}}
        target = 'data.json' if rpm == 25 else f'synthetic-{rpm}.json'
        (out / target).write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
    (out / 'manifest.json').write_text(json.dumps({'schemaVersion': 1, 'examples': manifests}, indent=2) + '\n')
    print(f'Exported 39 real examples and 78 synthetic scenarios; real holdout {np.mean(pred == y.iloc[test].fault_category):.3%}; Python/tree parity verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.raw_dir, args.output)
