"""Export real held-out measurements and separate models for 25, 50 and 75 RPM.

The retained synthesize() helper reproduces historical stress tests only.
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


def export(raw_dir, archives_dir, out):
    import zipfile
    from recover_measurements import read_measurement
    raw_dir, archives_dir, out = Path(raw_dir), Path(archives_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    categories = json.loads((fd.ROOT / 'data/manifests/25_rpm_categories.json').read_text())
    publisher = json.loads((fd.ROOT / 'data/publisher.json').read_text())
    manifests = []
    for rpm in (25, 50, 75):
        path = fd.feature_path(rpm, 'time')
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
        examples = []
        archive = zipfile.ZipFile(archives_dir / publisher['files'][str(rpm)]['name']) if rpm != 25 else None
        try:
            manifest = pd.read_csv(fd.ROOT / f'data/manifests/{rpm}_rpm.csv')
            for c in sorted(y.fault_category.unique()):
                i = min(int(i) for i in test if y.iloc[i].fault_category == c)
                source = df.iloc[i].source_file
                blob = (raw_dir / source).read_bytes() if archive is None else archive.read(manifest.iloc[i].archive_member)
                samples = read_measurement(blob)
                vector = list(fd.signal_features(samples, 'time').values())
                np.testing.assert_allclose(vector, X.iloc[i].values, rtol=1e-8, atol=1e-10)
                digest = hashlib.sha256(blob).hexdigest()
                if rpm != 25:
                    assert digest == manifest.iloc[i].sha256
                prediction = predict(vector)
                assert prediction == int(model.predict(pd.DataFrame([vector], columns=X.columns))[0])
                example = {'id': i, 'label': categories[str(c)], 'category': int(c), 'source': source,
                    'sha256': digest, 'sourceSha256': digest, 'features': vector,
                    'signal': overview(samples), 'samples': len(samples), 'expectedPrediction': prediction,
                    'rpm': rpm, 'kind': 'real', 'downloadName': filename(categories[str(c)], rpm, i, 'real'),
                    'labelBasis': 'source_filename', 'rawCsv': None}
                examples.append(example)
                manifests.append({k: v for k, v in example.items() if k not in ('features', 'signal', 'expectedPrediction')})
        finally:
            if archive is not None:
                archive.close()
        payload = {'schemaVersion': 3, 'rpm': rpm, 'kind': 'real', 'tree': tree,
            'categories': categories, 'examples': examples, 'features': list(X.columns),
            'trainCount': len(train), 'testCount': len(test), 'accuracy': float(np.mean(pred == y.iloc[test].fault_category)),
            'seed': 42, 'trainIndices': train.tolist(), 'testIndices': test.tolist(),
            'featureTableSha256': sha256(path), 'modelTrainingRpm': rpm,
            'sourceDataset': publisher['url'], 'doi': publisher['doi'], 'license': publisher['license'],
            'samplingRateHz': 6400, 'sensorUnits': {'Tachometer': 'V', **{s: 'g' for s in fd.SENSORS[1:]}},
            'archiveUrl': publisher['files'][str(rpm)]['url'],
            'versions': {'sklearn': fd.sklearn.__version__, 'numpy': np.__version__}}
        target = 'data.json' if rpm == 25 else f'real-{rpm}.json'
        (out / target).write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
        print(f'{rpm} RPM: 39 REAL examples, category holdout {payload["accuracy"]:.3%}; full holdout Python/tree parity verified.')
    (out / 'manifest.json').write_text(json.dumps({'schemaVersion': 2, 'examples': manifests}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_dir', type=Path)
    parser.add_argument('archives_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    export(args.raw_dir, args.archives_dir, args.output)
