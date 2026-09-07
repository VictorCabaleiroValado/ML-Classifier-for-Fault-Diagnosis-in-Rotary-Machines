"""Validated feature extraction and reproducible exploratory fault classification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.fft import fft
from scipy.stats import kurtosis, skew
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
SENSORS = ('Tachometer', 'Motor', 'B1_Z', 'B1_Y', 'B1_X', 'B2_Z', 'B2_Y', 'B2_X', 'Gearbox')
TARGETS = ('fault_detected', 'fault_category')
MODELS = ('linear', 'logistic', 'tree', 'svm', 'mlp')


def feature_columns(domain):
    if domain not in ('time', 'frequency'):
        raise ValueError('Domain must be time or frequency.')
    stats = ('max', 'mean', 'var', 'std', 'rms', 'kurtosis', 'skew', 'ptp') if domain == 'time' else (
        'max', 'mean', 'var', 'std', 'sp', 'kurtosis', 'skew')
    return [f'{stat}_{sensor}' for sensor in SENSORS for stat in stats]


def feature_path(rpm, domain, data_dir=ROOT):
    if rpm not in (25, 50, 75):
        raise ValueError('RPM must be 25, 50 or 75.')
    feature_columns(domain)
    prefix = 'time_domain_feature_extraction' if domain == 'time' else 'FF_feature_extraction'
    return Path(data_dir) / f'{prefix}_{rpm}.csv'


def validate_features(df, domain):
    columns = feature_columns(domain)
    missing = set(columns + list(TARGETS)) - set(df.columns)
    if missing:
        raise ValueError('Invalid feature table: missing columns ' + ', '.join(sorted(missing)))
    if df.empty:
        raise ValueError('Feature table contains no observations.')
    # Explicit whitelist excludes both targets, saved indexes, filenames and group IDs.
    try:
        X = df[columns].apply(pd.to_numeric, errors='raise').astype(float)
        y = df[list(TARGETS)].apply(pd.to_numeric, errors='raise').astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError('Features and targets must contain numeric values.') from exc
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError('Feature table contains missing or infinite values; regenerate or repair the source data.')
    if not y.isin([0, 1])['fault_detected'].all():
        raise ValueError('fault_detected must be 0 (healthy) or 1 (fault).')
    if not (y['fault_category'] >= 1).all() or not (y == np.floor(y)).all().all():
        raise ValueError('Targets must be integers and fault_category must be positive.')
    y = y.astype(int)
    if y.groupby('fault_category')['fault_detected'].nunique().max() > 1:
        raise ValueError('A fault category has inconsistent detection labels.')
    if any(y[target].nunique() < 2 for target in TARGETS):
        raise ValueError('Both prediction tasks require at least two classes.')
    return X, y


def load_features(path, domain):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Feature table unavailable: {path}. Regenerate it from original measurements and labels.')
    if path.read_bytes()[:100].startswith(b'429:'):
        raise ValueError(f'{path.name} contains a downloaded error message, not data. Restore the original measurements.')
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise ValueError(f'Cannot read feature CSV: {path.name}') from exc
    X, y = validate_features(df, domain)
    return df, X, y


def split_indices(y, seed=42, test_size=0.2, groups=None):
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between 0 and 1.')
    indices = np.arange(len(y))
    if groups is None:
        try:
            train, test = train_test_split(indices, test_size=test_size, random_state=seed,
                                           stratify=y['fault_category'])
        except ValueError as exc:
            raise ValueError('Not enough observations per category for the requested stratified split.') from exc
    else:
        groups = pd.Series(groups).reset_index(drop=True)
        if len(groups) != len(y) or groups.isna().any() or groups.nunique() < 2:
            raise ValueError('Group IDs must be present for every row, with at least two distinct groups.')
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(indices, groups=groups))
    for target in TARGETS:
        if set(y.iloc[train][target]) != set(y[target]) or set(y.iloc[test][target]) != set(y[target]):
            raise ValueError(f'The split does not represent every {target} class in both partitions. Supply more independent data or adjust the split.')
    return train, test


def build_model(name, seed=42):
    if name == 'tree':
        return DecisionTreeClassifier(random_state=seed)
    factories = {
        'linear': lambda: LinearRegression(),
        'logistic': lambda: LogisticRegression(max_iter=3000, random_state=seed),
        'svm': lambda: SVC(),
        'mlp': lambda: MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=3000, random_state=seed),
    }
    if name not in factories:
        raise ValueError(f'Unknown model: {name}')
    # Fit preprocessing only on the training partition, independently for each target.
    return make_pipeline(StandardScaler(), factories[name]())


def evaluate(path, domain, model='tree', seed=42, test_size=0.2, group_column=None, allow_unverified_data=False):
    df, X, y = load_features(path, domain)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    catalog = ROOT / 'data' / 'provenance.json'
    provenance = json.loads(catalog.read_text()).get(digest, {}) if catalog.exists() else {}
    if provenance.get('status') == 'unverified_legacy' and not allow_unverified_data:
        raise ValueError('Historical labels and feature semantics could not be verified against raw measurements. '
                         'Use the reconstructed 25 RPM tables, or --allow-unverified-data for legacy exploration only.')
    if group_column and (group_column not in df or group_column in feature_columns(domain) + list(TARGETS)):
        raise ValueError('Group column must be a separate metadata column in the feature table.')
    groups = df[group_column] if group_column else None
    train, test = split_indices(y, seed, test_size, groups)
    result = {
        'data_file': Path(path).name, 'data_sha256': digest,
        'data_provenance': provenance or {'status': 'user_supplied_unverified'},
        'domain': domain, 'model': model, 'seed': seed, 'test_size': test_size,
        'split': 'grouped' if groups is not None else 'stratified_by_category',
        'group_column': group_column, 'train_rows': len(train), 'test_rows': len(test),
        'feature_count': len(X.columns), 'train_indices': train.tolist(), 'test_indices': test.tolist(),
        'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__},
        'caveat': 'Exploratory holdout results; original run independence and label provenance require source verification.',
        'tasks': {},
    }
    for target in TARGETS:
        estimator = build_model(model, seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            estimator.fit(X.iloc[train], y.iloc[train][target])
            predicted = estimator.predict(X.iloc[test])
        observed = y.iloc[test][target]
        task = {'warnings': [str(w.message) for w in caught]}
        if model == 'linear':
            task.update(mse=mean_squared_error(observed, predicted), r2=r2_score(observed, predicted),
                        caveat='Regression on integer category codes is a legacy baseline, not classification accuracy.')
        else:
            labels = sorted(y[target].unique().tolist())
            task.update(report=classification_report(observed, predicted, labels=labels, output_dict=True, zero_division=0),
                        labels=labels, confusion_matrix=confusion_matrix(observed, predicted, labels=labels).tolist())
        result['tasks'][target] = task
    return result


def signal_features(samples, domain):
    """One observation per file; each input column is a named sensor."""
    feature_columns(domain)
    if list(samples.columns) != list(SENSORS) or len(samples) < 4:
        raise ValueError('Each measurement needs all nine sensor columns and at least four samples.')
    values = samples.apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Measurement contains missing or infinite sensor samples.')
    output = []
    for x in values.T:
        x = np.abs(fft(x)) if domain == 'frequency' else x
        constant = np.ptp(x) == 0
        output.extend([np.max(x), np.mean(x), np.var(x), np.std(x),
                       np.sqrt(np.mean(x ** 2)) if domain == 'time' else np.mean(x ** 2),
                       0.0 if constant else kurtosis(x), 0.0 if constant else skew(x)])
        if domain == 'time':
            output.append(np.ptp(x))
    if not np.isfinite(output).all():
        raise ValueError('Signal statistics overflowed or are undefined; inspect measurement scale.')
    return dict(zip(feature_columns(domain), output))


def extract(manifest_path, raw_dir, output, domain, start=1, end=None, legacy_skip_rows=None, overwrite=False):
    """Manifest rows, not filename order, determine labels and source identity."""
    manifest = pd.read_csv(manifest_path)
    required = {'filename', *TARGETS}
    if not required.issubset(manifest.columns) or manifest.empty:
        raise ValueError('Manifest requires filename, fault_detected and fault_category columns and at least one row.')
    if manifest['filename'].isna().any() or manifest['filename'].duplicated().any():
        raise ValueError('Manifest filenames must be nonempty and unique.')
    if start < 1 or start > len(manifest) or (end is not None and (end < start or end > len(manifest))):
        raise ValueError('Invalid manifest row range (1-based, inclusive).')
    if legacy_skip_rows is not None and legacy_skip_rows < 0:
        raise ValueError('legacy_skip_rows cannot be negative.')
    output, raw_dir = Path(output), Path(raw_dir).resolve()
    if output.exists() and not overwrite:
        raise ValueError(f'Output exists: {output}. Use --overwrite explicitly to replace it.')
    rows = []
    for _, record in manifest.iloc[start - 1:end].iterrows():
        source = (raw_dir / str(record['filename'])).resolve()
        if raw_dir not in source.parents or not source.is_file():
            raise ValueError(f'Measurement missing or outside raw directory: {record["filename"]}')
        if legacy_skip_rows is None:
            try:
                raw = pd.read_csv(source, low_memory=False)
            except UnicodeDecodeError:
                raw = pd.read_csv(source, encoding='ISO-8859-1', low_memory=False)
            if not set(SENSORS).issubset(raw.columns):
                raise ValueError(f'{source.name}: expected named sensor columns. For verified legacy alternating columns, use --legacy-skip-rows.')
            samples = raw[list(SENSORS)]
        else:
            if len(pd.read_csv(source, nrows=0).columns) != 18:
                raise ValueError('Legacy input requires exactly 18 alternating time/sensor columns.')
            # Read only the numeric sensor columns. Skip metadata after the header,
            # rather than materializing every timestamp and sample as strings.
            samples = pd.read_csv(source, usecols=list(range(1, 18, 2)),
                                  skiprows=range(1, legacy_skip_rows + 1), dtype=float)
            samples.columns = SENSORS
        row = signal_features(samples, domain)
        row.update({target: record[target] for target in TARGETS})
        row['source_file'] = str(record['filename'])
        if 'group_id' in manifest:
            if pd.isna(record['group_id']):
                raise ValueError('Manifest group_id must be present for each selected measurement.')
            row['group_id'] = record['group_id']
        rows.append(row)
    table = pd.DataFrame(rows)
    # Extraction subsets may legitimately contain a single class. Validate numeric
    # features and labels here; the classifier enforces multi-class sufficiency.
    if not np.isfinite(table[feature_columns(domain)].to_numpy()).all():
        raise ValueError('Extracted features contain nonfinite values.')
    labels = table[list(TARGETS)].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(labels.to_numpy()).all() or not (labels == np.floor(labels)).all().all() or not labels['fault_detected'].isin([0, 1]).all() or not (labels['fault_category'] >= 1).all():
        raise ValueError('Manifest labels must be integer categories >= 1 and binary detection flags.')
    if labels.groupby('fault_category')['fault_detected'].nunique().max() > 1:
        raise ValueError('Manifest assigns inconsistent detection labels to a category.')
    table[list(TARGETS)] = labels.astype(int)
    if output.resolve() == Path(manifest_path).resolve() or output.resolve() in [(raw_dir / str(f)).resolve() for f in manifest.filename]:
        raise ValueError('Output must not replace a manifest or source measurement.')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement without explicit opt-in.
    with output.open('w' if overwrite else 'x') as handle:
        table.to_csv(handle, index=False)
    return table


def choose(prompt, choices):
    while True:
        print(prompt)
        for key, value in choices.items():
            print(f'  {key}: {value}')
        answer = input('> ').strip()
        if answer in choices:
            return choices[answer]
        print('Invalid selection. Please choose one of the displayed options.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('evaluate', help='Evaluate a validated feature table')
    run.add_argument('--rpm', type=int, choices=(25, 50, 75), required=True)
    run.add_argument('--domain', choices=('time', 'frequency'), required=True)
    run.add_argument('--model', choices=MODELS, default='tree')
    run.add_argument('--data-dir', type=Path, default=ROOT)
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--test-size', type=float, default=0.2)
    run.add_argument('--group-column')
    run.add_argument('--allow-unverified-data', action='store_true', help='Explicitly opt into legacy tables whose original measurements are unavailable')
    run.add_argument('--output', type=Path)
    sub.add_parser('menu', help='Interactive classifier')
    ex = sub.add_parser('extract', help='Extract features from explicitly labeled measurements')
    ex.add_argument('--rpm', type=int, choices=(25, 50, 75), required=True)
    ex.add_argument('--domain', choices=('time', 'frequency'), required=True)
    ex.add_argument('--raw-dir', type=Path, required=True)
    ex.add_argument('--manifest', type=Path, required=True)
    ex.add_argument('--output', type=Path)
    ex.add_argument('--start', type=int, default=1)
    ex.add_argument('--end', type=int)
    ex.add_argument('--legacy-skip-rows', type=int)
    ex.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'extract':
            output = args.output or feature_path(args.rpm, args.domain)
            table = extract(args.manifest, args.raw_dir, output, args.domain, args.start, args.end,
                            args.legacy_skip_rows, args.overwrite)
            print(f'Saved {len(table)} observations to {output}')
        elif args.command == 'menu':
            while True:
                rpm = choose('Operating speed', {'1': 25, '2': 50, '3': 75})
                domain = choose('Feature domain', {'1': 'frequency', '2': 'time'})
                model = choose('Model (linear is a regression baseline)', dict(zip(('1', '2', '3', '4', '5'), MODELS)))
                try:
                    print(json.dumps(evaluate(feature_path(rpm, domain), domain, model), indent=2, allow_nan=False))
                except (ValueError, OSError) as exc:
                    print(f'Cannot evaluate: {exc}', file=sys.stderr)
                if choose('Run another model?', {'1': True, '0': False}) is False:
                    break
        else:
            result = evaluate(feature_path(args.rpm, args.domain, args.data_dir), args.domain,
                              args.model, args.seed, args.test_size, args.group_column, args.allow_unverified_data)
            payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open('x') as handle:
                    handle.write(payload)
            else:
                print(payload, end='')
        return 0
    except (ValueError, OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    except (EOFError, KeyboardInterrupt):
        print('\nCancelled.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
