import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import fault_diagnosis as fd


@pytest.fixture
def table():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(80, 72)), columns=fd.feature_columns('time'))
    df['fault_category'] = np.tile([1, 2], 40)
    df['fault_detected'] = df['fault_category'] - 1
    df['group_id'] = np.repeat(np.arange(20), 4)
    df['Unnamed: 0'] = np.arange(80)
    return df


def test_targets_and_metadata_never_enter_predictors(table):
    X, _ = fd.validate_features(table, 'time')
    assert len(X.columns) == 72
    assert not set(fd.TARGETS + ('group_id', 'Unnamed: 0')) & set(X.columns)


@pytest.mark.parametrize('bad', [np.nan, np.inf, 'not numeric'])
def test_bad_features_rejected(table, bad):
    table = table.astype({table.columns[0]: object})
    table.iloc[0, 0] = bad
    with pytest.raises(ValueError):
        fd.validate_features(table, 'time')


def test_inconsistent_labels_rejected(table):
    table.loc[0, 'fault_detected'] = 1
    with pytest.raises(ValueError, match='inconsistent'):
        fd.validate_features(table, 'time')


def test_group_split_prevents_shared_runs(table):
    _, y = fd.validate_features(table, 'time')
    train, test = fd.split_indices(y, groups=table.group_id)
    assert not set(table.iloc[train].group_id) & set(table.iloc[test].group_id)
    assert set(train).isdisjoint(test)


def test_stratified_split_is_repeatable(table):
    _, y = fd.validate_features(table, 'time')
    a = fd.split_indices(y)
    b = fd.split_indices(y)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert y.iloc[a[1]].fault_category.value_counts().tolist() == [8, 8]


def test_scaler_uses_training_only(table):
    X, y = fd.validate_features(table, 'time')
    train, test = fd.split_indices(y)
    X.iloc[test] = 1000000
    model = fd.build_model('logistic').fit(X.iloc[train], y.iloc[train].fault_detected)
    np.testing.assert_allclose(model[0].mean_, X.iloc[train].mean())
    assert not np.allclose(model[0].mean_, X.mean())


@pytest.mark.parametrize('model', fd.MODELS)
def test_all_models_return_both_tasks(tmp_path, table, model):
    path = tmp_path / 'features.csv'
    table.to_csv(path, index=False)
    result = fd.evaluate(path, 'time', model)
    assert set(result['tasks']) == set(fd.TARGETS)
    assert result['feature_count'] == 72
    json.dumps(result, allow_nan=False)
    if model == 'linear':
        assert 'mse' in result['tasks']['fault_detected']
    else:
        assert len(result['tasks']['fault_detected']['confusion_matrix']) == 2


def test_tree_evaluation_repeatable(tmp_path, table):
    path = tmp_path / 'table.csv'
    table.to_csv(path, index=False)
    assert fd.evaluate(path, 'time') == fd.evaluate(path, 'time')


@pytest.mark.parametrize('domain', ['time', 'frequency'])
def test_constant_signals_have_finite_features(domain):
    samples = pd.DataFrame(np.ones((10, 9)), columns=fd.SENSORS)
    result = fd.signal_features(samples, domain)
    assert len(result) == (72 if domain == 'time' else 63)
    assert np.isfinite(list(result.values())).all()
    if domain == 'time':
        assert result['rms_Motor'] == 1
        assert result['std_Motor'] == 0


def test_fft_features_known_signal():
    samples = pd.DataFrame(np.tile([1., 0., -1., 0.], (9, 1)).T, columns=fd.SENSORS)
    result = fd.signal_features(samples, 'frequency')
    assert result['max_Motor'] == pytest.approx(2)
    assert result['sp_Motor'] == pytest.approx(2)


def make_raw(tmp_path):
    rows = []
    for i, name in enumerate(['z.csv', 'a.csv', 'middle.csv']):
        pd.DataFrame(np.full((8, 9), i + 1), columns=fd.SENSORS).to_csv(tmp_path / name, index=False)
        rows.append({'filename': name, 'fault_category': i + 1, 'fault_detected': int(i != 1), 'group_id': f'run-{i}'})
    manifest = tmp_path / 'labels.csv'
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def test_subset_preserves_explicit_labels_and_source_order(tmp_path):
    manifest = make_raw(tmp_path)
    result = fd.extract(manifest, tmp_path, tmp_path/'features.csv', 'time', start=2, end=2)
    assert result.iloc[0].fault_category == 2
    assert result.iloc[0].fault_detected == 0
    assert result.iloc[0].source_file == 'a.csv'
    assert result.iloc[0].group_id == 'run-1'
    assert result.iloc[0].mean_Motor == 2


def test_extractions_do_not_accumulate_or_overwrite(tmp_path):
    manifest = make_raw(tmp_path)
    output = tmp_path/'features.csv'
    first = fd.extract(manifest, tmp_path, output, 'time')
    second = fd.extract(manifest, tmp_path, tmp_path/'second.csv', 'time')
    pd.testing.assert_frame_equal(first, second)
    before = output.read_bytes()
    with pytest.raises(ValueError, match='exists'):
        fd.extract(manifest, tmp_path, output, 'time')
    assert output.read_bytes() == before


def test_missing_raw_file_does_not_write_partial_output(tmp_path):
    manifest = make_raw(tmp_path)
    (tmp_path/'a.csv').unlink()
    output = tmp_path/'features.csv'
    with pytest.raises(ValueError, match='missing'):
        fd.extract(manifest, tmp_path, output, 'time')
    assert not output.exists()


def test_corrupt_download_rejected(tmp_path):
    path = tmp_path/'bad.csv'
    path.write_text('429: Too Many Requests\n')
    with pytest.raises(ValueError, match='error message'):
        fd.load_features(path, 'time')


@pytest.mark.parametrize('domain', ['time', 'frequency'])
@pytest.mark.parametrize('rpm', [25, 50, 75])
def test_real_tables_are_valid_and_tree_runs(domain, rpm):
    result = fd.evaluate(fd.feature_path(rpm, domain), domain, allow_unverified_data=True)
    assert result['train_rows'] == 780
    assert result['test_rows'] == 195
    assert len(result['tasks']['fault_category']['labels']) == 39


def test_cli_from_another_working_directory(tmp_path):
    run = subprocess.run([sys.executable, str(fd.ROOT/'fault_diagnosis.py'), 'evaluate', '--rpm', '25',
                          '--domain', 'time'], cwd=tmp_path, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)['feature_count'] == 72


def test_menu_recovers_from_invalid_input(tmp_path):
    run = subprocess.run([sys.executable, str(fd.ROOT/'ML_Classifer_Code_with_Menu.py')], cwd=tmp_path,
                         input='invalid\n1\n1\n3\n0\n', text=True, capture_output=True)
    assert run.returncode == 0, run.stderr
    assert 'Invalid selection' in run.stdout
    assert 'confusion_matrix' in run.stdout


@pytest.mark.parametrize('script', ['Time_Domain_FE.py', 'Time_Domain_FE_with_Menu.py',
                                  'Frequency_Domain_FE.py', 'Frequency_Domain_FE_with_Menu.py'])
def test_extractor_entrypoints_help(script):
    run = subprocess.run([sys.executable, str(fd.ROOT/script), '--help'], text=True, capture_output=True)
    assert run.returncode == 0
    assert '--manifest' in run.stdout


def test_legacy_tables_require_explicit_opt_in():
    with pytest.raises(ValueError, match='allow-unverified-data'):
        fd.evaluate(fd.feature_path(75, 'time'), 'time')


def test_legacy_layout_preserves_named_sensor_order(tmp_path):
    raw = pd.DataFrame(np.tile(np.arange(18), (8, 1)), columns=[f'column{i}' for i in range(18)])
    raw.to_csv(tmp_path/'raw.csv', index=False)
    manifest = tmp_path/'manifest.csv'
    pd.DataFrame([{'filename':'raw.csv', 'fault_detected':0, 'fault_category':1}]).to_csv(manifest,index=False)
    table = fd.extract(manifest,tmp_path,tmp_path/'output.csv','time',legacy_skip_rows=0)
    assert table.iloc[0].mean_Tachometer == 1
    assert table.iloc[0].mean_Motor == 3
    assert table.iloc[0].mean_Gearbox == 17


def test_manifest_and_raw_data_cannot_be_overwritten(tmp_path):
    manifest = make_raw(tmp_path)
    with pytest.raises(ValueError, match='must not replace'):
        fd.extract(manifest,tmp_path,manifest,'time',overwrite=True)


def test_outputs_use_classifier_names():
    assert fd.feature_path(75,'frequency').name == 'FF_feature_extraction_75.csv'
    assert fd.feature_path(75,'time').name == 'time_domain_feature_extraction_75.csv'


def test_rebuilt_domains_match_manifest_labels():
    manifest = pd.read_csv(fd.ROOT/'data/manifests/25_rpm.csv')
    for domain in ('time','frequency'):
        df, _, y = fd.load_features(fd.feature_path(25,domain),domain)
        assert df.source_file.tolist() == manifest.filename.tolist()
        pd.testing.assert_frame_equal(y,manifest[list(fd.TARGETS)])
        assert y.fault_category.value_counts().eq(25).all()
        assert df.loc[df.source_file.str.startswith('No Fault'), 'fault_detected'].eq(0).all()


def test_provenance_matches_current_bundled_files():
    import hashlib
    catalog = json.loads((fd.ROOT/'data/provenance.json').read_text())
    for domain in ('time','frequency'):
        for rpm in (25,50,75):
            path=fd.feature_path(rpm,domain)
            record=catalog[hashlib.sha256(path.read_bytes()).hexdigest()]
            assert record['status'] == ('reconstructed_from_raw' if rpm==25 else 'unverified_legacy')
