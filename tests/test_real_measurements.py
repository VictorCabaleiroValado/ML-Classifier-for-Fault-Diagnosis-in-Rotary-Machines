"""Guard against inherited labels, incorrect sensor mappings and demo leakage."""
import csv
import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest

import fault_diagnosis as fd
from recover_measurements import parse_filename, read_measurement, rebuild


def test_filename_labels_require_known_condition_speed_and_trial():
    categories = json.loads((fd.ROOT / 'data/manifests/25_rpm_categories.json').read_text())
    assert parse_filename('Bearing (2) Fault (inner) 50 Trial 3.csv', 50, categories) == (31, 3)
    assert parse_filename('Bearing (1) Ball 50 Trial 1.csv', 50, categories) == (11, 1)
    assert parse_filename('Bearing (1) Inner & Bearing (2) Combination 50 Trial 1.csv', 50, categories) == (16, 1)
    assert parse_filename('Shaft Fault ( Coupling End Bent) 75 Trial 1.csv', 75, categories) == (39, 1)
    assert parse_filename('No Fault 75 Trial 25.csv', 75, categories) == (37, 25)
    for filename in ['No Fault 50 Trial 1.csv', 'New Fault 75 Trial 1.csv', 'No Fault 75 Trial 26.csv']:
        with pytest.raises(ValueError):
            parse_filename(filename, 75, categories)


def test_source_header_validates_channels_units_and_sampling_interval():
    channels = ['Tachometer', 'Motor', 'Bearing 1 Z', 'Bearing 1 Y', 'Bearing 1 X',
                'Bearing 2 Z', 'Bearing 2 Y', 'Bearing 2 X', 'Gearbox']
    rows = [[x for v in ['2022-09-10'] * 9 for x in ['Timestamp', v]],
            [x for v in [1 / 6400] * 9 for x in ['Interval', v]],
            [x for v in channels for x in ['Channel name', v]],
            [x for v in ['V'] + ['g'] * 8 for x in ['Unit', v]]]
    rows += [[x for j in range(9) for x in [i / 6400, j + i]] for i in range(4)]
    def blob():
        out = io.StringIO(); csv.writer(out).writerows(rows)
        return out.getvalue().encode()
    samples = read_measurement(blob())
    assert list(samples.columns) == list(fd.SENSORS)
    assert samples.iloc[0].Motor == 1
    rows[2][3] = 'Gearbox'
    with pytest.raises(ValueError, match='channel order'):
        read_measurement(blob())


def test_corrupt_archive_is_rejected_before_outputs(tmp_path):
    archive = tmp_path / 'invalid.zip'; archive.write_bytes(b'not publisher data')
    with pytest.raises(ValueError, match='checksum/size'):
        rebuild(archive, 50)


@pytest.mark.parametrize('rpm', [25, 50, 75])
def test_real_demo_is_held_out_and_uses_its_own_speed_model(rpm):
    data = json.loads((fd.ROOT / 'demo' / ('data.json' if rpm == 25 else f'real-{rpm}.json')).read_text())
    path = fd.feature_path(rpm, 'time')
    df, X, y = fd.load_features(path, 'time')
    train, test = fd.split_indices(y)
    model = fd.build_model('tree'); model.fit(X.iloc[train], y.iloc[train].fault_category)
    assert data['kind'] == 'real' and data['modelTrainingRpm'] == data['rpm'] == rpm
    assert data['featureTableSha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert data['accuracy'] == np.mean(model.predict(X.iloc[test]) == y.iloc[test].fault_category)
    assert data['trainIndices'] == train.tolist() and data['testIndices'] == test.tolist()
    for e in data['examples']:
        i = e['id']
        assert i == min(j for j in test if y.iloc[j].fault_category == e['category'])
        assert e['labelBasis'] == 'source_filename' and e['kind'] == 'real'
        assert e['source'] == df.iloc[i].source_file
        assert e['expectedPrediction'] == model.predict(X.iloc[[i]])[0]
        np.testing.assert_allclose(e['features'], X.iloc[i].values, rtol=1e-8, atol=1e-10)
        if rpm != 25:
            manifest = pd.read_csv(fd.ROOT / f'data/manifests/{rpm}_rpm.csv')
            assert e['sha256'] == manifest.iloc[i].sha256
            assert e['samples'] == manifest.iloc[i].samples
