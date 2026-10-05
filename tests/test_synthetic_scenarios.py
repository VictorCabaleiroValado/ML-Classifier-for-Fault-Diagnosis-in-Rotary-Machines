"""Integrity tests: synthetic data must never masquerade as measured validation."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'demo'))
from export_demo import synthesize, SCENARIO_SAMPLES
import fault_diagnosis as fd


def test_transform_is_deterministic_and_does_not_mutate_source():
    samples = pd.DataFrame(np.random.default_rng(7).normal(size=(13000, 9)), columns=fd.SENSORS)
    before = samples.copy()
    first = synthesize(samples, 50, 12)
    pd.testing.assert_frame_equal(first, synthesize(samples, 50, 12))
    pd.testing.assert_frame_equal(samples, before)
    assert first.shape == (SCENARIO_SAMPLES, 9)
    assert not first.equals(synthesize(samples, 75, 12))
    assert not first.equals(synthesize(samples, 50, 13))
    with pytest.raises(ValueError):
        synthesize(samples, 25, 12)
    with pytest.raises(ValueError):
        synthesize(samples.iloc[:5], 50, 12)
    samples.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        synthesize(samples, 50, 12)


@pytest.mark.parametrize('rpm', [50, 75])
def test_published_scenarios_have_exact_csv_features_and_heldout_parents(rpm):
    real = json.loads((ROOT / 'demo/data.json').read_text())
    data = json.loads((ROOT / f'demo/synthetic-{rpm}.json').read_text())
    parents = {e['id']: e for e in real['examples']}
    assert data['accuracy'] is None
    assert data['kind'] == 'synthetic'
    assert data['tree'] == real['tree']
    assert data['trainIndices'] == real['trainIndices']
    assert len(data['examples']) == 39
    for e in data['examples']:
        parent = parents[e['parentRow']]
        assert e['id'] not in real['trainIndices']
        assert e['sourceSha256'] == parent['sha256']
        assert e['parentPrediction'] == parent['expectedPrediction']
        assert e['parentMotorRms'] == parent['features'][real['features'].index('rms_Motor')]
        assert e['label'] == parent['label']
        assert e['labelBasis'] == 'inherited_from_25rpm_parent'
        assert e['kind'] == 'synthetic' and e['rpm'] == rpm
        path = ROOT / 'demo' / e['rawCsv']
        assert path.name.startswith(f'synthetic_{rpm}rpm_')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == e['sha256']
        samples = pd.read_csv(path)
        assert samples.shape == (SCENARIO_SAMPLES, 9)
        np.testing.assert_allclose(list(fd.signal_features(samples, 'time').values()), e['features'], rtol=1e-12)
        assert all(0 <= p[0] < SCENARIO_SAMPLES for p in e['signal'])
