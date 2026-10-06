"""Rebuild both feature domains from checksum-verified publisher ZIP archives.

No extraction to disk, positional labels, or synthetic observations are used.
Download the original archives from the URLs in data/publisher.json first.
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import fault_diagnosis as fd


def digest_file(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def condition_key(label):
    # Documented cross-speed naming aliases; preserve original spelling in manifests.
    label = re.sub(r'\s+', ' ', label.strip()).casefold()
    label = re.sub(r'\(\s+', '(', label)
    label = label.replace('combintion', 'combination')
    label = re.sub(r'(bearing \([12]\)) fault \((ball|combination|inner(?: race)?|outer race)\)',
                   lambda m: m[1] + ' ' + m[2].replace(' race', ''), label)
    return label


def parse_filename(filename, rpm, categories):
    match = re.fullmatch(r'(.+?)\s+' + str(rpm) + r'\s+Trial\s+(\d+)\.csv', filename, re.I)
    if not match:
        raise ValueError(f'Unexpected measurement filename: {filename}')
    keys = {condition_key(label): int(category) for category, label in categories.items()}
    key, trial = condition_key(match[1]), int(match[2])
    if key not in keys or not 1 <= trial <= 25:
        raise ValueError(f'Unknown condition or trial: {filename}')
    return keys[key], trial


def read_measurement(blob):
    header = list(csv.reader(io.StringIO(blob[:8192].decode('utf-8-sig'))))[:4]
    channels = ['Tachometer', 'Motor', 'Bearing 1 Z', 'Bearing 1 Y', 'Bearing 1 X',
                'Bearing 2 Z', 'Bearing 2 Y', 'Bearing 2 X', 'Gearbox']
    if len(header) != 4 or any(len(row) != 18 for row in header):
        raise ValueError('Expected 18 alternating timestamp/sensor columns.')
    if header[2][1::2] != channels or header[3][1::2] != ['V'] + ['g'] * 8:
        raise ValueError('Unexpected channel order or physical units.')
    if not all(float(v) == 1 / 6400 for v in header[1][1::2]):
        raise ValueError('Unexpected sampling interval.')
    samples = pd.read_csv(io.BytesIO(blob), usecols=list(range(1, 18, 2)),
                          skiprows=range(1, 4), dtype=float)
    samples.columns = fd.SENSORS
    if len(samples) < 4 or not np.isfinite(samples.to_numpy()).all():
        raise ValueError('Invalid numeric measurement samples.')
    return samples


def process_member(args):
    archive, member, category, trial = args
    with zipfile.ZipFile(archive) as source:
        blob = source.read(member)  # zipfile also verifies this member's CRC.
    samples = read_measurement(blob)
    metadata = {'source_file': Path(member).name, 'fault_category': category,
                'fault_detected': int(category != 37)}
    record = {'filename': metadata['source_file'], 'archive_member': member,
              'fault_detected': metadata['fault_detected'], 'fault_category': category,
              'trial': trial, 'samples': len(samples), 'sha256': hashlib.sha256(blob).hexdigest()}
    header = next(csv.reader(io.StringIO(blob[:8192].decode('utf-8-sig'))))
    if len(set(header[1::2])) != 1:
        raise ValueError('Channel acquisition timestamps disagree.')
    record['acquisition_timestamp'] = header[1]
    metadata['acquisition_timestamp'] = header[1]
    return record, {**fd.signal_features(samples, 'time'), **metadata}, {
        **fd.signal_features(samples, 'frequency'), **metadata}


def rebuild(archive, rpm, workers=4):
    archive = Path(archive)
    publisher = json.loads((fd.ROOT / 'data/publisher.json').read_text())
    source = publisher['files'][str(rpm)]
    if archive.stat().st_size != source['bytes'] or digest_file(archive, 'md5') != source['md5']:
        raise ValueError('Publisher archive checksum/size mismatch; refusing to rebuild.')
    archive_sha = digest_file(archive)
    print(f'{rpm} RPM: publisher MD5 and size verified; SHA-256 {archive_sha}', flush=True)
    categories = json.loads((fd.ROOT / 'data/manifests/25_rpm_categories.json').read_text())
    with zipfile.ZipFile(archive) as z:
        members = sorted(n for n in z.namelist() if n.lower().endswith('.csv') and not n.startswith('__MACOSX/'))
    if len(members) != 975 or len(set(Path(n).name for n in members)) != 975:
        raise ValueError('Expected exactly 975 uniquely named measurement CSVs.')
    tasks = [(str(archive), name, *parse_filename(Path(name).name, rpm, categories)) for name in members]
    pairs = [(task[2], task[3]) for task in tasks]
    if len(set(pairs)) != 975 or set(pairs) != {(c, t) for c in range(1, 40) for t in range(1, 26)}:
        raise ValueError('Missing or duplicate condition/trial combination.')
    records, time_rows, frequency_rows = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for record, time, frequency in pool.map(process_member, tasks, chunksize=4):
            records.append(record); time_rows.append(time); frequency_rows.append(frequency)
            if len(records) % 50 == 0:
                print(f'{rpm} RPM: {len(records)}/975 files checked and both domains extracted', flush=True)
    manifest = pd.DataFrame(records)
    if manifest.sha256.duplicated().any():
        raise ValueError('Duplicate source bytes found; investigate potential split leakage.')
    tables = {'time': pd.DataFrame(time_rows), 'frequency': pd.DataFrame(frequency_rows)}
    for domain, table in tables.items():
        fd.validate_features(table, domain)
    manifest.to_csv(fd.ROOT / f'data/manifests/{rpm}_rpm.csv', index=False)
    (fd.ROOT / f'data/manifests/{rpm}_rpm_categories.json').write_text(json.dumps(categories, indent=2) + '\n')
    catalog_path = fd.ROOT / 'data/provenance.json'
    catalog = json.loads(catalog_path.read_text())
    for domain, table in tables.items():
        output = fd.feature_path(rpm, domain)
        table.to_csv(output, index=False)
        catalog[digest_file(output)] = {'status': 'reconstructed_from_raw', 'rpm': rpm,
            'rows': len(table), 'manifest': f'data/manifests/{rpm}_rpm.csv',
            'manifest_sha256': digest_file(fd.ROOT / f'data/manifests/{rpm}_rpm.csv'),
            'archive_sha256': archive_sha, 'publisher_md5_verified': source['md5'],
            'doi': publisher['doi'], 'legacy_skip_rows': 3,
            'label_basis': 'source_filename', 'independent_run_validation': False}
    catalog_path.write_text(json.dumps(catalog, indent=2) + '\n')
    print(f'{rpm} RPM: rebuilt 975 rows; samples {manifest.samples.min()}–{manifest.samples.max()}; 39 balanced conditions.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--rpm', type=int, choices=(50, 75), required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    rebuild(args.archive, args.rpm, args.workers)
