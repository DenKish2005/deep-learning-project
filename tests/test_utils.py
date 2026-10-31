"""Offline schema tests; no datasets, project model, training or CUDA required."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from src.utils import ConfigError, load_config, validate_config, validate_controlled_configs


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pair():
    return [load_config(ROOT / 'configs' / name) for name in ('clean_cnn.yaml', 'robust_cnn.yaml')]


def change(cfg, field, value):
    parts = field.split('.')
    for key in parts[:-1]:
        cfg = cfg[key]
    cfg[parts[-1]] = value


def test_config_schema_and_controlled_fields(pair):
    clean, robust = pair
    validate_controlled_configs(clean, robust)
    assert clean['training']['epochs'] == 40
    assert clean['seeds'] == [42, 123, 2026]
    assert clean['augmentation']['probabilities'] == [1, 0, 0, 0]
    assert robust['augmentation']['probabilities'] == [0.5, 0.17, 0.17, 0.16]
    # Schema tests read configs only; actual NPZ identity is checked separately opt-in.
    assert clean['dataset']['split_sha256'] == robust['dataset']['split_sha256']
    assert len(clean['dataset']['split_sha256']) == 64
    assert clean['dataset']['normalization_sha256'] == robust['dataset']['normalization_sha256']
    assert len(clean['dataset']['normalization_sha256']) == 64
    assert clean['corruptions']['blur_sigma'] == [0.3, 0.7, 1.1, 1.5, 2.0]


@pytest.mark.parametrize('field,value,diagnostic', [
    ('schema_version', 2, 'schema_version'),
    ('schema_version', True, 'schema_version'),
    ('schema_version', '1', 'schema_version'),
    ('schema_version', 1.0, 'schema_version'),
    ('training.epochs', 0, 'training.epochs'),
    ('training.epochs', True, 'expected integer'),
    ('training.batch_size', -1, 'training.batch_size'),
    ('training.batch_size', 1.5, 'expected integer'),
    ('training.lr', 0, 'training.lr'),
    ('training.lr', -0.1, 'training.lr'),
    ('training.lr', '0.001', 'expected finite number'),
    ('training.lr', float('nan'), 'expected finite number'),
    ('training.eps', float('inf'), 'expected finite number'),
    ('training.weight_decay', False, 'expected finite number'),
    ('training.weight_decay', -1, 'training.weight_decay'),
    ('training.betas', [0.9, 1], r'betas\[1\]'),
    ('training.betas', [0.9], 'length 2'),
    ('training.amp', 'false', 'expected boolean'),
    ('training.early_stopping', 0, 'expected boolean'),
    ('training.gradient_clip', 1.0, 'gradient_clip'),
    ('training.horizontal_flip_p', 1.1, 'horizontal_flip_p'),
    ('dropout', 1, 'dropout'),
    ('dropout', -0.1, 'dropout'),
    ('num_classes', 0, 'num_classes'),
    ('seeds', [42, 123, True], 'expected integer'),
    ('seeds', [42, 123, -1], 'seeds'),
    ('seeds', [42, 123], 'length 3'),
    ('dataset.train_size', 0, 'train_size'),
    ('dataset.split_seed', 2**63, 'split_seed'),
    ('dataset.name', 'cifar100', 'dataset.name'),
    ('architecture', 'resnet18', 'architecture'),
    ('training.optimizer', 'Adam', 'optimizer'),
    ('runtime.device', 'magic', 'runtime.device'),
    ('runtime.num_workers', -1, 'num_workers'),
    ('runtime.num_workers', True, 'expected integer'),
    ('runtime.pin_memory', 1, 'pin_memory'),
    ('runtime.deterministic', 'true', 'expected boolean'),
    ('runtime.persistent_workers', True, 'requires num_workers > 0'),
    ('dataset.split_sha256', 'a' * 63, '64 lowercase hex'),
    ('dataset.normalization_sha256', 'A' * 64, '64 lowercase hex'),
    ('dataset.split_sha256', 123, '64 lowercase hex'),
    ('dataset.root', '', 'repository-relative path'),
    ('dataset.split_path', '/tmp/split.npz', 'repository-relative path'),
    ('output.root', 'C:/private/runs', 'repository-relative path'),
    ('output.root', '../runs', 'repository-relative path'),
    ('output.root', 'outputs\\runs', 'repository-relative path'),
    ('output.root', ' ', 'repository-relative path'),
    ('output.root', '.', 'repository-relative path'),
    ('output.root', None, 'repository-relative path'),
])
def test_invalid_types_ranges_and_schema(pair, field, value, diagnostic):
    cfg = deepcopy(pair[0])
    change(cfg, field, value)
    with pytest.raises(ConfigError, match=diagnostic):
        validate_config(cfg)


@pytest.mark.parametrize('value,diagnostic', [
    ([0.5, 0.17, 0.17], 'length 4'),
    ([0.5, 0.17, 0.17, -0.16], r'probabilities\[3\]'),
    ([1.1, 0, 0, 0], r'probabilities\[0\]'),
    ([True, 0, 0, 0], 'finite number'),
    ([float('nan'), 0, 0, 0], 'finite number'),
    ([0.5, 0.17, 0.17, 0.1], 'sum to 1'),
    ([0.4, 0.2, 0.2, 0.2], 'protocol v1 requires'),
])
def test_probabilities(pair, value, diagnostic):
    cfg = deepcopy(pair[1])
    cfg['augmentation']['probabilities'] = value
    with pytest.raises(ConfigError, match=diagnostic):
        validate_config(cfg)


@pytest.mark.parametrize('field,value', [
    ('augmentation.enabled', True),
    ('augmentation.probabilities', [0.5, 0.17, 0.17, 0.16]),
    ('augmentation.train_severities', [1, 2, 4]),
    ('augmentation.train_severities', [1, 2, 2]),
    ('evaluation.severities', [0, 1, 2, 3, 4]),
    ('evaluation.severities', [5, 4, 3, 2, 1]),
    ('evaluation.severities', [1, 2, 3, 4, 6]),
    ('corruptions.noise_sigma', [0.02, 0.05, 0.08, 0.12]),
    ('corruptions.noise_sigma', [0.02, 0.05, 0.08, 0.12, 0.2]),
    ('corruptions.noise_sigma', [0, 0.05, 0.08, 0.12, 0.18]),
    ('corruptions.blur_sigma', [0.7, 0.3, 1.1, 1.5, 2.0]),
    ('corruptions.blur_kernel_rule', 'arbitrary'),
    ('corruptions.blur_padding', 'zeros'),
    ('corruptions.brightness_factor', [0.9, 0.8, 0.7, 0.55, 0]),
    ('corruptions.brightness_factor', [0.9, 0.8, 0.7, 0.55, 1.1]),
    ('corruptions.brightness_factor', [0.9, 0.8, 0.7, 0.55, 0.5]),
    ('training.epochs', 41),
    ('training.lr', 0.01),
    ('training.t_max', 39),
    ('training.early_stopping', True),
    ('training.amp', True),
    ('runtime.deterministic', False),
    ('seeds', [42, 42, 2026]),
    ('seeds', [123, 42, 2026]),
])
def test_core_protocol_changes_rejected(pair, field, value):
    cfg = deepcopy(pair[0])
    change(cfg, field, value)
    with pytest.raises(ConfigError, match=field.replace('.', r'\.')):
        validate_config(cfg)


@pytest.mark.parametrize('section', [None, 'dataset', 'training', 'augmentation', 'corruptions', 'evaluation', 'runtime', 'output'])
def test_required_and_unknown_keys(pair, section):
    for unknown in (False, True):
        cfg = deepcopy(pair[0])
        target = cfg if section is None else cfg[section]
        if unknown:
            target['typo'] = 1
        else:
            target.pop(next(iter(target)))
        with pytest.raises(ConfigError, match='unknown keys: typo' if unknown else 'missing required keys:'):
            validate_config(cfg)


@pytest.mark.parametrize('cfg', [None, {}, [], 'config', {1: 'bad'}])
def test_non_mapping_and_non_string_keys(cfg):
    with pytest.raises(ConfigError):
        validate_config(cfg)


@pytest.mark.parametrize('text', [
    '', '# comment-only', '[]', 'name: [',
    'schema_version: 1\nschema_version: 2',
    'training:\n  lr: 0.001\n  lr: 0.002',
    '1: value', '!!python/object/apply:os.system [echo unsafe]',
    '---\nschema_version: 1\n---\nschema_version: 1',
])
def test_invalid_yaml(tmp_path, text):
    path = tmp_path / 'invalid.yaml'
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ConfigError, match='invalid.yaml'):
        load_config(path)


def test_read_errors(tmp_path):
    with pytest.raises(ConfigError, match='cannot read config'):
        load_config(tmp_path / 'missing.yaml')
    path = tmp_path / 'bad.yaml'
    path.write_bytes(b'\xff\xfe\xff')
    with pytest.raises(ConfigError, match='cannot read config'):
        load_config(path)


def test_load_config_does_not_resolve_or_create_dataset(pair, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = deepcopy(pair[0])
    cfg['dataset']['root'] = 'future/raw'
    cfg['dataset']['split_path'] = 'future/split.npz'
    cfg['dataset']['normalization_path'] = 'future/stats.json'
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    assert load_config(path) == cfg
    assert list(tmp_path.iterdir()) == [path]
    # A syntactically valid supplied hash is not evidence of a verified artifact.
    cfg['dataset']['split_sha256'] = 'a' * 64
    validate_config(cfg)


@pytest.mark.parametrize('field,value', [
    ('dataset.root', 'other/raw'),
    ('dataset.split_path', 'other/split.npz'),
    ('dataset.normalization_path', 'other/stats.json'),
    ('dataset.split_sha256', 'a' * 64),
    ('dataset.normalization_sha256', 'b' * 64),
    ('runtime.device', 'cpu'),
    ('runtime.num_workers', 2),
    ('runtime.pin_memory', False),
])
def test_controlled_settings_mismatch(pair, field, value):
    clean, robust = deepcopy(pair)
    change(robust, field, value)
    validate_config(robust)  # Individually valid; the comparison must still fail.
    with pytest.raises(ConfigError, match='controlled setting differs'):
        validate_controlled_configs(clean, robust)


def test_allowed_output_changes_and_shared_relocation(pair):
    clean, robust = deepcopy(pair)
    for key in robust['output']:
        robust['output'][key] = 'other/' + key
    for cfg in (clean, robust):
        cfg['dataset']['root'] = 'shared/raw'
        cfg['runtime']['device'] = 'cpu'
        cfg['runtime']['num_workers'] = 2
        cfg['runtime']['persistent_workers'] = True
    before = deepcopy((clean, robust))
    validate_controlled_configs(clean, robust)
    assert (clean, robust) == before  # Validation does not rewrite user config.


def test_pair_roles_and_invalid_controls(pair):
    clean, robust = pair
    for configs in ((robust, clean), (clean, clean), (robust, robust)):
        with pytest.raises(ConfigError, match='expected clean_cnn then robust_cnn'):
            validate_controlled_configs(*configs)
    broken = deepcopy(robust)
    broken['dropout'] = 0.3
    with pytest.raises(ConfigError, match='dropout'):
        validate_controlled_configs(clean, broken)


def test_validation_cli(tmp_path):
    command = [sys.executable, '-m', 'src.utils', '--configs',
               'configs/clean_cnn.yaml', 'configs/robust_cnn.yaml', '--controlled-pair']
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'controlled settings equal' in result.stdout
    assert 'NOT verified' in result.stdout
    invalid = tmp_path / 'invalid.yaml'
    invalid.write_text('# empty', encoding='utf-8')
    result = subprocess.run([sys.executable, '-m', 'src.utils', '--configs', str(invalid)],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert 'Configuration error' in result.stderr and 'nonempty mapping' in result.stderr
    result = subprocess.run(command[:-2] + ['--controlled-pair'], cwd=ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert 'exactly two configs' in result.stderr
