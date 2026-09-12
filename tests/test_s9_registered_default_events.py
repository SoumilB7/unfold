"""Actual registered-source premises remain distinct from checkpoint occurrences."""
from dataclasses import replace

import pytest

from model_unfolder.evidence import config_access
from model_unfolder.evidence.config_registration import (
    RegisteredConstructorDefaultPremise,
    registered_constructor_default,
    registered_constructor_default_premise,
)
from model_unfolder.evidence.document import DocumentBinding
from model_unfolder.evidence.receipts import value_status_hash
from test_support.s9_fixtures.diffusion_config_binding import CONFIG, REGISTERED_SOURCE, _inputs


def _setup(tmp_path, *, config=None):
    config = ({key: value for key, value in CONFIG.items() if key != 'layers'}
              if config is None else config)
    index, root, binding, _topology, _companions = _inputs(
        tmp_path, source=REGISTERED_SOURCE, config=config)
    default = registered_constructor_default(index, root, binding.prepared, ('layers',))
    assert default is not None
    return index, root, binding, default


def _premise(index, root, binding, default, *, source_obj=None, spellings=('layers',)):
    return registered_constructor_default_premise(
        index, root, binding.prepared,
        binding.document if source_obj is None else source_obj,
        default, component='root.denoiser', spellings=spellings)


def _resolve(binding, premise, **kwargs):
    options = dict(component='root.denoiser', class_defaults={'layers': 4},
                   default_premise=premise)
    options.update(kwargs)
    return config_access.resolve(binding.document, 'layers', (), **options)


def test_registered_source_default_is_exact_from_first_event_without_overlay(tmp_path):
    index, root, binding, default = _setup(tmp_path)
    assert 'layers' not in binding.prepared.class_overlay
    with config_access.bound_document(binding), config_access.capture_events() as ledger:
        premise = _premise(index, root, binding, default)
        assert type(premise) is RegisteredConstructorDefaultPremise
        resolution = _resolve(binding, premise)
        resolution.bind(reader='actual_source_reader', fact_owner='root.denoiser', fact_key='depth')
        decision = resolution.consume_decision(
            reader='actual_source_reader', fact_owner='root.denoiser', fact_key='depth',
            mechanism='depth', status='class_default', expected_value=4)
    assert resolution.selected_path is None and resolution.selected_alias is None
    assert resolution.state == 'absent' and resolution.value == 4
    assert decision.occurrence is None
    assert ledger.events and all(event.intent == 'absent_default' and not event.present
                                 and event.path_exact and event.config_path == 'layers'
                                 and event.provenance == 'class_default' for event in ledger.events)
    consuming = [event for event in ledger.events if event.mechanism == 'depth']
    assert len(consuming) == 1
    assert consuming[0].value_status_hash == value_status_hash(4, 'class_default')


@pytest.mark.parametrize('changed', [
    {'class_defaults': {'layers': 99}},
    {'component': 'root.foreign'},
])
def test_mismatched_resolution_rejected_before_any_event(tmp_path, changed):
    index, root, binding, default = _setup(tmp_path)
    with config_access.bound_document(binding), config_access.capture_events() as ledger:
        premise = _premise(index, root, binding, default)
        with pytest.raises(ValueError):
            _resolve(binding, premise, **changed)
        assert ledger.events == []


def test_registered_premise_cannot_borrow_foreign_source_container_or_document(tmp_path):
    index, root, binding, default = _setup(tmp_path)
    with config_access.bound_document(binding):
        with pytest.raises(ValueError):
            _premise(index, root, binding, default, source_obj=dict(binding.document))
        premise = _premise(index, root, binding, default)
    foreign = replace(binding.prepared, checkpoint=dict(binding.prepared.checkpoint),
                      document=dict(binding.prepared.document))
    with config_access.bound_document(DocumentBinding('root', (), foreign)):
        with pytest.raises(ValueError):
            premise.validate()


@pytest.mark.parametrize('alias_value', [4, None])
def test_present_alias_including_null_prevents_default_premise(tmp_path, alias_value):
    config = {key: value for key, value in CONFIG.items() if key != 'layers'}
    config['depth'] = alias_value
    index, root, binding, default = _setup(tmp_path, config=config)
    with config_access.bound_document(binding):
        with pytest.raises(ValueError):
            _premise(index, root, binding, default, spellings=('layers', 'depth'))


def test_registered_premise_rejects_foreign_index_or_source_declaration(tmp_path):
    index, root, binding, default = _setup(tmp_path / 'first')
    other_index, other_root, other_binding, other_default = _setup(tmp_path / 'other')
    with config_access.bound_document(binding):
        premise = _premise(index, root, binding, default)
        with pytest.raises(ValueError):
            replace(premise, index=other_index)
        with pytest.raises(ValueError):
            replace(premise, default=other_default)
        with pytest.raises(TypeError):
            _premise(index, root, binding, {'value': 4})


def test_registered_resolution_revalidates_value_and_channels_without_event_capture(tmp_path):
    index, root, binding, default = _setup(tmp_path)
    with config_access.bound_document(binding):
        premise = _premise(index, root, binding, default)
        resolution = _resolve(binding, premise)
        with pytest.raises(ValueError, match="resolution changed its exact supplying value"):
            replace(resolution, value=99).consume(
                fact_owner='root.denoiser', fact_key='depth', reader='actual_source_reader')
        binding.prepared.class_overlay['layers'] = 99
        with pytest.raises(ValueError, match="supplying channels changed"):
            resolution.consume(fact_owner='root.denoiser', fact_key='depth', reader='actual_source_reader')
