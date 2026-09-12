"""Shared source fixtures; no test-module imports or test collection side effects."""
from __future__ import annotations

from model_unfolder.evidence.class_default_value import model_hidden_size_class_default
from model_unfolder.evidence.config_access import bound_document
from model_unfolder.evidence.document import DocumentBinding, PreparedDocument, checkpoint_provenance
from model_unfolder.evidence.facts import EvidenceFact, SourceSpan
from model_unfolder.evidence.models import SourceBundle
from model_unfolder.evidence.program_index import build_program_index
from model_unfolder.evidence.reader_claims import qualify_reader_fact


def _read(tmp_path, *, default='64', annotation='ExactConfig', support_component='root',
          checkpoint=None, overlay=None, config_path=(), model_suffix='', inherited=False,
          reader_document=None, failure=None, before_super='', after_super=''):
    model = tmp_path / 'modeling_default.py'
    support = tmp_path / 'configuration_default.py'
    model.write_text(f'''from transformers.modeling_utils import PreTrainedModel
from .configuration_default import ExactConfig
class Base(PreTrainedModel):
    config: {annotation}
class Wrapper(Base):
    def __init__(self, config):
{before_super}        super().__init__(config)
{after_super}{model_suffix}
''')
    support.write_text(f'''class {'Parent' if inherited else 'ExactConfig'}:
    hidden_size: int = {default}
''' + ('class ExactConfig(Parent):\n    pass\n' if inherited else ''))
    bundle = SourceBundle(source='local', files=(str(model),),
        component_files={'root': (str(model),)}, supporting_files={support_component: (str(support),)},
        component_architectures={'root': 'Wrapper'}, architecture='Wrapper')
    index = build_program_index(bundle)
    checkpoint = {} if checkpoint is None else checkpoint
    document = PreparedDocument(dict(checkpoint) if reader_document is None else reader_document,
                                dict(checkpoint),
                                {'hidden_size': 64} if overlay is None else overlay,
                                provenance=checkpoint_provenance(checkpoint), failure=failure)
    with bound_document(DocumentBinding('root', (), document)):
        result = model_hidden_size_class_default(index, bundle, config_path)
    return result, index, document


def _fact(result, index, document):
    projection = result.claim_witness.project('model', 'hidden_size', document)
    fact = EvidenceFact(key='hidden_size', owner='model', value=projection.value,
        status='class_default', completeness='complete', config_paths=(),
        source_spans=tuple(dict.fromkeys(SourceSpan(component=span.source.component_key or 'root',
                        file=span.source.canonical_path, line=span.line)
                        for origin in result.provenance for span in origin.spans)))
    return qualify_reader_fact(fact, result, index, document)
