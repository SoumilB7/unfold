"""One parameter pass preserves each constructed card's original ordered rows."""
from types import SimpleNamespace

from model_unfolder.adapters.diffusor.unet_projection import project_unet


class ParameterKey(str):
    partitions = 0

    def rpartition(self, separator):
        type(self).partitions += 1
        return super().rpartition(separator)


class ParameterRows(dict):
    item_scans = 0

    def items(self):
        self.item_scans += 1
        return super().items()


class ShapeRow(dict):
    reads = []

    def __getitem__(self, key):
        if key == 'shape':
            type(self).reads.append(self)
        return super().__getitem__(key)


def original_parameters(path, shapes):
    # Exact original per-card selection, including its order and row references.
    return [(key, row) for key, row in shapes['parameters'].items()
            if key.rpartition('.')[0] == path]


def project(modules, parameters, *, stages=()):
    shapes = {'parameters': parameters,
              'by_module': {path: 100 + number for number, path in enumerate(modules)}}
    values = {
        'constructed_modules': modules,
        'constructed_parameter_shapes': shapes,
        'constructed_stage_relations': {
            'producer_stages': list(stages), 'intermediate_stages': [],
            'consumer_stages': [], 'producer_field': 'down', 'consumer_field': 'up',
            'unresolved_relations': [{'reason': 'Original open relation'}]},
    }
    # These are projection-only rows, not forged qualified summary proofs.
    facts = {'root.denoiser.' + key: SimpleNamespace(value=value)
             for key, value in values.items()}
    ir = project_unet(facts=facts, handoffs={}, name='Grouping', architecture='Fixture')
    assert ir.construction_summary is None
    assert ir.extras['unet']['parameter_shapes'] is shapes
    assert ir.warnings == ['Original open relation']
    return ir, shapes


def cards(ir):
    def walk(block):
        if block.get('role') == 'constructed':
            yield block
        for child in block.get('children', ()):
            yield from walk(child)
    return [card for block in ir.extras['render']['loop_blocks'] for card in walk(block)]


def test_original_parent_order_and_row_identity_survive_real_projection():
    modules = {
        '': {'class_name': 'Root', 'children': ['α', 'ab', 'empty']},
        'α': {'class_name': 'Parent', 'children': ['child']},
        'α.child': {'class_name': 'Child', 'children': []},
        'ab': {'class_name': 'Sibling', 'children': []},
        'empty': {'class_name': 'Empty', 'children': []},
    }
    shared = ShapeRow(shape=[7, 2])
    parameters = ParameterRows([
        (ParameterKey('α.child.last'), ShapeRow(shape=[3])),
        (ParameterKey('α.z'), shared),
        (ParameterKey('root_weight'), ShapeRow(shape=[99])),
        (ParameterKey('ab.weight'), ShapeRow(shape=[5, 4])),
        (ParameterKey('α.a'), shared),
        (ParameterKey('α.'), ShapeRow(shape=[])),
        (ParameterKey('absent.weight'), ShapeRow(shape=[999])),
    ])
    expected = {path: original_parameters(path, {'parameters': parameters})
                for path in modules if path}
    ShapeRow.reads = []
    ir, shapes = project(modules, parameters, stages=('α',))
    actual = {card['source_instance_path']: card for card in cards(ir)}
    assert list(actual) == ['α', 'α.child', 'ab', 'empty']
    for path, card in actual.items():
        lines = [f"{key.rpartition('.')[2]}: " + ' × '.join(
            map(str, dict.__getitem__(row, 'shape'))) for key, row in expected[path]]
        assert card['facts'] == [f"{shapes['by_module'][path]:,} parameters in subtree", *lines,
                                *(['Cross-attention role: investigation_missing · owner: S8']
                                  if path == 'α' else [])]
        assert card['source_fact_keys'] == [
            'root.denoiser.constructed_modules', 'root.denoiser.constructed_parameter_shapes',
            *(['root.denoiser.constructed_stage_relations'] if path == 'α' else [])]
    # Card recursion visits child first; every actual shape access uses the
    # original row object, including both aliases, not a reconstructed copy.
    expected_reads = [row for path in ('α.child', 'α', 'ab', 'empty')
                      for _, row in expected[path]]
    assert len(ShapeRow.reads) == len(expected_reads)
    assert all(actual is expected for actual, expected in zip(ShapeRow.reads, expected_reads))
    assert parameters['α.z'] is parameters['α.a'] is shared


def test_many_modules_partition_each_parameter_once_for_grouping():
    paths = [f'module_{number}' for number in range(96)]
    modules = {'': {'class_name': 'Root', 'children': paths},
               **{path: {'class_name': 'Constructed', 'children': []} for path in paths}}
    # Interleave modules so iteration order cannot be replaced by name sorting.
    parameters = ParameterRows((ParameterKey(f'{path}.{suffix}'), ShapeRow(shape=[number + 1]))
                               for suffix in ('z', 'a', 'weight', 'bias')
                               for number, path in enumerate(reversed(paths)))
    shapes = {'parameters': parameters}
    ParameterKey.partitions = 0
    expected = {path: original_parameters(path, shapes) for path in paths}
    assert parameters.item_scans == len(paths)
    assert ParameterKey.partitions == len(paths) * len(parameters)
    parameters.item_scans = 0
    ParameterKey.partitions = 0
    ShapeRow.reads = []
    ir, _ = project(modules, parameters)
    assert parameters.item_scans == 1
    # One parent extraction per parameter, plus the unchanged displayed suffix.
    assert ParameterKey.partitions == 2 * len(parameters)
    actual = cards(ir)
    assert [card['source_instance_path'] for card in actual] == paths
    for card in actual:
        rows = expected[card['source_instance_path']]
        assert card['facts'][1:] == [
            f"{str(key).rpartition('.')[2]}: " + ' × '.join(map(str, dict.__getitem__(row, 'shape')))
            for key, row in rows]
    expected_reads = [row for path in paths for _, row in expected[path]]
    assert len(ShapeRow.reads) == len(expected_reads)
    assert all(actual is expected for actual, expected in zip(ShapeRow.reads, expected_reads))


def test_empty_parameter_population_preserves_empty_card_chips():
    modules = {'': {'class_name': 'Root', 'children': ['empty']},
               'empty': {'class_name': 'Empty', 'children': []}}
    parameters = ParameterRows()
    ir, shapes = project(modules, parameters)
    assert parameters.item_scans == 1
    assert cards(ir)[0]['facts'] == [f"{shapes['by_module']['empty']:,} parameters in subtree"]
