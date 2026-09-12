"""Exhaustive saved-byte UUID substitution audit; never an acceptance normalization."""
from pathlib import Path
import argparse
import gzip
import hashlib
import html
import json
import re

ROOT = re.compile(rb'<div id="(uf-[0-9a-f]{10})" class="uf-root">')
SVG = re.compile(rb'<svg\b[^>]*>.*?</svg\s*>', re.S | re.I)
PAYLOAD = re.compile(r'<script type="application/json" data-uf-card-payload="([0-9a-f]{64})" data-uf-card-mount="([^"]*)">(.*?)</script>', re.S)
SLOT = re.compile(r'@@UNFOLD_CARD_SLOT_(\d+)@@')
MOUNT = '@@UNFOLD_PRESENTATION_MOUNT@@'
ATTR = re.compile(rb'([\w:-]+)\s*=\s*([\'"])(.*?)\2', re.S)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def integer(value, size):
    if type(value) is not int or not 0 <= value < size:
        raise ValueError('invalid saved payload index')
    return value


def canonical_html(data):
    """Finite saved-JSON reconstruction; does not import or execute the renderer.

    The stored canonical hash validates reconstructed neutral bytes. This is
    not a runtime interaction check or the product inverse's complete shell
    validation. The original packed HTML remains untouched and hash-pinned.
    """
    document = data.decode()
    receipts = []
    for match in reversed(list(PAYLOAD.finditer(document))):
        expected, declared_mount, serialized = match.groups()
        payload = json.loads(serialized)
        if payload.get('version') != 1 or type(payload.get('version')) is not int:
            raise ValueError('unsupported saved payload version')
        identities, previous = [], ''
        for count, suffix in payload['identities']:
            if type(count) is not int or not 0 <= count <= len(previous) or not isinstance(suffix, str):
                raise ValueError('invalid identity prefix')
            previous = previous[:count] + suffix
            identities.append(previous)
        templates = []
        for parts in payload['templates']:
            template = ''.join(value if isinstance(value, str) else payload['svgs'][integer(value, len(payload['svgs']))] for value in parts)
            used = {int(m.group(1)) for m in SLOT.finditer(template)}
            if used != set(range(len(used))):
                raise ValueError('noncontiguous identity slots')
            templates.append((template, len(used)))
        cards = []
        for template_index, binding, own_slot in payload['cards']:
            template, count = templates[integer(template_index, len(templates))]
            if len(binding) != count:
                raise ValueError('wrong template binding count')
            values = [identities[integer(value, len(identities))] for value in binding]
            own = values[integer(own_slot, len(values))]
            card = SLOT.sub(lambda m: values[integer(int(m.group(1)), len(values))], template)
            own_match = re.search(r'\bdata-card-id="([^"]+)"', card)
            if own_match is None or own_match.group(1) != own:
                raise ValueError('wrong card identity')
            cards.append(card)
        order = [piece for piece in payload['canonical'] if not isinstance(piece, str)]
        if order != list(range(len(cards))):
            raise ValueError('canonical card order differs')
        placed = [integer(value, len(cards)) for _depth, indices in payload['containers'] for value in indices]
        if sorted(placed) != list(range(len(cards))):
            raise ValueError('missing or duplicate deferred card placement')
        neutral = ''.join(piece if isinstance(piece, str) else cards[integer(piece, len(cards))] for piece in payload['canonical'])
        if sha(neutral.encode()) != expected:
            raise ValueError('stored canonical bytes do not reproduce declared hash')
        mount = payload['mount_id']
        if not isinstance(mount, str) or MOUNT in mount or html.unescape(declared_mount) != mount:
            raise ValueError('payload mount mismatch')
        restored = neutral.replace(MOUNT, mount)
        begin, end = f'<!--uf-packed-start:{expected}-->', f'<!--uf-packed-end:{expected}-->'
        start, finish = document.rfind(begin, 0, match.start()), document.find(end, match.end())
        if start < 0 or finish < 0:
            raise ValueError('payload envelope missing')
        document = document[:start] + restored + document[finish + len(end):]
        receipts.append({'canonical_sha256': expected, 'mount_id': mount, 'cards': len(cards),
                         'canonical_bytes': len(restored.encode()), 'strict_stored_hash_equal': True})
    return document.encode(), receipts


def location(svg, offset):
    start = svg.rfind(b'<', 0, offset)
    finish = svg.find(b'>', start)
    if start >= 0 and finish >= offset:
        tag = svg[start:finish + 1]
        for match in ATTR.finditer(tag):
            if start + match.start(3) <= offset < start + match.end(3):
                return {'kind': 'attribute', 'attribute': match.group(1).decode(),
                        'attribute_value': match.group(3).decode(), 'tag_start_byte': start}
        return {'kind': 'unclassified_tag_text', 'tag_start_byte': start}
    return {'kind': 'text_or_style_content', 'nearby': svg[max(0, offset-40):offset+55].decode()}


def svg_audit(before, after, old_root, new_root):
    a, b = SVG.findall(before), SVG.findall(after)
    rows, leftovers = [], []
    for index in range(max(len(a), len(b))):
        if index >= len(a) or index >= len(b):
            leftovers.append({'svg_index': index, 'kind': 'svg_added_or_removed'})
            continue
        positions = [m.start() for m in re.finditer(re.escape(old_root), a[index])]
        substituted = a[index].replace(old_root, new_root)
        equal = substituted == b[index]
        row = {'svg_index': index, 'before_sha256': sha(a[index]), 'after_sha256': sha(b[index]),
               'raw_byte_equal': a[index] == b[index], 'only_paired_uuid_substitution_equal': equal,
               'substitutions': [{'byte_offset': offset, **location(a[index], offset)} for offset in positions]}
        rows.append(row)
        if not equal:
            leftovers.append({'svg_index': index, 'kind': 'bytes_remain_after_exact_uuid_substitution',
                              'substituted_sha256': sha(substituted), 'after_sha256': sha(b[index])})
    return {'before_svg_count': len(a), 'after_svg_count': len(b), 'rows': rows, 'leftovers': leftovers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--current', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    targets = json.loads(args.plan.read_text())['targets']
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows, input_pins = [], {}
    for target in targets:
        slug = target['slug']
        paths = [args.baseline/slug/'page.html.gz', args.current/slug/'page.html.gz']
        original = []
        for path in paths:
            raw = path.read_bytes()
            input_pins[str(path)] = sha(raw)
            original.append(gzip.decompress(raw))
        roots = [ROOT.findall(value) for value in original]
        if any(len(value) != 1 for value in roots):
            raise ValueError('expected exactly one observed top-level UUID mount per page')
        old_root, new_root = roots[0][0], roots[1][0]
        eager = svg_audit(*original, old_root, new_root)
        expanded = [canonical_html(value) for value in original]
        canonical = svg_audit(expanded[0][0], expanded[1][0], old_root, new_root)
        artifact = {'slug': slug, 'before_root': old_root.decode(), 'after_root': new_root.decode(),
                    'raw_html_sha256': [sha(value) for value in original], 'eager_svg': eager,
                    'stored_canonical_svg': canonical, 'payload_reconstruction': [value[1] for value in expanded],
                    'scope': 'Diagnostic finite replacement only. Raw output diffs remain; no acceptance normalization or browser verdict.'}
        (out/f'{slug}.json.gz').write_bytes(gzip.compress((json.dumps(artifact, indent=2)+'\n').encode(), mtime=0))
        rows.append({'slug': slug, 'before_root': old_root.decode(), 'after_root': new_root.decode(),
                     'eager_svg_counts': [eager['before_svg_count'], eager['after_svg_count']],
                     'canonical_svg_counts': [canonical['before_svg_count'], canonical['after_svg_count']],
                     'eager_leftovers': len(eager['leftovers']), 'canonical_leftovers': len(canonical['leftovers']),
                     'canonical_substitutions': sum(len(row['substitutions']) for row in canonical['rows'])})
    if any(sha(Path(path).read_bytes()) != expected for path, expected in input_pins.items()):
        raise ValueError('raw saved HTML changed during audit')
    (out/'result.json').write_text(json.dumps({'scope': 'Exhaustive raw/deferred SVG UUID diagnostic; no output acceptance.', 'models': rows, 'input_pins': input_pins}, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
