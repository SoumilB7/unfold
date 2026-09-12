"""Inline browser behavior for the HTML renderer."""
from __future__ import annotations


def _click_script(mount_id: str) -> str:
    """Inline JS for click-to-inspect."""
    return f"""
<script>
(function() {{
  var root = document.getElementById('{mount_id}');
  if (!root) return;

  var l1 = root.querySelectorAll('.uf-section-arch .uf-node');
  if (!l1.length) l1 = root.querySelectorAll('.uf-section-body .uf-node');
  var panels = Array.prototype.slice.call(root.querySelectorAll('.uf-inspect-panel'));
  var panelSizes = [
    'uf-panel-hint',
    'uf-panel-compact',
    'uf-panel-list',
    'uf-panel-diagram-compact',
    'uf-panel-diagram',
    'uf-panel-diagram-tall'
  ];

  function sourcePanelFor(index) {{
    return index === 0 ? root.querySelector('.uf-section-arch') : panels[index - 1];
  }}

  function sourceNodesFor(index) {{
    var source = sourcePanelFor(index);
    if (!source) return [];
    if (index === 0) return Array.prototype.slice.call(l1);
    return Array.prototype.slice.call(source.querySelectorAll('.uf-card-svg .uf-node'));
  }}

  function selectedClassFor(index) {{
    return index === 0 ? 'uf-selected' : 'uf-nested-selected';
  }}

  function setPanelSize(panel, card) {{
    if (!panel) return;
    panelSizes.forEach(function(cls) {{ panel.classList.remove(cls); }});
    var size = card && card.getAttribute('data-card-size') ? card.getAttribute('data-card-size') : 'compact';
    panel.classList.add('uf-panel-' + size);
    if (card && card.getAttribute('data-svg-width')) {{
      panel.style.setProperty('--uf-card-svg-width', card.getAttribute('data-svg-width'));
      panel.style.setProperty('--uf-card-svg-height', card.getAttribute('data-svg-height') || '');
    }} else {{
      panel.style.removeProperty('--uf-card-svg-width');
      panel.style.removeProperty('--uf-card-svg-height');
    }}
  }}

  function clearPanelsFrom(index) {{
    for (var i = index; i < panels.length; i++) {{
      if (i > 0) panels[i].classList.remove('uf-nested-active');
      panels[i].querySelectorAll('.uf-card-detail[data-card-id]').forEach(function(p) {{
        p.style.display = 'none';
      }});
      setPanelSize(panels[i], null);
      sourceNodesFor(i).forEach(function(n) {{
        n.classList.remove(selectedClassFor(i));
      }});
    }}
  }}

  function showPanel(index, id) {{
    clearPanelsFrom(index + 1);
    var panel = panels[index];
    if (!panel) return;

    var selectedClass = selectedClassFor(index);
    sourceNodesFor(index).forEach(function(n) {{
      if (n.getAttribute('data-id') === id) n.classList.add(selectedClass);
      else n.classList.remove(selectedClass);
    }});

    var found = false;
    var activeCard = null;
    panel.querySelectorAll('.uf-card-detail[data-card-id]').forEach(function(p) {{
      var active = p.getAttribute('data-card-id') === id;
      p.style.display = active ? 'block' : 'none';
      if (active) {{
        found = true;
        if (!activeCard) activeCard = p;
      }}
    }});

    if (id === 'default' || !found) {{
      if (index > 0) panel.classList.remove('uf-nested-active');
      if (!found) {{
        sourceNodesFor(index).forEach(function(n) {{
          n.classList.remove(selectedClass);
        }});
      }}
      setPanelSize(panel, activeCard);
    }} else {{
      if (index > 0) panel.classList.add('uf-nested-active');
      setPanelSize(panel, activeCard);
    }}
  }}

  l1.forEach(function(n) {{
    n.style.cursor = 'pointer';
    n.addEventListener('click', function(e) {{
      e.stopPropagation();
      if (n.classList.contains('uf-selected')) {{ showPanel(0, 'default'); }}
      else {{ showPanel(0, n.getAttribute('data-id')); }}
    }});
  }});

  root.querySelectorAll('.uf-card-svg .uf-node').forEach(function(n) {{
    n.style.cursor = 'pointer';
  }});

  root.addEventListener('click', function(e) {{
    var node = e.target.closest ? e.target.closest('.uf-card-svg .uf-node') : null;
    if (!node || !root.contains(node)) return;

    var sourcePanel = node.closest('.uf-inspect-panel');
    var index = panels.indexOf(sourcePanel) + 1;
    if (index <= 0 || index >= panels.length) return;

    e.stopPropagation();
    if (node.classList.contains('uf-nested-selected')) {{ showPanel(index, 'default'); }}
    else {{ showPanel(index, node.getAttribute('data-id')); }}
  }});
}})();
</script>
"""


def _lazy_click_script(mount_id: str, digest: str) -> str:
    """Decode stored strings on demand; never execute the stored canonical JS."""
    import json

    script = r'''
<script>
(function() {
  'use strict';
  var root = document.getElementById(@MOUNT@);
  var data = Array.from(document.querySelectorAll('script[data-uf-card-payload="' + @DIGEST@ + '"]')).find(function(item) {
    return item.getAttribute('data-uf-card-mount') === @MOUNT@;
  });
  if (!root || !data) throw new Error('Missing inspect-card payload');
  var p = JSON.parse(data.textContent);
  function fail() { throw new Error('Invalid inspect-card payload'); }
  function list(v) { if (!Array.isArray(v)) fail(); return v; }
  function text(v) { if (typeof v !== 'string') fail(); return v; }
  function ref(v, n) { if (!Number.isInteger(v) || v < 0 || v >= n) fail(); return v; }
  if (!p || p.version !== 1 || p.mount_id !== @MOUNT@) fail();
  var identities = [], previous = '';
  list(p.identities).forEach(function(row) {
    if (list(row).length !== 2) fail();
    var chars = Array.from(previous), n = row[0];
    if (!Number.isInteger(n) || n < 0 || n > chars.length) fail();
    previous = chars.slice(0, n).join('') + text(row[1]);
    identities.push(previous);
  });
  var svg = list(p.svgs).map(text), slotCounts = [];
  var templates = list(p.templates).map(function(parts) {
    var value = list(parts).map(function(v) { return typeof v === 'string' ? v : svg[ref(v, svg.length)]; }).join('');
    var slots = new Set(), re = /@@UNFOLD_CARD_SLOT_(\d+)@@/g, m;
    if (value.replace(re, '').indexOf('@@UNFOLD_CARD_SLOT_') >= 0) fail();
    while ((m = re.exec(value))) slots.add(Number(m[1]));
    for (var i = 0; i < slots.size; i++) if (!slots.has(i)) fail();
    slotCounts.push(slots.size);
    return value;
  });
  var cards = list(p.cards);
  cards.forEach(function(row) {
    if (list(row).length !== 3) fail();
    var t = ref(row[0], templates.length), binding = list(row[1]);
    if (binding.length !== slotCounts[t]) fail();
    binding.forEach(function(v) { ref(v, identities.length); });
    ref(row[2], binding.length);
    var own = templates[t].match(/\bdata-card-id="@@UNFOLD_CARD_SLOT_(\d+)@@"/);
    if (!own || Number(own[1]) !== row[2]) fail();
  });
  var declared = list(p.containers), seen = new Set();
  declared.forEach(function(row) {
    if (list(row).length !== 2 || !Number.isInteger(row[0]) || row[0] < 2) fail();
    list(row[1]).forEach(function(v) { ref(v, cards.length); if (seen.has(v)) fail(); seen.add(v); });
  });
  if (seen.size !== cards.length) fail();
  var ordinal = 0;
  list(p.canonical).forEach(function(v) {
    if (typeof v !== 'string' && ref(v, cards.length) !== ordinal++) fail();
  });
  if (ordinal !== cards.length) fail();
  var entityDecoder = document.createElement('textarea');
  function attribute(value) {
    if (value.indexOf('&') < 0) return value;
    entityDecoder.innerHTML = value;
    return entityDecoder.value;
  }
  function mounted(value) { return value.split('@@UNFOLD_PRESENTATION_MOUNT@@').join(@MOUNT@); }
  function cardId(i) { var row = cards[i]; return attribute(mounted(identities[row[1][row[2]]])); }
  function cardHTML(i) {
    var row = cards[i];
    return mounted(templates[row[0]].replace(/@@UNFOLD_CARD_SLOT_(\d+)@@/g, function(_, slot) {
      return identities[row[1][ref(Number(slot), row[1].length)]];
    }));
  }
  var panels = Array.from(root.querySelectorAll('.uf-inspect-panel'));
  var containers = [], markerSeen = new Set();
  root.querySelectorAll('template[data-uf-card-container]').forEach(function(marker) {
    var i = ref(Number(marker.getAttribute('data-uf-card-container')), declared.length);
    if (markerSeen.has(i)) fail(); markerSeen.add(i);
    var panel = marker.closest('.uf-inspect-panel'), panelIndex = panels.indexOf(panel);
    if (panelIndex < 0 || Number(panel.getAttribute('data-depth')) !== declared[i][0]) fail();
    var byId = new Map();
    declared[i][1].forEach(function(record) {
      var id = cardId(record); if (!byId.has(id)) byId.set(id, []); byId.get(id).push(record);
    });
    containers.push({marker: marker, panel: panelIndex, byId: byId, active: []});
  });
  if (markerSeen.size !== declared.length) fail();
  var sizes = ['hint','compact','list','diagram-compact','diagram','diagram-tall'];
  var top = root.querySelectorAll('.uf-section-arch .uf-node');
  if (!top.length) top = root.querySelectorAll('.uf-section-body .uf-node');
  function nodes(index) {
    if (index === 0) return Array.from(top);
    return Array.from(panels[index - 1].querySelectorAll('.uf-card-svg .uf-node'));
  }
  function setSize(panel, card) {
    sizes.forEach(function(s) { panel.classList.remove('uf-panel-' + s); });
    panel.classList.add('uf-panel-' + (card ? card.getAttribute('data-card-size') || 'compact' : 'compact'));
    if (card && card.hasAttribute('data-svg-width')) {
      panel.style.setProperty('--uf-card-svg-width', card.getAttribute('data-svg-width'));
      panel.style.setProperty('--uf-card-svg-height', card.getAttribute('data-svg-height') || '');
    } else {
      panel.style.removeProperty('--uf-card-svg-width'); panel.style.removeProperty('--uf-card-svg-height');
    }
  }
  function removeCards(c) { c.active.forEach(function(n) { n.remove(); }); c.active = []; }
  function insert(c, id, shown) {
    var result = [];
    (c.byId.get(id) || []).forEach(function(record) {
      var holder = document.createElement('template'); holder.innerHTML = cardHTML(record);
      var card = holder.content.firstElementChild;
      if (!card || card.getAttribute('data-card-id') !== id || holder.content.children.length !== 1) fail();
      if (shown) card.style.display = 'block';
      card.querySelectorAll('.uf-node').forEach(function(n) { n.style.cursor = 'pointer'; });
      c.marker.parentNode.insertBefore(card, c.marker);
      c.active.push(card); result.push(card);
    });
    return result;
  }
  function clearFrom(index) {
    for (var i = index; i < panels.length; i++) {
      if (i > 0) panels[i].classList.remove('uf-nested-active');
      containers.forEach(function(c) { if (c.panel === i) removeCards(c); });
      setSize(panels[i], null);
      nodes(i).forEach(function(n) { n.classList.remove(i === 0 ? 'uf-selected' : 'uf-nested-selected'); });
    }
  }
  function show(index, id) {
    clearFrom(index + 1);
    if (!panels[index]) return;
    var selected = index === 0 ? 'uf-selected' : 'uf-nested-selected';
    nodes(index).forEach(function(n) { n.classList.toggle(selected, n.getAttribute('data-id') === id); });
    var active = [];
    containers.forEach(function(c) {
      if (c.panel === index) { removeCards(c); active.push.apply(active, insert(c, id, true)); }
    });
    if (id === 'default' || !active.length) {
      if (index > 0) panels[index].classList.remove('uf-nested-active');
      if (!active.length) nodes(index).forEach(function(n) { n.classList.remove(selected); });
    } else if (index > 0) panels[index].classList.add('uf-nested-active');
    setSize(panels[index], active[0] || null);
  }
  containers.forEach(function(c) { insert(c, 'default', false); });
  top.forEach(function(n) {
    n.style.cursor = 'pointer';
    n.addEventListener('click', function(e) {
      e.stopPropagation(); show(0, n.classList.contains('uf-selected') ? 'default' : n.getAttribute('data-id'));
    });
  });
  root.addEventListener('click', function(e) {
    var node = e.target.closest ? e.target.closest('.uf-card-svg .uf-node') : null;
    if (!node || !root.contains(node)) return;
    var index = panels.indexOf(node.closest('.uf-inspect-panel')) + 1;
    if (index <= 0 || index >= panels.length) return;
    e.stopPropagation(); show(index, node.classList.contains('uf-nested-selected') ? 'default' : node.getAttribute('data-id'));
  });
  root.setAttribute('data-uf-cards-ready', 'true');
})();
</script>
'''
    return script.replace('@MOUNT@', json.dumps(mount_id).replace('<', '\\u003c')).replace('@DIGEST@', json.dumps(digest))
