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

  panels.slice(1).forEach(function(panel, panelOffset) {{
    var index = panelOffset + 1;
    sourceNodesFor(index).forEach(function(n) {{
      n.style.cursor = 'pointer';
      n.addEventListener('click', function(e) {{
        e.stopPropagation();
        if (n.classList.contains('uf-nested-selected')) {{ showPanel(index, 'default'); }}
        else {{ showPanel(index, n.getAttribute('data-id')); }}
      }});
    }});
  }});
}})();
</script>
"""
