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
  var l2cards = root.querySelectorAll('.uf-inspect .uf-card-detail');
  var l3 = root.querySelectorAll('.uf-inspect .uf-card-svg .uf-node');
  var l3cards = root.querySelectorAll('.uf-sub-inspect .uf-card-detail');
  var l3box = root.querySelector('.uf-sub-inspect');

  function showL2(id) {{
    l2cards.forEach(function(p) {{
      p.style.display = p.classList.contains('uf-card-' + id) ? 'block' : 'none';
    }});
    l1.forEach(function(n) {{
      if (n.getAttribute('data-id') === id) n.classList.add('uf-selected');
      else n.classList.remove('uf-selected');
    }});
    showL3('default');
  }}

  function showL3(id) {{
    l3cards.forEach(function(p) {{
      p.style.display = p.classList.contains('uf-l3-' + id) ? 'block' : 'none';
    }});
    l3.forEach(function(n) {{
      if (n.getAttribute('data-id') === id) n.classList.add('uf-sub-selected');
      else n.classList.remove('uf-sub-selected');
    }});
    if (l3box) {{
      if (id === 'default') l3box.classList.remove('uf-sub-active');
      else l3box.classList.add('uf-sub-active');
    }}
  }}

  l1.forEach(function(n) {{
    n.style.cursor = 'pointer';
    n.addEventListener('click', function(e) {{
      e.stopPropagation();
      if (n.classList.contains('uf-selected')) {{ showL2('default'); }}
      else {{ showL2(n.getAttribute('data-id')); }}
    }});
  }});

  l3.forEach(function(n) {{
    n.style.cursor = 'pointer';
    n.addEventListener('click', function(e) {{
      e.stopPropagation();
      if (n.classList.contains('uf-sub-selected')) {{ showL3('default'); }}
      else {{ showL3(n.getAttribute('data-id')); }}
    }});
  }});
}})();
</script>
"""
