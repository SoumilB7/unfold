/**
 * unfold renderer.
 * Consumes an IR JSON and renders an interactive architecture diagram.
 *
 * Public API (attached to window):
 *   Unfold.render(ir, mountElement)
 *
 * The IR shape is documented in unfold/ir.py.
 */
(function (global) {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const FONT_HEAD = '"Caveat","Patrick Hand","Comic Sans MS",cursive';
  const FONT_BODY = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif';
  const FONT_MONO = 'ui-monospace, "JetBrains Mono", "SF Mono", Menlo, monospace';
  const GAP = 6;

  // role-based palette — different block types get visually distinct hues
  // so the eye can find "attention" or "FFN" without reading every label.
  const C = {
    bg_outer:    "#F4F1F8",   // pale lavender
    bg_inner:    "#E5F0FF",   // pale azure (the "transformer block" backdrop)
    bg_card:     "#FFFFFF",
    canvas:      "#FAFBFD",

    embed:       "#7C3AED",   // violet
    norm:        "#475569",   // slate
    attn:        "#0D9488",   // teal
    attn_mla:    "#0F766E",   // darker teal — MLA gets its own shade
    ffn:         "#D97706",   // amber
    ffn_moe:     "#C2410C",   // burnt orange — MoE
    output:      "#059669",   // emerald
    add:         "#1F2937",   // dark slate
    text_block:  "#FFFFFF",

    arrow:       "#475569",
    text:        "#0F172A",
    muted:       "#64748B",
    border:      "#E2E8F0",
    badge_bg:    "#EEF2FF",
    badge_text:  "#3730A3",
    badge_warm:  "#FEF3C7",
    badge_warm_t:"#92400E",
    badge_cool:  "#CCFBF1",
    badge_cool_t:"#115E59",
  };

  function el(tag, attrs) {
    const n = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) {
      if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    }
    return n;
  }

  function fmtInt(n) {
    if (n == null) return "?";
    return Number(n).toLocaleString();
  }

  function fmtH(n) {
    // human param count; matches Python's humanize()
    if (n == null) return "?";
    n = Number(n);
    const units = [["T",1e12],["B",1e9],["M",1e6],["K",1e3]];
    for (const [u, s] of units) {
      if (n >= s) {
        const v = n / s;
        if (v >= 100) return v.toFixed(0) + u;
        if (v >= 10)  return v.toFixed(1) + u;
        return v.toFixed(2) + u;
      }
    }
    return String(Math.round(n));
  }

  function defs(svg) {
    const d = el("defs");
    const m = el("marker", {
      id: "uf-arrow", viewBox: "0 0 10 10", refX: 8, refY: 5,
      markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse"
    });
    m.appendChild(el("path", {
      d: "M2 1L8 5L2 9", fill: "none", stroke: "context-stroke",
      "stroke-width": 1.5, "stroke-linecap": "round", "stroke-linejoin": "round"
    }));
    d.appendChild(m);

    // soft drop shadow used on blocks
    const f = el("filter", { id: "uf-shadow", x: "-20%", y: "-20%", width: "140%", height: "140%" });
    f.appendChild(el("feGaussianBlur", { in: "SourceAlpha", stdDeviation: 1.2 }));
    f.appendChild(el("feOffset", { dx: 0, dy: 1, result: "off" }));
    const ct = el("feComponentTransfer");
    const fa = el("feFuncA", { type: "linear", slope: "0.18" });
    ct.appendChild(fa); f.appendChild(ct);
    const m2 = el("feMerge");
    m2.appendChild(el("feMergeNode"));
    m2.appendChild(el("feMergeNode", { in: "SourceGraphic" }));
    f.appendChild(m2);
    d.appendChild(f);

    // subtle stripe pattern for MoE blocks
    const p = el("pattern", { id: "uf-moe-stripe", patternUnits: "userSpaceOnUse",
                              width: 6, height: 6, patternTransform: "rotate(45)" });
    p.appendChild(el("rect", { width: 6, height: 6, fill: C.ffn_moe }));
    p.appendChild(el("rect", { x: 0, width: 2, height: 6, fill: "rgba(255,255,255,0.13)" }));
    d.appendChild(p);

    svg.appendChild(d);
  }

  function regionRect(x, y, w, h, fill, opts) {
    opts = opts || {};
    return el("rect", {
      x, y, width: w, height: h, rx: 18, ry: 18, fill,
      stroke: opts.stroke || "none", "stroke-width": opts.strokeWidth || 0
    });
  }

  function rectBlock(svg, id, x, y, w, h, label, opts) {
    opts = opts || {};
    const fontSize = opts.fontSize || 17;
    const fill = opts.fill || C.attn;
    const isStripe = opts.stripe;
    const g = el("g", { class: "uf-node", "data-id": id, "data-role": opts.role || "" });
    g.setAttribute("style", "cursor:pointer;");
    g.appendChild(el("rect", {
      x, y, width: w, height: h, rx: 11, ry: 11,
      fill: isStripe ? "url(#uf-moe-stripe)" : fill,
      stroke: "rgba(15, 23, 42, 0.18)", "stroke-width": 0.6,
      filter: "url(#uf-shadow)"
    }));
    const lines = Array.isArray(label) ? label : [label];
    const lineH = fontSize + 3;
    const startY = y + h / 2 - ((lines.length - 1) * lineH) / 2;
    lines.forEach((line, i) => {
      const t = el("text", {
        x: x + w / 2, y: startY + i * lineH,
        "text-anchor": "middle", "dominant-baseline": "central",
        fill: C.text_block, "font-family": FONT_HEAD,
        "font-size": fontSize, "pointer-events": "none"
      });
      t.textContent = line;
      g.appendChild(t);
    });
    if (opts.tag) {
      const tagW = 36, tagH = 14;
      const tx = x + w - tagW - 6, ty = y + 6;
      g.appendChild(el("rect", {
        x: tx, y: ty, width: tagW, height: tagH, rx: 4, ry: 4,
        fill: "rgba(255,255,255,0.22)", stroke: "rgba(255,255,255,0.4)", "stroke-width": 0.5
      }));
      const tt = el("text", {
        x: tx + tagW / 2, y: ty + tagH / 2 + 0.5,
        "text-anchor": "middle", "dominant-baseline": "central",
        fill: "#FFFFFF", "font-family": FONT_BODY, "font-size": 9,
        "font-weight": 600, "letter-spacing": "0.08em", "pointer-events": "none"
      });
      tt.textContent = opts.tag;
      g.appendChild(tt);
    }
    svg.appendChild(g);
    return {
      el: g, kind: "rect",
      left: x, right: x + w, top: y, bottom: y + h,
      cx: x + w / 2, cy: y + h / 2, w, h
    };
  }

  function plusBlock(svg, id, cx, cy, sym) {
    const r = 13;
    const g = el("g", { class: "uf-node", "data-id": id, style: "cursor:pointer;" });
    g.appendChild(el("circle", {
      cx, cy, r, fill: C.add, stroke: "rgba(255,255,255,0.6)", "stroke-width": 1,
      filter: "url(#uf-shadow)"
    }));
    const t = el("text", {
      x: cx, y: cy + 1, "text-anchor": "middle", "dominant-baseline": "central",
      fill: C.text_block, "font-family": FONT_HEAD, "font-size": 22, "pointer-events": "none"
    });
    t.textContent = sym || "+";
    g.appendChild(t);
    svg.appendChild(g);
    return {
      el: g, kind: "plus",
      left: cx - r, right: cx + r, top: cy - r, bottom: cy + r,
      cx, cy, r
    };
  }

  function vLine(svg, fromNode, toNode, opts) {
    opts = opts || {};
    const line = el("line", {
      x1: fromNode.cx, y1: fromNode.bottom,
      x2: fromNode.cx, y2: toNode.top - GAP,
      stroke: C.arrow, "stroke-width": 1.4, "stroke-linecap": "round",
      "marker-end": "url(#uf-arrow)", fill: "none"
    });
    svg.appendChild(line);
    if (opts.label) {
      const midY = (fromNode.bottom + toNode.top) / 2;
      const t = el("text", {
        x: fromNode.cx + 10, y: midY,
        "text-anchor": "start", "dominant-baseline": "central",
        fill: C.muted, "font-family": FONT_MONO, "font-size": 10
      });
      t.textContent = opts.label;
      svg.appendChild(t);
    }
  }

  function elbowPath(svg, x1, y1, x2, y2) {
    let d;
    if (Math.abs(x2 - x1) < 1 || Math.abs(y2 - y1) < 1) {
      d = `M ${x1} ${y1} L ${x2} ${y2}`;
    } else {
      const sx = Math.sign(x2 - x1);
      const sy = Math.sign(y2 - y1);
      const r = Math.min(10, Math.abs(x2 - x1) / 2, Math.abs(y2 - y1) / 2);
      d = `M ${x1} ${y1} ` +
          `L ${x1} ${y2 - sy * r} ` +
          `Q ${x1} ${y2} ${x1 + sx * r} ${y2} ` +
          `L ${x2} ${y2}`;
    }
    svg.appendChild(el("path", {
      d, fill: "none", stroke: C.arrow,
      "stroke-width": 1.4, "stroke-linecap": "round", "stroke-linejoin": "round",
      "marker-end": "url(#uf-arrow)"
    }));
  }

  function residualLoopRight(svg, fromNode, toNode, lane) {
    const r = 12;
    const startX = fromNode.right;
    const startY = fromNode.cy;
    const endX = toNode.right;
    const endY = toNode.cy;
    const d = `M ${startX} ${startY} ` +
              `L ${lane - r} ${startY} ` +
              `Q ${lane} ${startY} ${lane} ${startY - r} ` +
              `L ${lane} ${endY + r} ` +
              `Q ${lane} ${endY} ${lane - r} ${endY} ` +
              `L ${endX + GAP} ${endY}`;
    svg.appendChild(el("path", {
      d, fill: "none", stroke: C.arrow,
      "stroke-width": 1.4, "stroke-linecap": "round", "stroke-linejoin": "round",
      "stroke-dasharray": "0",
      "marker-end": "url(#uf-arrow)"
    }));
    // tiny "residual" hint
    const midY = (startY + endY) / 2;
    const t = el("text", {
      x: lane + 8, y: midY, "text-anchor": "start", "dominant-baseline": "central",
      fill: C.muted, "font-family": FONT_BODY, "font-size": 10, "font-style": "italic"
    });
    t.textContent = "residual";
    svg.appendChild(t);
  }

  function describeAttention(a) {
    if (a.kind === "mla") {
      return `Multi-head latent attention · ${a.num_heads} heads · KV LoRA ${fmtInt(a.kv_lora_rank)}` +
             (a.q_lora_rank ? ` · Q LoRA ${fmtInt(a.q_lora_rank)}` : "");
    }
    if (a.kind === "gqa") {
      return `Grouped-query · ${a.num_heads} Q heads / ${a.num_kv_heads} KV heads · head dim ${fmtInt(a.head_dim)}`;
    }
    if (a.kind === "mqa") {
      return `Multi-query · ${a.num_heads} Q heads / 1 KV head`;
    }
    return `Multi-head · ${a.num_heads} heads · head dim ${fmtInt(a.head_dim)}`;
  }

  function describeFFN(f) {
    if (f.kind === "moe") {
      const sparsity = f.num_experts && f.num_experts_per_tok
        ? ` · ${(100 * f.num_experts_per_tok / f.num_experts).toFixed(1)}% active`
        : "";
      return `MoE · ${fmtInt(f.num_experts)} experts · top-${f.num_experts_per_tok}` +
             (f.num_shared_experts ? ` + ${f.num_shared_experts} shared` : "") +
             sparsity + ` · expert hidden ${fmtInt(f.expert_intermediate_size || f.intermediate_size)}`;
    }
    const gated = f.gated ? "gated " : "";
    return `${gated}FFN · ${f.activation} · hidden ${fmtInt(f.intermediate_size)}`;
  }

  function archBadges(ir, info) {
    const badges = [];
    const a = info.dominant.spec.attention;
    const f = info.dominant.spec.ffn;

    if (a.kind === "mla") badges.push({ text: "MLA", tone: "cool", title: "Multi-head latent attention" });
    else if (a.kind === "gqa") badges.push({ text: `GQA ${a.num_heads}/${a.num_kv_heads}`, tone: "cool", title: "Grouped-query attention" });
    else if (a.kind === "mqa") badges.push({ text: "MQA", tone: "cool", title: "Multi-query attention" });
    else badges.push({ text: "MHA", tone: "cool", title: "Multi-head attention" });

    if (f.kind === "moe") {
      badges.push({
        text: `MoE ${f.num_experts_per_tok}/${f.num_experts}`,
        tone: "warm",
        title: `Mixture of experts — top-${f.num_experts_per_tok} of ${f.num_experts}`
      });
    } else {
      badges.push({ text: "Dense FFN", tone: "warm", title: "Dense feed-forward" });
    }

    if (info.groups.length > 1) {
      badges.push({ text: `${info.groups.length} layer types`, tone: "default" });
    }

    if (a.mask === "sliding") {
      badges.push({ text: `SWA ${fmtInt(a.window_size)}`, tone: "default", title: "Sliding-window attention" });
    }

    return badges;
  }

  function makeInfo(ir) {
    const groups = [];
    let cur = null;
    ir.layers.forEach(l => {
      const sig = JSON.stringify([
        l.attention.kind, l.attention.mask, l.attention.window_size,
        l.ffn.kind, l.ffn.num_experts, l.norm_kind, l.norm_placement
      ]);
      if (cur && cur.sig === sig) cur.indices.push(l.index);
      else { cur = { sig, indices: [l.index], spec: l }; groups.push(cur); }
    });

    const dominant = groups.reduce((a, b) =>
      a.indices.length >= b.indices.length ? a : b
    );

    const a = dominant.spec.attention;
    const f = dominant.spec.ffn;

    return {
      groups,
      dominant,
      meta: {
        tok_text: ["Tokenized text", "Input token IDs · shape [batch, seq_len]"],
        embed: ["Token embedding", `${fmtInt(ir.vocab_size)} × ${fmtInt(ir.hidden_size)}` +
                (ir.tie_word_embeddings ? " (tied with output)" : "")],
        rms1: ["Pre-attention norm", `RMSNorm · dim ${fmtInt(ir.hidden_size)}`],
        attn: ["Attention", describeAttention(a)],
        add1: ["Residual add", "block input + attention output"],
        rms2: ["Pre-FFN norm", `RMSNorm · dim ${fmtInt(ir.hidden_size)}`],
        ffn: [f.kind === "moe" ? "Mixture of experts" : "Feed-forward", describeFFN(f)],
        add2: ["Residual add", "post-attention + FFN output"],
        final_rms: ["Final norm", `RMSNorm · dim ${fmtInt(ir.hidden_size)}`],
        lm_head: ["LM head", `${fmtInt(ir.hidden_size)} → ${fmtInt(ir.vocab_size)}` +
                  (ir.tie_word_embeddings ? " (tied)" : "")],
      }
    };
  }

  function buildArchitectureView(ir, info) {
    const W = 720, H = 1080;
    const svg = el("svg", { width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img" });
    const t = el("title"); t.textContent = `${ir.name} architecture`; svg.appendChild(t);
    defs(svg);

    // outer card
    svg.appendChild(regionRect(40, 30, W - 80, H - 60, C.bg_outer));

    const cx = W / 2;

    // ── repeating transformer block ───────────────────────────────────────
    const innerX = 110, innerY = 240, innerW = W - 220, innerH = 580;
    svg.appendChild(regionRect(innerX, innerY, innerW, innerH, C.bg_inner));

    const a = info.dominant.spec.attention;
    const f = info.dominant.spec.ffn;

    const attnLabel = a.kind === "mla" ? ["Multi-head latent", "attention"]
                   : a.kind === "gqa" ? ["Grouped-query", "attention"]
                   : a.kind === "mqa" ? ["Multi-query", "attention"]
                   : ["Multi-head", "attention"];

    const ffnLabel = f.kind === "moe"
      ? [`MoE · ${fmtInt(f.num_experts)} experts`, `top-${f.num_experts_per_tok} per token`]
      : ["Feed-forward"];

    const tokText  = rectBlock(svg, "tok_text",  cx - 110, H - 120, 220, 48, "Tokenized text",
                               { fill: C.embed, role: "embed", fontSize: 16 });
    const embed    = rectBlock(svg, "embed",     cx - 130, H - 200, 260, 48, "Token embedding",
                               { fill: C.embed, role: "embed" });

    const rms1     = rectBlock(svg, "rms1",      cx - 80,  innerY + 470, 160, 38, "RMSNorm",
                               { fill: C.norm, role: "norm", fontSize: 15 });
    const attn     = rectBlock(svg, "attn",      cx - 115, innerY + 360, 230, 70, attnLabel,
                               { fill: a.kind === "mla" ? C.attn_mla : C.attn,
                                 role: "attn", tag: a.kind.toUpperCase() });
    const add1     = plusBlock (svg, "add1",     cx,       innerY + 320);
    const rms2     = rectBlock(svg, "rms2",      cx - 80,  innerY + 230, 160, 38, "RMSNorm",
                               { fill: C.norm, role: "norm", fontSize: 15 });
    const ffn      = rectBlock(svg, "ffn",       cx - 115, innerY + 130, 230, 70, ffnLabel,
                               { fill: f.kind === "moe" ? C.ffn_moe : C.ffn,
                                 role: "ffn", stripe: f.kind === "moe",
                                 tag: f.kind === "moe" ? "MOE" : "FFN" });
    const add2     = plusBlock (svg, "add2",     cx,       innerY + 90);

    const finalRms = rectBlock(svg, "final_rms", cx - 80,  170, 160, 38, "Final RMSNorm",
                               { fill: C.norm, role: "norm", fontSize: 15 });
    const lmHead   = rectBlock(svg, "lm_head",   cx - 130, 90,  260, 48, "Linear · LM head",
                               { fill: C.output, role: "output" });

    // wires bottom-up
    vLine(svg, tokText, embed);
    vLine(svg, embed, rms1, { label: `[B, T, ${fmtInt(ir.hidden_size)}]` });
    vLine(svg, rms1, attn);
    vLine(svg, attn, add1);
    vLine(svg, add1, rms2);
    vLine(svg, rms2, ffn);
    vLine(svg, ffn, add2);
    vLine(svg, add2, finalRms);
    vLine(svg, finalRms, lmHead);

    // arrow up off the LM head
    svg.appendChild(el("line", {
      x1: cx, y1: lmHead.top, x2: cx, y2: lmHead.top - 28,
      stroke: C.arrow, "stroke-width": 1.4, "stroke-linecap": "round",
      "marker-end": "url(#uf-arrow)", fill: "none"
    }));
    const logits = el("text", {
      x: cx, y: lmHead.top - 38, "text-anchor": "middle",
      fill: C.muted, "font-family": FONT_MONO, "font-size": 11
    });
    logits.textContent = `logits [B, T, ${fmtInt(ir.vocab_size)}]`;
    svg.appendChild(logits);

    // residual loops on the right
    const lane = innerX + innerW - 28;
    residualLoopRight(svg, rms1, add1, lane);
    residualLoopRight(svg, rms2, add2, lane);

    // "× N layers" label, prominently
    const repeatBg = el("rect", {
      x: innerX + innerW - 92, y: innerY + 12, width: 80, height: 26,
      rx: 13, ry: 13, fill: "rgba(255,255,255,0.7)",
      stroke: C.border, "stroke-width": 0.5
    });
    svg.appendChild(repeatBg);
    const repeat = el("text", {
      x: innerX + innerW - 52, y: innerY + 25, "text-anchor": "middle",
      "dominant-baseline": "central",
      fill: C.text, "font-family": FONT_HEAD, "font-size": 20
    });
    repeat.textContent = `× ${ir.layers.length}`;
    svg.appendChild(repeat);

    // header label inside the inner card
    const blockLabel = el("text", {
      x: innerX + 18, y: innerY + 26, "text-anchor": "start",
      fill: C.muted, "font-family": FONT_BODY, "font-size": 11,
      "letter-spacing": "0.12em", "font-weight": 600
    });
    blockLabel.textContent = "TRANSFORMER BLOCK";
    svg.appendChild(blockLabel);

    if (info.groups.length > 1) {
      const note = el("text", {
        x: innerX + 18, y: innerY + 44, "text-anchor": "start",
        fill: C.muted, "font-family": FONT_BODY, "font-size": 11, "font-style": "italic"
      });
      note.textContent = `${info.groups.length} variants — see "Layer map"`;
      svg.appendChild(note);
    }

    // small annotation: "input embedding" / "output projection" on outer card
    [
      { y: H - 220, text: "EMBEDDING" },
      { y: 70, text: "OUTPUT" }
    ].forEach(o => {
      const t = el("text", {
        x: 64, y: o.y, "text-anchor": "start",
        fill: C.muted, "font-family": FONT_BODY, "font-size": 10,
        "letter-spacing": "0.14em", "font-weight": 600
      });
      t.textContent = o.text;
      svg.appendChild(t);
    });

    return svg;
  }

  function buildMoeView(ir, info) {
    const W = 720, H = 560;
    const svg = el("svg", { width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img" });
    defs(svg);
    svg.appendChild(regionRect(40, 30, W - 80, H - 60, C.bg_outer));

    const ffn = info.dominant.spec.ffn;
    const cx = W / 2;
    const router = rectBlock(svg, "router", cx - 80, H - 110, 160, 50, "Router",
                             { fill: C.norm, role: "router" });
    const sumNode = plusBlock(svg, "add_moe", cx, 80);

    const expertY = 230, expertW = 130, expertH = 60;
    const slots = [
      { x: 110,            label: ["Expert", "1"],     id: "expert_1" },
      { x: 270,            label: ["Expert", "k"],     id: "expert_k" },
      { x: 430,            label: ["Expert", "k+1"],   id: "expert_kp1" },
      { x: W - 110 - expertW, label: ["Expert", `${ffn.num_experts || "N"}`], id: "expert_n" }
    ];
    const experts = slots.map(s => rectBlock(svg, s.id, s.x, expertY, expertW, expertH, s.label, {
      fill: C.ffn_moe, stripe: true, role: "expert", fontSize: 16
    }));

    // dots between expert k and k+1
    const dotsX = (experts[1].right + experts[2].left) / 2;
    const dotsY = expertY + expertH / 2;
    for (let i = -2; i <= 2; i++) {
      svg.appendChild(el("circle", { cx: dotsX + i * 7, cy: dotsY, r: 2.5, fill: C.muted }));
    }

    // "(N total)" annotation
    const annot = el("text", {
      x: experts[3].right, y: experts[3].bottom + 22, "text-anchor": "end",
      fill: C.muted, "font-family": FONT_BODY, "font-size": 12, "font-style": "italic"
    });
    annot.textContent = `${ffn.num_experts || "N"} experts total`;
    svg.appendChild(annot);

    experts.forEach(e => {
      svg.appendChild(el("line", {
        x1: e.cx, y1: router.top - GAP, x2: e.cx, y2: e.bottom + GAP,
        stroke: C.arrow, "stroke-width": 1.4, "stroke-linecap": "round",
        "marker-end": "url(#uf-arrow)", fill: "none"
      }));
    });

    experts.forEach(e => {
      const sx = sumNode.cx;
      const targetX = sx + (e.cx < sx ? -sumNode.r - GAP : sumNode.r + GAP);
      elbowPath(svg, e.cx, e.top - GAP, targetX, sumNode.cy);
    });

    // sparsity callout at top-right
    if (ffn.num_experts && ffn.num_experts_per_tok) {
      const sparsity = (100 * ffn.num_experts_per_tok / ffn.num_experts).toFixed(1);
      const cgX = W - 240, cgY = 60, cgW = 200, cgH = 56;
      svg.appendChild(el("rect", {
        x: cgX, y: cgY, width: cgW, height: cgH, rx: 10, ry: 10,
        fill: C.bg_card, stroke: C.border, "stroke-width": 0.5
      }));
      const lbl = el("text", {
        x: cgX + 12, y: cgY + 16, fill: C.muted,
        "font-family": FONT_BODY, "font-size": 10,
        "letter-spacing": "0.12em", "font-weight": 600
      });
      lbl.textContent = "ACTIVE PER TOKEN";
      svg.appendChild(lbl);
      const big = el("text", {
        x: cgX + 12, y: cgY + 42, fill: C.text,
        "font-family": FONT_HEAD, "font-size": 24
      });
      big.textContent = `${ffn.num_experts_per_tok} / ${ffn.num_experts}  ·  ${sparsity}%`;
      svg.appendChild(big);
    }

    const inLabel = el("text", {
      x: cx, y: H - 22, "text-anchor": "middle",
      fill: C.text, "font-family": FONT_HEAD, "font-size": 18
    });
    inLabel.textContent = `top-${ffn.num_experts_per_tok || "k"} of ${ffn.num_experts || "N"} experts active per token`;
    svg.appendChild(inLabel);

    return svg;
  }

  function buildFfnView(ir, info) {
    const W = 720, H = 560;
    const svg = el("svg", { width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img" });
    defs(svg);
    svg.appendChild(regionRect(40, 30, W - 80, H - 60, C.bg_outer));

    const ffn = info.dominant.spec.ffn;
    const cx = W / 2;
    const actName = (ffn.activation || "silu").toUpperCase();

    const downProj = rectBlock(svg, "down_proj", cx - 90, 90, 180, 50, "Linear (down)",
                               { fill: C.ffn, role: "ffn" });
    const mulNode  = plusBlock (svg, "mul", cx, 210, "×");
    const silu     = rectBlock(svg, "silu",      cx - 240, 300, 180, 50, actName,
                               { fill: C.norm, role: "act" });
    const upProj   = rectBlock(svg, "up_proj",   cx + 60,  300, 180, 50, "Linear (up)",
                               { fill: C.ffn, role: "ffn" });
    const gateProj = rectBlock(svg, "gate_proj", cx - 240, 420, 180, 50, "Linear (gate)",
                               { fill: C.ffn, role: "ffn" });

    vLine(svg, mulNode, downProj);
    elbowPath(svg, silu.cx, silu.top - GAP, mulNode.cx - mulNode.r - GAP, mulNode.cy);
    elbowPath(svg, upProj.cx, upProj.top - GAP, mulNode.cx + mulNode.r + GAP, mulNode.cy);
    vLine(svg, gateProj, silu);

    const inputY = H - 35;
    const branchY = inputY - 22;
    svg.appendChild(el("circle", { cx, cy: branchY, r: 3, fill: C.arrow }));
    svg.appendChild(el("path", {
      d: `M ${cx} ${inputY} L ${cx} ${branchY}`,
      fill: "none", stroke: C.arrow, "stroke-width": 1.4, "stroke-linecap": "round"
    }));
    elbowPath(svg, cx, branchY, gateProj.cx, gateProj.bottom + GAP);
    elbowPath(svg, cx, branchY, upProj.cx, upProj.bottom + GAP);

    const inLabel = el("text", {
      x: cx, y: H - 14, "text-anchor": "middle",
      fill: C.text, "font-family": FONT_HEAD, "font-size": 16
    });
    inLabel.textContent = "x  (input)";
    svg.appendChild(inLabel);

    const dimLabel = el("text", {
      x: cx, y: 36, "text-anchor": "middle",
      fill: C.muted, "font-family": FONT_MONO, "font-size": 11
    });
    dimLabel.textContent = `intermediate hidden = ${fmtInt(ffn.expert_intermediate_size || ffn.intermediate_size)}`;
    svg.appendChild(dimLabel);

    return svg;
  }

  function buildLayerMap(ir, info) {
    const W = 720, H = 240;
    const svg = el("svg", { width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img" });
    defs(svg);
    svg.appendChild(regionRect(40, 30, W - 80, H - 60, C.bg_card, { stroke: C.border, strokeWidth: 0.5 }));

    const palette = ["#0D9488", "#D97706", "#7C3AED", "#0EA5E9", "#DC2626", "#65A30D", "#9333EA"];
    const sigToColor = {};
    info.groups.forEach((g, i) => { sigToColor[g.sig] = palette[i % palette.length]; });

    const stripX = 80, stripY = 90, stripW = W - 160, stripH = 36;
    const n = ir.layers.length;
    const colW = stripW / n;

    ir.layers.forEach((l, i) => {
      const sig = JSON.stringify([
        l.attention.kind, l.attention.mask, l.attention.window_size,
        l.ffn.kind, l.ffn.num_experts, l.norm_kind, l.norm_placement
      ]);
      svg.appendChild(el("rect", {
        x: stripX + i * colW, y: stripY,
        width: Math.max(colW - 0.5, 1), height: stripH,
        fill: sigToColor[sig], opacity: 0.95
      }));
    });

    svg.appendChild(el("rect", {
      x: stripX, y: stripY, width: stripW, height: stripH,
      fill: "none", stroke: C.text, "stroke-width": 0.4, rx: 4, ry: 4
    }));

    // axis ticks: 0 and N-1
    [0, n - 1].forEach(idx => {
      const x = stripX + (idx + 0.5) * colW;
      const t = el("text", {
        x, y: stripY + stripH + 16, "text-anchor": "middle",
        fill: C.muted, "font-family": FONT_MONO, "font-size": 10
      });
      t.textContent = `L${idx}`;
      svg.appendChild(t);
    });

    const title = el("text", {
      x: stripX, y: 70, fill: C.text,
      "font-family": FONT_BODY, "font-size": 12, "font-weight": 600
    });
    title.textContent = `${n} layers · ${info.groups.length} ${info.groups.length === 1 ? "type" : "types"}`;
    svg.appendChild(title);

    let lx = stripX, ly = stripY + stripH + 44;
    info.groups.forEach((g) => {
      const ffnK = g.spec.ffn.kind === "moe" ? "MoE" : "Dense";
      const mask = g.spec.attention.mask === "sliding" ? "SWA" : "full";
      const labelTxt = `${g.spec.attention.kind.toUpperCase()} + ${ffnK} (${mask}) · L${g.indices[0]}–L${g.indices[g.indices.length - 1]} · ${g.indices.length}×`;
      svg.appendChild(el("rect", { x: lx, y: ly - 9, width: 12, height: 12, fill: sigToColor[g.sig], rx: 2 }));
      const txt = el("text", {
        x: lx + 18, y: ly, "dominant-baseline": "central",
        fill: C.text, "font-family": FONT_BODY, "font-size": 12
      });
      txt.textContent = labelTxt;
      svg.appendChild(txt);
      ly += 20;
    });

    return svg;
  }

  // ── chrome around the canvas: stats banner, tabs, info panel ─────────────
  function statsBanner(ir) {
    const root = document.createElement("div");
    const params = ir.params || {};
    const items = [
      ["Layers",   String(ir.layers.length)],
      ["Hidden",   fmtInt(ir.hidden_size)],
      ["Vocab",    fmtInt(ir.vocab_size)],
      ["Context",  ir.max_position_embeddings ? fmtInt(ir.max_position_embeddings) : "—"],
      ["Params",   params.is_sparse
                   ? `${params.total_h} (${params.active_h} active)`
                   : (params.total_h || "?")],
    ];
    root.style.cssText = `
      display:grid;grid-template-columns:repeat(${items.length},minmax(0,1fr));
      gap:1px;background:${C.border};border:0.5px solid ${C.border};
      border-radius:10px;overflow:hidden;margin-bottom:14px;
    `;
    items.forEach(([k, v]) => {
      const cell = document.createElement("div");
      cell.style.cssText = `padding:10px 14px;background:${C.bg_card};`;
      const k1 = document.createElement("div");
      k1.style.cssText = `font-size:10px;letter-spacing:0.12em;color:${C.muted};font-weight:600;`;
      k1.textContent = k.toUpperCase();
      const v1 = document.createElement("div");
      v1.style.cssText = `font-family:${FONT_HEAD};font-size:22px;color:${C.text};margin-top:2px;line-height:1.05;`;
      v1.textContent = v;
      cell.appendChild(k1); cell.appendChild(v1);
      root.appendChild(cell);
    });
    return root;
  }

  function badgeRow(badges) {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;";
    badges.forEach(b => {
      const bg  = b.tone === "warm" ? C.badge_warm
                : b.tone === "cool" ? C.badge_cool
                : C.badge_bg;
      const fg  = b.tone === "warm" ? C.badge_warm_t
                : b.tone === "cool" ? C.badge_cool_t
                : C.badge_text;
      const span = document.createElement("span");
      span.title = b.title || "";
      span.style.cssText = `
        display:inline-flex;align-items:center;height:22px;padding:0 9px;
        background:${bg};color:${fg};border-radius:11px;
        font-size:11px;font-weight:600;letter-spacing:0.02em;
      `;
      span.textContent = b.text;
      row.appendChild(span);
    });
    return row;
  }

  function render(ir, mount) {
    const info = makeInfo(ir);

    mount.innerHTML = "";
    mount.style.fontFamily = FONT_BODY;
    mount.style.color = C.text;

    // header: name + badges
    const head = document.createElement("div");
    head.style.cssText = "margin-bottom:12px;";
    const name = document.createElement("div");
    name.style.cssText = `font-family:${FONT_HEAD};font-size:30px;line-height:1;color:${C.text};`;
    name.textContent = ir.name;
    head.appendChild(name);
    const sub = document.createElement("div");
    sub.style.cssText = `color:${C.muted};font-size:12px;margin-top:4px;font-family:${FONT_MONO};`;
    sub.textContent = ir.architecture;
    head.appendChild(sub);
    head.appendChild(badgeRow(archBadges(ir, info)));
    mount.appendChild(head);

    // stats grid
    mount.appendChild(statsBanner(ir));

    // tab strip
    const tabs = document.createElement("div");
    tabs.style.cssText = `
      display:flex;gap:4px;margin-bottom:10px;padding:4px;background:${C.bg_card};
      border:0.5px solid ${C.border};border-radius:9px;width:fit-content;
    `;
    const tabDefs = [
      { v: "arch", label: "Architecture", enabled: true },
      { v: "moe",  label: "MoE",          enabled: info.dominant.spec.ffn.kind === "moe" },
      { v: "ffn",  label: info.dominant.spec.ffn.kind === "moe" ? "Expert" : "FFN", enabled: true },
      { v: "map",  label: "Layer map",    enabled: true },
    ];
    const tabBtns = [];
    tabDefs.forEach(td => {
      const b = document.createElement("button");
      b.dataset.view = td.v;
      b.textContent = td.label;
      b.disabled = !td.enabled;
      b.className = "uf-tab";
      tabs.appendChild(b);
      tabBtns.push(b);
    });
    mount.appendChild(tabs);

    // styles, scoped via the mount id
    const style = document.createElement("style");
    style.textContent = `
      .uf-tab {
        appearance:none;border:0;background:transparent;color:${C.muted};
        padding:7px 14px;border-radius:6px;font:600 12px ${FONT_BODY};
        cursor:pointer;transition:background .15s,color .15s;
      }
      .uf-tab:hover:not(:disabled) { background:#F1F5F9;color:${C.text}; }
      .uf-tab.uf-active { background:${C.text};color:#FFFFFF; }
      .uf-tab:disabled { opacity:.4;cursor:not-allowed; }
      .uf-node rect, .uf-node circle { transition: filter .15s, transform .15s; transform-origin: center; transform-box: fill-box; }
      .uf-node:hover rect, .uf-node:hover circle { filter: brightness(1.08) drop-shadow(0 2px 4px rgba(0,0,0,.18)); }
      .uf-node.uf-selected rect, .uf-node.uf-selected circle { stroke: #FACC15 !important; stroke-width: 2.5 !important; }
      .uf-canvas svg { display:block; max-width:100%; height:auto; animation: uf-fade .35s ease-out; }
      @keyframes uf-fade { from { opacity:0; transform:translateY(4px); } to { opacity:1; transform:none; } }
    `;
    mount.appendChild(style);

    const canvas = document.createElement("div");
    canvas.className = "uf-canvas";
    canvas.style.cssText = `
      background:${C.canvas};border:0.5px solid ${C.border};
      border-radius:12px;padding:8px;
    `;
    mount.appendChild(canvas);

    const panel = document.createElement("div");
    panel.style.cssText = `
      margin-top:10px;padding:12px 14px;background:${C.bg_card};
      border:0.5px solid ${C.border};border-radius:10px;
      font-size:13px;color:${C.muted};min-height:42px;line-height:1.45;
    `;
    panel.innerHTML = `<span style="color:${C.text}">Click any block</span> to inspect its dimensions and role.`;
    mount.appendChild(panel);

    function attachClicks(svg) {
      svg.querySelectorAll(".uf-node").forEach(node => {
        node.addEventListener("click", () => {
          svg.querySelectorAll(".uf-node.uf-selected").forEach(n => n.classList.remove("uf-selected"));
          node.classList.add("uf-selected");
          const id = node.getAttribute("data-id");
          const meta = info.meta[id];
          if (meta) {
            panel.innerHTML =
              `<span style="color:${C.text};font-weight:600">${meta[0]}</span>` +
              `<span style="margin:0 8px;color:${C.border}">|</span>` +
              `<span>${meta[1]}</span>`;
          }
        });
      });
    }

    function show(view) {
      canvas.innerHTML = "";
      let svg;
      if (view === "moe" && info.dominant.spec.ffn.kind === "moe") svg = buildMoeView(ir, info);
      else if (view === "ffn") svg = buildFfnView(ir, info);
      else if (view === "map") svg = buildLayerMap(ir, info);
      else svg = buildArchitectureView(ir, info);
      canvas.appendChild(svg);
      attachClicks(svg);
      tabBtns.forEach(b => b.classList.toggle("uf-active", b.dataset.view === view));
    }

    tabBtns.forEach(b => b.addEventListener("click", () => {
      if (!b.disabled) show(b.dataset.view);
    }));

    show("arch");
  }

  global.Unfold = { render };
})(typeof window !== "undefined" ? window : this);
