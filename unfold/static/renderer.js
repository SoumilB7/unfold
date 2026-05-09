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

  // unified mint/green theme
  const C = {
    bg_outer:    "#E1F5EE",   // pale mint  (outer card)
    bg_inner:    "#9FE1CB",   // mint       (inner "transformer block" card)
    bg_card:     "#FFFFFF",
    canvas:      "#F4FBF8",

    block:       "#0F6E56",   // unified dark green for all blocks
    block_alt:   "#0E5C48",   // darker shade for stroke
    text_block:  "#FFFFFF",

    arrow:       "#0F6E56",
    text:        "#04342C",
    muted:       "#5F7C73",
    border:      "#B6DDCB",

    badge_bg:    "#D6F1E4",
    badge_text:  "#0E5C48",
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

    const f = el("filter", { id: "uf-shadow", x: "-20%", y: "-20%", width: "140%", height: "140%" });
    f.appendChild(el("feGaussianBlur", { in: "SourceAlpha", stdDeviation: 1 }));
    f.appendChild(el("feOffset", { dx: 0, dy: 1, result: "off" }));
    const ct = el("feComponentTransfer");
    const fa = el("feFuncA", { type: "linear", slope: "0.16" });
    ct.appendChild(fa); f.appendChild(ct);
    const m2 = el("feMerge");
    m2.appendChild(el("feMergeNode"));
    m2.appendChild(el("feMergeNode", { in: "SourceGraphic" }));
    f.appendChild(m2);
    d.appendChild(f);

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
    const fontSize = opts.fontSize || 18;
    const g = el("g", { class: "uf-node", "data-id": id });
    g.setAttribute("style", "cursor:pointer;");
    g.appendChild(el("rect", {
      x, y, width: w, height: h, rx: 11, ry: 11,
      fill: C.block,
      stroke: C.block_alt, "stroke-width": 0.6,
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
    svg.appendChild(g);
    return {
      el: g, kind: "rect",
      left: x, right: x + w, top: y, bottom: y + h,
      cx: x + w / 2, cy: y + h / 2, w, h
    };
  }

  function plusBlock(svg, id, cx, cy, sym) {
    const r = 14;
    const g = el("g", { class: "uf-node", "data-id": id, style: "cursor:pointer;" });
    g.appendChild(el("circle", {
      cx, cy, r, fill: C.block, stroke: C.block_alt, "stroke-width": 0.6,
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

  // ── arrow routing convention ────────────────────────────────────────────
  // START coordinate is exactly ON the source block's edge (the line touches
  // the block).  END coordinate is offset by GAP from the destination's
  // edge, so the arrowhead sits in the gap with room to breathe and visibly
  // points INTO the destination.

  // Straight vertical arrow connecting two stacked blocks.  Auto-detects
  // whether `from` is above or below `to` based on cy.
  function vLine(svg, fromNode, toNode) {
    let y1, y2;
    if (fromNode.cy > toNode.cy) {        // from is below; arrow goes UP
      y1 = fromNode.top;
      y2 = toNode.bottom + GAP;
    } else {                              // from is above; arrow goes DOWN
      y1 = fromNode.bottom;
      y2 = toNode.top - GAP;
    }
    svg.appendChild(el("line", {
      x1: fromNode.cx, y1, x2: fromNode.cx, y2,
      stroke: C.arrow, "stroke-width": 1.6, "stroke-linecap": "round",
      "marker-end": "url(#uf-arrow)", fill: "none"
    }));
  }

  // Straight vertical arrow at column `x`, from y1 to y2 (caller passes
  // already-correct edge coordinates).
  function vSeg(svg, x, y1, y2) {
    svg.appendChild(el("line", {
      x1: x, y1, x2: x, y2,
      stroke: C.arrow, "stroke-width": 1.6, "stroke-linecap": "round",
      "marker-end": "url(#uf-arrow)", fill: "none"
    }));
  }

  // VH elbow: VERTICAL first, then HORIZONTAL.  Arrow ENDS horizontal —
  // the arrowhead enters its target from the side.  Use this for
  // "branch into the side of a +/× node" wires.
  function elbowVH(svg, x1, y1, x2, y2) {
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
      "stroke-width": 1.6, "stroke-linecap": "round", "stroke-linejoin": "round",
      "marker-end": "url(#uf-arrow)"
    }));
  }

  // HV elbow: HORIZONTAL first, then VERTICAL.  Arrow ENDS vertical —
  // the arrowhead enters its target from the top or bottom.  Use this for
  // "fanned-out branch into the bottom of a block" wires.
  function elbowHV(svg, x1, y1, x2, y2) {
    let d;
    if (Math.abs(x2 - x1) < 1 || Math.abs(y2 - y1) < 1) {
      d = `M ${x1} ${y1} L ${x2} ${y2}`;
    } else {
      const sx = Math.sign(x2 - x1);
      const sy = Math.sign(y2 - y1);
      const r = Math.min(10, Math.abs(x2 - x1) / 2, Math.abs(y2 - y1) / 2);
      d = `M ${x1} ${y1} ` +
          `L ${x2 - sx * r} ${y1} ` +
          `Q ${x2} ${y1} ${x2} ${y1 + sy * r} ` +
          `L ${x2} ${y2}`;
    }
    svg.appendChild(el("path", {
      d, fill: "none", stroke: C.arrow,
      "stroke-width": 1.6, "stroke-linecap": "round", "stroke-linejoin": "round",
      "marker-end": "url(#uf-arrow)"
    }));
  }

  // residual loop: out the right side of a block, up and around, back into
  // the right side of a + node above.
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
      "stroke-width": 1.6, "stroke-linecap": "round", "stroke-linejoin": "round",
      "marker-end": "url(#uf-arrow)"
    }));
  }

  // ── description helpers (used by info panel) ───────────────────────────
  function describeAttention(a) {
    if (a.kind === "mla") {
      return `Multi-head latent attention · ${a.num_heads} heads · KV LoRA ${fmtInt(a.kv_lora_rank)}` +
             (a.q_lora_rank ? ` · Q LoRA ${fmtInt(a.q_lora_rank)}` : "");
    }
    if (a.kind === "gqa") {
      return `Grouped-query · ${a.num_heads} Q / ${a.num_kv_heads} KV heads · head dim ${fmtInt(a.head_dim)}`;
    }
    if (a.kind === "mqa") {
      return `Multi-query · ${a.num_heads} Q / 1 KV head`;
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

    if (a.kind === "mla") badges.push({ text: "MLA", title: "Multi-head latent attention" });
    else if (a.kind === "gqa") badges.push({ text: `GQA ${a.num_heads}/${a.num_kv_heads}`, title: "Grouped-query attention" });
    else if (a.kind === "mqa") badges.push({ text: "MQA", title: "Multi-query attention" });
    else badges.push({ text: "MHA", title: "Multi-head attention" });

    if (f.kind === "moe") {
      badges.push({
        text: `MoE ${f.num_experts_per_tok}/${f.num_experts}`,
        title: `Mixture of experts — top-${f.num_experts_per_tok} of ${f.num_experts}`
      });
    } else {
      badges.push({ text: "Dense FFN", title: "Dense feed-forward" });
    }
    if (info.groups.length > 1) badges.push({ text: `${info.groups.length} layer types` });
    if (a.mask === "sliding") badges.push({ text: `SWA ${fmtInt(a.window_size)}`, title: "Sliding-window attention" });
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
      groups, dominant,
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

  // ── architecture view ───────────────────────────────────────────────────
  function buildArchitectureView(ir, info) {
    const W = 720, H = 1080;
    const svg = el("svg", { width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img" });
    const t = el("title"); t.textContent = `${ir.name} architecture`; svg.appendChild(t);
    defs(svg);

    svg.appendChild(regionRect(40, 30, W - 80, H - 60, C.bg_outer));

    const cx = W / 2;
    const innerX = 110, innerY = 240, innerW = W - 220, innerH = 580;
    svg.appendChild(regionRect(innerX, innerY, innerW, innerH, C.bg_inner));

    const a = info.dominant.spec.attention;
    const f = info.dominant.spec.ffn;

    const attnLabel = a.kind === "mla" ? ["Multi-Head Latent", "Attention"]
                   : a.kind === "gqa" ? ["Grouped-Query", "Attention"]
                   : a.kind === "mqa" ? ["Multi-Query", "Attention"]
                   : ["Multi-Head", "Attention"];

    const ffnLabel = f.kind === "moe" ? "MoE" : "Feed-Forward";

    const tokText  = rectBlock(svg, "tok_text",  cx - 110, H - 120, 220, 48, "Tokenized text", { fontSize: 18 });
    const embed    = rectBlock(svg, "embed",     cx - 130, H - 200, 260, 48, "Token Embedding layer");

    const rms1     = rectBlock(svg, "rms1",      cx - 80,  innerY + 470, 160, 40, "RMSNorm", { fontSize: 18 });
    const attn     = rectBlock(svg, "attn",      cx - 115, innerY + 360, 230, 70, attnLabel, { fontSize: 18 });
    const add1     = plusBlock (svg, "add1",     cx,       innerY + 320);
    const rms2     = rectBlock(svg, "rms2",      cx - 80,  innerY + 230, 160, 40, "RMSNorm", { fontSize: 18 });
    const ffn      = rectBlock(svg, "ffn",       cx - 80,  innerY + 130, 160, 50, ffnLabel);
    const add2     = plusBlock (svg, "add2",     cx,       innerY + 90);

    const finalRms = rectBlock(svg, "final_rms", cx - 90,  170, 180, 40, "Final RMSNorm", { fontSize: 18 });
    const lmHead   = rectBlock(svg, "lm_head",   cx - 130, 90,  260, 48, "Linear output layer");

    // wires bottom-up
    vLine(svg, tokText, embed);
    vLine(svg, embed, rms1);
    vLine(svg, rms1, attn);
    vLine(svg, attn, add1);
    vLine(svg, add1, rms2);
    vLine(svg, rms2, ffn);
    vLine(svg, ffn, add2);
    vLine(svg, add2, finalRms);
    vLine(svg, finalRms, lmHead);

    // arrow leaving the LM head upward
    svg.appendChild(el("line", {
      x1: cx, y1: lmHead.top, x2: cx, y2: lmHead.top - 32,
      stroke: C.arrow, "stroke-width": 1.6, "stroke-linecap": "round",
      "marker-end": "url(#uf-arrow)", fill: "none"
    }));

    // residual loops on the right
    const lane = innerX + innerW - 28;
    residualLoopRight(svg, rms1, add1, lane);
    residualLoopRight(svg, rms2, add2, lane);

    // "× N" label, top-right of inner card
    const repeatBg = el("rect", {
      x: innerX + innerW - 78, y: innerY + 12, width: 66, height: 26,
      rx: 13, ry: 13, fill: "rgba(255,255,255,0.65)",
      stroke: C.border, "stroke-width": 0.5
    });
    svg.appendChild(repeatBg);
    const repeat = el("text", {
      x: innerX + innerW - 45, y: innerY + 25, "text-anchor": "middle",
      "dominant-baseline": "central",
      fill: C.text, "font-family": FONT_HEAD, "font-size": 20
    });
    repeat.textContent = `× ${ir.layers.length}`;
    svg.appendChild(repeat);

    return svg;
  }

  // ── MoE view ────────────────────────────────────────────────────────────
  function buildMoeView(ir, info) {
    const W = 720, H = 580;
    const svg = el("svg", { width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img" });
    defs(svg);
    svg.appendChild(regionRect(40, 30, W - 80, H - 60, C.bg_outer));

    const ffn = info.dominant.spec.ffn;
    const cx = W / 2;
    const router = rectBlock(svg, "router", cx - 80, H - 110, 160, 50, "Router");
    const sumNode = plusBlock(svg, "add_moe", cx, 80);

    const expertY = 240, expertW = 130, expertH = 60;
    const slots = [
      { x: 110,                label: ["Feed forward", "(expert 1)"],   id: "expert_1" },
      { x: 270,                label: ["Feed forward", "(expert k)"],   id: "expert_k" },
      { x: 430,                label: ["Feed forward", "(expert k+1)"], id: "expert_kp1" },
      { x: W - 110 - expertW,  label: ["Feed forward", `(expert ${ffn.num_experts || "N"})`], id: "expert_n" }
    ];
    const experts = slots.map(s => rectBlock(svg, s.id, s.x, expertY, expertW, expertH, s.label, { fontSize: 14 }));

    // dots between expert k and k+1
    const dotsX = (experts[1].right + experts[2].left) / 2;
    const dotsY = expertY + expertH / 2;
    for (let i = -2; i <= 2; i++) {
      svg.appendChild(el("circle", { cx: dotsX + i * 7, cy: dotsY, r: 2.5, fill: C.muted }));
    }

    const annot = el("text", {
      x: experts[3].right, y: experts[3].bottom + 22, "text-anchor": "end",
      fill: C.text, "font-family": FONT_HEAD, "font-size": 18, "font-style": "italic"
    });
    annot.textContent = `(${ffn.num_experts || "N"})`;
    svg.appendChild(annot);

    // router → each expert
    // (start at router.top, end one GAP below each expert)
    experts.forEach(e => {
      vSeg(svg, e.cx, router.top, e.bottom + GAP);
    });

    // each expert → sum  (start at expert.top, end at sum's left/right edge)
    experts.forEach(e => {
      const sx = sumNode.cx;
      const targetX = sx + (e.cx < sx ? -sumNode.r - GAP : sumNode.r + GAP);
      elbowVH(svg, e.cx, e.top, targetX, sumNode.cy);
    });

    // top-k callout
    if (ffn.num_experts && ffn.num_experts_per_tok) {
      const sparsity = (100 * ffn.num_experts_per_tok / ffn.num_experts).toFixed(1);
      const cgX = W - 224, cgY = 56, cgW = 184, cgH = 56;
      svg.appendChild(el("rect", {
        x: cgX, y: cgY, width: cgW, height: cgH, rx: 10, ry: 10,
        fill: C.bg_card, stroke: C.border, "stroke-width": 0.5
      }));
      const lbl = el("text", {
        x: cgX + 12, y: cgY + 18, fill: C.muted,
        "font-family": FONT_BODY, "font-size": 10,
        "letter-spacing": "0.12em", "font-weight": 600
      });
      lbl.textContent = "ACTIVE PER TOKEN";
      svg.appendChild(lbl);
      const big = el("text", {
        x: cgX + 12, y: cgY + 44, fill: C.text,
        "font-family": FONT_HEAD, "font-size": 22
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

  // ── FFN view (gated SwiGLU) ─────────────────────────────────────────────
  function buildFfnView(ir, info) {
    const W = 720, H = 600;
    const svg = el("svg", { width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img" });
    defs(svg);
    svg.appendChild(regionRect(40, 30, W - 80, H - 60, C.bg_outer));

    const ffn = info.dominant.spec.ffn;
    const cx = W / 2;
    const actName = (ffn.activation || "silu").toUpperCase();

    // layout (top → bottom)
    const downProj = rectBlock(svg, "down_proj", cx - 90, 80,  180, 50, "Linear (down)");
    const mulNode  = plusBlock (svg, "mul",      cx, 200, "×");
    const silu     = rectBlock(svg, "silu",      cx - 270, 300, 180, 50, actName);
    const upProj   = rectBlock(svg, "up_proj",   cx + 90,  300, 180, 50, "Linear (up)");
    const gateProj = rectBlock(svg, "gate_proj", cx - 270, 430, 180, 50, "Linear (gate)");

    // input branch dot at bottom-center
    const branchY = H - 70;
    svg.appendChild(el("circle", { cx, cy: branchY, r: 4, fill: C.arrow }));

    // ── arrows, all carefully routed ───────────────────────────────────────
    // input → gate  (HV: go left, then up; arrowhead enters gate from below)
    elbowHV(svg, cx, branchY, gateProj.cx, gateProj.bottom + GAP);
    // input → up    (HV: go right, then up; arrowhead enters up from below)
    elbowHV(svg, cx, branchY, upProj.cx, upProj.bottom + GAP);

    // gate → silu  (vertical; arrowhead enters silu from below)
    vLine(svg, gateProj, silu);

    // silu → ×   (VH: up, then right; arrowhead enters × from the LEFT side)
    // Start touches silu's top edge.
    elbowVH(svg, silu.cx, silu.top, mulNode.cx - mulNode.r - GAP, mulNode.cy);
    // up → ×     (VH: up, then left; arrowhead enters × from the RIGHT side)
    elbowVH(svg, upProj.cx, upProj.top, mulNode.cx + mulNode.r + GAP, mulNode.cy);

    // × → down  (vertical; arrowhead enters down from below)
    vLine(svg, mulNode, downProj);

    // arrow leaving down upward
    svg.appendChild(el("line", {
      x1: cx, y1: downProj.top, x2: cx, y2: downProj.top - 32,
      stroke: C.arrow, "stroke-width": 1.6, "stroke-linecap": "round",
      "marker-end": "url(#uf-arrow)", fill: "none"
    }));

    // labels
    const inLabel = el("text", {
      x: cx, y: H - 22, "text-anchor": "middle",
      fill: C.text, "font-family": FONT_HEAD, "font-size": 18
    });
    inLabel.textContent = "x  (input)";
    svg.appendChild(inLabel);

    const dimLabel = el("text", {
      x: cx, y: 50, "text-anchor": "middle",
      fill: C.muted, "font-family": FONT_MONO, "font-size": 11
    });
    dimLabel.textContent = `intermediate hidden = ${fmtInt(ffn.expert_intermediate_size || ffn.intermediate_size)}`;
    svg.appendChild(dimLabel);

    return svg;
  }

  // ── layer map ───────────────────────────────────────────────────────────
  function buildLayerMap(ir, info) {
    const W = 720, H = 240;
    const svg = el("svg", { width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img" });
    defs(svg);
    svg.appendChild(regionRect(40, 30, W - 80, H - 60, C.bg_card, { stroke: C.border, strokeWidth: 0.5 }));

    // green-toned palette for the RLE stripes (so layer types are still
    // distinguishable but stay within the theme)
    const palette = ["#0F6E56", "#1D9E75", "#0E7C8C", "#3C3489", "#993C1D", "#185FA5", "#65A30D"];
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

  // ── chrome (single outer card holding everything) ───────────────────────
  function statsBanner(ir) {
    const root = document.createElement("div");
    const params = ir.params || {};
    const items = [
      ["Layers",  String(ir.layers.length)],
      ["Hidden",  fmtInt(ir.hidden_size)],
      ["Vocab",   fmtInt(ir.vocab_size)],
      ["Context", ir.max_position_embeddings ? fmtInt(ir.max_position_embeddings) : "—"],
      ["Params",  params.is_sparse
                  ? `${params.total_h} (${params.active_h} act.)`
                  : (params.total_h || "?")],
    ];
    root.style.cssText = `
      display:grid;grid-template-columns:repeat(${items.length},minmax(0,1fr));
      gap:1px;background:${C.border};border:0.5px solid ${C.border};
      border-radius:8px;overflow:hidden;margin-bottom:18px;
    `;
    items.forEach(([k, v]) => {
      const cell = document.createElement("div");
      cell.style.cssText = `padding:8px 12px;background:${C.bg_card};`;
      const k1 = document.createElement("div");
      k1.style.cssText = `font-size:9.5px;letter-spacing:0.12em;color:${C.muted};font-weight:600;`;
      k1.textContent = k.toUpperCase();
      const v1 = document.createElement("div");
      v1.style.cssText = `font-family:${FONT_HEAD};font-size:19px;color:${C.text};margin-top:2px;line-height:1.05;`;
      v1.textContent = v;
      cell.appendChild(k1); cell.appendChild(v1);
      root.appendChild(cell);
    });
    return root;
  }

  function section(label, sub, svg) {
    const s = document.createElement("div");
    s.className = "uf-section";
    s.style.cssText = "margin-top:18px;";
    const head = document.createElement("div");
    head.style.cssText = `display:flex;align-items:baseline;gap:10px;margin-bottom:6px;`;
    const l = document.createElement("span");
    l.style.cssText = `font-size:10.5px;letter-spacing:0.14em;font-weight:700;color:${C.text};`;
    l.textContent = label.toUpperCase();
    head.appendChild(l);
    if (sub) {
      const su = document.createElement("span");
      su.style.cssText = `font-size:11px;color:${C.muted};`;
      su.textContent = sub;
      head.appendChild(su);
    }
    s.appendChild(head);
    const body = document.createElement("div");
    body.className = "uf-svg";
    body.style.cssText = `background:${C.canvas};border:0.5px solid ${C.border};border-radius:10px;padding:6px;`;
    body.appendChild(svg);
    s.appendChild(body);
    return s;
  }

  function badgeRow(badges) {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;";
    badges.forEach(b => {
      const span = document.createElement("span");
      span.title = b.title || "";
      span.style.cssText = `
        display:inline-flex;align-items:center;height:22px;padding:0 9px;
        background:${C.badge_bg};color:${C.badge_text};border-radius:11px;
        font-size:11px;font-weight:600;letter-spacing:0.02em;
      `;
      span.textContent = b.text;
      row.appendChild(span);
    });
    return row;
  }

  // Load Caveat from Google Fonts so the handwritten labels actually render
  // as Caveat instead of falling back to Comic Sans MS.
  function ensureFonts() {
    if (typeof document === "undefined" || document.getElementById("uf-fonts")) return;
    const link = document.createElement("link");
    link.id = "uf-fonts";
    link.rel = "stylesheet";
    link.href = "https://fonts.googleapis.com/css2?family=Caveat:wght@500;700&display=swap";
    document.head.appendChild(link);
  }

  function render(ir, mount) {
    ensureFonts();
    const info = makeInfo(ir);
    const isMoe = info.dominant.spec.ffn.kind === "moe";

    mount.innerHTML = "";
    mount.style.fontFamily = FONT_BODY;
    mount.style.color = C.text;
    mount.style.maxWidth = "720px";

    // single outer card holds everything
    const card = document.createElement("div");
    card.style.cssText = `
      background:${C.bg_card};
      border:0.5px solid ${C.border};
      border-radius:14px;
      padding:22px 24px 20px;
      box-shadow:0 1px 3px rgba(15,23,42,0.04);
    `;
    mount.appendChild(card);

    // ── stylesheet (scoped via class names) ──
    const style = document.createElement("style");
    style.textContent = `
      .uf-node rect, .uf-node circle { transition: filter .15s; }
      .uf-node:hover rect, .uf-node:hover circle { filter: brightness(1.08) drop-shadow(0 2px 4px rgba(0,0,0,.18)); }
      .uf-node.uf-selected rect, .uf-node.uf-selected circle { stroke: #FACC15 !important; stroke-width: 2.5 !important; }
      .uf-svg svg { display:block; max-width:100%; height:auto; animation: uf-fade .3s ease-out; }
      @keyframes uf-fade { from { opacity:0; transform:translateY(2px); } to { opacity:1; transform:none; } }
    `;
    mount.appendChild(style);

    // header: name, architecture, badges
    const head = document.createElement("div");
    head.style.cssText = "margin-bottom:14px;";
    const name = document.createElement("div");
    name.style.cssText = `font-family:${FONT_HEAD};font-size:26px;line-height:1;color:${C.text};`;
    name.textContent = ir.name;
    head.appendChild(name);
    const sub = document.createElement("div");
    sub.style.cssText = `color:${C.muted};font-size:11px;margin-top:3px;font-family:${FONT_MONO};`;
    sub.textContent = ir.architecture;
    head.appendChild(sub);
    head.appendChild(badgeRow(archBadges(ir, info)));
    card.appendChild(head);

    // stats grid
    card.appendChild(statsBanner(ir));

    // ── stacked sections (no tabs) ──
    card.appendChild(section(
      "Architecture",
      `Per-layer block · repeats × ${ir.layers.length}`,
      buildArchitectureView(ir, info)
    ));

    if (isMoe) {
      card.appendChild(section(
        "Mixture of experts",
        info.dominant.spec.ffn.num_experts && info.dominant.spec.ffn.num_experts_per_tok
          ? `top-${info.dominant.spec.ffn.num_experts_per_tok} of ${info.dominant.spec.ffn.num_experts} active per token`
          : "router → top-k experts → weighted sum",
        buildMoeView(ir, info)
      ));
    }

    card.appendChild(section(
      isMoe ? "Expert FFN" : "FFN block",
      `Gated ${(info.dominant.spec.ffn.activation || "silu").toUpperCase()} · hidden ` +
      fmtInt(info.dominant.spec.ffn.expert_intermediate_size || info.dominant.spec.ffn.intermediate_size),
      buildFfnView(ir, info)
    ));

    card.appendChild(section(
      "Layer map",
      info.groups.length === 1
        ? "All layers structurally identical"
        : `${info.groups.length} layer types across ${ir.layers.length} layers`,
      buildLayerMap(ir, info)
    ));

    // info panel — updates when any block in any view is clicked
    const panel = document.createElement("div");
    panel.style.cssText = `
      margin-top:18px;padding:11px 14px;background:${C.canvas};
      border:0.5px solid ${C.border};border-radius:9px;
      font-size:12.5px;color:${C.muted};min-height:38px;line-height:1.45;
    `;
    panel.innerHTML = `<span style="color:${C.text}">Click any block</span> to inspect its dimensions and role.`;
    card.appendChild(panel);

    // wire clicks across every section's SVG; selecting a node deselects
    // siblings within its own SVG (so each diagram has its own selection)
    card.querySelectorAll(".uf-svg svg").forEach(svg => {
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
    });
  }

  global.Unfold = { render };
})(typeof window !== "undefined" ? window : this);
