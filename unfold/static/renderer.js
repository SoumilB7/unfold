/**
 * transformer_viz renderer.
 * Consumes an IR JSON and renders an interactive architecture diagram.
 *
 * Public API (attached to window):
 *   TransformerViz.render(ir, mountElement)
 *
 * The IR shape is documented in transformer_viz/ir.py.
 */
(function (global) {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const FONT = '"Caveat","Patrick Hand","Comic Sans MS",cursive';
  const GAP = 5;

  const C = {
    bg_outer: "#E1F5EE",
    bg_inner: "#9FE1CB",
    block: "#0F6E56",
    stroke: "#085041",
    text_on_block: "#FFFFFF",
    arrow: "#0F6E56",
    canvas_text: "#04342C",
    panel: "#F4FBF8",
    panel_border: "#9FE1CB",
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

  function defs() {
    const d = el("defs");
    const m = el("marker", {
      id: "tv-arrow", viewBox: "0 0 10 10", refX: 8, refY: 5,
      markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse"
    });
    m.appendChild(el("path", {
      d: "M2 1L8 5L2 9", fill: "none", stroke: "context-stroke",
      "stroke-width": 1.5, "stroke-linecap": "round", "stroke-linejoin": "round"
    }));
    d.appendChild(m);
    return d;
  }

  function regionRect(x, y, w, h, fill) {
    return el("rect", { x, y, width: w, height: h, rx: 18, ry: 18, fill });
  }

  function rectBlock(svg, id, x, y, w, h, label, opts) {
    opts = opts || {};
    const fontSize = opts.fontSize || 18;
    const g = el("g", { class: "tv-node", "data-id": id, style: "cursor:pointer;" });
    g.appendChild(el("rect", {
      x, y, width: w, height: h, rx: 12, ry: 12,
      fill: C.block, stroke: C.stroke, "stroke-width": 0.5
    }));
    const lines = Array.isArray(label) ? label : [label];
    const lineH = fontSize + 4;
    const startY = y + h / 2 - ((lines.length - 1) * lineH) / 2;
    lines.forEach((line, i) => {
      const t = el("text", {
        x: x + w / 2, y: startY + i * lineH,
        "text-anchor": "middle", "dominant-baseline": "central",
        fill: C.text_on_block, "font-family": FONT,
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
    const g = el("g", { class: "tv-node", "data-id": id, style: "cursor:pointer;" });
    g.appendChild(el("circle", {
      cx, cy, r, fill: C.block, stroke: C.stroke, "stroke-width": 0.5
    }));
    const t = el("text", {
      x: cx, y: cy, "text-anchor": "middle", "dominant-baseline": "central",
      fill: C.text_on_block, "font-family": FONT, "font-size": 22, "pointer-events": "none"
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

  function vLine(svg, fromNode, toNode) {
    svg.appendChild(el("line", {
      x1: fromNode.cx, y1: fromNode.bottom,
      x2: fromNode.cx, y2: toNode.top - GAP,
      stroke: C.arrow, "stroke-width": 1.6, "stroke-linecap": "round",
      "marker-end": "url(#tv-arrow)", fill: "none"
    }));
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
      "stroke-width": 1.6, "stroke-linecap": "round", "stroke-linejoin": "round",
      "marker-end": "url(#tv-arrow)"
    }));
  }

  function residualLoopRight(svg, fromNode, toNode, lane) {
    const r = 14;
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
      "marker-end": "url(#tv-arrow)"
    }));
  }

  function describeAttention(a) {
    if (a.kind === "mla") {
      return `Multi-head latent attention · ${a.num_heads} heads · KV LoRA ${fmtInt(a.kv_lora_rank)}` +
             (a.q_lora_rank ? ` · Q LoRA ${fmtInt(a.q_lora_rank)}` : "");
    }
    if (a.kind === "gqa") {
      return `Grouped-query attention · ${a.num_heads} Q heads / ${a.num_kv_heads} KV heads`;
    }
    if (a.kind === "mqa") {
      return `Multi-query attention · ${a.num_heads} Q heads / 1 KV head`;
    }
    return `Multi-head attention · ${a.num_heads} heads`;
  }

  function describeFFN(f) {
    if (f.kind === "moe") {
      return `MoE · ${fmtInt(f.num_experts)} experts · top-${f.num_experts_per_tok} per token` +
             (f.num_shared_experts ? ` · ${f.num_shared_experts} shared` : "") +
             ` · expert hidden ${fmtInt(f.expert_intermediate_size || f.intermediate_size)}`;
    }
    const gated = f.gated ? "gated " : "";
    return `Dense ${gated}FFN · activation ${f.activation} · hidden ${fmtInt(f.intermediate_size)}`;
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

    return {
      groups,
      dominant,
      meta: {
        tok_text: ["Tokenized text", "Input token IDs · [batch, seq_len]"],
        embed: ["Token embedding", `${fmtInt(ir.vocab_size)} × ${fmtInt(ir.hidden_size)}` +
                (ir.tie_word_embeddings ? " (tied with output)" : "")],
        rms1: ["Pre-attention norm", `dim ${fmtInt(ir.hidden_size)}`],
        attn: ["Attention", describeAttention(dominant.spec.attention)],
        add1: ["Residual add", "block input + attention output"],
        rms2: ["Pre-FFN norm", `dim ${fmtInt(ir.hidden_size)}`],
        ffn: [dominant.spec.ffn.kind === "moe" ? "Mixture of experts" : "Feed-forward",
              describeFFN(dominant.spec.ffn)],
        add2: ["Residual add", "post-attn + FFN output"],
        final_rms: ["Final norm", `dim ${fmtInt(ir.hidden_size)}`],
        lm_head: ["Linear output", `${fmtInt(ir.hidden_size)} → ${fmtInt(ir.vocab_size)}` +
                  (ir.tie_word_embeddings ? " (tied)" : "")],
      }
    };
  }

  function buildArchitectureView(ir, info) {
    const W = 680, H = 1020;
    const svg = el("svg", { width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img" });
    const t = el("title"); t.textContent = `${ir.name} architecture`; svg.appendChild(t);
    const dc = el("desc"); dc.textContent = `${ir.layers.length} layers`; svg.appendChild(dc);
    svg.appendChild(defs());

    const cx = W / 2;
    svg.appendChild(regionRect(80, 30, W - 160, H - 60, C.bg_outer));

    const innerX = 130, innerY = 250, innerW = W - 260, innerH = 540;
    svg.appendChild(regionRect(innerX, innerY, innerW, innerH, C.bg_inner));

    const ffnLabel = info.dominant.spec.ffn.kind === "moe" ? "MoE" : "FFN";
    const attnLabel = info.dominant.spec.attention.kind === "mla"
      ? ["Multi-head latent", "attention"]
      : info.dominant.spec.attention.kind === "gqa"
        ? ["Grouped-query", "attention"]
        : ["Multi-head", "attention"];

    const tokText  = rectBlock(svg, "tok_text",  cx - 110, 920, 220, 50, "Tokenized text");
    const embed    = rectBlock(svg, "embed",     cx - 130, 830, 260, 50, "Token embedding layer");

    const rms1     = rectBlock(svg, "rms1",      cx - 90,  innerY + 440, 180, 46, "RMSNorm");
    const attn     = rectBlock(svg, "attn",      cx - 110, innerY + 340, 220, 64, attnLabel);
    const add1     = plusBlock (svg, "add1",     cx,       innerY + 300);
    const rms2     = rectBlock(svg, "rms2",      cx - 90,  innerY + 210, 180, 46, "RMSNorm");
    const ffn      = rectBlock(svg, "ffn",       cx - 60,  innerY + 130, 120, 50, ffnLabel);
    const add2     = plusBlock (svg, "add2",     cx,       innerY + 80);

    const finalRms = rectBlock(svg, "final_rms", cx - 90,  150, 180, 46, "Final RMSNorm");
    const lmHead   = rectBlock(svg, "lm_head",   cx - 130, 70,  260, 50, "Linear output layer");

    vLine(svg, tokText, embed);
    vLine(svg, embed, rms1);
    vLine(svg, rms1, attn);
    vLine(svg, attn, add1);
    vLine(svg, add1, rms2);
    vLine(svg, rms2, ffn);
    vLine(svg, ffn, add2);
    vLine(svg, add2, finalRms);
    vLine(svg, finalRms, lmHead);

    svg.appendChild(el("line", {
      x1: cx, y1: lmHead.top, x2: cx, y2: lmHead.top - 28,
      stroke: C.arrow, "stroke-width": 1.6, "stroke-linecap": "round",
      "marker-end": "url(#tv-arrow)", fill: "none"
    }));

    const lane = innerX + innerW - 36;
    residualLoopRight(svg, rms1, add1, lane);
    residualLoopRight(svg, rms2, add2, lane);

    const repeat = el("text", {
      x: innerX + innerW - 14, y: innerY + 30, "text-anchor": "end",
      fill: C.canvas_text, "font-family": FONT, "font-size": 18, "font-style": "italic"
    });
    repeat.textContent = `× ${ir.layers.length}`;
    svg.appendChild(repeat);

    if (info.groups.length > 1) {
      const note = el("text", {
        x: innerX + 18, y: innerY + 30, "text-anchor": "start",
        fill: C.canvas_text, "font-family": FONT, "font-size": 14, "font-style": "italic"
      });
      note.textContent = `${info.groups.length} layer types — see Layer map`;
      svg.appendChild(note);
    }

    return svg;
  }

  function buildMoeView(ir, info) {
    const W = 680, H = 540;
    const svg = el("svg", { width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img" });
    svg.appendChild(defs());
    svg.appendChild(regionRect(80, 30, W - 160, H - 60, C.bg_outer));

    const ffn = info.dominant.spec.ffn;
    const cx = W / 2;
    const router = rectBlock(svg, "router", cx - 70, H - 110, 140, 50, "Router");
    const sumNode = plusBlock(svg, "add_moe", cx, 80);

    const expertY = 240, expertW = 130, expertH = 58;
    const slots = [
      { x: 110, label: ["Feed forward", "(expert 1)"],   id: "expert_1" },
      { x: 260, label: ["Feed forward", `(expert k)`],   id: "expert_k" },
      { x: 420, label: ["Feed forward", `(expert k+1)`], id: "expert_kp1" },
      { x: 570 - expertW, label: ["Feed forward", `(expert N)`], id: "expert_n" }
    ];
    const experts = slots.map(s => rectBlock(svg, s.id, s.x, expertY, expertW, expertH, s.label, { fontSize: 14 }));

    const dotsX = (experts[1].right + experts[2].left) / 2;
    const dotsY = expertY + expertH / 2;
    for (let i = -2; i <= 2; i++) {
      svg.appendChild(el("circle", { cx: dotsX + i * 7, cy: dotsY, r: 2.5, fill: C.block }));
    }

    const annot = el("text", {
      x: experts[3].right, y: experts[3].bottom + 22, "text-anchor": "end",
      fill: C.canvas_text, "font-family": FONT, "font-size": 18, "font-style": "italic"
    });
    annot.textContent = `(${ffn.num_experts || "N"})`;
    svg.appendChild(annot);

    experts.forEach(e => {
      svg.appendChild(el("line", {
        x1: e.cx, y1: router.top - GAP, x2: e.cx, y2: e.bottom + GAP,
        stroke: C.arrow, "stroke-width": 1.6, "stroke-linecap": "round",
        "marker-end": "url(#tv-arrow)", fill: "none"
      }));
    });

    experts.forEach(e => {
      const sx = sumNode.cx;
      const targetX = sx + (e.cx < sx ? -sumNode.r - GAP : sumNode.r + GAP);
      elbowPath(svg, e.cx, e.top - GAP, targetX, sumNode.cy);
    });

    const inLabel = el("text", {
      x: cx, y: H - 22, "text-anchor": "middle",
      fill: C.canvas_text, "font-family": FONT, "font-size": 16
    });
    inLabel.textContent = `top-${ffn.num_experts_per_tok || "k"} of ${ffn.num_experts || "N"} experts active per token`;
    svg.appendChild(inLabel);

    return svg;
  }

  function buildFfnView(ir, info) {
    const W = 680, H = 540;
    const svg = el("svg", { width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img" });
    svg.appendChild(defs());
    svg.appendChild(regionRect(80, 30, W - 160, H - 60, C.bg_outer));

    const ffn = info.dominant.spec.ffn;
    const cx = W / 2;
    const actName = ffn.activation ? ffn.activation.toUpperCase() : "SiLU";

    const downProj = rectBlock(svg, "down_proj", cx - 80, 80, 160, 50, "Linear (down)");
    const mulNode  = plusBlock (svg, "mul", cx, 200, "×");
    const silu     = rectBlock(svg, "silu",      cx - 220, 290, 170, 50, actName);
    const upProj   = rectBlock(svg, "up_proj",   cx + 50,  290, 170, 50, "Linear (up)");
    const gateProj = rectBlock(svg, "gate_proj", cx - 220, 410, 170, 50, "Linear (gate)");

    vLine(svg, mulNode, downProj);
    elbowPath(svg, silu.cx, silu.top - GAP, mulNode.cx - mulNode.r - GAP, mulNode.cy);
    elbowPath(svg, upProj.cx, upProj.top - GAP, mulNode.cx + mulNode.r + GAP, mulNode.cy);
    vLine(svg, gateProj, silu);

    const inputY = H - 35;
    const branchY = inputY - 22;
    svg.appendChild(el("circle", { cx, cy: branchY, r: 3, fill: C.arrow }));
    svg.appendChild(el("path", {
      d: `M ${cx} ${inputY} L ${cx} ${branchY}`,
      fill: "none", stroke: C.arrow, "stroke-width": 1.6, "stroke-linecap": "round"
    }));
    elbowPath(svg, cx, branchY, gateProj.cx, gateProj.bottom + GAP);
    elbowPath(svg, cx, branchY, upProj.cx, upProj.bottom + GAP);

    const inLabel = el("text", {
      x: cx, y: H - 14, "text-anchor": "middle",
      fill: C.canvas_text, "font-family": FONT, "font-size": 16
    });
    inLabel.textContent = "x (input)";
    svg.appendChild(inLabel);

    const dimLabel = el("text", {
      x: cx, y: 30, "text-anchor": "middle",
      fill: C.canvas_text, "font-family": FONT, "font-size": 14, "font-style": "italic"
    });
    dimLabel.textContent = `hidden ${fmtInt(ffn.expert_intermediate_size || ffn.intermediate_size)}`;
    svg.appendChild(dimLabel);

    return svg;
  }

  function buildLayerMap(ir, info) {
    const W = 680, H = 200;
    const svg = el("svg", { width: "100%", viewBox: `0 0 ${W} ${H}`, role: "img" });
    svg.appendChild(defs());
    svg.appendChild(regionRect(80, 30, W - 160, H - 60, C.bg_outer));

    const palette = ["#0F6E56", "#1D9E75", "#3C3489", "#993C1D", "#185FA5", "#854F0B"];
    const sigToColor = {};
    info.groups.forEach((g, i) => { sigToColor[g.sig] = palette[i % palette.length]; });

    const stripX = 110, stripY = 90, stripW = W - 220, stripH = 36;
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
        fill: sigToColor[sig], opacity: 0.92
      }));
    });

    svg.appendChild(el("rect", {
      x: stripX, y: stripY, width: stripW, height: stripH,
      fill: "none", stroke: C.stroke, "stroke-width": 0.5, rx: 4, ry: 4
    }));

    const title = el("text", {
      x: stripX, y: 70, fill: C.canvas_text,
      "font-family": FONT, "font-size": 16
    });
    title.textContent = `${n} layers · ${info.groups.length} ${info.groups.length === 1 ? "type" : "types"}`;
    svg.appendChild(title);

    let lx = stripX, ly = stripY + stripH + 30;
    info.groups.forEach((g, i) => {
      const ffnK = g.spec.ffn.kind === "moe" ? "MoE" : "Dense";
      const mask = g.spec.attention.mask === "sliding" ? "sliding" : "full";
      const labelTxt = `${g.spec.attention.kind.toUpperCase()} + ${ffnK} (${mask}) · layers ${g.indices[0]}–${g.indices[g.indices.length - 1]}`;
      svg.appendChild(el("rect", { x: lx, y: ly - 10, width: 12, height: 12, fill: sigToColor[g.sig], rx: 2 }));
      const txt = el("text", {
        x: lx + 18, y: ly, fill: C.canvas_text,
        "font-family": FONT, "font-size": 14
      });
      txt.textContent = labelTxt;
      svg.appendChild(txt);
      lx = stripX;
      ly += 22;
    });

    return svg;
  }

  function render(ir, mount) {
    const info = makeInfo(ir);

    mount.innerHTML = "";
    mount.style.fontFamily = "system-ui, -apple-system, 'Segoe UI', sans-serif";

    const header = document.createElement("div");
    header.style.cssText = "display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;align-items:center;font-size:13px;";
    header.innerHTML = `
      <div style="display:flex;gap:6px;padding:4px;background:${C.panel};border:0.5px solid ${C.panel_border};border-radius:8px;">
        <button data-view="arch" class="tv-btn tv-active">Architecture</button>
        <button data-view="moe"  class="tv-btn"${info.dominant.spec.ffn.kind === "moe" ? "" : " disabled"}>MoE detail</button>
        <button data-view="ffn"  class="tv-btn">${info.dominant.spec.ffn.kind === "moe" ? "Expert" : "FFN"} detail</button>
        <button data-view="map"  class="tv-btn">Layer map</button>
      </div>
      <div style="margin-left:auto;color:#5F5E5A;">
        <strong style="color:${C.canvas_text}">${ir.name}</strong>
        &nbsp;·&nbsp; ${ir.layers.length} layers
        &nbsp;·&nbsp; hidden ${fmtInt(ir.hidden_size)}
        &nbsp;·&nbsp; vocab ${fmtInt(ir.vocab_size)}
      </div>
    `;
    mount.appendChild(header);

    const style = document.createElement("style");
    style.textContent = `
      .tv-btn { padding:6px 12px;font-size:13px;border:0.5px solid #D3D1C7;background:transparent;color:#5F5E5A;border-radius:6px;cursor:pointer;font-family:inherit; }
      .tv-btn.tv-active { border:0.5px solid ${C.block};background:${C.bg_outer};color:${C.canvas_text}; }
      .tv-btn:disabled { opacity:0.4;cursor:not-allowed; }
    `;
    mount.appendChild(style);

    const canvas = document.createElement("div");
    canvas.style.cssText = `background:#fff;border:0.5px solid #D3D1C7;border-radius:12px;padding:8px;`;
    mount.appendChild(canvas);

    const panel = document.createElement("div");
    panel.style.cssText = `margin-top:12px;padding:12px 14px;background:${C.panel};border:0.5px solid ${C.panel_border};border-radius:8px;font-size:13px;color:#5F5E5A;min-height:40px;`;
    panel.textContent = "Click any block to inspect its dimensions and role.";
    mount.appendChild(panel);

    function attachClicks(svg) {
      svg.querySelectorAll(".tv-node").forEach(node => {
        node.addEventListener("click", () => {
          const id = node.getAttribute("data-id");
          const meta = info.meta[id];
          if (meta) {
            panel.innerHTML = `<strong style="color:${C.canvas_text}">${meta[0]}</strong> &nbsp;·&nbsp; ${meta[1]}`;
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
      header.querySelectorAll(".tv-btn").forEach(b => {
        b.classList.toggle("tv-active", b.dataset.view === view);
      });
    }

    header.querySelectorAll(".tv-btn").forEach(b => {
      b.addEventListener("click", () => {
        if (!b.disabled) show(b.dataset.view);
      });
    });

    show("arch");
  }

  global.TransformerViz = { render };
})(typeof window !== "undefined" ? window : this);
