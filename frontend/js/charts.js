/* Tiny SVG chart helpers (no external library, works offline).
 *
 * Every chart is drawn at the pixel width of its container (1 SVG unit = 1 CSS pixel), so text never
 * shrinks or grows with the layout, and it is redrawn automatically when the container is resized.
 * Marks follow one style: thin bars with a 4px rounded data end anchored to the baseline, a 2px gap between
 * neighbouring bars, recessive grid lines, values in text colours (never the series colour), and a hover
 * tooltip on every mark. */
/* eslint-disable no-unused-vars */
const Charts = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const tip = () => document.getElementById("tooltip");

  function el(tag, attrs = {}, parent = null, text = null) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) n.setAttribute(k, v);
    if (text !== null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }

  // ------------------------------------------------------------------ sizing and redraw on resize
  // Redraw on the next frame (coalesces bursts of resize events) and ignore transient tiny sizes,
  // e.g. while a tab is being hidden or the browser is capturing a screenshot.
  const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver((entries) => {
    for (const e of entries) {
      const c = e.target;
      if (!c.__render || e.contentRect.width < 200) continue;
      cancelAnimationFrame(c.__raf);
      c.__raf = requestAnimationFrame(() => {
        const w = Math.round(c.getBoundingClientRect().width);
        if (w >= 200 && Math.abs(w - (c.__w || 0)) > 3) c.__render();
      });
    }
  }) : null;

  function measure(container, fallback) {
    const w = Math.round(container.getBoundingClientRect().width);
    return Math.max(260, w > 0 ? w : fallback);
  }

  function track(container, render, W) {
    container.__render = render;
    container.__w = W;
    if (observer && !container.__observed) { observer.observe(container); container.__observed = true; }
  }

  // ------------------------------------------------------------------ tooltip
  function showTip(evt, html) {
    const t = tip();
    if (!t) return;
    t.innerHTML = html;
    t.hidden = false;
    const pad = 14;
    let x = evt.clientX + pad, y = evt.clientY + pad;
    const r = t.getBoundingClientRect();
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
    t.style.left = `${Math.max(4, x)}px`;
    t.style.top = `${Math.max(4, y)}px`;
  }
  function hideTip() { const t = tip(); if (t) t.hidden = true; }

  function hit(parent, attrs, html) {
    const r = el("rect", { ...attrs, fill: "transparent" }, parent);
    r.addEventListener("mousemove", (e) => showTip(e, html));
    r.addEventListener("mouseleave", hideTip);
    return r;
  }

  // ------------------------------------------------------------------ geometry helpers
  // Bar with 4px rounded data end, square at the baseline.
  function vbarPath(x, yTop, w, h, r = 4) {
    if (h <= 0.5) return `M${x},${yTop + h}h${w}`;
    const rr = Math.min(r, w / 2, h);
    return `M${x},${yTop + h}V${yTop + rr}Q${x},${yTop} ${x + rr},${yTop}H${x + w - rr}Q${x + w},${yTop} ${x + w},${yTop + rr}V${yTop + h}Z`;
  }
  function hbarPath(x0, y, len, h, r = 4) {
    // len may be negative (bar grows to the left of x0)
    if (Math.abs(len) < 0.5) return `M${x0},${y}v${h}`;
    const rr = Math.min(r, h / 2, Math.abs(len));
    const x1 = x0 + len;
    if (len > 0) return `M${x0},${y}H${x1 - rr}Q${x1},${y} ${x1},${y + rr}V${y + h - rr}Q${x1},${y + h} ${x1 - rr},${y + h}H${x0}Z`;
    return `M${x0},${y}H${x1 + rr}Q${x1},${y} ${x1},${y + rr}V${y + h - rr}Q${x1},${y + h} ${x1 + rr},${y + h}H${x0}Z`;
  }

  /** Round axis: max and step from {1, 2, 2.5, 5} x 10^k with at most `maxTicks` intervals. */
  function niceScale(v, maxTicks = 4) {
    if (!(v > 0)) return { max: 1, step: 0.25 };
    const raw = v / maxTicks;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 2.5, 5, 10]) {
      const step = m * p;
      if (Math.ceil(v / step - 1e-9) <= maxTicks) return { max: Math.ceil(v / step - 1e-9) * step, step };
    }
    return { max: 10 * p * maxTicks, step: 10 * p };
  }
  /** Fewest decimals that print every multiple of `step` (times `scale`) exactly: 5 -> 0, 2.5 -> 1, 0.25 -> 2. */
  const decimalsFor = (step, scale = 1) => {
    const s = step * scale;
    for (let d = 0; d <= 3; d++) if (Math.abs(s * 10 ** d - Math.round(s * 10 ** d)) < 1e-6) return d;
    return 3;
  };

  /** Split a category label into at most two lines (at a hyphen or the space nearest the middle) when it is
   *  wider than `maxWidth` pixels; 6.4px per character approximates the 12px label font. */
  function wrapLabel(text, maxWidth) {
    const t = String(text);
    if (t.length * 6.4 <= maxWidth) return [t];
    const h = t.indexOf("-");
    if (h > 0 && h < t.length - 1) return [t.slice(0, h + 1), t.slice(h + 1)];
    const spaces = [...t.matchAll(/ /g)].map((mm) => mm.index);
    if (!spaces.length) return [t];
    const cut = spaces.reduce((a, b) => (Math.abs(b - t.length / 2) < Math.abs(a - t.length / 2) ? b : a));
    return [t.slice(0, cut), t.slice(cut + 1)];
  }
  function catLabel(svg, x, y, lines) {
    const t = el("text", { x, y, "text-anchor": "middle", class: "cat-label" }, svg);
    lines.forEach((ln, k) => el("tspan", { x, dy: k ? 13 : 0 }, t, ln));
    return t;
  }

  /** Vertical bars, one series. opts: categories, values, color, yMax, threshold, thresholdLabel,
   *  format(v), tickFormat(v, step), highlight (index), tooltip(i), height */
  function bars(container, o) {
    const render = () => {
      container.innerHTML = "";
      const W = measure(container, 460), H = o.height || 230;
      container.__w = W;
      const hasThr = o.threshold !== undefined && o.threshold !== null;
      const m = { t: 22, r: hasThr ? (W < 420 ? 62 : 74) : 10, b: 0, l: 42 };
      const iw = W - m.l - m.r, n = o.categories.length, band = iw / n;
      const labels = o.categories.map((c) => wrapLabel(c, band - 4));
      const wrapped = labels.some((l) => l.length > 1);
      m.b = (o.highlight !== undefined ? 38 : 26) + (wrapped ? 13 : 0);
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "chart", role: "img", "aria-label": o.aria || "" }, container);
      const sc = niceScale(o.yMax || Math.max(...o.values, hasThr ? o.threshold : 0) * 1.12);
      const ih = H - m.t - m.b;
      const y = (v) => m.t + ih - (v / sc.max) * ih;
      const tf = o.tickFormat || ((v) => o.format(v, true));
      for (let v = 0; v <= sc.max + 1e-12; v += sc.step) {
        const yy = y(v);
        el("line", { x1: m.l, x2: m.l + iw, y1: yy, y2: yy, class: v === 0 ? "base-line" : "grid-line" }, svg);
        el("text", { x: m.l - 6, y: yy + 4, "text-anchor": "end", class: "tick" }, svg, tf(v, sc.step));
      }
      const bw = Math.max(8, Math.min(46, band * 0.62 - 2));
      const xs = o.categories.map((_, i) => m.l + band * i + (band - bw) / 2);
      // layers: bars, then the threshold line, then value labels (with a halo) so the line never hides a number
      o.categories.forEach((c, i) => {
        const v = o.values[i], yy = y(v);
        const p = el("path", { d: vbarPath(xs[i], yy, bw, m.t + ih - yy), fill: o.color, class: "bar" + (i === o.highlight ? " current" : "") }, svg);
        if (o.dim && o.dim[i]) p.setAttribute("fill-opacity", "0.55");
        catLabel(svg, xs[i] + bw / 2, m.t + ih + 16, labels[i]);
        if (i === o.highlight) el("text", { x: xs[i] + bw / 2, y: m.t + ih + (wrapped ? 43 : 30), "text-anchor": "middle", class: "tick" }, svg, "this order");
      });
      if (hasThr) {
        const ty = y(o.threshold);
        el("line", { x1: m.l, x2: m.l + iw, y1: ty, y2: ty, class: "thr-line" }, svg);
        const t = el("text", { x: m.l + iw + 6, y: ty - 3, class: "thr-label" }, svg);
        el("tspan", { x: m.l + iw + 6, dy: 0 }, t, o.thresholdLabel || "high risk");
        el("tspan", { x: m.l + iw + 6, dy: 13 }, t, `≥ ${o.format(o.threshold)}`);
      }
      o.categories.forEach((c, i) => {
        el("text", { x: xs[i] + bw / 2, y: y(o.values[i]) - 6, "text-anchor": "middle", class: "value-label" }, svg, o.format(o.values[i]));
      });
      o.categories.forEach((c, i) => {
        hit(svg, { x: m.l + band * i, y: m.t, width: band, height: ih + 4 }, o.tooltip ? o.tooltip(i) : `${c}: ${o.format(o.values[i])}`);
      });
    };
    render();
    track(container, render, measure(container, 460));
  }

  /** Grouped vertical bars. opts: categories, series [{name,color,values}], format, tickFormat, yMax, tooltip(i) */
  function groupedBars(container, o) {
    const render = () => {
      container.innerHTML = "";
      container.appendChild(legend(o.series.map((s) => ({ name: s.name, color: s.color }))));
      const holder = document.createElement("div");
      container.appendChild(holder);
      const W = measure(container, 900), H = o.height || 300, m = { t: 18, r: 10, b: 30, l: 44 };
      container.__w = W;
      const labels = o.categories.map((c) => wrapLabel(c, (W - m.l - m.r) / o.categories.length - 4));
      if (labels.some((l) => l.length > 1)) m.b += 13;
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "chart", role: "img", "aria-label": o.aria || "" }, holder);
      const all = o.series.flatMap((s) => s.values).filter((v) => v !== null && v !== undefined && !Number.isNaN(v));
      const sc = niceScale(o.yMax || Math.max(...all) * 1.1);
      const iw = W - m.l - m.r, ih = H - m.t - m.b;
      const y = (v) => m.t + ih - (v / sc.max) * ih;
      const tf = o.tickFormat || ((v) => o.format(v, true));
      for (let v = 0; v <= sc.max + 1e-12; v += sc.step) {
        const yy = y(v);
        el("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy, class: v === 0 ? "base-line" : "grid-line" }, svg);
        el("text", { x: m.l - 6, y: yy + 4, "text-anchor": "end", class: "tick" }, svg, tf(v, sc.step));
      }
      const n = o.categories.length, band = iw / n, ns = o.series.length;
      const gw = Math.min(band * 0.8, ns * 54), bw = gw / ns;
      o.categories.forEach((c, i) => {
        const x0 = m.l + band * i + (band - gw) / 2;
        o.series.forEach((s, j) => {
          const v = s.values[i];
          if (v === null || v === undefined || Number.isNaN(v)) return;
          const x = x0 + j * bw + 1, w = bw - 2, yy = y(v);
          el("path", { d: vbarPath(x, yy, w, m.t + ih - yy), fill: s.color, class: "bar" }, svg);
          if (o.labels !== false && w >= 18) el("text", { x: x + w / 2, y: yy - 5, "text-anchor": "middle", class: "value-label" }, svg, o.format(v));
        });
        catLabel(svg, m.l + band * i + band / 2, H - m.b + 18, labels[i]);
        const html = `<strong>${c}</strong><br>` + o.series.map((s) => `${s.name}: ${s.values[i] === null || s.values[i] === undefined ? "n/a" : o.format(s.values[i])}`).join("<br>");
        hit(svg, { x: m.l + band * i, y: m.t, width: band, height: ih }, o.tooltip ? o.tooltip(i) : html);
      });
    };
    render();
    track(container, render, measure(container, 900));
  }

  /** Horizontal bars. rows [{label, value, color, err, note}], format, xMax, ref {value,label}, labelWidth */
  function hbars(container, o) {
    const render = () => {
      // keep a heading the caller inserted before the chart
      [...container.querySelectorAll("svg.chart")].forEach((s) => s.remove());
      const rowH = o.rowH || 26, W = measure(container, 640), lw = Math.min(o.labelWidth || 230, Math.round(W * 0.4));
      container.__w = W;
      const H = o.rows.length * rowH + 30;
      const m = { t: 6, r: 64, b: 24, l: lw };
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "chart", role: "img", "aria-label": o.aria || "" }, container);
      const vals = o.rows.map((r) => r.value + (r.err || 0)).filter((v) => Number.isFinite(v));
      const xMin = o.xMin || 0;
      const sc = niceScale(o.xMax || Math.max(...vals, o.ref ? o.ref.value : 0, 1e-9) * 1.05);
      const xMax = o.xMax || sc.max;
      const iw = W - m.l - m.r;
      const x = (v) => m.l + ((v - xMin) / (xMax - xMin)) * iw;
      const tf = o.tickFormat || ((v) => o.format(v, true));
      for (let v = xMin; v <= xMax + 1e-12; v += sc.step) {
        const xx = x(v);
        el("line", { x1: xx, x2: xx, y1: m.t, y2: H - m.b, class: v === xMin ? "base-line" : "grid-line" }, svg);
        el("text", { x: xx, y: H - 6, "text-anchor": "middle", class: "tick" }, svg, tf(v, sc.step));
      }
      o.rows.forEach((r, i) => {
        const yy = m.t + i * rowH + 4, h = rowH - 9;
        el("text", { x: m.l - 8, y: yy + h / 2 + 4, "text-anchor": "end", class: "cat-label" }, svg, r.label);
        if (!Number.isFinite(r.value)) {
          el("text", { x: m.l + 4, y: yy + h / 2 + 4, class: "tick" }, svg, "n/a");
          return;
        }
        el("path", { d: hbarPath(x(xMin), yy, x(r.value) - x(xMin), h), fill: r.color, class: "bar" }, svg);
        if (r.err) el("line", { x1: x(r.value - r.err), x2: x(r.value + r.err), y1: yy + h / 2, y2: yy + h / 2, stroke: "#4a5468", "stroke-width": 1 }, svg);
        el("text", { x: x(r.value + (r.err || 0)) + 6, y: yy + h / 2 + 4, class: "value-label" }, svg, o.format(r.value));
        hit(svg, { x: 0, y: yy - 2, width: W, height: rowH }, `<strong>${r.label}</strong><br>${o.format(r.value)}${r.err ? ` ± ${o.format(r.err)}` : ""}${r.note ? `<br>${r.note}` : ""}`);
      });
      if (o.ref) {
        const xx = x(o.ref.value);
        el("line", { x1: xx, x2: xx, y1: m.t, y2: H - m.b, class: "thr-line" }, svg);
        el("text", { x: xx + 4, y: m.t + 10, class: "thr-label" }, svg, o.ref.label);
      }
    };
    render();
    track(container, render, measure(container, 640));
  }

  /** Diverging horizontal bars around zero. rows [{label, value}], format, maxAbs, colorPos, colorNeg */
  function divergingBars(container, o) {
    const render = () => {
      container.innerHTML = "";
      const rowH = 26, W = measure(container, 420), lw = Math.min(170, Math.round(W * 0.4));
      container.__w = W;
      const H = o.rows.length * rowH + 6;
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "chart", role: "img", "aria-label": o.aria || "" }, container);
      const maxAbs = o.maxAbs || Math.max(1e-9, ...o.rows.map((r) => Math.abs(r.value)));
      const room = 62;                               // space for the value label on either side
      const half = Math.max(30, (W - lw - 8 - 2 * room) / 2), x0 = lw + 8 + room + half;
      el("line", { x1: x0, x2: x0, y1: 0, y2: H, class: "base-line" }, svg);
      o.rows.forEach((r, i) => {
        const yy = 4 + i * rowH, h = rowH - 9, len = (r.value / maxAbs) * half;
        el("text", { x: lw, y: yy + h / 2 + 4, "text-anchor": "end", class: "cat-label" }, svg, r.label);
        el("path", { d: hbarPath(x0, yy, len, h), fill: r.value >= 0 ? o.colorPos : o.colorNeg, class: "bar" }, svg);
        const lx = len >= 0 ? x0 + len + 5 : x0 + len - 5;
        el("text", { x: lx, y: yy + h / 2 + 4, "text-anchor": len >= 0 ? "start" : "end", class: "value-label" }, svg, o.format(r.value));
        hit(svg, { x: 0, y: yy - 3, width: W, height: rowH }, `<strong>${r.label}</strong><br>${Math.abs(r.value) < 5e-5 ? "no effect" : `${r.value >= 0 ? "raises" : "lowers"} the risk by ${o.format(Math.abs(r.value)).replace("+", "")}`}`);
      });
    };
    render();
    track(container, render, measure(container, 420));
  }

  function legend(items) {
    const d = document.createElement("div");
    d.className = "legend";
    for (const it of items) {
      const s = document.createElement("span");
      const i = document.createElement("i");
      i.style.background = it.color;
      s.appendChild(i);
      s.appendChild(document.createTextNode(it.name));
      d.appendChild(s);
    }
    return d;
  }

  const pctTick = (v, step) => `${(v * 100).toFixed(decimalsFor(step, 100))}%`;

  return { bars, groupedBars, hbars, divergingBars, legend, hideTip, niceScale, pctTick };
})();
