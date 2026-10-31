/* Fair Returns front end: plain JavaScript, talks to the Flask API under /api. */
(() => {
  "use strict";
  const C = { baseline: "#2a78d6", ours: "#eb6834", aqua: "#1baf7a", violet: "#6250d6", raise: "#e34948", lower: "#8b94a5", muted: "#898781" };
  const ROLE_COLOR = { baseline: C.baseline, "fairness baseline": C.aqua, reference: C.muted, ablation: C.violet, ours: C.ours, variant: "#eda100" };
  const $ = (sel) => document.querySelector(sel);
  const state = { meta: null, products: new Map(), examples: [], invOffset: 0, invLimit: 25, currentInvoice: null, loaded: {} };

  // ------------------------------------------------------------------ utilities
  const pct = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : `${(v * 100).toFixed(d)}%`);
  const pts = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(2)} pts`;
  const num = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const title = (s) => String(s || "").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
  const short = (g) => (state.meta ? state.meta.group_short[g] || g : g);

  function setStatus(msg, isError = false) {
    const s = $("#status");
    s.textContent = msg || "";
    s.classList.toggle("error", isError);
  }

  async function api(path, opts = {}) {
    const init = { headers: {}, ...opts };
    if (opts.json) {
      init.method = opts.method || "POST";
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.json);
    }
    const res = await fetch(path, init);
    let data = null;
    try { data = await res.json(); } catch (_e) { /* non-JSON */ }
    if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
    return data;
  }

  const ICON_ALERT = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.5 15 14H1L8 1.5Z" fill="currentColor"/><path d="M8 6v4M8 11.6v.2" stroke="#fff" stroke-width="1.6" stroke-linecap="round"/></svg>';
  const ICON_OK = '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="currentColor"/><path d="m4.8 8.2 2.1 2.1 4.3-4.4" stroke="#fff" stroke-width="1.6" fill="none" stroke-linecap="round"/></svg>';
  const riskChip = (high) => (high ? `<span class="chip high">${ICON_ALERT}High risk</span>` : `<span class="chip low">${ICON_OK}Below threshold</span>`);

  // ------------------------------------------------------------------ tabs
  function showTab(name) {
    const tabs = ["order", "invoices", "fairness", "genai", "about"];
    if (!tabs.includes(name)) name = "order";
    document.querySelectorAll(".tab").forEach((t) => { t.hidden = t.dataset.tab !== name; });
    document.querySelectorAll(".tabs a").forEach((a) => a.setAttribute("aria-selected", String(a.dataset.tab === name)));
    Charts.hideTip();
    if (!state.loaded[name]) {
      state.loaded[name] = true;
      ({ invoices: loadInvoices, fairness: loadFairness, genai: loadGenAI, about: loadAbout }[name] || (() => {}))();
    }
  }
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

  // ------------------------------------------------------------------ init
  async function init() {
    setStatus("Loading…");
    try {
      state.meta = await api("/api/meta");
    } catch (e) {
      setStatus("Server not ready: " + e.message, true);
      $("#cf-charts").innerHTML = `<p class="empty">The API did not answer (${esc(e.message)}). Run <code>python run_pipeline.py</code>, then <code>python -m fairreturn.backend</code>.</p>`;
      return;
    }
    fillCountrySelect();
    const groupSel = $("#inv-group");
    state.meta.groups.forEach((g) => groupSel.add(new Option(g, g)));
    await loadProducts("");
    try {
      state.examples = await api("/api/examples");
      const sel = $("#example-select");
      state.examples.forEach((ex, i) => sel.add(new Option(ex.title + (ex.actual_returned ? " (was returned)" : ""), String(i))));
    } catch (_e) { /* examples are optional */ }
    fillCompareSelect();
    const llm = state.meta.llm || {};
    showTab(location.hash.slice(1) || "order");
    // the form must be filled before the first check, otherwise the scores would belong to another order
    if (state.examples.length) {
      await applyExample(state.examples[0]);
      $("#example-select").value = "0";
    } else {
      pickProduct(state.products.values().next().value);
    }
    await runCheck();
    setStatus(`Assistant: ${llm.backend === "api" ? llm.model : "rule-based (no API key)"}`);
  }

  // Which model the order check compares ours with (any loaded model except ours).
  function fillCompareSelect() {
    const sel = $("#compare-model");
    if (!sel) return;
    sel.innerHTML = "";
    state.meta.models.filter((m) => m.name !== state.meta.ours).forEach((m) => sel.add(new Option(m.label, m.name)));
    state.compare = state.meta.models.some((m) => m.name === state.meta.baseline) ? state.meta.baseline : sel.value;
    sel.value = state.compare;
    sel.addEventListener("change", () => { state.compare = sel.value; runCheck(); });
  }

  function describeInputs(m) {
    if (!m) return "";
    return { aware: "uses every field, including the country", unaware: "country column removed, postage line kept",
      blind: "country and postage removed", notags: "country removed, no product tags" }[m.feature_set] || "";
  }

  function fillCountrySelect() {
    const sel = $("#country");
    const weights = state.meta.country_weights || {};
    sel.innerHTML = "";
    const groups = state.meta.groups;
    groups.forEach((g) => {
      const og = document.createElement("optgroup");
      og.label = g;
      const names = Object.keys((state.meta.countries_by_group || {})[g] || {});
      const list = names.length ? names : (state.meta.countries || []).filter((c) => groupOf(c) === g);
      (list.length ? list : [g]).forEach((c) => og.appendChild(new Option(c, c)));
      sel.appendChild(og);
    });
    sel.value = "United Kingdom";
    void weights;
  }
  function groupOf(country) {
    const direct = ["United Kingdom", "Germany", "France", "EIRE"];
    if (direct.includes(country)) return country;
    const eu = ["Netherlands", "Spain", "Belgium", "Portugal", "Italy", "Sweden", "Finland", "Austria", "Denmark", "Greece", "Cyprus", "Poland", "Lithuania", "Malta", "Czech Republic", "European Community", "Norway", "Switzerland", "Iceland", "Channel Islands"];
    return eu.includes(country) ? "Other Europe" : "Rest of World";
  }

  // ------------------------------------------------------------------ products
  let searchTimer = null;
  async function loadProducts(q) {
    const list = await api(`/api/products?q=${encodeURIComponent(q)}&limit=40`);
    const dl = $("#product-list");
    dl.innerHTML = "";
    list.forEach((p) => {
      state.products.set(p.stock_code, p);
      dl.appendChild(new Option(`${title(p.description)} (${p.stock_code})`));
    });
    return list;
  }
  function pickProduct(p) {
    if (!p) return;
    state.product = p;
    $("#product-search").value = `${title(p.description)} (${p.stock_code})`;
    $("#unit-price").value = Number(p.typical_price || 1).toFixed(2);
    const card = $("#product-card");
    card.hidden = false;
    card.innerHTML = `<strong>${esc(title(p.description))}</strong>
      <span>Stock code ${esc(p.stock_code)}, usual price £${Number(p.typical_price).toFixed(2)}, ${Number(p.n_lines).toLocaleString()} sales, ${Number(p.history_returns).toLocaleString()} returns</span>
      <span class="tagline">
        <span class="tag">${esc(String(p.tag_category).replace(/_/g, " "))}</span>
        <span class="tag">${esc(p.tag_material)}</span>
        ${p.tag_fragile ? '<span class="tag fragile">fragile</span>' : ""}
        ${p.tag_seasonal ? '<span class="tag">seasonal</span>' : ""}
        <span class="tag" title="How the attributes were produced">tags: ${esc(String(p.tag_source || "rules").replace(/_/g, " "))}</span>
      </span>`;
  }
  $("#product-search").addEventListener("input", (e) => {
    const v = e.target.value;
    const m = v.match(/\(([^()]+)\)\s*$/);
    if (m && state.products.has(m[1])) { pickProduct(state.products.get(m[1])); return; }
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadProducts(v), 200);
  });

  // ------------------------------------------------------------------ order form
  document.querySelectorAll('input[name="ctype"]').forEach((r) => r.addEventListener("change", () => {
    $("#history-fields").hidden = document.querySelector('input[name="ctype"]:checked').value === "new";
  }));

  function readOrder() {
    const isNew = document.querySelector('input[name="ctype"]:checked').value === "new";
    const postage = $("#postage").value;
    return {
      stock_code: state.product ? state.product.stock_code : "",
      quantity: Number($("#quantity").value), unit_price: Number($("#unit-price").value),
      basket_lines: Number($("#basket-lines").value), basket_value: Number($("#basket-value").value),
      order_date: $("#order-date").value, country: $("#country").value,
      has_postage: postage === "auto" ? "auto" : postage === "1",
      customer: isNew ? { new: true } : {
        prior_invoices: Number($("#prior-invoices").value), prior_items: Number($("#prior-items").value),
        prior_returns: Number($("#prior-returns").value), tenure_days: Number($("#tenure-days").value), recency_days: 30,
      },
    };
  }

  async function applyExample(ex) {
    let p = state.products.get(ex.stock_code);
    if (!p) { const found = await loadProducts(ex.stock_code); p = found.find((x) => x.stock_code === ex.stock_code); }
    if (p) pickProduct(p);
    $("#quantity").value = ex.quantity;
    $("#unit-price").value = Number(ex.unit_price).toFixed(2);
    $("#basket-lines").value = ex.basket_lines;
    $("#basket-value").value = Math.round(ex.basket_value);
    $("#order-date").value = String(ex.order_date).slice(0, 16);
    $("#country").value = ex.country;
    $("#postage").value = ex.has_postage ? "1" : "0";
    const isNew = !!(ex.customer && ex.customer.new);
    document.querySelector(`input[name="ctype"][value="${isNew ? "new" : "returning"}"]`).checked = true;
    $("#history-fields").hidden = isNew;
    if (!isNew) {
      $("#prior-invoices").value = ex.customer.prior_invoices;
      $("#prior-items").value = ex.customer.prior_items;
      $("#prior-returns").value = ex.customer.prior_returns;
      $("#tenure-days").value = ex.customer.tenure_days;
    }
  }
  $("#example-select").addEventListener("change", async (e) => {
    if (e.target.value === "") return;
    await applyExample(state.examples[Number(e.target.value)]);
    runCheck();
  });
  $("#order-form").addEventListener("submit", (e) => { e.preventDefault(); runCheck(); });
  $("#country").addEventListener("change", () => runCheck());
  $("#move-proxies").addEventListener("change", () => runCheck());

  let checkSeq = 0;
  async function runCheck() {
    const order = readOrder();
    const seq = ++checkSeq;          // ignore answers to an older check that arrive late
    setStatus("Scoring…");
    try {
      const mode = $("#move-proxies").checked ? "with_proxies" : "country_only";
      const [pred, cf] = await Promise.all([api("/api/predict", { json: order }), api("/api/counterfactual", { json: { ...order, proxy_mode: mode } })]);
      if (seq !== checkSeq) return;
      const cmp = pred.models.some((m) => m.name === state.compare) ? state.compare : pred.baseline;
      renderVerdicts(pred, cmp);
      renderCounterfactual(cf, pred, cmp);
      renderFactors(pred, cmp);
      setStatus("");
      $("#assistant").innerHTML = '<p class="empty">Asking the assistant…</p>';
      const a = await api("/api/assistant", { json: order });
      if (seq !== checkSeq) return;
      renderAssistant(a);
    } catch (e) {
      setStatus(e.message, true);
    }
  }

  function renderVerdicts(pred, cmp) {
    const pick = (n) => pred.models.find((m) => m.name === n);
    const b = pick(cmp), o = pick(pred.ours);
    const card = (m, cls, who) => `<div class="verdict ${cls}">
      <span class="who">${who}</span>
      <span class="num">${pct(m.risk)}<small>chance of return</small></span>
      ${riskChip(m.high_risk)}
      <span class="who">High risk means at or above ${pct(m.threshold)} (the top ${Math.round(state.meta.high_risk_share * 100)}% of orders)</span>
    </div>`;
    $("#verdicts").innerHTML = card(b, "baseline", `<strong>${esc(b.label)}</strong>, ${esc(describeInputs(b))}`)
      + card(o, "ours", `<strong>Our model: ${esc(o.label)}</strong>, ${esc(describeInputs(o))}, trained to forget the country`);
  }

  function renderCounterfactual(cf, pred, cmp) {
    const box = $("#cf-charts");
    box.innerHTML = "";
    const names = [cmp, pred.ours];
    const cats = cf.groups.map(short);
    const yMax = Math.max(...names.flatMap((n) => [...cf.models[n].risks, cf.models[n].threshold])) * 1.12;
    const cur = cf.groups.indexOf(cf.original_group);
    names.forEach((n, k) => {
      const m = cf.models[n];
      const panel = document.createElement("div");
      panel.className = "cf-panel";
      const color = k === 0 ? C.baseline : C.ours;
      const flat = m.spread < 0.00005;
      panel.innerHTML = `<h4><span class="swatch" style="background:${color}"></span>${esc(m.label)}</h4>
        <p class="cf-spread">${flat ? "<strong>Identical</strong> for all six country groups"
          : `Range across countries <strong>${(m.spread * 100).toFixed(2)} pts</strong>, highest is ${m.ratio.toFixed(2)}× the lowest`}</p>`;
      const holder = document.createElement("div");
      panel.appendChild(holder);
      box.appendChild(panel);
      Charts.bars(holder, {
        categories: cats, values: m.risks, color, yMax, threshold: m.threshold, highlight: cur,
        format: (v) => pct(v), tickFormat: Charts.pctTick,
        aria: `${m.label}: risk by country group`,
        tooltip: (i) => `<strong>${esc(cf.groups[i])}</strong> (as ${esc(cf.countries[i])})<br>risk ${pct(m.risks[i])}${m.flags[i] ? ", flagged high-risk" : ""}<br>postage line: ${cf.postage[i] ? "yes" : "no"}`,
      });
    });
    const note = document.createElement("p");
    note.className = "note";
    note.style.gridColumn = "1 / -1";
    note.textContent = cf.proxy_mode === "with_proxies"
      ? "Moving the customer also moves the postage line to what is usual for that country, because postage is set by the destination."
      : "Only the country field changes; the postage line stays as entered.";
    box.appendChild(note);
    // the same numbers as a table (for screen readers, printing and exact values)
    const det = document.createElement("details");
    det.className = "prompt";
    det.style.gridColumn = "1 / -1";
    det.innerHTML = `<summary>Show the numbers</summary><div class="table-wrap"><table><thead><tr><th>Country group</th><th>Scored as</th><th>Postage line</th>
      ${names.map((n) => `<th class="num">${esc(cf.models[n].label)}</th>`).join("")}</tr></thead><tbody>
      ${cf.groups.map((g, i) => `<tr><td>${esc(g)}${i === cur ? " (this order)" : ""}</td><td>${esc(cf.countries[i])}</td><td>${cf.postage[i] ? "yes" : "no"}</td>
        ${names.map((n) => `<td class="num"><span class="flag ${cf.models[n].flags[i] ? "on" : ""}">${pct(cf.models[n].risks[i], 2)}</span></td>`).join("")}</tr>`).join("")}
      </tbody></table></div>`;
    box.appendChild(det);
  }

  function renderFactors(pred, cmp) {
    const box = $("#factors");
    box.innerHTML = "";
    box.appendChild(Charts.legend([{ name: "raises the risk", color: C.raise }, { name: "lowers the risk", color: C.lower }]));
    const cols = document.createElement("div");
    cols.className = "factor-cols";
    box.appendChild(cols);
    const both = pred.models.filter((x) => [cmp, pred.ours].includes(x.name));
    const maxAbs = Math.max(...both.flatMap((x) => x.factors.map((f) => Math.abs(f.delta))), 1e-6);
    [cmp, pred.ours].forEach((n) => {
      const m = pred.models.find((x) => x.name === n);
      const col = document.createElement("div");
      col.innerHTML = `<h4>${esc(m.label)}</h4><div></div>`;
      cols.appendChild(col);
      const rows = m.factors.slice(0, 7).map((f) => ({ label: f.group, value: f.delta }));
      Charts.divergingBars(col.lastChild, { rows, maxAbs, format: pts, colorPos: C.raise, colorNeg: C.lower, aria: `Factor contributions for ${m.label}` });
    });
  }

  function renderAssistant(a) {
    const src = String(a.source || "");
    $("#assistant").innerHTML = `<div class="assist">
      <span class="action">${esc(a.action_label)}</span>
      <p class="msg">${esc(a.message)}</p>
      <span class="meta">Written by ${src.startsWith("llm:") ? esc(src.slice(4)) : "the rule-based fallback"}. Guardrail: ${esc(a.guardrail)}.</span>
    </div>`;
  }

  // ------------------------------------------------------------------ real invoices
  async function loadInvoices() {
    const g = $("#inv-group").value, r = $("#inv-return").value, q = $("#inv-q").value, sort = $("#inv-sort").value;
    const data = await api(`/api/invoices?group=${encodeURIComponent(g)}&has_return=${r}&q=${encodeURIComponent(q)}&sort=${sort}&limit=${state.invLimit}&offset=${state.invOffset}`);
    const list = $("#inv-list");
    list.innerHTML = data.invoices.length ? "" : '<p class="empty">No invoices match these filters.</p>';
    data.invoices.forEach((inv) => {
      const b = document.createElement("button");
      b.className = "inv-item";
      b.type = "button";
      b.dataset.invoice = inv.invoice;
      b.setAttribute("aria-current", String(inv.invoice === state.currentInvoice));
      b.innerHTML = `<span class="t">Invoice ${esc(inv.invoice)}</span>
        <span class="r ${inv.returned_lines ? "ret" : ""}">${inv.returned_lines ? `${inv.returned_lines} returned` : "no return"}</span>
        <span class="s">${esc(inv.country)}, ${new Date(inv.date).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}, ${inv.lines} lines, £${Math.round(inv.value).toLocaleString()}${inv.country_effect > 0 ? ` · <span class="effect" title="Change in the lines the country-aware shop model flags when only the customer's country changes">${inv.group === "United Kingdom" ? `+${inv.country_effect} flagged if the customer lived in Germany` : `+${inv.country_effect} flagged because of the country`}</span>` : ""}</span>`;
      b.addEventListener("click", () => { state.currentInvoice = inv.invoice; openInvoice(inv.invoice, ""); list.querySelectorAll(".inv-item").forEach((x) => x.setAttribute("aria-current", "false")); b.setAttribute("aria-current", "true"); });
      list.appendChild(b);
    });
    const page = Math.floor(state.invOffset / state.invLimit) + 1, pages = Math.max(1, Math.ceil(data.total / state.invLimit));
    $("#inv-page").textContent = `Page ${page} of ${pages} (${data.total} invoices)`;
    $("#inv-prev").disabled = state.invOffset === 0;
    $("#inv-next").disabled = state.invOffset + state.invLimit >= data.total;
    if (!state.currentInvoice && data.invoices.length) {
      const first = sort === "country_effect" ? data.invoices[0]
        : (data.invoices.find((i) => i.group !== "United Kingdom" && i.returned_lines === 0) || data.invoices[0]);
      state.currentInvoice = first.invoice;
      openInvoice(first.invoice, "");
      list.querySelectorAll(".inv-item").forEach((x) => x.setAttribute("aria-current", String(x.dataset.invoice === first.invoice)));
    }
  }
  ["#inv-group", "#inv-return", "#inv-sort"].forEach((s) => $(s).addEventListener("change", () => { state.invOffset = 0; loadInvoices(); }));
  let invTimer = null;
  $("#inv-q").addEventListener("input", () => { clearTimeout(invTimer); invTimer = setTimeout(() => { state.invOffset = 0; loadInvoices(); }, 250); });
  $("#inv-prev").addEventListener("click", () => { state.invOffset = Math.max(0, state.invOffset - state.invLimit); loadInvoices(); });
  $("#inv-next").addEventListener("click", () => { state.invOffset += state.invLimit; loadInvoices(); });

  async function openInvoice(id, asGroup) {
    const box = $("#inv-detail");
    box.innerHTML = '<p class="empty">Scoring the invoice…</p>';
    try {
      const orig = await api(`/api/invoices/${encodeURIComponent(id)}`);
      const shown = asGroup && asGroup !== orig.original_group ? await api(`/api/invoices/${encodeURIComponent(id)}?as_group=${encodeURIComponent(asGroup)}`) : orig;
      renderInvoice(orig, shown);
    } catch (e) { box.innerHTML = `<p class="empty">${esc(e.message)}</p>`; }
  }

  function renderInvoice(orig, d) {
    const meta = state.meta;
    const b = d.summary[state.compare] ? state.compare : meta.baseline, o = meta.ours;
    const opts = meta.groups.map((g) => `<option value="${esc(g)}" ${g === d.shown_as_group ? "selected" : ""}>${esc(g)}${g === orig.original_group ? " (actual)" : ""}</option>`).join("");
    const moved = d.shown_as_group !== orig.original_group;
    const strip = [b, o].map((n) => {
      const s = d.summary[n], s0 = orig.summary[n];
      const diff = s.flagged - s0.flagged;
      return `<div><strong>${s.flagged} of ${d.n_lines}</strong>lines flagged by ${esc(s.label)}${moved ? ` <span class="${diff > 0 ? "delta-up" : ""}">(${diff >= 0 ? "+" : ""}${diff} vs actual country)</span>` : ""}</div>`;
    }).join("") + `<div><strong>${d.returned_lines} of ${d.n_lines}</strong>lines actually returned</div>`;
    const rows = d.lines.map((l) => `<tr>
        <td>${esc(title(l.description))}<span class="role">${esc(String(l.tag_category).replace(/_/g, " "))}${l.tag_fragile ? ", fragile" : ""}</span></td>
        <td class="num">${l.quantity}</td><td class="num">£${l.price.toFixed(2)}</td>
        <td>${l.returned ? '<span class="returned-mark">Returned</span>' : "—"}</td>
        <td class="num"><span class="flag ${l.flags[b] ? "on" : ""}">${pct(l.risks[b])}</span></td>
        <td class="num"><span class="flag ${l.flags[o] ? "on" : ""}">${pct(l.risks[o])}</span></td>
        <td>${esc((l.ours_top_factors[0] || {}).group || "")}</td></tr>`).join("");
    $("#inv-detail").innerHTML = `<div class="inv-head">
        <div><h2>Invoice ${esc(d.invoice)}</h2>
          <p class="facts">${esc(orig.original_country)}, customer ${esc(d.customer_id)}, ${new Date(d.date).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" })}. Postage line: ${orig.has_postage ? "yes" : "no"}. Customer had ${Math.round(d.customer.prior_invoices)} earlier orders and returned ${Math.round(d.customer.prior_returns)} of ${Math.round(d.customer.prior_items)} items.</p></div>
        <label class="switcher">Show as if the customer lived in <select id="as-group">${opts}</select></label>
      </div>
      <div class="summary-strip">${strip}</div>
      <div class="table-wrap"><table>
        <thead><tr><th>Product</th><th class="num">Qty</th><th class="num">Price</th><th>Outcome</th>
          <th class="num">${esc(d.summary[b].label)}</th><th class="num">${esc(d.summary[o].label)}</th><th>Main factor (ours)</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
      <p class="note">Red cells are flagged high-risk. Moving the customer changes only the country and the postage line; the products, quantities and the customer's own history stay the same.</p>`;
    $("#as-group").addEventListener("change", (e) => openInvoice(d.invoice, e.target.value));
  }

  // ------------------------------------------------------------------ batch scoring (CSV upload)
  let csvUrl = null;
  async function scoreCsv(text, name) {
    const out = $("#csv-result");
    out.innerHTML = `<p class="empty">Scoring ${esc(name)}…</p>`;
    try {
      const res = await fetch("/api/score-csv", { method: "POST", headers: { "Content-Type": "text/csv" }, body: text });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      const b = data.summary[state.compare] ? state.compare : state.meta.baseline, o = state.meta.ours;
      const label = (n) => (state.meta.models.find((m) => m.name === n) || {}).label || n;
      const strip = [b, o].map((n) => `<div><strong>${data.summary[n].flagged} of ${data.rows}</strong>lines flagged by ${esc(label(n))}</div>`).join("")
        + `<div><strong>${data.rows}</strong>product lines scored${data.skipped_rows ? ` (${data.skipped_rows} other rows skipped: postage, fees, returns)` : ""}</div>`;
      const rows = data.preview.slice(0, 25).map((r) => `<tr><td>${esc(r.invoice)}</td><td>${esc(title(r.description))}</td>
          <td class="num">${r.quantity}</td><td>${esc(r.country)}</td><td>${r.customer_id ? esc(r.customer_id) : "first order"}</td>
          <td class="num"><span class="flag ${r[`high_risk_${b}`] ? "on" : ""}">${pct(r[`risk_${b}`])}</span></td>
          <td class="num"><span class="flag ${r[`high_risk_${o}`] ? "on" : ""}">${pct(r[`risk_${o}`])}</span></td></tr>`).join("");
      out.innerHTML = `<div class="summary-strip">${strip}</div><div class="table-wrap"><table><thead><tr><th>Invoice</th><th>Product</th>
        <th class="num">Qty</th><th>Country</th><th>Customer</th><th class="num">${esc(label(b))}</th><th class="num">${esc(label(o))}</th></tr></thead>
        <tbody>${rows}</tbody></table></div>${data.rows > 25 ? `<p class="note">Showing the first 25 of ${data.rows} lines; the download has all of them and every model.</p>` : ""}`;
      if (csvUrl) URL.revokeObjectURL(csvUrl);
      csvUrl = URL.createObjectURL(new Blob([data.csv], { type: "text/csv" }));
      const a = $("#csv-download");
      a.href = csvUrl;
      a.download = `scored_${name.replace(/\.csv$/i, "")}.csv`;
      a.hidden = false;
    } catch (e) {
      out.innerHTML = `<p class="empty">${esc(e.message)}</p>`;
      $("#csv-download").hidden = true;
    }
  }
  $("#csv-score").addEventListener("click", async () => {
    const f = $("#csv-file").files[0];
    if (!f) { $("#csv-result").innerHTML = '<p class="empty">Choose a CSV file first, or try the sample file.</p>'; return; }
    scoreCsv(await f.text(), f.name);
  });
  $("#csv-sample").addEventListener("click", async () => {
    const r = await fetch("/static/sample_orders.csv");
    scoreCsv(await r.text(), "sample_orders.csv");
  });

  // ------------------------------------------------------------------ fairness
  async function loadFairness() {
    try {
      const [fair, res] = await Promise.all([api("/api/fairness"), api("/api/results")]);
      // one fixed colour per model (same as the report figures)
      const show = [["gbdt_aware", C.baseline], ["gbdt_blind", C.violet], ["gbdt_reweigh", C.aqua], ["vib_grl", C.ours]];
      const groups = state.meta.groups.filter((g) => g !== "EIRE");
      const series = show.filter(([m]) => fair.rows.some((r) => r.method === m)).map(([m, color]) => ({
        name: (fair.rows.find((r) => r.method === m) || {}).label || m, color,
        values: groups.map((g) => { const r = fair.rows.find((x) => x.method === m && x.group === g); return r ? r.fpr : null; }),
      }));
      Charts.groupedBars($("#fpr-chart"), { categories: groups, series, format: (v) => `${(v * 100).toFixed(0)}`, tickFormat: Charts.pctTick, aria: "False-alarm rate by country group" });
      $("#fpr-note").textContent = "Bar labels are percentages (seed 0). EIRE is left out of the chart because its test lines come from three customers.";
      renderResultsTable(res);
      renderGallery(res.figures);
    } catch (e) {
      $("#fpr-chart").innerHTML = `<p class="empty">${esc(e.message)}</p>`;
    }
  }

  function renderResultsTable(res) {
    const rows = res.summary || [];
    if (!rows.length) { $("#results-table").innerHTML = '<p class="empty">No results yet. Run the experiments first.</p>'; return; }
    const f = (r, k, d = 3) => {
      const m = r[`${k}_mean`], s = r[`${k}_std`];
      if (m === null || m === undefined) return "—";
      return s && s > Math.pow(10, -(d + 1)) ? `${num(m, d)} ± ${num(s, d)}` : num(m, d);
    };
    const body = rows.map((r) => `<tr class="${r.role === "ours" ? "ours" : ""}">
      <td>${esc(r.label)}<span class="role">${esc(r.role)}</span></td>
      <td class="num">${f(r, "auc")}</td><td class="num">${f(r, "pr_auc")}</td>
      <td class="num">${f(r, "fpr_gap")}</td><td class="num">${f(r, "tpr_gap")}</td><td class="num">${f(r, "dp_gap")}</td>
      <td class="num">${r.cf_flip_rate_mean === null ? "—" : pct(r.cf_flip_rate_mean, 2)}</td>
      <td class="num">${f(r, "output_probe")}</td><td class="num">${f(r, "probe_best")}</td>
      <td class="num">${r.size_kb_mean ? r.size_kb_mean.toFixed(1) : "—"}</td></tr>`).join("");
    $("#results-table").innerHTML = `<table><thead><tr><th>Method</th><th class="num">AUC</th><th class="num">PR-AUC</th>
      <th class="num">ΔFPR</th><th class="num">ΔTPR</th><th class="num">ΔDP</th><th class="num">Flip rate</th>
      <th class="num">Score probe</th><th class="num">z probe</th><th class="num">Size (KB)</th></tr></thead><tbody>${body}</tbody></table>`;
    const ip = res.input_probe || {};
    const chance = 1 / state.meta.groups.length;
    $("#results-note").textContent = `Lower is fairer for every column except AUC and PR-AUC. Flip rate: share of test lines whose high-risk flag changes when only the country and postage change. Probes: balanced accuracy of a fresh classifier recovering the country group (chance ${chance.toFixed(3)})` +
      (ip.blind ? `; from the inputs our model gets (country and postage removed) it still reaches ${ip.blind.probe_best.toFixed(3)}, because customer history and timing carry the country.` : ".");
  }

  function renderGallery(figs) {
    const caps = {
      "fig_data_overview.png": "Return rates by country group and month", "fig_accuracy_fairness.png": "Accuracy and fairness of every method",
      "fig_group_rates.png": "False-alarm and flag rates by country group", "fig_leakage.png": "How much country information remains",
      "fig_pareto.png": "Accuracy vs fairness as the adversary gets stronger", "fig_tsne.png": "Learned representation coloured by country",
      "fig_training.png": "VIB + GRL training curves", "fig_product_tags.png": "Return rates by LLM/rule product attributes",
      "fig_llm_audit.png": "LLM audit by prompt condition", "fig_stress_test.png": "Stress test with injected label bias",
      "fig_postage_proxy.png": "How models use the postage line (a country proxy)",
    };
    const g = $("#figure-gallery");
    g.innerHTML = figs.length ? "" : '<p class="empty">No figures yet. Run <code>python -m fairreturn.experiments.make_figures</code>.</p>';
    figs.forEach((f) => {
      const fig = document.createElement("figure");
      fig.innerHTML = `<a href="/figures/${encodeURIComponent(f)}" target="_blank" rel="noopener"><img src="/figures/${encodeURIComponent(f)}" alt="${esc(caps[f] || f)}" loading="lazy"></a><figcaption>${esc(caps[f] || f)}</figcaption>`;
      g.appendChild(fig);
    });
  }

  // ------------------------------------------------------------------ GenAI
  const COND_TEXT = {
    plain: ["Plain", "country and postage shown"], instructed: ["Instructed", "plus “do not use country”"],
    masked: ["Masked", "country hidden, postage shown"], tool: ["With our tool", "our debiased score, no country or postage"],
  };

  async function loadGenAI() {
    try {
      const [audit, tags] = await Promise.all([api("/api/audit"), api("/api/tags")]);
      renderAudit(audit);
      renderTags(tags);
    } catch (e) { $("#audit-summary").innerHTML = `<p class="empty">${esc(e.message)}</p>`; }
  }

  function renderAudit(audit) {
    const box = $("#audit-summary");
    if (!audit.available) {
      box.innerHTML = '<p class="empty">The audit has not been run. Run <code>python -m fairreturn.genai.llm_audit</code>.</p>';
      return;
    }
    const mock = audit.summary.some((s) => s.is_mock);
    $("#genai-banner").innerHTML = mock
      ? '<div class="banner warn">These LLM results come from the offline mock backend, which has a country bias built in to test the pipeline. Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL and re-run the audit before presenting numbers.</div>'
      : `<div class="banner info">LLM audited: ${esc(audit.llm.model)}.</div>`;
    const s = audit.summary;
    const kp = (c) => s.find((x) => x.condition === c) || {};
    box.innerHTML = `<div class="kpis">
        <div class="kpi"><strong>${(kp("plain").foreign_surcharge ?? 0) >= 0 ? "+" : ""}${num(kp("plain").foreign_surcharge, 1)}</strong><span>points of risk added when a UK order is moved abroad (plain prompt)</span></div>
        <div class="kpi"><strong>${pct(kp("plain").cf_flip_rate, 0)}</strong><span>decisions that flip when only the country changes (plain)</span></div>
        <div class="kpi"><strong>${pct(kp("tool").cf_flip_rate, 0)}</strong><span>decisions that flip with our tool</span></div>
        <div class="kpi"><strong>${pct(kp("plain").mentions_country_rate, 0)}</strong><span>plain answers whose reason mentions country or postage</span></div>
      </div><div class="two-col"><div id="audit-chart-1"></div><div id="audit-chart-2"></div></div>`;
    const rows = (key) => s.map((x) => ({ label: COND_TEXT[x.condition][0], value: x[key], color: x.condition === "tool" ? C.ours : C.baseline }));
    $("#audit-chart-1").innerHTML = "<h3>Decisions that flip between twins</h3>";
    Charts.hbars($("#audit-chart-1"), { rows: rows("cf_flip_rate"), format: (v) => pct(v, 0), tickFormat: Charts.pctTick, labelWidth: 150, aria: "Flip rate by prompt" });
    $("#audit-chart-2").innerHTML = "<h3>Average risk change between twins</h3>";
    Charts.hbars($("#audit-chart-2"), { rows: rows("mean_abs_delta_risk"), format: (v) => `${v.toFixed(1)} pts`, tickFormat: (v) => `${+v.toFixed(2)}`, labelWidth: 150, aria: "Mean risk change by prompt" });

    const list = $("#pair-list");
    list.innerHTML = "";
    audit.pairs.sort((a, b) => Math.abs(b.plain_delta || 0) - Math.abs(a.plain_delta || 0)).forEach((p, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "inv-item";
      b.innerHTML = `<span class="t">${esc(p.original.country)} to ${esc(p.twin.country)}</span>
        <span class="r">${p.plain_delta === null ? "" : `${p.plain_delta >= 0 ? "+" : ""}${p.plain_delta.toFixed(0)} pts`}</span>
        <span class="s">Pair ${esc(p.pair_id)}, ${p.returned ? "was returned" : "not returned"}</span>`;
      b.addEventListener("click", () => { list.querySelectorAll(".inv-item").forEach((x) => x.setAttribute("aria-current", "false")); b.setAttribute("aria-current", "true"); openPair(p.pair_id); });
      list.appendChild(b);
      if (i === 0) { b.setAttribute("aria-current", "true"); openPair(p.pair_id); }
    });
  }

  async function openPair(id) {
    const p = await api(`/api/audit/${encodeURIComponent(id)}`);
    const countryWords = /country|nationality|germany|german|france|french|uk\b|united kingdom|british|irish|eire|europe|foreign|abroad|international|postage|overseas/i;
    const rows = Object.entries(p.conditions).map(([c, v]) => {
      const ans = (a, who) => `<div class="answer ${countryWords.test(a.reason || "") ? "country-mention" : ""}">
          <span class="who">${who}</span><div class="risk">${a.risk === null ? "n/a" : `${Math.round(a.risk)}%`}</div>${esc(a.reason)}</div>`;
      return `<div class="answer-row"><div class="cond">${COND_TEXT[c][0]}<small>${COND_TEXT[c][1]}</small></div>
        ${ans(v.original, `As it happened: ${esc(p.original.country)}, postage ${p.original.postage ? "yes" : "no"}`)}
        ${ans(v.twin, `Twin: ${esc(p.twin.country)}, postage ${p.twin.postage ? "yes" : "no"}`)}</div>`;
    }).join("");
    $("#pair-detail").innerHTML = `<h3>Pair ${esc(p.pair_id)}: the same order line from ${esc(p.original.country)} and from ${esc(p.twin.country)}</h3>
      <p class="lede" style="margin:4px 0 10px">Actual outcome: ${p.returned ? "returned" : "not returned"}. A red edge marks a reason that mentions country or postage.</p>
      <div class="answers">${rows}</div>
      <details class="prompt"><summary>Show the plain prompt</summary><pre>${esc(p.conditions.plain.prompt_original)}</pre></details>
      <details class="prompt"><summary>Show the prompt with our tool</summary><pre>${esc(p.conditions.tool.prompt_original)}</pre></details>`;
  }

  function renderTags(t) {
    const box = $("#tag-eval");
    if (!t.available) { box.innerHTML = '<p class="empty">Run <code>python -m fairreturn.genai.product_tagger</code> to tag products with an LLM.</p>'; return; }
    const ev = t.eval;
    const fields = Object.entries(ev.llm_vs_rules || {});
    const agree = fields.map(([f, m]) => `<tr><td>${esc(f)}</td><td class="num">${m.n}</td><td class="num">${pct(m.accuracy, 1)}</td><td class="num">${num(m.kappa, 2)}</td></tr>`).join("");
    const dis = (ev.disagreement_examples || []).slice(0, 8).map((d) => `<tr><td>${esc(title(d.description))}</td><td>${esc(d.llm.replace(/_/g, " "))}</td><td>${esc(d.rules.replace(/_/g, " "))}</td></tr>`).join("");
    const human = ev.human || {};
    box.innerHTML = `${ev.is_mock ? '<div class="banner warn" style="margin-bottom:12px">Product tags from the mock backend: the pipeline works, but these are keyword rules with deliberate noise, not LLM output.</div>' : ""}
      <div class="kpis">
        <div class="kpi"><strong>${pct(ev.validity, 1)}</strong><span>LLM answers that used only allowed values</span></div>
        <div class="kpi"><strong>${Number(ev.n_products_tagged).toLocaleString()}</strong><span>products tagged by the LLM</span></div>
        <div class="kpi"><strong>${pct(ev.coverage_lines_valid_llm, 0)}</strong><span>of all purchase lines covered</span></div>
        <div class="kpi"><strong>${esc(t.tags_used_in_models || "rules")}</strong><span>tags used by the trained models</span></div>
      </div>
      <div class="two-col">
        <div><h3>Agreement with keyword rules</h3><table><thead><tr><th>Attribute</th><th class="num">Products</th><th class="num">Agreement</th><th class="num">Cohen’s κ</th></tr></thead><tbody>${agree}</tbody></table>
          <p class="note">Human check: ${esc(human.status || "not started")}. Two teammates label data/processed/human_tag_sample.csv; the agreement is then computed automatically.</p></div>
        <div><h3>Where the LLM and the rules disagree</h3><table><thead><tr><th>Product</th><th>LLM</th><th>Rules</th></tr></thead><tbody>${dis || '<tr><td colspan="3">No disagreements.</td></tr>'}</tbody></table></div>
      </div>`;
  }

  // ------------------------------------------------------------------ about
  async function loadAbout() {
    const r = state.meta.report, s = state.meta.split_sizes;
    const cell = (v, l) => `<div><strong>${v}</strong>${l}</div>`;
    const rates = Object.entries(state.meta.return_rate_by_group || {}).sort((a, b) => b[1] - a[1]).map(([g, v]) => `${g} ${pct(v, 2)}`).join(", ");
    $("#about-data").innerHTML = `<p class="lede" style="margin-bottom:10px">UCI Online Retail II: every transaction of a UK online gift wholesaler, Dec 2009 to Dec 2011 (CC BY 4.0).</p>
      <div class="datagrid">
        ${cell(Number(r.raw_rows).toLocaleString(), "raw rows in the two sheets")}
        ${cell(Number(r.duplicate_rows_removed).toLocaleString(), "duplicate rows removed, incl. the 9-day sheet overlap")}
        ${cell(pct(r.share_without_customer, 0), "rows without a customer id, dropped")}
        ${cell(Number(r.purchase_lines).toLocaleString(), "purchase lines")}
        ${cell(Number(r.cancellation_lines).toLocaleString(), "cancellation lines")}
        ${cell(pct(r.cancellations_matched_share, 0), "cancellations matched to a purchase")}
        ${cell(`${r.lag_days_median} days`, "median time to return")}
        ${cell(Number(s.train.lines).toLocaleString(), `training lines (Mar 2010 to Feb 2011), ${pct(s.train.return_rate, 2)} returned`)}
        ${cell(Number(s.test.lines).toLocaleString(), `test lines (Jun to Oct 2011), ${pct(s.test.return_rate, 2)} returned`)}
      </div><p class="note">Return rate by country group: ${esc(rates)}.</p>`;
    try {
      const docs = await api("/api/docs");
      $("#about-api").innerHTML = `<div class="api-list">${docs.map((d) => `<div><code>${esc(d.method)} ${esc(d.path)}</code> ${esc(d.description)}</div>`).join("")}</div>`;
    } catch (_e) { /* optional */ }
  }

  init();
})();
