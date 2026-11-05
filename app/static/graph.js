// Knowledge graph (after nashsu/llm_wiki): sigma.js + graphology, ForceAtlas2 layout, Louvain
// communities. Size = links (square root), colour = area / community / type, edge width =
// relevance (links ×3, parent/child ×2, shared sources ×4, common neighbours, same type).
// Hover a node to see its neighbourhood; click to pin it and see why pages are related.
(() => {
  "use strict";
  const box = document.getElementById("graph");
  if (!box || typeof Sigma === "undefined" || typeof graphology === "undefined") return;
  const lib = graphologyLibrary;
  const Graph = graphology.Graph || graphology;
  const css = getComputedStyle(document.documentElement);
  const v = (n) => css.getPropertyValue(n).trim();
  const PALETTE = ["#f2a516", "#4e9af1", "#3fbf7f", "#e0605e", "#a97ff0", "#2fc4c4",
                   "#e58ac7", "#9fbf3f", "#f5e05a", "#7c8cf5", "#c9a86a", "#58b3e8"];
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const store = { get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
                  set(k, x) { try { localStorage.setItem(k, JSON.stringify(x)); } catch {} } };

  fetch("/graph.json").then((r) => r.json()).then((data) => {
    const g = new Graph({ type: "undirected" });
    for (const { type, ...n } of data.nodes) g.addNode(n.id, { ...n, kind: type, label: n.title, x: 0, y: 0, size: 1 });   // sigma reserves "type"
    for (const e of data.edges) g.addEdge(e.source, e.target, { weight: e.weight, why: e.why });

    // Layout: cached positions keep the picture stable between visits; new pages start near
    // their parent (or their area's centre) and ForceAtlas2 settles everything.
    const cached = store.get("kb-graph-pos") || {};
    const areas = [...new Set(data.nodes.map((n) => n.folder))].sort();
    g.forEachNode((id, a) => {
      const i = areas.indexOf(a.folder), t = (2 * Math.PI * i) / areas.length;
      const p = cached[id] || cached[id.split("/").slice(0, -1).join("/")];
      g.mergeNodeAttributes(id, p ? { x: p[0] + Math.random(), y: p[1] + Math.random() }
                                  : { x: 100 * Math.cos(t) + 10 * Math.random(), y: 100 * Math.sin(t) + 10 * Math.random() });
    });
    const allCached = data.nodes.every((n) => cached[n.id]);
    lib.layoutForceAtlas2.assign(g, { iterations: allCached ? 40 : 500, getEdgeWeight: "weight",
      settings: { ...lib.layoutForceAtlas2.inferSettings(g), gravity: 0.6, scalingRatio: 12, barnesHutOptimize: false,
                  edgeWeightInfluence: 0.6, linLogMode: false } });
    const pos = {};
    g.forEachNode((id, a) => (pos[id] = [Math.round(a.x * 10) / 10, Math.round(a.y * 10) / 10]));
    store.set("kb-graph-pos", pos);

    // Communities and graph insights.
    lib.communitiesLouvain.assign(g, { getEdgeWeight: "weight", resolution: 1 });
    const comms = {};
    g.forEachNode((id, a) => (comms[a.community] ||= []).push(id));
    const commName = {}, cohesion = {};
    for (const [c, ids] of Object.entries(comms)) {
      const top = ids.slice().sort((x, y) => g.degree(y) - g.degree(x))[0];
      commName[c] = g.getNodeAttribute(top, "title");
      let inside = 0;
      for (const x of ids) for (const y of g.neighbors(x)) if (g.getNodeAttribute(y, "community") == c) inside++;
      cohesion[c] = ids.length > 1 ? inside / 2 / ((ids.length * (ids.length - 1)) / 2) : 1;
    }
    g.forEachNode((id, a) => g.mergeNodeAttributes(id, {
      size: a.portal ? 15 : (a.hub ? 9 : 4) + 2.2 * Math.sqrt(a.degree),
    }));

    // Colouring.
    let mode = store.get("kb-graph-mode") || "folder";
    const groupOf = (a) => (mode === "community" ? String(a.community) : mode === "type" ? a.kind : a.folder);
    const groups = () => {
      const count = {};
      g.forEachNode((_, a) => (count[groupOf(a)] = (count[groupOf(a)] || 0) + 1));
      return Object.keys(count).sort((x, y) => count[y] - count[x]).map((k, i) => ({ key: k, n: count[k], colour: PALETTE[i % PALETTE.length] }));
    };
    let colours = {};
    const recolour = () => {
      colours = Object.fromEntries(groups().map((x) => [x.key, x.colour]));
      g.forEachNode((id, a) => g.setNodeAttribute(id, "color", a.status === "deprecated" ? v("--muted") : colours[groupOf(a)]));
      legend();
    };

    // State: hovered / pinned node, isolated group, highlighted set (insights), search.
    let hovered = null, pinned = null, onlyGroup = null, marked = null;
    let showAll = store.get("kb-graph-labels") === true;
    const focus = () => hovered || pinned;
    const edgeBase = v("--border"), accent = v("--accent"), muted = v("--muted");
    const renderer = new Sigma(g, box, {
      labelColor: { color: v("--text") }, labelFont: "system-ui, sans-serif", labelSize: 12, labelWeight: "600",
      labelDensity: 0.9, labelGridCellSize: 70, labelRenderedSizeThreshold: showAll ? 0 : 9,
      defaultEdgeColor: edgeBase, zIndex: true, minCameraRatio: 0.05, maxCameraRatio: 4,
      // Hovered label in the page's colours (sigma's default is a white box, unreadable in dark mode).
      hoverRenderer(ctx, d, s) {
        const size = s.labelSize, font = `${s.labelWeight} ${size}px ${s.labelFont}`;
        ctx.font = font;
        const w = d.label ? ctx.measureText(d.label).width + 12 : 0, h = size + 10;
        ctx.fillStyle = v("--surface-2"); ctx.strokeStyle = accent; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(d.x + d.size + 4, d.y - h / 2, w, h, 6) : ctx.rect(d.x + d.size + 4, d.y - h / 2, w, h);
        if (w) { ctx.fill(); ctx.stroke(); }
        ctx.beginPath(); ctx.arc(d.x, d.y, d.size + 3, 0, Math.PI * 2); ctx.strokeStyle = accent; ctx.lineWidth = 2; ctx.stroke();
        if (d.label) { ctx.fillStyle = v("--text"); ctx.fillText(d.label, d.x + d.size + 10, d.y + size / 3); }
      },
      nodeReducer(id, a) {
        const out = { ...a };
        const f = focus();
        const dim = (f && id !== f && !g.areNeighbors(id, f))
          || (onlyGroup !== null && groupOf(a) !== onlyGroup)
          || (marked && !marked.has(id));
        if (dim) { out.color = edgeBase; out.label = ""; out.zIndex = 0; }
        else { out.zIndex = 1; if (f && (id === f || g.areNeighbors(id, f))) out.forceLabel = true; }
        if (marked && marked.has(id)) out.forceLabel = true;
        if (id === f) { out.highlighted = true; out.zIndex = 2; }
        return out;
      },
      edgeReducer(e, a) {
        const out = { ...a, size: Math.min(0.6 + a.weight * 0.35, 5) };
        const f = focus();
        const [s, t] = g.extremities(e);
        if (f) {
          if (s === f || t === f) { out.color = accent; out.zIndex = 1; } else out.hidden = true;
        } else if (onlyGroup !== null || marked) {
          const keep = (x) => (onlyGroup === null || groupOf(g.getNodeAttributes(x)) === onlyGroup) && (!marked || marked.has(x));
          if (!keep(s) || !keep(t)) out.hidden = true;
        } else out.color = a.weight >= 6 ? muted : edgeBase;
        return out;
      },
    });
    const refresh = () => renderer.refresh();

    // Info panel: what this page is, and its neighbours by relevance with the reason.
    const why = { link: "links", child: "parent/child", source: "shared source", cites: "cites" };
    const info = (id) => {
      const panel = $("graph-info");
      if (!id) { panel.innerHTML = `<p class="muted">Hover a page to see its neighbourhood, click to pin it. Scroll to zoom, drag to pan.</p>`; return; }
      const a = g.getNodeAttributes(id);
      const nb = g.edges(id).map((e) => ({ id: g.opposite(id, e), ...g.getEdgeAttributes(e) })).sort((x, y) => y.weight - x.weight);
      panel.innerHTML = `<h3><a href="/c/${esc(id)}">${esc(a.title)}</a></h3>
        <p class="muted"><code>${esc(id)}</code> · ${esc(a.kind)}${a.status !== "stable" ? " · " + esc(a.status) : ""}${a.stale ? " · due for re-check" : ""}</p>
        <p>${esc(a.description)}</p>
        <p class="muted">${nb.length} connection${nb.length === 1 ? "" : "s"} · cluster “${esc(commName[a.community])}”</p>
        <ul class="nb">${nb.map((x) => `<li><a href="#" data-node="${esc(x.id)}">${esc(g.getNodeAttribute(x.id, "title"))}</a>
          <span class="muted">${x.weight.toFixed(1)} · ${x.why.map((w) => why[w] || w).join(", ")}</span></li>`).join("")}</ul>
        <p><a class="btn small primary" href="/c/${esc(id)}">Open page</a></p>`;
    };
    $("graph-info").addEventListener("click", (ev) => {
      const t = ev.target.closest("[data-node]");
      if (t) { ev.preventDefault(); pin(t.dataset.node); }
    });
    const camera = renderer.getCamera();
    const goTo = (id) => {
      const d = renderer.getNodeDisplayData(id);
      if (d) camera.animate({ x: d.x, y: d.y, ratio: Math.min(camera.ratio, 0.45) }, { duration: 400 });
    };
    const pin = (id) => { pinned = id; marked = null; info(id); goTo(id); refresh(); };

    renderer.on("enterNode", ({ node }) => { hovered = node; info(node); refresh(); box.style.cursor = "pointer"; });
    renderer.on("leaveNode", () => { hovered = null; info(pinned); refresh(); box.style.cursor = ""; });
    renderer.on("clickNode", ({ node }) => (pinned === node ? (location.href = "/c/" + node) : pin(node)));
    renderer.on("doubleClickNode", ({ node, preventSigmaDefault }) => { preventSigmaDefault(); location.href = "/c/" + node; });
    renderer.on("clickStage", () => { pinned = null; marked = null; onlyGroup = null; info(null); legend(); refresh(); });

    // Legend: one row per colour with its count; click to show only that group.
    function legend() {
      const rows = groups().map((x) => {
        const label = mode === "community" ? `${commName[x.key]} (${Math.round(cohesion[x.key] * 100)}% cohesion)` : x.key;
        return `<li><button type="button" class="${onlyGroup === x.key ? "on" : ""}" data-group="${esc(x.key)}">
          <i style="background:${x.colour}"></i>${esc(label)}<span class="muted"> ${x.n}</span></button></li>`;
      });
      $("graph-legend").innerHTML = rows.join("");
    }
    $("graph-legend").addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-group]");
      if (!b) return;
      onlyGroup = onlyGroup === b.dataset.group ? null : b.dataset.group;
      pinned = null; marked = null; legend(); refresh();
    });

    // Controls.
    $("graph-mode").value = mode;
    $("graph-mode").addEventListener("change", (ev) => { mode = ev.target.value; store.set("kb-graph-mode", mode); onlyGroup = null; recolour(); refresh(); });
    $("graph-labels").checked = showAll;
    $("graph-labels").addEventListener("change", (ev) => {
      showAll = ev.target.checked; store.set("kb-graph-labels", showAll);
      renderer.setSetting("labelRenderedSizeThreshold", showAll ? 0 : 9);
    });
    $("graph-in").onclick = () => camera.animatedZoom({ duration: 250 });
    $("graph-out").onclick = () => camera.animatedUnzoom({ duration: 250 });
    $("graph-fit").onclick = () => camera.animatedReset({ duration: 300 });
    const search = $("graph-search");
    $("graph-titles").innerHTML = data.nodes.map((n) => `<option value="${esc(n.title)}">`).join("");
    search.addEventListener("input", () => {
      const q = search.value.trim().toLowerCase();
      if (!q) { marked = null; refresh(); return; }
      const hits = data.nodes.filter((n) => (n.title + " " + n.id).toLowerCase().includes(q)).map((n) => n.id);
      marked = new Set(hits); pinned = null;
      const exact = data.nodes.find((n) => n.title.toLowerCase() === q);
      if (exact) pin(exact.id); else refresh();
    });

    // Insights: isolated pages, bridges between clusters, loose clusters.
    const isolated = g.filterNodes((id, a) => !a.portal && g.degree(id) <= 1);
    const bridges = g.filterNodes((id) => new Set(g.neighbors(id).map((x) => g.getNodeAttribute(x, "community"))).size >= 3)
      .sort((x, y) => g.degree(y) - g.degree(x)).slice(0, 8);
    const sparse = Object.entries(comms).filter(([c, ids]) => ids.length >= 3 && cohesion[c] < 0.15);
    const item = (ids, text) => `<li><button type="button" data-mark="${esc(ids.join("|"))}">${text}</button></li>`;
    $("graph-insights").innerHTML = [
      isolated.length ? `<h3>Barely connected <span class="muted">${isolated.length}</span></h3><ul>${isolated.map((id) => item([id], esc(g.getNodeAttribute(id, "title")))).join("")}</ul>` : "",
      bridges.length ? `<h3>Bridges between clusters</h3><ul>${bridges.map((id) => item([id, ...g.neighbors(id)], esc(g.getNodeAttribute(id, "title")))).join("")}</ul>` : "",
      sparse.length ? `<h3>Loose clusters</h3><ul>${sparse.map(([c, ids]) => item(ids, `${esc(commName[c])} · ${ids.length} pages, ${Math.round(cohesion[c] * 100)}% cohesion`)).join("")}</ul>` : "",
    ].join("") || `<p class="muted">No isolated pages, bridges or loose clusters.</p>`;
    $("graph-insights").addEventListener("click", (ev) => {
      const b = ev.target.closest("[data-mark]");
      if (!b) return;
      const ids = b.dataset.mark.split("|");
      if (ids.length === 1) return pin(ids[0]);
      pinned = null; onlyGroup = null; marked = new Set(ids); goTo(ids[0]); refresh();
    });

    $("count").textContent = `${data.nodes.length} pages, ${data.edges.length} connections, ${Object.keys(comms).length} clusters`;
    recolour(); info(null); refresh();
  });
})();
