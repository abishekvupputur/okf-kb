// /pairs: heatmap of meaning similarity between pages (from /pairs.json), with links, open and dismissed pairs.
(() => {
  const cv = document.getElementById("pairs-map"), tip = document.getElementById("pairs-tip");
  if (!cv) return;
  const LO = 0.3, HI = 0.95;
  // dark blue -> teal -> yellow; readable on both themes
  const stops = [[24, 32, 72], [33, 110, 140], [80, 180, 130], [250, 220, 90]];
  const color = (v) => {
    const t = Math.max(0, Math.min(1, (v - LO) / (HI - LO))) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i;
    return `rgb(${stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * f)).join(",")})`;
  };
  let d = null;
  const draw = () => {
    if (!d || !d.ids) return;
    const css = getComputedStyle(document.documentElement);
    const fg = css.getPropertyValue("--text").trim(), muted = css.getPropertyValue("--muted").trim();
    const accent = css.getPropertyValue("--accent").trim(), bg = css.getPropertyValue("--bg").trim();
    const n = d.ids.length, area = d.ids.map((c) => c.split("/")[0]);
    const lw = Math.min(110, Math.max(64, cv.parentElement.clientWidth * 0.12));
    const size = Math.max(200, Math.min(900, cv.parentElement.clientWidth) - lw), cell = size / n;
    const dpr = window.devicePixelRatio || 1;
    cv.width = (size + lw) * dpr; cv.height = (size + lw) * dpr;
    cv.style.width = cv.style.height = (size + lw) + "px";
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = bg; g.fillRect(0, 0, size + lw, size + lw);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      g.fillStyle = i === j ? muted : color(d.cos[i][j]);
      g.fillRect(lw + j * cell, lw + i * cell, Math.ceil(cell), Math.ceil(cell));
    }
    const both = (list, fn) => list.forEach(([a, b]) => { fn(a, b); fn(b, a); });
    g.lineWidth = Math.max(0.8, cell * 0.16);
    g.strokeStyle = fg;
    both(d.linked, (i, j) => { g.beginPath(); g.arc(lw + (j + .5) * cell, lw + (i + .5) * cell, Math.max(1.2, cell * .28), 0, 7); g.stroke(); });
    g.strokeStyle = accent;
    both(d.candidates, (i, j) => g.strokeRect(lw + j * cell + 1, lw + i * cell + 1, cell - 2, cell - 2));
    g.strokeStyle = muted;
    both(d.dismissed, (i, j) => {
      const x = lw + j * cell, y = lw + i * cell;
      g.beginPath(); g.moveTo(x + 1, y + 1); g.lineTo(x + cell - 1, y + cell - 1); g.moveTo(x + cell - 1, y + 1); g.lineTo(x + 1, y + cell - 1); g.stroke();
    });
    g.font = "11px " + css.getPropertyValue("--font"); g.strokeStyle = fg; g.lineWidth = 1;
    area.forEach((a, i) => {
      if (i && area[i - 1] === a) return;
      let e = i; while (e < n && area[e] === a) e++;
      const x = lw + i * cell, w = (e - i) * cell;
      g.strokeRect(x, x, w, w);
      if (w < 10) return;
      g.fillStyle = muted; g.textBaseline = "middle";
      g.textAlign = "right"; g.fillText(a, lw - 6, x + w / 2);
      g.save(); g.translate(x + w / 2, lw - 6); g.rotate(-Math.PI / 2); g.textAlign = "left"; g.fillText(a, 0, 0); g.restore();
    });
    cv._geo = { lw, cell, n };
  };
  const at = (e) => {
    const r = cv.getBoundingClientRect(), { lw, cell, n } = cv._geo || {};
    if (!cell) return null;
    const j = Math.floor((e.clientX - r.left - lw) / cell), i = Math.floor((e.clientY - r.top - lw) / cell);
    return i >= 0 && j >= 0 && i < n && j < n ? [i, j] : null;
  };
  const has = (list, i, j) => list.some(([a, b]) => (a === i && b === j) || (a === j && b === i));
  cv.addEventListener("mousemove", (e) => {
    const p = at(e);
    if (!p) { tip.hidden = true; return; }
    const [i, j] = p;
    const state = i === j ? "same page" : has(d.linked, i, j) ? "linked" : has(d.candidates, i, j) ? "open pair: not linked"
      : has(d.dismissed, i, j) ? "dismissed: no link needed" : "not linked";
    tip.replaceChildren(...[d.titles[i], d.titles[j], `cos ${d.cos[i][j].toFixed(3)} · ${state}`].map((t) => {
      const el = document.createElement("div"); el.textContent = t; return el; }));
    const wr = cv.parentElement.getBoundingClientRect();
    tip.hidden = false;
    tip.style.left = Math.min(wr.width - tip.offsetWidth, e.clientX - wr.left + 12) + "px";
    tip.style.top = (e.clientY - wr.top + 14) + "px";
  });
  cv.addEventListener("mouseleave", () => { tip.hidden = true; });
  cv.addEventListener("click", (e) => { const p = at(e); if (p) location.href = "/c/" + d.ids[p[1]]; });
  fetch("/pairs.json").then((r) => r.json()).then((j) => { d = j; draw(); });
  let t; addEventListener("resize", () => { clearTimeout(t); t = setTimeout(draw, 150); });
})();
