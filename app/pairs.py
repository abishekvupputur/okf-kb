"""Pages close in meaning that don't link each other, and how well the links follow meaning.

Each page is the mean of its section vectors (scrubbed of footnotes and link targets, embed.scrub).
A pair is a candidate when one page is among the other's K nearest pages, neither links the other,
and they aren't parent and child. Neighbour rank instead of a cosine floor, so it works on every
model's scale. Measured 2026-11-10 (stella, 108 pages): a page's nearest page was already linked for
89 % of pages, its 3 nearest for 70 %; the rest are the candidates.

A pair judged as needing no link is recorded with a fingerprint of both pages
(`.kb/no-link.json` in the bundle, committed with it) and stays quiet until either page changes.
"""

import hashlib
import json
import posixpath

import numpy as np

from . import embed, okf

K = 3
DECISIONS = ".kb/no-link.json"
SKIP = ("archive/", "sources/", "journal/")


def key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def fingerprint(meta: dict, body: str) -> str:
    return hashlib.sha1(okf.dump(meta, body).encode()).hexdigest()[:16]


def load(root) -> dict:
    try:
        return json.loads((root / DECISIONS).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def record(root, docs, a: str, b: str, reason: str, by: str) -> None:
    """Note that a and b need no link, as of their current text. Caller commits."""
    pages = {c: (m, body) for c, m, body in docs}
    d = load(root)
    d[key(a, b)] = {"reason": reason, "by": by, "at": okf.now_iso(),
                    "fp": {c: fingerprint(*pages[c]) for c in (a, b)}}
    (root / DECISIONS).parent.mkdir(exist_ok=True)
    (root / DECISIONS).write_text(json.dumps(d, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def analyse(user_dir: str, root, docs) -> dict | None:
    """The pair check for one bundle, or None while embeddings are unavailable or building."""
    cur = [(c, m, b) for c, m, b in docs if m.get("type") != "Source" and not c.startswith(SKIP)
           and okf.signals(m)["status"] != "deprecated"]
    ids = [c for c, _, _ in cur]
    if len(ids) < 2:
        return None
    S = embed.page_matrix(user_dir, docs, ids)
    if S is None:
        return None
    ix = {c: i for i, c in enumerate(ids)}
    linked = {key(c, l) for c, _, b in cur for l in okf.links(c, b) if l in ix and l != c}
    family = {key(c, posixpath.dirname(c)) for c in ids if posixpath.dirname(c) in ix}
    n = len(ids)
    order = np.argsort(-S, axis=1)
    rank = np.empty((n, n), int)
    for i in range(n):
        o = order[i][order[i] != i]
        rank[i, o] = np.arange(1, n)
        rank[i, i] = 0
    # How well links follow meaning: of each linked page's 1 / K nearest pages, the share it links.
    with_links = [i for i in range(n) if any(key(ids[i], ids[j]) in linked for j in range(n) if j != i)]
    def p_at(k):
        hits = [sum(key(ids[i], ids[j]) in linked for j in order[i][order[i] != i][:k]) / k for i in with_links]
        return round(float(np.mean(hits)), 3) if hits else None
    fps = {c: fingerprint(m, b) for c, m, b in cur}
    decided = load(root)
    candidates, dismissed = [], []
    for i in range(n):
        for j in range(i + 1, n):
            r = int(min(rank[i, j], rank[j, i]))
            k = key(ids[i], ids[j])
            if r > K or k in linked or k in family:
                continue
            pair = {"a": ids[i], "b": ids[j], "similarity": round(float(S[i, j]), 3), "rank": r}
            d = decided.get(k)
            if d and d.get("fp") == {ids[i]: fps[ids[i]], ids[j]: fps[ids[j]]}:
                dismissed.append({**pair, "reason": d.get("reason")})
            else:
                if d:
                    pair["recheck"] = {"reason": d.get("reason"), "at": d.get("at"), "why": "a page changed since"}
                candidates.append(pair)
    candidates.sort(key=lambda p: -p["similarity"])
    return {"ids": ids, "matrix": S, "linked": linked, "family": family, "candidates": candidates,
            "dismissed": dismissed,
            "match": {"pages": n, "pages_with_links": len(with_links), "nearest_linked": p_at(1),
                      f"top{K}_linked": p_at(K), "random_pair_linked": round(len(linked) / (n * (n - 1) / 2), 3)}}
