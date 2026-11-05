"""The bundle as a weighted graph: what is related to what, and how strongly.

Relevance between two pages (after nashsu/llm_wiki) adds up four signals:
  direct link (either way), or a citation of a raw source  ×3
  parent/child in the tree (hub)      ×2
  shared source (a source cited by ≤ 6 pages; ones cited everywhere say nothing)  ×4 each
  common neighbours (Adamic-Adar)     ×1.5
  same type, only on top of another signal  ×1
Used by the web graph (/graph.json) and kb_read's `related` list.
"""

import math
import posixpath
from collections import defaultdict
from itertools import combinations

from . import okf

W_LINK, W_TREE, W_SOURCE, W_AA, W_TYPE = 3.0, 2.0, 4.0, 1.5, 1.0
SOURCE_MAX_PAGES = 6


def _source_key(src: dict) -> str:
    return str(src.get("resource") or "").strip().lower().rstrip("/")


def build(docs: list[tuple[str, dict, str]]) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    """(nodes, edges): nodes by id with degree; edges keyed (a, b) with a < b, each with
    `weight` and the `why` signals."""
    known = {c: m for c, m, _ in docs}
    nbrs: dict[str, set[str]] = defaultdict(set)
    edges: dict[tuple[str, str], dict] = {}

    def edge(a: str, b: str) -> dict:
        key = (a, b) if a < b else (b, a)
        return edges.setdefault(key, {"weight": 0.0, "why": []})

    for c, _, body in docs:
        for l in okf.links(c, body):
            if l in known and l != c:
                e = edge(c, l)
                if "link" not in e["why"]:
                    e["weight"] += W_LINK
                    e["why"].append("link")
                nbrs[c].add(l)
                nbrs[l].add(c)
        parent = posixpath.dirname(c)
        if parent in known:
            e = edge(c, parent)
            e["weight"] += W_TREE
            e["why"].append("child")
            nbrs[c].add(parent)
            nbrs[parent].add(c)

    for c, m, _ in docs:                       # a page citing a raw source in sources/ (the citation layer)
        for src in okf.cited(m, known):
            if src != c:
                e = edge(c, src)
                e["weight"] += W_LINK
                e["why"].append("cites")
                nbrs[c].add(src)
                nbrs[src].add(c)

    cited: dict[str, list[str]] = defaultdict(list)
    for c, m, _ in docs:
        for key in {_source_key(s) for s in m.get("sources") or [] if isinstance(s, dict)} - {""}:
            cited[key].append(c)
    for pages in cited.values():
        if 2 <= len(pages) <= SOURCE_MAX_PAGES:
            for a, b in combinations(sorted(pages), 2):
                e = edge(a, b)
                e["weight"] += W_SOURCE
                if "source" not in e["why"]:
                    e["why"].append("source")

    for (a, b), e in edges.items():
        aa = sum(1 / math.log(len(nbrs[z])) for z in nbrs[a] & nbrs[b] if len(nbrs[z]) > 1)
        if aa:
            e["weight"] += W_AA * aa
        if known[a].get("type") == known[b].get("type"):
            e["weight"] += W_TYPE
        e["weight"] = round(e["weight"], 2)

    nodes = {}
    for c, m in known.items():
        nodes[c] = {"id": c, "title": m.get("title") or posixpath.basename(c), "type": m.get("type", "?"),
                    "folder": c.split("/", 1)[0] if "/" in c else c,
                    "status": okf.signals(m)["status"], "trust": okf.trust_tier(m), "stale": okf.is_stale(m),
                    "description": m.get("description") or "",
                    "portal": "/" not in c, "hub": any(posixpath.dirname(k) == c for k in known),
                    "degree": len(nbrs[c])}
    return nodes, edges


def related(cid: str, docs: list[tuple[str, dict, str]], k: int = 6) -> list[dict]:
    """The pages most related to `cid`, strongest first, including ones it doesn't link to
    but shares sources or neighbours with."""
    _, edges = build(docs)
    nbrs: dict[str, set[str]] = defaultdict(set)
    for a, b in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
    scores = {}
    for (a, b), e in edges.items():
        if cid in (a, b):
            scores[b if a == cid else a] = (e["weight"], e["why"])
    for other in {z2 for z in nbrs[cid] for z2 in nbrs[z]} - nbrs[cid] - {cid}:
        aa = sum(1 / math.log(len(nbrs[z])) for z in nbrs[cid] & nbrs[other] if len(nbrs[z]) > 1)
        if aa >= 0.8:
            scores[other] = (round(W_AA * aa, 2), ["neighbours"])
    titles = {c: m.get("title") or c for c, m, _ in docs}
    top = sorted(scores.items(), key=lambda x: -x[1][0])[:k]
    return [{"id": c, "title": titles[c], "score": s, "why": why} for c, (s, why) in top]
