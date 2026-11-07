"""MCP tools: each signed-in user works in their own OKF bundle.

The caller is identified from the OAuth token (OIDC subject + email). Writes
from Claude are recorded as `generated.by: <author>` (an agent actor); only a person
confirming content makes it `human:` — see kb_verify.
"""

import asyncio
import difflib
import hashlib
import os
import posixpath
import re
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_access_token
from pydantic import Field

from . import embed, graph, guard, okf, search, store
from .store import Bundle, User

# Exact facts (search, identifiers, old values) are found by keywords; embeddings (embed.py) only
# add what is related by meaning. Cosine thresholds for the model in embed.py. They were set for
# bge-small-en-v1.5 on a real bundle (exact copy 0.97, reworded duplicate 0.84, related pages ~0.80-0.89,
# a short claim ~0.70-0.76 against the section stating it) and carried over to each later model by
# quantile: the same share of section pairs (or question->section scores for SEED_MIN) lies above them.
# Current values: stella-en-400M-v5, mapped 2026-11-05 from bge-small's 0.60 / 0.65 / 0.70 / 0.75 / 0.80 / 0.93
# on 549 sections (298,068 pairs) and the 40 benchmark questions.
SEED_MIN = 0.37        # kb_context: a page this close to the question becomes a starting page (question->section)
TEXT_MIN = 0.54        # kb_search(similar_to=<text>) default floor
INGEST_MIN = 0.61      # kb_ingest: pages like the source's slices
PAGE_MIN = 0.69        # kb_search(similar_to=<page>) default floor
TOPIC_MIN = 0.76       # after an edit or ingest: a section elsewhere about the same thing
DUPLICATE_MIN = 0.91   # kb_write: an existing page very likely covers the new one
# kb_context's "no evidence" gate and ranking (set 2026-11-05 on the 40 benchmark questions and 15 off-topic
# ones: answerable questions' best section scored >= 0.471, off-topic ones <= 0.387).
ABSTAIN_SIM = 0.43     # abstain when the best section is below this by meaning ...
ABSTAIN_COV = 0.6      # ... and no page holds this share of the question's term weight (idf)
SEM_LOW, SEM_FULL = 0.30, 0.65   # meaning similarity mapped onto 0..1 to rank it next to keyword coverage
if embed.MODEL == "voyage-context-4":
    # The values above mapped by quantile (2026-11-06; 521 sections, 80 questions + 15 off-topic; kit's
    # map_thresholds.py). ABSTAIN_SIM 0.24 instead of the mapped 0.259: answerable questions' best section
    # >= 0.291, off-topic <= 0.218; 0.24 lets 3 more questions that share no words with their page through.
    SEED_MIN, TEXT_MIN, INGEST_MIN, PAGE_MIN, TOPIC_MIN, DUPLICATE_MIN = 0.181, 0.259, 0.348, 0.472, 0.653, 0.889
    ABSTAIN_SIM, SEM_LOW, SEM_FULL = 0.24, 0.106, 0.554
PAGE_SHARE = 0.35      # one page's share of the budget at most (at least 900 characters)
RERANK_TOP = 6         # kb_context: how many of the best candidate pages the reranker (KB_RERANKER) re-orders
                       # (12 until 2026-11-06: whole pages made that ~20-36K rerank tokens per question)

PUBLIC_URL = os.environ["PUBLIC_URL"].rstrip("/")
DEFAULT_AUTHOR = "claude/kb-mcp"
SOURCES_PORTAL = (
    {"type": "Portal", "title": "Sources",
     "description": "Raw sources as ingested (articles, documents, transcripts), stored unchanged; the pages that use them cite them.",
     "tags": ["portal", "sources"]},
    "Raw material behind the knowledge base, each stored exactly as it was ingested with `kb_ingest`. "
    "These pages never change; what was learned from them lives in the pages that cite them.\n")
NAVIGATE = ("Drill down instead of reading everything: kb_list(folder) for one level, "
            "kb_read(hub) for an entity's summary and its children, kb_read(id, section=…) "
            "for one part of a long page, kb_search for a specific fact.")


async def _me() -> User:
    token = get_access_token()
    if token is None:
        raise ToolError("not signed in")
    try:
        return await store.user_for(token.claims or {})
    except PermissionError as e:
        raise ToolError(str(e)) from None


async def _run(fn, *args):
    try:
        return await asyncio.to_thread(fn, *args)
    except okf.ConceptError as e:
        raise ToolError(str(e)) from None


def _summary(cid: str, meta: dict) -> dict:
    s = okf.signals(meta)
    return {
        "id": cid,
        "type": meta.get("type"),
        "title": meta.get("title") or posixpath.basename(cid),
        "description": meta.get("description"),
        "tags": meta.get("tags", []),
        "status": s["status"],
        "trust": s["trust"],
        "stale": s["stale"],
        "updated": (meta.get("generated") or {}).get("at") if isinstance(meta.get("generated"), dict) else None,
    }


def _updated(meta: dict) -> str | None:
    gen = meta.get("generated")
    return gen.get("at") if isinstance(gen, dict) else None


def _brief(cid: str, meta: dict) -> dict:
    """An index line: enough to decide whether to open the concept."""
    s = okf.signals(meta)
    out = {"id": cid, "title": meta.get("title") or posixpath.basename(cid),
           "type": meta.get("type"), "description": meta.get("description")}
    if s["status"] != "stable":
        out["status"] = s["status"]
    if s["stale"]:
        out["stale"] = True
    return out


def _hit_section(body: str, terms: list[str]) -> str | None:
    """Heading of the section holding the first search hit, to read with kb_read(section=…)."""
    low = body.lower()
    pos = min((low.find(t) for t in terms if t in low), default=-1)
    if pos < 0:
        return None
    inside = [h for h in okf.sections(body) if h["start"] <= pos < h["end"]]
    return inside[-1]["heading"] if inside else None


def _matches(meta: dict, type_: str | None, tag: str | None, status: str | None) -> bool:
    if type_ and str(meta.get("type", "")).lower() != type_.lower():
        return False
    if tag and tag.lower() not in [t.lower() for t in meta.get("tags", [])]:
        return False
    if status and okf.signals(meta)["status"] != status:
        return False
    return True


def _snippet(body: str, terms: list[str]) -> str:
    low = body.lower()
    pos = min((low.find(t) for t in terms if t in low), default=-1)
    start = max(0, pos - 80) if pos >= 0 else 0
    text = re.sub(r"\s+", " ", body[start:start + 240]).strip()
    return ("…" if start else "") + text


def _section_at(body: str, pos: int) -> str | None:
    inside = [h for h in okf.sections(body) if h["start"] <= pos < h["end"]]
    return inside[-1]["heading"] if inside else None


def _page_text(meta: dict, body: str) -> str:
    """What an edit can change that other pages may repeat: body, description, infobox values."""
    box = " ".join(f"{k} {v}" for k, v in (meta.get("infobox") or {}).items())
    return f"{meta.get('description', '')}\n{box}\n{body}"


def _check_also(user_dir: str, cid: str, old_meta: dict, old_body: str, meta: dict, body: str, docs) -> dict:
    """After an edit: where else the knowledge base may now be out of date.
    Accuracy: values the edit removed (keywords) that other pages still state.
    Relations: sections elsewhere about the same thing as the sections the edit changed (meaning).
    Journal days and archived pages record history and are skipped. Never raises."""
    out: dict = {}
    try:
        gone = search.removed_values(_page_text(old_meta, old_body), _page_text(meta, body))[:10]
        stale = []
        for c, m, other in docs:
            if c == cid or m.get("type") == "Source" or c.startswith(("journal/", "archive/", "sources/")):
                continue
            for t in gone:
                pos = search.find_value(other, t)   # whole token: 0.100 is not inside 100.100.100.100
                if pos >= 0:
                    stale.append({"id": c, "section": _section_at(other, pos), "still_says": t})
        if stale:
            out["old_values_elsewhere"] = stale[:15]
        changed = embed.changed_passages(cid, old_meta, old_body, meta, body)[:3]
        if changed:
            seen = []
            for c, sim, their, mine in embed.similar(user_dir, docs, cid, changed, 8, TOPIC_MIN):
                if not c.startswith(("journal/", "archive/", "sources/")):
                    seen.append({"id": c, "section": their, "similarity": sim, "your_section": mine or "(lead)"})
            if seen:
                out["same_topic_elsewhere"] = seen[:8]
        if out:
            out["check_also"] = ("old_values_elsewhere: exact values this edit removed that these pages still state "
                                 "(update them unless they describe history). same_topic_elsewhere: sections about "
                                 "the same thing by meaning; read them, they may need the same change.")
    except Exception as e:   # a hint must never fail a write
        out = {"check_also_error": str(e)[:200]}
    return out


def _backlinks(docs: list[tuple[str, dict, str]], cid: str) -> list[str]:
    return [other for other, _, body in docs if other != cid and cid in okf.links(other, body)]


def register(mcp: FastMCP) -> None:

    async def _overview() -> dict:
        """Start here. Summary of your knowledge bundle: how many concepts of each type,
        the most used tags, what changed recently, and how much needs review (stale,
        draft, unverified). Also returns the bundle's root index."""
        me = await _me()
        b = Bundle(me)
        docs = await _run(b.all)
        types: dict[str, int] = {}
        tags: dict[str, int] = {}
        trust = {"unverified": 0, "machine-confirmed": 0, "human-reviewed": 0}
        stale = draft = 0
        for _, meta, _ in docs:
            types[str(meta.get("type", "?"))] = types.get(str(meta.get("type", "?")), 0) + 1
            for t in meta.get("tags", []):
                tags[t] = tags.get(t, 0) + 1
            s = okf.signals(meta)
            trust[s["trust"]] += 1
            stale += s["stale"]
            draft += s["status"] == "draft"
        recent = sorted(docs, key=lambda d: _updated(d[1]) or "", reverse=True)[:5]
        known = {c for c, _, _ in docs}
        folders: dict[str, int] = {}
        for cid, _, _ in docs:
            top = cid.split("/", 1)[0] if "/" in cid else ""
            if top:
                folders[top] = folders.get(top, 0) + 1
        return {
            "owner": me.name, "bundle": me.dir, "okf_version": okf.OKF_VERSION, "concepts": len(docs),
            "folders": [{"folder": f, "concepts": n, **({"hub": f} if f in known else {})}
                        for f, n in sorted(folders.items())],
            "start_here": [_brief(c, m) for c, m, _ in docs if "/" not in c],
            "types": dict(sorted(types.items(), key=lambda x: -x[1])),
            "top_tags": dict(sorted(tags.items(), key=lambda x: -x[1])[:15]),
            "trust": trust, "stale": stale, "drafts": draft,
            "recently_updated": [{"id": c, "title": m.get("title") or c, "updated": _updated(m)} for c, m, _ in recent],
            "next": NAVIGATE,
            "web": PUBLIC_URL,
        }

    async def _keyword_search(
        query: Annotated[str, Field(description="Words to look for in titles, descriptions, tags and text")],
        type: Annotated[str | None, Field(description="Only this concept type, e.g. 'Playbook'")] = None,
        tag: Annotated[str | None, Field(description="Only concepts with this tag")] = None,
        status: Annotated[Literal["draft", "stable", "deprecated"] | None, Field(description="Only this lifecycle status")] = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
        include_sources: Annotated[bool, Field(description="Also search the raw sources in sources/ (the citation layer); normally only when checking a claim")] = False,
    ) -> list[dict]:
        """Search your knowledge for a specific fact or page. Ranks by matches in title
        (strongest, plus a bonus for the whole query), tags, description, infobox, ID and
        body, and returns a snippet and the `section` holding the hit (read just that with
        kb_read(id, section=…)). To answer a question from several pages, use kb_context.
        Check the result's `trust` and `stale` before relying on it."""
        me = await _me()
        terms = search.tokenize(query)
        if not terms:
            raise ToolError("query needs at least one word that isn't a stop word")
        alt = search.expand(terms)   # synonyms, half weight
        results = []
        for cid, meta, body in await _run(Bundle(me).all):
            if not _matches(meta, type, tag, status) or (meta.get("type") == "Source" and not include_sources and type != "Source"):
                continue
            score = search.score(cid, meta, body, terms, query, alt)
            if score:
                results.append((score, cid, meta, body))
        results.sort(key=lambda r: (-r[0], r[1]))
        return [{**_summary(cid, meta), "score": score, "snippet": _snippet(body, terms),
                 **({"section": s} if (s := _hit_section(body, terms)) else {})}
                for score, cid, meta, body in results[:limit]]

    @mcp.tool
    async def kb_context(
        question: Annotated[str, Field(description="The question to answer from the knowledge base")],
        budget_chars: Annotated[int, Field(ge=2000, le=200000, description="How much page text to return in total (≈4 chars per token). The default covers most questions; raise it (8000-16000) for broad questions or when the answer isn't covered.")] = 4000,
        include_sources: Annotated[bool, Field(description="Also draw on raw sources (sources/); by default they are only listed as citations to visit")] = False,
    ) -> dict:
        """The first tool for any question the knowledge base may answer: use it before
        kb_search or kb_read. One call finds the right pages by keywords AND by meaning (so it
        also finds pages worded differently from the question), follows their links, and
        returns the relevant text within a budget, numbered for citing. Keyword search alone
        misses paraphrases: in tests it put the right page first for 47 of 80 questions,
        kb_context for 71. If it says `no_evidence`, the knowledge base has no answer.
        How: keyword and meaning matches seed the pages, the relevance graph (links,
        parent/child, shared sources, common neighbours) adds related pages two hops out with
        decay, and the best-ranked pages fill the budget — whole when small, otherwise their
        lead plus the sections that match. Pages are numbered: cite them as [1], [2] in your answer. Pages that
        didn't fit are listed under `more` for kb_read. Raw sources the pages cite are listed under
        `sources_cited`: open one with kb_read only to check a claim. A useful synthesis can
        be filed back with kb_write."""
        me = await _me()
        terms = search.tokenize(question)
        if not terms:
            raise ToolError("the question needs at least one word that isn't a stop word")
        docs = await _run(Bundle(me).all)
        by_id = {c: (m, body) for c, m, body in docs}
        if not include_sources:
            docs = [d for d in docs if d[1].get("type") != "Source"]
        alt = search.expand(terms)   # synonyms, half weight: seeds even when the question uses other words
        weights = await _run(search.idf, docs, terms + alt)   # rare terms count more
        hits = {c: s for c, m, body in docs if (s := search.score(c, m, body, terms, question, alt, weights))}
        cov = {c: search.coverage(c, by_id[c][0], by_id[c][1], terms, weights) for c in hits}
        # Starting pages by meaning too (related even without shared words). Exact facts still come
        # from keywords.
        allowed, by_keyword = {c for c, _, _ in docs}, set(hits)
        ranked = [r for r in await _run(embed.rank, me.dir, docs, question, 10) if r[0] in allowed]
        top_sim, best_cov = (ranked[0][1] if ranked else 0.0), max(cov.values(), default=0.0)
        # No evidence: nothing is about the question by meaning and no page holds most of its rare
        # words. Without it the best of many weak keyword hits came back as if it were an answer.
        # Only when meaning search is ready (while the index builds, keywords alone can't tell).
        if top_sim < ABSTAIN_SIM and best_cov < ABSTAIN_COV and await _run(embed.available) and not embed.building():
            return {"question": question, "pages": [], "more": [], "no_evidence": True,
                    "note": f"Nothing in the knowledge base is about this (best match by meaning {top_sim:.2f}, "
                            f"best keyword coverage {best_cov:.0%}). Say the knowledge base has no answer, or try other words."}
        sem = [(c, s) for c, s, _ in ranked if s >= SEED_MIN][:4]
        sem_heading = {c: h for c, _, h in ranked}
        by_meaning = [c for c, _ in sem]
        if not hits and not sem:
            return {"question": question, "pages": [], "more": [],
                    "note": "Nothing matches. Try other words, kb_list the portals, or say the knowledge base has no answer."}
        _, edges = await _run(graph.build, docs)
        adj: dict[str, list[tuple[str, float]]] = {}
        for (a, b), e in edges.items():
            adj.setdefault(a, []).append((b, e["weight"]))
            adj.setdefault(b, []).append((a, e["weight"]))
        # One scale for both methods: keyword pages by how much of the question they cover (frequency
        # breaks ties), meaning pages by similarity mapped onto 0..1.
        top = max(hits.values(), default=1) or 1
        rank = {c: cov[c] * (0.5 + 0.5 * s / top) for c, s in hits.items()}
        for c, s in sem:
            rank[c] = max(rank.get(c, 0), min(1.0, max(0.05, (s - SEM_LOW) / (SEM_FULL - SEM_LOW))))
            hits.setdefault(c, 0)
        via: dict[str, str] = {}
        seeds = list(dict.fromkeys([*sorted(by_keyword, key=lambda c: -rank[c])[:5], *by_meaning]))
        frontier = {c: rank[c] for c in seeds}
        for decay in (0.5, 0.25):                       # two hops, decaying
            nxt: dict[str, float] = {}
            for c, s in frontier.items():
                wmax = max((w for _, w in adj.get(c, [])), default=1)
                for n, w in adj.get(c, []):
                    gain = s * decay * w / wmax
                    if gain > 0.05:
                        nxt[n] = max(nxt.get(n, 0), gain)
                        via.setdefault(n, c)
            for n, g in nxt.items():
                # max, not sum: links bring in related pages, but a hub linked from several starting
                # pages must not climb above the pages that actually match (it did: 9 of 14 misses).
                rank[n] = max(rank.get(n, 0), g)
            frontier = nxt
        order = sorted(rank, key=lambda c: -rank[c])
        # Second stage: a cross-encoder reads the question together with each top candidate's best
        # section (the one meaning found, else the best keyword section, else the lead) and re-orders them.
        if embed.RERANKER and len(order) > 1:
            def passage(c):
                meta, body = by_id[c]
                if embed.RERANK_WHOLE_PAGE:
                    return f"{meta.get('title') or c}. {meta.get('description', '')}\n{body}"[:12000]
                secs = dict(embed.sections(c, meta, body))
                kw = search.section_scores(body, okf.sections(body), terms, weights)
                heading = sem_heading.get(c) or (kw[0][1]["heading"] if kw else "")
                return (secs.get(heading) or secs.get(""))[:1500]
            top = order[:RERANK_TOP]
            if scores := await _run(embed.rerank, question, [passage(c) for c in top]):
                order = [c for _, c in sorted(zip(scores, top), key=lambda x: -x[0])] + order[RERANK_TOP:]
        pages, more, left = [], [], budget_chars
        page_cap = max(budget_chars * PAGE_SHARE, 900)   # no page may crowd out the rest
        for c in order:
            meta, body = by_id[c]
            if left < 600:
                more.append(_brief(c, meta))
                continue
            cap = min(left, page_cap)
            if len(body) <= cap:
                text, partial = body, False
            else:
                lead = guard.lead(body)[:min(1200, int(cap))]
                parts, used = [lead], len(lead)
                # the section the meaning match came from first, then the keyword-matching ones
                matched = [h for h in okf.sections(body) if h["heading"] == sem_heading.get(c)][:1]
                for h in [*matched, *(h for _, h in search.section_scores(body, okf.sections(body), terms, weights) if h not in matched)]:
                    chunk = body[h["start"]:h["end"]]
                    if used + len(chunk) > cap:
                        continue
                    parts.append(chunk.strip())
                    used += len(chunk)
                text, partial = "\n\n".join(parts), True
            left -= len(text)
            n = len(pages) + 1
            pages.append({"n": n, "id": c, "title": meta.get("title") or c, "trust": okf.trust_tier(meta),
                          "stale": okf.is_stale(meta), **({"status": meta["status"]} if meta.get("status") not in (None, "stable") else {}),
                          "found_by": ("both" if c in by_keyword and c in by_meaning else "search" if c in by_keyword
                                       else "meaning" if c in by_meaning else f"related to {via.get(c)}"),
                          "score": round(rank[c], 2), **({"partial": True} if partial else {}), "text": text})
        return {"question": question, "cite_as": "[n] after a statement, n = page number below",
                "pages": pages, "more": more[:15],
                **({"sources_cited": cited} if (cited := sorted({s for p in pages for s in okf.cited(by_id[p["id"]][0], by_id)} - {p["id"] for p in pages})) else {}),
                "hint": "Read a page in `more` (or a partial page's other sections) with kb_read if the answer isn't here.",
                **({"file_back": "If your answer combines these pages into something new (a comparison, a synthesis, a "
                                 "conclusion worth keeping), file it with kb_write as type Comparison or Synthesis under "
                                 "analyses/ (or under the hub it belongs to), citing the pages in `sources`."}
                   if len(pages) >= 3 else {})}

    async def _similar(
        concept_id: Annotated[str | None, Field(description="A page: find sections elsewhere about the same things as its sections")] = None,
        text: Annotated[str | None, Field(description="Or a text (a claim, a paragraph, a draft): where in the kb is this already covered?")] = None,
        limit: Annotated[int, Field(ge=1, le=30)] = 10,
        min_similarity: Annotated[float | None, Field(ge=0.0, le=1.0, description=f"Cosine similarity floor. Default {TEXT_MIN} for a text, {PAGE_MIN} for a page")] = None,
        include_history: Annotated[bool, Field(description="Also return journal days, archive/ and sources/ (history; left out by default because they repeat old values)")] = False,
    ) -> dict:
        """Sections on other pages that are about the same thing as a page or a text, by
        meaning (embeddings), whether or not they share words. For maintenance: contradiction
        candidates (same topic, check whether the values agree), missing links (`linked`:
        false), duplicates, and where new information belongs. Similarity says two sections
        are about the same topic, not that they agree: read both before concluding. For exact
        values and identifiers use kb_search."""
        me = await _me()
        if bool(concept_id) == bool(text):
            raise ToolError("pass either concept_id or text")
        docs = await _run(Bundle(me).all)
        by_id = {c: (m, body) for c, m, body in docs}
        cid = None
        if concept_id:
            cid = await _run(Bundle(me).resolve, okf.normalize_id(concept_id)) or okf.normalize_id(concept_id)
            if cid not in by_id:
                raise ToolError(f"no concept '{cid}'")
        if not await _run(embed.available):
            return {"similar": [], "note": "embeddings are unavailable right now; use kb_search"}
        floor = min_similarity if min_similarity is not None else (TEXT_MIN if text else PAGE_MIN)
        hits = await _run(embed.similar, me.dir, docs, cid, [("text", text)] if text else None, limit * 3, floor)   # history is filtered below
        mine_links = set(okf.links(cid, by_id[cid][1])) if cid else set()
        out = []
        for c, sim, their, yours in hits:
            if c not in by_id or (not include_history and c.startswith(("journal/", "archive/", "sources/"))):
                continue
            linked = c in mine_links or (cid in okf.links(c, by_id[c][1]) if cid else False)
            out.append({**_brief(c, by_id[c][0]), "similarity": sim, "their_section": their or "(lead)",
                        **({"your_section": yours or "(lead)", "linked": linked} if cid else {})})
        out = out[:limit]
        note = ("the index is still being built after a restart or a large change; try again in a few minutes"
                if embed.building() else
                "nothing this close in meaning; try the kb's own terms (system names), a lower min_similarity, or kb_search")
        return {"similar": out, **({} if out else {"note": note})}

    @mcp.tool
    async def kb_ingest(
        title: Annotated[str, Field(description="Name of the source, e.g. the article or document title")],
        content: Annotated[str, Field(description="The source's full text (markdown or plain), stored unchanged")],
        resource: Annotated[str | None, Field(description="Where it came from: URL, file path, or 'conversation with <who> on <date>'")] = None,
        author: Annotated[str, Field(description="Your model, e.g. 'claude/opus-5.5'")] = DEFAULT_AUTHOR,
    ) -> dict:
        """Step 1 of ingesting a source (an article, document, transcript, notes): stores it
        unchanged in the citation layer (sources/, immutable, deduplicated by hash), where pages
        cite it and readers open it only to check a claim, and returns what
        you need to analyse it against the knowledge base: the existing pages it touches
        (with the matching sections, to spot contradictions), the structure to file new
        pages into, and the steps to follow. Step 2 is yours: write a summary of the source
        and create or update the entity and concept pages it affects (kb_write/kb_update),
        citing it in `sources`, then update the portal/hub summaries and record questions
        for the user as `review` items."""
        me = await _me()
        b = Bundle(me)
        if len(content) > guard.SOURCE_LIMIT:
            raise ToolError(f"source is {len(content)} chars (limit {guard.SOURCE_LIMIT}); ingest it in parts")
        if leaked := guard.secrets(content):
            raise ToolError(f"the source looks like it contains a secret ({', '.join(leaked)}); remove it first")
        digest = hashlib.sha256(content.encode()).hexdigest()
        async with store.lock(me):
            docs = await _run(b.all)
            existing = next((c for c, m, _ in docs if m.get("sha256") == digest), None)
            if existing:
                sid, created = existing, False
            else:
                base = "sources/" + (re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "source")
                sid, n = base, 2
                while await _run(b.exists, sid):
                    sid, n = f"{base}-{n}", n + 1
                if not await _run(b.exists, "sources"):
                    await _run(b.save, "sources", dict(SOURCES_PORTAL[0], generated={"by": "claude/kb-mcp", "at": okf.now_iso()}), SOURCES_PORTAL[1], "Creation: [Sources](/sources.md)")
                meta = {"type": "Source", "title": title, "description": f"Raw source as ingested on {okf.now_iso()[:10]}: {title}"[:guard.DESC_LIMIT],
                        "tags": ["source"], "resource": resource, "sha256": digest,
                        "generated": {"by": me.actor if author == "user" else author, "at": okf.now_iso()},
                        "sources": [{"id": "original", "resource": resource or "provided in conversation", "title": title}]}
                meta = {k: v for k, v in meta.items() if v not in (None, "")}
                await _run(b.save, sid, meta, content, f"Ingest: [{title}](/{sid}.md)")
                created = True
            docs = await _run(b.all)
        terms = search.tokenize(title + " " + content[:20000])[:60]
        title_terms = search.tokenize(title)   # what the source is about outweighs words it happens to use
        scored = sorted(((3 * search.score(c, m, body, title_terms) + search.score(c, m, body, terms), c, m, body)
                         for c, m, body in docs if c != sid and m.get("type") != "Source"), key=lambda x: -x[0])
        touched, by_id = [], {c: (m, body) for c, m, body in docs}
        for s, c, m, body in scored[:10]:
            if not s:
                break
            secs = [h["heading"] for _, h in search.section_scores(body, okf.sections(body), terms)[:3]]
            touched.append({**_brief(c, m), "matching_sections": secs, "found_by": "keywords"})
        # Pages about the same things by meaning (the source's opening, in a few slices), even when
        # they use other words. Sources and journal days are left out: they aren't where facts go.
        slices = [(f"source part {i + 1}", content[i * 1500:(i + 1) * 1500]) for i in range(min(4, (len(content) + 1499) // 1500))]
        have = {t["id"] for t in touched}
        for c, sim, their, _ in await _run(embed.similar, me.dir, docs, sid, [(f"{title}", title), *slices], 10, INGEST_MIN):
            if c in have:
                next(t for t in touched if t["id"] == c)["found_by"] = "both"
            elif c in by_id and not c.startswith(("sources/", "journal/")) and len(touched) < 15:
                touched.append({**_brief(c, by_id[c][0]), "matching_sections": [their] if their else [],
                                "found_by": "meaning", "similarity": sim})
        return {
            "source": sid, "created": created, "chars": len(content),
            "pages_it_touches": touched,
            "portals": [_brief(c, m) for c, m, _ in docs if "/" not in c],
            "steps": [
                "1. Analyse: list the source's key entities, concepts and claims; compare each with the pages above "
                "(kb_read their matching_sections) and note agreements, contradictions and what's new. For each claim: "
                "a page already states it the same way → skip; states it differently → edit; no page covers it → "
                "create one. kb_search(similar_to='<a claim>') checks a single claim.",
                f"2. Summarise: create a page summarising the source under the portal it belongs to, citing "
                f"{{id: 'src', resource: '/{sid}.md', title: '{title}'}} in sources.",
                "3. Update: edit each affected entity/concept page in place (kb_update) with the new facts and the "
                "source citation; create pages for important entities or concepts that have none (under the right hub).",
                "4. Contradictions: don't silently overwrite — state which is current and why, and add a `review` "
                "item (fields: {review: ['…question for the user…']}) where a human must decide.",
                "5. Summaries: update the hub and portal pages whose overview changed. Finish with kb_review.",
            ],
        }

    async def _list(
        folder: Annotated[str, Field(description="Folder to list, e.g. 'infrastructure'; empty for the top level")] = "",
        recursive: Annotated[bool, Field(description="Everything below the folder instead of one level (big; prefer drilling down)")] = False,
        type: Annotated[str | None, Field(description="Only this concept type (searches below the folder)")] = None,
        tag: Annotated[str | None, Field(description="Only concepts with this tag (searches below the folder)")] = None,
    ) -> dict:
        """One level of the tree, like an index page: the concepts directly in `folder`
        (id, title, one-line description) and its subfolders with concept counts. A concept
        and a folder of the same name (e.g. 'projects/garden' and
        'projects/garden/') are a hub page and its children. Drill down with
        another kb_list or kb_read rather than listing everything."""
        me = await _me()
        folder = okf.normalize_folder(folder) if folder else ""
        docs = await _run(Bundle(me).all, folder)
        deep = recursive or bool(type or tag)
        here = [(c, m) for c, m, _ in docs if deep or posixpath.dirname(c) == folder]
        sub: dict[str, int] = {}
        for c, _, _ in docs:
            rest = c[len(folder) + 1:] if folder else c
            if "/" in rest:
                name = f"{folder}/{rest.split('/', 1)[0]}" if folder else rest.split("/", 1)[0]
                sub[name] = sub.get(name, 0) + 1
        return {"folder": folder or "/",
                "concepts": [_brief(c, m) for c, m in here if _matches(m, type, tag, None)],
                **({} if deep else {"subfolders": [{"folder": f + "/", "concepts": n} for f, n in sorted(sub.items())]})}

    async def _read(
        concept_id: Annotated[str, Field(description="Concept ID, e.g. 'projects/garden'")],
        section: Annotated[str | None, Field(description="Only this section: its heading text (exact or a unique part), as listed in `outline`")] = None,
        full: Annotated[bool, Field(description="Return the whole body even when it's long")] = False,
    ) -> dict:
        """Read one concept: frontmatter, body, trust/stale signals, links, backlinks, and
        its child concepts (the folder of the same name) and the most `related` pages
        (links, shared sources, common neighbours). Long pages come back as their
        lead plus an `outline` of sections with sizes; read just the part you need with
        `section`, or pass full=true. Prefer sections: they keep your context small.
        The frontmatter's `infobox` holds the page's key facts. An old ID of a moved page
        still works (`redirected_from`)."""
        me = await _me()
        b = Bundle(me)
        asked = okf.normalize_id(concept_id) if concept_id else ""
        cid = await _run(b.resolve, asked) or asked
        meta, body = await _run(b.read, cid)
        docs = await _run(b.all)
        known = {c for c, _, _ in docs}
        heads = okf.sections(body)
        outline = [{"heading": h["heading"], "level": h["level"], "chars": h["end"] - h["start"]} for h in heads]
        out: dict[str, Any] = {"id": cid, **({"redirected_from": asked} if asked != cid else {}),
                               "frontmatter": meta, "signals": okf.signals(meta), "chars": len(body)}
        parent = posixpath.dirname(cid)
        if parent in known:
            out["parent"] = parent
        if section:
            h = okf.find_section(body, section)
            if h is None:
                raise ToolError(f"no section {section!r} in '{cid}'; sections: {', '.join(o['heading'] for o in outline) or 'none'}")
            part = body[h["start"]:h["end"]]
            out.update(section=h["heading"], body=part, outline=outline)
        elif full or len(body) <= guard.READ_LIMIT or not heads:
            part = body
            out["body"] = body
        else:
            lead = body[:heads[0]["start"]].strip()
            part = lead[:1500]
            out.update(body=part + ("…" if len(lead) > 1500 else ""), outline=outline, truncated=True,
                       hint=f"{len(body)} chars: read a section with section='<heading>', or full=true")
        children = [(c, m) for c, m, _ in docs if posixpath.dirname(c) == cid]
        out.update(
            links=[{"id": l, "exists": l in known} for l in okf.links(cid, part)],
            backlinks=_backlinks(docs, cid),
            **({"cited_by": cb} if (cb := [c for c, m, _ in docs if cid in okf.cited(m, known)]) else {}),
            related=graph.related(cid, docs),
            **({"children": [_brief(c, m) for c, m in children]} if children else {}),
            url=f"{PUBLIC_URL}/c/{cid}",
        )
        return out

    @mcp.tool
    async def kb_write(
        concept_id: Annotated[str, Field(description="Under a portal and an existing parent, e.g. 'projects/garden/irrigation'")],
        type: Annotated[str, Field(description=f"One of {', '.join(guard.TYPES)}")],
        body: Annotated[str, Field(description=f"Markdown: a 1–3 sentence lead, then sections; ≤ {guard.HARD_LIMIT} chars (else child pages under '<id>/'); "
                                               "links [text](/folder/id.md), footnotes [^source-id]")],
        title: Annotated[str, Field(description="Unique title")],
        description: Annotated[str, Field(description=f"One sentence (≤ {guard.DESC_LIMIT} chars)")],
        tags: Annotated[list[str], Field(description="At least one")],
        infobox: Annotated[dict[str, Any] | None, Field(description="Key facts, label → short value; expected on hubs")] = None,
        sources: Annotated[list[dict[str, Any]] | None, Field(description="[{id, resource (required), title, author, last_modified}]")] = None,
        status: Literal["draft", "stable", "deprecated"] | None = None,
        stale_after: Annotated[str | None, Field(description="ISO datetime to re-check by")] = None,
        resource: Annotated[str | None, Field(description="URI of the thing described")] = None,
        extra: Annotated[dict[str, Any] | None, Field(description="Any other frontmatter keys")] = None,
        author: Annotated[str, Field(description="Your model, e.g. 'claude/opus-5.5'; 'user' if dictated verbatim")] = DEFAULT_AUTHOR,
        overwrite: Annotated[bool, Field(description="Replace an existing page (history stays in git)")] = False,
        shrink_ok: Annotated[bool, Field(description="Allow dropping over half the page (content moved to child pages)")] = False,
    ) -> dict:
        """Create a page (or replace one with overwrite=true; for edits use kb_update).
        Writes that break the structure are refused with the reason. A new page returns
        `closest_pages` by meaning: merge into one of them if it already covers this."""
        me = await _me()
        cid = okf.normalize_id(concept_id)
        b = Bundle(me)
        async with store.lock(me):
            existed = await _run(b.exists, cid)
            if existed and not overwrite:
                raise ToolError(f"'{cid}' already exists — read it and use kb_update, or pass overwrite=true")
            old_meta, old_body = await _run(b.read, cid) if existed else ({}, None)
            given = {"type": type, "title": title, "description": description, "tags": tags,
                     "infobox": infobox, "aliases": old_meta.get("aliases"),
                     "resource": resource, "status": status, "stale_after": stale_after, "sources": sources,
                     "generated": {"by": me.actor if author == "user" else author, "at": okf.now_iso()}}
            # A parameter left unset must not erase the same key passed in `extra` (clients with an
            # older tool schema can only send newer fields, such as infobox, that way).
            meta = {**(extra or {}), **{k: v for k, v in given.items() if v not in (None, [], "", {})}}
            meta = {k: v for k, v in meta.items() if v not in (None, [], "", {})}
            warnings = await _run(lambda: guard.check(b, cid, meta, body, old_body=old_body,
                                                      creating=not existed, shrink_ok=shrink_ok))
            rev = await _run(b.save, cid, meta, body,
                             f"{'Update' if existed else 'Creation'}: [{title or cid}](/{cid}.md)")
        docs = await _run(b.all)
        if existed:
            extra_out = await _run(_check_also, me.dir, cid, old_meta, old_body, meta, body, docs)
        else:   # a new page: the closest existing pages by meaning (titles are checked by the guard).
            # A reworded duplicate can score about like a related page, so list them for the writer
            # to judge; at DUPLICATE_MIN and above it's very likely the same.
            near = await _run(embed.similar, me.dir, docs, cid, [("new page", f"{title}. {description}\n{body}")], 3, TOPIC_MIN)
            extra_out = {"closest_pages": [{"id": c, "section": h, "similarity": s,
                                            **({"likely_duplicate": True} if s >= DUPLICATE_MIN else {})} for c, s, h, _ in near],
                         "check_also": "Closest existing pages by meaning. If one already covers this, merge into it "
                                       "(kb_update) and delete the new page; otherwise link them where they relate."} if near else {}
        return {"id": cid, "created": not existed, "revision": rev, "url": f"{PUBLIC_URL}/c/{cid}",
                **({"warnings": warnings} if warnings else {}), **extra_out}

    @mcp.tool
    async def kb_update(
        concept_id: str,
        replacements: Annotated[list[dict[str, str]] | None, Field(description="[{old: exact text (must occur once), new: replacement}], in order")] = None,
        append: Annotated[str | None, Field(description="Markdown added at the end")] = None,
        fields: Annotated[dict[str, Any] | None, Field(description="Frontmatter keys to set, e.g. {infobox: {...}, status: 'deprecated'}")] = None,
        remove_fields: list[str] | None = None,
        author: Annotated[str, Field(description="Your model, e.g. 'claude/opus-5.5'; 'user' if dictated")] = DEFAULT_AUTHOR,
        shrink_ok: Annotated[bool, Field(description="Allow dropping over half the page (content moved to child pages)")] = False,
    ) -> dict:
        """Edit a page in place: exact-text replacements, an append, frontmatter fields.
        Replace outdated values (also in description and infobox) instead of adding a
        contradicting note. Returns `check_also`: old values other pages still state
        (keywords) and same-topic sections (meaning), which may need the same change."""
        me = await _me()
        cid = okf.normalize_id(concept_id)
        b = Bundle(me)
        async with store.lock(me):
            cid = await _run(b.resolve, cid) or cid
            meta, body = await _run(b.read, cid)
            old_meta, old_body = dict(meta), body
            for r in replacements or []:
                old, new = r.get("old", ""), r.get("new", "")
                if not old or body.count(old) != 1:
                    raise ToolError(f"replacement text must occur exactly once (found {body.count(old) if old else 0}×): {old[:80]!r}")
                body = body.replace(old, new)
            if append:
                body = body.rstrip() + "\n\n" + append.strip() + "\n"
            for k in remove_fields or []:
                if k == "type":
                    raise ToolError("'type' is required by OKF and can't be removed")
                meta.pop(k, None)
            if fields:
                if "verified" in fields or "generated" in fields:
                    raise ToolError("use kb_verify for 'verified'; 'generated' is set automatically")
                if "aliases" in fields:
                    raise ToolError("'aliases' records old IDs and is kept by kb_move")
                meta.update(fields)
            meta = {k: v for k, v in meta.items() if v not in (None, [], "", {})}
            meta["generated"] = {"by": me.actor if author == "user" else author, "at": okf.now_iso()}
            warnings = await _run(lambda: guard.check(b, cid, meta, body, old_body=old_body, shrink_ok=shrink_ok))
            rev = await _run(b.save, cid, meta, body, f"Update: [{meta.get('title') or cid}](/{cid}.md)")
        extra_out = await _run(_check_also, me.dir, cid, old_meta, old_body, meta, body, await _run(b.all))
        return {"id": cid, "revision": rev, "url": f"{PUBLIC_URL}/c/{cid}",
                **({"warnings": warnings} if warnings else {}), **extra_out}

    @mcp.tool
    async def kb_move(old_id: str, new_id: str) -> dict:
        """Rename or move a concept together with its child pages (the folder of the same
        name). Absolute links to them are updated, and each old ID keeps
        working as an alias (like a wiki redirect). The new place must follow the structure
        (under a portal and an existing parent)."""
        me = await _me()
        old, new = okf.normalize_id(old_id), okf.normalize_id(new_id)
        b = Bundle(me)
        async with store.lock(me):
            if not await _run(b.exists, old):
                raise ToolError(f"no concept '{old}'")
            if new == old or new.startswith(old + "/"):
                raise ToolError("can't move a page into itself")
            subtree = [old] + [c for c in await _run(b.concept_ids, old)]
            targets = {c: new + c[len(old):] for c in subtree}
            if taken := [t for t in targets.values() if await _run(b.exists, t)]:
                raise ToolError(f"already exists: {', '.join(taken)}")
            meta, _ = await _run(b.read, old)
            if errors := guard.placement(new, meta, lambda c: c in targets.values() or b.exists(c)):
                raise ToolError("Refused: " + " · ".join(errors))
            updated, rev = set(), None
            for c in sorted(subtree, key=lambda c: c.count("/")):   # parents first
                rev, upd = await _run(b.move, c, targets[c])
                updated.update(upd)
        return {"id": new, "moved": targets, "revision": rev, "links_updated_in": sorted(updated)}

    @mcp.tool
    async def kb_delete(
        concept_id: str,
        force: Annotated[bool, Field(description="Delete even though other pages link to it (they'll show it as missing)")] = False,
    ) -> dict:
        """Delete a concept. It stays in git history (kb_history / kb_restore can bring it
        back). Prefer status 'deprecated' + kb_move to archive/ for knowledge that's outdated
        but referenced. Pages with child pages can't be deleted; pages that others link to
        need force=true (fix those links first)."""
        me = await _me()
        cid = okf.normalize_id(concept_id)
        b = Bundle(me)
        async with store.lock(me):
            if not await _run(b.exists, cid):
                raise ToolError(f"no concept '{cid}'")
            if children := await _run(b.concept_ids, cid):
                raise ToolError(f"'{cid}' is the hub of {len(children)} page(s) ({', '.join(children[:5])}); "
                                f"move or delete those first, or kb_move the whole subtree to archive/")
            if not force and (linked := _backlinks(await _run(b.all), cid)):
                raise ToolError(f"linked from {', '.join(linked[:8])}: update those pages, deprecate and archive it "
                                f"instead, or pass force=true")
            rev = await _run(b.delete, cid)
        return {"deleted": cid, "revision": rev, "restore_with": f"kb_restore('{cid}', <revision before {rev}>)"}

    @mcp.tool
    async def kb_verify(
        concept_id: str,
        confirmed_by_user: Annotated[bool, Field(description="True ONLY if the user explicitly confirmed this content is correct in this conversation. Records human:<user> (human-reviewed tier).")] = False,
        verifier: Annotated[str, Field(description="When not user-confirmed: the actor that checked it against its sources, e.g. 'claude/opus-5.5'")] = DEFAULT_AUTHOR,
        stale_after: Annotated[str | None, Field(description="Optionally set when it should be re-checked next (ISO datetime)")] = None,
    ) -> dict:
        """Record that a concept was checked and is correct. Without the user's explicit
        confirmation this is only machine-confirmed. Doesn't change the content."""
        me = await _me()
        cid = okf.normalize_id(concept_id)
        b = Bundle(me)
        async with store.lock(me):
            meta, body = await _run(b.read, cid)
            by = me.actor if confirmed_by_user else verifier
            if by.startswith("human:") and not confirmed_by_user:
                raise ToolError("only the user can produce a human: verification — set confirmed_by_user")
            entries = [v for v in okf.verified_list(meta) if v.get("by") != by]
            meta["verified"] = [*entries, {"by": by, "at": okf.now_iso()}]
            if stale_after:
                meta["stale_after"] = stale_after
            rev = await _run(b.save, cid, meta, body, f"Verification: [{meta.get('title') or cid}](/{cid}.md) by {by}")
        return {"id": cid, "trust": okf.trust_tier(meta), "revision": rev}

    async def _review_queue(limit: Annotated[int, Field(ge=1, le=100)] = 30) -> dict:
        """What needs attention: the user's answers to review questions (`answered_by_user`: work
        each answer into its page with kb_update, then clear it with remove_fields=["answered"]),
        open questions for the user, stale concepts (past stale_after), drafts, unverified
        concepts, and links to concepts that don't exist yet."""
        me = await _me()
        docs = await _run(Bundle(me).all)
        known = {c for c, _, _ in docs}
        stale = [_summary(c, m) for c, m, _ in docs if okf.is_stale(m)]
        drafts = [_summary(c, m) for c, m, _ in docs if okf.signals(m)["status"] == "draft"]
        unverified = [_summary(c, m) for c, m, _ in docs
                      if okf.trust_tier(m) == "unverified" and okf.signals(m)["status"] != "deprecated"]
        broken = [{"from": c, "to": l} for c, _, body in docs for l in okf.links(c, body) if l not in known]
        questions = [{"id": c, "title": m.get("title") or c, "review": m["review"]} for c, m, _ in docs if m.get("review")]
        answers = [{"id": c, "title": m.get("title") or c, "answered": m["answered"]} for c, m, _ in docs if m.get("answered")]
        return {"answered_by_user": answers, "questions_for_user": questions[:limit], "stale": stale[:limit], "drafts": drafts[:limit],
                "unverified": unverified[:limit], "missing_link_targets": broken[:limit]}

    async def _lint(
        folder: Annotated[str, Field(description="Only check concepts under this folder; empty for all")] = "",
        limit: Annotated[int, Field(ge=1, le=100)] = 25,
    ) -> dict:
        """Health check of the bundle's structure (run it after reorganising, and now and
        then): pages breaking the write rules (`rule_errors`: placement, lead, size,
        metadata, secrets; `rule_warnings`: getting long, deprecated outside archive/),
        broken links, orphans nothing links to, hubs that don't mention or summarise their
        children, hubs without an infobox, current pages linking to deprecated ones, and
        possible duplicates. It is mechanical: contradictions between pages, claims that
        newer knowledge superseded, and topics mentioned on several pages without one of
        their own still need you to read the pages it points to."""
        me = await _me()
        folder = okf.normalize_folder(folder) if folder else ""
        docs = await _run(Bundle(me).all)
        known = {c: m for c, m, _ in docs}
        inbound: dict[str, set[str]] = {c: set() for c in known}
        for c, _, body in docs:
            for l in okf.links(c, body):
                if l in inbound and l != c:
                    inbound[l].add(c)
        for c, m, _ in docs:
            for s in okf.cited(m, known):
                inbound[s].add(c)
        mine = [(c, m, body) for c, m, body in docs if not folder or c == folder or c.startswith(folder + "/")]
        current = lambda m: okf.signals(m)["status"] != "deprecated"
        rules = {c: (*guard.shape(c, m, body), guard.placement(c, m, known.__contains__)) for c, m, body in mine}
        hubs = {c for c in known if any(posixpath.dirname(k) == c for k in known)}
        out: dict[str, list] = {
            "rule_errors": [{"id": c, "errors": e + p} for c, (e, _, p) in rules.items() if e or p],
            "rule_warnings": [{"id": c, "warnings": w} for c, (_, w, _) in rules.items() if w],
            "broken_links": [{"from": c, "to": l} for c, _, body in mine for l in okf.links(c, body) if l not in known],
            "orphans": [c for c, m, _ in mine if not inbound[c] and current(m) and "/" in c and m.get("type") != "Source"],
            "uncited_sources": [c for c, m, _ in mine if m.get("type") == "Source" and not inbound[c]],
            "unlinked_children": [
                {"hub": c, "children": missing} for c, m, body in mine if c in hubs and c not in guard.HUBLESS
                if (missing := [k for k in known if posixpath.dirname(k) == c and k not in okf.links(c, body)])],
            "hubs_without_infobox": [c for c, m, _ in mine if c in hubs and "/" in c and not m.get("infobox")
                                     and not c.startswith(tuple(f"{h}/" for h in guard.HUBLESS))],
            "links_to_deprecated": [{"from": c, "to": l} for c, m, body in mine
                                    if current(m) and m.get("type") != "Log" and c != "archive"
                                    for l in okf.links(c, body) if l in known and not current(known[l])],
        }
        titles = [(c, str(m.get("title") or c).lower()) for c, m, _ in mine]
        out["possible_duplicates"] = [
            {"a": a, "b": b} for i, (a, ta) in enumerate(titles) for b, tb in titles[i + 1:]
            if difflib.SequenceMatcher(None, ta, tb).ratio() >= 0.8
            and re.sub(r"\d", "", ta) != re.sub(r"\d", "", tb)    # dated series (journal days) aren't duplicates
            and not (a.startswith("sources/") or b.startswith("sources/"))]   # nor a summary next to its source
        counts = {k: len(v) for k, v in out.items()}
        return {"checked": len(mine), "counts": counts, **{k: v[:limit] for k, v in out.items() if v},
                "also_check": "Read the pages above where needed for contradictions, superseded claims and "
                              "topics that deserve their own page; fix with kb_update/kb_move, then lint again."}

    async def _history(concept_id: str | None = None, limit: Annotated[int, Field(ge=1, le=100)] = 20) -> list[dict]:
        """Change history (git) of one concept, or of the whole bundle if no ID is given."""
        me = await _me()
        cid = okf.normalize_id(concept_id) if concept_id else None
        return await _run(Bundle(me).history, cid, limit)

    async def _diff(
        concept_id: str,
        revision: Annotated[str, Field(description="Commit id from kb_history(concept_id)")],
        include_old: Annotated[bool, Field(description="Also return the concept's full text at that revision")] = False,
    ) -> dict:
        """What changed in a concept from a revision to its current version, as a unified diff
        (frontmatter included). Use it to see what an older version said, e.g. before kb_restore.
        Works on your own bundle only."""
        me = await _me()
        cid = okf.normalize_id(concept_id)
        b = Bundle(me)
        rows = await _run(b.changes, cid, 100)
        row = next((r for r in rows if len(revision) >= 4 and
                    (r["revision"].startswith(revision) or revision.startswith(r["revision"]))), None)
        if row is None:
            raise ToolError(f"{revision!r} isn't in the history of '{cid}' — get revision ids from kb_history('{cid}')")
        if row["files"] and row["files"][0][0] == "D":
            raise ToolError("the concept was deleted in that revision; pick an earlier one")
        old = await _run(b.text_at, cid, row["revision"], row["path"])
        exists = await _run(b.exists, cid)
        new = await _run(b.path(cid).read_text, "utf-8") if exists else ""
        lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"{cid} @ {row['revision']}",
                                          f"{cid} (current)" if exists else f"{cid} (deleted)", lineterm="", n=3))
        diff = "\n".join(lines)
        if len(diff) > 60000:
            diff = diff[:60000] + "\n… (diff truncated; use include_old for the full old text)"
        return {"id": cid,
                "from": {k: row[k] for k in ("revision", "at", "message")},
                "to": "current" if exists else "deleted (empty)",
                "same": not lines,
                "added": sum(1 for l in lines[2:] if l.startswith("+")),
                "removed": sum(1 for l in lines[2:] if l.startswith("-")),
                "diff": diff,
                **({"old_text": old} if include_old else {})}

    @mcp.tool
    async def kb_restore(concept_id: str, revision: Annotated[str, Field(description="Commit id from kb_history")]) -> dict:
        """Bring back a concept as it was at a revision (also undeletes). Recorded as a new change."""
        me = await _me()
        cid = okf.normalize_id(concept_id)
        b = Bundle(me)
        async with store.lock(me):
            meta, body = okf.parse(await _run(b.text_at, cid, revision))
            errors = await _run(guard.placement, cid, meta, b.exists)
            if leaked := guard.secrets(body + "\n" + str(meta)):
                errors.append(f"that version contains a secret ({', '.join(leaked)})")
            if errors:
                raise ToolError("Refused: " + " · ".join(errors))
            rev = await _run(b.save, cid, meta, body, f"Update: restored [{meta.get('title') or cid}](/{cid}.md) from {revision}")
        return {"id": cid, "revision": rev}

    # ── The merged tools (2026-11-04: 22 → 14). Each dispatches to the functions above. ──

    @mcp.tool
    async def kb_list(
        folder: Annotated[str, Field(description="Folder to list, e.g. 'infrastructure'; empty: the overview plus the top level")] = "",
        recursive: Annotated[bool, Field(description="Everything below the folder (big; prefer drilling down)")] = False,
        type: Annotated[str | None, Field(description="Only this concept type")] = None,
        tag: Annotated[str | None, Field(description="Only concepts with this tag")] = None,
    ) -> dict:
        """Start here. One level of the tree, like an index page: the concepts in `folder`
        (id, title, description) and its subfolders. Without a folder, also the bundle's
        overview: counts by type, top tags, recent changes, how much needs review."""
        listing = await _list(folder, recursive, type, tag)
        if folder or recursive or type or tag:
            return listing
        return {"overview": await _overview(), **listing}

    @mcp.tool
    async def kb_read(
        concept_id: Annotated[str, Field(description="Concept ID, e.g. 'projects/garden'")],
        section: Annotated[str | None, Field(description="Only this section (its heading, as listed in `outline`)")] = None,
        full: Annotated[bool, Field(description="The whole body even when it's long")] = False,
    ) -> dict:
        """Read one concept: frontmatter (its `infobox` holds the key facts), body,
        trust/stale signals, links, backlinks, children and related pages. Long pages come
        back as their lead plus an `outline`; read just the part you need with `section`.
        Old IDs of moved pages still work."""
        return await _read(concept_id.strip(), section, full)

    @mcp.tool
    async def kb_search(
        query: Annotated[str | None, Field(description="Words to look for (keywords): the way to find exact facts, values and identifiers")] = None,
        similar_to: Annotated[str | None, Field(description="Instead of query: a page ID or a text (claim, paragraph, draft); finds sections elsewhere about the same thing by meaning")] = None,
        type: Annotated[str | None, Field(description="Keywords: only this concept type")] = None,
        tag: Annotated[str | None, Field(description="Keywords: only concepts with this tag")] = None,
        status: Annotated[Literal["draft", "stable", "deprecated"] | None, Field(description="Keywords: only this status")] = None,
        limit: Annotated[int, Field(ge=1, le=50)] = 10,
        include_sources: Annotated[bool, Field(description="Keywords: also search raw sources in sources/")] = False,
        min_similarity: Annotated[float | None, Field(ge=0.0, le=1.0, description=f"Meaning: floor (default {TEXT_MIN} for a text, {PAGE_MIN} for a page)")] = None,
        include_history: Annotated[bool, Field(description="Meaning: also journal days, archive/ and sources/")] = False,
    ) -> dict:
        """Find pages. With `query`: keyword search for a specific fact or page, ranked by
        title, tags, description, infobox and body, with a snippet and the `section` holding
        the hit. With `similar_to`: sections on other pages about the same thing by meaning,
        even without shared words (where a fact is already written, contradiction
        candidates, missing links via `linked`: false, duplicates); similarity means same
        topic, not agreement, so read both. To answer a question, use kb_context instead
        (keywords and meaning together); kb_search is for an exact value, identifier or name,
        or every page stating an old value. Check `trust` and `stale` before relying on a result."""
        if bool(query) == bool(similar_to):
            raise ToolError("pass either query (keywords) or similar_to (a page ID or a text)")
        if query:
            return {"results": await _keyword_search(query, type, tag, status, limit, include_sources)}
        me = await _me()
        def _is_page() -> bool:   # anything that isn't a valid, existing page ID is a text
            try:
                return Bundle(me).resolve(okf.normalize_id(similar_to)) is not None
            except okf.ConceptError:
                return False
        is_page = "\n" not in similar_to and len(similar_to) < 200 and await _run(_is_page)
        return await _similar(similar_to if is_page else None, None if is_page else similar_to,
                              min(limit, 30), min_similarity, include_history)

    @mcp.tool
    async def kb_history(
        concept_id: Annotated[str | None, Field(description="A concept; omit for the whole bundle")] = None,
        revision: Annotated[str | None, Field(description="With concept_id: what changed from this commit to now (a diff)")] = None,
        include_old: Annotated[bool, Field(description="With revision: also the full text at that revision")] = False,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
    ) -> dict:
        """Change history (git): the commits of one concept or of the whole bundle, newest
        first. With `revision`: the diff from that commit to the current version (e.g. to
        see what an older version said, before kb_restore)."""
        if revision:
            if not concept_id:
                raise ToolError("a diff needs concept_id and revision")
            return await _diff(concept_id, revision, include_old)
        return {"commits": await _history(concept_id, limit)}

    @mcp.tool
    async def kb_review(
        folder: Annotated[str, Field(description="Structure check only under this folder; empty for all")] = "",
        limit: Annotated[int, Field(ge=1, le=100)] = 30,
    ) -> dict:
        """Maintenance view. `lint`: the structure check (rule errors and warnings, broken
        links, orphans, hubs not listing children or without an infobox, links to deprecated
        pages, possible duplicates). `queue`: the user's answers to apply
        (`answered_by_user`: work each into its page, then kb_update(id,
        remove_fields=["answered"])), open questions, stale pages and drafts. Contradictions
        and superseded claims still need reading (kb_search similar_to helps find them)."""
        return {"lint": await _lint(folder, limit), "queue": await _review_queue(limit)}
