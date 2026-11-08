# How it works

What happens inside okf-wiki-mcp when Claude asks a question, writes a page or checks the bundle's
health. Each diagram matches the code it names. Thresholds are the defaults for the local model
(`stella-en-400M-v5`); `tools.py` holds the mapped values for `voyage-context-4`.

1. [Keyword search](#1-keyword-search-lexical): exact facts, values and names
2. [Embeddings](#2-embeddings-meaning): what is related by meaning
3. [Relevance graph](#3-relevance-graph): how pages connect, and hops
4. [kb_context](#4-kb_context-answering-a-question): answering a question with all three
5. [Maintenance](#5-maintenance-write-checks-lint-and-gaps): write checks, lint and gaps

---

## 1. Keyword search (lexical)

`search.py`, used by `kb_search(query=…)`, as one half of `kb_context`, and by `kb_ingest`.
Keywords stay in charge of **exact facts**: versions, IPs, hostnames and identifiers.

```mermaid
flowchart TD
    Q["query"] --> T["tokenize<br/>lowercase · drop stop words (en + de)<br/>stem: backups→backup, policies→polic<br/>identifiers kept whole: 0.45.1, web-01<br/>CJK runs → bigrams"]
    T --> S["synonyms from KB_SYNONYMS<br/>(count at half weight)"]
    T --> F
    S --> F["score each page by field<br/>title ×6 · tags ×4 · description ×3<br/>infobox ×3 · ID / aliases ×2<br/>body: hits, at most 5"]
    F --> I["× idf per term (in kb_context):<br/>rare words count more"]
    I --> B["+10 if the whole query<br/>appears in the title"]
    B --> R["ranked pages<br/>with snippet + the section holding the hit"]
```

A term must **start a word**: `backup` finds *backups*, but `point` doesn't match inside
*endpoint*. `coverage()` measures the share of the question's idf weight a page contains (0–1), so a
page holding only one common word of the question doesn't look like an answer.

## 2. Embeddings (meaning)

`embed.py`. Embeddings find what is **about the same thing**, even with no words in common. They
only add relations; they never decide facts.

```mermaid
flowchart TD
    subgraph Index["Index (background, low priority)"]
        P["page"] --> SEC["sections: the lead + each heading's own text<br/>prefixed with the page title (and heading)<br/>cut to 1000 chars"]
        SEC --> M["model<br/>stella-en-400M-v5 (local, ONNX)<br/>or voyage-context-4 (API, whole page as context)"]
        M --> V[("vectors per user<br/>/data/embeddings/USER.npz<br/>keyed by a hash of the text")]
    end
    V --> RK["rank(question)<br/>best section per page"]
    V --> SIM["similar(page or text)<br/>closest sections on other pages"]
    RK --> U1["kb_context starting pages ≥ 0.37<br/>no-evidence gate below 0.43"]
    SIM --> U2["kb_search similar_to=text ≥ 0.54<br/>kb_search similar_to=page ≥ 0.69"]
    SIM --> U3["kb_ingest: pages the source touches ≥ 0.61<br/>after an edit: same topic elsewhere ≥ 0.76<br/>kb_write: likely duplicate ≥ 0.91"]
```

A write re-embeds only the sections whose text changed (with Voyage, the whole page, since every
section's vector depends on its page). After a restart nothing is re-embedded. If the model or the API
is unavailable, every caller falls back to keywords.

## 3. Relevance graph

`graph.py`. Pages are nodes, and an edge's weight adds up the signals that connect two pages
(after [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki)).

```mermaid
flowchart LR
    subgraph Signals["Edge weight between two pages"]
        L["direct link, either way,<br/>or citing a raw source<br/>+3"]
        H["parent / child<br/>(hub and its pages)<br/>+2"]
        SS["shared source<br/>cited by 2–6 pages<br/>+4 each"]
        AA["common neighbours<br/>Adamic-Adar<br/>+1.5 × score"]
        TY["same type,<br/>only on top of another signal<br/>+1"]
    end
    Signals --> E["weighted edges"]
    E --> R1["kb_read 'related':<br/>direct edges + 2-hop pages<br/>with Adamic-Adar ≥ 0.8, top 6"]
    E --> R2["web /graph:<br/>interactive map (sigma.js)"]
    E --> R3["kb_context:<br/>expansion, 2 hops"]
```

A source cited everywhere says nothing about how two pages relate, so shared sources only count
when at most 6 pages cite them. **Hops** in `kb_context`:

```mermaid
flowchart LR
    S(("starting page<br/>score s")) -- "hop 1: s × 0.5 × w / w_max" --> N1(("neighbour"))
    N1 -- "hop 2: × 0.25 × w / w_max" --> N2(("neighbour's<br/>neighbour"))
    N2 -. "gain ≤ 0.05" .-> X["dropped"]
```

A page reached from several starting pages keeps its **best** gain, not the sum, so a busy hub
can't climb above the pages that actually match.

## 4. kb_context: answering a question

`tools.py: kb_context`. This is the first tool for any question: keywords and meaning both pick
starting pages, the graph adds related ones, and the best pages fill a character budget.

```mermaid
flowchart TD
    Q["question"] --> KW["keyword hits<br/>(idf-weighted score + coverage)"]
    Q --> EM["meaning: top 10 pages<br/>by best section"]
    KW --> GATE{"best meaning below 0.43<br/>and best coverage below 60%?"}
    EM --> GATE
    GATE -- yes --> NE["no_evidence:<br/>the kb has no answer"]
    GATE -- no --> SEED["starting pages<br/>top 5 by keywords + up to 4 by meaning ≥ 0.37"]
    SEED --> ONE["one scale, 0..1<br/>keywords: coverage × (0.5 + 0.5 × score / best)<br/>meaning: similarity 0.30–0.65 mapped to 0–1"]
    ONE --> HOP["graph expansion<br/>2 hops, decay 0.5 then 0.25"]
    HOP --> RR{"KB_RERANKER set?"}
    RR -- yes --> RER["reranker re-orders the top 6<br/>(cross-encoder, or Voyage rerank on whole pages)"]
    RR -- no --> FILL
    RER --> FILL["fill the budget (default 4000 chars)<br/>one page ≤ 35% of it (at least 900)<br/>small page: whole · long page: lead + matching sections"]
    FILL --> OUT["numbered pages [1], [2] … to cite<br/>found_by: search / meaning / both / related to X<br/>'more' for pages that didn't fit · sources cited"]
```

With raw sources excluded (the default), `sources/` pages are only listed as citations to open
when a claim needs checking.

## 5. Maintenance: write checks, lint and gaps

Three layers keep the bundle healthy: **guardrails** refuse a bad write, **check-also hints** point
at what else the write made stale, and **`kb_review`** checks the whole bundle.

```mermaid
flowchart TD
    W["kb_write / kb_update"] --> G{"guard.check"}
    G -- refused --> WHY["the reason is returned:<br/>wrong placement · no lead/title/description/tags<br/>over 12000 chars · duplicate title · secret<br/>literal backslash-n escapes · shrinks by over half"]
    G -- ok --> C[("git commit")]
    C --> CA["check_also"]
    CA --> OV["old_values_elsewhere<br/>identifiers the edit removed (0.45.1, web-01)<br/>still stated on other pages, matched whole"]
    CA --> ST["same_topic_elsewhere<br/>changed passages vs other sections<br/>by meaning ≥ 0.76"]
    C --> DUP["kb_write only: closest pages,<br/>likely_duplicate ≥ 0.91"]
    ING["kb_ingest"] --> TOUCH["pages_it_touches<br/>keywords (title ×3) + meaning ≥ 0.61<br/>with the matching sections"]
```

```mermaid
flowchart TD
    R["kb_review"] --> LINT["lint (structure)"]
    R --> QUEUE["queue (people)"]
    LINT --> L1["rule errors / warnings<br/>placement, lead, size, metadata, secrets<br/>getting long, deprecated outside archive/"]
    LINT --> L2["broken links"]
    LINT --> L3["orphans: nothing links to them"]
    LINT --> L4["uncited sources"]
    LINT --> L5["hubs not listing their children<br/>hubs without an infobox"]
    LINT --> L6["current pages linking to deprecated ones"]
    LINT --> L7["possible duplicates<br/>titles ≥ 80% alike"]
    QUEUE --> Q1["answered_by_user:<br/>answers from the web UI to work in"]
    QUEUE --> Q2["open review questions"]
    QUEUE --> Q3["stale (past stale_after) · drafts · unverified"]
    QUEUE --> Q4["missing link targets"]
```

**Gap analysis**: finding what's missing, not what's wrong.

```mermaid
flowchart LR
    subgraph Mechanical["Found by the server"]
        G1["missing link targets:<br/>pages linked to that don't exist yet"]
        G2["orphans and unlinked children:<br/>pages the structure doesn't reach"]
        G3["similar_to=page, linked: false<br/>same topic, no link between them"]
        G4["uncited sources:<br/>ingested but used by no page"]
    end
    subgraph Judgement["Left to Claude reading the pages"]
        J1["contradictions:<br/>same topic, different values"]
        J2["claims newer knowledge superseded"]
        J3["topics mentioned on several pages<br/>without a page of their own"]
    end
    Mechanical --> FIX["kb_update · kb_write · kb_move<br/>then kb_review again"]
    Judgement --> FIX
```

The checks are mechanical on purpose: similarity says two sections are about the same topic, not
that they agree, so deciding a contradiction always means reading both.
