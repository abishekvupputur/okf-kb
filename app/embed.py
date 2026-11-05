"""Meaning-based layer of the kb: one embedding per page section (stella-en-400M-v5 through
FastEmbed on ONNX Runtime, no PyTorch), used next to the keyword search, never instead of it.

Model choice (2026-11-05): best of 18 runs on 40 paraphrased questions against a real bundle: the right
page first for 35 of 40 and in the top 3 for all 40 (bge-small-en-v1.5, used before: 25 and 35).

- Sections: a page's lead and each heading's own text, prefixed with the page title and heading,
  cut to CUT characters (shorter sections keep the memory peak down: measured peak ~390 MB for a
  full build of ~150 pages).
- Vectors are kept per user in /data/embeddings/<user>.npz, keyed by a hash of the section text,
  so a restart re-embeds nothing and a write re-embeds only the sections it changed.
- CPU: KB_EMBED_THREADS ONNX threads (default 1), and the embedding work runs at low priority (nice).
- Questions get the model's retrieval instruction (QUERY_PREFIX); sections and texts compared with
  sections are embedded as they are.
- If FastEmbed or the model is unavailable, `available()` is False and callers use keywords only.
"""

import hashlib
import logging
import os
import re
import ctypes
import threading
from pathlib import Path

import numpy as np

from .store import DATA

MODEL = "Qdrant/stella-en-400M-v5-doc-onnx"   # MIT; one model for questions and sections
DIM = 1024
QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
THREADS = int(os.environ.get("KB_EMBED_THREADS", "1"))
MODEL_DIR = os.environ.get("KB_MODELS", "/srv/models")   # baked into the image (Dockerfile)
INDEX_DIR = DATA / "embeddings"
CUT = int(os.environ.get("KB_EMBED_CUT", "1000"))
BATCH = 4
NICE = 15
QUERY_MAX_NEW = 40               # sections a query may embed itself (a write touches a few)

log = logging.getLogger("kb.embed")
_lock = threading.Lock()          # one embedding job at a time (model + index files)
_model = None
_failed = False
_index: dict[str, dict] = {}      # user dir -> {"keys": [...], "vecs": ndarray, "labels": [(cid, heading)]}


def _load_model():
    global _model, _failed
    if _model is None and not _failed:
        try:
            import onnxruntime
            onnxruntime.disable_telemetry_events()   # no usage telemetry from this box
            from fastembed import TextEmbedding
            _model = TextEmbedding(MODEL, cache_dir=MODEL_DIR, threads=THREADS)
        except Exception as e:   # missing package or model: keyword search keeps working
            _failed = True
            log.error("embeddings unavailable, keyword search only: %s", e)
    return _model


def available() -> bool:
    return _load_model() is not None


def sections(cid: str, meta: dict, body: str, cut: int | None = CUT) -> list[tuple[str, str]]:
    """(heading or '', text to embed) for the lead and every heading's own text, each prefixed
    with the page title (and heading) and cut to `cut` characters (None: the whole text)."""
    title = str(meta.get("title") or cid)
    marks = [(m.start(), m.group(2).strip()) for m in re.finditer(r"(?m)^(#{1,6}) +(.+)$", body)]
    lead = body[:marks[0][0] if marks else len(body)]
    out = [("", f"{title}. {meta.get('description', '')}\n{lead[:cut]}")]
    for i, (start, heading) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(body)
        text = body[start:end].split("\n", 1)[1] if "\n" in body[start:end] else ""
        out.append((heading, f"{title} — {heading}\n{text[:cut]}"))
    return out


def changed_passages(cid: str, old_meta: dict, old_body: str, meta: dict, body: str) -> list[tuple[str, str]]:
    """What an edit changed, as passages to compare: for each section whose full text differs,
    about CUT characters around the first difference (an append deep in a long section counts too),
    prefixed with the section's title line."""
    before = dict(sections(cid, old_meta, old_body, None))
    out = []
    for heading, text in sections(cid, meta, body, None):
        old = before.get(heading)
        if old == text:
            continue
        head, _, rest = text.partition("\n")
        old_rest = (old or "").partition("\n")[2]
        pos = next((i for i, (a, b) in enumerate(zip(rest, old_rest)) if a != b), min(len(rest), len(old_rest)))
        start = max(0, min(pos - CUT // 3, len(rest) - CUT))
        out.append((heading, f"{head}\n{rest[start:start + CUT]}"))
    return out


def _key(text: str) -> str:
    return hashlib.sha1(f"{MODEL}\n{text}".encode()).hexdigest()


def _trim():
    """Hand memory freed after embedding back to the system (glibc keeps it otherwise, so the
    process would stay at its peak, ~390 MB, instead of ~290 MB)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _low_priority():
    try:
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), NICE)   # this thread only (Linux)
    except OSError:
        pass


def _ensure(user_dir: str, docs, max_new: int | None = None) -> dict | None:
    """Bring the user's index in line with their pages; embeds only sections it hasn't seen.
    With max_new (the query path), more new sections than that means a full build is due:
    return None and let the background warm-up do it, so a search never waits minutes."""
    model = _load_model()
    if model is None:
        return None
    wanted = [(cid, h, t) for cid, meta, body in docs if meta.get("type") != "Source"
              for h, t in sections(cid, meta, body)]
    keys = [_key(t) for _, _, t in wanted]
    idx = _index.get(user_dir)
    if idx is None:
        path = INDEX_DIR / f"{user_dir}.npz"
        idx = {"keys": [], "vecs": np.zeros((0, DIM), np.float32)}
        if path.exists():
            try:
                z = np.load(path, allow_pickle=False)
                idx = {"keys": list(z["keys"]), "vecs": z["vecs"]}
            except Exception as e:
                log.warning("embedding index for %s unreadable, rebuilding: %s", user_dir, e)
    have = dict(zip(idx["keys"], range(len(idx["keys"]))))
    missing = [i for i, k in enumerate(keys) if k not in have]
    if max_new is not None and len(missing) > max_new:
        _start_warm()
        return None
    labels = [(c, h) for c, h, _ in wanted]
    # Vectors change only when sections were added or removed. Labels (page ID, heading) are
    # recomputed whenever they differ: a moved page keeps its text, so its fingerprints, but
    # gets a new ID. Labels aren't stored, so they are rebuilt once after every restart.
    vectors_changed = bool(missing) or set(have) != set(keys)
    if vectors_changed or idx.get("labels") != labels:
        new_vecs = {}
        if missing:
            _low_priority()
            texts = [wanted[i][2] for i in missing]
            for i, v in zip(missing, model.passage_embed(texts, batch_size=BATCH)):
                new_vecs[keys[i]] = np.asarray(v, np.float32)
            log.info("embedded %d sections for %s", len(missing), user_dir)
            _trim()
        vecs = np.stack([new_vecs[k] if k in new_vecs else idx["vecs"][have[k]] for k in keys]) if keys \
            else np.zeros((0, DIM), np.float32)
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        idx = {"keys": keys, "vecs": vecs, "labels": labels}
        if vectors_changed:
            try:   # a failed save only costs re-embedding those sections after the next restart
                INDEX_DIR.mkdir(parents=True, exist_ok=True)
                tmp = INDEX_DIR / f".{user_dir}.tmp.npz"
                np.savez(tmp, keys=np.array(keys), vecs=vecs)
                os.replace(tmp, INDEX_DIR / f"{user_dir}.npz")
            except OSError as e:
                log.warning("could not save the embedding index for %s: %s", user_dir, e)
        _index[user_dir] = idx
    return idx


def rank(user_dir: str, docs, query: str, limit: int = 20) -> list[tuple[str, float, str]]:
    """Pages by meaning: [(id, similarity of its best section, that section's heading)]."""
    if not _lock.acquire(timeout=5):   # a build is running: keywords only for now
        return []
    try:
        idx = _ensure(user_dir, docs, max_new=QUERY_MAX_NEW)
        if idx is None or not len(idx["keys"]):
            return []
        q = np.asarray(next(iter(_model.query_embed([QUERY_PREFIX + query]))), np.float32)
    finally:
        _lock.release()
    return _best_per_page(idx, idx["vecs"] @ (q / max(np.linalg.norm(q), 1e-9)), limit)


def similar(user_dir: str, docs, cid: str | None = None, texts: list[tuple[str, str]] | None = None,
            limit: int = 10, min_sim: float = 0.0) -> list[tuple[str, float, str, str]]:
    """Sections on other pages closest to a page (each of its sections) or to texts given as
    [(label, text)]: [(id, similarity, their heading, your heading/label)]. Texts are embedded
    as passages (like the sections), not as search queries."""
    if not _lock.acquire(timeout=5):
        return []
    try:
        idx = _ensure(user_dir, docs, max_new=QUERY_MAX_NEW)
        if idx is None or not len(idx["keys"]):
            return []
        if texts:
            _low_priority()
            vecs = _model.passage_embed([t[:CUT] for _, t in texts], batch_size=BATCH)
            mine = [(label, np.asarray(v, np.float32)) for (label, _), v in zip(texts, vecs)]
        else:
            mine = [(h, idx["vecs"][i]) for i, (c, h) in enumerate(idx["labels"]) if c == cid]
    finally:
        _lock.release()
    best: dict[str, tuple[float, str, str]] = {}
    for my_heading, v in mine:
        sims = idx["vecs"] @ (v / max(np.linalg.norm(v), 1e-9))
        for i in np.argsort(-sims)[:50]:
            c, h = idx["labels"][i]
            s = float(sims[i])
            if c != cid and s >= min_sim and s > best.get(c, (-1,))[0]:
                best[c] = (s, h, my_heading)
    out = sorted(((c, round(s, 3), h, mh) for c, (s, h, mh) in best.items()), key=lambda x: -x[1])
    return out[:limit]


def _best_per_page(idx, sims, limit):
    best: dict[str, tuple[float, str]] = {}
    for i in np.argsort(-sims)[: limit * 8]:
        c, h = idx["labels"][i]
        if float(sims[i]) > best.get(c, (-1.0,))[0]:
            best[c] = (float(sims[i]), h)
    return sorted(((c, round(s, 3), h) for c, (s, h) in best.items()), key=lambda x: -x[1])[:limit]


_warm_jobs: list = []
_warm_thread: threading.Thread | None = None


def warm(users_and_docs) -> None:
    """Build every user's index in the background (low priority); called at startup."""
    _warm_jobs[:] = list(users_and_docs)
    _start_warm()


def building() -> bool:
    """True while the background job is embedding (after a restart or a large change)."""
    return _warm_thread is not None and _warm_thread.is_alive()


def _start_warm() -> None:
    global _warm_thread
    if _warm_thread is not None and _warm_thread.is_alive():
        return

    def run():
        for user_dir, load_docs in _warm_jobs:
            try:
                with _lock:
                    _ensure(user_dir, load_docs())
            except Exception as e:
                log.error("embedding warm-up for %s failed: %s", user_dir, e)
    _warm_thread = threading.Thread(target=run, name="embed-warm", daemon=True)
    _warm_thread.start()
