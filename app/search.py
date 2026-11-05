"""Keyword search over the bundle (phase 1 of the query pipeline, after nashsu/llm_wiki).

Words are split and stop words dropped; CJK runs become bigrams. A page scores by where the
terms appear (title strongest, then tags, description, ID/aliases, body), plus a bonus when
the whole query appears in the title.
"""

import math
import re
from functools import lru_cache

STOP = set("""a an and are as at be but by can could did do does for from had has have how i if in into is it
its me my no not of on or our should so than that the their them then there these they this to was we were
what when where which who why will with would you your about after all also any been before being between
both each few more most other over same some such only own under until up very via just ever per der die das
und ist ein eine mit für von zu den dem des im""".split())
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]+")
TITLE_PHRASE_BONUS = 10

# Words the user may use for what the pages name differently. Each group's members find each
# other at half weight (see expand/score), so a page using the exact word still ranks first.
# Keep it short; single lowercase words, already stemmed forms are fine.
SYNONYM_GROUPS = [
    ["sso", "oidc"],
]


def stem(w: str) -> str:
    """A common prefix of a word's forms, so a substring match finds all of them:
    backups/backup -> backup, policies/policy -> polic, patches -> patch, renamed/renaming -> renam.
    Identifiers (digits, dots, hyphens) and short words are left alone."""
    if len(w) < 4 or not w.isalpha():
        return w
    for suffix, keep in (("ies", 4), ("ing", 4), ("ed", 4)):
        if w.endswith(suffix) and len(w) - len(suffix) >= keep:
            return w[:-len(suffix)]
    if w.endswith("y") and len(w) >= 6 and w[-2] not in "aeiou":   # policy -> polic, not relay -> rela
        return w[:-1]
    if w.endswith("es") and w[:-2].endswith(("s", "x", "z", "ch", "sh")) and len(w) - 2 >= 3:
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]   # shares -> share, then the -e rule below -> shar
    if w.endswith("e") and len(w) >= 5:   # rename -> renam (finds renamed, renaming), update -> updat
        return w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    text = text.lower()
    terms = [stem(w) for w in re.findall(r"[^\W_]+(?:[-.][^\W_]+)*", _CJK.sub(" ", text)) if len(w) > 1 and w not in STOP]
    for run in _CJK.findall(text):
        terms += [run] if len(run) < 2 else [run[i:i + 2] for i in range(len(run) - 1)]
    return list(dict.fromkeys(terms))


_SYNONYMS = {stem(w): [stem(o) for o in g if o != w] for g in SYNONYM_GROUPS for w in g}


def expand(terms: list[str]) -> list[str]:
    """Synonyms of the query terms (not already among them), for score(..., alt=...)."""
    return list(dict.fromkeys(s for t in terms for s in _SYNONYMS.get(t, []) if s not in terms))


_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@-]*[A-Za-z0-9]")


def removed_values(old: str, new: str) -> list[str]:
    """Values an edit took out: identifier-like tokens (versions, IPs, hosts, names with digits,
    dots, hyphens, or a capital letter, e.g. 0.45.1, 7d, app-stack-db-1, OpenSSL) that occur in
    `old` but no longer in `new`. Ordinary words are left out: they'd match everywhere."""
    gone = set(_VALUE.findall(old)) - set(_VALUE.findall(new))
    return sorted(t for t in gone if len(t) > 2 and t.lower() not in STOP
                  and not re.fullmatch(r"[A-Z]{2,}s", t)   # plural acronyms (IDs, APIs, URLs) are ordinary words
                  and (any(c.isdigit() for c in t) or any(c in "._:/@-" for c in t) or any(c.isupper() for c in t[1:])))


def find_value(text: str, value: str) -> int:
    """Position of `value` in `text` as a whole token (not inside a longer value: 0.100 must not
    match 100.100.100.100, kb-1 must not match kb-12), or -1."""
    m = re.search(rf"(?<![A-Za-z0-9._:/@-]){re.escape(value)}(?![A-Za-z0-9_:/@-]|[.-][A-Za-z0-9])", text)
    return m.start() if m else -1


@lru_cache(maxsize=4096)
def _word(term: str) -> re.Pattern:
    # A term (a stem) must start a word: backup finds backups, point doesn't match inside endpoint.
    return re.compile(rf"(?<![^\W_]){re.escape(term)}")


def count(text: str, term: str) -> int:
    """How often `term` starts a word in `text` (lowercase)."""
    return len(_word(term).findall(text))


def _fields(cid: str, meta: dict, body: str) -> tuple[str, str, str, str, str, str]:
    return (str(meta.get("title", "")).lower(), " ".join(meta.get("tags", [])).lower(),
            str(meta.get("description", "")).lower(),
            " ".join(f"{k} {v}" for k, v in (meta.get("infobox") or {}).items()).lower(),
            " ".join([cid, *(meta.get("aliases") or [])]).lower(), body.lower())


def idf(docs, terms: list[str]) -> dict[str, float]:
    """Weight per term: rare terms count more (a term on no page gets the highest weight)."""
    n = len(docs)
    texts = [" ".join(_fields(c, m, b)) for c, m, b in docs]
    return {t: math.log((n + 1) / (sum(1 for x in texts if count(x, t)) + 1)) + 1 for t in terms}


def coverage(cid: str, meta: dict, body: str, terms: list[str], weights: dict[str, float]) -> float:
    """Share of the question's term weight (idf) that the page contains, 0..1. Unlike score(), it is
    absolute: a page holding only a common word of the question covers little of it."""
    text = " ".join(_fields(cid, meta, body))
    total = sum(weights.get(t, 1) for t in terms)
    return sum(weights.get(t, 1) for t in terms if count(text, t)) / total if total else 0.0


def score(cid: str, meta: dict, body: str, terms: list[str], phrase: str = "", alt: list[str] = (),
          weights: dict[str, float] | None = None) -> float:
    """Terms count fully, `alt` (synonyms from expand) at half weight; with `weights`, rare terms more."""
    return _score(cid, meta, body, terms, phrase, weights) + _score(cid, meta, body, alt, "", weights) / 2


def _score(cid: str, meta: dict, body: str, terms: list[str], phrase: str = "", weights: dict | None = None) -> float:
    title, tags, desc, box, ids, low = _fields(cid, meta, body)
    s = sum((weights or {}).get(t, 1) * (6 * bool(count(title, t)) + 4 * bool(count(tags, t)) + 3 * bool(count(desc, t))
                                         + 3 * bool(count(box, t)) + 2 * bool(count(ids, t)) + min(count(low, t), 5))
            for t in terms)
    if s and phrase and len(phrase) > 3 and phrase.lower() in title:
        s += TITLE_PHRASE_BONUS
    return s


def section_scores(body: str, sections: list[dict], terms: list[str],
                   weights: dict[str, float] | None = None) -> list[tuple[float, dict]]:
    """Leaf sections ranked by how many term hits they hold (heading hits count double; with
    `weights`, rare terms more, so the section holding the question's specific word wins)."""
    leaves = [h for h in sections if not any(k["start"] > h["start"] and k["end"] <= h["end"] for k in sections)]
    out = []
    for h in leaves:
        text = body[h["start"]:h["end"]].lower()
        s = sum((weights or {}).get(t, 1) * (min(count(text, t), 5) + 2 * bool(count(h["heading"].lower(), t))) for t in terms)
        if s:
            out.append((s, h))
    return sorted(out, key=lambda x: -x[0])
