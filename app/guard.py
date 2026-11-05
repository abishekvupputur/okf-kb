"""Write guardrails: the bundle's structure is enforced, not just recommended.

The layout works like a wiki (e.g. Tolkien Gateway): the root holds only portals,
one per top-level folder; every page sits under an existing parent (its portal, or a
hub page of the same name as its folder); every page opens with a short lead; long
topics are split into child pages instead of growing without end. `check` runs before
every kb_write / kb_update / kb_move and raises GuardError for a bad write; softer
problems come back as warnings. kb_review (its lint part) runs the same rules over existing pages.
"""

import posixpath
import re

from . import okf


class GuardError(okf.ConceptError):
    """A write that would break the bundle's structure or leak a secret."""


TYPES = ("Portal", "Project", "Reference", "Playbook", "Decision", "Comparison", "Synthesis", "Log", "Person",
         "Session", "Note", "Metric", "Source")
READ_LIMIT = 6000      # kb_read returns longer bodies as lead + outline unless asked
SOFT_LIMIT = 9000      # warn: time to split
HARD_LIMIT = 12000     # refuse: split into child pages first
SECTION_LIMIT = 4000   # a section this long wants its own page
DESC_LIMIT = 300       # descriptions appear in every index and listing
LEAD_MIN, LEAD_MAX = 40, 1500
SHRINK = 0.5           # a rewrite keeping less than this share of a page is refused without shrink_ok
HUBLESS = {"sessions", "sources"}  # working notes (Claude Code /session) and raw sources: no hub, lead or infobox rules
SOURCE_LIMIT = 200000  # raw sources (sources/, type Source) are stored whole, up to this size

_SECRETS = [
    (re.compile(r"tskey-[A-Za-z0-9-]{10,}"), "auth key"),
    (re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}"), "GitHub token"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), "Google API key"),
    (re.compile(r"GOCSPX-[A-Za-z0-9_-]{10,}"), "Google OAuth client secret"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "JWT / tunnel token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"\b(sk-[A-Za-z0-9]{20,}|xox[abp]-[A-Za-z0-9-]{10,})"), "API token"),
    (re.compile(r"(?i)\b[A-Z0-9_]*(SECRET|PASSWORD|PASSPHRASE|TOKEN|API_?KEY)[A-Z0-9_]*\s*[:=]\s*['\"]?[A-Za-z0-9+/_=-]{16,}"),
     "a value after a secret-looking key"),
]


def lead(body: str) -> str:
    heads = okf.sections(body)
    return (body[:heads[0]["start"]] if heads else body).strip()


def _free(cid: str) -> bool:
    return cid.split("/", 1)[0] in HUBLESS


def placement(cid: str, meta: dict, exists) -> list[str]:
    """Where a concept may live. `exists(id)` tells whether a concept exists."""
    parts = cid.split("/")
    if len(parts) == 1:
        return [] if meta.get("type") == "Portal" else [
            f"'{cid}' is at the root, which holds only portals (type Portal). File it under a portal, "
            f"e.g. 'reference/{cid}', or make it a Portal for a new top-level folder."]
    if meta.get("type") == "Portal":
        return [f"'{cid}': portals live at the root only."]
    if not exists(parts[0]):
        return [f"top-level folder '{parts[0]}' has no portal page '{parts[0]}'. Use an existing folder "
                f"(kb_list()) or create the portal first (type Portal)."]
    parent = posixpath.dirname(cid)
    if len(parts) > 2 and not _free(cid) and not exists(parent):
        return [f"'{cid}' needs its parent hub '{parent}' first: a short page summarising what the "
                f"folder '{parent}/' is about. Create it, or file this page one level up."]
    return []


def secrets(text: str) -> list[str]:
    return [label for rx, label in _SECRETS if rx.search(text)]


def shape(cid: str, meta: dict, body: str) -> tuple[list[str], list[str]]:
    """(errors, warnings) about the page itself, independent of the rest of the bundle."""
    errors, warnings = [], []
    if meta.get("type") not in TYPES:
        errors.append(f"type must be one of {', '.join(TYPES)} (got {meta.get('type')!r})")
    for key in ("title", "description", "tags"):
        if not meta.get(key):
            errors.append(f"'{key}' is required, so the page can be found from indexes and search")
    if len(str(meta.get("description") or "")) > DESC_LIMIT:
        errors.append(f"description is {len(str(meta['description']))} chars; keep it to one sentence (≤ {DESC_LIMIT})")
    review = meta.get("review")
    if review is not None and not (isinstance(review, list) and all(isinstance(r, str) and r.strip() for r in review)):
        errors.append("review must be a list of questions for the user (strings)")
    is_source = meta.get("type") == "Source"
    if is_source != cid.startswith("sources/"):
        errors.append("raw sources (type Source) live in sources/ and nothing else does; use kb_ingest to add one")
    if is_source:
        if len(body) > SOURCE_LIMIT:
            errors.append(f"source is {len(body)} chars (limit {SOURCE_LIMIT}); ingest it in parts")
    elif len(body) > HARD_LIMIT:
        errors.append(f"body is {len(body)} chars (limit {HARD_LIMIT}). Split it: keep a summary here and move "
                      f"sections into child pages under '{cid}/'")
    elif len(body) > SOFT_LIMIT:
        warnings.append(f"body is {len(body)} chars; split sections into child pages under '{cid}/' before it reaches {HARD_LIMIT}")
    if not _free(cid):
        ld = lead(body)
        if len(ld) < LEAD_MIN:
            errors.append("start the body with a lead: 1–3 sentences saying what this is, before the first heading")
        elif len(ld) > LEAD_MAX:
            warnings.append(f"lead is {len(ld)} chars; keep it short and move detail into sections")
        for h in okf.sections(body) if meta.get("type") != "Log" else []:   # a day's log is one list
            if h["end"] - h["start"] > SECTION_LIMIT and not any(
                    k["start"] > h["start"] and k["end"] <= h["end"] for k in okf.sections(body)):
                warnings.append(f"section '{h['heading']}' is {h['end'] - h['start']} chars; consider its own page")
    if not is_source and re.search(r"\\n(#{1,6} |\\n|\| |- )", body.replace("```", "\0")):
        errors.append("the body contains literal '\\n' sequences instead of line breaks (an escaping mistake); "
                      "send real newlines")
    if leaked := secrets(body + "\n" + str(meta)):
        errors.append(f"looks like it contains a secret ({', '.join(leaked)}). Store only the key's name and "
                      f"where it lives, never the value")
    if okf.signals(meta)["status"] == "deprecated" and not cid.startswith("archive/"):
        warnings.append(f"deprecated pages belong in archive/: kb_move('{cid}', 'archive/{posixpath.basename(cid)}')")
    return errors, warnings


def check(bundle, cid: str, meta: dict, body: str, *, old_body: str | None = None,
          creating: bool = False, shrink_ok: bool = False) -> list[str]:
    """Raise GuardError for a bad write; return warnings for the caller to pass on."""
    errors, warnings = shape(cid, meta, body)
    errors += placement(cid, meta, bundle.exists)
    if old_body is not None and cid.startswith("sources/") and body != old_body:
        errors.append("raw sources are immutable: record what changed in the pages that cite it, or ingest the new version")
    if old_body is not None and len(old_body) > 1000 and len(body) < SHRINK * len(old_body) and not shrink_ok:
        errors.append(f"this would cut the page from {len(old_body)} to {len(body)} chars. If you're splitting it, "
                      f"write the child pages first, then pass shrink_ok=true")
    if creating:
        title = str(meta.get("title") or "").strip().lower()
        for other, m, _ in bundle.all():
            if title and str(m.get("title") or "").strip().lower() == title:
                errors.append(f"'{other}' already has the title {meta['title']!r}: update it instead of duplicating")
            if cid in (m.get("aliases") or []):
                errors.append(f"'{cid}' is an old name of '{other}' (moved there): update that page instead")
    if errors:
        raise GuardError("Refused: " + " · ".join(errors))
    return warnings
