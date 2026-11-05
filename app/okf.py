"""Open Knowledge Format (OKF) v0.2: parsing, writing and the derived signals.

A concept is a markdown file with YAML frontmatter; its ID is the path inside the
bundle without ".md". index.md and log.md are reserved. Spec:
https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md
"""

import posixpath
import re
from datetime import date, datetime, timezone

import yaml
from fastmcp.exceptions import ToolError

OKF_VERSION = "0.2"
RESERVED = {"index", "log"}
STATUSES = ("draft", "stable", "deprecated")
_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")
_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


class ConceptError(ToolError, ValueError):
    """Invalid concept ID, frontmatter or content (shown to the caller as is)."""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_id(concept_id: str) -> str:
    """'Projects/Garden Plan.md' -> 'projects/garden-plan'. Rejects anything path-like or reserved."""
    cid = concept_id.strip().strip("/")
    if cid.endswith(".md"):
        cid = cid[:-3]
    cid = re.sub(r"\s+", "-", cid.lower())
    parts = cid.split("/")
    if not cid or len(parts) > 8 or any(not _SEGMENT.match(p) or p in (".", "..") for p in parts):
        raise ConceptError(
            f"invalid concept id {concept_id!r}: use lowercase path segments of a-z, 0-9, '.', '_', '-' "
            "separated by '/', e.g. 'projects/garden'")
    if parts[-1] in RESERVED:
        raise ConceptError(f"'{parts[-1]}' is reserved by OKF (index.md / log.md)")
    return cid


def normalize_folder(folder: str) -> str:
    folder = (folder or "").strip().strip("/")
    if not folder:
        return ""
    parts = folder.lower().split("/")
    if any(not _SEGMENT.match(p) or p in (".", "..") for p in parts):
        raise ConceptError(f"invalid folder {folder!r}")
    return "/".join(parts)


def _plain(value):
    """YAML timestamps -> ISO strings, recursively, so everything is JSON-friendly."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def parse(text: str) -> tuple[dict, str]:
    """(frontmatter, body). A file without frontmatter yields ({}, text)."""
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", text, re.S)
    if not match:
        return {}, text
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(meta, dict):
        return {}, text
    return _plain(meta), text[match.end():].lstrip("\n")


# Conventional key order when writing; unknown keys keep their order after these.
_ORDER = ["type", "title", "description", "resource", "tags", "aliases", "infobox", "status", "stale_after",
          "generated", "verified", "sources", "usage_window"]


def dump(meta: dict, body: str) -> str:
    ordered = {k: meta[k] for k in _ORDER if k in meta and meta[k] not in (None, [], "")}
    ordered.update({k: v for k, v in meta.items() if k not in ordered and v not in (None, [], "")})
    front = yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, width=100).strip()
    return f"---\n{front}\n---\n\n{body.strip()}\n"


def validate(meta: dict) -> None:
    if not str(meta.get("type") or "").strip():
        raise ConceptError("frontmatter needs a non-empty 'type' (e.g. Note, Playbook, Reference, Person, Project)")
    if "status" in meta and meta["status"] not in STATUSES:
        raise ConceptError(f"status must be one of {', '.join(STATUSES)}")
    if "tags" in meta and not (isinstance(meta["tags"], list) and all(isinstance(t, str) for t in meta["tags"])):
        raise ConceptError("tags must be a list of strings")
    if "aliases" in meta and not (isinstance(meta["aliases"], list) and all(isinstance(a, str) for a in meta["aliases"])):
        raise ConceptError("aliases must be a list of old concept IDs")
    box = meta.get("infobox")
    if box is not None and not (isinstance(box, dict) and all(
            isinstance(v, (str, int, float)) or (isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v))
            for v in box.values())):
        raise ConceptError("infobox must map labels to short values (text, numbers or lists of them)")
    if "stale_after" in meta:
        parse_time(meta["stale_after"], "stale_after")
    for i, src in enumerate(meta.get("sources") or []):
        if not isinstance(src, dict) or not src.get("resource"):
            raise ConceptError(f"sources[{i}] needs a 'resource' (URL, bundle path or scope description)")
    for v in verified_list(meta):
        if not isinstance(v, dict) or not v.get("by"):
            raise ConceptError("each verified entry needs 'by' (an actor) and 'at'")


def parse_time(value, field: str = "time") -> datetime:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise ConceptError(f"{field} must be an ISO 8601 datetime like 2026-12-31T00:00:00Z") from None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def verified_list(meta: dict) -> list:
    v = meta.get("verified")
    if v is None:
        return []
    return [v] if isinstance(v, dict) else list(v)


def trust_tier(meta: dict) -> str:
    """§5.3: unverified < machine-confirmed < human-reviewed."""
    actors = [str(v.get("by", "")) for v in verified_list(meta) if isinstance(v, dict)]
    if not actors:
        return "unverified"
    return "human-reviewed" if any(a.startswith("human:") for a in actors) else "machine-confirmed"


def is_stale(meta: dict) -> bool:
    if not meta.get("stale_after"):
        return False
    try:
        return datetime.now(timezone.utc) >= parse_time(meta["stale_after"])
    except ConceptError:
        return False


def signals(meta: dict) -> dict:
    """The derived, consumer-side view of a concept's trust and lifecycle."""
    verified = verified_list(meta)
    return {
        "status": meta.get("status", "stable"),
        "trust": trust_tier(meta),
        "stale": is_stale(meta),
        "last_verified": max((str(v.get("at", "")) for v in verified if isinstance(v, dict)), default=None),
        "generated": meta.get("generated"),
    }


def links(concept_id: str, body: str) -> list[str]:
    """Concept IDs this body links to (absolute '/x/y.md' or relative './y.md' links)."""
    out = []
    base = posixpath.dirname(concept_id)
    prose = re.sub(r"`[^`\n]*`", "", re.sub(r"(?ms)^[ \t]*(```|~~~).*?^[ \t]*\1[^\n]*$", "", body))  # code isn't a link
    for _, target in _LINK.findall(prose):
        resolved = resolve_link(base, target)
        if resolved and resolved not in out:
            out.append(resolved)
    return out


def resolve_link(base_dir: str, target: str) -> str | None:
    """A markdown link target -> concept ID, or None for external/non-concept links."""
    target = target.split("#", 1)[0]
    if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or not target.endswith(".md"):
        return None
    path = target[1:] if target.startswith("/") else posixpath.join(base_dir, target)
    path = posixpath.normpath(path)
    if path.startswith("..") or path.startswith("/"):
        return None
    cid = path[:-3]
    return None if posixpath.basename(cid) in RESERVED else cid


_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")


def sections(body: str) -> list[dict]:
    """The body's headings in order, each with the character span of its section
    (up to the next heading of the same or a higher level). Fenced code is skipped."""
    heads, pos, fenced = [], 0, False
    for line in body.splitlines(keepends=True):
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and (m := _HEADING.match(line.rstrip("\r\n"))):
            heads.append({"level": len(m[1]), "heading": m[2].strip(), "start": pos})
        pos += len(line)
    for i, h in enumerate(heads):
        h["end"] = next((n["start"] for n in heads[i + 1:] if n["level"] <= h["level"]), len(body))
    return heads


def find_section(body: str, name: str) -> dict | None:
    """The section whose heading equals `name` (case-insensitive), else the first one containing it."""
    want = name.strip().lstrip("#").strip().lower()
    heads = sections(body)
    return (next((h for h in heads if h["heading"].lower() == want), None)
            or next((h for h in heads if want and want in h["heading"].lower()), None))


def cited(meta: dict, known) -> list[str]:
    """Concepts this page cites in its `sources` (e.g. raw sources: resource '/sources/x.md')."""
    out = []
    for s in meta.get("sources") or []:
        res = str(s.get("resource") or "").strip() if isinstance(s, dict) else ""
        cid = resolve_link("", res if res.endswith(".md") else res + ".md") if res and " " not in res else None
        if cid and cid in known and cid not in out:
            out.append(cid)
    return out


def actor_for_user(username: str) -> str:
    return f"human:{username}"
