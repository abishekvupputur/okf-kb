"""Delete day change logs from sources/ once they are older than KEEP_DAYS (default 90).

A change log is a source whose id ends in `change-log-<YYYY-MM-DD>` (stored by /journal-ingest); its facts
already live on the topic pages. Citations to a deleted log are turned into plain text so no link dangles:
the footnote link becomes its title plus "(change log, deleted after N days)", and `sources` entries keep
their id and title with that note as the resource. Run daily: docker exec kb python -m app.prune [--dry-run]
"""
import datetime as dt
import os
import re
import sys

from . import store

KEEP_DAYS = int(os.environ.get("KB_CHANGELOG_DAYS", "90"))
_ID = re.compile(r"^sources/(?:.*-)?change-log-(\d{4}-\d\d-\d\d)$")


def expired(cid: str, today: dt.date) -> bool:
    m = _ID.match(cid)
    return bool(m) and (today - dt.date.fromisoformat(m.group(1))).days > KEEP_DAYS


def unlink(meta: dict, body: str, gone: set[str]) -> tuple[dict, str, bool]:
    """Turn links and `sources` entries pointing at deleted logs into plain text."""
    note = f"(change log, deleted after {KEEP_DAYS} days)"
    new = re.sub(r"\[([^\]]+)\]\(/(sources/[^)]+?)\.md\)",
                 lambda m: f"{m.group(1)} {note}" if m.group(2) in gone else m.group(0), body)
    changed = new != body
    for s in meta.get("sources") or []:
        if str(s.get("resource", "")).removeprefix("/").removesuffix(".md") in gone:
            s["resource"] = note
            changed = True
    return meta, new, changed


def run(user_dir: str, today: dt.date, dry: bool) -> list[str]:
    b = store.Bundle(store.user_by_dir(user_dir))
    docs = b.all()
    gone = {c for c, _, _ in docs if expired(c, today)}
    if not gone or dry:
        return sorted(gone)
    for c, m, body in docs:
        if c not in gone and not c.startswith("sources/"):
            m, body, changed = unlink(m, body, gone)
            if changed:
                b.save(c, m, body, f"Update: citations of change logs older than {KEEP_DAYS} days made plain text")
    for c in sorted(gone):
        b.delete(c)
    return sorted(gone)


if __name__ == "__main__":
    t = dt.date(2026, 11, 6)
    assert expired("sources/web-01-change-log-2026-08-01", t) and not expired("sources/web-01-change-log-2026-10-20", t)
    assert not expired("sources/nginx-1-27-release-notes", t)
    _, txt, ch = unlink({"sources": [{"id": "a", "resource": "/sources/x-change-log-2026-01-01.md"}]},
                        "see [log](/sources/x-change-log-2026-01-01.md) and [y](/sources/y.md)", {"sources/x-change-log-2026-01-01"})
    assert ch and "[log]" not in txt and "[y](/sources/y.md)" in txt
    dry = "--dry-run" in sys.argv
    for u in store._load_users().values():
        print(u["dir"], ("would delete" if dry else "deleted"), run(u["dir"], dt.date.today(), dry))
