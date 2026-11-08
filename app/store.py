"""Per-user OKF bundles on disk, each a git repository.

  /data/users.json            OIDC subject -> {dir, email, name}
  /data/bundles/<dir>/        that user's bundle: concepts, index.md files, log.md, .git

Users are keyed by the OIDC `sub` (stable even if a username changes); the
directory name is fixed the first time someone signs in. Every change is a git
commit, so nothing is lost by an edit or delete.
"""

import asyncio
import json
import os
import posixpath
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import okf

DATA = Path(os.environ.get("KB_DATA", "/data"))
BUNDLES = DATA / "bundles"
USERS_FILE = DATA / "users.json"
_users_lock = asyncio.Lock()
_bundle_locks: dict[str, asyncio.Lock] = {}
_parsed: dict[str, tuple[float, dict, str]] = {}  # path -> (mtime, meta, body); Bundle objects are short-lived


@dataclass(frozen=True)
class User:
    sub: str
    dir: str
    email: str
    name: str

    @property
    def actor(self) -> str:
        return okf.actor_for_user(self.dir)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _load_users() -> dict:
    return json.loads(USERS_FILE.read_text()) if USERS_FILE.exists() else {}


async def user_for(claims: dict) -> User:
    """The bundle owner for a set of OIDC claims, creating the bundle on first use."""
    sub = str(claims.get("sub") or "")
    email = str(claims.get("email") or "").lower()
    if not sub or not email:
        raise PermissionError("the sign-in didn't include a user id and email")
    name = str(claims.get("name") or claims.get("preferred_username") or email)
    async with _users_lock:
        users = _load_users()
        entry = users.get(sub)
        if not entry:
            base = re.sub(r"[^a-z0-9-]+", "-", str(claims.get("preferred_username") or email.split("@")[0]).lower()).strip("-") or "user"
            taken = {u["dir"] for u in users.values()}
            dir_name, n = base, 2
            while dir_name in taken:
                dir_name, n = f"{base}-{n}", n + 1
            entry = users[sub] = {"dir": dir_name, "email": email, "name": name}
            _atomic_write(USERS_FILE, json.dumps(users, indent=2))
        elif entry.get("email") != email or entry.get("name") != name:
            entry.update(email=email, name=name)
            _atomic_write(USERS_FILE, json.dumps(users, indent=2))
    user = User(sub=sub, dir=entry["dir"], email=email, name=name)
    await asyncio.to_thread(Bundle(user).ensure)
    return user


def user_by_dir(dir_name: str) -> User | None:
    for sub, u in _load_users().items():
        if u["dir"] == dir_name:
            return User(sub=sub, dir=u["dir"], email=u["email"], name=u["name"])
    return None


def lock(user: User) -> asyncio.Lock:
    return _bundle_locks.setdefault(user.dir, asyncio.Lock())


class Bundle:
    """One user's OKF bundle. Methods are blocking: call them via asyncio.to_thread."""

    def __init__(self, user: User):
        self.user = user
        self.root = BUNDLES / user.dir

    # ── git ──────────────────────────────────────────────────────────────────
    def _git(self, *args: str, check: bool = True) -> str:
        env = {**os.environ, "GIT_AUTHOR_NAME": self.user.name, "GIT_AUTHOR_EMAIL": self.user.email,
               "GIT_COMMITTER_NAME": "okf-wiki-mcp", "GIT_COMMITTER_EMAIL": "okf-wiki-mcp@localhost"}
        out = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True, env=env)
        if check and out.returncode != 0:
            raise RuntimeError(f"git {args[0]} failed: {out.stderr.strip()[:300]}")
        return out.stdout

    def ensure(self) -> None:
        if (self.root / ".git").exists():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self._git("init", "-q", "-b", "main")
        self._git("config", "core.quotepath", "false")
        self._write_raw("log.md", "# Update Log\n")
        self._rebuild_indexes([""])
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "Create knowledge bundle")

    def commit(self, message: str) -> str | None:
        self._git("add", "-A")
        if not self._git("status", "--porcelain").strip():
            return None
        self._git("commit", "-q", "-m", message)
        return self._git("rev-parse", "--short", "HEAD").strip()

    def history(self, concept_id: str | None, limit: int) -> list[dict]:
        args = ["log", f"-{limit}", "--format=%h%x1f%aI%x1f%an%x1f%s"]
        if concept_id:
            args += ["--follow", "--", f"{concept_id}.md"]
        rows = [line.split("\x1f") for line in self._git(*args).splitlines() if line]
        return [{"revision": r[0], "at": r[1], "author": r[2], "message": r[3]} for r in rows]

    def changes(self, concept_id: str | None, limit: int) -> list[dict]:
        """Like history, plus the files each commit touched: [(status, path)], status A/M/D/R.
        For one concept (followed across moves), its path at that commit is `path`."""
        args = ["log", f"-{limit}", "--format=%x1e%h%x1f%aI%x1f%an%x1f%s", "--name-status"]
        if concept_id:
            args += ["--follow", "--", f"{concept_id}.md"]
        out = []
        for block in self._git(*args).split("\x1e")[1:]:
            head, *lines = block.strip("\n").split("\n")
            rev, at, author, message = head.split("\x1f")
            files = [(f.split("\t")[0][:1], f.split("\t")[-1]) for f in lines if "\t" in f]
            out.append({"revision": rev, "at": at, "author": author, "message": message, "files": files,
                        "path": files[0][1] if concept_id and files else None})
        return out

    def text_at(self, concept_id: str, revision: str, path: str | None = None) -> str:
        """A concept's markdown at a revision (`path`: its file then, if it has moved since)."""
        if not re.fullmatch(r"[0-9a-f]{4,40}", revision):
            raise okf.ConceptError("revision must be a commit id from kb_history")
        out = subprocess.run(["git", "-C", str(self.root), "show", f"{revision}:{path or concept_id + '.md'}"],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise okf.ConceptError(f"{concept_id} doesn't exist at revision {revision}")
        return out.stdout

    # ── files ────────────────────────────────────────────────────────────────
    def path(self, concept_id: str) -> Path:
        return self.root / f"{concept_id}.md"

    def _write_raw(self, rel: str, text: str) -> None:
        _atomic_write(self.root / rel, text)

    def exists(self, concept_id: str) -> bool:
        return self.path(concept_id).is_file()

    def read(self, concept_id: str) -> tuple[dict, str]:
        p = self.path(concept_id)
        if not p.is_file():
            raise okf.ConceptError(f"no concept '{concept_id}' — use kb_search or kb_list to find IDs")
        mtime = p.stat().st_mtime
        hit = _parsed.get(str(p))
        if hit and hit[0] == mtime:
            return dict(hit[1]), hit[2]
        meta, body = okf.parse(p.read_text(encoding="utf-8"))
        _parsed[str(p)] = (mtime, meta, body)
        return dict(meta), body

    def resolve(self, concept_id: str) -> str | None:
        """The concept now living at `concept_id`: itself, or the one listing it in `aliases`
        (it was moved or merged there). None if neither exists."""
        if self.exists(concept_id):
            return concept_id
        return next((c for c, m, _ in self.all() if concept_id in (m.get("aliases") or [])), None)

    def concept_ids(self, folder: str = "") -> list[str]:
        base = self.root / folder if folder else self.root
        if not base.is_dir():
            return []
        out = []
        for p in base.rglob("*.md"):
            rel = p.relative_to(self.root).as_posix()
            if rel.startswith(".git/") or posixpath.basename(rel)[:-3] in okf.RESERVED:
                continue
            out.append(rel[:-3])
        return sorted(out)

    def all(self, folder: str = "") -> list[tuple[str, dict, str]]:
        docs = []
        for cid in self.concept_ids(folder):
            try:
                meta, body = self.read(cid)
            except (OSError, okf.ConceptError):
                continue
            docs.append((cid, meta, body))
        return docs

    def save(self, concept_id: str, meta: dict, body: str, log_entry: str) -> str | None:
        okf.validate(meta)
        self._write_raw(f"{concept_id}.md", okf.dump(meta, body))
        self._after_change([concept_id], log_entry)
        return self.commit(f"{log_entry.split(':', 1)[0]}: {concept_id}")

    def delete(self, concept_id: str) -> str | None:
        self.path(concept_id).unlink()
        self._prune_empty_dirs(posixpath.dirname(concept_id))
        self._after_change([concept_id], f"Deletion: removed `{concept_id}`")
        return self.commit(f"Deletion: {concept_id}")

    def move(self, old: str, new: str) -> tuple[str | None, list[str]]:
        """Rename a concept and rewrite absolute links to it in every other concept."""
        target = self.path(new)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.path(old), target)
        self._prune_empty_dirs(posixpath.dirname(old))
        # Like a wiki redirect: the old id keeps resolving (resolve(), kb_read, /c/<old>).
        meta, body = okf.parse(target.read_text(encoding="utf-8"))
        meta["aliases"] = [a for a in [*(meta.get("aliases") or []), old] if a != new]
        _atomic_write(target, okf.dump(meta, body))
        updated = []
        pattern = re.compile(r"\]\(/" + re.escape(old) + r"\.md(#[^)]*)?\)")
        for cid in self.concept_ids():
            p = self.path(cid)
            text = p.read_text(encoding="utf-8")
            new_text = pattern.sub(lambda m: f"](/{new}.md{m.group(1) or ''})", text)
            if new_text != text:
                _atomic_write(p, new_text)
                updated.append(cid)
        self._after_change([old, new], f"Update: moved `{old}` to [{new}](/{new}.md)")
        return self.commit(f"Move: {old} -> {new}"), updated

    def _prune_empty_dirs(self, folder: str) -> None:
        while folder:
            d = self.root / folder
            leftovers = [p for p in d.iterdir() if p.name != "index.md"] if d.is_dir() else []
            if leftovers:
                return
            if (d / "index.md").exists():
                (d / "index.md").unlink()
            if d.is_dir():
                d.rmdir()
            folder = posixpath.dirname(folder)

    # ── index.md / log.md (OKF §8, §9) ──────────────────────────────────────
    def _after_change(self, concept_ids: list[str], log_entry: str) -> None:
        folders = {""}
        for cid in concept_ids:
            d = posixpath.dirname(cid)
            while d:
                folders.add(d)
                d = posixpath.dirname(d)
        self._rebuild_indexes(sorted(folders, key=len, reverse=True))
        self._append_log(log_entry)

    def _rebuild_indexes(self, folders: list[str]) -> None:
        for folder in folders:
            d = self.root / folder if folder else self.root
            if not d.is_dir():
                continue
            concepts, subdirs = [], []
            for p in sorted(d.iterdir()):
                if p.name.startswith("."):
                    continue
                if p.is_dir():
                    count = len(self.concept_ids(p.relative_to(self.root).as_posix()))
                    if count:
                        subdirs.append(f"* [{p.name}/]({p.name}/) - {count} concept{'s' * (count != 1)}")
                elif p.suffix == ".md" and p.stem not in okf.RESERVED:
                    meta, _ = okf.parse(p.read_text(encoding="utf-8"))
                    title = meta.get("title") or p.stem
                    desc = f" - {meta['description']}" if meta.get("description") else ""
                    concepts.append(f"* [{title}]({p.name}){desc} `{meta.get('type', '?')}`")
            parts = ["---", f'okf_version: "{okf.OKF_VERSION}"', "---", ""] if not folder else []
            parts.append(f"# {folder or self.user.name + ' — knowledge'}")
            if subdirs:
                parts += ["", "## Folders", "", *subdirs]
            if concepts:
                parts += ["", "## Concepts", "", *concepts]
            if not subdirs and not concepts:
                parts += ["", "_Empty._"]
            self._write_raw(f"{folder}/index.md" if folder else "index.md", "\n".join(parts) + "\n")

    def _append_log(self, entry: str) -> None:
        path = self.root / "log.md"
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else ["# Update Log"]
        today = datetime.now(timezone.utc).date().isoformat()
        head = lines[:1] or ["# Update Log"]
        rest = lines[1:]
        while rest and not rest[0].strip():
            rest = rest[1:]
        bold, _, text = entry.partition(": ")
        bullet = f"* **{bold}**: {text}"
        if rest and rest[0] == f"## {today}":
            rest = [rest[0], bullet, *rest[1:]]
        else:
            rest = [f"## {today}", bullet, "", *rest]
        self._write_raw("log.md", "\n".join([*head, "", *rest]).rstrip() + "\n")

    def export_tar(self) -> bytes:
        """The bundle as a .tar.gz with its git history: a folder `<dir>-knowledge/` that is both a
        portable OKF bundle and a complete git repository (`git log` works after extracting).
        Hold lock(user) while calling it, so no commit lands halfway through."""
        out = subprocess.run(["tar", "czf", "-", "-C", str(BUNDLES), "--transform",
                              f"s,^{self.user.dir},{self.user.dir}-knowledge,", self.user.dir],
                             capture_output=True, check=True)
        return out.stdout
