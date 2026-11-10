"""Browser side of the knowledge base, as middleware in front of FastMCP.

  /                  home: stats, recent pages, what needs attention, all folders, connect Claude
  /c/<id>            one page (rendered) with its sources, links and review actions
  /c/<id>.md         raw markdown
  /h/<id>[?from=r]   the page's change history, and what changed between revision r and now
  /history           every change in your bundle (your own git repository)
  /search?q=        full search results: keywords, then pages by meaning (embeddings); the top bar
                     searches live by keyword via /search.json
  /graph             link graph of your pages
  /pairs             heatmap of meaning similarity between pages vs their links (pairs.py)
  /export.tar.gz     your bundle with its git history (owner only), as a .tar.gz
  /healthz           liveness, no login
  /static/*          stylesheet + scripts

Every page needs a sign-in, and each user only ever sees their own bundle: nothing can be
shared or published.

Login is the OIDC code flow + PKCE (same OAuth client as the MCP proxy),
giving an 8-hour signed session cookie. POSTs must come from this origin (CSRF).
Everything else (/mcp, OAuth endpoints, metadata) passes through to FastMCP.
"""

import asyncio
import base64
import difflib
import hashlib
import hmac
import json
import os
import posixpath
import re
import secrets
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode
from zoneinfo import ZoneInfo

import httpx
from starlette.requests import Request
from starlette.responses import (FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse,
                                 Response)
from starlette.types import ASGIApp, Receive, Scope, Send

from . import embed, graph, guard, okf, pairs, search, store
from .tools import SEED_MIN
from .render import ago, esc, plain, render
from .store import Bundle, User

PUBLIC_URL = os.environ["PUBLIC_URL"].rstrip("/")
CALLBACK = "/web/callback"
SESSION_COOKIE, FLOW_COOKIE = "kb_session", "kb_login"
SESSION_TTL = 8 * 3600
# Cloudflare challenges library User-Agents from this server's IP.
IDP_HEADERS = {"User-Agent": f"okf-wiki-mcp/1.0 (+{PUBLIC_URL})"}
STATIC = Path(__file__).parent / "static"
# Asset URLs carry a content hash, so a deploy is never hidden behind a cached old copy.
ASSET_VERSION = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(STATIC.rglob("*")) if p.is_file())).hexdigest()[:10]
CSP = ("default-src 'none'; style-src 'self' 'unsafe-inline'; img-src https: data:; "
       "script-src 'self'; connect-src 'self'; form-action 'self'; "
       "frame-ancestors 'none'; base-uri 'none'")
_http = httpx.AsyncClient(timeout=20)
_oidc: dict = {}


# ── signed cookies ─────────────────────────────────────────────────────────────

def _key() -> bytes:
    return hashlib.sha256(("kb-web:" + os.environ["JWT_SIGNING_KEY"]).encode()).digest()


def _sign(data: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(data).encode()).decode()
    return body + "." + hmac.new(_key(), body.encode(), hashlib.sha256).hexdigest()


def _unsign(value: str | None) -> dict | None:
    if not value or "." not in value:
        return None
    body, sig = value.rsplit(".", 1)
    if not hmac.compare_digest(sig, hmac.new(_key(), body.encode(), hashlib.sha256).hexdigest()):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(body))
    except ValueError:
        return None
    return data if data.get("exp", 0) > time.time() else None


def _cookie(resp: Response, name: str, value: str, max_age: int) -> None:
    resp.set_cookie(name, value, max_age=max_age, path="/", secure=True, httponly=True, samesite="lax")


# ── OIDC login ────────────────────────────────────────────────────────────────

async def _config() -> dict:
    if not _oidc:
        r = await _http.get(os.environ["OIDC_CONFIG_URL"], headers=IDP_HEADERS)
        r.raise_for_status()
        _oidc.update(r.json())
    return _oidc


def _safe_next(path: str | None) -> str:
    return path if path and path.startswith("/") and not path.startswith("//") else "/"


async def _login(next_path: str) -> Response:
    cfg = await _config()
    state, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    resp = RedirectResponse(cfg["authorization_endpoint"] + "?" + urlencode({
        "response_type": "code", "client_id": os.environ["OIDC_CLIENT_ID"],
        "redirect_uri": PUBLIC_URL + CALLBACK, "scope": " ".join(s for s in store.OIDC_SCOPES if s != "offline_access"), "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256",
    }), status_code=302)
    _cookie(resp, FLOW_COOKIE, _sign({"state": state, "verifier": verifier, "next": _safe_next(next_path),
                                      "exp": time.time() + 600}), 600)
    return resp


async def _callback(request: Request) -> Response:
    p = request.query_params
    if p.get("error"):
        return PlainTextResponse(f"Login failed: {p.get('error_description') or p['error']}", 403)
    flow = _unsign(request.cookies.get(FLOW_COOKIE))
    if not flow or not hmac.compare_digest(flow["state"], p.get("state", "")):
        return PlainTextResponse("Login expired or invalid — open the knowledge base again.", 400)
    cfg = await _config()
    tok = await _http.post(cfg["token_endpoint"], data={
        "grant_type": "authorization_code", "code": p.get("code", ""),
        "redirect_uri": PUBLIC_URL + CALLBACK, "code_verifier": flow["verifier"],
    }, auth=(os.environ["OIDC_CLIENT_ID"], os.environ["OIDC_CLIENT_SECRET"]), headers=IDP_HEADERS)
    if tok.status_code != 200:
        return PlainTextResponse("Login failed (token exchange).", 403)
    info = await _http.get(cfg["userinfo_endpoint"],
                           headers={**IDP_HEADERS, "Authorization": f"Bearer {tok.json()['access_token']}"})
    if info.status_code != 200:
        return PlainTextResponse("Login failed (user info).", 403)
    try:
        user = await store.user_for(info.json())
    except PermissionError as e:
        return PlainTextResponse(str(e), 403)
    resp = RedirectResponse(flow["next"], status_code=302)
    _cookie(resp, SESSION_COOKIE, _sign({"sub": user.sub, "dir": user.dir, "email": user.email,
                                         "name": user.name, "exp": time.time() + SESSION_TTL}), SESSION_TTL)
    resp.delete_cookie(FLOW_COOKIE, path="/")
    return resp


def _session_user(request: Request) -> User | None:
    s = _unsign(request.cookies.get(SESSION_COOKIE))
    return User(sub=s["sub"], dir=s["dir"], email=s["email"], name=s["name"]) if s else None


# ── words for humans ──────────────────────────────────────────────────────────

def _actor(actor: str | None, viewer: User | None) -> str:
    """'claude/opus-5.5' -> 'Claude (opus-5.5)', 'human:alex' -> 'you' / 'alex'."""
    actor = str(actor or "")
    if actor.startswith("human:"):
        return "you" if viewer and actor == viewer.actor else actor[6:]
    if actor.startswith("process:"):
        return f"an automated process ({actor[8:]})"
    producer, _, version = actor.partition("/")
    name = {"claude": "Claude", "user": "you"}.get(producer, producer or "someone")
    return f"{name} ({version})" if version and version not in ("mcp", "kb-mcp") else name


def _title(cid: str, meta: dict) -> str:
    return str(meta.get("title") or posixpath.basename(cid).replace("-", " ").capitalize())


def _trust_pill(meta: dict, viewer: User | None) -> str:
    verified = sorted((v for v in okf.verified_list(meta) if isinstance(v, dict)), key=lambda v: str(v.get("at", "")))
    tier = okf.trust_tier(meta)
    if tier == "human-reviewed":
        who = _actor([v for v in verified if str(v.get("by", "")).startswith("human:")][-1].get("by"), viewer)
        return f'<span class="pill good" title="A person confirmed this page">Reviewed by {esc(who)}</span>'
    if tier == "machine-confirmed":
        who = _actor(verified[-1].get("by"), viewer).split(" (")[0]
        return f'<span class="pill info" title="Checked against its sources, not yet by a person">Checked by {esc(who)}</span>'
    return '<span class="pill plain" title="Nobody has confirmed this page yet">Not reviewed yet</span>'


def _pills(meta: dict, viewer: User | None, tags: bool = True, link_tags: bool = True) -> str:
    out = [f'<span class="pill type plain">{esc(meta.get("type", "Note"))}</span>', _trust_pill(meta, viewer)]
    status = okf.signals(meta)["status"]
    if status == "draft":
        out.append('<span class="pill warn" title="Not finished yet">Draft</span>')
    elif status == "deprecated":
        out.append('<span class="pill bad" title="Kept for reference; no longer current">Outdated</span>')
    if okf.is_stale(meta):
        out.append(f'<span class="pill bad" title="Re-check was due {esc(str(meta.get("stale_after", ""))[:10])}">Due for re-check</span>')
    if tags:
        for t in meta.get("tags", []):
            out.append(f'<a class="pill tag" href="/search?tag={quote(t)}">#{esc(t)}</a>' if link_tags
                       else f'<span class="pill tag">#{esc(t)}</span>')
    return f'<div class="pills">{"".join(out)}</div>'


def _attention(meta: dict) -> list[str]:
    reasons = ["Question for you"] if meta.get("review") else []
    if okf.is_stale(meta):
        reasons.append("Due for re-check")
    if okf.signals(meta)["status"] == "draft":
        reasons.append("Draft")
    if okf.trust_tier(meta) == "unverified" and okf.signals(meta)["status"] != "deprecated":
        reasons.append("Not reviewed yet")
    return reasons


def _updated(meta: dict) -> str:
    gen = meta.get("generated") if isinstance(meta.get("generated"), dict) else {}
    return str(gen.get("at") or "")


# ── page chrome ────────────────────────────────────────────────────────────────

def _topbar(user: User | None, q: str = "") -> str:
    brand = '<a class="brand" href="/"><span class="mark" aria-hidden="true">📚</span>Knowledge</a>'
    if user is None:
        return f'<header class="topbar">{brand}<nav><a href="/">Sign in</a></nav></header>'
    return f"""<header class="topbar">
  {brand}
  <form class="search" role="search" action="/search" method="get">
    <input type="search" id="q" name="q" value="{esc(q)}" placeholder="Search your knowledge" aria-label="Search your knowledge" autocomplete="off">
    <kbd>/</kbd>
    <ol id="results" hidden></ol>
  </form>
  <nav>
    <a href="/">Home</a><a href="/graph">Graph</a><a href="/pairs">Pairs</a><a href="/history">History</a>
    <span class="who">{esc(user.name.split()[0] if user.name else user.email)}</span>{'' if store.NO_AUTH else '<a href="/logout">Sign out</a>'}
  </nav>
</header>"""


def _page(title: str, main: str, user: User | None, *, sidebar: str = "", toc: str = "", q: str = "",
          scripts: str = "", description: str = "", csp: str = CSP) -> HTMLResponse:
    if sidebar or toc:
        side = (f'<details class="sidebar" open><summary>Contents</summary>{sidebar}</details>'
                if sidebar else "<span></span>")
        content = f'<div class="layout">{side}<main class="prose">{main}</main>{toc}</div>'
    else:
        content = f'<main class="wrap">{main}</main>'
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>{esc(title)} · Knowledge</title>
  <meta name="description" content="{esc(description or title)}">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-title" content="Knowledge">
  <meta name="apple-mobile-web-app-status-bar-style" content="black">
  <meta name="theme-color" content="#0e0e10" media="(prefers-color-scheme: dark)">
  <meta name="theme-color" content="#fbfaf7" media="(prefers-color-scheme: light)">
  <link rel="icon" type="image/png" sizes="32x32" href="/static/icon-32.png?v={ASSET_VERSION}">
  <link rel="apple-touch-icon" sizes="180x180" href="/static/icon-180.png?v={ASSET_VERSION}">
  <link rel="stylesheet" href="/static/style.css?v={ASSET_VERSION}">
  {scripts}
  <script src="/static/kb.js?v={ASSET_VERSION}" data-index="/search.json" defer></script>
</head>
<body class="docs">
{_topbar(user, q)}
{content}
<footer class="footer kb"><div class="wrap">
  <span>Open Knowledge Format v{okf.OKF_VERSION} · plain markdown in git, yours to export any time</span>
  <span><a href="https://github.com/GoogleCloudPlatform/open-knowledge-format">About OKF</a></span>
</div></footer>
</body>
</html>"""
    resp = HTMLResponse(doc)
    resp.headers.update({"Content-Security-Policy": csp, "Cache-Control": "no-store",
                         "X-Content-Type-Options": "nosniff", "Referrer-Policy": "same-origin"})
    return resp


def _message(title: str, text: str, user: User | None, back: str = "/") -> HTMLResponse:
    return _page(title, f'<section class="block"><h1>{esc(title)}</h1><p>{text}</p>'
                        f'<p><a class="btn" href="{esc(back)}">Back</a></p></section>', user)


# Top-level folders: reading order, colour (fc-N in style.css) and icon. Others get a stable
# colour from their name and a plain folder icon.
FOLDERS = {
    "": (0, 0, "📄"), "projects": (1, 0, "🧩"), "infrastructure": (2, 1, "🖥️"), "runbooks": (3, 2, "📘"),
    "decisions": (4, 7, "⚖️"), "research": (5, 3, "🔬"), "work": (6, 6, "💼"), "reference": (7, 5, "📎"),
    "people": (8, 4, "👤"), "journal": (9, 3, "📓"), "meta": (10, 8, "⚙️"),
}


def _top(folder: str) -> str:
    return folder.split("/", 1)[0]


def _folder_style(folder: str) -> tuple[int, int, str]:
    top = _top(folder)
    if top in FOLDERS:
        return FOLDERS[top]
    return (50, sum(map(ord, top)) % 8, "📁")


def _folder_label(folder: str) -> str:
    """'projects/garden' -> 'Projects › garden'."""
    if not folder:
        return "General"
    first, *rest = folder.split("/")
    return " › ".join([first.replace("-", " ").capitalize(), *rest])


def _chip(folder: str, link: bool = True) -> str:
    _, colour, icon = _folder_style(folder)
    inner = f'<span aria-hidden="true">{icon}</span>{esc(_folder_label(folder))}'
    anchor = f"/#f-{esc(folder or 'general').replace('/', '-')}"
    return (f'<a class="chip fc-{colour}" href="{anchor}">{inner}</a>' if link
            else f'<span class="chip fc-{colour}">{inner}</span>')


def _ordered(docs: list[tuple[str, dict, str]]) -> list[tuple[str, dict, str]]:
    """Reading order: folders in FOLDERS order (sub-folders after their parent), then by title."""
    return sorted(docs, key=lambda d: (_folder_style(posixpath.dirname(d[0]))[0], posixpath.dirname(d[0]),
                                       _title(d[0], d[1]).lower()))


def _sidebar(docs: list[tuple[str, dict, str]], current: str | None, href=lambda c: f"/c/{c}") -> str:
    out, folder = [], None
    for cid, meta, _ in _ordered(docs):
        f = posixpath.dirname(cid)
        if f != folder:
            if folder is not None:
                out.append("</ul>")
            _, colour, icon = _folder_style(f)
            out.append(f'<p class="group fc-{colour}"><span aria-hidden="true">{icon}</span>{esc(_folder_label(f))}</p><ul>')
            folder = f
        cur = ' aria-current="page"' if cid == current else ""
        out.append(f'<li><a href="{esc(href(cid))}"{cur}>{esc(_title(cid, meta))}</a></li>')
    if folder is not None:
        out.append("</ul>")
    return "".join(out)


def _toc(outline: list[tuple[int, str, str]]) -> str:
    if not outline:
        return ""
    items = "".join(f'<li class="l{lvl}"><a href="#{esc(i)}">{esc(t)}</a></li>' for lvl, i, t in outline)
    return f'<nav class="toc" aria-label="On this page"><p>On this page</p><ul>{items}</ul></nav>'


def _csrf_ok(request: Request) -> bool:
    origin = request.headers.get("origin") or ""
    if origin:
        return origin == PUBLIC_URL
    return (request.headers.get("referer") or "").startswith(PUBLIC_URL + "/")


# ── pages ──────────────────────────────────────────────────────────────────────

_REASON_CLASS = {"Question for you": "warn", "Due for re-check": "bad", "Draft": "warn", "Not reviewed yet": "plain"}


def _attention_row(cid: str, meta: dict, reasons: list[str]) -> str:
    pills = "".join(f'<span class="pill {_REASON_CLASS[r]}">{esc(r)}</span>' for r in reasons)
    return (f'<li><span class="t"><a href="/c/{esc(cid)}">{esc(_title(cid, meta))}</a></span>'
            f'<span class="m">{_chip(posixpath.dirname(cid))}{pills}</span></li>')


def _row(cid: str, meta: dict, user: User, *, folder: bool, when: bool) -> str:
    """One readable line per page: title and summary on the left, facts on the right."""
    desc = f'<span class="d">{esc(meta["description"])}</span>' if meta.get("description") else ""
    facts = [_chip(posixpath.dirname(cid)) if folder else f'<span class="pill type plain">{esc(meta.get("type", "Note"))}</span>',
             _trust_pill(meta, user)]
    if okf.is_stale(meta):
        facts.append('<span class="pill bad">Due for re-check</span>')
    if when:
        gen = meta.get("generated") if isinstance(meta.get("generated"), dict) else {}
        facts.append(f'<span class="ago" title="{esc(gen.get("at", ""))}">{esc(ago(gen.get("at")))}</span>')
    return (f'<li><span class="t"><a href="/c/{esc(cid)}">{esc(_title(cid, meta))}</a>{desc}</span>'
            f'<span class="m">{"".join(facts)}</span></li>')


def _folder_group(folder: str, items: list[tuple[str, dict]], user: User) -> str:
    _, colour, icon = _folder_style(folder)
    count = f'{len(items)} page{"s" * (len(items) != 1)}'
    rows = "".join(_row(c, m, user, folder=False, when=False) for c, m in items)
    return (f'<section class="folder-group fc-{colour}" id="f-{esc(folder or "general").replace("/", "-")}">'
            f'<h3><span aria-hidden="true">{icon}</span>{esc(_folder_label(folder))}<span class="count">{count}</span></h3>'
            f'<ul class="list">{rows}</ul></section>')


async def _home(user: User) -> Response:
    docs = await asyncio.to_thread(Bundle(user).all)
    first = (user.name or user.email).split()[0]
    reviewed = sum(1 for _, m, _ in docs if okf.trust_tier(m) == "human-reviewed")
    attention = [(c, m, _attention(m)) for c, m, _ in docs if _attention(m)]
    attention.sort(key=lambda a: (not a[1].get("review"), not okf.is_stale(a[1]), a[0]))   # questions first
    questions = [(c, m, q) for c, m, _ in docs for q in m.get("review") or []]
    q_rows = "".join(f'<li><a href="/c/{esc(c)}#questions"><b>{esc(q)}</b></a>'
                     f'<span class="muted"> · {esc(_title(c, m))}</span></li>' for c, m, q in questions)
    q_block = (f'<section class="block questions-home" id="questions"><h2>Questions for you <span class="count">{len(questions)}</span></h2>'
               f'<p class="lead muted">Claude couldn\'t settle these from the knowledge base. Open one to answer it on its page.</p>'
               f'<ul class="list">{q_rows}</ul></section>') if questions else ""
    last = max((_updated(m) for _, m, _ in docs), default="")

    recent = sorted(docs, key=lambda d: _updated(d[1]), reverse=True)[:8]
    recent_rows = "".join(_row(c, m, user, folder=True, when=True) for c, m, _ in recent)

    att_rows = "".join(_attention_row(c, m, reasons) for c, m, reasons in attention[:12])
    more = f'<p class="muted">…and {len(attention) - 12} more.</p>' if len(attention) > 12 else ""

    folders: dict[str, list] = {}
    for c, m, _ in _ordered(docs):
        folders.setdefault(posixpath.dirname(c), []).append((c, m))
    folder_groups = "".join(_folder_group(f, items, user) for f, items in folders.items())
    jump = "".join(_chip(f) for f in folders)

    # Portals: the root pages, one per top-level folder (like a wiki's main page).
    portal_cards = "".join(
        f'<a class="card portal" href="/c/{esc(c)}"><h3>{esc(_title(c, m))}</h3>'
        f'<p>{esc(m.get("description") or "")}</p>'
        f'<span class="count">{len([k for k, _, _ in docs if k.startswith(c + "/")])} pages</span></a>'
        for c, m, _ in sorted((d for d in docs if "/" not in d[0]), key=lambda d: _title(d[0], d[1]).lower()))
    portals = (f'<section class="block" id="portals"><h2>Portals</h2>'
               f'<p class="lead muted">Start here: each portal summarises one area and leads to its pages.</p>'
               f'<div class="cards">{portal_cards}</div></section>') if portal_cards else ""

    mcp = f"{PUBLIC_URL}/mcp"
    main = f"""
<section class="hero kb">
  <h1>{esc(first)}'s knowledge</h1>
  <p class="tagline">{len(docs)} page{"s" * (len(docs) != 1)}{f" · last change {esc(ago(last))}" if last else ""} · written with Claude, reviewed by you</p>
  <div class="actions">
    <a class="btn primary" href="#connect">Connect Claude</a>
    <a class="btn" href="/graph">Explore the graph</a>
    <a class="btn" href="/export.tar.gz?at={int(time.time())}" title="All your pages with their full git history (.tar.gz)">Download everything (git)</a>
  </div>
</section>
<div class="stats">
  <a class="stat" href="#all"><b>{len(docs)}</b><span>pages</span></a>
  <a class="stat" href="#all"><b>{reviewed}</b><span>reviewed by you</span></a>
  <a class="stat" href="#attention"><b>{len(attention)}</b><span>need your attention</span></a>
</div>
{q_block}
{portals}
<section class="block"><h2>Recently updated</h2>
  <p class="lead muted">What changed last — Claude keeps these up to date as things change.</p>
  {f'<ul class="list">{recent_rows}</ul>' if recent_rows else '<p class="empty">Nothing here yet. Connect Claude and ask it to remember something.</p>'}
</section>
<section class="block" id="attention"><h2>Needs your attention</h2>
  <p class="lead muted">Pages nobody has confirmed yet, drafts, and facts that are due for a re-check. Open one and press <b>Mark as reviewed</b> once it's right.</p>
  {f'<ul class="list">{att_rows}</ul>{more}' if att_rows else '<p class="empty">All caught up — everything is reviewed and fresh.</p>'}
</section>
<section class="block" id="all"><h2>All pages</h2>
  <p class="lead muted">Grouped by folder. Jump to:</p>
  <div class="jump">{jump}</div>
  {folder_groups or '<p class="empty">No pages yet.</p>'}
</section>
<section class="block" id="connect"><h2>Connect Claude</h2>
  <p class="lead muted">Claude reads and writes these pages for you. {'There is no sign-in: this server has one user.' if store.NO_AUTH else 'Each person who signs in gets their own private set.'}</p>
  <div class="cards">
    <div class="card"><h3>Claude (web, desktop, mobile)</h3>
      <p>Settings → Connectors → <b>Add custom connector</b>. Paste the URL below, leave the OAuth fields empty, then {'Connect.' if store.NO_AUTH else 'Connect and sign in.'}</p></div>
    <div class="card"><h3>Claude Code</h3>
      <p><code>claude mcp add --transport http kb {esc(mcp)}</code>{'' if store.NO_AUTH else ', then <code>/mcp</code> → kb → Authenticate'}.</p></div>
  </div>
  <div class="endpoint"><b>{esc(mcp)}</b> <button class="btn small" type="button" data-copy="{esc(mcp)}">Copy</button></div>
</section>"""
    return _page("Home", main, user, description=f"{first}'s knowledge base")


def _inline(text: str, known: dict) -> str:
    """Infobox values: escaped text with `code` and [label](/id.md) links (internal or http)."""
    out = esc(str(text))
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)

    def link(m: re.Match) -> str:
        label, target = m[1], m[2]
        if target.startswith(("http://", "https://")):
            return f'<a href="{target}">{label}</a>'
        cid = okf.resolve_link("", target)
        return f'<a href="/c/{cid}">{label}</a>' if cid in known else label
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, out)


def _infobox(meta: dict, known: dict) -> str:
    box = meta.get("infobox")
    if not isinstance(box, dict) or not box:
        return ""
    rows = "".join(
        f'<tr><th>{esc(str(k))}</th><td>{"<br>".join(_inline(x, known) for x in v) if isinstance(v, list) else _inline(v, known)}</td></tr>'
        for k, v in box.items())
    return f'<aside class="infobox"><table>{rows}</table></aside>'


def _navbox(cid: str, known: dict) -> str:
    """Like a wiki navbox: this page's parent hub with its pages, and this page's own children."""
    def items(parent: str) -> str:
        kids = sorted((c for c in known if posixpath.dirname(c) == parent), key=lambda c: _title(c, known[c]).lower())
        return "".join(f'<li{" class=here" if c == cid else ""}><a href="/c/{esc(c)}">{esc(_title(c, known[c]))}</a></li>'
                       for c in kids)
    boxes = []
    if mine := items(cid):
        boxes.append(f'<div class="navbox"><h2>Pages under {esc(_title(cid, known[cid]))}</h2><ul>{mine}</ul></div>')
    parent = posixpath.dirname(cid)
    if parent and parent in known:
        boxes.append(f'<div class="navbox"><h2><a href="/c/{esc(parent)}">{esc(_title(parent, known[parent]))}</a></h2>'
                     f'<ul>{items(parent)}</ul></div>')
    return "".join(boxes)


async def _concept(user: User, cid: str) -> Response:
    b = Bundle(user)
    try:
        meta, body = await asyncio.to_thread(b.read, cid)
    except okf.ConceptError:
        moved = await asyncio.to_thread(b.resolve, cid)
        if moved:   # an old name, like a wiki redirect
            return RedirectResponse(f"/c/{moved}", status_code=301)
        return _message("Page not found", f"There's no page called <code>{esc(cid)}</code>. It may have been moved or deleted.", user)
    docs = await asyncio.to_thread(b.all)
    known = {c: m for c, m, _ in docs}
    title = _title(cid, meta)
    content, outline = render(cid, body, lambda t: f"/c/{t}" if t in known else None, title)

    # Breadcrumbs through the portal and hub pages above this one.
    parts, trail = cid.split("/"), []
    for i in range(1, len(parts)):
        anc = "/".join(parts[:i])
        trail.append(f'<a href="/c/{esc(anc)}">{esc(_title(anc, known[anc]))}</a>' if anc in known
                     else _chip(anc))
    crumbs = f'<p class="crumbs"><a href="/">Knowledge</a>{"".join(f" › {t}" for t in trail)}</p>'
    lead = f'<p class="lead">{esc(meta["description"])}</p>' if meta.get("description") else ""
    content = _infobox(meta, known) + content
    navbox = _navbox(cid, known)
    gen = meta.get("generated") if isinstance(meta.get("generated"), dict) else {}
    byline = f'Updated {esc(ago(gen.get("at")))} by {esc(_actor(gen.get("by"), user))}'
    if meta.get("stale_after"):
        due = str(meta["stale_after"])[:10]
        byline += f' · {"re-check was due" if okf.is_stale(meta) else "re-check by"} {esc(due)}'

    # Where it comes from
    srcs = []
    for s in meta.get("sources") or []:
        res = str(s.get("resource", ""))
        label = esc(s.get("title") or res)
        stored = okf.cited({"sources": [s]}, known)   # a raw source in the citation layer
        link = (f'<a href="{esc(res)}">{label}</a>' if res.startswith(("http://", "https://")) else
                f'<a href="/c/{esc(stored[0])}">{label}</a>' if stored else label)
        extra = [x for x in (_actor(s["author"], user) if s.get("author") else "",
                             f'changed {ago(s["last_modified"])}' if s.get("last_modified") else "") if x]
        srcs.append(f'<li>{link}{f" <span class=muted>· {esc(" · ".join(extra))}</span>" if extra else ""}</li>')
    resource = (f'<dt>About</dt><dd><a href="{esc(meta["resource"])}">{esc(meta["resource"])}</a></dd>'
                if str(meta.get("resource", "")).startswith("http") else
                (f'<dt>About</dt><dd>{esc(meta["resource"])}</dd>' if meta.get("resource") else ""))

    # Connections
    out_links = okf.links(cid, body)
    backlinks = [c for c, _, bd in docs if c != cid and cid in okf.links(c, bd)]
    cited_by = [c for c, m, _ in docs if cid in okf.cited(m, known)]
    li = lambda c: (f'<li><a href="/c/{esc(c)}">{esc(_title(c, known[c]))}</a></li>' if c in known
                    else f'<li><span class="muted">{esc(c)} (not written yet)</span></li>')

    # Review
    verified = sorted((v for v in okf.verified_list(meta) if isinstance(v, dict)), key=lambda v: str(v.get("at", "")), reverse=True)
    checks = "".join(f'<li>{esc(_actor(v.get("by"), user)).capitalize()} · {esc(ago(v.get("at")))}</li>' for v in verified)

    order = [c for c, _, _ in _ordered(docs)]
    i = order.index(cid) if cid in order else -1
    prev = order[i - 1] if i > 0 else None
    nxt = order[i + 1] if 0 <= i < len(order) - 1 else None
    pager = ('<nav class="pager">'
             + (f'<a href="/c/{esc(prev)}"><small>Previous</small>{esc(_title(prev, known[prev]))}</a>' if prev else "<span></span>")
             + (f'<a class="next" href="/c/{esc(nxt)}"><small>Next</small>{esc(_title(nxt, known[nxt]))}</a>' if nxt else "<span></span>")
             + "</nav>")

    main = f"""{crumbs}<h1>{esc(title)}</h1>{lead}
{_pills(meta, user)}
<p class="byline">{byline}</p>
{_questions_box(cid, meta)}
{content}
{navbox}
<div class="about">
  <div class="panel"><h2>Where this comes from</h2>
    {f"<dl>{resource}</dl>" if resource else ""}
    {f"<ul>{''.join(srcs)}</ul>" if srcs else '<p class="muted">No sources recorded.</p>'}
  </div>
  <div class="panel"><h2>Connections</h2>
    <dl><dt>Links to</dt><dd>{f"<ul>{''.join(li(c) for c in out_links)}</ul>" if out_links else '<span class="muted">nothing</span>'}</dd>
        <dt>Linked from</dt><dd>{f"<ul>{''.join(li(c) for c in backlinks)}</ul>" if backlinks else '<span class="muted">nothing yet</span>'}</dd>
        {f"<dt>Cited by</dt><dd><ul>{''.join(li(c) for c in cited_by)}</ul></dd>" if cited_by else ""}</dl>
  </div>
  <div class="panel"><h2>Review</h2>
    {f"<ul>{checks}</ul>" if checks else '<p class="muted">Nobody has checked this page yet.</p>'}
    <div class="actions-row" style="margin-top:14px">
      <form method="post" action="/c/{esc(cid)}/verify"><button class="btn primary" title="Records that you confirmed this page is correct">Mark as reviewed</button></form>
    </div>
  </div>
</div>
{pager}
<p class="edit"><span><a href="/h/{esc(cid)}">History &amp; changes</a> ·<a href="/c/{esc(cid)}.md">Raw markdown</a></span>
<span><code>{esc(cid)}</code></span></p>"""
    return _page(title, main, user, sidebar=_sidebar(docs, cid), toc=_toc(outline),
                 description=str(meta.get("description") or title))


# ── history and diffs (owner only) ──────────────────────────────────────────────

BERLIN = ZoneInfo("Europe/Berlin")


def _when(iso: str) -> str:
    """Commit time as Berlin date and time, e.g. '2026-10-22 20:25'."""
    return datetime.fromisoformat(iso).astimezone(BERLIN).strftime("%Y-%m-%d %H:%M")


def _diff(old: str, new: str) -> str:
    """Line diff of two markdown texts (frontmatter included) as HTML."""
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=3))[2:]
    if not lines:
        return '<p class="muted">No differences: this version is the same as the current one.</p>'
    added = sum(1 for l in lines if l.startswith("+"))
    removed = sum(1 for l in lines if l.startswith("-"))
    out = []
    for l in lines:
        if l.startswith("@@"):
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", l)
            out.append(f'<span class="hunk">around line {m[1] if m else "?"}</span>')
        else:
            cls = {"+": "add", "-": "del"}.get(l[:1], "ctx")
            out.append(f'<span class="{cls}">{esc(l) or " "}</span>')
    return (f'<p class="diffstat"><span class="add">+{added}</span> <span class="del">−{removed}</span> lines</p>'
            f'<pre class="diff">{"".join(out)}</pre>')


async def _history(user: User, cid: str, rev: str | None) -> Response:
    b = Bundle(user)
    rows = await asyncio.to_thread(b.changes, cid, 100)
    if not rows:
        return _message("No history", f"No changes recorded for <code>{esc(cid)}</code>.", user)
    exists = b.exists(cid)
    title = _title(cid, (await asyncio.to_thread(b.read, cid))[0]) if exists else cid
    # Older versions to compare with now: every revision of this page except the current one and
    # deletions. Only revisions from this page's own history are accepted.
    older = [r for r in rows[1 if exists else 0:] if r["files"] and r["files"][0][0] != "D"]
    pick = next((r for r in older if r["revision"] == rev), None) or (older[0] if older else None)
    if pick:
        try:
            old = await asyncio.to_thread(b.text_at, cid, pick["revision"], pick["path"])
            new = await asyncio.to_thread(b.path(cid).read_text, "utf-8") if exists else ""
            changes = _diff(old, new)
        except (okf.ConceptError, OSError):
            changes = '<p class="muted">That version can\'t be read.</p>'
        options = "".join(f'<option value="{esc(r["revision"])}"{" selected" if r is pick else ""}>'
                          f'{esc(_when(r["at"]))} · {esc(r["message"])} · {esc(r["revision"])}</option>' for r in older)
        compare = (f'<section id="changes"><h2>What changed since an older version</h2>'
                   f'<form class="compare" method="get" action="/h/{esc(cid)}#changes">'
                   f'<label for="from">Compare the {"current version" if exists else "deleted page (now empty)"} with</label>'
                   f'<select id="from" name="from" data-autosubmit>{options}</select>'
                   f'<button class="btn small">Show changes</button></form>{changes}</section>')
    else:
        compare = '<p class="muted">There is only one version of this page so far, so there is nothing to compare.</p>'
    current = rows[0]["revision"] if exists else None
    items = "".join(
        f'<li><span class="grow">{esc(r["message"])}<small>{esc(_when(r["at"]))} · {esc(ago(r["at"]))}</small></span>'
        f'<code>{esc(r["revision"])}</code>'
        + ('<span class="muted">current</span>' if r["revision"] == current else
           f'<a class="btn small" href="/h/{esc(cid)}?from={esc(r["revision"])}#changes">Compare with now</a>'
           if r in older else "")
        + "</li>" for r in rows)
    main = (f'<p class="crumbs"><a href="/">Knowledge</a> › <a href="/c/{esc(cid)}">{esc(title)}</a></p>'
            f'<h1>History &amp; changes</h1><p class="lead">Every change to this page is kept in your own git repository. '
            f'Ask Claude to restore an older version with <code>kb_restore</code> and the revision id.</p>'
            f'{compare}<h2>All versions</h2><ul class="rows">{items}</ul>')
    return _page(f"History · {title}", main, user)


async def _bundle_history(user: User) -> Response:
    rows = await asyncio.to_thread(Bundle(user).changes, None, 150)

    def page(status: str, path: str, rev: str) -> str:
        if not path.endswith(".md") or posixpath.basename(path)[:-3] in okf.RESERVED:
            return ""
        cid = path[:-3]
        if status == "D":
            return f'<a class="chip" href="/h/{esc(cid)}">{esc(cid)} <em>(deleted)</em></a>'
        return f'<a class="chip" href="/h/{esc(cid)}?from={esc(rev)}#changes" title="What changed since this version">{esc(cid)}</a>'

    items = "".join(
        f'<li><span class="grow">{esc(r["message"])}<small>{esc(_when(r["at"]))} · {esc(ago(r["at"]))}</small>'
        f'<span class="chips">{"".join(page(s, p, r["revision"]) for s, p in r["files"])}</span></span>'
        f'<code>{esc(r["revision"])}</code></li>' for r in rows)
    main = (f'<p class="crumbs"><a href="/">Knowledge</a></p><h1>History</h1>'
            f'<p class="lead">The latest {len(rows)} changes to your knowledge, newest first. Your pages have their own '
            f'git repository, separate from everyone else\'s. Pick a page to see what changed in it since that version.</p>'
            f'<ul class="rows history">{items}</ul>')
    return _page("History", main, user)


def _search_hits(docs, terms: list[str], tag: str | None) -> list[tuple[int, str, dict, str]]:
    hits = []
    for cid, meta, body in docs:
        if tag and tag.lower() not in [t.lower() for t in meta.get("tags", [])]:
            continue
        title, desc, tags = _title(cid, meta).lower(), str(meta.get("description", "")).lower(), " ".join(meta.get("tags", [])).lower()
        low = body.lower()
        score = 1 if not terms else 0
        for t in terms:
            if t not in title and t not in desc and t not in tags and t not in low and t not in cid:
                score = 0
                break
            score += 6 * (t in title) + 4 * (t in tags) + 3 * (t in desc) + min(low.count(t), 5)
        if score:
            hits.append((score, cid, meta, body))
    return sorted(hits, key=lambda h: (-h[0], h[1]))


async def _search(request: Request, user: User) -> Response:
    q = (request.query_params.get("q") or "").strip()
    tag = request.query_params.get("tag")
    docs = await asyncio.to_thread(Bundle(user).all)
    terms = [search.stem(t) for t in re.findall(r"\w+", q.lower()) if len(t) > 1]   # backups finds backup
    hits = _search_hits(docs, terms, tag)
    what = f"#{tag}" if tag else f"“{q}”"
    items = "".join(f'<li><span class="grow"><a href="/c/{esc(c)}">{esc(_title(c, m))}</a>'
                    f'<small>{esc(m.get("description") or plain(b)[:180])}</small></span>{_pills(m, user, tags=False)}</li>'
                    for _, c, m, b in hits[:50])
    main = (f'<section class="block"><h1>Search</h1><p class="lead muted">{len(hits)} page{"s" * (len(hits) != 1)} for {esc(what)}</p>'
            f'{f"<ul class=rows>{items}</ul>" if items else "<p class=empty>Nothing matches. Try fewer or different words.</p>"}</section>')
    if q and not tag:
        main += await _meaning_results(user, docs, q, {c for _, c, _, _ in hits[:50]})
    return _page(f"Search {what}", main, user, q=q)


async def _meaning_results(user: User, docs, q: str, shown: set[str]) -> str:
    """Pages whose sections are about the question by meaning (embeddings, like kb_context's starting
    pages), linked to the section that matched; pages already in the keyword results are left out."""
    by_id = {c: (m, b) for c, m, b in docs}
    ranked = await asyncio.to_thread(embed.rank, user.dir, docs, q, 15)
    rows = []
    for c, sim, heading in ranked:
        if sim < SEED_MIN or c in shown or c not in by_id:
            continue
        meta, body = by_id[c]
        anchor = next((slug for _, slug, text in render(c, body, lambda t: None, _title(c, meta))[1] if text in (heading, plain(heading).strip())), "") if heading else ""
        where = f' · section <a href="/c/{esc(c)}#{esc(anchor)}">{esc(heading)}</a>' if anchor else (f" · section {esc(heading)}" if heading else "")
        rows.append(f'<li><span class="grow"><a href="/c/{esc(c)}{f"#{esc(anchor)}" if anchor else ""}">{esc(_title(c, meta))}</a>'
                    f'<small>{esc(meta.get("description") or plain(body)[:180])}{where}</small></span>'
                    f'<span class="pill plain" title="similarity of the best-matching section">{sim:.2f}</span></li>')
    if rows:
        listing = f'<ul class="rows">{"".join(rows[:10])}</ul>'
    elif embed.building():
        listing = '<p class="empty">The meaning index is being built; only keyword results for now. Try again in a few minutes.</p>'
    else:
        listing = '<p class="empty">Nothing else is close in meaning.</p>'
    return (f'<section class="block"><h2>By meaning</h2><p class="lead muted">Pages about the same thing in other words, '
            f'linked to the section that matched.</p>{listing}</section>')


async def _search_index(user: User) -> Response:
    """Per-section entries for the live search in the top bar ."""
    docs = await asyncio.to_thread(Bundle(user).all)
    entries = []
    for cid, meta, body in docs:
        title = _title(cid, meta)
        html, _ = render(cid, body, lambda t: None, title)
        cuts = list(re.finditer(r'<h([23]) id="([^"]+)">(.*?)</h\1>', html, re.S))
        intro = re.sub(r"<[^>]+>", " ", html[:cuts[0].start()] if cuts else html)
        entries.append({"p": f"c/{cid}", "t": title, "h": "", "a": "",
                        "x": re.sub(r"\s+", " ", f'{meta.get("description", "")} {" ".join(meta.get("tags", []))} {intro}').strip()[:700]})
        for i, m in enumerate(cuts):
            end = cuts[i + 1].start() if i + 1 < len(cuts) else len(html)
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html[m.end():end])).strip()
            heading = re.sub(r'<a class="anchor".*?</a>|<[^>]+>', "", m[3]).strip()
            entries.append({"p": f"c/{cid}", "t": title, "h": heading, "a": m[2], "x": text[:700]})
    return JSONResponse(entries, headers={"Cache-Control": "no-store"})


async def _verify(user: User, cid: str) -> Response:
    b = Bundle(user)
    async with store.lock(user):
        try:
            meta, body = await asyncio.to_thread(b.read, cid)
        except okf.ConceptError:
            return PlainTextResponse("not found", 404)
        entries = [v for v in okf.verified_list(meta) if v.get("by") != user.actor]
        meta["verified"] = [*entries, {"by": user.actor, "at": okf.now_iso()}]
        await asyncio.to_thread(b.save, cid, meta, body,
                                f"Verification: [{meta.get('title') or cid}](/{cid}.md) by {user.actor}")
    return RedirectResponse(f"/c/{cid}", status_code=303)


async def _answer(request: Request, user: User, cid: str) -> Response:
    """Answer one of a page's review questions. The answer is kept in `answered` until Claude
    (the weekly lint, or kb_review in any session) has worked it into the page and cleared it."""
    form = await request.form()
    question, answer = str(form.get("question", "")).strip(), str(form.get("answer", "")).strip()[:2000]
    if not answer:
        return _message("Nothing to save", "Type an answer first.", user, back=f"/c/{cid}")
    if guard.secrets(answer):
        return _message("Not saved", "That answer looks like it contains a secret. Name the key and where it lives "
                        "instead of its value.", user, back=f"/c/{cid}")
    b = Bundle(user)
    async with store.lock(user):
        try:
            meta, body = await asyncio.to_thread(b.read, cid)
        except okf.ConceptError:
            return PlainTextResponse("not found", 404)
        if question not in (meta.get("review") or []):
            return RedirectResponse(f"/c/{cid}", status_code=303)   # already answered (e.g. a second tab)
        meta["review"] = [q for q in meta["review"] if q != question]
        meta["answered"] = [*(meta.get("answered") or []),
                            {"question": question, "answer": answer, "by": user.actor, "at": okf.now_iso()}]
        if not meta["review"]:
            del meta["review"]
        await asyncio.to_thread(b.save, cid, meta, body,
                                f"Update: answered a question on [{meta.get('title') or cid}](/{cid}.md)")
    return RedirectResponse(f"/c/{cid}#questions", status_code=303)


def _questions_box(cid: str, meta: dict) -> str:
    """The page's open questions for the user, each with an answer form, and answers waiting for Claude."""
    open_q = meta.get("review") or []
    waiting = [a for a in meta.get("answered") or [] if isinstance(a, dict)]
    if not open_q and not waiting:
        return ""
    forms = "".join(
        f'<form class="question" method="post" action="/c/{esc(cid)}/answer">'
        f'<p><b>{esc(q)}</b></p><input type="hidden" name="question" value="{esc(q)}">'
        f'<textarea name="answer" rows="2" maxlength="2000" placeholder="Your answer" required></textarea>'
        f'<button class="btn small primary">Save answer</button></form>' for q in open_q)
    done = "".join(f'<li><b>{esc(a.get("question", ""))}</b><br>{esc(a.get("answer", ""))} '
                   f'<span class="muted">· {esc(ago(a.get("at")))}, waiting for Claude to update the page</span></li>'
                   for a in waiting)
    return (f'<section class="questions" id="questions"><h2>{"Questions for you" if open_q else "Your answers"}</h2>'
            + (f'<p class="muted">Claude couldn\'t settle these from the knowledge base. Your answer is worked into '
               f'the page on the next lint run (Mondays 09:00) or the next time Claude reads its review queue.</p>{forms}' if open_q else "")
            + (f'<ul class="answered">{done}</ul>' if done else "") + "</section>")


async def _graph_json(user: User) -> Response:
    docs = await asyncio.to_thread(Bundle(user).all)
    nodes, edges = graph.build(docs)
    return JSONResponse({"nodes": list(nodes.values()),
                         "edges": [{"source": a, "target": b, **e} for (a, b), e in edges.items()]},
                        headers={"Cache-Control": "no-store"})


async def _pairs_data(user: User) -> tuple[dict | None, dict]:
    b = Bundle(user)
    docs = await asyncio.to_thread(b.all)
    return await asyncio.to_thread(pairs.analyse, user.dir, b.root, docs), {c: m for c, m, _ in docs}


async def _pairs_json(user: User) -> Response:
    pa, known = await _pairs_data(user)
    if pa is None:
        return JSONResponse({"building": True})
    ix = {c: i for i, c in enumerate(pa["ids"])}
    return JSONResponse({
        "ids": pa["ids"], "titles": [_title(c, known[c]) for c in pa["ids"]],
        "cos": [[round(float(v), 3) for v in row] for row in pa["matrix"]],
        "linked": [[ix[a], ix[b]] for a, b in (k.split("|") for k in pa["linked"])],
        "candidates": [[ix[p["a"]], ix[p["b"]]] for p in pa["candidates"]],
        "dismissed": [[ix[p["a"]], ix[p["b"]]] for p in pa["dismissed"]]})


async def _pairs_page(user: User) -> Response:
    pa, known = await _pairs_data(user)
    if pa is None:
        return _message("Pairs", "Embeddings are unavailable or still being built. Try again in a minute.", user)
    m = pa["match"]
    pct = lambda v: "–" if v is None else f"{v:.0%}"
    link = lambda c: f'<a href="/c/{esc(c)}">{esc(_title(c, known[c]))}</a>'
    def rows(items, extra):
        return "".join(f'<tr><td class="num">{p["similarity"]:.2f}</td><td>{link(p["a"])}</td><td>{link(p["b"])}</td>'
                       f'<td>{extra(p)}</td></tr>' for p in items) or \
            '<tr><td colspan="4" class="muted">None.</td></tr>'
    cand = rows(pa["candidates"], lambda p: esc(f'was dismissed: {p["recheck"]["reason"]}') if p.get("recheck") else "")
    dism = rows(pa["dismissed"], lambda p: esc(p.get("reason") or ""))
    head = '<thead><tr><th class="num">cos</th><th>Page</th><th>Page</th><th>Note</th></tr></thead>'
    main = f"""<section class="block graph-page pairs-page"><h1>Pairs</h1>
<p class="lead muted">How closely pages are related by meaning, and whether they link each other. Text is compared
without footnotes and link targets, so pages don't look alike just because they cite or link the same things.</p>
<div class="pair-stats">
  <div class="panel"><span class="muted">Nearest page already linked</span><b>{pct(m["nearest_linked"])}</b></div>
  <div class="panel"><span class="muted">Of the {pairs.K} nearest, linked</span><b>{pct(m[f"top{pairs.K}_linked"])}</b></div>
  <div class="panel"><span class="muted">A random pair is linked</span><b>{pct(m["random_pair_linked"])}</b></div>
  <div class="panel"><span class="muted">Open pairs / dismissed</span><b>{len(pa["candidates"])} / {len(pa["dismissed"])}</b></div>
</div>
<h2>Every page against every page</h2>
<p class="muted">Sorted by path, so each area is a block on the diagonal. Brighter is closer. Ringed: linked.
Amber square: open pair (close, not linked). Cross: dismissed (no link needed). Hover a cell for the pages; click to open one.</p>
<div class="pairs-wrap"><canvas id="pairs-map" role="img" aria-label="Heatmap of meaning similarity between all pages"></canvas>
<div id="pairs-tip" class="pairs-tip" hidden></div></div>
<h2>Open pairs</h2>
<p class="muted">Among each other's {pairs.K} nearest pages by meaning, not linked either way, not parent and child.
The weekly lint links them or records why no link is needed.</p>
<div class="table-wrap"><table class="pairs">{head}<tbody>{cand}</tbody></table></div>
<h2>Dismissed</h2>
<p class="muted">Judged as needing no link. A pair comes back when either page changes.</p>
<div class="table-wrap"><table class="pairs">{head}<tbody>{dism}</tbody></table></div>
</section>"""
    return _page("Pairs", main, user, scripts=f'<script src="/static/pairs.js?v={ASSET_VERSION}" defer></script>')


VENDOR = ("graphology.umd.min.js", "graphology-library.min.js", "sigma.min.js")


def _graph_page(user: User) -> Response:
    main = """<section class="block graph-page"><h1>Graph</h1>
<p class="lead muted"><span id="count">Loading…</span>. Bigger dots have more connections; thicker lines are
stronger relations (links, parent and child pages, shared sources, common neighbours).</p>
<div class="graph-tools">
  <input id="graph-search" type="search" list="graph-titles" placeholder="Find a page…" autocomplete="off">
  <datalist id="graph-titles"></datalist>
  <label>Colour by <select id="graph-mode"><option value="folder">Area</option>
    <option value="community">Cluster</option><option value="type">Type</option></select></label>
  <label class="check"><input id="graph-labels" type="checkbox"> All labels</label>
  <span class="zoom-btns"><button class="btn small" id="graph-in" type="button" title="Zoom in">+</button>
  <button class="btn small" id="graph-out" type="button" title="Zoom out">−</button>
  <button class="btn small" id="graph-fit" type="button" title="Show everything">Fit</button></span>
</div>
<div class="graph-wrap">
  <div id="graph"></div>
  <aside class="graph-side">
    <div class="panel" id="graph-info"></div>
    <div class="panel"><h2>Legend</h2><ul class="graph-legend" id="graph-legend"></ul></div>
    <div class="panel graph-insights"><h2>Insights</h2><div id="graph-insights"></div></div>
  </aside>
</div></section>"""
    scripts = "".join(f'<script src="/static/vendor/{f}?v={ASSET_VERSION}" defer></script>' for f in VENDOR)
    scripts += f'<script src="/static/graph.js?v={ASSET_VERSION}" defer></script>'
    return _page("Graph", main, user, scripts=scripts)


class WebGate:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        response = await self._route(Request(scope, receive))
        if response is None:
            return await self.app(scope, receive, send)
        # Nothing personal may be stored by Cloudflare or any other shared cache. Cloudflare caches
        # by file extension (.gz, .tar, …) and ignores the session cookie, so without this a signed-in
        # export was served to anyone (2026-10-22). Only the public static assets may be cached.
        if not scope["path"].startswith("/static/"):
            response.headers["Cache-Control"] = "private, no-store"
        await response(scope, receive, send)

    async def _route(self, request: Request) -> Response | None:
        path, method = request.url.path, request.method
        if path == "/healthz":
            return PlainTextResponse("ok")
        if path == CALLBACK:
            return await _callback(request)
        if path in ("/static/style.css", "/static/kb.js", "/static/graph.js", "/static/pairs.js",
                    "/static/icon-32.png", "/static/icon-180.png"):
            return FileResponse(STATIC / path.rsplit("/", 1)[1], headers={"Cache-Control": "max-age=31536000, immutable"})
        if path.startswith("/static/vendor/") and path.rsplit("/", 1)[1] in VENDOR:
            return FileResponse(STATIC / "vendor" / path.rsplit("/", 1)[1], headers={"Cache-Control": "max-age=31536000, immutable"})
        browser_paths = path in ("/", "/graph", "/graph.json", "/pairs", "/pairs.json", "/history", "/search", "/search.json",
                                 "/export.tar.gz", "/logout") \
            or path.startswith(("/c/", "/h/"))
        if path == "/mcp" and method == "GET" and "authorization" not in request.headers \
                and "text/html" in request.headers.get("accept", ""):
            return RedirectResponse("/#connect", status_code=302)
        if not browser_paths:
            return None   # MCP + OAuth endpoints

        if path == "/logout":
            if store.NO_AUTH:
                return RedirectResponse("/", status_code=302)
            resp = RedirectResponse((await _config()).get("end_session_endpoint") or "/", status_code=302)
            resp.delete_cookie(SESSION_COOKIE, path="/")
            return resp
        user = await store.local_user() if store.NO_AUTH else _session_user(request)
        if user is None:
            return await _login(path + (f"?{request.url.query}" if request.url.query else "")) if method == "GET" \
                else PlainTextResponse("signed out — reload the page", 401)
        if method == "POST":
            if not _csrf_ok(request):
                return PlainTextResponse("cross-site request refused", 403)
            if path.startswith("/c/") and path.endswith("/verify"):
                try:
                    return await _verify(user, okf.normalize_id(path[3:-len("/verify")]))
                except okf.ConceptError:
                    return PlainTextResponse("not found", 404)
            if path.startswith("/c/") and path.endswith("/answer"):
                try:
                    return await _answer(request, user, okf.normalize_id(path[3:-len("/answer")]))
                except okf.ConceptError:
                    return PlainTextResponse("not found", 404)
            return PlainTextResponse("not found", 404)
        if path == "/":
            return await _home(user)
        if path == "/search":
            return await _search(request, user)
        if path == "/search.json":
            return await _search_index(user)
        if path == "/graph":
            return _graph_page(user)
        if path == "/graph.json":
            return await _graph_json(user)
        if path == "/pairs":
            return await _pairs_page(user)
        if path == "/pairs.json":
            return await _pairs_json(user)
        if path == "/export.tar.gz":
            async with store.lock(user):   # a consistent snapshot of the files and .git
                data = await asyncio.to_thread(Bundle(user).export_tar)
            return Response(data, media_type="application/gzip",
                            headers={"Content-Disposition": f'attachment; filename="{quote(user.dir)}-knowledge-git.tar.gz"'})
        if path == "/history":
            return await _bundle_history(user)
        if path.startswith("/h/"):
            try:
                return await _history(user, okf.normalize_id(path[3:]), request.query_params.get("from"))
            except okf.ConceptError:
                return PlainTextResponse("not found", 404)
        if path.startswith("/c/"):
            raw = path.endswith(".md")
            try:
                cid = okf.normalize_id(path[3:])
            except okf.ConceptError:
                return PlainTextResponse("not found", 404)
            if raw:
                try:
                    text = await asyncio.to_thread(Bundle(user).path(cid).read_text, "utf-8")
                except OSError:
                    return PlainTextResponse("not found", 404)
                return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")
            return await _concept(user, cid)
        return PlainTextResponse("not found", 404)
