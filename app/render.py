"""Markdown -> safe HTML for the web view and shared pages.

Raw HTML in markdown is not rendered (html=False), and the output goes through
nh3 (an allow-list sanitizer) as well, because shared pages show content to
people other than its author. Links between concepts are rewritten to the web
view: inside a share they stay within the share or turn into plain text.

The page title is the <h1>, so body headings move down one level (# -> h2), a
leading heading that just repeats the title is dropped, and h2/h3 get ids,
a "#" link each, and an entry in the "On this page" outline.
"""

import html
import posixpath
import re
from datetime import datetime, timezone
from typing import Callable

import nh3
from markdown_it import MarkdownIt
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.tasklists import tasklists_plugin

from . import okf

_md = (MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False})
       .enable(["table", "strikethrough"]).use(footnote_plugin).use(tasklists_plugin))

_TAGS = {"a", "p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "strong", "em", "del", "s", "code", "pre",
         "blockquote", "ul", "ol", "li", "table", "thead", "tbody", "tr", "th", "td", "img", "sup", "sub",
         "section", "input", "span", "div"}
_ATTRS = {"a": {"href", "title", "id", "class", "aria-label"}, "img": {"src", "alt", "title"}, "th": {"style"},
          "td": {"style"}, "li": {"id", "class"}, "sup": {"id", "class"}, "section": {"class"},
          "input": {"type", "checked", "disabled", "class"}, "ol": {"start", "class"}, "ul": {"class"},
          "code": {"class"}, "span": {"class"}, "div": {"class"},
          "h2": {"id"}, "h3": {"id"}, "h4": {"id"}}
_HEADING = re.compile(r'<h([23]) id="([^"]+)">(.*?)</h\1>', re.S)


def _slug(text: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"
    slug, n = base, 2
    while slug in taken:
        slug, n = f"{base}-{n}", n + 1
    taken.add(slug)
    return slug


def render(concept_id: str, body: str, link_for: Callable[[str], str | None],
           title: str | None = None) -> tuple[str, list[tuple[int, str, str]]]:
    """(html, outline). link_for(concept_id) gives the URL for a linked concept, or None
    to show the link text without a link (e.g. outside a share). Outline entries are
    (level, id, text) for h2/h3."""
    base = posixpath.dirname(concept_id)
    tokens = _md.parse(body)

    # Drop a leading heading that only repeats the page title.
    if (title and len(tokens) >= 3 and tokens[0].type == "heading_open"
            and tokens[1].content.strip().lower() == title.strip().lower()):
        tokens = tokens[3:]

    taken: set[str] = set()
    outline = []
    for i, tok in enumerate(tokens):
        if tok.type in ("heading_open", "heading_close"):
            level = min(int(tok.tag[1]) + 1, 6)
            tok.tag = f"h{level}"
            if tok.type == "heading_open" and level in (2, 3):
                text = plain(tokens[i + 1].content)
                slug = _slug(text, taken)
                tok.attrSet("id", slug)
                outline.append((level, slug, text))
        for child in tok.children or []:
            if child.type == "link_open":
                href = child.attrGet("href") or ""
                target = okf.resolve_link(base, href)
                if target is not None:
                    url = link_for(target)
                    if url is None:
                        child.attrSet("href", "#")
                        child.attrSet("class", "unavailable")
                        child.attrSet("title", "not part of what's shared")
                    else:
                        child.attrSet("href", url)
                elif href.endswith("/") and not href.startswith(("http:", "https:")):
                    child.attrSet("href", "#")
    rendered = _md.renderer.render(tokens, _md.options, {})
    safe = nh3.clean(rendered, tags=_TAGS, attributes=_ATTRS, url_schemes={"http", "https", "mailto"},
                     link_rel="noopener noreferrer nofollow")
    # A "#" link on every section heading, so any section can be shared.
    safe = _HEADING.sub(lambda m: f'<h{m[1]} id="{m[2]}">{m[3]}<a class="anchor" href="#{m[2]}" '
                                  f'aria-label="Link to this section">#</a></h{m[1]}>', safe)
    return safe, outline


def to_html(concept_id: str, body: str, link_for: Callable[[str], str | None]) -> str:
    return render(concept_id, body, link_for)[0]


def plain(markdown: str) -> str:
    """Rough plain text of a markdown fragment, for search snippets."""
    text = re.sub(r"```.*?```", " ", markdown, flags=re.S)
    text = re.sub(r"\[\^[^\]]+\]:?", "", text)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_>#|]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ago(iso: str | None) -> str:
    """'3 hours ago', 'yesterday', '12 Oct 2026' for an ISO timestamp."""
    if not iso:
        return "at an unknown time"
    try:
        then = okf.parse_time(iso)
    except okf.ConceptError:
        return str(iso)
    secs = (datetime.now(timezone.utc) - then).total_seconds()
    if secs < 0:
        return then.strftime("%-d %b %Y")
    for limit, div, unit in ((60, 1, None), (3600, 60, "minute"), (86400, 3600, "hour")):
        if secs < limit:
            if unit is None:
                return "just now"
            n = int(secs // div)
            return f"{n} {unit}{'s' * (n != 1)} ago"
    days = int(secs // 86400)
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    return then.strftime("%-d %b %Y")


def esc(value) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)
