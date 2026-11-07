"""Knowledge base MCP server: one OKF bundle per user.

Auth: OIDCProxy runs the MCP OAuth flow (dynamic client registration, PKCE,
token exchange) in front of an OIDC provider; the caller's
subject + email pick their bundle. The browser side (web.py) uses the same client.

Environment:
  PUBLIC_URL                 where this server is reached, e.g. https://kb.example.com
  OIDC_CONFIG_URL            the provider's .well-known/openid-configuration
  OIDC_CLIENT_ID / _SECRET   the OAuth2 client's credentials
  JWT_SIGNING_KEY            signs the tokens this server hands to MCP clients (and web cookies)
  STORAGE_ENCRYPTION_KEY     Fernet key encrypting stored client registrations/tokens
"""

import logging
import os

import uvicorn
from cryptography.fernet import Fernet
from fastmcp import FastMCP
from fastmcp.server.auth.oidc_proxy import OIDCProxy
from key_value.aio.stores.disk import DiskStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper
from starlette.middleware import Middleware

from . import embed, store, tools, web

# The kb's own progress messages ("embedded 549 sections for <user>") at INFO; libraries stay at WARNING.
logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s")
logging.getLogger("kb").setLevel(logging.INFO)


class OfflineAccessProxy(OIDCProxy):
    """Some OIDC providers only issue refresh tokens for offline_access, which MCP clients don't
    ask for. Without one, Claude's token dies after an hour and the connector drops, so
    ask for it upstream at login and on every refresh."""

    def _prepare_scopes_for_upstream_refresh(self, scopes: list[str]) -> list[str]:
        return scopes if "offline_access" in scopes else [*scopes, "offline_access"]


def build_auth() -> OfflineAccessProxy:
    return OfflineAccessProxy(
        config_url=os.environ["OIDC_CONFIG_URL"],
        client_id=os.environ["OIDC_CLIENT_ID"],
        client_secret=os.environ["OIDC_CLIENT_SECRET"],
        base_url=os.environ["PUBLIC_URL"],
        required_scopes=["openid", "profile", "email"],
        extra_authorize_params={"scope": "openid profile email offline_access"},
        jwt_signing_key=os.environ["JWT_SIGNING_KEY"],
        client_storage=FernetEncryptionWrapper(
            key_value=DiskStore(directory="/data/oauth"),
            fernet=Fernet(os.environ["STORAGE_ENCRYPTION_KEY"]),
        ),
    )


INSTRUCTIONS = """\
The user's personal knowledge base: a wiki in Open Knowledge Format (OKF v0.2),
markdown pages with YAML frontmatter, one private bundle per user, maintained by Claude. It ALWAYS
holds the latest facts. Keeping it current is part of finishing any
task, not a separate request.

READ IT FIRST: before answering questions about the user's systems, servers, homelab, projects,
repos, configuration, accounts, people or plans, look here.
- A question: START with kb_context(question): it matches by keywords and meaning and returns numbered
  pages; cite them as [1], [2]. Raise budget_chars only if the answer isn't covered. Use kb_search
  for an exact value, identifier or name, not instead of kb_context.
- Navigating, progressively (don't load pages you don't need): kb_list() (overview + top level) →
  kb_list(folder) (one level) → kb_read(hub) (lead, infobox, outline, children) → kb_read(id,
  section=…). kb_search(query) finds a specific fact by keywords and names its section;
  kb_search(similar_to=page or text) finds what is about the same thing by meaning. Long pages come back as lead + outline; pass full=true
  only when you need all of it. Old page IDs still resolve.

WRITE WITHOUT BEING ASKED, in the same session, as soon as a change is done and confirmed:
- a config, version, schedule, IP, port, hostname, account, access rule or dependency changed;
- something was added, removed, moved, fixed or decided (and why); a problem was diagnosed
  (symptom, cause, fix); the user stated a fact, a preference, or said "remember"/"note this";
- a useful comparison or conclusion was worked out from several pages (file it back).
Not for: secret values (only key names and where they live), throwaway chat, unconfirmed claims
(ask the user, or file a `review` question).

HOW TO WRITE:
- Find every page the change touches: kb_search for the new topic AND the old value (old version,
  schedule, IP). Edit in place with kb_update (body, description, tags, infobox) so no page states
  the old value. Never append a contradicting note; git keeps history (kb_history; with revision=… it shows the diff).
- New things: kb_write. Read meta/maintaining-this-kb once before creating pages. The server
  enforces the structure and says why when it refuses a write; fix the write, don't work around it:
  the root holds only portals; a page sits under its portal or a hub (a page with the same ID as
  its folder: projects/garden is the hub of projects/garden/irrigation), and the hub
  comes first; every page opens with a 1–3 sentence lead; hubs carry an `infobox` of key facts and
  summarise their children; title, one-sentence description and tags are required, titles unique;
  past ~9000 characters split into child pages (over 12000 is refused).
- Ideas that recur across systems → concepts/; comparisons and syntheses → analyses/ (type
  Comparison/Synthesis); obsolete pages → status 'deprecated' (say what replaced them) and
  kb_move to archive/ (old IDs become aliases).
- Sources: cite in `sources` (URL, path, or "conversation with the user on <date>") with footnotes.
  A document worth keeping (release notes, spec, article, the user's pasted text): kb_ingest it
  first; it is stored unchanged in sources/ (the citation layer, left out of search unless asked)
  and returns the pages it touches. Then cite '/sources/<id>.md'.
- No journal or day-log pages: the facts go on the topic pages. A day's change log is raw material:
  kb_ingest it as a source (title '<system> change log <date>') and cite it.
- Questions only the user can answer go in a page's `review` field (shown on the kb home page).
  kb_review lists the user's answers under queue.answered_by_user: work each into its page, cite
  {resource: "answer from the user on <date> (kb web)", author: "human:<user>"}, then
  kb_update(id, remove_fields=["answered"]).
- Pass your model as `author` (e.g. claude/opus-5.5). Never mark anything human-verified unless
  the user explicitly confirmed it. After reorganising, run
  kb_review and fix what its lint part reports.

FINISH: tell the user in one line which kb pages you created or updated."""

mcp = FastMCP(name="kb", instructions=INSTRUCTIONS, auth=build_auth())
tools.register(mcp)


# Prompts: the same rules on demand, for clients that show them as commands
# (Claude Code: /mcp__<connector>__record_session). The instructions above make Claude do this
# unprompted; these are for when the user wants it done now, explicitly.
@mcp.prompt(name="record_session",
            description="Record in the knowledge base everything this conversation changed, decided, fixed or learned")
def record_session() -> str:
    return ("Go through this conversation and record in the knowledge base everything that changed, was decided, "
            "fixed, diagnosed or learned about my systems, projects, configuration, accounts, people or plans, "
            "following the knowledge base's instructions: kb_search for each topic and for the old values, edit the "
            "affected pages in place with kb_update, create pages under the right hub only for genuinely new things, "
            "and file anything you are "
            "unsure of as a `review` question instead of guessing. No secret values. Finish with kb_review, then list "
            "the pages you created or updated.")


@mcp.prompt(name="ingest_source",
            description="Ingest a document (article, release notes, spec, pasted text) into the knowledge base")
def ingest_source(what: str) -> str:
    return (f"Ingest this into the knowledge base: {what}\n\n"
            "1. Get its full text (read the file or page; if it is pasted, use it as given) and store it with "
            "kb_ingest (title, content, resource = where it came from).\n"
            "2. Analyse it against the pages kb_ingest returns: read their matching sections and list the key "
            "facts, what is new, what agrees and what contradicts.\n"
            "3. Write a summary page under the right portal or hub, citing '/sources/<id>.md'.\n"
            "4. Update every affected page in place with the new facts and the citation; where a contradiction "
            "needs my decision, add a `review` question instead of overwriting.\n"
            "5. Update the hub and portal summaries, run kb_review, and list the pages you changed.")

app = mcp.http_app(middleware=[Middleware(web.WebGate)])


if __name__ == "__main__":
    # Build (or catch up) every user's section embeddings in the background, at low priority.
    embed.warm([(u["dir"], lambda d=u["dir"]: store.Bundle(store.user_by_dir(d)).all())
                for u in store._load_users().values()])
    uvicorn.run(app, host="0.0.0.0", port=8000, proxy_headers=True, forwarded_allow_ips="*")
