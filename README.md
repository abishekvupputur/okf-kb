# okf-kb

A self-hosted personal knowledge base that Claude maintains for you: a wiki of markdown pages in
[Open Knowledge Format](https://github.com/GoogleCloudPlatform/open-knowledge-format) (OKF v0.2), served as an **MCP server** (so
Claude can read and write it) plus a small **web UI** to browse, review and answer its questions.

- One private bundle per signed-in user: a git repository of markdown files with YAML frontmatter.
  Every change is a commit, so history, diffs and restore come for free.
- Nothing is ever shared or published: every page needs a sign-in and each user only sees their own
  bundle.
- Keyword search plus section embeddings (`stella-en-400M-v5`, MIT, via FastEmbed on ONNX Runtime,
  no PyTorch; ~3 GB RAM while building the index), a relevance graph, a structure lint, and guards that refuse secrets and malformed pages.
- Sign-in through any OIDC provider; the MCP OAuth flow for
  Claude's connector is handled by FastMCP's `OIDCProxy`.

## Run

```sh
docker build -t okf-kb .
docker run -d --name kb -p 8000:8000 -v kb-data:/data --env-file kb.env okf-kb
```

`kb.env`:

| Variable | What |
|---|---|
| `PUBLIC_URL` | Where the server is reached, e.g. `https://kb.example.com` (HTTPS, behind a reverse proxy or tunnel) |
| `OIDC_CONFIG_URL` | Your provider's `.well-known/openid-configuration` |
| `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET` | A confidential OAuth2 client; redirect URIs `<PUBLIC_URL>/auth/callback` and `<PUBLIC_URL>/web/callback` |
| `JWT_SIGNING_KEY` | Random string; signs the tokens handed to MCP clients and the web session cookie |
| `KB_EMBED_THREADS` | Optional: CPU threads for embedding (default 1) |
| `KB_SYNONYMS` | Optional: search synonym groups, `;` between groups, spaces between words, e.g. `sso oidc;backup snapshot` (default `sso oidc`) |
| `STORAGE_ENCRYPTION_KEY` | A Fernet key (`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`) encrypting stored client registrations |

The provider must return `sub` and `email`, and should issue refresh tokens for `offline_access`
(otherwise Claude's connector drops after the access token expires). Limit who may sign in at the
provider: everyone who can sign in gets their own bundle.

Then in Claude: Settings → Connectors → Add custom connector → `<PUBLIC_URL>/mcp`.

## Data

Everything lives in `/data`: `bundles/<user>/` (one git repo each), `users.json` (OIDC subject →
bundle), `embeddings/` (rebuildable), `oauth/` (encrypted client registrations). Back up `/data`.
The data never belongs in this repository.

## Design

- **Bundle:** a git repo of markdown concepts with YAML frontmatter (`type` required; `title`,
  `description`, `tags`, `sources`, `generated`, `verified`, `status`, `stale_after` optional). Every
  folder gets a generated `index.md` (the root one carries `okf_version`), and the root has a
  `log.md`. The OKF spec is the contract.
- **Structure, enforced on every write** (`guard.py`): the root holds only portals (type `Portal`);
  every other page sits under its portal or a hub (a page with the same ID as its folder:
  `projects/garden` is the hub of `projects/garden/irrigation`). Pages open with a 1–3 sentence
  lead; hubs carry an `infobox`. Writes over 12000 characters, without lead/title/description/tags,
  with a duplicate title, a secret, literal `\n` escapes, or cutting a page by more than half
  (without `shrink_ok`) are refused with the reason. `kb_move` moves subtrees and keeps old IDs as
  `aliases`.
- **Reading is progressive:** `kb_list` shows one level; `kb_read` returns long pages as lead +
  outline (`section=` reads one part). `kb_context(question)` answers from search + graph expansion
  with numbered pages to cite. `kb_ingest` stores a raw source unchanged under `sources/` (the
  citation layer, left out of search unless asked).
- **Trust:** writes from Claude carry `generated.by: <agent>`; `human:<user>` verification only comes
  from the web "Mark as reviewed" button or `kb_verify(confirmed_by_user=True)`.
- **Users:** `users.json` maps the OIDC `sub` to a bundle dir (fixed at first sign-in), so the web
  UI and Claude always see the same bundle. Each bundle is its own repo; histories never mix.
- **Web:** CSP allows scripts from self only (sigma, graphology vendored in `static/vendor/`), POSTs
  must carry this origin (CSRF), nothing but `/static/` may be cached (`Cache-Control: private,
  no-store`; CDNs cache by file extension and would otherwise serve `/export.tar.gz` to anyone).
- **Agent guidance** is the MCP server `instructions` (`server.py`), which every client receives,
  plus the prompts `record_session` and `ingest_source`.

Code: `okf.py` (parse/validate), `guard.py` (write rules, lint), `graph.py` (relevance graph),
`search.py` (keywords), `embed.py` (embeddings), `store.py` (users, bundles, git), `tools.py` (MCP
tools), `web.py` (browser UI), `render.py` (markdown → sanitized HTML), `server.py` (entry point).

**Writing without the MCP connector:** `store.Bundle.save` commits but does not run the
guardrails, so call `guard.check` first, as the tools do:

```sh
docker exec -i kb python - <<'EOF'
import sys; sys.path.insert(0, "/srv")
from app import guard, okf, store
b = store.Bundle(store.user_by_dir("<dir>"))
cid = "projects/garden"
meta, body = b.read(cid); old = body
body = body.replace("old fact", "new fact")
meta["generated"] = {"by": "claude/<model>", "at": okf.now_iso()}
print(guard.check(b, cid, meta, body, old_body=old))   # raises GuardError on a bad write
b.save(cid, meta, body, f"Update: [{meta['title']}](/{cid}.md)")
EOF
```
