# okf-kb

A self-hosted personal knowledge base that Claude maintains for you: a wiki of markdown pages in
[Open Knowledge Format](https://github.com/okf-spec) (OKF v0.2), served as an **MCP server** (so
Claude can read and write it) plus a small **web UI** to browse, review and answer its questions.

- One private bundle per signed-in user: a git repository of markdown files with YAML frontmatter.
  Every change is a commit, so history, diffs and restore come for free.
- Nothing is ever shared or published: every page needs a sign-in and each user only sees their own
  bundle.
- Keyword search plus section embeddings (`bge-small-en-v1.5` via FastEmbed on ONNX Runtime, no
  PyTorch), a relevance graph, a structure lint, and guards that refuse secrets and malformed pages.
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
| `STORAGE_ENCRYPTION_KEY` | A Fernet key (`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`) encrypting stored client registrations |

The provider must return `sub` and `email`, and should issue refresh tokens for `offline_access`
(otherwise Claude's connector drops after the access token expires). Limit who may sign in at the
provider: everyone who can sign in gets their own bundle.

Then in Claude: Settings → Connectors → Add custom connector → `<PUBLIC_URL>/mcp`.

## Data

Everything lives in `/data`: `bundles/<user>/` (one git repo each), `users.json` (OIDC subject →
bundle), `embeddings/` (rebuildable), `oauth/` (encrypted client registrations). Back up `/data`.
The data never belongs in this repository.

## Checks

```sh
python -m app.test_search                   # no dependencies beyond the code
docker exec kb python -m app.test_embed
docker exec kb python -m app.test_tools     # read-only, against the first bundle
```
