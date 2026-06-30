# AI Support Automation

[![CI](https://github.com/AnteAndrijasevi/ai-support-automation/actions/workflows/ci.yml/badge.svg)](https://github.com/AnteAndrijasevi/ai-support-automation/actions/workflows/ci.yml)

A backend service that automates triage of incoming customer support tickets.
It ingests a ticket (subject + body), calls an LLM to classify it (category,
urgency, sentiment), drafts a suggested reply, and persists everything in
Postgres along with a confidence flag for human review. It's exposed both as a
REST API and as MCP tools, and ships with an evaluation harness that scores
the pipeline against a hand-labeled set of sample tickets and checks drafted
replies for likely hallucinations.

**All ticket data in this repo (fixtures, seed data, golden set) is
synthetic** -- invented tickets for a fictional product, no real company or
customer data anywhere.

## Architecture

```mermaid
flowchart TB
    RESTClient[REST client] --> API[FastAPI app]
    MCPClient[MCP client] --> MCPServer[MCP server]

    API --> Pipeline[ticket_pipeline service]
    MCPServer --> Pipeline

    Pipeline --> LLMClient[["LLMClient protocol"]]
    Pipeline --> DB[(PostgreSQL)]
    LLMClient --> AnthropicImpl[AnthropicLLMClient]
    AnthropicImpl --> AnthropicAPI((Anthropic API))

    subgraph Offline["Offline evaluation (manual, real API calls)"]
        Eval[eval/run_eval.py] --> GoldenSet[eval/golden_set.json]
        Eval --> LLMClient
        Eval --> Report[eval/report.md]
    end
```

The REST API and the MCP server are two entry points into the same
`app/services/ticket_pipeline.py` module -- neither one has its own copy of
the classification logic, so they can't drift out of sync with each other.
The evaluation harness is a deliberately separate, offline path: it talks to
the same `LLMClient` directly, but is never invoked by the API and never runs
in the default CI workflow, since it costs real API calls.

## Tech stack

- **Python 3.12, FastAPI, Pydantic v2** -- async throughout, request/response
  validation, and OpenAPI docs for free.
- **PostgreSQL via SQLAlchemy (async) + Alembic** -- a real relational schema
  with check constraints on the AI-assigned enum fields, migrated rather than
  `create_all`'d.
- **Anthropic API behind an `LLMClient` Protocol** (`app/llm/base.py`) -- the
  rest of the codebase (API routes, the pipeline service, the MCP server, the
  eval harness) depends only on this protocol, never on the Anthropic SDK
  directly. `AnthropicLLMClient` is the only implementation today, but
  swapping or adding a provider means writing one new class that satisfies
  the protocol -- nothing else changes. This is a structural `typing.Protocol`,
  not just a comment promising future flexibility: a test stub
  (`tests/fakes.py::FakeLLMClient`) satisfies it with no inheritance at all.
- **Anthropic tool-use for structured output** -- classification is returned
  via a forced tool call (`tool_choice`), not parsed out of free text.
- **pytest + pytest-asyncio** -- LLM calls are mocked in the main suite (see
  Testing below); tests run against a real local Postgres, not SQLite, so
  the check constraints and the `Uuid` column type behave exactly as in
  production.
- **Docker + docker-compose** -- `app` + `db` services; migrations run
  automatically on container start.
- **GitHub Actions** -- one workflow lints and runs the mocked test suite on
  every push/PR; a separate, manually-triggered workflow runs the real-API
  evaluation harness and uploads its report as an artifact.
- **Structured (JSON) logging** -- a `/health` endpoint, and per-request
  latency + per-LLM-call token usage logged as structured fields.
- **MCP server** (`app/mcp/server.py`) -- exposes `classify_ticket` and
  `create_ticket` as MCP tools, alongside (not instead of) the REST API.

## API

Once running, interactive docs are at `/docs` (Swagger UI) and `/redoc`.

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `POST /tickets` | Ingest a ticket, run it through the AI pipeline, return the result |
| `GET /tickets/{id}` | Fetch a single ticket |
| `GET /tickets` | List tickets, filterable by category/urgency/sentiment/confidence_flag/human_reviewed, paginated |
| `POST /tickets/batch-import` | Bulk-ingest tickets from a CSV file (`subject`, `body` columns) |

If the LLM call fails (timeout, rate limit, API error), the ticket is still
saved -- it's returned with a `classification_error` field instead of losing
the submission. A database outage returns a clean `503` instead of a stack
trace.

## Setup

### Docker quickstart

```bash
cp .env.example .env   # add your ANTHROPIC_API_KEY if you want real classifications
docker compose up --build
```

This brings up Postgres and the app, applying Alembic migrations
automatically before the server starts. The API is then at
`http://localhost:8000` (`/docs` for interactive docs).

### Local development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # set DATABASE_URL and (optionally) ANTHROPIC_API_KEY

# requires a running Postgres reachable at DATABASE_URL, e.g.:
docker compose up -d db

alembic upgrade head
python -m scripts.seed       # optional: load 18 synthetic sample tickets (untriaged)

uvicorn app.main:app --reload
```

Pre-commit hooks (ruff lint + format) can be installed with
`pre-commit install`.

### MCP server

```bash
python -m app.mcp.server
```

Runs over stdio, exposing `classify_ticket` and `create_ticket` as tools for
an MCP-compatible client.

## Running tests

The main test suite mocks all LLM calls (no API key or network access
needed) but runs against a real local Postgres, so check constraints and
column types are exercised exactly as in production:

```bash
docker compose up -d db   # or any reachable Postgres
pytest                     # runs with coverage by default (see pyproject.toml)
```

`TEST_DATABASE_URL` defaults to
`postgresql+asyncpg://postgres:postgres@localhost:5432/ai_support_test` and
is created automatically if it doesn't exist. Override it via the environment
if your local Postgres uses different credentials.

This is the suite that runs in CI on every push/PR -- it's free to run and
never touches the real Anthropic API.

## Running the evaluation harness

This is the **one place** in the repo that makes real, billed Anthropic API
calls, which is why it's not part of the default CI workflow:

```bash
python -m eval.run_eval --output-file eval/report.md
```

It runs `eval/golden_set.json` (18 hand-labeled synthetic tickets) through
the real pipeline, scores category/urgency accuracy against the expected
labels, and runs a heuristic faithfulness check (`eval/faithfulness.py`) that
flags drafted replies containing a reference number, dollar amount, or email
address that doesn't appear anywhere in the source ticket -- the cheapest,
highest-signal class of hallucination to catch without a second LLM-as-judge
call. The report is printed to stdout and, with `--output-file`, also written
as markdown.

In CI, this lives in a separate `workflow_dispatch`-only workflow
(`.github/workflows/eval.yml`) that you trigger manually from the Actions
tab; it uploads the report as a build artifact. It needs an
`ANTHROPIC_API_KEY` repository secret to run there.
