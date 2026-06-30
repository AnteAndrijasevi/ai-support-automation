# AI Support Automation

[![CI](https://github.com/AnteAndrijasevi/ai-support-automation/actions/workflows/ci.yml/badge.svg)](https://github.com/AnteAndrijasevi/ai-support-automation/actions/workflows/ci.yml)

A backend service that triages incoming customer support tickets with an LLM:
classification (category, urgency, sentiment), a drafted reply, and a confidence
flag for human review. All sample/demo data in this repo is synthetic.

> Full documentation (architecture diagram, setup instructions, testing and
> evaluation-harness guides) is in progress.

## Tech stack

- Python 3.12, FastAPI, Pydantic v2
- PostgreSQL via SQLAlchemy (async) + Alembic migrations
- Anthropic API behind a provider-agnostic `LLMClient` protocol
- pytest + pytest-asyncio, with LLM calls mocked in the main test suite
- Docker + docker-compose
- GitHub Actions (lint + mocked tests on every push; a separate manual
  workflow for the real-API evaluation harness)
