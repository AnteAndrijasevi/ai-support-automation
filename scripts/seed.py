"""Load synthetic sample tickets from fixtures/sample_tickets.json into the database.

Seeded tickets are inserted as raw, untriaged ingestions (category/urgency/sentiment
left null) -- the same state a ticket is in immediately after POST /tickets writes it
but before the AI pipeline runs. This script does not call the LLM; it only exercises
the data layer. Use the API (or eval/run_eval.py) to see the AI pipeline in action.

Usage:
    python -m scripts.seed
"""

import asyncio
import json
import logging
from pathlib import Path

from app.db import async_session_factory
from app.models import Ticket

logger = logging.getLogger(__name__)

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "sample_tickets.json"


async def seed() -> int:
    tickets_data = json.loads(FIXTURE_PATH.read_text())

    async with async_session_factory() as session, session.begin():
        for entry in tickets_data:
            session.add(Ticket(subject=entry["subject"], body=entry["body"]))

    return len(tickets_data)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    count = await seed()
    logger.info("Seeded %d tickets from %s", count, FIXTURE_PATH.name)


if __name__ == "__main__":
    asyncio.run(main())
