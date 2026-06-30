import logging
import sys

from pythonjsonlogger.json import JsonFormatter

# Base fields always present on a record. Per-call "extra" fields (e.g. latency_ms,
# input_tokens, ticket_id) are merged into the JSON output automatically by JsonFormatter
# without needing to be listed here, so log calls that omit them don't raise.
BASE_LOG_FIELDS = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(BASE_LOG_FIELDS, rename_fields={"asctime": "timestamp"}))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
