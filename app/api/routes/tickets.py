import csv
import io
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.dependencies import get_llm_client
from app.llm.base import LLMClient
from app.models import Category, ConfidenceFlag, Sentiment, Ticket, Urgency
from app.schemas import (
    BatchImportResponse,
    BatchImportRowResult,
    TicketCreate,
    TicketListResponse,
    TicketRead,
)
from app.services.ticket_pipeline import ingest_ticket

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tickets", tags=["tickets"])

MAX_BATCH_ROWS = 200
REQUIRED_CSV_COLUMNS = {"subject", "body"}


def _to_ticket_read(result_ticket: Ticket, classification_error: str | None = None) -> TicketRead:
    return TicketRead.model_validate(result_ticket).model_copy(
        update={"classification_error": classification_error}
    )


@router.post(
    "",
    response_model=TicketRead,
    status_code=201,
    summary="Ingest a ticket and run it through the AI triage pipeline",
)
async def create_ticket(
    payload: TicketCreate,
    db: AsyncSession = Depends(get_db),
    llm_client: LLMClient = Depends(get_llm_client),
) -> TicketRead:
    result = await ingest_ticket(db, llm_client, subject=payload.subject, body=payload.body)
    return _to_ticket_read(result.ticket, result.classification_error)


@router.get(
    "/{ticket_id}",
    response_model=TicketRead,
    summary="Fetch a single ticket by id",
)
async def get_ticket(ticket_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> TicketRead:
    ticket = await db.get(Ticket, ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return TicketRead.model_validate(ticket)


@router.get(
    "",
    response_model=TicketListResponse,
    summary="List tickets, optionally filtered by category/urgency/sentiment",
)
async def list_tickets(
    db: AsyncSession = Depends(get_db),
    category: Category | None = None,
    urgency: Urgency | None = None,
    sentiment: Sentiment | None = None,
    confidence_flag: ConfidenceFlag | None = None,
    human_reviewed: bool | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> TicketListResponse:
    filters = []
    if category is not None:
        filters.append(Ticket.category == category)
    if urgency is not None:
        filters.append(Ticket.urgency == urgency)
    if sentiment is not None:
        filters.append(Ticket.sentiment == sentiment)
    if confidence_flag is not None:
        filters.append(Ticket.confidence_flag == confidence_flag)
    if human_reviewed is not None:
        filters.append(Ticket.human_reviewed == human_reviewed)

    count_stmt = select(func.count()).select_from(Ticket).where(*filters)
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = (
        select(Ticket)
        .where(*filters)
        .order_by(Ticket.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    tickets = (await db.execute(stmt)).scalars().all()

    return TicketListResponse(
        items=[TicketRead.model_validate(t) for t in tickets],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/batch-import",
    response_model=BatchImportResponse,
    summary="Bulk-ingest tickets from a CSV file with 'subject' and 'body' columns",
)
async def batch_import_tickets(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    llm_client: LLMClient = Depends(get_llm_client),
) -> BatchImportResponse:
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="File is not valid UTF-8 text") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or not REQUIRED_CSV_COLUMNS.issubset(set(reader.fieldnames)):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must include columns: {sorted(REQUIRED_CSV_COLUMNS)}",
        )

    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file has no data rows")
    if len(rows) > MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=400, detail=f"CSV has {len(rows)} rows; max is {MAX_BATCH_ROWS}"
        )

    results: list[BatchImportRowResult] = []
    for i, row in enumerate(rows, start=1):
        subject = (row.get("subject") or "").strip()
        body = (row.get("body") or "").strip()
        if not subject or not body:
            results.append(
                BatchImportRowResult(
                    row=i, status="failed", error="subject and body are both required"
                )
            )
            continue

        try:
            ingest_result = await ingest_ticket(db, llm_client, subject=subject, body=body)
        except Exception as exc:  # noqa: BLE001 - isolate one bad row from the rest of the batch
            logger.exception("Unexpected error importing row %d", i)
            results.append(BatchImportRowResult(row=i, status="failed", error=str(exc)))
            continue

        results.append(
            BatchImportRowResult(row=i, status="created", ticket_id=ingest_result.ticket.id)
        )

    created = sum(1 for r in results if r.status == "created")
    return BatchImportResponse(
        total_rows=len(rows),
        created=created,
        failed=len(rows) - created,
        results=results,
    )
