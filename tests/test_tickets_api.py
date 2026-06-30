import uuid

from app.llm.base import LLMTimeoutError, TicketClassification


async def test_create_ticket_happy_path(client, fake_llm):
    fake_llm.response = TicketClassification(
        category="billing",
        urgency="high",
        sentiment="negative",
        confidence=0.92,
        draft_reply="Sorry about the duplicate charge -- we'll refund it.",
        input_tokens=80,
        output_tokens=25,
    )

    response = await client.post(
        "/tickets",
        json={"subject": "Charged twice", "body": "I was billed twice this month."},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["category"] == "billing"
    assert body["urgency"] == "high"
    assert body["confidence_flag"] == "ok"
    assert body["classification_error"] is None
    assert uuid.UUID(body["id"])


async def test_create_ticket_rejects_empty_subject(client):
    response = await client.post("/tickets", json={"subject": "", "body": "something"})
    assert response.status_code == 422


async def test_create_ticket_rejects_missing_body(client):
    response = await client.post("/tickets", json={"subject": "Help"})
    assert response.status_code == 422


async def test_create_ticket_survives_llm_timeout(client, fake_llm):
    fake_llm.error = LLMTimeoutError("Anthropic API timed out after 30s")

    response = await client.post("/tickets", json={"subject": "Help", "body": "It's broken."})

    assert response.status_code == 201
    body = response.json()
    assert body["category"] is None
    assert body["classification_error"] == "Anthropic API timed out after 30s"


async def test_get_ticket_returns_created_ticket(client):
    create_resp = await client.post("/tickets", json={"subject": "s", "body": "b"})
    ticket_id = create_resp.json()["id"]

    response = await client.get(f"/tickets/{ticket_id}")

    assert response.status_code == 200
    assert response.json()["id"] == ticket_id


async def test_get_ticket_404_for_unknown_id(client):
    response = await client.get(f"/tickets/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_get_ticket_422_for_malformed_id(client):
    response = await client.get("/tickets/not-a-uuid")
    assert response.status_code == 422


async def test_list_tickets_filters_by_category(client, fake_llm):
    fake_llm.response = TicketClassification(
        category="billing",
        urgency="low",
        sentiment="neutral",
        confidence=0.9,
        draft_reply="Reply",
    )
    await client.post("/tickets", json={"subject": "s1", "body": "b1"})

    fake_llm.response = TicketClassification(
        category="technical",
        urgency="low",
        sentiment="neutral",
        confidence=0.9,
        draft_reply="Reply",
    )
    await client.post("/tickets", json={"subject": "s2", "body": "b2"})

    response = await client.get("/tickets", params={"category": "billing"})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["category"] == "billing"


async def test_list_tickets_filters_by_urgency_sentiment_and_review_state(client, fake_llm):
    fake_llm.response = TicketClassification(
        category="technical",
        urgency="critical",
        sentiment="negative",
        confidence=0.95,
        draft_reply="Reply",
    )
    await client.post("/tickets", json={"subject": "s1", "body": "b1"})

    fake_llm.response = TicketClassification(
        category="technical",
        urgency="low",
        sentiment="positive",
        confidence=0.95,
        draft_reply="Reply",
    )
    await client.post("/tickets", json={"subject": "s2", "body": "b2"})

    response = await client.get(
        "/tickets",
        params={"urgency": "critical", "sentiment": "negative", "human_reviewed": "false"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["urgency"] == "critical"


async def test_list_tickets_filters_by_confidence_flag(client, fake_llm):
    fake_llm.response = TicketClassification(
        category="general", urgency="low", sentiment="neutral", confidence=0.2, draft_reply="Reply"
    )
    await client.post("/tickets", json={"subject": "s1", "body": "b1"})

    response = await client.get("/tickets", params={"confidence_flag": "low_confidence"})

    assert response.status_code == 200
    assert response.json()["total"] == 1


async def test_list_tickets_paginates(client):
    for i in range(3):
        await client.post("/tickets", json={"subject": f"s{i}", "body": f"b{i}"})

    response = await client.get("/tickets", params={"limit": 2, "offset": 0})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert body["limit"] == 2


async def test_batch_import_creates_and_skips_invalid_rows(client):
    csv_content = (
        "subject,body\n"
        "Slow app,The app is slow today.\n"
        ",Missing subject\n"
        "Crash,App crashes on launch.\n"
    )
    files = {"file": ("tickets.csv", csv_content.encode(), "text/csv")}

    response = await client.post("/tickets/batch-import", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["total_rows"] == 3
    assert body["created"] == 2
    assert body["failed"] == 1
    assert body["results"][1]["status"] == "failed"


async def test_batch_import_rejects_non_utf8_file(client):
    files = {"file": ("tickets.csv", b"subject,body\n\xff\xfe garbage,bad\n", "text/csv")}
    response = await client.post("/tickets/batch-import", files=files)
    assert response.status_code == 400


async def test_batch_import_rejects_missing_columns(client):
    files = {"file": ("tickets.csv", b"foo,bar\n1,2\n", "text/csv")}
    response = await client.post("/tickets/batch-import", files=files)
    assert response.status_code == 400


async def test_batch_import_rejects_empty_file(client):
    files = {"file": ("tickets.csv", b"subject,body\n", "text/csv")}
    response = await client.post("/tickets/batch-import", files=files)
    assert response.status_code == 400


async def test_batch_import_rejects_too_many_rows(client):
    header = "subject,body\n"
    rows = "\n".join(f"s{i},b{i}" for i in range(201))
    files = {"file": ("tickets.csv", (header + rows).encode(), "text/csv")}
    response = await client.post("/tickets/batch-import", files=files)
    assert response.status_code == 400
