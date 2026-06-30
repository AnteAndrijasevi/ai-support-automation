from app.models import Category, Sentiment, Urgency

SYSTEM_PROMPT = """You are a customer support triage assistant for a software product.

For each incoming ticket, you must:
1. Classify it into exactly one category, urgency level, and sentiment.
2. Draft a short, professional reply to the customer.
3. Report your own confidence in this classification.

Rules:
- Base the reply ONLY on information present in the ticket. Do not invent order numbers,
  names, dates, refund amounts, or any other detail that was not stated by the customer.
- If the ticket lacks information needed to fully resolve it, write a reply that
  acknowledges the issue and asks a clarifying question instead of guessing.
- "urgency" reflects how time-sensitive the issue is to the customer (e.g. a complete
  loss of access before a deadline is "critical"; a cosmetic suggestion is "low").
- "confidence" reflects how certain you are about the category/urgency/sentiment
  classification, not about the reply text. Use a low confidence (below 0.5) when the
  ticket is ambiguous, mixes multiple issues, or barely matches any category.
- Always respond by calling the classify_ticket tool. Never respond with plain text.
"""

USER_PROMPT_TEMPLATE = """Subject: {subject}

Body:
{body}"""

CLASSIFY_TICKET_TOOL = {
    "name": "classify_ticket",
    "description": "Record the triage classification and drafted reply for a support ticket.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [c.value for c in Category],
                "description": "The single best-fitting category for this ticket.",
            },
            "urgency": {
                "type": "string",
                "enum": [u.value for u in Urgency],
                "description": "How time-sensitive this issue is to the customer.",
            },
            "sentiment": {
                "type": "string",
                "enum": [s.value for s in Sentiment],
                "description": "The customer's emotional tone in the ticket.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Self-rated confidence in the category/urgency/sentiment call.",
            },
            "draft_reply": {
                "type": "string",
                "description": "A short, professional reply to send the customer.",
            },
        },
        "required": ["category", "urgency", "sentiment", "confidence", "draft_reply"],
    },
}
