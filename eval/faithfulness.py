"""A deliberately simple, heuristic faithfulness check.

Not an LLM-as-judge (that would cost a second API call per ticket and add its
own hallucination risk) -- instead it pattern-matches the kinds of specific,
authoritative-sounding details a model is most likely to fabricate (reference
numbers, dollar amounts, email addresses) and flags any that appear in the
drafted reply but not in the original ticket text. It will not catch every
hallucination (e.g. fabricated free-text claims), but it catches the riskiest
category cheaply: invented specifics that look like real account data.
"""

import re

REFERENCE_CODE_PATTERN = re.compile(r"\b[A-Za-z]{2,6}-?\d{3,}\b")
MONEY_PATTERN = re.compile(r"\$\s?\d+(?:\.\d{2})?")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

PATTERNS = (REFERENCE_CODE_PATTERN, MONEY_PATTERN, EMAIL_PATTERN)


def find_unsupported_claims(draft_reply: str, source_text: str) -> list[str]:
    """Return reply substrings (codes/amounts/emails) absent from the source ticket."""
    source_lower = source_text.lower()
    flagged: list[str] = []
    for pattern in PATTERNS:
        for match in pattern.findall(draft_reply):
            if match.lower() not in source_lower and match not in flagged:
                flagged.append(match)
    return flagged
