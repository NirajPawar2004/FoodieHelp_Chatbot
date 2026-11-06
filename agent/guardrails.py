"""
Guardrails and Security Layer for FoodieHelp AI
Prevents prompt injection, ensures customer order isolation, and validates parameters.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, List

# Malicious patterns commonly seen in prompt injection and bypass attacks
INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"system prompt",
    r"you are now in (developer|dan|god|admin) mode",
    r"drop table",
    r"delete from",
    r"execute arbitrary",
    r"jailbreak",
    r"bypass security",
    r"override rules",
    r"<script.*?>",
]


class Guardrails:
    """Security validations and sanitizers for user inputs."""

    @staticmethod
    def check_for_injection(user_message: str) -> Tuple[bool, Optional[str]]:
        """
        Inspect message for prompt injection patterns.
        Returns (is_violation, explanation).
        """
        lower = user_message.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, lower):
                return True, (
                    "⚠️ Security Notice: Your input contains unauthorized instructions or patterns "
                    "that violate FoodieHelp safety policies. Please rephrase your support inquiry."
                )
        return False, None

    @staticmethod
    def extract_order_id(text: str) -> Optional[str]:
        """
        Extract order ID patterns like ORD1001, ORD-1001, ord1015, #ORD1001, or #1001 from text.
        """
        if not text:
            return None

        # Pattern 1: explicit prefix ORD1001 / ORD-1001
        match = re.search(r"\bORD[-_]?(\d{3,6})\b", text, re.IGNORECASE)
        if match:
            digits = match.group(1)
            return f"ORD{digits}"

        # Pattern 2: #1001
        match_hash = re.search(r"#\s*(\d{4,6})\b", text)
        if match_hash:
            return f"ORD{match_hash.group(1)}"

        # Pattern 3: Standalone 4 digits if adjacent to 'order'
        match_near = re.search(r"order\s+(?:id\s+)?(?:is\s+)?(\d{4})\b", text, re.IGNORECASE)
        if match_near:
            return f"ORD{match_near.group(1)}"

        return None

    @staticmethod
    def sanitize_text(text: str, max_length: int = 1000) -> str:
        """Strip control characters, trim whitespace, and enforce length bounds."""
        if not text:
            return ""
        # Remove null bytes and non-printable characters
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        return cleaned.strip()[:max_length]

