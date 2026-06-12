"""Targeted runtime patches for ADK behaviors with no config hook.

Applied once by ``build_app()`` (idempotent). Each patch documents the
upstream gap; drop it when ADK grows the corresponding knob.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_applied = False


def apply_adk_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True
    _patch_tool_call_argument_parsing()


def _patch_tool_call_argument_parsing() -> None:
    """Make malformed tool-call arguments survivable.

    When the model's output is cut off mid tool call (e.g. a long
    write_artifact hitting the max_tokens cap), ADK's strict
    ``_parse_tool_call_arguments`` raises JSONDecodeError and the whole
    invocation dies. Returning ``{}`` instead routes the call into ADK's
    missing-mandatory-args path (function_tool.py), which sends the model a
    retryable error response — the run continues and the model can try again
    with shorter content.
    """
    from google.adk.models import lite_llm

    original = lite_llm._parse_tool_call_arguments

    def tolerant_parse(arguments):
        try:
            return original(arguments)
        except json.JSONDecodeError as exc:
            preview = arguments[:120] if isinstance(arguments, str) else arguments
            logger.warning(
                "dropping malformed tool-call arguments (%s); len=%s preview=%r"
                " — returning {} so the model gets a retryable tool error",
                exc,
                len(arguments) if isinstance(arguments, str) else "?",
                preview,
            )
            return {}

    tolerant_parse._original = original  # for tests / unpatching
    lite_llm._parse_tool_call_arguments = tolerant_parse
