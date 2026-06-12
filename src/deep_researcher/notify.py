"""Desktop notifications for unattended periods (design §11.4).

macOS: osascript; Linux: notify-send; otherwise no-op. Optional webhook
(Slack-compatible JSON {"text": ...}) via NOTIFY_WEBHOOK_URL. Never raises —
a failed notification must not break a run.
"""

from __future__ import annotations

import logging
import subprocess
import sys

from .config import get_settings

logger = logging.getLogger(__name__)


def notify(title: str, message: str) -> None:
    settings = get_settings()
    if not settings.desktop_notifications:
        return
    try:
        if sys.platform == "darwin":
            script = f'display notification "{_esc(message)}" with title "{_esc(title)}"'
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, timeout=5, check=False,
            )
        elif sys.platform.startswith("linux"):
            subprocess.run(
                ["notify-send", title, message],
                capture_output=True, timeout=5, check=False,
            )
    except Exception:
        logger.debug("desktop notification failed", exc_info=True)
    if settings.notify_webhook_url:
        try:
            import httpx

            httpx.post(
                settings.notify_webhook_url,
                json={"text": f"*{title}*\n{message}"},
                timeout=5,
            )
        except Exception:
            logger.warning("notification webhook failed", exc_info=True)


def _esc(text: str) -> str:
    return text.replace("\\", "").replace('"', "'")[:200]
