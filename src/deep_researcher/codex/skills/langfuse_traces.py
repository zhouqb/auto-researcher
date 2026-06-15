#!/usr/bin/env python3
"""Langfuse trace lookup for root-cause analysis (Codex skill).

Dropped into a repo-improvement workspace as ``.dr_langfuse.py`` when Langfuse
keys are configured. The diagnosis turn runs it to inspect *why* specific eval
cases failed — pulling the agent's actual trajectory (LLM prompts/outputs and
tool calls) from tracing, beyond what cases.jsonl records.

The eval harness tags each case's root span as ``eval_case:<case_id>`` with
``ada.correct`` / ``ada.category`` (see auto_data_analysis/observability.py), so
traces are findable by case id or by failure.

Usage (reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from env):

    python .dr_langfuse.py --failed [--limit 20]      # list recent failed cases
    python .dr_langfuse.py --case bird_173_financial  # one case's full trajectory

Self-contained (stdlib only); never raises on a bad response — prints a hint.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

CASE_PREFIX = "eval_case:"


def auth_header(public_key: str, secret_key: str) -> str:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return f"Basic {token}"


def api_get(host: str, path: str, public_key: str, secret_key: str,
            params: dict | None = None, *, timeout: float = 30.0) -> dict:
    url = f"{host.rstrip('/')}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Authorization": auth_header(public_key, secret_key),
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _truncate(value, n: int = 1200) -> str:
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    return s if len(s) <= n else s[:n] + f"…[+{len(s) - n} chars]"


def _data_list(payload) -> list[dict]:
    """The `data` array from an API payload, normalized to dict entries only —
    tolerates error payloads / version skew (`{"data": {}}`, non-dict items)."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def find_case_trace(host, pk, sk, case_id: str) -> dict | None:
    """The trace whose root span is named eval_case:<case_id>, or None."""
    name = f"{CASE_PREFIX}{case_id}"
    data = _data_list(api_get(host, "/api/public/traces", pk, sk,
                              {"name": name, "limit": 1}))
    return data[0] if data else None


def list_failed(host, pk, sk, limit: int = 20) -> list[dict]:
    """Recent eval-case traces with ada.correct == false (best-effort:
    correctness lands in trace metadata via the OTLP span attributes)."""
    out: list[dict] = []
    page = api_get(host, "/api/public/traces", pk, sk, {"limit": max(limit * 5, 50)})
    for t in _data_list(page):
        if not (t.get("name") or "").startswith(CASE_PREFIX):
            continue
        meta = t.get("metadata") if isinstance(t.get("metadata"), dict) else {}
        correct = meta.get("ada.correct")
        if correct is False or str(correct).lower() == "false":
            out.append(t)
        if len(out) >= limit:
            break
    return out


def trace_observations(host, pk, sk, trace_id: str) -> list[dict]:
    data = _data_list(api_get(host, "/api/public/observations", pk, sk,
                              {"traceId": trace_id, "limit": 200}))
    # chronological so the trajectory reads top-to-bottom
    return sorted(data, key=lambda o: o.get("startTime") or "")


def format_trajectory(host: str, trace: dict, obs: list[dict]) -> str:
    tid = trace.get("id", "?")
    meta = trace.get("metadata") or {}
    lines = [
        f"# {trace.get('name', '?')}",
        f"trace: {host.rstrip('/')}/trace/{tid}",
        f"metadata: {json.dumps({k: v for k, v in meta.items() if k.startswith('ada.')})}",
        "",
    ]
    for o in obs:
        kind = o.get("type", "OBSERVATION")
        name = o.get("name", "")
        lines.append(f"## [{kind}] {name}")
        if o.get("input") is not None:
            lines.append(f"  input:  {_truncate(o['input'])}")
        if o.get("output") is not None:
            lines.append(f"  output: {_truncate(o['output'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect Langfuse traces for eval cases.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--case", help="case id, e.g. bird_173_financial")
    g.add_argument("--failed", action="store_true", help="list recent failed cases")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args(argv)

    pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    if not (pk and sk):
        print("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — cannot query "
              "traces. Fall back to the eval/out/*/cases.jsonl dump.", file=sys.stderr)
        return 2

    try:
        if args.failed:
            failed = list_failed(host, pk, sk, args.limit)
            if not failed:
                print("No failed eval-case traces found (or correctness not in "
                      "trace metadata). Check eval/out/*/cases.jsonl.")
                return 0
            for t in failed:
                meta = t.get("metadata") or {}
                print(f"{t.get('name')}  category={meta.get('ada.category', '?')}  "
                      f"{host.rstrip('/')}/trace/{t.get('id')}")
            return 0

        trace = find_case_trace(host, pk, sk, args.case)
        if not trace:
            print(f"No trace named {CASE_PREFIX}{args.case}. It may not have been "
                  "traced (Langfuse off during that run) — see cases.jsonl.")
            return 1
        obs = trace_observations(host, pk, sk, trace["id"])
        print(format_trajectory(host, trace, obs))
        return 0
    except (urllib.error.URLError, OSError, ValueError) as e:
        # URLError/OSError: unreachable host / timeout. ValueError (incl.
        # JSONDecodeError): a non-JSON or truncated body. Either way, fall back.
        print(f"Langfuse request failed ({type(e).__name__}: {e}). Is "
              "LANGFUSE_HOST reachable and returning JSON? Fall back to "
              "eval/out/*/cases.jsonl.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
