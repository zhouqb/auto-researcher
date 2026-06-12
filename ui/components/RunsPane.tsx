"use client";

import { useCallback, useEffect, useState } from "react";
import { api, Budget, RunInfo } from "@/lib/api";

const STATUS_DOT: Record<RunInfo["status"], string> = {
  running: "bg-yellow-400 animate-pulse",
  completed: "bg-green-500",
  failed: "bg-red-500",
  timeout: "bg-orange-500",
  killed: "bg-zinc-600",
};

export default function RunsPane({ projectId }: { projectId: string }) {
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [budget, setBudget] = useState<Budget | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [r, b] = await Promise.all([api.runs(projectId), api.budget(projectId)]);
      setRuns(r);
      setBudget(b);
    } catch {
      /* gateway may be momentarily busy */
    }
  }, [projectId]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, [refresh]);

  const totals = budget?.totals ?? {};

  return (
    <div className="space-y-3 text-sm">
      {Object.keys(totals).length > 0 && (
        <div className="rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-xs dark:border-zinc-700 dark:bg-zinc-900">
          <span className="font-semibold">Budget</span> · runs:{" "}
          {budget?.entries.length ?? 0} · in:{" "}
          {(totals.input_tokens ?? 0).toLocaleString()} tok · out:{" "}
          {(totals.output_tokens ?? 0).toLocaleString()} tok · wallclock:{" "}
          {totals.wallclock_s ?? 0}s
        </div>
      )}
      {runs.length === 0 && (
        <p className="text-zinc-500">No experiment runs yet.</p>
      )}
      {runs.map((run) => (
        <details
          key={run.run_id}
          open={run.status === "running"}
          className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-700"
        >
          <summary className="flex cursor-pointer items-center gap-2">
            <span className={`h-2.5 w-2.5 rounded-full ${STATUS_DOT[run.status]}`} />
            <code className="text-xs">{run.experiment}</code>
            <span className="text-xs text-zinc-500">{run.run_id}</span>
            <span className="ml-auto text-xs font-medium">{run.status}</span>
            {run.status === "running" && (
              <button
                onClick={(e) => {
                  e.preventDefault();
                  api.killRun(projectId, run.run_id).then(refresh);
                }}
                className="rounded bg-red-600 px-2 py-0.5 text-xs text-white hover:bg-red-700"
              >
                Kill
              </button>
            )}
          </summary>
          <div className="mt-2 space-y-2 text-xs">
            {Object.keys(run.usage).length > 0 && (
              <p className="text-zinc-500">
                tokens in/out: {(run.usage.input_tokens ?? 0).toLocaleString()}/
                {(run.usage.output_tokens ?? 0).toLocaleString()} ·{" "}
                {run.wallclock_s}s
              </p>
            )}
            {run.metrics && (
              <pre className="overflow-x-auto rounded bg-zinc-100 p-2 dark:bg-zinc-900">
                {JSON.stringify(run.metrics, null, 2)}
              </pre>
            )}
            {run.commands.map((cmd, i) => (
              <pre
                key={i}
                className="overflow-x-auto rounded bg-zinc-100 p-1.5 font-mono dark:bg-zinc-900"
              >
                $ {cmd}
              </pre>
            ))}
            {run.files_changed.length > 0 && (
              <p className="text-zinc-500">
                files: {run.files_changed.join(", ")}
              </p>
            )}
            {run.last_message && <p>{run.last_message}</p>}
          </div>
        </details>
      ))}
    </div>
  );
}
