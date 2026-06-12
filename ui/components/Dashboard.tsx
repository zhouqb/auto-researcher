"use client";

import { useEffect, useState } from "react";
import { GATEWAY_URL } from "@/lib/api";

interface Card {
  id: string;
  last_update_time: number;
  has_report: boolean;
  running_runs: number;
  total_runs: number;
  budget_totals: Record<string, number>;
  artifact_count: number;
}

export default function Dashboard({
  onSelect,
}: {
  onSelect: (pid: string) => void;
}) {
  const [cards, setCards] = useState<Card[]>([]);

  useEffect(() => {
    const load = () =>
      fetch(`${GATEWAY_URL}/api/dashboard`, { cache: "no-store" })
        .then((r) => r.json())
        .then(setCards)
        .catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  if (cards.length === 0)
    return (
      <main className="flex flex-1 items-center justify-center text-sm text-zinc-500">
        No projects yet — start one in the sidebar.
      </main>
    );

  return (
    <main className="flex-1 overflow-y-auto p-6">
      <h2 className="mb-4 text-sm font-semibold text-zinc-600 dark:text-zinc-300">
        Projects
      </h2>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {cards.map((c) => (
          <button
            key={c.id}
            onClick={() => onSelect(c.id)}
            className="rounded-xl border border-zinc-200 bg-white p-4 text-left shadow-sm hover:border-blue-400 hover:shadow dark:border-zinc-700 dark:bg-zinc-900"
          >
            <div className="flex items-center gap-2">
              <span className="truncate text-sm font-semibold">{c.id}</span>
              {c.running_runs > 0 && (
                <span className="ml-auto flex items-center gap-1 text-xs text-yellow-600">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-yellow-400" />
                  {c.running_runs} running
                </span>
              )}
              {c.running_runs === 0 && c.has_report && (
                <span className="ml-auto text-xs text-green-600">✓ report</span>
              )}
            </div>
            <p className="mt-2 text-xs text-zinc-500">
              {c.artifact_count} artifacts · {c.total_runs} runs
              {c.budget_totals.input_tokens
                ? ` · ${(c.budget_totals.input_tokens / 1000).toFixed(0)}k tok in`
                : ""}
            </p>
            <p className="mt-1 text-[11px] text-zinc-400">
              updated{" "}
              {new Date(c.last_update_time * 1000).toLocaleString(undefined, {
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </p>
          </button>
        ))}
      </div>
    </main>
  );
}
