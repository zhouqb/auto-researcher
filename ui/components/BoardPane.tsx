"use client";

import { useEffect, useState } from "react";
import { api, BoardItem } from "@/lib/api";

const COLUMNS = [
  "backlog",
  "in_progress",
  "blocked",
  "awaiting_review",
  "done",
  "killed",
] as const;

const COLUMN_STYLE: Record<string, string> = {
  backlog: "border-zinc-300",
  in_progress: "border-yellow-400",
  blocked: "border-red-400",
  awaiting_review: "border-blue-400",
  done: "border-green-500",
  killed: "border-zinc-600",
};

export default function BoardPane({ projectId }: { projectId: string }) {
  const [items, setItems] = useState<BoardItem[]>([]);

  useEffect(() => {
    const load = () =>
      api
        .board(projectId)
        .then((b) => setItems(b.items))
        .catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [projectId]);

  if (items.length === 0)
    return (
      <p className="text-sm text-zinc-500">
        No board yet — created when the plan is saved.
      </p>
    );

  const present = COLUMNS.filter((c) => items.some((i) => i.status === c));

  return (
    <div className="grid grid-cols-2 gap-3">
      {present.map((col) => (
        <div key={col}>
          <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-zinc-500">
            {col.replace("_", " ")}
          </h3>
          <div className="space-y-1.5">
            {items
              .filter((i) => i.status === col)
              .map((item) => (
                <div
                  key={item.id}
                  className={`rounded-md border-l-4 ${COLUMN_STYLE[col]} bg-zinc-50 p-2 text-xs dark:bg-zinc-900`}
                >
                  <span className="mr-1 rounded bg-zinc-200 px-1 py-0.5 font-mono text-[10px] dark:bg-zinc-700">
                    {item.type ?? "?"}
                  </span>
                  {item.title ?? item.id}
                  {item.status_reason && (
                    <p className="mt-0.5 italic text-zinc-500">
                      {item.status_reason}
                    </p>
                  )}
                </div>
              ))}
          </div>
        </div>
      ))}
    </div>
  );
}
