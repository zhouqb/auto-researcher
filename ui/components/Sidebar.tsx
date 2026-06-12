"use client";

import { useEffect, useState } from "react";
import { api, Project } from "@/lib/api";

export default function Sidebar({
  projectId,
  onSelect,
  onDeleted,
}: {
  projectId: string | null;
  onSelect: (pid: string) => void;
  onDeleted: (pid: string) => void;
}) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [question, setQuestion] = useState("");
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);

  const refresh = () => api.projects().then(setProjects).catch(() => {});
  useEffect(() => {
    refresh();
  }, []);

  const create = async () => {
    if (!question.trim()) return;
    setCreating(true);
    try {
      const { id } = await api.createProject(question.trim());
      await refresh();
      onSelect(id);
      setQuestion("");
    } finally {
      setCreating(false);
    }
  };

  const remove = async (pid: string) => {
    if (!confirm(`Delete project "${pid}" and all its artifacts? This cannot be undone.`))
      return;
    setDeleting(pid);
    try {
      await api.deleteProject(pid);
      onDeleted(pid);
    } catch (e) {
      alert(`Delete failed: ${e instanceof Error ? e.message : e}`);
    } finally {
      setDeleting(null);
      refresh();
    }
  };

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-800 dark:bg-zinc-950">
      <h1 className="mb-3 text-sm font-bold">🔬 Deep Researcher</h1>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="New research question…"
        rows={3}
        className="mb-1.5 w-full rounded-md border border-zinc-300 bg-white p-2 text-xs dark:border-zinc-700 dark:bg-zinc-900"
      />
      <button
        onClick={create}
        disabled={creating || !question.trim()}
        className="mb-4 rounded-md bg-blue-600 py-1.5 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-40"
      >
        {creating ? "Creating…" : "Start project"}
      </button>
      <div className="min-h-0 flex-1 space-y-1 overflow-y-auto">
        {projects
          .sort((a, b) => b.last_update_time - a.last_update_time)
          .map((p) => (
            <div
              key={p.id}
              className={`group flex items-center rounded-md ${
                p.id === projectId
                  ? "bg-blue-100 dark:bg-blue-950"
                  : "hover:bg-zinc-200 dark:hover:bg-zinc-800"
              }`}
            >
              <button
                onClick={() => onSelect(p.id)}
                className={`min-w-0 flex-1 truncate px-2 py-1.5 text-left text-xs ${
                  p.id === projectId
                    ? "font-medium text-blue-900 dark:text-blue-200"
                    : ""
                }`}
              >
                {p.id}
              </button>
              <button
                onClick={() => remove(p.id)}
                disabled={deleting === p.id}
                title="Delete project"
                aria-label={`Delete project ${p.id}`}
                className="shrink-0 px-1.5 text-xs text-zinc-400 opacity-0 hover:text-red-600 group-hover:opacity-100 disabled:opacity-100"
              >
                {deleting === p.id ? "…" : "✕"}
              </button>
            </div>
          ))}
      </div>
    </aside>
  );
}
