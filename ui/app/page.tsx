"use client";

import { useEffect, useState } from "react";
import { CopilotKit, CopilotChat } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";
import Sidebar from "@/components/Sidebar";
import BoardPane from "@/components/BoardPane";
import RunsPane from "@/components/RunsPane";
import ArtifactsPane from "@/components/ArtifactsPane";
import Dashboard from "@/components/Dashboard";
import { api } from "@/lib/api";

const TABS = ["Runs", "Board", "Artifacts"] as const;
type Tab = (typeof TABS)[number];

function ResumeBanner({ projectId }: { projectId: string }) {
  const [resumable, setResumable] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .status(projectId)
      .then((s) => setResumable(s.resumable_invocation))
      .catch(() => setResumable(null));
  }, [projectId]);

  if (!resumable) return null;
  return (
    <div className="flex items-center gap-3 border-b border-amber-300 bg-amber-50 px-4 py-2 text-xs text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200">
      The last run was interrupted before finishing.
      <button
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          setError(null);
          try {
            await api.resume(projectId);
            setResumable(null);
          } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
          } finally {
            setBusy(false);
          }
        }}
        className="rounded bg-amber-600 px-2 py-1 font-medium text-white hover:bg-amber-700 disabled:opacity-50"
      >
        {busy ? "Resuming…" : "▶ Resume"}
      </button>
      {error && <span className="text-red-700 dark:text-red-400">{error}</span>}
    </div>
  );
}

export default function Home() {
  const [projectId, setProjectId] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("Runs");

  return (
    <CopilotKit
      runtimeUrl="/api/copilotkit"
      agent="deep_researcher"
      showDevConsole={false}
    >
      <div className="flex h-screen overflow-hidden">
        <Sidebar projectId={projectId} onSelect={setProjectId} />

        {projectId ? (
          <>
            <main className="flex min-w-0 flex-1 flex-col">
              <header className="border-b border-zinc-200 px-4 py-2 text-sm font-semibold dark:border-zinc-800">
                {projectId}
              </header>
              <ResumeBanner projectId={projectId} />
              <div className="min-h-0 flex-1">
                <CopilotChat
                  key={projectId}
                  agentId="deep_researcher"
                  threadId={projectId}
                  className="h-full"
                />
              </div>
            </main>

            <section className="flex w-[430px] shrink-0 flex-col border-l border-zinc-200 dark:border-zinc-800">
              <nav className="flex border-b border-zinc-200 dark:border-zinc-800">
                {TABS.map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`px-4 py-2 text-xs font-medium ${
                      tab === t
                        ? "border-b-2 border-blue-600 text-blue-700 dark:text-blue-300"
                        : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </nav>
              <div className="min-h-0 flex-1 overflow-y-auto p-3">
                {tab === "Runs" && <RunsPane projectId={projectId} />}
                {tab === "Board" && <BoardPane projectId={projectId} />}
                {tab === "Artifacts" && <ArtifactsPane projectId={projectId} />}
              </div>
            </section>
          </>
        ) : (
          <Dashboard onSelect={setProjectId} />
        )}
      </div>
    </CopilotKit>
  );
}
