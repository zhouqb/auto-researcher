"use client";

import { useCallback, useEffect, useState } from "react";
import dynamic from "next/dynamic";
import Markdown from "./Markdown";
import {
  api,
  ArtifactContent,
  ArtifactMeta,
  LineageGraph,
} from "@/lib/api";

const VegaEmbed = dynamic(
  () => import("react-vega").then((m) => m.VegaEmbed),
  { ssr: false },
) as React.ComponentType<{ spec: object; options?: { actions?: boolean } }>;

function Viewer({ content }: { content: ArtifactContent }) {
  if (content.base64 && content.mime_type?.startsWith("image/")) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={`data:${content.mime_type};base64,${content.base64}`}
        alt={content.path}
        className="max-w-full rounded"
      />
    );
  }
  if (content.text === undefined) return <p>Binary artifact.</p>;
  if (content.path.endsWith(".vega.json")) {
    try {
      return (
        <VegaEmbed spec={JSON.parse(content.text)} options={{ actions: false }} />
      );
    } catch {
      /* fall through to raw */
    }
  }
  if (content.path.endsWith(".json")) {
    return (
      <pre className="overflow-x-auto rounded bg-zinc-100 p-2 text-xs dark:bg-zinc-900">
        {content.text}
      </pre>
    );
  }
  return <Markdown>{content.text}</Markdown>;
}

export default function ArtifactsPane({ projectId }: { projectId: string }) {
  const [artifacts, setArtifacts] = useState<ArtifactMeta[]>([]);
  const [selected, setSelected] = useState<ArtifactMeta | null>(null);
  const [content, setContent] = useState<ArtifactContent | null>(null);
  const [lineage, setLineage] = useState<LineageGraph | null>(null);

  useEffect(() => {
    const load = () => api.artifacts(projectId).then(setArtifacts).catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [projectId]);

  const open = useCallback(
    async (art: ArtifactMeta, version?: number) => {
      setSelected(art);
      setContent(await api.artifactContent(projectId, art.path, version));
      setLineage(await api.lineage(projectId, art.id));
    },
    [projectId],
  );

  if (selected && content) {
    const related = lineage
      ? lineage.edges
          .map((e) => {
            const otherId = e.child === selected.id ? e.parent : e.child;
            const node = lineage.nodes[otherId];
            const direction = e.child === selected.id ? "→" : "←";
            return node ? { node, label: `${direction} ${e.relation}` } : null;
          })
          .filter((x): x is NonNullable<typeof x> => x !== null)
      : [];
    return (
      <div className="space-y-3 text-sm">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setSelected(null)}
            className="rounded bg-zinc-200 px-2 py-0.5 text-xs hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600"
          >
            ← All artifacts
          </button>
          <code className="text-xs">{content.path}</code>
          {content.versions.length > 1 && (
            <select
              className="ml-auto rounded border border-zinc-300 bg-transparent px-1 py-0.5 text-xs dark:border-zinc-600"
              value={content.version}
              onChange={(e) => open(selected, Number(e.target.value))}
            >
              {content.versions.map((v) => (
                <option key={v} value={v}>
                  v{v}
                </option>
              ))}
            </select>
          )}
        </div>
        {related.length > 0 && (
          <div className="rounded border border-zinc-200 p-2 text-xs dark:border-zinc-700">
            <span className="font-semibold">Lineage:</span>{" "}
            {related.map(({ node, label }, i) => {
              const target = artifacts.find((a) => a.id === node.id);
              return (
                <button
                  key={i}
                  disabled={!target}
                  onClick={() => target && open(target)}
                  className="mr-2 text-blue-600 hover:underline disabled:text-zinc-400 dark:text-blue-400"
                >
                  {label} {node.path}@v{node.version}
                </button>
              );
            })}
          </div>
        )}
        <Viewer content={content} />
      </div>
    );
  }

  return (
    <div className="space-y-1.5 text-sm">
      {artifacts.length === 0 && (
        <p className="text-zinc-500">No artifacts yet.</p>
      )}
      {artifacts.map((a) => (
        <button
          key={a.id}
          onClick={() => open(a)}
          className="block w-full rounded-md border border-zinc-200 p-2 text-left hover:bg-zinc-50 dark:border-zinc-700 dark:hover:bg-zinc-900"
        >
          <div className="flex items-center gap-2">
            <span className="rounded bg-zinc-200 px-1 py-0.5 font-mono text-[10px] dark:bg-zinc-700">
              {a.kind}
            </span>
            <span className="truncate text-xs font-medium">
              {a.title ?? a.path}
            </span>
            <span className="ml-auto text-[10px] text-zinc-500">v{a.version}</span>
          </div>
          {a.summary && (
            <p className="mt-0.5 truncate text-xs text-zinc-500">{a.summary}</p>
          )}
        </button>
      ))}
    </div>
  );
}
