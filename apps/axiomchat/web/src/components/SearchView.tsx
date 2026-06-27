import { useState } from "react";
import type { Message, PublicState } from "../types";
import { api } from "../api";
import { channelById, userById } from "../selectors";
import { navigate } from "../router";
import { formatShortDate } from "../util/time";
import { renderMessageText } from "../util/text";
import { Avatar } from "./Avatar";

export function SearchView({ state }: { state: PublicState }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Message[] | null>(null);
  const [loading, setLoading] = useState(false);

  async function run(): Promise<void> {
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    try {
      setResults(await api.search(q));
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <header className="border-b border-line px-5 py-3">
        <h1 className="text-[17px] font-bold text-ink">Search</h1>
      </header>

      <div className="border-b border-line px-5 py-3">
        <div className="flex gap-2">
          <input
            data-testid="search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void run();
            }}
            placeholder="Search messages…"
            aria-label="Search messages"
            className="flex-1 rounded-md border border-line bg-base px-3 py-2 text-sm text-ink placeholder:text-ink-3 focus:border-accent/60 focus:outline-none"
          />
          <button
            type="button"
            data-testid="search-button"
            onClick={() => void run()}
            className="rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white transition hover:bg-accent-hover"
          >
            Search
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-3">
        {loading && <p className="text-sm text-ink-3">Searching…</p>}
        {!loading && results !== null && (
          <p className="mb-3 text-xs text-ink-3">
            {results.length} {results.length === 1 ? "result" : "results"}
          </p>
        )}
        {!loading &&
          results?.map((m) => {
            const author = userById(state, m.authorId);
            const channel = channelById(state, m.channelId);
            return (
              <button
                key={m.id}
                type="button"
                data-testid={`search-result-${m.id}`}
                onClick={() =>
                  navigate(
                    m.channelId.startsWith("dm_")
                      ? { kind: "dm", id: m.channelId }
                      : { kind: "channel", id: m.channelId },
                  )
                }
                className="mb-2 block w-full rounded-lg border border-line bg-base p-3 text-left transition hover:border-accent/50 hover:bg-raised"
              >
                <div className="mb-1 flex items-center gap-2 text-xs text-ink-3">
                  <span className="font-semibold text-ink-2">
                    {channel?.kind === "dm" ? "DM" : `#${channel?.name ?? "?"}`}
                  </span>
                  <span>·</span>
                  <span>{formatShortDate(m.ts)}</span>
                </div>
                <div className="flex items-start gap-2">
                  {author && <Avatar user={author} size="sm" />}
                  <div className="min-w-0">
                    <div className="text-sm font-semibold text-ink">{author?.name ?? "Unknown"}</div>
                    <div className="break-words text-sm text-ink-2">
                      {renderMessageText(m.text, state.users, state.currentUserId)}
                    </div>
                  </div>
                </div>
              </button>
            );
          })}
        {!loading && results !== null && results.length === 0 && (
          <div className="mt-10 text-center text-sm text-ink-3">
            No messages match “{query}”.
          </div>
        )}
        {!loading && results === null && (
          <div className="mt-10 text-center text-sm text-ink-3">
            Search across every channel and DM in the workspace.
          </div>
        )}
      </div>
    </>
  );
}
