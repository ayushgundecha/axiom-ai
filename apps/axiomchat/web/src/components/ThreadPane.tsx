import { useMemo } from "react";
import type { PublicState } from "../types";
import { channelById, threadReplies } from "../selectors";
import { actions } from "../store";
import { MessageRow } from "./MessageRow";
import { Composer } from "./Composer";

export function ThreadPane({
  state,
  rootId,
  onClose,
}: {
  state: PublicState;
  rootId: string;
  onClose: () => void;
}) {
  const root = useMemo(() => state.messages.find((m) => m.id === rootId), [state, rootId]);
  const replies = useMemo(() => threadReplies(state, rootId), [state, rootId]);
  const channel = root ? channelById(state, root.channelId) : undefined;

  if (!root) {
    return (
      <aside data-testid="thread-pane" className="flex w-[380px] shrink-0 flex-col border-l border-line bg-panel">
        <ThreadHeader title="Thread" subtitle="" onClose={onClose} />
        <div className="flex flex-1 items-center justify-center text-sm text-ink-3">
          This thread is no longer available.
        </div>
      </aside>
    );
  }

  return (
    <aside
      data-testid="thread-pane"
      className="flex w-[380px] shrink-0 flex-col border-l border-line bg-panel"
      aria-label="Thread"
    >
      <ThreadHeader
        title="Thread"
        subtitle={channel ? `#${channel.name}` : ""}
        onClose={onClose}
        resolved={root.resolved}
        onResolve={() => void actions.resolve(rootId, !root.resolved)}
        resolveTestId={`resolve-thread-${rootId}`}
      />

      <div className="flex-1 overflow-y-auto py-2">
        <MessageRow state={state} message={root} grouped={false} showThreadFooter={false} />

        <div className="my-2 flex items-center gap-3 px-5 text-xs font-semibold text-ink-3">
          <span>
            {replies.length} {replies.length === 1 ? "reply" : "replies"}
          </span>
          <span className="h-px flex-1 bg-line" />
        </div>

        {replies.map((m, i) => {
          const prev = replies[i - 1];
          const grouped =
            !!prev && prev.authorId === m.authorId && m.ts - prev.ts < 5 * 60_000;
          return (
            <MessageRow key={m.id} state={state} message={m} grouped={grouped} showThreadFooter={false} />
          );
        })}

        {replies.length === 0 && (
          <p className="px-5 py-3 text-sm text-ink-3">No replies yet. Start the conversation.</p>
        )}
      </div>

      <Composer
        state={state}
        channelId={root.channelId}
        threadRootId={rootId}
        inputTestId={`reply-input-${rootId}`}
        sendTestId={`reply-send-${rootId}`}
        placeholder="Reply…"
        autoFocus
      />
    </aside>
  );
}

function ThreadHeader({
  title,
  subtitle,
  onClose,
  resolved,
  onResolve,
  resolveTestId,
}: {
  title: string;
  subtitle: string;
  onClose: () => void;
  resolved?: boolean;
  onResolve?: () => void;
  resolveTestId?: string;
}) {
  return (
    <header className="flex items-center justify-between border-b border-line px-4 py-3">
      <div>
        <div className="text-[15px] font-bold text-ink">{title}</div>
        {subtitle && <div className="text-xs text-ink-3">{subtitle}</div>}
      </div>
      <div className="flex items-center gap-1">
        {onResolve && resolveTestId && (
          <button
            type="button"
            data-testid={resolveTestId}
            onClick={onResolve}
            aria-pressed={resolved}
            className={`flex items-center gap-1 rounded-md border px-2 py-1 text-xs font-medium transition ${
              resolved
                ? "border-success/50 bg-success/15 text-success"
                : "border-line text-ink-2 hover:bg-raised hover:text-ink"
            }`}
          >
            {resolved ? "✓ Resolved" : "Resolve"}
          </button>
        )}
        <button
          type="button"
          aria-label="Close thread"
          onClick={onClose}
          className="grid h-7 w-7 place-items-center rounded-md text-ink-2 transition hover:bg-raised hover:text-ink"
        >
          <CloseIcon />
        </button>
      </div>
    </header>
  );
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
      <path d="m4 4 8 8M12 4l-8 8" strokeLinecap="round" />
    </svg>
  );
}
