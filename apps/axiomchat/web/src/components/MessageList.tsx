import { useLayoutEffect, useMemo, useRef } from "react";
import type { Channel, Message, PublicState } from "../types";
import { channelMessages } from "../selectors";
import { dayKey, formatDayDivider } from "../util/time";
import { MessageRow } from "./MessageRow";

const GROUP_WINDOW_MS = 5 * 60_000; // group consecutive msgs from same author within 5 min

export function MessageList({
  state,
  channel,
  onOpenThread,
  activeThreadId,
}: {
  state: PublicState;
  channel: Channel;
  onOpenThread: (rootId: string) => void;
  activeThreadId: string | null;
}) {
  const messages = useMemo(() => channelMessages(state, channel.id), [state, channel.id]);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Index of the first unread top-level message (drives the new-messages divider).
  const firstUnread = useMemo(
    () => messages.findIndex((m) => m.ts > channel.lastReadTs && m.authorId !== state.currentUserId),
    [messages, channel.lastReadTs, state.currentUserId],
  );

  // Pin the view to the newest message on channel switch / new post — Slack's
  // bottom-anchored log. Deterministic given identical content.
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [channel.id, messages.length]);

  return (
    <div ref={scrollRef} data-testid="message-list" className="flex-1 overflow-y-auto py-3" role="log">
      {messages.length === 0 && <EmptyChannel name={channel.name} />}
      {messages.map((m, i) => {
        const prev = messages[i - 1];
        const newDay = !prev || dayKey(prev.ts) !== dayKey(m.ts);
        const grouped = isGrouped(prev, m) && !newDay && i !== firstUnread;
        return (
          <div key={m.id}>
            {newDay && <DayDivider ts={m.ts} />}
            {i === firstUnread && <NewMessagesDivider />}
            <MessageRow
              state={state}
              message={m}
              grouped={grouped}
              onOpenThread={onOpenThread}
              active={activeThreadId === m.id}
            />
          </div>
        );
      })}
    </div>
  );
}

function isGrouped(prev: Message | undefined, cur: Message): boolean {
  return (
    !!prev &&
    prev.authorId === cur.authorId &&
    cur.ts - prev.ts < GROUP_WINDOW_MS &&
    !cur.pinned
  );
}

function DayDivider({ ts }: { ts: number }) {
  return (
    <div className="relative my-3 flex items-center justify-center" role="separator">
      <span className="absolute inset-x-5 top-1/2 h-px -translate-y-1/2 bg-line" />
      <span className="relative rounded-full border border-line bg-panel px-3 py-0.5 text-xs font-semibold text-ink-2">
        {formatDayDivider(ts)}
      </span>
    </div>
  );
}

function NewMessagesDivider() {
  return (
    <div
      data-testid="new-messages-divider"
      className="relative my-2 flex items-center"
      role="separator"
      aria-label="New messages"
    >
      <span className="h-px flex-1 bg-danger/60" />
      <span className="px-3 text-[11px] font-bold uppercase tracking-wide text-danger">New</span>
    </div>
  );
}

function EmptyChannel({ name }: { name: string }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-ink-3">
      <div className="grid h-12 w-12 place-items-center rounded-full bg-raised text-xl">#</div>
      <div className="text-sm">
        This is the start of <span className="font-semibold text-ink-2">#{name}</span>.
      </div>
      <div className="text-xs">No messages yet — say hello 👋</div>
    </div>
  );
}
