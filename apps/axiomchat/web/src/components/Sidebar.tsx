import { useMemo } from "react";
import type { Channel, PublicState } from "../types";
import type { Route } from "../router";
import {
  dmChannels,
  dmPartner,
  hasUnreadMention,
  namedChannels,
  unreadCount,
} from "../selectors";
import { Avatar } from "./Avatar";

export function Sidebar({ state, route }: { state: PublicState; route: Route | null }) {
  const channels = useMemo(() => namedChannels(state), [state]);
  const dms = useMemo(() => dmChannels(state), [state]);
  const activeId = route && "id" in route ? route.id : null;

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-line bg-sidebar">
      <WorkspaceHeader name={state.workspace.name} />

      <nav className="flex-1 overflow-y-auto px-2 pb-4" aria-label="Channels and direct messages">
        <a
          href="#/search"
          data-testid="search-link"
          className="mb-2 mt-1 flex items-center gap-2 rounded-md px-2 py-1.5 text-sm text-ink-2 transition hover:bg-raised"
        >
          <SearchIcon />
          <span>Search messages…</span>
        </a>

        <SectionHeader label="Channels" actionTestId="add-channel" actionLabel="Add channel" />
        <ul data-testid="channel-list" className="mb-4 space-y-px" role="list">
          {channels.map((c) => (
            <ChannelLink
              key={c.id}
              channel={c}
              active={activeId === c.id}
              unread={unreadCount(state, c)}
              mention={hasUnreadMention(state, c)}
            />
          ))}
        </ul>

        <SectionHeader label="Direct Messages" />
        <ul className="space-y-px" role="list">
          {dms.map((c) => {
            const partner = dmPartner(state, c);
            if (!partner) return null;
            return (
              <li key={c.id}>
                <a
                  href={`#/dm/${c.id}`}
                  data-testid={`dm-link-${c.id}`}
                  aria-current={activeId === c.id ? "page" : undefined}
                  className={rowClass(activeId === c.id)}
                >
                  <Avatar user={partner} size="sm" presence dotRing="border-sidebar" />
                  <span className="truncate">{partner.name}</span>
                  <UnreadBadge channelId={c.id} count={unreadCount(state, c)} />
                </a>
              </li>
            );
          })}
        </ul>
      </nav>
    </aside>
  );
}

function WorkspaceHeader({ name }: { name: string }) {
  return (
    <header
      data-testid="workspace-header"
      className="flex items-center justify-between border-b border-line px-4 py-3"
    >
      <div className="min-w-0">
        <div className="truncate text-[15px] font-bold text-ink">{name}</div>
        <div className="mt-0.5 flex items-center gap-1.5 text-xs text-ink-3">
          <span className="h-2 w-2 rounded-full bg-presence-active" />
          <span>Jordan Lee</span>
        </div>
      </div>
      <button
        type="button"
        aria-label="New message"
        className="grid h-7 w-7 place-items-center rounded-md text-ink-2 transition hover:bg-raised hover:text-ink"
      >
        <PencilIcon />
      </button>
    </header>
  );
}

function SectionHeader({
  label,
  actionTestId,
  actionLabel,
}: {
  label: string;
  actionTestId?: string;
  actionLabel?: string;
}) {
  return (
    <div className="group/section mb-1 mt-3 flex items-center justify-between px-2">
      <span className="text-xs font-semibold uppercase tracking-wide text-ink-3">{label}</span>
      {actionTestId && (
        <button
          type="button"
          data-testid={actionTestId}
          aria-label={actionLabel}
          className="grid h-5 w-5 place-items-center rounded text-ink-3 transition hover:bg-raised hover:text-ink"
        >
          <PlusIcon />
        </button>
      )}
    </div>
  );
}

function ChannelLink({
  channel,
  active,
  unread,
  mention,
}: {
  channel: Channel;
  active: boolean;
  unread: number;
  mention: boolean;
}) {
  const bold = unread > 0 && !active;
  return (
    <li>
      <a
        href={`#/channel/${channel.id}`}
        data-testid={`channel-link-${channel.id}`}
        aria-current={active ? "page" : undefined}
        className={rowClass(active)}
      >
        <span className={`shrink-0 text-ink-3 ${active ? "text-ink" : ""}`}>
          {channel.kind === "private" ? <LockIcon /> : <HashIcon />}
        </span>
        <span className={`truncate ${bold ? "font-semibold text-ink" : ""}`}>{channel.name}</span>
        {mention && (
          <span className="ml-auto h-1.5 w-1.5 shrink-0 rounded-full bg-danger" aria-label="mention" />
        )}
        <UnreadBadge channelId={channel.id} count={unread} pushRight={!mention} />
      </a>
    </li>
  );
}

function UnreadBadge({
  channelId,
  count,
  pushRight = true,
}: {
  channelId: string;
  count: number;
  pushRight?: boolean;
}) {
  if (count <= 0) return null;
  return (
    <span
      data-testid={`unread-${channelId}`}
      className={`${pushRight ? "ml-auto" : ""} grid h-[18px] min-w-[18px] shrink-0 place-items-center rounded-full bg-danger px-1 text-[11px] font-bold text-white`}
    >
      {count}
    </span>
  );
}

function rowClass(active: boolean): string {
  return [
    "flex items-center gap-2 rounded-md px-2 py-1 text-sm transition",
    active
      ? "bg-accent/90 text-white"
      : "text-ink-2 hover:bg-raised hover:text-ink",
  ].join(" ");
}

// --- inline icons (no network deps) ---------------------------------------

function HashIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="currentColor" aria-hidden="true">
      <path d="M6.4 2 5.9 5H3v1.5h2.6l-.5 3H2v1.5h2.8L4.3 14h1.5l.5-2.5h3L8.8 14h1.5l.5-2.5H14V10h-2.4l.5-3H15V5.5h-2.6L13 2h-1.5l-.6 3h-3L8.5 2H7l-.6 3h-3z" />
    </svg>
  );
}
function LockIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="currentColor" aria-hidden="true">
      <path d="M8 1a3 3 0 0 0-3 3v2H4.5A1.5 1.5 0 0 0 3 7.5v5A1.5 1.5 0 0 0 4.5 14h7a1.5 1.5 0 0 0 1.5-1.5v-5A1.5 1.5 0 0 0 11.5 6H11V4a3 3 0 0 0-3-3Zm1.5 5h-3V4a1.5 1.5 0 0 1 3 0v2Z" />
    </svg>
  );
}
function SearchIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
      <circle cx="7" cy="7" r="4.5" />
      <path d="m11 11 3 3" strokeLinecap="round" />
    </svg>
  );
}
function PlusIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="currentColor" aria-hidden="true">
      <path d="M7.25 2.5h1.5V7.25H13.5v1.5H8.75V13.5h-1.5V8.75H2.5v-1.5h4.75z" />
    </svg>
  );
}
function PencilIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor" aria-hidden="true">
      <path d="M11.7 1.5a1.2 1.2 0 0 1 1.7 0l1.1 1.1a1.2 1.2 0 0 1 0 1.7l-7.7 7.7-3 .8.8-3 7.8-7.8Zm-7 9.4-.4 1.5 1.5-.4 6.6-6.6-1.1-1.1-6.6 6.6Z" />
    </svg>
  );
}
