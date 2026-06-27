import type { Message, PublicState, User } from "../types";
import { threadParticipants, threadReplyCount, userById } from "../selectors";
import { actions } from "../store";
import { formatClock } from "../util/time";
import { renderMessageText } from "../util/text";
import { Avatar } from "./Avatar";

const QUICK_REACT = "👍";

export function MessageRow({
  state,
  message,
  grouped,
  onOpenThread,
  active = false,
  showThreadFooter = true,
}: {
  state: PublicState;
  message: Message;
  grouped: boolean;
  onOpenThread?: (rootId: string) => void;
  active?: boolean;
  showThreadFooter?: boolean;
}) {
  const author = userById(state, message.authorId);
  if (!author) return null;
  const replyCount = showThreadFooter ? threadReplyCount(state, message.id) : 0;

  return (
    <div
      data-testid={`message-${message.id}`}
      className={`group relative flex gap-2 px-5 ${grouped ? "py-0.5" : "mt-2 py-0.5"} ${
        active ? "bg-accent-soft" : "hover:bg-raised/60"
      } transition`}
    >
      {/* left gutter: avatar (ungrouped) or hover timestamp (grouped) */}
      <div className="w-10 shrink-0 pt-0.5">
        {grouped ? (
          <span className="row-actions block pt-1 text-right text-[10px] leading-4 text-ink-3">
            {formatClock(message.ts).replace(/ [AP]M$/, "")}
          </span>
        ) : (
          <Avatar user={author} size="md" />
        )}
      </div>

      <div className="min-w-0 flex-1">
        {!grouped && (
          <div className="flex items-baseline gap-2">
            <span
              data-testid={`message-author-${message.id}`}
              className="text-sm font-semibold text-ink"
            >
              {author.name}
            </span>
            {author.role !== "member" && <RoleTag role={author.role} />}
            <span className="text-xs text-ink-3">{formatClock(message.ts)}</span>
          </div>
        )}

        {message.pinned && <PinnedTag />}

        <div
          data-testid={`message-text-${message.id}`}
          className="whitespace-pre-wrap break-words text-[15px] leading-[22px] text-ink-2"
        >
          {renderMessageText(message.text, state.users, state.currentUserId)}
          {message.resolved && <ResolvedTag />}
        </div>

        <ReactionPills state={state} message={message} />

        {replyCount > 0 && onOpenThread && (
          <ThreadFooter
            message={message}
            participants={threadParticipants(state, message.id)}
            count={replyCount}
            onOpen={() => onOpenThread(message.id)}
          />
        )}
      </div>

      <RowActions message={message} onOpenThread={onOpenThread} />
    </div>
  );
}

function ReactionPills({ state, message }: { state: PublicState; message: Message }) {
  const entries = Object.entries(message.reactions).filter(([, ids]) => ids.length > 0);
  if (entries.length === 0) return null;
  return (
    <div className="mt-1 flex flex-wrap gap-1">
      {entries.map(([emoji, ids]) => {
        const mine = ids.includes(state.currentUserId);
        return (
          <button
            key={emoji}
            type="button"
            data-testid={`reaction-${message.id}-${emoji}`}
            onClick={() => void actions.react(message.id, emoji)}
            title={ids
              .map((id) => userById(state, id)?.name ?? id)
              .join(", ")}
            className={`flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs transition ${
              mine
                ? "border-accent/60 bg-accent-soft text-accent-hover"
                : "border-line bg-raised text-ink-2 hover:border-line/80"
            }`}
          >
            <span>{emoji}</span>
            <span className="font-medium tabular-nums">{ids.length}</span>
          </button>
        );
      })}
    </div>
  );
}

function ThreadFooter({
  message,
  participants,
  count,
  onOpen,
}: {
  message: Message;
  participants: User[];
  count: number;
  onOpen: () => void;
}) {
  return (
    <button
      type="button"
      data-testid={`thread-count-${message.id}`}
      onClick={onOpen}
      className="mt-1 flex items-center gap-2 rounded-md border border-transparent px-1 py-0.5 text-xs text-accent-hover transition hover:border-line hover:bg-raised"
    >
      <span className="flex -space-x-1">
        {participants.slice(0, 3).map((u) => (
          <Avatar key={u.id} user={u} size="sm" dotRing="border-panel" />
        ))}
      </span>
      <span className="font-semibold">
        {count} {count === 1 ? "reply" : "replies"}
      </span>
      <span className="text-ink-3">View thread</span>
    </button>
  );
}

function RowActions({
  message,
  onOpenThread,
}: {
  message: Message;
  onOpenThread?: (rootId: string) => void;
}) {
  return (
    <div className="row-actions absolute -top-3 right-4 flex items-center gap-0.5 rounded-md border border-line bg-overlay p-0.5 shadow-popover">
      <ActionButton
        testId={`react-${message.id}`}
        label="Add reaction"
        onClick={() => void actions.react(message.id, QUICK_REACT)}
      >
        <EmojiIcon />
      </ActionButton>
      {onOpenThread && (
        <ActionButton
          testId={`thread-open-${message.id}`}
          label="Reply in thread"
          onClick={() => onOpenThread(message.threadRootId ?? message.id)}
        >
          <ThreadIcon />
        </ActionButton>
      )}
      <ActionButton
        testId={`pin-${message.id}`}
        label={message.pinned ? "Unpin message" : "Pin message"}
        onClick={() => void actions.pin(message.id)}
        active={message.pinned}
      >
        <PinIcon />
      </ActionButton>
    </div>
  );
}

function ActionButton({
  testId,
  label,
  onClick,
  active = false,
  children,
}: {
  testId: string;
  label: string;
  onClick: () => void;
  active?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      data-testid={testId}
      aria-label={label}
      title={label}
      onClick={onClick}
      className={`grid h-7 w-7 place-items-center rounded transition hover:bg-raised ${
        active ? "text-accent-hover" : "text-ink-2 hover:text-ink"
      }`}
    >
      {children}
    </button>
  );
}

function RoleTag({ role }: { role: User["role"] }) {
  return (
    <span className="rounded bg-raised px-1 text-[10px] font-medium uppercase tracking-wide text-ink-3">
      {role}
    </span>
  );
}
function PinnedTag() {
  return (
    <div className="mb-0.5 flex items-center gap-1 text-[11px] font-medium text-warning">
      <PinIcon />
      <span>Pinned</span>
    </div>
  );
}
function ResolvedTag() {
  return (
    <span className="ml-2 inline-flex items-center gap-1 rounded bg-success/15 px-1.5 py-0.5 align-middle text-[11px] font-medium text-success">
      ✓ Resolved
    </span>
  );
}

function EmojiIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
      <circle cx="8" cy="8" r="6.2" />
      <path d="M5.8 6.4h.01M10.2 6.4h.01M5.5 9.5c.7.8 1.5 1.2 2.5 1.2s1.8-.4 2.5-1.2" strokeLinecap="round" />
    </svg>
  );
}
function ThreadIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
      <path d="M2.5 4.5h11v6h-5l-3 2.5v-2.5h-3z" strokeLinejoin="round" />
    </svg>
  );
}
function PinIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="currentColor" aria-hidden="true">
      <path d="M9.4 1.6a1 1 0 0 0-1.7.7v.3L4.9 6H3a1 1 0 0 0-.7 1.7L5 10.4 1.5 14.5l4.1-3.5 2.7 2.7A1 1 0 0 0 10 13V11l3.4-2.8h.3a1 1 0 0 0 .7-1.7l-5-5z" />
    </svg>
  );
}
