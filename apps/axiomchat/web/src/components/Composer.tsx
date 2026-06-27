import { useLayoutEffect, useRef, useState } from "react";
import type { PublicState, User } from "../types";
import { dmPartner, namedChannels, dmChannels } from "../selectors";
import { actions } from "../store";
import { navigate } from "../router";
import { Avatar } from "./Avatar";

const MAX_HEIGHT = 200;
const MENU_LIMIT = 6;

interface MentionToken {
  start: number;
  query: string;
}

/**
 * Message composer. Reused by the channel/DM view (with channel-select) and the
 * thread reply pane (with reply-* testids, no channel select). Supports
 * @-mention autocomplete, auto-grow, Enter-to-send / Shift+Enter newline.
 */
export function Composer({
  state,
  channelId,
  threadRootId,
  inputTestId,
  sendTestId,
  placeholder,
  showChannelSelect = false,
  autoFocus = false,
  onSent,
}: {
  state: PublicState;
  channelId: string;
  threadRootId?: string;
  inputTestId: string;
  sendTestId: string;
  placeholder: string;
  showChannelSelect?: boolean;
  autoFocus?: boolean;
  onSent?: () => void;
}) {
  const [text, setText] = useState("");
  const [mention, setMention] = useState<MentionToken | null>(null);
  const [menuIndex, setMenuIndex] = useState(0);
  const ref = useRef<HTMLTextAreaElement>(null);

  const candidates = mention ? matchUsers(state.users, mention.query) : [];

  // Auto-grow the textarea to fit its content (capped).
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [text]);

  function refreshMention(value: string, caret: number): void {
    const upto = value.slice(0, caret);
    const m = /(^|\s)@([a-z0-9_]*)$/i.exec(upto);
    if (m) {
      setMention({ start: caret - m[2].length - 1, query: m[2] });
      setMenuIndex(0);
    } else {
      setMention(null);
    }
  }

  function onChange(e: React.ChangeEvent<HTMLTextAreaElement>): void {
    setText(e.target.value);
    refreshMention(e.target.value, e.target.selectionStart ?? e.target.value.length);
  }

  function acceptMention(user: User): void {
    if (!mention) return;
    const before = text.slice(0, mention.start);
    const after = text.slice(mention.start + 1 + mention.query.length);
    const next = `${before}@${user.handle} ${after}`;
    setText(next);
    setMention(null);
    const caret = mention.start + user.handle.length + 2;
    requestAnimationFrame(() => {
      const el = ref.current;
      if (el) {
        el.focus();
        el.setSelectionRange(caret, caret);
      }
    });
  }

  async function send(): Promise<void> {
    const body = text.trim();
    if (!body) return;
    setText("");
    setMention(null);
    await actions.post({ channelId, text: body, threadRootId });
    onSent?.();
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>): void {
    if (mention && candidates.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setMenuIndex((i) => (i + 1) % candidates.length);
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setMenuIndex((i) => (i - 1 + candidates.length) % candidates.length);
        return;
      }
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        acceptMention(candidates[menuIndex]);
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setMention(null);
        return;
      }
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <div className="px-4 pb-4 pt-1">
      {showChannelSelect && <ChannelSelect state={state} channelId={channelId} />}
      <div className="relative rounded-lg border border-line bg-base focus-within:border-accent/60 focus-within:shadow-focus">
        {mention && candidates.length > 0 && (
          <MentionMenu
            candidates={candidates}
            index={menuIndex}
            onPick={acceptMention}
            onHover={setMenuIndex}
          />
        )}
        <textarea
          ref={ref}
          data-testid={inputTestId}
          rows={1}
          autoFocus={autoFocus}
          value={text}
          onChange={onChange}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          aria-label={placeholder}
          className="block max-h-[200px] w-full resize-none bg-transparent px-3 py-2.5 text-[15px] text-ink placeholder:text-ink-3 focus:outline-none"
        />
        <div className="flex items-center justify-between px-2 pb-2">
          <span className="px-1 text-[11px] text-ink-3">
            <kbd className="font-sans font-semibold">Enter</kbd> to send ·{" "}
            <kbd className="font-sans font-semibold">Shift+Enter</kbd> for newline
          </span>
          <button
            type="button"
            data-testid={sendTestId}
            onClick={() => void send()}
            disabled={text.trim().length === 0}
            className="flex items-center gap-1 rounded-md bg-accent px-3 py-1 text-sm font-semibold text-white transition enabled:hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
          >
            <SendIcon />
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

function ChannelSelect({ state, channelId }: { state: PublicState; channelId: string }) {
  const channels = namedChannels(state);
  const dms = dmChannels(state);
  return (
    <div className="mb-1.5 flex items-center gap-2 px-1 text-xs text-ink-3">
      <span>Posting to</span>
      <select
        data-testid="channel-select"
        aria-label="Select channel to post to"
        value={channelId}
        onChange={(e) => {
          const id = e.target.value;
          navigate(
            id.startsWith("dm_") ? { kind: "dm", id } : { kind: "channel", id },
          );
        }}
        className="rounded border border-line bg-base px-2 py-1 text-xs text-ink-2 focus:border-accent/60 focus:outline-none"
      >
        <optgroup label="Channels">
          {channels.map((c) => (
            <option key={c.id} value={c.id}>
              #{c.name}
            </option>
          ))}
        </optgroup>
        <optgroup label="Direct messages">
          {dms.map((c) => {
            const p = dmPartner(state, c);
            return (
              <option key={c.id} value={c.id}>
                {p?.name ?? c.name}
              </option>
            );
          })}
        </optgroup>
      </select>
    </div>
  );
}

function MentionMenu({
  candidates,
  index,
  onPick,
  onHover,
}: {
  candidates: User[];
  index: number;
  onPick: (u: User) => void;
  onHover: (i: number) => void;
}) {
  return (
    <div
      data-testid="mention-menu"
      role="listbox"
      className="absolute bottom-full left-0 z-10 mb-2 w-72 overflow-hidden rounded-lg border border-line bg-overlay py-1 shadow-popover"
    >
      <div className="px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-ink-3">
        Members
      </div>
      {candidates.map((u, i) => (
        <button
          key={u.id}
          type="button"
          data-testid={`mention-${u.id}`}
          role="option"
          aria-selected={i === index}
          onMouseEnter={() => onHover(i)}
          onMouseDown={(e) => {
            e.preventDefault();
            onPick(u);
          }}
          className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm transition ${
            i === index ? "bg-accent/90 text-white" : "text-ink-2 hover:bg-raised"
          }`}
        >
          <Avatar user={u} size="sm" />
          <span className="font-medium">{u.name}</span>
          <span className={i === index ? "text-white/80" : "text-ink-3"}>@{u.handle}</span>
        </button>
      ))}
    </div>
  );
}

function matchUsers(users: User[], query: string): User[] {
  const q = query.toLowerCase();
  return users
    .filter((u) => u.handle.toLowerCase().includes(q) || u.name.toLowerCase().includes(q))
    .sort((a, b) => a.handle.localeCompare(b.handle))
    .slice(0, MENU_LIMIT);
}

function SendIcon() {
  return (
    <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="currentColor" aria-hidden="true">
      <path d="M1.6 2.2 14.5 8 1.6 13.8 3 8.8l6-0.8-6-0.8z" />
    </svg>
  );
}
