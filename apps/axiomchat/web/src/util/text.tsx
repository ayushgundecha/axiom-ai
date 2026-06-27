/** Render message text, highlighting @mentions of known users. */
import type { ReactNode } from "react";
import type { User } from "../types";

/**
 * Split `text` on @handle tokens and wrap mentions of real users in a
 * highlighted pill. The current user's own mentions get a stronger highlight.
 */
export function renderMessageText(
  text: string,
  users: User[],
  currentUserId: string,
): ReactNode[] {
  const byHandle = new Map(users.map((u) => [u.handle.toLowerCase(), u]));
  const parts = text.split(/(@[a-z0-9_]+)/gi);
  return parts.map((part, i) => {
    const m = /^@([a-z0-9_]+)$/i.exec(part);
    const user = m ? byHandle.get(m[1].toLowerCase()) : undefined;
    if (user) {
      const isMe = user.id === currentUserId;
      return (
        <span
          key={i}
          className={
            isMe
              ? "rounded bg-mention px-1 font-medium text-mention-ink"
              : "rounded px-0.5 font-medium text-accent-hover"
          }
        >
          @{user.handle}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}
