/**
 * Pure, deterministic selectors over PublicState. The render layer derives
 * everything it shows through these — no Date.now(), no Math.random(), no
 * hidden client state. Given the same PublicState they always return the same
 * result, which is what keeps screenshots reproducible.
 */
import type { Channel, Message, PublicState, User } from "./types";

export function userById(state: PublicState, id: string): User | undefined {
  return state.users.find((u) => u.id === id);
}

export function currentUser(state: PublicState): User | undefined {
  return userById(state, state.currentUserId);
}

export function channelById(state: PublicState, id: string): Channel | undefined {
  return state.channels.find((c) => c.id === id);
}

/** Named channels (public + private), alphabetical — Slack's default order. */
export function namedChannels(state: PublicState): Channel[] {
  return state.channels
    .filter((c) => c.kind !== "dm")
    .slice()
    .sort((a, b) => a.name.localeCompare(b.name));
}

/** DM conversations, most-recently-active first. */
export function dmChannels(state: PublicState): Channel[] {
  const lastTs = (c: Channel): number =>
    state.messages
      .filter((m) => m.channelId === c.id)
      .reduce((mx, m) => Math.max(mx, m.ts), 0);
  return state.channels
    .filter((c) => c.kind === "dm")
    .slice()
    .sort((a, b) => lastTs(b) - lastTs(a));
}

/** The other participant in a DM (not the current user). */
export function dmPartner(state: PublicState, channel: Channel): User | undefined {
  const otherId = channel.memberIds.find((id) => id !== state.currentUserId);
  return otherId ? userById(state, otherId) : undefined;
}

/** Top-level messages in a channel (thread replies excluded), oldest first. */
export function channelMessages(state: PublicState, channelId: string): Message[] {
  return state.messages
    .filter((m) => m.channelId === channelId && !m.threadRootId)
    .sort((a, b) => a.ts - b.ts);
}

/** Replies belonging to a thread root, oldest first. */
export function threadReplies(state: PublicState, rootId: string): Message[] {
  return state.messages
    .filter((m) => m.threadRootId === rootId)
    .sort((a, b) => a.ts - b.ts);
}

export function threadReplyCount(state: PublicState, rootId: string): number {
  return state.messages.reduce((n, m) => (m.threadRootId === rootId ? n + 1 : n), 0);
}

/** The participants who have replied in a thread (for stacked avatars). */
export function threadParticipants(state: PublicState, rootId: string): User[] {
  const ids: string[] = [];
  for (const m of threadReplies(state, rootId)) {
    if (!ids.includes(m.authorId)) ids.push(m.authorId);
  }
  return ids.map((id) => userById(state, id)).filter((u): u is User => Boolean(u));
}

/** Unread count for a channel: top-level messages after lastReadTs not authored by me. */
export function unreadCount(state: PublicState, channel: Channel): number {
  return state.messages.reduce((n, m) => {
    const counts =
      m.channelId === channel.id &&
      !m.threadRootId &&
      m.ts > channel.lastReadTs &&
      m.authorId !== state.currentUserId;
    return counts ? n + 1 : n;
  }, 0);
}

/** Whether a channel has any mention of the current user in its unread tail. */
export function hasUnreadMention(state: PublicState, channel: Channel): boolean {
  return state.messages.some(
    (m) =>
      m.channelId === channel.id &&
      m.ts > channel.lastReadTs &&
      m.mentions.includes(state.currentUserId),
  );
}

export function pinnedMessages(state: PublicState, channelId: string): Message[] {
  return state.messages
    .filter((m) => m.channelId === channelId && m.pinned)
    .sort((a, b) => a.ts - b.ts);
}
