/**
 * Public domain types — mirrors the server's PublicState (apps/axiomchat/src/types.ts)
 * with all hidden `_` ground-truth fields removed. The SPA only ever sees these.
 */

export type ChannelKind = "public" | "private" | "dm";
export type UserStatus = "active" | "away" | "dnd" | "offline";
export type UserRole = "owner" | "admin" | "member" | "guest";
export type Reactions = Record<string, string[]>;

export interface User {
  id: string;
  name: string;
  handle: string;
  role: UserRole;
  status: UserStatus;
  title?: string;
}

export interface Channel {
  id: string;
  name: string;
  kind: ChannelKind;
  topic: string;
  memberIds: string[];
  lastReadTs: number;
}

export interface Message {
  id: string;
  channelId: string;
  threadRootId?: string;
  authorId: string;
  text: string;
  ts: number;
  mentions: string[];
  reactions: Reactions;
  pinned: boolean;
  resolved: boolean;
}

export interface PublicState {
  workspace: { id: string; name: string; seed: number; scale: string };
  currentUserId: string;
  users: User[];
  channels: Channel[];
  messages: Message[];
}

export interface PostInput {
  channelId: string;
  text: string;
  threadRootId?: string;
  authorId?: string;
}

export type PatchOp =
  | { op: "react"; emoji: string; userId?: string }
  | { op: "pin" }
  | { op: "resolve"; value?: boolean }
  | { op: "edit"; text: string };
