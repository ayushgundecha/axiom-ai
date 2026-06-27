/**
 * AxiomChat domain model.
 *
 * Public fields are what the SPA and agents see via `GET /api/state`.
 * Hidden ground-truth fields are prefixed with `_` and live SERVER-SIDE
 * ONLY — `GET /api/state` strips every `_`-prefixed key, while the
 * privileged `GET /api/_oracle/state` exposes them. The hidden labels are
 * the substrate Pillar 2 will turn into rewards; Pillar 1 only populates
 * and gates them, it does not score them.
 */

export type ChannelKind = "public" | "private" | "dm";

/** Presence, drives the sidebar status dots. */
export type UserStatus = "active" | "away" | "dnd" | "offline";

/** Workspace role. */
export type UserRole = "owner" | "admin" | "member" | "guest";

/** Incident severity used by the #incident ground-truth scenario. */
export type Severity = "SEV1" | "SEV2" | "SEV3";

/** emoji -> list of userIds who reacted with it. */
export type Reactions = Record<string, string[]>;

export interface User {
  id: string;
  name: string;
  handle: string;
  role: UserRole;
  status: UserStatus;
  /** Job title — purely cosmetic, shown in hovercards/profile. */
  title?: string;
}

export interface Channel {
  id: string;
  name: string;
  kind: ChannelKind;
  topic: string;
  memberIds: string[];
  /**
   * Timestamp of the last message the current user has "read". Messages with
   * `ts > lastReadTs` are unread — this drives the sidebar badge AND the
   * new-messages divider deterministically (no clocks involved).
   */
  lastReadTs: number;
}

export interface Message {
  id: string;
  channelId: string;
  /** Set on replies; the id of the thread's root message. */
  threadRootId?: string;
  authorId: string;
  text: string;
  /** Epoch ms, derived from the seed — never Date.now(). */
  ts: number;
  /** userIds mentioned via @handle in `text`. */
  mentions: string[];
  reactions: Reactions;
  pinned: boolean;
  resolved: boolean;

  // --- Hidden ground-truth labels (server-only, stripped from /api/state) ---
  /** Which seeded scenario this message belongs to (support, incident, ...). */
  _scenario?: string;
  /** This message is a question that expects an answer. */
  _isQuestion?: boolean;
  /** Facts a correct answer to this question must contain. */
  _answerFacts?: string[];
  /** The user who should own/triage this request. */
  _correctAssigneeId?: string;
  /** Facts a correct summary of this thread must include. */
  _summaryFacts?: string[];
  /** Incident severity ground truth. */
  _severity?: Severity;
}

/** Top-level in-memory workspace state. */
export interface Workspace {
  id: string;
  name: string;
  /** The seed this workspace was generated from (echoed for debugging). */
  seed: number;
  /** Scale preset used for generation. */
  scale: string;
  /** The user the agent acts as. */
  currentUserId: string;
  users: User[];
  channels: Channel[];
  messages: Message[];
}

/** A Message with all `_`-prefixed ground-truth fields removed. */
export type PublicMessage = Omit<
  Message,
  | "_scenario"
  | "_isQuestion"
  | "_answerFacts"
  | "_correctAssigneeId"
  | "_summaryFacts"
  | "_severity"
>;

/** The public shape returned by GET /api/state — never contains `_` fields. */
export interface PublicState {
  workspace: { id: string; name: string; seed: number; scale: string };
  currentUserId: string;
  users: User[];
  channels: Channel[];
  messages: PublicMessage[];
}
