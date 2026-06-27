/**
 * In-memory workspace store.
 *
 * Holds the full Workspace (including hidden `_` ground-truth fields) and is
 * the single mutation point. Public reads go through `toPublic*` helpers that
 * strip every `_`-prefixed key, so the hidden labels can only escape via the
 * token-gated oracle endpoint (see server.ts / A6).
 *
 * Runtime mutations (post/react/pin/resolve/edit) stay deterministic:
 * new messages get `maxTs + 1min`, never Date.now(). Same seed + same action
 * sequence => byte-identical state and screenshots.
 */
import { seedWorkspace } from "./seed";
import type {
  Channel,
  Message,
  PublicMessage,
  PublicState,
  Severity,
  User,
  Workspace,
} from "./types";

const MINUTE = 60_000;

/** Strip every `_`-prefixed key from a message. */
export function toPublicMessage(m: Message): PublicMessage {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(m)) {
    if (!k.startsWith("_")) out[k] = v;
  }
  return out as unknown as PublicMessage;
}

/** Resolve @handle tokens to userIds present in the roster. */
export function extractMentions(text: string, users: User[]): string[] {
  const byHandle = new Map(users.map((u) => [u.handle.toLowerCase(), u.id]));
  const out: string[] = [];
  const re = /@([a-z0-9_]+)/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const id = byHandle.get(m[1].toLowerCase());
    if (id && !out.includes(id)) out.push(id);
  }
  return out;
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

export class StoreError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export class Store {
  private ws: Workspace;
  private nextMsgNum = 1;

  constructor(seed = 1, scale = "medium") {
    this.ws = seedWorkspace(seed, scale);
    this.nextMsgNum = this.ws.messages.length + 1;
  }

  // --- lifecycle -----------------------------------------------------------

  reset(seed: number, scale?: string): { seed: number; scale: string; messages: number } {
    this.ws = seedWorkspace(seed, scale);
    this.nextMsgNum = this.ws.messages.length + 1;
    return { seed: this.ws.seed, scale: this.ws.scale, messages: this.ws.messages.length };
  }

  health(): { status: string; seed: number; messages: number } {
    return { status: "ok", seed: this.ws.seed, messages: this.ws.messages.length };
  }

  // --- public reads (hidden fields stripped) -------------------------------

  publicState(): PublicState {
    return {
      workspace: {
        id: this.ws.id,
        name: this.ws.name,
        seed: this.ws.seed,
        scale: this.ws.scale,
      },
      currentUserId: this.ws.currentUserId,
      users: this.ws.users,
      channels: this.ws.channels,
      messages: this.ws.messages.map(toPublicMessage),
    };
  }

  channels(): Channel[] {
    return this.ws.channels;
  }

  /** Top-level messages in a channel (thread replies excluded), oldest first. */
  channelMessages(channelId: string): PublicMessage[] {
    if (!this.ws.channels.some((c) => c.id === channelId)) {
      throw new StoreError(404, `Unknown channel '${channelId}'`);
    }
    return this.ws.messages
      .filter((m) => m.channelId === channelId && !m.threadRootId)
      .sort((a, b) => a.ts - b.ts)
      .map(toPublicMessage);
  }

  /** A thread: root message followed by its replies, oldest first. */
  thread(rootId: string): PublicMessage[] {
    const root = this.ws.messages.find((m) => m.id === rootId);
    if (!root) throw new StoreError(404, `Unknown thread root '${rootId}'`);
    const replies = this.ws.messages
      .filter((m) => m.threadRootId === rootId)
      .sort((a, b) => a.ts - b.ts);
    return [root, ...replies].map(toPublicMessage);
  }

  search(query: string): PublicMessage[] {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return this.ws.messages
      .filter((m) => m.text.toLowerCase().includes(q))
      .sort((a, b) => b.ts - a.ts)
      .map(toPublicMessage);
  }

  // --- mutations -----------------------------------------------------------

  private maxTs(): number {
    return this.ws.messages.reduce((mx, m) => Math.max(mx, m.ts), 0);
  }

  postMessage(input: PostInput): PublicMessage {
    const text = (input.text ?? "").trim();
    if (!text) throw new StoreError(400, "text is required");

    let channelId = input.channelId;
    if (input.threadRootId) {
      const root = this.ws.messages.find((m) => m.id === input.threadRootId);
      if (!root) throw new StoreError(404, `Unknown thread root '${input.threadRootId}'`);
      channelId = root.channelId; // replies always live in the root's channel
    }
    if (!this.ws.channels.some((c) => c.id === channelId)) {
      throw new StoreError(404, `Unknown channel '${channelId}'`);
    }

    const authorId = input.authorId ?? this.ws.currentUserId;
    if (!this.ws.users.some((u) => u.id === authorId)) {
      throw new StoreError(400, `Unknown author '${authorId}'`);
    }

    const msg: Message = {
      id: `m${this.nextMsgNum++}`,
      channelId,
      threadRootId: input.threadRootId,
      authorId,
      text,
      ts: this.maxTs() + MINUTE, // deterministic — never Date.now()
      mentions: extractMentions(text, this.ws.users),
      reactions: {},
      pinned: false,
      resolved: false,
    };
    this.ws.messages.push(msg);
    return toPublicMessage(msg);
  }

  patchMessage(id: string, patch: PatchOp): PublicMessage {
    const msg = this.ws.messages.find((m) => m.id === id);
    if (!msg) throw new StoreError(404, `Unknown message '${id}'`);

    switch (patch.op) {
      case "react": {
        const emoji = patch.emoji;
        if (!emoji) throw new StoreError(400, "emoji is required to react");
        const userId = patch.userId ?? this.ws.currentUserId;
        const list = msg.reactions[emoji] ?? [];
        msg.reactions[emoji] = list.includes(userId)
          ? list.filter((u) => u !== userId)
          : [...list, userId];
        if (msg.reactions[emoji].length === 0) delete msg.reactions[emoji];
        break;
      }
      case "pin":
        msg.pinned = !msg.pinned;
        break;
      case "resolve":
        msg.resolved = patch.value ?? !msg.resolved;
        break;
      case "edit": {
        const text = (patch.text ?? "").trim();
        if (!text) throw new StoreError(400, "text is required to edit");
        msg.text = text;
        msg.mentions = extractMentions(text, this.ws.users);
        break;
      }
      default:
        throw new StoreError(400, "unknown patch op");
    }
    return toPublicMessage(msg);
  }

  // --- privileged (oracle) -------------------------------------------------

  /**
   * Full state INCLUDING hidden `_` labels, plus derived ground-truth answers.
   * Only the token-gated /api/_oracle/state route calls this. The `derived`
   * block is the privileged "truth" Pillar 2 will reward against (and which a
   * cheap proxy reward must NOT have access to).
   */
  oracleView(): OracleView {
    return { ...this.ws, derived: this.derive() };
  }

  private replyCount(rootId: string): number {
    return this.ws.messages.filter((m) => m.threadRootId === rootId).length;
  }

  private handleOf(userId: string): string {
    return this.ws.users.find((u) => u.id === userId)?.handle ?? "";
  }

  /** Reduce the hidden `_` labels into structured ground-truth answers. */
  private derive(): Derived {
    const m = this.ws.messages;
    return {
      questions: m
        .filter((x) => x._scenario === "support_question")
        .map((x) => ({
          messageId: x.id,
          channelId: x.channelId,
          answerFacts: x._answerFacts ?? [],
          answered: this.replyCount(x.id) > 0,
          resolved: x.resolved,
        })),
      incidents: m
        .filter((x) => x._scenario === "incident")
        .map((x) => ({
          messageId: x.id,
          channelId: x.channelId,
          severity: x._severity as Severity,
          summaryFacts: x._summaryFacts ?? [],
          replyCount: this.replyCount(x.id),
        })),
      requests: m
        .filter((x) => x._scenario === "dm_request")
        .map((x) => ({
          messageId: x.id,
          channelId: x.channelId,
          correctAssigneeId: x._correctAssigneeId ?? "",
          correctAssigneeHandle: this.handleOf(x._correctAssigneeId ?? ""),
        })),
      triage: m
        .filter((x) => x._scenario === "triage")
        .map((x) => ({
          messageId: x.id,
          severity: x._severity as Severity,
          correctAssigneeId: x._correctAssigneeId ?? "",
        })),
    };
  }
}

/** Structured ground-truth answers derived from the hidden `_` labels. */
export interface Derived {
  questions: {
    messageId: string;
    channelId: string;
    answerFacts: string[];
    answered: boolean;
    resolved: boolean;
  }[];
  incidents: {
    messageId: string;
    channelId: string;
    severity: Severity;
    summaryFacts: string[];
    replyCount: number;
  }[];
  requests: {
    messageId: string;
    channelId: string;
    correctAssigneeId: string;
    correctAssigneeHandle: string;
  }[];
  triage: { messageId: string; severity: Severity; correctAssigneeId: string }[];
}

export type OracleView = Workspace & { derived: Derived };
