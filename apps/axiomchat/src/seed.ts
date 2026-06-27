/**
 * Deterministic workspace seeding.
 *
 * Everything here is a pure function of (seed, scale). No Date.now(), no
 * Math.random() — the only entropy source is `mulberry32(seed)`, consumed in
 * a fixed order. Same seed => byte-identical workspace; different seed =>
 * different workspace. Timestamps are `BASE_EPOCH + seedDerivedOffset`.
 *
 * Hidden `_`-prefixed ground-truth labels are embedded per scenario
 * (an unanswered #support question, a #incidents thread, a DM request, and a
 * #triage backlog). They are the substrate Pillar 2 will reward against;
 * Pillar 1 only populates and gates them.
 */
import type {
  Channel,
  Message,
  Reactions,
  Severity,
  User,
  UserStatus,
  Workspace,
} from "./types";

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

/** mulberry32 — a fast, well-distributed 32-bit seeded PRNG. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function next(): number {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Thin deterministic helpers over a mulberry32 stream. */
export class Rng {
  private readonly r: () => number;
  constructor(seed: number) {
    this.r = mulberry32(seed);
  }
  /** float in [0, 1). */
  next(): number {
    return this.r();
  }
  /** integer in [min, max] inclusive. */
  int(min: number, max: number): number {
    return min + Math.floor(this.r() * (max - min + 1));
  }
  /** true with probability p. */
  bool(p = 0.5): boolean {
    return this.r() < p;
  }
  /** pick one element. */
  pick<T>(arr: readonly T[]): T {
    return arr[Math.floor(this.r() * arr.length)];
  }
  /** Fisher–Yates shuffle into a new array. */
  shuffle<T>(arr: readonly T[]): T[] {
    const out = arr.slice();
    for (let i = out.length - 1; i > 0; i--) {
      const j = Math.floor(this.r() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }
  /** pick `n` distinct elements (or all, if n >= length). */
  sample<T>(arr: readonly T[], n: number): T[] {
    return this.shuffle(arr).slice(0, Math.min(n, arr.length));
  }
}

// ---------------------------------------------------------------------------
// Time
// ---------------------------------------------------------------------------

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** Fixed base epoch: 2024-06-03T09:00:00Z (a Monday). Never Date.now(). */
export const BASE_EPOCH = Date.UTC(2024, 5, 3, 9, 0, 0);

// ---------------------------------------------------------------------------
// Scale presets
// ---------------------------------------------------------------------------

export type Scale = "small" | "medium" | "large";

interface ScalePreset {
  fillerPerChannel: [number, number]; // [min, max] filler messages per channel
  optionalChannels: number; // how many of the optional channels to include
  dms: number; // number of DM conversations
}

const SCALES: Record<Scale, ScalePreset> = {
  small: { fillerPerChannel: [4, 7], optionalChannels: 0, dms: 1 },
  medium: { fillerPerChannel: [7, 12], optionalChannels: 2, dms: 2 },
  large: { fillerPerChannel: [12, 18], optionalChannels: 3, dms: 3 },
};

function normalizeScale(scale: string | undefined): Scale {
  return scale === "small" || scale === "large" ? scale : "medium";
}

// ---------------------------------------------------------------------------
// Fixed pools
// ---------------------------------------------------------------------------

/** The current user (the agent). Always present, stable id across seeds. */
const ME: User = {
  id: "u_me",
  name: "Jordan Lee",
  handle: "jordan",
  role: "member",
  status: "active",
  title: "Software Engineer",
};

interface UserSeed {
  id: string;
  name: string;
  handle: string;
  role: User["role"];
  title: string;
}

/** Stable roster (a small startup eng team). Presence is seeded per workspace. */
const ROSTER: readonly UserSeed[] = [
  { id: "u_emma", name: "Emma Wilson", handle: "emma", role: "owner", title: "Engineering Manager" },
  { id: "u_maya", name: "Maya Chen", handle: "maya", role: "admin", title: "Eng Lead" },
  { id: "u_diego", name: "Diego Santos", handle: "diego", role: "member", title: "Backend Engineer" },
  { id: "u_priya", name: "Priya Nair", handle: "priya", role: "member", title: "Frontend Engineer" },
  { id: "u_sam", name: "Sam Okafor", handle: "sam", role: "member", title: "Site Reliability Eng" },
  { id: "u_lena", name: "Lena Petrova", handle: "lena", role: "member", title: "Product Manager" },
  { id: "u_tom", name: "Tom Becker", handle: "tom", role: "member", title: "Product Designer" },
  { id: "u_aisha", name: "Aisha Khan", handle: "aisha", role: "member", title: "Data Engineer" },
  { id: "u_ravi", name: "Ravi Mehta", handle: "ravi", role: "member", title: "DevOps Engineer" },
  { id: "u_carlos", name: "Carlos Diaz", handle: "carlos", role: "guest", title: "Support Engineer" },
];

const STATUSES: readonly UserStatus[] = ["active", "active", "active", "away", "dnd", "offline"];

interface ChannelSeed {
  id: string;
  name: string;
  topic: string;
  /** members by handle; "*" means the whole roster + me. */
  members: string[] | "*";
  /** message pool used for filler chatter. */
  filler: readonly string[];
  optional?: boolean;
}

const GENERAL_FILLER = [
  "Morning all ☕ — standup in 10.",
  "Reminder: lunch & learn on the new event bus is Thursday.",
  "Welcome to the new folks joining this week! 🎉",
  "Office is closed Monday for the holiday.",
  "Coffee machine on the 4th floor is fixed.",
  "Sprint review notes are in the wiki.",
  "Anyone have the link to the Q3 roadmap deck?",
  "Kudos to the platform team for the latency win 🚀",
  "Heads up: SSO maintenance window tonight 22:00–23:00 UTC.",
  "Don't forget to submit your timesheets by EOD Friday.",
];

const ENGINEERING_FILLER = [
  "Bumped the Node base image to 20-slim across services.",
  "The flaky `checkout.e2e` test should be stable now — pinned the seed.",
  "Can someone review the connection-pool PR? It's small.",
  "Migrated the search index to the new analyzer, p95 is down 18%.",
  "Reminder to rebase onto main before merging, CI is much faster now.",
  "Added structured logging to the ledger service.",
  "Feature flag `new_billing_ui` is at 25% rollout, metrics look clean.",
  "Heads up: deprecating the v1 webhooks endpoint next sprint.",
  "Profiling shows the N+1 in the invoices export — fix incoming.",
  "Tag cut: 4.2.1 is on staging, smoke tests green.",
];

const DESIGN_FILLER = [
  "Pushed the new empty-state illustrations to Figma.",
  "Dark theme tokens are finalized — handing off to eng.",
  "Can we get 8px more padding on the message composer?",
  "New avatar color ramp is more accessible (AA on dark).",
  "Prototype for threaded replies is ready for review.",
  "Iconography pass done for the sidebar.",
];

const RANDOM_FILLER = [
  "What's everyone reading this week? 📚",
  "Friday demo was 🔥",
  "New espresso blend in the kitchen, highly recommend.",
  "Anyone up for a lunch run?",
  "Cat tax: 🐈 (photo in thread)",
  "TIL you can pin messages here.",
];

const CHANNELS: readonly ChannelSeed[] = [
  {
    id: "c_general",
    name: "general",
    topic: "Company-wide announcements and chatter",
    members: "*",
    filler: GENERAL_FILLER,
  },
  {
    id: "c_engineering",
    name: "engineering",
    topic: "Eng discussion, deploys, and reviews",
    members: ["maya", "diego", "priya", "sam", "aisha", "ravi", "emma"],
    filler: ENGINEERING_FILLER,
  },
  {
    id: "c_support",
    name: "support",
    topic: "Customer-reported issues — triage and answer",
    members: ["carlos", "maya", "diego", "priya", "sam"],
    filler: GENERAL_FILLER,
  },
  {
    id: "c_incidents",
    name: "incidents",
    topic: "Active incidents and postmortems",
    members: ["sam", "ravi", "maya", "diego", "emma"],
    filler: ENGINEERING_FILLER,
  },
  {
    id: "c_design",
    name: "design",
    topic: "Product design and UX",
    members: ["tom", "priya", "lena", "emma"],
    filler: DESIGN_FILLER,
    optional: true,
  },
  {
    id: "c_random",
    name: "random",
    topic: "Non-work banter",
    members: "*",
    filler: RANDOM_FILLER,
    optional: true,
  },
  {
    id: "c_triage",
    name: "triage",
    topic: "Private: bug backlog awaiting assignment",
    members: ["maya", "diego", "sam", "aisha", "ravi"],
    filler: ENGINEERING_FILLER,
    optional: true,
  },
];

// ---------------------------------------------------------------------------
// Scenario variant pools (text + matching hidden ground truth)
// ---------------------------------------------------------------------------

interface SupportVariant {
  text: string;
  answerFacts: string[];
}
const SUPPORT_VARIANTS: readonly SupportVariant[] = [
  {
    text: "Enterprise customer says SSO via Okta broke after the 4.2 release — SAML responses now fail with `invalid_audience`. Anyone know the fix?",
    answerFacts: ["audience", "4.2", "Okta", "entityId", "SAML"],
  },
  {
    text: "Customer on the Growth plan reports webhook deliveries stopped after they rotated their signing secret. Retries all 401. What's the correct way to re-sync?",
    answerFacts: ["signing secret", "401", "webhook", "rotate", "re-sync"],
  },
  {
    text: "A user can't export invoices over 50k rows — the download 504s. Is there a documented limit or workaround?",
    answerFacts: ["504", "50k", "invoices", "export", "pagination"],
  },
];

interface IncidentVariant {
  rootText: string;
  severity: Severity;
  summaryFacts: string[];
  replies: { handle: string; text: string }[];
}
const INCIDENT_VARIANTS: readonly IncidentVariant[] = [
  {
    rootText:
      "🚨 Elevated 5xx on checkout-api starting ~09:14 UTC. Error rate ~12%, p99 latency 3.2s. Investigating.",
    severity: "SEV2",
    summaryFacts: ["checkout-api", "5xx", "12%", "p99 3.2s", "09:14"],
    replies: [
      { handle: "maya", text: "Connection pool to payments-db looks saturated — max_conns pegged." },
      { handle: "ravi", text: "Rolling back deploy 4.2.1 now." },
      { handle: "sam", text: "Rollback complete, error rate back to baseline. Monitoring for 30m." },
    ],
  },
  {
    rootText:
      "🚨 SEV1: auth-service is down, login success rate dropped to 0% at 14:02 UTC. All regions affected.",
    severity: "SEV1",
    summaryFacts: ["auth-service", "0%", "14:02", "all regions", "login"],
    replies: [
      { handle: "ravi", text: "Redis primary failed over and the replica wasn't promoted — fixing." },
      { handle: "maya", text: "Promoted replica manually, sessions recovering." },
      { handle: "sam", text: "Login success rate back to 99.9%. Writing the postmortem." },
    ],
  },
];

interface DmRequestVariant {
  text: string;
  assigneeHandle: string;
}
const DM_REQUEST_VARIANTS: readonly DmRequestVariant[] = [
  {
    text: "Can someone own migrating the billing webhooks to the new event bus before the 4.3 cutover? It touches Stripe + the ledger service.",
    assigneeHandle: "diego",
  },
  {
    text: "We need an owner for the incident postmortem automation — pulling timelines from #incidents into the wiki. Who has bandwidth?",
    assigneeHandle: "ravi",
  },
];

interface TriageItem {
  text: string;
  severity: Severity;
  assigneeHandle: string;
}
const TRIAGE_ITEMS: readonly TriageItem[] = [
  { text: "BUG: intermittent 500 on /api/invoices/export for tenants > 50k rows.", severity: "SEV3", assigneeHandle: "aisha" },
  { text: "BUG: password reset emails delayed 20+ min during peak.", severity: "SEV2", assigneeHandle: "ravi" },
  { text: "BUG: stale balance shown on the billing dashboard after refunds.", severity: "SEV2", assigneeHandle: "diego" },
  { text: "BUG: search returns deleted threads for ~5 min (index lag).", severity: "SEV3", assigneeHandle: "maya" },
  { text: "BUG: mobile composer loses draft on rotation.", severity: "SEV3", assigneeHandle: "priya" },
];

const EMOJIS = ["👍", "🎉", "🚀", "👀", "✅", "🔥", "❤️", "🙏", "💯", "😄"] as const;

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/** Internal mutable builder threading the RNG, id counter, and message sink. */
class Builder {
  readonly rng: Rng;
  readonly users: User[];
  readonly byHandle: Map<string, User>;
  readonly messages: Message[] = [];
  private n = 0;

  constructor(rng: Rng, users: User[]) {
    this.rng = rng;
    this.users = users;
    this.byHandle = new Map(users.map((u) => [u.handle, u]));
  }

  id(): string {
    this.n += 1;
    return `m${this.n}`;
  }

  handleId(handle: string): string {
    return this.byHandle.get(handle)?.id ?? "u_me";
  }

  /** Parse @handle tokens in text into mentioned userIds present in the workspace. */
  mentions(text: string): string[] {
    const out: string[] = [];
    const re = /@([a-z0-9_]+)/gi;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      const u = this.byHandle.get(m[1].toLowerCase());
      if (u && !out.includes(u.id)) out.push(u.id);
    }
    return out;
  }

  /** Seeded reaction set for a message (0–3 emojis, each from 1–4 reactors). */
  reactions(memberIds: string[]): Reactions {
    const out: Reactions = {};
    const count = this.rng.bool(0.45) ? this.rng.int(1, 3) : 0;
    const emojis = this.rng.sample(EMOJIS, count);
    for (const e of emojis) {
      const reactors = this.rng.sample(memberIds, this.rng.int(1, Math.min(4, memberIds.length)));
      if (reactors.length > 0) out[e] = reactors;
    }
    return out;
  }

  push(msg: Partial<Message> & Pick<Message, "channelId" | "authorId" | "text" | "ts">): Message {
    const full: Message = {
      id: this.id(),
      threadRootId: undefined,
      mentions: this.mentions(msg.text),
      reactions: {},
      pinned: false,
      resolved: false,
      ...msg,
    };
    this.messages.push(full);
    return full;
  }
}

// ---------------------------------------------------------------------------
// seedWorkspace
// ---------------------------------------------------------------------------

export function seedWorkspace(seed: number, scale?: string): Workspace {
  const sc = normalizeScale(scale);
  const preset = SCALES[sc];
  const rng = new Rng(seed >>> 0);

  // --- Users: stable roster, seeded presence -------------------------------
  const users: User[] = [
    ME,
    ...ROSTER.map((u) => ({ ...u, status: rng.pick(STATUSES) })),
  ];
  const b = new Builder(rng, users);

  // --- Choose channels by scale --------------------------------------------
  const required = CHANNELS.filter((c) => !c.optional);
  const optional = rng.sample(
    CHANNELS.filter((c) => c.optional),
    preset.optionalChannels,
  );
  const chosen = [...required, ...optional];

  const channels: Channel[] = [];

  for (const cs of chosen) {
    const memberHandles =
      cs.members === "*" ? users.map((u) => u.handle) : ["jordan", ...cs.members];
    const memberIds = memberHandles
      .map((h) => b.byHandle.get(h)?.id)
      .filter((x): x is string => Boolean(x));

    // Per-channel timeline: start somewhere in the 3-day window.
    let ts = BASE_EPOCH + rng.int(0, 2) * DAY + rng.int(0, 6) * HOUR;
    const authors = memberIds.filter((id) => id !== "u_me");

    const fillerCount = rng.int(preset.fillerPerChannel[0], preset.fillerPerChannel[1]);
    const lines = rng.shuffle(cs.filler).slice(0, fillerCount);

    for (const line of lines) {
      ts += rng.int(3, 70) * MINUTE;
      if (rng.bool(0.18)) ts += DAY; // occasional day jump → day dividers
      const author = rng.pick(authors.length ? authors : memberIds);
      const m = b.push({ channelId: cs.id, authorId: author, text: line, ts });
      m.reactions = b.reactions(memberIds);
    }

    // Pin one announcement in #general.
    if (cs.id === "c_general" && b.messages.length > 0) {
      const target = b.messages.find((m) => m.channelId === "c_general");
      if (target) target.pinned = true;
    }

    // Inject scenario content.
    ts = injectScenario(b, cs, ts);

    // Unread tail: last k messages in this channel are unread.
    const chanMsgs = b.messages.filter((m) => m.channelId === cs.id);
    const k = rng.int(0, Math.min(3, chanMsgs.length));
    const lastReadTs =
      k === 0 || chanMsgs.length === 0
        ? (chanMsgs.at(-1)?.ts ?? BASE_EPOCH)
        : chanMsgs[chanMsgs.length - 1 - k].ts;

    channels.push({
      id: cs.id,
      name: cs.name,
      kind: cs.id === "c_triage" ? "private" : "public",
      topic: cs.topic,
      memberIds,
      lastReadTs,
    });
  }

  // --- DMs -----------------------------------------------------------------
  const dmPartners = rng.sample(
    ROSTER.filter((u) => u.handle !== "carlos"),
    preset.dms,
  );
  // Ensure the DM-request scenario partner (lena) is present for medium+.
  if (sc !== "small" && !dmPartners.some((u) => u.handle === "lena")) {
    const lena = ROSTER.find((u) => u.handle === "lena");
    if (lena) dmPartners[0] = lena;
  }

  for (const partnerSeed of dmPartners) {
    const partner = b.byHandle.get(partnerSeed.handle);
    if (!partner) continue;
    const memberIds = ["u_me", partner.id];
    let ts = BASE_EPOCH + rng.int(0, 2) * DAY + rng.int(2, 8) * HOUR;

    const smallTalk = [
      "hey, do you have 10 min later today?",
      "sent you the doc — lmk what you think",
      "thanks for the review earlier 🙏",
      "are you joining the sync at 3?",
    ];
    const lines = rng.sample(smallTalk, rng.int(2, 3));
    for (const line of lines) {
      ts += rng.int(5, 90) * MINUTE;
      const author = rng.bool() ? "u_me" : partner.id;
      b.push({ channelId: `dm_${partner.id}`, authorId: author, text: line, ts });
    }

    // DM request scenario goes into the lena DM.
    if (partner.handle === "lena") {
      const v = rng.pick(DM_REQUEST_VARIANTS);
      ts += rng.int(20, 120) * MINUTE;
      b.push({
        channelId: `dm_${partner.id}`,
        authorId: partner.id,
        text: v.text,
        ts,
        _scenario: "dm_request",
        _isQuestion: true,
        _correctAssigneeId: b.handleId(v.assigneeHandle),
      });
    }

    const chanMsgs = b.messages.filter((m) => m.channelId === `dm_${partner.id}`);
    channels.push({
      id: `dm_${partner.id}`,
      name: partner.name,
      kind: "dm",
      topic: "",
      memberIds,
      lastReadTs: chanMsgs.length > 1 ? chanMsgs[chanMsgs.length - 2].ts : BASE_EPOCH,
    });
  }

  return {
    id: "ws_axiom",
    name: "Axiom Labs",
    seed: seed >>> 0,
    scale: sc,
    currentUserId: "u_me",
    users,
    channels,
    messages: b.messages,
  };
}

/**
 * Inject the labeled scenario for a channel (if any) and return the advanced
 * timestamp cursor. Each scenario carries hidden `_` ground truth.
 */
function injectScenario(b: Builder, cs: ChannelSeed, tsIn: number): number {
  let ts = tsIn;
  const rng = b.rng;

  if (cs.id === "c_support") {
    // An UNANSWERED question: a thread root with zero replies, unresolved.
    const v = rng.pick(SUPPORT_VARIANTS);
    ts += rng.int(15, 90) * MINUTE;
    b.push({
      channelId: cs.id,
      authorId: b.handleId("carlos"),
      text: v.text,
      ts,
      _scenario: "support_question",
      _isQuestion: true,
      _answerFacts: v.answerFacts,
    });
  }

  if (cs.id === "c_incidents") {
    // An incident THREAD: labeled root + replies.
    const v = rng.pick(INCIDENT_VARIANTS);
    ts += rng.int(20, 120) * MINUTE;
    const root = b.push({
      channelId: cs.id,
      authorId: b.handleId("sam"),
      text: v.rootText,
      ts,
      _scenario: "incident",
      _severity: v.severity,
      _summaryFacts: v.summaryFacts,
    });
    for (const reply of v.replies) {
      ts += rng.int(2, 18) * MINUTE;
      b.push({
        channelId: cs.id,
        authorId: b.handleId(reply.handle),
        text: reply.text,
        ts,
        threadRootId: root.id,
      });
    }
  }

  if (cs.id === "c_triage") {
    // A backlog of labeled bug items awaiting assignment.
    const items = rng.sample(TRIAGE_ITEMS, rng.int(3, TRIAGE_ITEMS.length));
    for (const item of items) {
      ts += rng.int(30, 180) * MINUTE;
      b.push({
        channelId: cs.id,
        authorId: b.handleId("maya"),
        text: item.text,
        ts,
        _scenario: "triage",
        _severity: item.severity,
        _correctAssigneeId: b.handleId(item.assigneeHandle),
      });
    }
  }

  return ts;
}
