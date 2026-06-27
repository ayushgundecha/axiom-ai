import { describe, it, expect, beforeEach } from "vitest";
import request from "supertest";
import { createApp } from "../src/server";
import { Store } from "../src/store";

/** Recursively assert no object key anywhere starts with "_". */
function assertNoHiddenKeys(value: unknown, path = "$"): void {
  if (Array.isArray(value)) {
    value.forEach((v, i) => assertNoHiddenKeys(v, `${path}[${i}]`));
  } else if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) {
      expect(k.startsWith("_"), `leaked hidden key '${k}' at ${path}`).toBe(false);
      assertNoHiddenKeys(v, `${path}.${k}`);
    }
  }
}

function app(seed = 1, scale = "medium") {
  return createApp(new Store(seed, scale));
}

describe("GET /api/health", () => {
  it("returns ok", async () => {
    const res = await request(app()).get("/api/health");
    expect(res.status).toBe(200);
    expect(res.body.status).toBe("ok");
  });
});

describe("GET /api/state (no-leak guarantee)", () => {
  it("never exposes any _-prefixed field, at any depth", async () => {
    const res = await request(app()).get("/api/state");
    expect(res.status).toBe(200);
    assertNoHiddenKeys(res.body);
    // And the raw payload contains no hidden substrings.
    expect(JSON.stringify(res.body)).not.toContain('"_');
  });

  it("includes workspace, users, channels, messages", async () => {
    const res = await request(app()).get("/api/state");
    expect(res.body.currentUserId).toBe("u_me");
    expect(res.body.users.length).toBeGreaterThan(0);
    expect(res.body.channels.length).toBeGreaterThan(0);
    expect(res.body.messages.length).toBeGreaterThan(0);
  });
});

describe("GET /api/channels and messages", () => {
  it("lists channels", async () => {
    const res = await request(app()).get("/api/channels");
    expect(res.status).toBe(200);
    expect(res.body.some((c: { id: string }) => c.id === "c_general")).toBe(true);
  });

  it("returns channel messages sorted oldest-first, top-level only", async () => {
    const res = await request(app()).get("/api/channels/c_incidents/messages");
    expect(res.status).toBe(200);
    const ts = res.body.map((m: { ts: number }) => m.ts);
    expect(ts).toEqual([...ts].sort((a, b) => a - b));
    expect(res.body.every((m: { threadRootId?: string }) => !m.threadRootId)).toBe(true);
  });

  it("404s on an unknown channel", async () => {
    const res = await request(app()).get("/api/channels/nope/messages");
    expect(res.status).toBe(404);
  });
});

describe("GET /api/threads/:rootId", () => {
  it("returns the root followed by replies", async () => {
    const a = app();
    const msgs = (await request(a).get("/api/channels/c_incidents/messages")).body;
    const root = msgs[msgs.length - 1];
    const res = await request(a).get(`/api/threads/${root.id}`);
    expect(res.status).toBe(200);
    expect(res.body[0].id).toBe(root.id);
  });

  it("404s on an unknown thread root", async () => {
    expect((await request(app()).get("/api/threads/nope")).status).toBe(404);
  });
});

describe("POST /api/messages", () => {
  it("creates a message with a deterministic timestamp", async () => {
    const post = { channelId: "c_general", text: "ship it 🚀" };
    const r1 = await request(app()).post("/api/messages").send(post);
    const r2 = await request(app()).post("/api/messages").send(post);
    expect(r1.status).toBe(200);
    expect(r1.body.ts).toBe(r2.body.ts); // identical across fresh stores
    expect(r1.body.authorId).toBe("u_me");
  });

  it("parses @mentions into userIds", async () => {
    const res = await request(app())
      .post("/api/messages")
      .send({ channelId: "c_general", text: "hey @maya and @diego" });
    expect(res.body.mentions).toEqual(expect.arrayContaining(["u_maya", "u_diego"]));
  });

  it("routes a reply into the root's channel", async () => {
    const a = app();
    const msgs = (await request(a).get("/api/channels/c_incidents/messages")).body;
    const root = msgs[msgs.length - 1];
    const res = await request(a)
      .post("/api/messages")
      .send({ channelId: "c_general", text: "on it", threadRootId: root.id });
    expect(res.body.channelId).toBe("c_incidents"); // overridden to root's channel
    expect(res.body.threadRootId).toBe(root.id);
  });

  it("400s on empty text, 404s on unknown channel", async () => {
    expect((await request(app()).post("/api/messages").send({ channelId: "c_general", text: "  " })).status).toBe(400);
    expect((await request(app()).post("/api/messages").send({ channelId: "nope", text: "hi" })).status).toBe(404);
  });
});

describe("PATCH /api/messages/:id", () => {
  let a: ReturnType<typeof app>;
  let id: string;
  beforeEach(async () => {
    a = app();
    id = (await request(a).post("/api/messages").send({ channelId: "c_general", text: "x" })).body.id;
  });

  it("toggles a reaction on and off", async () => {
    const on = await request(a).patch(`/api/messages/${id}`).send({ op: "react", emoji: "🎉" });
    expect(on.body.reactions["🎉"]).toContain("u_me");
    const off = await request(a).patch(`/api/messages/${id}`).send({ op: "react", emoji: "🎉" });
    expect(off.body.reactions["🎉"]).toBeUndefined();
  });

  it("toggles pin and sets resolved", async () => {
    expect((await request(a).patch(`/api/messages/${id}`).send({ op: "pin" })).body.pinned).toBe(true);
    expect((await request(a).patch(`/api/messages/${id}`).send({ op: "resolve", value: true })).body.resolved).toBe(true);
  });

  it("edits text and re-derives mentions", async () => {
    const res = await request(a).patch(`/api/messages/${id}`).send({ op: "edit", text: "now mentioning @sam" });
    expect(res.body.text).toBe("now mentioning @sam");
    expect(res.body.mentions).toContain("u_sam");
  });

  it("404s on an unknown message", async () => {
    expect((await request(a).patch("/api/messages/nope").send({ op: "pin" })).status).toBe(404);
  });
});

describe("POST /api/search", () => {
  it("matches message text case-insensitively", async () => {
    const a = app();
    await request(a).post("/api/messages").send({ channelId: "c_general", text: "UNIQUE_TOKEN_XYZ here" });
    const res = await request(a).post("/api/search").send({ query: "unique_token_xyz" });
    expect(res.status).toBe(200);
    expect(res.body.length).toBe(1);
  });

  it("returns nothing for an empty query", async () => {
    expect((await request(app()).post("/api/search").send({ query: "" })).body).toEqual([]);
  });
});

describe("POST /api/reset", () => {
  it("re-seeds: same seed => identical state, different seed => different", async () => {
    const a = app();
    await request(a).post("/api/reset").send({ seed: 9, scale: "medium" });
    const s1 = (await request(a).get("/api/state")).body;
    await request(a).post("/api/reset").send({ seed: 10, scale: "medium" });
    const s2 = (await request(a).get("/api/state")).body;
    await request(a).post("/api/reset").send({ seed: 9, scale: "medium" });
    const s3 = (await request(a).get("/api/state")).body;
    expect(s1).toEqual(s3);
    expect(s1).not.toEqual(s2);
  });
});

describe("GET /api/_oracle/state (token-gated)", () => {
  beforeEach(() => {
    process.env.AXIOMCHAT_ORACLE_TOKEN = "test-oracle-token";
  });

  it("403s without a token and with a wrong token", async () => {
    expect((await request(app()).get("/api/_oracle/state")).status).toBe(403);
    expect((await request(app()).get("/api/_oracle/state").set("X-Oracle-Token", "wrong")).status).toBe(403);
  });

  it("200s with a valid token and exposes hidden labels + derived truth", async () => {
    const res = await request(app()).get("/api/_oracle/state").set("X-Oracle-Token", "test-oracle-token");
    expect(res.status).toBe(200);
    const labeled = res.body.messages.filter((m: Record<string, unknown>) =>
      Object.keys(m).some((k) => k.startsWith("_")),
    );
    expect(labeled.length).toBeGreaterThan(0);
    expect(res.body.derived.questions.length).toBeGreaterThan(0);
    expect(res.body.derived.incidents.length).toBeGreaterThan(0);
    expect(res.body.derived.requests.length).toBeGreaterThan(0);
  });
});
