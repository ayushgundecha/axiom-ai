import { describe, it, expect } from "vitest";
import { mulberry32, seedWorkspace, BASE_EPOCH } from "../src/seed";

describe("mulberry32", () => {
  it("is deterministic for a given seed", () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    const seqA = [a(), a(), a(), a()];
    const seqB = [b(), b(), b(), b()];
    expect(seqA).toEqual(seqB);
  });

  it("produces values in [0, 1)", () => {
    const r = mulberry32(7);
    for (let i = 0; i < 1000; i++) {
      const v = r();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("differs across seeds", () => {
    expect(mulberry32(1)()).not.toEqual(mulberry32(2)());
  });
});

describe("seedWorkspace determinism", () => {
  it("yields byte-identical workspaces for the same seed", () => {
    const a = JSON.stringify(seedWorkspace(123, "medium"));
    const b = JSON.stringify(seedWorkspace(123, "medium"));
    expect(a).toEqual(b);
  });

  it("yields different workspaces for different seeds", () => {
    const a = JSON.stringify(seedWorkspace(1, "medium"));
    const b = JSON.stringify(seedWorkspace(2, "medium"));
    expect(a).not.toEqual(b);
  });

  it("treats seed as unsigned 32-bit (stable, no NaN)", () => {
    const w = seedWorkspace(0, "medium");
    expect(w.seed).toBe(0);
    expect(w.messages.length).toBeGreaterThan(0);
  });
});

describe("seedWorkspace timestamps", () => {
  it("derives all timestamps from BASE_EPOCH as integers (never Date.now)", () => {
    const w = seedWorkspace(5, "medium");
    for (const m of w.messages) {
      expect(Number.isInteger(m.ts)).toBe(true);
      expect(m.ts).toBeGreaterThanOrEqual(BASE_EPOCH);
    }
  });

  it("BASE_EPOCH is the fixed 2024-06-03T09:00:00Z constant", () => {
    expect(BASE_EPOCH).toBe(Date.UTC(2024, 5, 3, 9, 0, 0));
  });
});

describe("seedWorkspace scenarios + hidden labels", () => {
  it("populates all four labeled scenarios", () => {
    const w = seedWorkspace(1, "large");
    const scenarios = new Set(
      w.messages.map((m) => m._scenario).filter((s): s is string => Boolean(s)),
    );
    expect(scenarios).toContain("support_question");
    expect(scenarios).toContain("incident");
    expect(scenarios).toContain("dm_request");
    expect(scenarios).toContain("triage");
  });

  it("the support question is unanswered and unresolved", () => {
    const w = seedWorkspace(1, "medium");
    const q = w.messages.find((m) => m._scenario === "support_question");
    expect(q).toBeDefined();
    expect(q?._isQuestion).toBe(true);
    expect((q?._answerFacts ?? []).length).toBeGreaterThan(0);
    expect(q?.resolved).toBe(false);
    const replies = w.messages.filter((m) => m.threadRootId === q?.id);
    expect(replies.length).toBe(0);
  });

  it("the incident thread carries severity, summary facts, and replies", () => {
    const w = seedWorkspace(1, "medium");
    const root = w.messages.find((m) => m._scenario === "incident");
    expect(root?._severity).toMatch(/SEV[123]/);
    expect((root?._summaryFacts ?? []).length).toBeGreaterThan(0);
    const replies = w.messages.filter((m) => m.threadRootId === root?.id);
    expect(replies.length).toBeGreaterThan(0);
  });
});

describe("seedWorkspace scale", () => {
  it("scales workspace size with the scale preset", () => {
    const small = seedWorkspace(1, "small");
    const medium = seedWorkspace(1, "medium");
    const large = seedWorkspace(1, "large");
    expect(small.messages.length).toBeLessThan(medium.messages.length);
    expect(medium.messages.length).toBeLessThan(large.messages.length);
  });

  it("defaults unknown scale to medium", () => {
    expect(seedWorkspace(1, "bogus").scale).toBe("medium");
  });
});
