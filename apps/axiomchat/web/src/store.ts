/**
 * Single source-of-truth store.
 *
 * The whole UI renders from one PublicState snapshot hydrated from /api/state.
 * Action helpers (post/react/pin/resolve/edit) mutate via the API and then
 * refetch the canonical state — the client never optimistically invents data,
 * so render stays a pure function of server state (no Date.now / Math.random).
 */
import { useSyncExternalStore } from "react";
import { api } from "./api";
import type { PatchOp, PostInput, PublicState } from "./types";

interface Snapshot {
  state: PublicState | null;
  ready: boolean;
  error: string | null;
}

let snap: Snapshot = { state: null, ready: false, error: null };
const listeners = new Set<() => void>();

function emit(): void {
  for (const l of listeners) l();
}
function set(patch: Partial<Snapshot>): void {
  snap = { ...snap, ...patch };
  emit();
}
function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

/** Initial load — kicked off once at startup; flips `ready` on success. */
export async function hydrate(): Promise<void> {
  try {
    const state = await api.getState();
    set({ state, ready: true, error: null });
  } catch (e) {
    set({ error: e instanceof Error ? e.message : String(e), ready: true });
  }
}

/** Re-pull canonical state after a mutation. */
async function refetch(): Promise<void> {
  const state = await api.getState();
  set({ state });
}

/**
 * In-flight mutation counter, mirrored as a `data-busy` attribute on <html>.
 * Set synchronously when an action starts (before any await) and cleared when
 * its refetch resolves — so a Playwright-driven env can wait for the DOM to
 * settle on a *signal* (`html:not([data-busy])`) rather than a timeout.
 */
let pending = 0;
function setBusy(delta: number): void {
  pending += delta;
  if (typeof document !== "undefined") {
    document.documentElement.toggleAttribute("data-busy", pending > 0);
  }
}

export const actions = {
  post: async (input: PostInput): Promise<void> => {
    setBusy(1);
    try {
      await api.postMessage(input);
      await refetch();
    } finally {
      setBusy(-1);
    }
  },
  patch: async (id: string, patch: PatchOp): Promise<void> => {
    setBusy(1);
    try {
      await api.patchMessage(id, patch);
      await refetch();
    } finally {
      setBusy(-1);
    }
  },
  react: (id: string, emoji: string, userId?: string): Promise<void> =>
    actions.patch(id, { op: "react", emoji, userId }),
  pin: (id: string): Promise<void> => actions.patch(id, { op: "pin" }),
  resolve: (id: string, value?: boolean): Promise<void> =>
    actions.patch(id, { op: "resolve", value }),
  edit: (id: string, text: string): Promise<void> =>
    actions.patch(id, { op: "edit", text }),
};

// --- hooks -----------------------------------------------------------------

export function useReady(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => snap.ready,
    () => snap.ready,
  );
}

export function useChatState(): PublicState | null {
  return useSyncExternalStore(
    subscribe,
    () => snap.state,
    () => snap.state,
  );
}

export function useError(): string | null {
  return useSyncExternalStore(
    subscribe,
    () => snap.error,
    () => snap.error,
  );
}
