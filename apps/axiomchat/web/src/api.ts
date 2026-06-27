/** Thin fetch wrapper over the AxiomChat REST API (same-origin /api/*). */
import type { Message, PatchOp, PostInput, PublicState } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

const JSON_HEADERS = { "content-type": "application/json" };

export const api = {
  getState: (): Promise<PublicState> => fetch("/api/state").then(json<PublicState>),

  channelMessages: (channelId: string): Promise<Message[]> =>
    fetch(`/api/channels/${channelId}/messages`).then(json<Message[]>),

  thread: (rootId: string): Promise<Message[]> =>
    fetch(`/api/threads/${rootId}`).then(json<Message[]>),

  search: (query: string): Promise<Message[]> =>
    fetch("/api/search", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ query }),
    }).then(json<Message[]>),

  postMessage: (input: PostInput): Promise<Message> =>
    fetch("/api/messages", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify(input),
    }).then(json<Message>),

  patchMessage: (id: string, patch: PatchOp): Promise<Message> =>
    fetch(`/api/messages/${id}`, {
      method: "PATCH",
      headers: JSON_HEADERS,
      body: JSON.stringify(patch),
    }).then(json<Message>),

  reset: (seed: number, scale?: string): Promise<unknown> =>
    fetch("/api/reset", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ seed, scale }),
    }).then(json),
};
