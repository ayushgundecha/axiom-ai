/**
 * AxiomChat server — a deterministic, resettable mini-Slack.
 *
 * Express + TypeScript, fully in-memory. Serves the React SPA built to
 * web/dist and exposes the /api/* surface that the axiom WebApp environment
 * drives via Playwright. All workspace data is produced by a seeded PRNG
 * (see seed.ts) so a given seed yields byte-identical state every time.
 *
 * Structured as a `createApp(store)` factory so vitest can exercise the API
 * in-process without binding a port.
 */
import express, { type Express, type Request, type Response } from "express";
import path from "path";
import { Store, StoreError, type PatchOp } from "./store";

const PORT = Number(process.env.PORT ?? 3000);

/** Shared secret for the privileged oracle endpoint (read per-request so tests
 *  and the Python integration can override it). Override in any real run. */
function oracleToken(): string {
  return process.env.AXIOMCHAT_ORACLE_TOKEN ?? "axiom-oracle-dev-token";
}

/** Run a store operation, translating StoreError into an HTTP status. */
function handle(res: Response, fn: () => unknown): void {
  try {
    res.json(fn());
  } catch (err) {
    if (err instanceof StoreError) {
      res.status(err.status).json({ error: err.message });
      return;
    }
    res.status(500).json({ error: (err as Error).message });
  }
}

/** Mount the /api/* routes backed by `store`. */
export function mountApi(app: Express, store: Store): void {
  app.get("/api/health", (_req: Request, res: Response) => {
    res.json(store.health());
  });

  app.get("/api/state", (_req: Request, res: Response) => {
    res.json(store.publicState());
  });

  app.get("/api/channels", (_req: Request, res: Response) => {
    res.json(store.channels());
  });

  app.get("/api/channels/:id/messages", (req: Request, res: Response) => {
    handle(res, () => store.channelMessages(String(req.params.id)));
  });

  app.get("/api/threads/:rootId", (req: Request, res: Response) => {
    handle(res, () => store.thread(String(req.params.rootId)));
  });

  app.post("/api/messages", (req: Request, res: Response) => {
    handle(res, () =>
      store.postMessage({
        channelId: req.body?.channelId,
        text: req.body?.text,
        threadRootId: req.body?.threadRootId,
        authorId: req.body?.authorId,
      }),
    );
  });

  app.patch("/api/messages/:id", (req: Request, res: Response) => {
    handle(res, () => store.patchMessage(String(req.params.id), req.body as PatchOp));
  });

  app.post("/api/search", (req: Request, res: Response) => {
    handle(res, () => store.search(String(req.body?.query ?? "")));
  });

  // Deterministic re-seed. The axiom WebApp env posts {seed, scale} here on
  // every episode reset (via the _reset_server hook) — same seed => same state.
  app.post("/api/reset", (req: Request, res: Response) => {
    const seed = Number(req.body?.seed ?? 1);
    const scale = req.body?.scale as string | undefined;
    handle(res, () => ({ status: "reset", ...store.reset(seed, scale) }));
  });

  // Privileged: full state INCLUDING hidden `_` labels + derived ground truth.
  // Gated by the X-Oracle-Token header — 403 without a valid token. This is the
  // "oracle" the Pillar 2 reward machinery will consult; the live proxy reward
  // must never see it.
  app.get("/api/_oracle/state", (req: Request, res: Response) => {
    const token = req.header("X-Oracle-Token");
    if (!token || token !== oracleToken()) {
      res.status(403).json({ error: "forbidden: valid X-Oracle-Token required" });
      return;
    }
    res.json(store.oracleView());
  });
}

/** Build the Express app with all routes mounted. */
export function createApp(store: Store = new Store()): Express {
  const app = express();
  app.use(express.json({ limit: "1mb" }));

  mountApi(app, store);

  // Serve the built SPA. web/dist is produced by `npm run build:web`.
  const webDist = path.join(__dirname, "..", "web", "dist");
  app.use(express.static(webDist));

  // SPA fallback: any non-API GET returns index.html (hash router lives client-side).
  app.get(/^(?!\/api\/).*/, (_req: Request, res: Response) => {
    res.sendFile(path.join(webDist, "index.html"));
  });

  return app;
}

if (require.main === module) {
  const seed = Number(process.env.AXIOMCHAT_SEED ?? 1);
  const scale = process.env.AXIOMCHAT_SCALE ?? "medium";
  createApp(new Store(seed, scale)).listen(PORT, () => {
    // eslint-disable-next-line no-console
    console.log(`AxiomChat server running on port ${PORT} (seed=${seed}, scale=${scale})`);
  });
}
