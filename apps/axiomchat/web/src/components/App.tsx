import { useChatState, useError, useReady } from "../store";
import { Shell } from "./Shell";

/**
 * App root. Owns the `data-app-ready` marker that Playwright waits on:
 * it flips to "true" only after the initial /api/state load resolves, so the
 * environment waits on a signal rather than a timeout.
 */
export function App() {
  const ready = useReady();
  const state = useChatState();
  const error = useError();

  return (
    <div data-app-ready={ready ? "true" : "false"} className="h-full">
      {!ready && <LoadingScreen />}
      {ready && error && !state && <ErrorScreen message={error} />}
      {ready && state && <Shell state={state} />}
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="flex h-full items-center justify-center text-ink-3">
      <span className="text-sm">Loading AxiomChat…</span>
    </div>
  );
}

function ErrorScreen({ message }: { message: string }) {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="rounded-lg border border-danger/40 bg-panel px-6 py-4 text-center">
        <div className="text-sm font-medium text-danger">Failed to load workspace</div>
        <div className="mt-1 text-xs text-ink-3">{message}</div>
      </div>
    </div>
  );
}
