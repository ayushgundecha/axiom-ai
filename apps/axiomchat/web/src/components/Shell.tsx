import { useCallback, useEffect, useState } from "react";
import type { PublicState } from "../types";
import { namedChannels } from "../selectors";
import { navigate, useRoute } from "../router";
import { Sidebar } from "./Sidebar";
import { ChannelView } from "./ChannelView";
import { DMView } from "./DMView";
import { SearchView } from "./SearchView";
import { ThreadPane } from "./ThreadPane";

/**
 * The 3-pane Slack shell: sidebar | main column | optional thread right-pane.
 * Route (#/channel/:id, #/dm/:id, #/search) selects the main column; the
 * thread pane is ephemeral UI state opened from a message's reply action.
 */
export function Shell({ state }: { state: PublicState }) {
  const route = useRoute();
  const [threadRootId, setThreadRootId] = useState<string | null>(null);

  // Default to the first channel when no route is present.
  useEffect(() => {
    if (route === null) {
      const first = namedChannels(state)[0];
      if (first) navigate({ kind: "channel", id: first.id });
    }
  }, [route, state]);

  // Close the thread pane whenever the main route changes.
  const routeKey = route ? `${route.kind}:${"id" in route ? route.id : ""}` : "";
  useEffect(() => {
    setThreadRootId(null);
  }, [routeKey]);

  const openThread = useCallback((rootId: string) => setThreadRootId(rootId), []);
  const closeThread = useCallback(() => setThreadRootId(null), []);

  return (
    <div className="flex h-full w-full overflow-hidden bg-base text-ink">
      <Sidebar state={state} route={route} />
      <main className="flex min-w-0 flex-1">
        <section className="flex min-w-0 flex-1 flex-col bg-panel">
          {(!route || route.kind === "channel") && (
            <ChannelView
              state={state}
              channelId={route?.kind === "channel" ? route.id : (namedChannels(state)[0]?.id ?? "")}
              onOpenThread={openThread}
              activeThreadId={threadRootId}
            />
          )}
          {route?.kind === "dm" && (
            <DMView
              state={state}
              channelId={route.id}
              onOpenThread={openThread}
              activeThreadId={threadRootId}
            />
          )}
          {route?.kind === "search" && <SearchView state={state} />}
        </section>
        {threadRootId && (
          <ThreadPane state={state} rootId={threadRootId} onClose={closeThread} />
        )}
      </main>
    </div>
  );
}
