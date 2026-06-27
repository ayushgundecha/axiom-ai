import { useMemo } from "react";
import type { PublicState } from "../types";
import { channelById, dmPartner } from "../selectors";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";
import { Avatar } from "./Avatar";

const STATUS_LABEL: Record<string, string> = {
  active: "Active",
  away: "Away",
  dnd: "Do not disturb",
  offline: "Offline",
};

export function DMView({
  state,
  channelId,
  onOpenThread,
  activeThreadId,
}: {
  state: PublicState;
  channelId: string;
  onOpenThread: (rootId: string) => void;
  activeThreadId: string | null;
}) {
  const channel = useMemo(() => channelById(state, channelId), [state, channelId]);
  const partner = useMemo(() => (channel ? dmPartner(state, channel) : undefined), [state, channel]);

  if (!channel || !partner) {
    return (
      <div data-testid="dm-view" className="flex flex-1 items-center justify-center text-sm text-ink-3">
        Conversation not found.
      </div>
    );
  }

  return (
    <div data-testid="dm-view" className="flex min-h-0 flex-1 flex-col">
      <header
        data-testid="channel-header"
        className="flex items-center gap-2.5 border-b border-line px-5 py-2.5"
      >
        <Avatar user={partner} size="sm" presence dotRing="border-panel" />
        <div className="min-w-0">
          <h1 data-testid="channel-name" className="text-[15px] font-bold text-ink">
            {partner.name}
          </h1>
          <p data-testid="channel-details" className="text-xs text-ink-3">
            {partner.title ? `${partner.title} · ` : ""}
            {STATUS_LABEL[partner.status] ?? partner.status}
          </p>
        </div>
      </header>

      <MessageList
        state={state}
        channel={channel}
        onOpenThread={onOpenThread}
        activeThreadId={activeThreadId}
      />

      <Composer
        state={state}
        channelId={channel.id}
        inputTestId="message-input"
        sendTestId="send-button"
        placeholder={`Message ${partner.name}`}
      />
    </div>
  );
}
