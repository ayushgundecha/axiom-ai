import { useMemo } from "react";
import type { PublicState } from "../types";
import { channelById, pinnedMessages } from "../selectors";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";

export function ChannelView({
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
  const pins = useMemo(
    () => (channel ? pinnedMessages(state, channel.id) : []),
    [state, channel],
  );

  if (!channel) {
    return (
      <div className="flex flex-1 items-center justify-center text-sm text-ink-3">
        Channel not found.
      </div>
    );
  }

  return (
    <>
      <header
        data-testid="channel-header"
        className="flex items-center justify-between border-b border-line px-5 py-3"
      >
        <div className="min-w-0">
          <h1 data-testid="channel-name" className="flex items-center gap-1.5 text-[17px] font-bold text-ink">
            <span className="text-ink-3">{channel.kind === "private" ? "🔒" : "#"}</span>
            {channel.name}
          </h1>
          <p data-testid="channel-details" className="mt-0.5 truncate text-xs text-ink-3">
            {channel.topic || "No topic set"} · {channel.memberIds.length} members
            {pins.length > 0 && ` · ${pins.length} pinned`}
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
        showChannelSelect
        placeholder={`Message #${channel.name}`}
      />
    </>
  );
}
