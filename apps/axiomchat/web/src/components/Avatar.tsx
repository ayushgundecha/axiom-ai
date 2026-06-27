import type { User } from "../types";
import { avatarColor, initials } from "../util/avatar";

const PRESENCE: Record<User["status"], string> = {
  active: "bg-presence-active",
  away: "bg-presence-away",
  dnd: "bg-presence-dnd",
  offline: "bg-presence-offline",
};

const SIZES = {
  sm: "h-6 w-6 rounded text-[10px]",
  md: "h-9 w-9 rounded-md text-xs",
  lg: "h-10 w-10 rounded-md text-sm",
};

export function Avatar({
  user,
  size = "md",
  presence = false,
  dotRing = "border-panel",
}: {
  user: User;
  size?: keyof typeof SIZES;
  presence?: boolean;
  dotRing?: string;
}) {
  return (
    <span className="relative inline-flex shrink-0">
      <span
        className={`inline-flex items-center justify-center font-semibold text-white/95 ${SIZES[size]}`}
        style={{ backgroundColor: avatarColor(user.id) }}
        aria-hidden="true"
      >
        {initials(user.name)}
      </span>
      {presence && (
        <span
          className={`absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 ${dotRing} ${PRESENCE[user.status]}`}
          aria-label={`status: ${user.status}`}
        />
      )}
    </span>
  );
}
