/**
 * Deterministic time formatting.
 *
 * All formatting is done in UTC via the explicit getUTC* accessors so output
 * does NOT depend on the host timezone — a given epoch always renders the same
 * string, keeping screenshots reproducible across machines. We deliberately
 * avoid relative labels ("today"/"2h ago") since those would depend on a
 * wall-clock "now".
 */
const DAYS = [
  "Sunday",
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
];
const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

/** "9:42 AM" — 12-hour clock, UTC. */
export function formatClock(ts: number): string {
  const d = new Date(ts);
  let h = d.getUTCHours();
  const m = d.getUTCMinutes();
  const ampm = h >= 12 ? "PM" : "AM";
  h = h % 12;
  if (h === 0) h = 12;
  return `${h}:${String(m).padStart(2, "0")} ${ampm}`;
}

/** "Monday, June 3" — used by day dividers, UTC. */
export function formatDayDivider(ts: number): string {
  const d = new Date(ts);
  return `${DAYS[d.getUTCDay()]}, ${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`;
}

/** "Jun 3" — compact date for search results / DM previews. */
export function formatShortDate(ts: number): string {
  const d = new Date(ts);
  return `${MONTHS[d.getUTCMonth()].slice(0, 3)} ${d.getUTCDate()}`;
}

/** Stable per-UTC-day key, used to group messages under day dividers. */
export function dayKey(ts: number): string {
  const d = new Date(ts);
  return `${d.getUTCFullYear()}-${d.getUTCMonth()}-${d.getUTCDate()}`;
}
