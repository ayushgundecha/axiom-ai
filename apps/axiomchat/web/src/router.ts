/** Minimal hash router: #/channel/:id, #/dm/:id, #/search. */
import { useEffect, useState } from "react";

export type Route =
  | { kind: "channel"; id: string }
  | { kind: "dm"; id: string }
  | { kind: "search" };

export function parseHash(hash: string): Route | null {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (parts[0] === "channel" && parts[1]) return { kind: "channel", id: parts[1] };
  if (parts[0] === "dm" && parts[1]) return { kind: "dm", id: parts[1] };
  if (parts[0] === "search") return { kind: "search" };
  return null;
}

export function toHash(route: Route): string {
  switch (route.kind) {
    case "channel":
      return `#/channel/${route.id}`;
    case "dm":
      return `#/dm/${route.id}`;
    case "search":
      return `#/search`;
  }
}

export function navigate(route: Route): void {
  const next = toHash(route);
  if (window.location.hash !== next) window.location.hash = next;
}

/** Subscribe to the current route, re-rendering on hashchange. */
export function useRoute(): Route | null {
  const [route, setRoute] = useState<Route | null>(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = (): void => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}
