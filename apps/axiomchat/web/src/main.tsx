import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { App } from "./components/App";
import { hydrate } from "./store";

// Screenshot mode: ?screenshot=1 adds [data-screenshot] to <html>, which the
// stylesheet uses to kill transitions/animations for byte-stable captures.
if (new URLSearchParams(window.location.search).has("screenshot")) {
  document.documentElement.setAttribute("data-screenshot", "");
}

// Kick off the initial data load before first paint.
void hydrate();

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
