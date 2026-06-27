/**
 * AxiomChat design tokens.
 *
 * Dark-first, Slack-grade. Every component pulls from these semantic tokens so
 * the product reads as intentional rather than ad hoc. Colors are flat hex (no
 * runtime CSS-var indirection) so the screenshot output is deterministic.
 */
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Surfaces (darkest -> raised)
        base: "#0b0d12", // app background
        sidebar: "#14161d", // left rail
        panel: "#0f1117", // main column
        raised: "#171a22", // hovered rows / cards
        overlay: "#1b1f29", // popovers / menus
        line: "#242a36", // hairline borders
        "line-soft": "#1c212b",
        // Text
        ink: "#e8eaf0", // primary
        "ink-2": "#aab2c2", // secondary
        "ink-3": "#727b8d", // muted / metadata
        // Brand accent (modern indigo/violet)
        accent: "#6366f1",
        "accent-hover": "#818cf8",
        "accent-soft": "rgba(99,102,241,0.16)",
        // Semantic
        mention: "rgba(250,204,21,0.16)",
        "mention-ink": "#fde68a",
        danger: "#ef4444",
        success: "#22c55e",
        warning: "#eab308",
        // Presence dots
        "presence-active": "#22c55e",
        "presence-away": "#eab308",
        "presence-dnd": "#ef4444",
        "presence-offline": "#5b6472",
      },
      fontFamily: {
        // System stack only — no network fonts (keeps screenshots reproducible).
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      fontSize: {
        xs: ["11px", "16px"],
        sm: ["13px", "18px"],
        base: ["15px", "22px"],
        lg: ["17px", "24px"],
        xl: ["20px", "28px"],
        "2xl": ["24px", "32px"],
      },
      borderRadius: {
        sm: "6px",
        md: "8px",
        lg: "12px",
        xl: "16px",
      },
      boxShadow: {
        card: "0 1px 2px rgba(0,0,0,0.4)",
        popover: "0 8px 28px rgba(0,0,0,0.55)",
        focus: "0 0 0 2px rgba(99,102,241,0.55)",
      },
      transitionDuration: {
        DEFAULT: "150ms",
      },
    },
  },
  plugins: [],
};
