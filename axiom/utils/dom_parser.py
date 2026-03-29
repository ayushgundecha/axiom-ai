"""Simplified DOM extraction for LLM consumption.

Raw HTML is too verbose for LLM context windows — script tags, style blocks,
and irrelevant attributes waste tokens. This parser extracts only interactive
elements (input, button, a, select, textarea, label) and semantic structure
(h1-h4, p, li, span, div, form) with important attributes.

This mirrors the accessibility tree approach used by OSWorld and Deeptune's
computer-use environments. A full HTML page might be 50K tokens; the
simplified DOM is typically under 2K tokens.
"""

from __future__ import annotations

from html.parser import HTMLParser

# Tags that represent interactive elements agents can act on
INTERACTIVE_TAGS = frozenset({"input", "button", "a", "select", "textarea", "label", "option"})

# Tags that provide semantic structure
SEMANTIC_TAGS = frozenset({"h1", "h2", "h3", "h4", "p", "li", "span", "div", "form", "ul", "ol"})

# Tags whose entire content (including children) should be skipped
SKIP_TAGS = frozenset({"script", "style", "meta", "link", "head", "noscript", "svg"})

# Attributes worth preserving (everything else is noise)
IMPORTANT_ATTRS = frozenset({
    "id", "class", "type", "placeholder", "value", "href", "name",
    "role", "aria-label", "checked", "disabled", "data-testid",
})

# Max characters before text gets truncated
_TEXT_TRUNCATE_LEN = 100


class SimplifiedDOMParser(HTMLParser):
    """Extracts a simplified, token-efficient DOM from HTML.

    Only keeps interactive elements and semantic structure. Strips
    script, style, meta, link, head, and noscript tags entirely.
    """

    def __init__(self, max_depth: int = 15) -> None:
        super().__init__()
        self.result_lines: list[str] = []
        self._depth = 0
        self._skip_until: str | None = None
        self._max_depth = max_depth

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # If we're inside a skip block, ignore everything
        if self._skip_until is not None:
            return

        # Enter skip mode for tags like script, style, meta, etc.
        if tag in SKIP_TAGS:
            self._skip_until = tag
            return

        # Only process interactive and semantic tags
        if tag not in INTERACTIVE_TAGS and tag not in SEMANTIC_TAGS:
            return

        # Respect max depth
        if self._depth >= self._max_depth:
            return

        indent = "  " * self._depth
        attr_parts: list[str] = []
        for key, val in attrs:
            if key in IMPORTANT_ATTRS and val is not None:
                attr_parts.append(f'{key}="{val}"')
        attr_str = " " + " ".join(attr_parts) if attr_parts else ""
        self.result_lines.append(f"{indent}<{tag}{attr_str}>")
        self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        # Exit skip mode when we see the matching close tag
        if self._skip_until == tag:
            self._skip_until = None
            return

        if self._skip_until is not None:
            return

        if tag in INTERACTIVE_TAGS or tag in SEMANTIC_TAGS:
            self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_until is not None:
            return

        text = data.strip()
        if not text:
            return

        if self._depth > self._max_depth:
            return

        indent = "  " * self._depth
        if len(text) > _TEXT_TRUNCATE_LEN:
            text = text[:_TEXT_TRUNCATE_LEN] + "..."
        self.result_lines.append(f"{indent}{text}")


def extract_simplified_dom(html: str, max_depth: int = 15) -> str:
    """Convert raw HTML into a simplified DOM tree.

    Only keeps interactive elements and semantic structure.
    Drastically reduces token count for LLM consumption.

    Args:
        html: Raw HTML string.
        max_depth: Maximum nesting depth to include (default 15).

    Returns:
        Simplified DOM as a string, one element per line.
    """
    if not html.strip():
        return ""

    parser = SimplifiedDOMParser(max_depth=max_depth)
    parser.feed(html)
    return "\n".join(parser.result_lines)
