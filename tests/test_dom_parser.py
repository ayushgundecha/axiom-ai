"""Tests for axiom/utils/dom_parser.py — simplified DOM extraction.

Tests that raw HTML is correctly transformed into token-efficient DOM trees.
Written TDD-style before implementation.
"""



class TestSimplifiedDOMParser:
    """DOM parser must extract only interactive and semantic elements."""

    def test_extracts_interactive_elements(self) -> None:
        from axiom.utils.dom_parser import extract_simplified_dom

        html = """
        <html>
        <body>
            <form>
                <input type="text" id="name" placeholder="Enter name" />
                <button type="submit">Submit</button>
            </form>
        </body>
        </html>
        """
        result = extract_simplified_dom(html)

        assert "input" in result
        assert "button" in result
        assert 'placeholder="Enter name"' in result
        assert "Submit" in result

    def test_strips_script_tags(self) -> None:
        from axiom.utils.dom_parser import extract_simplified_dom

        html = """
        <html>
        <body>
            <script>var x = 'secret';</script>
            <p>Visible text</p>
        </body>
        </html>
        """
        result = extract_simplified_dom(html)

        assert "secret" not in result
        assert "script" not in result
        assert "Visible text" in result

    def test_strips_style_tags(self) -> None:
        from axiom.utils.dom_parser import extract_simplified_dom

        html = """
        <html>
        <head><style>.foo { color: red; }</style></head>
        <body><p>Content</p></body>
        </html>
        """
        result = extract_simplified_dom(html)

        assert "color: red" not in result
        assert "Content" in result

    def test_preserves_data_testid(self) -> None:
        from axiom.utils.dom_parser import extract_simplified_dom

        html = """
        <div>
            <input data-testid="todo-input" type="text" />
            <button data-testid="add-button">Add</button>
        </div>
        """
        result = extract_simplified_dom(html)

        assert 'data-testid="todo-input"' in result
        assert 'data-testid="add-button"' in result

    def test_preserves_semantic_structure(self) -> None:
        from axiom.utils.dom_parser import extract_simplified_dom

        html = """
        <html>
        <body>
            <h1>Title</h1>
            <p>Paragraph text</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </body>
        </html>
        """
        result = extract_simplified_dom(html)

        assert "h1" in result
        assert "Title" in result
        assert "li" in result
        assert "Item 1" in result
        assert "Item 2" in result

    def test_preserves_important_attributes(self) -> None:
        from axiom.utils.dom_parser import extract_simplified_dom

        html = """
        <a href="/about" id="nav-about" class="nav-link">About</a>
        <input type="checkbox" name="agree" checked />
        <select name="color">
            <option value="red">Red</option>
        </select>
        """
        result = extract_simplified_dom(html)

        assert 'href="/about"' in result
        assert 'id="nav-about"' in result
        assert 'type="checkbox"' in result

    def test_truncates_long_text(self) -> None:
        from axiom.utils.dom_parser import extract_simplified_dom

        long_text = "A" * 200
        html = f"<p>{long_text}</p>"
        result = extract_simplified_dom(html)

        # Text should be truncated (not full 200 chars)
        assert "..." in result
        assert len(result) < len(long_text)

    def test_max_depth_limits_nesting(self) -> None:
        from axiom.utils.dom_parser import extract_simplified_dom

        # Create deeply nested HTML
        html = "<div>" * 20 + "<p>Deep</p>" + "</div>" * 20
        result = extract_simplified_dom(html, max_depth=5)

        # Should not go infinitely deep
        # The exact behavior depends on implementation,
        # but the output should be bounded
        assert len(result.split("\n")) < 25

    def test_empty_html_returns_empty(self) -> None:
        from axiom.utils.dom_parser import extract_simplified_dom

        result = extract_simplified_dom("")
        assert result == "" or result.strip() == ""

    def test_todo_app_html_structure(self) -> None:
        """Test with HTML resembling the actual todo app."""
        from axiom.utils.dom_parser import extract_simplified_dom

        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Todo App</title>
            <link rel="stylesheet" href="styles.css">
        </head>
        <body>
            <div class="app">
                <h1>Todos</h1>
                <form id="todo-form" data-testid="todo-form">
                    <input type="text" id="todo-input" data-testid="todo-input"
                           placeholder="What needs to be done?" />
                    <button type="submit" data-testid="add-button">Add</button>
                </form>
                <ul id="todo-list" data-testid="todo-list">
                    <li data-testid="todo-item-1">
                        <input type="checkbox" data-testid="toggle-1" />
                        <span data-testid="todo-text-1">Buy milk</span>
                        <button data-testid="delete-1">x</button>
                    </li>
                </ul>
                <div id="stats" data-testid="stats">1/1 completed</div>
            </div>
            <script src="app.js"></script>
        </body>
        </html>
        """
        result = extract_simplified_dom(html)

        # Must include interactive elements
        assert "todo-input" in result
        assert "add-button" in result
        assert "toggle-1" in result
        assert "delete-1" in result

        # Must include semantic content
        assert "Todos" in result
        assert "Buy milk" in result

        # Must NOT include script references
        assert "app.js" not in result

        # Must NOT include meta/link tags
        assert "charset" not in result
        assert "stylesheet" not in result
