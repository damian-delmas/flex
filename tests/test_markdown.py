"""
Tests for flex.compile.markdown — COMPILE primitives.

Contract:
  normalize_headers(content) -> str (promoted headers, source untouched)
  extract_frontmatter(content) -> (dict, str) (YAML dict + body)
  split_sections(content, level=2) -> list[(title, content, position)]

Key principles tested:
  - Header normalization is ephemeral (promotes ## -> # if no H1)
  - Frontmatter extraction handles missing/malformed YAML gracefully
  - Section splitting respects header levels
  - Preamble before first header is captured

Run with: pytest tests/test_markdown.py -v
"""

import pytest
from flex.compile.markdown import normalize_headers, extract_frontmatter, split_sections


# =============================================================================
# Header Normalization
# =============================================================================

class TestNormalizeHeaders:
    """Ephemeral header promotion. Source files never modified."""

    def test_no_h1_promotes_h2_to_h1(self):
        content = "## Section A\ntext\n## Section B\nmore"
        result = normalize_headers(content)
        assert "# Section A" in result
        assert "# Section B" in result
        assert "## " not in result

    def test_has_h1_returns_unchanged(self):
        content = "# Title\n## Section\ntext"
        result = normalize_headers(content)
        assert result == content

    def test_promotes_h3_to_h2(self):
        content = "## Title\n### Sub\ntext"
        result = normalize_headers(content)
        assert "# Title" in result
        assert "## Sub" in result

    def test_no_headers_returns_unchanged(self):
        content = "Just plain text\nwith no headers"
        result = normalize_headers(content)
        assert result == content

    def test_empty_content(self):
        assert normalize_headers("") == ""

    def test_preserves_non_header_hashes(self):
        content = "## Title\nsome #tag in text"
        result = normalize_headers(content)
        assert "some #tag in text" in result


# =============================================================================
# Frontmatter Extraction
# =============================================================================

class TestExtractFrontmatter:
    """YAML frontmatter extraction."""

    def test_standard_frontmatter(self):
        content = "---\ntype: changelog\nstatus: active\n---\n# Title\nBody"
        fm, body = extract_frontmatter(content)
        assert fm['type'] == 'changelog'
        assert fm['status'] == 'active'
        assert body.startswith('# Title')

    def test_no_frontmatter(self):
        content = "# Title\nBody text"
        fm, body = extract_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_malformed_yaml(self):
        content = "---\n: invalid: yaml: [broken\n---\nBody"
        fm, body = extract_frontmatter(content)
        # Should gracefully return empty dict and original content
        assert isinstance(fm, dict)

    def test_empty_frontmatter(self):
        content = "---\n---\nBody"
        fm, body = extract_frontmatter(content)
        assert fm == {}
        assert body == "Body"

    def test_non_dict_frontmatter(self):
        content = "---\n- list\n- item\n---\nBody"
        fm, body = extract_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_frontmatter_with_keywords(self):
        content = '---\nkeywords: "sql, migration, plan"\n---\nBody'
        fm, body = extract_frontmatter(content)
        assert 'keywords' in fm


# =============================================================================
# Section Splitting
# =============================================================================

class TestSplitSections:
    """Split on ## headers into chunk-sized sections."""

    def test_basic_split(self):
        content = "## Section A\ntext a\n## Section B\ntext b"
        sections = split_sections(content)
        assert len(sections) == 2
        assert sections[0][0] == 'Section A'
        assert sections[1][0] == 'Section B'
        assert sections[0][2] == 0  # position
        assert sections[1][2] == 1

    def test_preamble_before_headers(self):
        content = "Some intro text\n\n## Section A\ntext"
        sections = split_sections(content)
        assert len(sections) == 2
        assert sections[0][0] == ''  # preamble has no title
        assert 'intro text' in sections[0][1]

    def test_no_headers(self):
        content = "Just plain text\nwith no headers at all"
        sections = split_sections(content)
        assert len(sections) == 1
        assert sections[0][0] == ''
        assert 'plain text' in sections[0][1]

    def test_h3_not_split(self):
        """Only splits on ## (level 2), not ### (level 3)."""
        content = "## Section\n### Subsection\ntext"
        sections = split_sections(content)
        assert len(sections) == 1
        assert '### Subsection' in sections[0][1]

    def test_section_content_includes_header(self):
        content = "## Title\nBody text"
        sections = split_sections(content)
        assert sections[0][1].startswith('## Title')

    def test_empty_content(self):
        sections = split_sections("")
        assert sections == []

    def test_custom_level(self):
        content = "# H1\ntext\n# H1B\nmore"
        sections = split_sections(content, level=1)
        assert len(sections) == 2
        assert sections[0][0] == 'H1'
        assert sections[1][0] == 'H1B'

    def test_positions_sequential(self):
        content = "## A\na\n## B\nb\n## C\nc"
        sections = split_sections(content)
        positions = [s[2] for s in sections]
        assert positions == [0, 1, 2]

    def test_after_normalization(self):
        """Integration: normalize then split.
        ## Title -> # Title (H1), ### Sub A -> ## Sub A (split boundary).
        """
        content = "## Title\n### Sub A\ntext\n### Sub B\nmore"
        normalized = normalize_headers(content)
        # After promotion: # Title, ## Sub A, ## Sub B
        sections = split_sections(normalized)
        assert len(sections) == 3
        assert sections[0][0] == ''  # preamble with # Title
        assert '# Title' in sections[0][1]
        assert sections[1][0] == 'Sub A'
        assert sections[2][0] == 'Sub B'
