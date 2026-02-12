"""
TDD Tests for flexsearch/compile/docpac.py — Plan 1

Tests parse_docpac(): walk a doc-pac directory, map folders to semantic metadata.

Contract:
  parse_docpac(root, facet=None, pattern='**/*.md') -> list[DocPacEntry]
  DocPacEntry: path, temporal, doc_type, facet, skip

These tests WILL FAIL until Plan 1 code is written.

Run with: pytest tests/test_docpac.py -v
"""
import pytest
from pathlib import Path


def _can_import():
    try:
        from flexsearch.compile.docpac import parse_docpac, DocPacEntry
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _can_import(),
    reason="flexsearch.compile.docpac not yet implemented (Plan 1)"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def docpac_tree(tmp_path):
    """Create a realistic doc-pac folder structure with .md files."""
    root = tmp_path / "context"
    # changes/code — past, changelog
    (root / "changes" / "code").mkdir(parents=True)
    (root / "changes" / "code" / "260211-refactor.md").write_text("# Refactor log")

    # changes/design — future, design
    (root / "changes" / "design").mkdir(parents=True)
    (root / "changes" / "design" / "260211-new-idea.md").write_text("# Design idea")

    # changes/testing — past, testing
    (root / "changes" / "testing").mkdir(parents=True)
    (root / "changes" / "testing" / "260211-tests.md").write_text("# Test results")

    # current — present, architecture
    (root / "current").mkdir(parents=True)
    (root / "current" / "architecture.md").write_text("# Architecture")
    (root / "current" / "schema.md").write_text("# Schema")

    # intended/proximate — future, plan
    (root / "intended" / "proximate").mkdir(parents=True)
    (root / "intended" / "proximate" / "migration.md").write_text("# Migration plan")

    # knowledge — exogenous, knowledge
    (root / "knowledge").mkdir(parents=True)
    (root / "knowledge" / "sqlite-fts.md").write_text("# SQLite FTS5 reference")

    # buffer — skip
    (root / "buffer").mkdir(parents=True)
    (root / "buffer" / "scratch.md").write_text("# Scratch notes")

    # _raw — skip
    (root / "_raw").mkdir(parents=True)
    (root / "_raw" / "dump.md").write_text("raw data")

    return root


@pytest.fixture
def nested_docpac(tmp_path):
    """Doc-pac with a recursive plan inside intended/proximate/."""
    root = tmp_path / "context"

    # Top-level current
    (root / "current").mkdir(parents=True)
    (root / "current" / "arch.md").write_text("# Top arch")

    # Nested plan: intended/proximate/sql-first/ IS a doc-pac
    plan = root / "intended" / "proximate" / "sql-first"
    (plan / "changes" / "code").mkdir(parents=True)
    (plan / "changes" / "code" / "260211.md").write_text("# SQL change")
    (plan / "current").mkdir(parents=True)
    (plan / "current" / "status.md").write_text("# SQL status")

    return root


# =============================================================================
# Folder Mapping Tests
# =============================================================================

class TestFolderMapping:
    """Folder names map to (temporal, doc_type) tuples."""

    def test_changes_code_is_past_changelog(self, docpac_tree):
        from flexsearch.compile.docpac import parse_docpac
        entries = parse_docpac(str(docpac_tree))
        code_entries = [e for e in entries if 'changes/code' in e.path]
        assert len(code_entries) == 1
        assert code_entries[0].temporal == 'past'
        assert code_entries[0].doc_type == 'changelog'

    def test_changes_design_is_future_design(self, docpac_tree):
        from flexsearch.compile.docpac import parse_docpac
        entries = parse_docpac(str(docpac_tree))
        design_entries = [e for e in entries if 'changes/design' in e.path]
        assert len(design_entries) == 1
        assert design_entries[0].temporal == 'future'
        assert design_entries[0].doc_type == 'design'

    def test_current_is_present_architecture(self, docpac_tree):
        from flexsearch.compile.docpac import parse_docpac
        entries = parse_docpac(str(docpac_tree))
        current_entries = [e for e in entries if '/current/' in e.path and not e.skip]
        assert len(current_entries) == 2  # architecture.md + schema.md
        for e in current_entries:
            assert e.temporal == 'present'
            assert e.doc_type == 'architecture'

    def test_intended_proximate_is_future_plan(self, docpac_tree):
        from flexsearch.compile.docpac import parse_docpac
        entries = parse_docpac(str(docpac_tree))
        plan_entries = [e for e in entries if 'intended/proximate' in e.path]
        assert len(plan_entries) >= 1
        for e in plan_entries:
            assert e.temporal == 'future'
            assert e.doc_type == 'plan'

    def test_knowledge_is_exogenous(self, docpac_tree):
        from flexsearch.compile.docpac import parse_docpac
        entries = parse_docpac(str(docpac_tree))
        know_entries = [e for e in entries if '/knowledge/' in e.path]
        assert len(know_entries) == 1
        assert know_entries[0].temporal == 'exogenous'
        assert know_entries[0].doc_type == 'knowledge'


# =============================================================================
# Skip Folders
# =============================================================================

class TestSkipFolders:
    """buffer/, _raw/, _qmem/, cache/ are marked skip=True."""

    def test_buffer_is_skipped(self, docpac_tree):
        from flexsearch.compile.docpac import parse_docpac
        entries = parse_docpac(str(docpac_tree))
        buffer_entries = [e for e in entries if '/buffer/' in e.path]
        assert all(e.skip for e in buffer_entries), "buffer/ files should be skipped"

    def test_raw_is_skipped(self, docpac_tree):
        from flexsearch.compile.docpac import parse_docpac
        entries = parse_docpac(str(docpac_tree))
        raw_entries = [e for e in entries if '/_raw/' in e.path]
        assert all(e.skip for e in raw_entries), "_raw/ files should be skipped"

    def test_indexable_excludes_skipped(self, docpac_tree):
        from flexsearch.compile.docpac import parse_docpac
        entries = parse_docpac(str(docpac_tree))
        indexable = [e for e in entries if not e.skip]
        for e in indexable:
            assert '/buffer/' not in e.path
            assert '/_raw/' not in e.path


# =============================================================================
# Recursion Tests
# =============================================================================

class TestRecursion:
    """Nested plans in intended/proximate/{name}/ recurse with facet={name}."""

    def test_nested_plan_gets_facet(self, nested_docpac):
        from flexsearch.compile.docpac import parse_docpac
        entries = parse_docpac(str(nested_docpac))
        sql_entries = [e for e in entries if 'sql-first' in e.path]
        assert len(sql_entries) >= 2
        for e in sql_entries:
            assert e.facet == 'sql-first', f"Nested plan entry should have facet='sql-first', got '{e.facet}'"

    def test_nested_changelog_is_past(self, nested_docpac):
        from flexsearch.compile.docpac import parse_docpac
        entries = parse_docpac(str(nested_docpac))
        nested_code = [e for e in entries if 'sql-first/changes/code' in e.path]
        assert len(nested_code) == 1
        assert nested_code[0].temporal == 'past'
        assert nested_code[0].doc_type == 'changelog'

    def test_top_level_has_no_facet(self, nested_docpac):
        from flexsearch.compile.docpac import parse_docpac
        entries = parse_docpac(str(nested_docpac))
        top_entries = [e for e in entries if '/current/arch.md' in e.path]
        assert len(top_entries) == 1
        assert top_entries[0].facet is None


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge cases and return format."""

    def test_returns_list_of_dataclass(self, docpac_tree):
        from flexsearch.compile.docpac import parse_docpac, DocPacEntry
        entries = parse_docpac(str(docpac_tree))
        assert isinstance(entries, list)
        assert all(isinstance(e, DocPacEntry) for e in entries)

    def test_paths_are_absolute(self, docpac_tree):
        from flexsearch.compile.docpac import parse_docpac
        entries = parse_docpac(str(docpac_tree))
        for e in entries:
            assert Path(e.path).is_absolute(), f"Path should be absolute: {e.path}"

    def test_empty_directory(self, tmp_path):
        from flexsearch.compile.docpac import parse_docpac
        empty = tmp_path / "empty"
        empty.mkdir()
        entries = parse_docpac(str(empty))
        assert entries == []

    def test_custom_pattern(self, docpac_tree):
        from flexsearch.compile.docpac import parse_docpac
        # Only .md files by default, but allow override
        entries_md = parse_docpac(str(docpac_tree), pattern='**/*.md')
        assert len(entries_md) > 0
