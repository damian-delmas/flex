"""
Tests for flex/retrieve/presets.py — DB-backed presets

Tests PresetLoader: read from _presets table, parse annotations, interpolate params, execute.

Actual API:
  PresetLoader(db)
  .execute(db, name, params={}) -> list[dict] | list[{query, results}]
  ._parse(text, name) -> dict

  install_presets(db, preset_dir) -> bake .sql files into _presets table

Run with: pytest tests/test_presets.py -v
"""
import sqlite3
import pytest
from pathlib import Path


def _can_import():
    try:
        from flex.retrieve.presets import PresetLoader, install_presets
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _can_import(),
    reason="flex.presets not yet implemented"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def loader(qmem_cell):
    """PresetLoader backed by qmem_cell's _presets table."""
    from flex.retrieve.presets import PresetLoader
    return PresetLoader(qmem_cell)


@pytest.fixture
def preset_dir(tmp_path):
    """Create a temp directory with .sql files for install_presets testing."""
    d = tmp_path / "presets"
    d.mkdir()

    (d / "test-single.sql").write_text("""\
-- @name: test-single
-- @description: A test single-query preset
-- @params: min_centrality (default: 0.3)
SELECT source_id, centrality
FROM _enrich_source_graph
WHERE centrality >= :min_centrality
ORDER BY centrality DESC
""")

    (d / "test-multi.sql").write_text("""\
-- @name: test-multi
-- @description: A test multi-query preset
-- @multi: true

-- @query: counts
SELECT COUNT(*) as n FROM _raw_chunks;

-- @query: sources
SELECT source_id, doc_type FROM _raw_sources ORDER BY file_date DESC;
""")

    return d


# =============================================================================
# Parsing
# =============================================================================

class TestParsing:
    """Annotation parsing from _presets sql text."""

    def test_parse_single_query(self, loader):
        preset = loader.load('hub-sources')
        assert len(preset['queries']) == 1
        assert not preset['multi']

    def test_parse_multi_query(self, loader):
        preset = loader.load('overview')
        assert preset['multi'] is True
        assert len(preset['queries']) == 2

    def test_parse_query_names(self, loader):
        preset = loader.load('overview')
        names = [q['name'] for q in preset['queries']]
        assert 'counts' in names
        assert 'sources' in names

    def test_list_presets(self, loader):
        names = loader.list_presets()
        assert 'hub-sources' in names
        assert 'overview' in names
        assert 'all-chunks' in names


# =============================================================================
# Execution
# =============================================================================

class TestExecution:
    """Execute presets against a live cell."""

    def test_single_query_returns_list(self, loader, qmem_cell):
        results = loader.execute(qmem_cell, 'all-chunks')
        assert isinstance(results, list)
        assert len(results) == 9  # qmem_cell has 9 chunks

    def test_param_interpolation(self, loader, qmem_cell):
        results = loader.execute(qmem_cell, 'hub-sources', {'min_centrality': 0.5})
        assert isinstance(results, list)
        # Only src-arch has centrality >= 0.5 (0.85)
        assert len(results) >= 1
        assert results[0]['source_id'] == 'src-arch'

    def test_multi_query_returns_list_of_query_results(self, loader, qmem_cell):
        results = loader.execute(qmem_cell, 'overview')
        assert isinstance(results, list)
        # Each entry has 'query' name and 'results' list
        names = [r['query'] for r in results]
        assert 'counts' in names
        assert 'sources' in names
        counts_entry = next(r for r in results if r['query'] == 'counts')
        assert isinstance(counts_entry['results'], list)

    def test_missing_preset_raises(self, loader, qmem_cell):
        with pytest.raises(KeyError):
            loader.execute(qmem_cell, 'nonexistent-preset')

    def test_result_is_dicts(self, loader, qmem_cell):
        results = loader.execute(qmem_cell, 'all-chunks')
        assert isinstance(results[0], dict)
        assert 'id' in results[0]
        assert 'content' in results[0]


# =============================================================================
# Defaults
# =============================================================================

class TestDefaults:
    """@params default parsing and application."""

    def test_defaults_parsed(self):
        from flex.retrieve.presets import PresetLoader
        parsed = PresetLoader._parse(
            "-- @params: limit (default: 15), offset (default: 0)\nSELECT 1;",
            'test')
        assert parsed['defaults'] == {'limit': 15, 'offset': 0}

    def test_defaults_applied(self, qmem_cell):
        from flex.retrieve.presets import PresetLoader
        # Insert a preset with defaults
        qmem_cell.execute(
            "INSERT OR REPLACE INTO _presets (name, description, params, sql) VALUES (?, ?, '', ?)",
            ('limited', 'Test with defaults',
             "-- @params: limit (default: 3)\nSELECT id FROM _raw_chunks LIMIT :limit"))
        qmem_cell.commit()
        loader = PresetLoader(qmem_cell)
        results = loader.execute(qmem_cell, 'limited')
        assert len(results) == 3

    def test_explicit_params_override_defaults(self, qmem_cell):
        from flex.retrieve.presets import PresetLoader
        qmem_cell.execute(
            "INSERT OR REPLACE INTO _presets (name, description, params, sql) VALUES (?, ?, '', ?)",
            ('limited', 'Test with defaults',
             "-- @params: limit (default: 3)\nSELECT id FROM _raw_chunks LIMIT :limit"))
        qmem_cell.commit()
        loader = PresetLoader(qmem_cell)
        results = loader.execute(qmem_cell, 'limited', {'limit': 1})
        assert len(results) == 1


# =============================================================================
# install_presets
# =============================================================================

class TestInstallPresets:
    """Bake .sql files into _presets table."""

    def test_install_from_dir(self, qmem_cell, preset_dir):
        from flex.retrieve.presets import install_presets
        install_presets(qmem_cell, preset_dir)
        # Verify presets were inserted
        rows = qmem_cell.execute("SELECT name FROM _presets WHERE name LIKE 'test-%'").fetchall()
        names = [r[0] for r in rows]
        assert 'test-single' in names
        assert 'test-multi' in names

    def test_installed_presets_executable(self, qmem_cell, preset_dir):
        from flex.retrieve.presets import install_presets, PresetLoader
        install_presets(qmem_cell, preset_dir)
        loader = PresetLoader(qmem_cell)
        results = loader.execute(qmem_cell, 'test-single', {'min_centrality': 0.5})
        assert len(results) >= 1

    def test_install_preserves_description(self, qmem_cell, preset_dir):
        from flex.retrieve.presets import install_presets
        install_presets(qmem_cell, preset_dir)
        row = qmem_cell.execute(
            "SELECT description FROM _presets WHERE name = 'test-single'").fetchone()
        assert row[0] == 'A test single-query preset'

    def test_install_nonexistent_dir(self, qmem_cell, tmp_path):
        from flex.retrieve.presets import install_presets
        # Should not raise
        install_presets(qmem_cell, tmp_path / "nope")

    def test_install_replaces_existing(self, qmem_cell, preset_dir):
        from flex.retrieve.presets import install_presets
        install_presets(qmem_cell, preset_dir)
        # Modify file and re-install
        (preset_dir / "test-single.sql").write_text(
            "-- @name: test-single\n-- @description: Updated\nSELECT 1;")
        install_presets(qmem_cell, preset_dir)
        row = qmem_cell.execute(
            "SELECT description FROM _presets WHERE name = 'test-single'").fetchone()
        assert row[0] == 'Updated'
