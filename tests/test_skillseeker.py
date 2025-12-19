"""Tests for Skillseeker"""

import pytest
import json
import os
from dataclasses import asdict
from unittest.mock import Mock, patch, AsyncMock
import skillseeker.skillseeker as skillseeker_module

from skillseeker import (
    Aggregator,
    Resource,
    ResourceType,
    Source,
)
from skillseeker import Verbosity, get_verbosity, set_verbosity
from skillseeker.skillseeker import (
    FirecrawlTracker,
    SkillsmpClient,
    SmitheryClient,
    GlamaClient,
)


class TestResource:
    """Test the Resource dataclass"""

    def test_resource_creation(self):
        r = Resource(
            name="test-skill",
            description="A test skill",
            type="skill",
            source="skillsmp",
            url="https://example.com/skill"
        )
        assert r.name == "test-skill"
        assert r.type == "skill"
        assert r.stars is None

    def test_resource_with_all_fields(self):
        r = Resource(
            name="complete-skill",
            description="Full skill",
            type="mcp_server",
            source="mcp_so",
            url="https://mcp.so/server/1",
            author="@anthropic",
            github_url="https://github.com/anthropic/server",
            stars=1500,
            downloads=50000,
            last_updated="2024-01-01",
            install_command="npx @anthropic/server",
            category="database",
            language="TypeScript",
            tags=["db", "postgres"]
        )
        assert r.stars == 1500
        assert r.tags == ["db", "postgres"]

    def test_resource_to_dict(self):
        r = Resource(
            name="dict-test",
            description="Test",
            type="plugin",
            source="skillsmp",
            url="https://test.com"
        )
        d = asdict(r)
        assert isinstance(d, dict)
        assert d["name"] == "dict-test"
        assert "stars" in d


class TestResourceTypes:
    """Test ResourceType enum"""

    def test_all_types_exist(self):
        assert ResourceType.SKILL.value == "skill"
        assert ResourceType.PLUGIN.value == "plugin"
        assert ResourceType.MCP_SERVER.value == "mcp_server"
        assert ResourceType.AGENT.value == "agent"
        assert ResourceType.COMMAND.value == "command"
        assert ResourceType.ALL.value == "all"


class TestSources:
    """Test Source enum"""

    def test_all_sources_exist(self):
        assert Source.SKILLSMP.value == "skillsmp"
        assert Source.SMITHERY.value == "smithery"
        assert Source.GLAMA.value == "glama"
        assert Source.MCP_SO.value == "mcp_so"
        assert Source.ALL.value == "all"


class TestFirecrawlTracker:
    """Test FireCrawl usage tracking"""

    def test_tracker_init(self):
        tracker = FirecrawlTracker(limit=5)
        assert tracker.limit == 5
        assert tracker.usage_count == 0
        assert tracker.permission_granted is False

    def test_can_use_within_limit(self):
        tracker = FirecrawlTracker(limit=5)
        assert tracker.can_use() is True
        tracker.increment()
        tracker.increment()
        assert tracker.can_use() is True

    def test_can_use_at_limit(self):
        tracker = FirecrawlTracker(limit=2)
        tracker.increment()
        tracker.increment()
        assert tracker.can_use() is False

    def test_unlimited_mode(self):
        tracker = FirecrawlTracker(limit=0)
        for _ in range(100):
            tracker.increment()
        assert tracker.can_use() is True

    def test_permission_allows_continued_use(self):
        tracker = FirecrawlTracker(limit=1)
        tracker.increment()
        assert tracker.can_use() is False
        tracker.permission_granted = True
        assert tracker.can_use() is True


class TestVerbosityControls:
    """Test verbosity flag handling"""

    def test_set_and_get_verbosity(self):
        original = get_verbosity()
        try:
            set_verbosity(Verbosity.WARNING)
            assert get_verbosity() == Verbosity.WARNING
        finally:
            set_verbosity(original)

    def test_filtered_print_respects_levels(self, monkeypatch):
        original = get_verbosity()
        set_verbosity(Verbosity.WARNING)

        captured = []

        def fake_print(*args, **kwargs):
            captured.append(args[0] if args else "")

        monkeypatch.setattr(skillseeker_module.console, "_raw_console_print", fake_print)

        try:
            skillseeker_module.console.print("[dim]info message[/dim]")
            skillseeker_module.console.print("[yellow]warn message[/yellow]")
            skillseeker_module.console.print("[red]error message[/red]")

            assert captured == ["[yellow]warn message[/yellow]", "[red]error message[/red]"]
        finally:
            set_verbosity(original)

    def test_explicit_verbosity_override(self, monkeypatch):
        original = get_verbosity()
        set_verbosity(Verbosity.ERROR)

        captured = []

        def fake_print(*args, **kwargs):
            captured.append(args[0] if args else "")

        monkeypatch.setattr(skillseeker_module.console, "_raw_console_print", fake_print)

        try:
            skillseeker_module.console.print("details", _verbosity=Verbosity.ERROR)
            skillseeker_module.console.print("hidden info", _verbosity=Verbosity.INFO)

            assert captured == ["details"]
        finally:
            set_verbosity(original)


class TestAggregator:
    """Test Aggregator class"""

    def test_init_without_api_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            agg = Aggregator()
            assert agg.firecrawl_app is None
            assert agg.skillsmp_api_key is None
            assert agg.smithery_api_key is None

    def test_init_with_env_keys(self):
        with patch.dict(os.environ, {
            "FIRECRAWL_API_KEY": "fc-key",
            "SKILLSMP_API_KEY": "sm-key",
            "SMITHERY_API_KEY": "st-key"
        }):
            agg = Aggregator()
            assert agg.firecrawl_api_key == "fc-key"
            assert agg.skillsmp_api_key == "sm-key"
            assert agg.smithery_api_key == "st-key"

    def test_init_with_direct_keys(self):
        agg = Aggregator(
            firecrawl_api_key="fc-direct",
            skillsmp_api_key="sm-direct"
        )
        assert agg.firecrawl_api_key == "fc-direct"
        assert agg.skillsmp_api_key == "sm-direct"

    def test_filter_by_type(self):
        agg = Aggregator()
        agg.results = [
            Resource(name="skill1", description="", type="skill", source="test", url=""),
            Resource(name="mcp1", description="", type="mcp_server", source="test", url=""),
            Resource(name="skill2", description="", type="skill", source="test", url=""),
        ]

        filtered = agg.filter_results(resource_type=ResourceType.SKILL)
        assert len(filtered) == 2
        assert all(r.type == "skill" for r in filtered)

    def test_filter_by_stars(self):
        agg = Aggregator()
        agg.results = [
            Resource(name="r1", description="", type="skill", source="test", url="", stars=100),
            Resource(name="r2", description="", type="skill", source="test", url="", stars=50),
            Resource(name="r3", description="", type="skill", source="test", url="", stars=200),
            Resource(name="r4", description="", type="skill", source="test", url=""),  # None
        ]

        filtered = agg.filter_results(min_stars=100)
        assert len(filtered) == 2
        assert all(r.stars >= 100 for r in filtered)

    def test_filter_by_category(self):
        agg = Aggregator()
        agg.results = [
            Resource(name="r1", description="", type="mcp_server", source="test", url="", category="database"),
            Resource(name="r2", description="", type="mcp_server", source="test", url="", category="cloud"),
            Resource(name="r3", description="", type="mcp_server", source="test", url="", category="Database Tools"),
        ]

        filtered = agg.filter_results(category="database")
        assert len(filtered) == 2  # Case insensitive, partial match

    def test_sort_by_stars(self):
        agg = Aggregator()
        results = [
            Resource(name="r1", description="", type="skill", source="test", url="", stars=50),
            Resource(name="r2", description="", type="skill", source="test", url="", stars=200),
            Resource(name="r3", description="", type="skill", source="test", url="", stars=100),
        ]

        sorted_results = agg.sort_results(results, by="stars", reverse=True)
        assert sorted_results[0].stars == 200
        assert sorted_results[1].stars == 100
        assert sorted_results[2].stars == 50

    def test_sort_with_none_values(self):
        agg = Aggregator()
        results = [
            Resource(name="r1", description="", type="skill", source="test", url="", stars=50),
            Resource(name="r2", description="", type="skill", source="test", url=""),  # None
            Resource(name="r3", description="", type="skill", source="test", url="", stars=100),
        ]

        sorted_results = agg.sort_results(results, by="stars", reverse=True)
        # None values should sort to end
        assert sorted_results[0].stars == 100
        assert sorted_results[1].stars == 50

    def test_firecrawl_usage_tracking(self):
        agg = Aggregator(firecrawl_limit=5)
        usage = agg.get_firecrawl_usage()
        assert usage["usage"] == 0
        assert usage["limit"] == 5
        assert usage["exceeded"] is False


class TestUnifiedSchema:
    """Test the unified output schema"""

    def test_export_schema_format(self):
        unified_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "type": {"type": "string", "enum": [t.value for t in ResourceType if t != ResourceType.ALL]},
                "source": {"type": "string", "enum": [s.value for s in Source if s != Source.ALL]},
                "url": {"type": "string", "format": "uri"},
                "author": {"type": "string"},
                "github_url": {"type": "string", "format": "uri"},
                "stars": {"type": "integer"},
                "downloads": {"type": "integer"},
                "last_updated": {"type": "string", "format": "date-time"},
                "install_command": {"type": "string"},
                "category": {"type": "string"},
                "language": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["name", "description", "type", "source", "url"]
        }

        # Validate it's valid JSON
        json_str = json.dumps(unified_schema)
        parsed = json.loads(json_str)
        assert parsed["type"] == "object"
        assert len(parsed["required"]) == 5


@pytest.mark.asyncio
class TestSkillsmpClient:
    """Test SkillsMP API client"""

    async def test_search_without_api_key(self):
        client = SkillsmpClient(api_key=None)
        results = await client.search("test")
        assert results == []
        await client.close()

    async def test_search_with_mock_response(self):
        client = SkillsmpClient(api_key="test-key")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"name": "test-skill", "description": "A test skill", "author": "test-author"}
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch.object(client.client, "get", new_callable=AsyncMock, return_value=mock_response):
            results = await client.search("test")
            assert len(results) == 1
            assert results[0].name == "test-skill"
            assert results[0].source == "skillsmp"

        await client.close()


@pytest.mark.asyncio
class TestSmitheryClient:
    """Test Smithery API client"""

    async def test_search_without_api_key(self):
        client = SmitheryClient(api_key=None)
        results = await client.search("test")
        assert results == []
        await client.close()

    async def test_search_with_mock_response(self):
        client = SmitheryClient(api_key="test-key")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "servers": [
                {
                    "qualifiedName": "owner/test-server",
                    "displayName": "Test Server",
                    "description": "A test MCP server"
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch.object(client.client, "get", new_callable=AsyncMock, return_value=mock_response):
            results = await client.search("test")
            assert len(results) == 1
            assert results[0].name == "Test Server"
            assert results[0].source == "smithery"
            assert results[0].type == "mcp_server"

        await client.close()


@pytest.mark.asyncio
class TestAsyncAggregator:
    """Test async aggregation functionality"""

    async def test_search_all_with_no_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            agg = Aggregator()
            results = await agg.search_all([Source.SKILLSMP, Source.SMITHERY], "test")
            assert results == []
            await agg.close()

    async def test_search_expands_all_source(self):
        agg = Aggregator()

        # Mock all search methods to return empty
        agg.skillsmp.search = AsyncMock(return_value=[])
        agg.smithery.search = AsyncMock(return_value=[])
        agg.glama.search_via_scrape = AsyncMock(return_value=[])

        await agg.search_all([Source.ALL], "test")

        # Should have tried all non-ALL sources
        await agg.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
