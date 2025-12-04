"""Tests for Skillseeker"""

import pytest
import json
from dataclasses import asdict
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.insert(0, ".")

from skillseeker import (
    Aggregator,
    Resource,
    ResourceType,
    Source,
    SKILLSMP_SCHEMA,
    CLAUDE_PLUGINS_SCHEMA,
    MCP_SCHEMA,
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
            source="claude_plugins",
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
    """Test Source enum and configurations"""
    
    def test_all_sources_exist(self):
        assert Source.SKILLSMP.value == "skillsmp"
        assert Source.CLAUDE_PLUGINS.value == "claude_plugins"
        assert Source.CLAUDEMARKETPLACES.value == "claudemarketplaces"
        assert Source.MCP_SO.value == "mcp_so"
    
    def test_source_configs_complete(self):
        for source in Source:
            if source == Source.ALL:
                continue
            config = Aggregator.SOURCES[source]
            assert "url" in config
            assert "schema" in config
            assert "prompt" in config
            assert "type" in config


class TestSchemas:
    """Test extraction schemas"""
    
    def test_skillsmp_schema_valid(self):
        assert SKILLSMP_SCHEMA["type"] == "array"
        assert "items" in SKILLSMP_SCHEMA
        props = SKILLSMP_SCHEMA["items"]["properties"]
        assert "name" in props
        assert "description" in props
    
    def test_mcp_schema_valid(self):
        assert MCP_SCHEMA["type"] == "array"
        props = MCP_SCHEMA["items"]["properties"]
        assert "name" in props
        assert "language" in props
        assert "install_command" in props
    
    def test_schema_required_fields(self):
        for schema in [SKILLSMP_SCHEMA, CLAUDE_PLUGINS_SCHEMA, MCP_SCHEMA]:
            required = schema["items"]["required"]
            assert "name" in required
            assert "description" in required


class TestAggregator:
    """Test Aggregator class"""
    
    def test_init_requires_api_key(self):
        with pytest.raises(ValueError):
            with patch.dict("os.environ", {}, clear=True):
                Aggregator()
    
    def test_init_with_api_key(self):
        agg = Aggregator(api_key="test-key")
        assert agg.api_key == "test-key"
    
    def test_filter_by_type(self):
        agg = Aggregator(api_key="test-key")
        agg.results = [
            Resource(name="skill1", description="", type="skill", source="test", url=""),
            Resource(name="mcp1", description="", type="mcp_server", source="test", url=""),
            Resource(name="skill2", description="", type="skill", source="test", url=""),
        ]
        
        filtered = agg.filter_results(resource_type=ResourceType.SKILL)
        assert len(filtered) == 2
        assert all(r.type == "skill" for r in filtered)
    
    def test_filter_by_stars(self):
        agg = Aggregator(api_key="test-key")
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
        agg = Aggregator(api_key="test-key")
        agg.results = [
            Resource(name="r1", description="", type="mcp_server", source="test", url="", category="database"),
            Resource(name="r2", description="", type="mcp_server", source="test", url="", category="cloud"),
            Resource(name="r3", description="", type="mcp_server", source="test", url="", category="Database Tools"),
        ]
        
        filtered = agg.filter_results(category="database")
        assert len(filtered) == 2  # Case insensitive, partial match
    
    def test_sort_by_stars(self):
        agg = Aggregator(api_key="test-key")
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
        agg = Aggregator(api_key="test-key")
        results = [
            Resource(name="r1", description="", type="skill", source="test", url="", stars=50),
            Resource(name="r2", description="", type="skill", source="test", url=""),  # None
            Resource(name="r3", description="", type="skill", source="test", url="", stars=100),
        ]
        
        sorted_results = agg.sort_results(results, by="stars", reverse=True)
        # None values should sort to end
        assert sorted_results[0].stars == 100
        assert sorted_results[1].stars == 50


class TestUnifiedSchema:
    """Test the unified output schema"""
    
    def test_export_schema_format(self):
        unified_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "type": {"type": "string", "enum": [t.value for t in ResourceType]},
                "source": {"type": "string", "enum": [s.value for s in Source]},
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
class TestAsyncScraping:
    """Test async scraping functionality"""
    
    async def test_scrape_source_mock(self):
        agg = Aggregator(api_key="test-key")
        
        # Mock the firecrawl response
        mock_response = {
            "data": [
                {"name": "test-skill", "description": "A test", "author": "@test/repo"}
            ]
        }
        
        with patch.object(agg.firecrawl, "extract", return_value=mock_response):
            results = await agg.scrape_source(Source.SKILLSMP)
            assert len(results) == 1
            assert results[0].name == "test-skill"
    
    async def test_scrape_source_handles_errors(self):
        agg = Aggregator(api_key="test-key")
        
        with patch.object(agg.firecrawl, "extract", side_effect=Exception("Network error")):
            results = await agg.scrape_source(Source.SKILLSMP)
            assert results == []  # Should return empty list on error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
