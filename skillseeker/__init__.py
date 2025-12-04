"""Claude Skills/MCP Aggregator - Search across multiple marketplaces"""

from .skillseeker import (
    Aggregator,
    Resource,
    ResourceType,
    Source,
    cli,
    SKILLSMP_SCHEMA,
    CLAUDE_PLUGINS_SCHEMA,
    MCP_SCHEMA,
    MARKETPLACE_SCHEMA,
)

__version__ = "0.1.0"
__all__ = [
    "Aggregator",
    "Resource",
    "ResourceType",
    "Source",
    "cli",
    "SKILLSMP_SCHEMA",
    "CLAUDE_PLUGINS_SCHEMA",
    "MCP_SCHEMA",
    "MARKETPLACE_SCHEMA",
]
