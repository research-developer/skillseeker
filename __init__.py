"""Claude Skills/MCP Aggregator - Search across multiple marketplaces"""

from .aggregator import (
    Aggregator,
    Resource,
    ResourceType,
    Source,
    cli,
)

__version__ = "0.1.0"
__all__ = ["Aggregator", "Resource", "ResourceType", "Source", "cli"]
