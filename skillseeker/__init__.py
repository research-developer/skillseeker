"""Skillseeker - Search across multiple Claude skill and MCP marketplaces"""

from .skillseeker import (
    Aggregator,
    Resource,
    ResourceType,
    Source,
    cli,
)

__version__ = "0.1.0"
__all__ = [
    "Aggregator",
    "Resource",
    "ResourceType",
    "Source",
    "cli",
]
