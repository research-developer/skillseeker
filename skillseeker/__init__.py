"""Skillseeker - Search across multiple Claude skill and MCP marketplaces"""

from .skillseeker import (
    Aggregator,
    Verbosity,
    Resource,
    ResourceType,
    Source,
    get_verbosity,
    cli,
    set_verbosity,
)

__version__ = "0.1.0"
__all__ = [
    "Aggregator",
    "Verbosity",
    "Resource",
    "ResourceType",
    "Source",
    "get_verbosity",
    "cli",
    "set_verbosity",
]
