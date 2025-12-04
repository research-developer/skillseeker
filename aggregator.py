#!/usr/bin/env python3
"""
Claude Skills/MCP Aggregator CLI
Aggregates skills, plugins, agents, and MCP servers from multiple marketplaces.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from firecrawl import FirecrawlApp

console = Console()


class ResourceType(str, Enum):
    SKILL = "skill"
    PLUGIN = "plugin"
    MCP_SERVER = "mcp_server"
    AGENT = "agent"
    COMMAND = "command"
    ALL = "all"


class Source(str, Enum):
    SKILLSMP = "skillsmp"
    CLAUDE_PLUGINS = "claude_plugins"
    CLAUDEMARKETPLACES = "claudemarketplaces"
    MCP_SO = "mcp_so"
    ALL = "all"


@dataclass
class Resource:
    """Unified schema for all resource types"""
    name: str
    description: str
    type: str
    source: str
    url: str
    author: Optional[str] = None
    github_url: Optional[str] = None
    stars: Optional[int] = None
    downloads: Optional[int] = None
    last_updated: Optional[str] = None
    install_command: Optional[str] = None
    category: Optional[str] = None
    language: Optional[str] = None
    tags: list = field(default_factory=list)


# Extraction schemas for each source
SKILLSMP_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the skill"},
            "description": {"type": "string", "description": "Description of what the skill does"},
            "author": {"type": "string", "description": "Author/creator username or org"},
            "github_url": {"type": "string", "description": "GitHub repository URL"},
            "stars": {"type": "integer", "description": "GitHub stars count"},
            "category": {"type": "string", "description": "Category like development, productivity, etc"},
            "install_command": {"type": "string", "description": "Command to install the skill"},
        },
        "required": ["name", "description"]
    }
}

CLAUDE_PLUGINS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the skill/plugin"},
            "description": {"type": "string", "description": "Full description text"},
            "author": {"type": "string", "description": "Author in format @username/repo"},
            "github_url": {"type": "string", "description": "GitHub repository URL"},
            "stars": {"type": "integer", "description": "Stars/popularity number"},
            "downloads": {"type": "integer", "description": "Download count"},
            "url": {"type": "string", "description": "Direct URL to the skill page"},
        },
        "required": ["name", "description"]
    }
}

MCP_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name of the MCP server"},
            "description": {"type": "string", "description": "Description of capabilities"},
            "author": {"type": "string", "description": "Author or organization"},
            "github_url": {"type": "string", "description": "GitHub/source repository URL"},
            "stars": {"type": "integer", "description": "GitHub stars"},
            "category": {"type": "string", "description": "Category like database, cloud, communication"},
            "language": {"type": "string", "description": "Implementation language (TypeScript, Python, etc)"},
            "install_command": {"type": "string", "description": "npx or uvx install command"},
        },
        "required": ["name", "description"]
    }
}

MARKETPLACE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Plugin/marketplace name"},
            "description": {"type": "string", "description": "Description"},
            "author": {"type": "string", "description": "Author/maintainer"},
            "github_url": {"type": "string", "description": "Repository URL"},
            "plugin_count": {"type": "integer", "description": "Number of plugins in marketplace"},
        },
        "required": ["name", "description"]
    }
}


class Aggregator:
    """Main aggregator class that scrapes and combines resources"""
    
    SOURCES = {
        Source.SKILLSMP: {
            "url": "https://skillsmp.com/",
            "search_url": "https://skillsmp.com/search?q={query}",
            "schema": SKILLSMP_SCHEMA,
            "prompt": "Extract all Claude skills/plugins shown on this page. For each skill, get the name, description, author, GitHub URL, star count, and category.",
            "type": ResourceType.SKILL
        },
        Source.CLAUDE_PLUGINS: {
            "url": "https://claude-plugins.dev/skills",
            "search_url": "https://claude-plugins.dev/skills?q={query}",
            "schema": CLAUDE_PLUGINS_SCHEMA,
            "prompt": "Extract all skills/plugins listed on this page. Get the name, description, author (in @username/repo format), GitHub URL, stars count, downloads, and the direct URL to each skill.",
            "type": ResourceType.SKILL
        },
        Source.CLAUDEMARKETPLACES: {
            "url": "https://claudemarketplaces.com/",
            "search_url": "https://claudemarketplaces.com/?q={query}",
            "schema": MARKETPLACE_SCHEMA,
            "prompt": "Extract all plugin marketplaces and plugins shown. Get name, description, author, GitHub URL, and plugin count if available.",
            "type": ResourceType.PLUGIN
        },
        Source.MCP_SO: {
            "url": "https://mcp.so/",
            "search_url": "https://mcp.so/search?q={query}",
            "schema": MCP_SCHEMA,
            "prompt": "Extract all MCP servers listed on this page. For each server, get the name, description, author, GitHub URL, stars, category, implementation language, and install command.",
            "type": ResourceType.MCP_SERVER
        }
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable required or pass api_key")
        self.firecrawl = FirecrawlApp(api_key=self.api_key)
        self.results: list[Resource] = []
    
    async def scrape_source(self, source: Source, query: Optional[str] = None) -> list[Resource]:
        """Scrape a single source"""
        config = self.SOURCES[source]
        url = config["search_url"].format(query=query) if query else config["url"]
        
        try:
            result = self.firecrawl.extract(
                urls=[url],
                prompt=config["prompt"],
                schema=config["schema"]
            )
            
            resources = []
            items = result.get("data", []) if isinstance(result, dict) else []
            
            # Handle nested data structure
            if isinstance(items, dict) and "items" in items:
                items = items["items"]
            elif not isinstance(items, list):
                items = [items] if items else []
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                resource = Resource(
                    name=item.get("name", "Unknown"),
                    description=item.get("description", ""),
                    type=config["type"].value,
                    source=source.value,
                    url=item.get("url", url),
                    author=item.get("author"),
                    github_url=item.get("github_url"),
                    stars=item.get("stars"),
                    downloads=item.get("downloads"),
                    install_command=item.get("install_command"),
                    category=item.get("category"),
                    language=item.get("language"),
                    tags=item.get("tags", [])
                )
                resources.append(resource)
            
            return resources
            
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to scrape {source.value}: {e}[/yellow]")
            return []
    
    async def scrape_all(self, sources: list[Source], query: Optional[str] = None) -> list[Resource]:
        """Scrape multiple sources in parallel"""
        if Source.ALL in sources:
            sources = [s for s in Source if s != Source.ALL]
        
        tasks = [self.scrape_source(source, query) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_resources = []
        for result in results:
            if isinstance(result, list):
                all_resources.extend(result)
            elif isinstance(result, Exception):
                console.print(f"[yellow]Error: {result}[/yellow]")
        
        self.results = all_resources
        return all_resources
    
    def filter_results(
        self,
        resource_type: Optional[ResourceType] = None,
        min_stars: Optional[int] = None,
        min_downloads: Optional[int] = None,
        category: Optional[str] = None,
        author: Optional[str] = None
    ) -> list[Resource]:
        """Filter aggregated results"""
        filtered = self.results
        
        if resource_type and resource_type != ResourceType.ALL:
            filtered = [r for r in filtered if r.type == resource_type.value]
        
        if min_stars:
            filtered = [r for r in filtered if r.stars and r.stars >= min_stars]
        
        if min_downloads:
            filtered = [r for r in filtered if r.downloads and r.downloads >= min_downloads]
        
        if category:
            filtered = [r for r in filtered if r.category and category.lower() in r.category.lower()]
        
        if author:
            filtered = [r for r in filtered if r.author and author.lower() in r.author.lower()]
        
        return filtered
    
    def sort_results(self, results: list[Resource], by: str = "stars", reverse: bool = True) -> list[Resource]:
        """Sort results by field"""
        def get_sort_key(r: Resource):
            val = getattr(r, by, None)
            if val is None:
                return -1 if reverse else float('inf')
            return val
        
        return sorted(results, key=get_sort_key, reverse=reverse)


def display_results(results: list[Resource], output_format: str = "table"):
    """Display results in various formats"""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return
    
    if output_format == "json":
        output = [asdict(r) for r in results]
        console.print_json(json.dumps(output, indent=2, default=str))
        return
    
    if output_format == "simple":
        for r in results:
            console.print(f"[bold cyan]{r.name}[/bold cyan] ({r.type})")
            console.print(f"  {r.description[:100]}..." if len(r.description) > 100 else f"  {r.description}")
            if r.github_url:
                console.print(f"  [link={r.github_url}]{r.github_url}[/link]")
            console.print()
        return
    
    # Table format (default)
    table = Table(
        title="Claude Skills/MCP Aggregator Results",
        box=box.ROUNDED,
        show_lines=True
    )
    
    table.add_column("Name", style="cyan", no_wrap=True, max_width=25)
    table.add_column("Type", style="magenta", max_width=10)
    table.add_column("Source", style="green", max_width=15)
    table.add_column("Description", max_width=40)
    table.add_column("⭐", justify="right", max_width=6)
    table.add_column("Author", style="yellow", max_width=20)
    
    for r in results[:50]:  # Limit display to 50
        desc = r.description[:80] + "..." if len(r.description) > 80 else r.description
        stars = str(r.stars) if r.stars else "-"
        author = r.author[:18] + ".." if r.author and len(r.author) > 20 else (r.author or "-")
        
        table.add_row(
            r.name,
            r.type,
            r.source,
            desc,
            stars,
            author
        )
    
    console.print(table)
    
    if len(results) > 50:
        console.print(f"\n[dim]Showing 50 of {len(results)} results. Use --format json for full output.[/dim]")


# CLI Commands
@click.group()
@click.option("--api-key", envvar="FIRECRAWL_API_KEY", help="FireCrawl API key")
@click.pass_context
def cli(ctx, api_key):
    """Claude Skills/MCP Aggregator - Search across multiple marketplaces"""
    ctx.ensure_object(dict)
    ctx.obj["api_key"] = api_key


@cli.command()
@click.option("-q", "--query", help="Search query")
@click.option("-s", "--source", type=click.Choice([s.value for s in Source]), multiple=True, default=["all"], help="Sources to search")
@click.option("-t", "--type", "resource_type", type=click.Choice([t.value for t in ResourceType]), default="all", help="Resource type filter")
@click.option("--min-stars", type=int, help="Minimum GitHub stars")
@click.option("--min-downloads", type=int, help="Minimum downloads")
@click.option("-c", "--category", help="Category filter")
@click.option("-a", "--author", help="Author filter")
@click.option("--sort", type=click.Choice(["stars", "downloads", "name"]), default="stars", help="Sort by field")
@click.option("-f", "--format", "output_format", type=click.Choice(["table", "json", "simple"]), default="table", help="Output format")
@click.option("-o", "--output", type=click.Path(), help="Save results to file")
@click.pass_context
def search(ctx, query, source, resource_type, min_stars, min_downloads, category, author, sort, output_format, output):
    """Search and aggregate Claude skills, plugins, and MCP servers"""
    api_key = ctx.obj.get("api_key")
    
    if not api_key:
        console.print("[red]Error: FIRECRAWL_API_KEY required. Set via --api-key or environment variable.[/red]")
        sys.exit(1)
    
    sources = [Source(s) for s in source]
    rtype = ResourceType(resource_type) if resource_type else None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Scraping sources...", total=None)
        
        try:
            aggregator = Aggregator(api_key)
            results = asyncio.run(aggregator.scrape_all(sources, query))
            
            progress.update(task, description="Filtering results...")
            results = aggregator.filter_results(
                resource_type=rtype,
                min_stars=min_stars,
                min_downloads=min_downloads,
                category=category,
                author=author
            )
            
            progress.update(task, description="Sorting results...")
            results = aggregator.sort_results(results, by=sort)
            
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
    
    # Display or save
    if output:
        output_data = [asdict(r) for r in results]
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        console.print(f"[green]Saved {len(results)} results to {output}[/green]")
    else:
        display_results(results, output_format)
    
    console.print(f"\n[dim]Found {len(results)} resources[/dim]")


@cli.command()
def sources():
    """List available sources"""
    table = Table(title="Available Sources", box=box.ROUNDED)
    table.add_column("Source", style="cyan")
    table.add_column("URL", style="blue")
    table.add_column("Type", style="green")
    
    for source, config in Aggregator.SOURCES.items():
        table.add_row(source.value, config["url"], config["type"].value)
    
    console.print(table)


@cli.command()
def types():
    """List available resource types"""
    console.print(Panel.fit(
        "\n".join([f"• [cyan]{t.value}[/cyan]" for t in ResourceType]),
        title="Resource Types"
    ))


@cli.command()
@click.option("-o", "--output", type=click.Path(), default="schema.json", help="Output file")
def schema(output):
    """Export the unified resource schema"""
    unified_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Resource name"},
            "description": {"type": "string", "description": "Description"},
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
    
    with open(output, "w") as f:
        json.dump(unified_schema, f, indent=2)
    
    console.print(f"[green]Schema exported to {output}[/green]")
    console.print_json(json.dumps(unified_schema, indent=2))


if __name__ == "__main__":
    cli()
