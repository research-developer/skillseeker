#!/usr/bin/env python3
"""
Skillseeker - Claude Skills/MCP Aggregator CLI
Aggregates skills, plugins, agents, and MCP servers from multiple marketplaces.
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus, urlparse

import click
import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich import box

# Load environment variables from .env file
load_dotenv()

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
    SMITHERY = "smithery"
    GLAMA = "glama"
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


class FirecrawlTracker:
    """Tracks FireCrawl API usage and prompts for permission when limit exceeded."""

    def __init__(self, limit: int = 10):
        self.limit = limit
        self.usage_count = 0
        self.permission_granted = False

    def can_use(self) -> bool:
        """Check if we can use FireCrawl (within limit or permission granted)."""
        if self.limit == 0:  # Unlimited
            return True
        if self.usage_count < self.limit:
            return True
        return self.permission_granted

    def request_permission(self) -> bool:
        """Ask user for permission to exceed the limit."""
        if self.permission_granted:
            return True

        console.print(f"\n[yellow]FireCrawl usage limit ({self.limit}) reached.[/yellow]")
        console.print(f"[dim]Current usage: {self.usage_count} requests[/dim]")

        self.permission_granted = Confirm.ask(
            "Allow additional FireCrawl requests?",
            default=False
        )
        return self.permission_granted

    def increment(self):
        """Record a FireCrawl API call."""
        self.usage_count += 1


class SkillsmpClient:
    """Client for SkillsMP API (https://skillsmp.com/docs/api)"""

    BASE_URL = "https://skillsmp.com/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query: str,
        page: int = 1,
        limit: int = 20,
        sort_by: str = "stars",
        use_ai: bool = False
    ) -> list[Resource]:
        """Search skills using keyword or AI semantic search."""
        if not self.api_key:
            return []

        endpoint = "/skills/ai-search" if use_ai else "/skills/search"
        url = f"{self.BASE_URL}{endpoint}"

        params = {"q": query}
        if not use_ai:
            params.update({
                "page": page,
                "limit": min(limit, 100),
                "sortBy": sort_by
            })

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = await self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            resources = []
            items = data.get("data", data.get("skills", []))
            if isinstance(items, dict):
                # AI search returns nested structure: data.data[].skill
                items = items.get("data", items.get("items", items.get("skills", [])))
            if not isinstance(items, list):
                items = [items] if items else []

            for item in items:
                if not isinstance(item, dict):
                    continue
                # AI search nests skill info under "skill" key
                if "skill" in item:
                    item = item["skill"]
                resource = Resource(
                    name=item.get("name", "Unknown"),
                    description=item.get("description", ""),
                    type=ResourceType.SKILL.value,
                    source=Source.SKILLSMP.value,
                    url=item.get("url", f"https://skillsmp.com/skills/{item.get('slug', '')}"),
                    author=item.get("author"),
                    github_url=item.get("github_url") or item.get("githubUrl"),
                    stars=item.get("stars"),
                    downloads=item.get("downloads"),
                    install_command=item.get("install_command") or item.get("installCommand"),
                    category=item.get("category"),
                    tags=item.get("tags", [])
                )
                resources.append(resource)

            return resources

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                console.print("[yellow]Warning: SkillsMP API key invalid or missing[/yellow]")
            else:
                console.print(f"[yellow]Warning: SkillsMP API error: {e}[/yellow]")
            return []
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to fetch from SkillsMP: {e}[/yellow]")
            return []

    async def close(self):
        await self.client.aclose()


class SmitheryClient:
    """Client for Smithery Registry API (https://registry.smithery.ai)"""

    BASE_URL = "https://registry.smithery.ai"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search(
        self,
        query: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> list[Resource]:
        """Search MCP servers in Smithery registry."""
        if not self.api_key:
            return []

        url = f"{self.BASE_URL}/servers"
        params = {"page": page, "pageSize": min(page_size, 100)}
        if query:
            params["q"] = query

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

        try:
            response = await self.client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            resources = []
            servers = data.get("servers", [])

            for server in servers:
                if not isinstance(server, dict):
                    continue

                qualified_name = server.get("qualifiedName", "")
                resource = Resource(
                    name=server.get("displayName") or qualified_name,
                    description=server.get("description", ""),
                    type=ResourceType.MCP_SERVER.value,
                    source=Source.SMITHERY.value,
                    url=server.get("homepage") or f"https://smithery.ai/server/{qualified_name}",
                    author=qualified_name.split("/")[0] if "/" in qualified_name else None,
                    github_url=server.get("repository"),
                    stars=server.get("useCount"),
                    install_command=f"npx @smithery/cli install {qualified_name}",
                    category=server.get("category"),
                    tags=server.get("tags", [])
                )
                resources.append(resource)

            return resources

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                console.print("[yellow]Warning: Smithery API key invalid or missing[/yellow]")
            else:
                console.print(f"[yellow]Warning: Smithery API error: {e}[/yellow]")
            return []
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to fetch from Smithery: {e}[/yellow]")
            return []

    async def close(self):
        await self.client.aclose()


class GlamaClient:
    """Client for Glama MCP Registry API (https://glama.ai/api/mcp/v1)"""

    BASE_URL = "https://glama.ai/api/mcp/v1"
    SEARCH_URL = "https://glama.ai/mcp/servers"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def get_server(self, owner: str, repo: str) -> Optional[Resource]:
        """Get details for a specific MCP server (no auth required)."""
        url = f"{self.BASE_URL}/servers/{owner}/{repo}"

        try:
            response = await self.client.get(url)
            response.raise_for_status()
            server = response.json()

            return Resource(
                name=server.get("name", f"{owner}/{repo}"),
                description=server.get("description", ""),
                type=ResourceType.MCP_SERVER.value,
                source=Source.GLAMA.value,
                url=f"https://glama.ai/mcp/servers/@{owner}/{repo}",
                author=owner,
                github_url=server.get("repository"),
                stars=server.get("stars"),
                install_command=server.get("installCommand"),
                category=server.get("category"),
                language=server.get("language"),
                tags=server.get("tags", [])
            )

        except Exception:
            return None

    async def search_via_scrape(
        self,
        query: str,
        firecrawl_app,
        tracker: FirecrawlTracker
    ) -> list[Resource]:
        """
        Search Glama using FireCrawl (no public search API).
        Falls back to scraping since Glama doesn't expose a search endpoint.
        """
        if not tracker.can_use():
            if not tracker.request_permission():
                console.print("[dim]Skipping Glama (FireCrawl limit reached)[/dim]")
                return []

        url = f"{self.SEARCH_URL}?query={quote_plus(query)}"

        try:
            tracker.increment()
            result = firecrawl_app.extract(
                urls=[url],
                prompt="Extract all MCP servers listed. For each server get: name, description, author/owner, GitHub URL, stars, category, and install command.",
                schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "author": {"type": "string"},
                            "github_url": {"type": "string"},
                            "stars": {"type": "integer"},
                            "category": {"type": "string"},
                            "install_command": {"type": "string"},
                        },
                        "required": ["name", "description"]
                    }
                }
            )

            resources = []
            items = result.get("data", []) if isinstance(result, dict) else []
            if isinstance(items, dict):
                items = items.get("items", [])
            if not isinstance(items, list):
                items = [items] if items else []

            for item in items:
                if not isinstance(item, dict):
                    continue
                resource = Resource(
                    name=item.get("name", "Unknown"),
                    description=item.get("description", ""),
                    type=ResourceType.MCP_SERVER.value,
                    source=Source.GLAMA.value,
                    url=f"https://glama.ai/mcp/servers",
                    author=item.get("author"),
                    github_url=item.get("github_url"),
                    stars=item.get("stars"),
                    install_command=item.get("install_command"),
                    category=item.get("category"),
                    tags=[]
                )
                resources.append(resource)

            return resources

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to search Glama: {e}[/yellow]")
            return []

    async def close(self):
        await self.client.aclose()


class McpSoClient:
    """Client for mcp.so - uses FireCrawl since no public API."""

    BASE_URL = "https://mcp.so"

    def __init__(self):
        pass

    async def search_via_scrape(
        self,
        query: str,
        firecrawl_app,
        tracker: FirecrawlTracker
    ) -> list[Resource]:
        """Search mcp.so using FireCrawl scraping."""
        if not tracker.can_use():
            if not tracker.request_permission():
                console.print("[dim]Skipping mcp.so (FireCrawl limit reached)[/dim]")
                return []

        url = f"{self.BASE_URL}/servers?q={quote_plus(query)}" if query else self.BASE_URL

        try:
            tracker.increment()
            result = firecrawl_app.extract(
                urls=[url],
                prompt="Extract all MCP servers listed on this page. For each server, get the name, description, author, GitHub URL, stars, category, implementation language, and install command.",
                schema={
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "author": {"type": "string"},
                            "github_url": {"type": "string"},
                            "stars": {"type": "integer"},
                            "category": {"type": "string"},
                            "language": {"type": "string"},
                            "install_command": {"type": "string"},
                        },
                        "required": ["name", "description"]
                    }
                }
            )

            resources = []
            items = result.get("data", []) if isinstance(result, dict) else []
            if isinstance(items, dict):
                items = items.get("items", [])
            if not isinstance(items, list):
                items = [items] if items else []

            for item in items:
                if not isinstance(item, dict):
                    continue
                resource = Resource(
                    name=item.get("name", "Unknown"),
                    description=item.get("description", ""),
                    type=ResourceType.MCP_SERVER.value,
                    source=Source.MCP_SO.value,
                    url=self.BASE_URL,
                    author=item.get("author"),
                    github_url=item.get("github_url"),
                    stars=item.get("stars"),
                    install_command=item.get("install_command"),
                    category=item.get("category"),
                    language=item.get("language"),
                    tags=[]
                )
                resources.append(resource)

            return resources

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to search mcp.so: {e}[/yellow]")
            return []


class Aggregator:
    """Main aggregator class that searches and combines resources from multiple sources."""

    def __init__(
        self,
        firecrawl_api_key: Optional[str] = None,
        skillsmp_api_key: Optional[str] = None,
        smithery_api_key: Optional[str] = None,
        firecrawl_limit: int = 10
    ):
        # API keys
        self.firecrawl_api_key = firecrawl_api_key or os.environ.get("FIRECRAWL_API_KEY")
        self.skillsmp_api_key = skillsmp_api_key or os.environ.get("SKILLSMP_API_KEY")
        self.smithery_api_key = smithery_api_key or os.environ.get("SMITHERY_API_KEY")

        # FireCrawl setup (for sources without APIs)
        self.firecrawl_app = None
        if self.firecrawl_api_key:
            try:
                from firecrawl import FirecrawlApp
                self.firecrawl_app = FirecrawlApp(api_key=self.firecrawl_api_key)
            except ImportError:
                console.print("[yellow]Warning: firecrawl-py not installed, some sources unavailable[/yellow]")

        # FireCrawl usage tracking
        limit = firecrawl_limit
        try:
            limit = int(os.environ.get("FIRECRAWL_LIMIT", firecrawl_limit))
        except ValueError:
            pass
        self.firecrawl_tracker = FirecrawlTracker(limit=limit)

        # API clients
        self.skillsmp = SkillsmpClient(self.skillsmp_api_key)
        self.smithery = SmitheryClient(self.smithery_api_key)
        self.glama = GlamaClient()
        self.mcp_so = McpSoClient()

        self.results: list[Resource] = []

    async def search_source(self, source: Source, query: Optional[str] = None) -> list[Resource]:
        """Search a single source."""
        if source == Source.SKILLSMP:
            if self.skillsmp_api_key:
                return await self.skillsmp.search(query or "", use_ai=True)
            elif self.firecrawl_app:
                console.print("[dim]SkillsMP: No API key, consider getting one at skillsmp.com/docs/api[/dim]")
                return []
            return []

        elif source == Source.SMITHERY:
            if self.smithery_api_key:
                return await self.smithery.search(query)
            else:
                console.print("[dim]Smithery: No API key, get one at smithery.ai/account/api-keys[/dim]")
                return []

        elif source == Source.GLAMA:
            if query and self.firecrawl_app:
                return await self.glama.search_via_scrape(query, self.firecrawl_app, self.firecrawl_tracker)
            return []

        elif source == Source.MCP_SO:
            if self.firecrawl_app:
                return await self.mcp_so.search_via_scrape(query or "", self.firecrawl_app, self.firecrawl_tracker)
            return []

        return []

    async def search_all(self, sources: list[Source], query: Optional[str] = None) -> list[Resource]:
        """Search multiple sources in parallel."""
        if Source.ALL in sources:
            sources = [s for s in Source if s != Source.ALL]

        tasks = [self.search_source(source, query) for source in sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_resources = []
        for i, result in enumerate(results):
            if isinstance(result, list):
                all_resources.extend(result)
            elif isinstance(result, Exception):
                console.print(f"[yellow]Error searching {sources[i].value}: {result}[/yellow]")

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
        """Filter aggregated results."""
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
        """Sort results by field."""
        def get_sort_key(r: Resource):
            val = getattr(r, by, None)
            if val is None:
                return -1 if reverse else float('inf')
            return val

        return sorted(results, key=get_sort_key, reverse=reverse)

    async def close(self):
        """Clean up HTTP clients."""
        await self.skillsmp.close()
        await self.smithery.close()
        await self.glama.close()

    def get_firecrawl_usage(self) -> dict:
        """Get FireCrawl usage statistics."""
        return {
            "usage": self.firecrawl_tracker.usage_count,
            "limit": self.firecrawl_tracker.limit,
            "exceeded": self.firecrawl_tracker.usage_count >= self.firecrawl_tracker.limit if self.firecrawl_tracker.limit > 0 else False
        }


def display_results(results: list[Resource], output_format: str = "table"):
    """Display results in various formats."""
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
        title="Skillseeker Results",
        box=box.ROUNDED,
        show_lines=True
    )

    table.add_column("Name", style="cyan", no_wrap=True, max_width=25)
    table.add_column("Type", style="magenta", max_width=10)
    table.add_column("Source", style="green", max_width=12)
    table.add_column("Description", max_width=40)
    table.add_column("Stars", justify="right", max_width=6)
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
@click.pass_context
def cli(ctx):
    """Skillseeker - Search across multiple Claude skill and MCP marketplaces"""
    ctx.ensure_object(dict)


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
    sources = [Source(s) for s in source]
    rtype = ResourceType(resource_type) if resource_type else None

    # Check for at least one usable source
    has_skillsmp = bool(os.environ.get("SKILLSMP_API_KEY"))
    has_smithery = bool(os.environ.get("SMITHERY_API_KEY"))
    has_firecrawl = bool(os.environ.get("FIRECRAWL_API_KEY"))

    if not (has_skillsmp or has_smithery or has_firecrawl):
        console.print("[red]Error: No API keys configured.[/red]")
        console.print("Set at least one of: SKILLSMP_API_KEY, SMITHERY_API_KEY, or FIRECRAWL_API_KEY")
        console.print("See .env.example for configuration options.")
        sys.exit(1)

    async def run_search():
        aggregator = Aggregator()
        try:
            results = await aggregator.search_all(sources, query)

            results = aggregator.filter_results(
                resource_type=rtype,
                min_stars=min_stars,
                min_downloads=min_downloads,
                category=category,
                author=author
            )

            results = aggregator.sort_results(results, by=sort)

            # Show FireCrawl usage if any was used
            usage = aggregator.get_firecrawl_usage()
            if usage["usage"] > 0:
                console.print(f"\n[dim]FireCrawl usage: {usage['usage']}/{usage['limit'] or '∞'} requests[/dim]")

            return results
        finally:
            await aggregator.close()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Searching sources...", total=None)

        try:
            results = asyncio.run(run_search())
            progress.update(task, description="Processing results...")
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
    """List available sources and their API status"""
    table = Table(title="Available Sources", box=box.ROUNDED)
    table.add_column("Source", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("API", style="blue")
    table.add_column("Status", style="yellow")

    has_skillsmp = bool(os.environ.get("SKILLSMP_API_KEY"))
    has_smithery = bool(os.environ.get("SMITHERY_API_KEY"))
    has_firecrawl = bool(os.environ.get("FIRECRAWL_API_KEY"))

    table.add_row(
        "skillsmp",
        "skill",
        "REST API",
        "[green]✓ Ready[/green]" if has_skillsmp else "[yellow]⚠ No API key[/yellow]"
    )
    table.add_row(
        "smithery",
        "mcp_server",
        "REST API",
        "[green]✓ Ready[/green]" if has_smithery else "[yellow]⚠ No API key[/yellow]"
    )
    table.add_row(
        "glama",
        "mcp_server",
        "FireCrawl",
        "[green]✓ Ready[/green]" if has_firecrawl else "[red]✗ Needs FireCrawl[/red]"
    )
    table.add_row(
        "mcp_so",
        "mcp_server",
        "FireCrawl",
        "[green]✓ Ready[/green]" if has_firecrawl else "[red]✗ Needs FireCrawl[/red]"
    )

    console.print(table)

    console.print("\n[dim]Configure API keys in .env file. See .env.example for options.[/dim]")


@cli.command()
def types():
    """List available resource types"""
    console.print(Panel.fit(
        "\n".join([f"• [cyan]{t.value}[/cyan]" for t in ResourceType if t != ResourceType.ALL]),
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

    with open(output, "w") as f:
        json.dump(unified_schema, f, indent=2)

    console.print(f"[green]Schema exported to {output}[/green]")
    console.print_json(json.dumps(unified_schema, indent=2))


@cli.command()
def usage():
    """Show current FireCrawl usage limits"""
    limit = os.environ.get("FIRECRAWL_LIMIT", "10")
    has_firecrawl = bool(os.environ.get("FIRECRAWL_API_KEY"))

    console.print(Panel.fit(
        f"[bold]FireCrawl Configuration[/bold]\n\n"
        f"API Key: {'[green]Configured[/green]' if has_firecrawl else '[red]Not set[/red]'}\n"
        f"Request Limit: {limit if limit != '0' else 'Unlimited'}\n\n"
        f"[dim]Set FIRECRAWL_LIMIT in .env to adjust (0 = unlimited)[/dim]",
        title="Usage Limits"
    ))


def get_skill_install_path(skill_name: str, global_install: bool) -> Path:
    """Get the installation path for a skill."""
    if global_install:
        base = Path.home() / ".claude" / "skills"
    else:
        base = Path.cwd() / ".claude" / "skills"
    return base / skill_name


def parse_github_url(url: str) -> tuple[str, str, str, str]:
    """
    Parse a GitHub URL and return (owner, repo, branch, path).
    Supports various GitHub URL formats:
    - https://github.com/owner/repo
    - https://github.com/owner/repo/tree/branch/path
    - https://github.com/owner/repo/blob/branch/path/file.md
    - https://raw.githubusercontent.com/owner/repo/branch/path
    """
    parsed = urlparse(url)

    if "raw.githubusercontent.com" in parsed.netloc:
        # Raw URL: /owner/repo/branch/path
        parts = parsed.path.strip("/").split("/")
        owner, repo, branch = parts[0], parts[1], parts[2]
        path = "/".join(parts[3:]) if len(parts) > 3 else ""
        return owner, repo, branch, path

    # Regular GitHub URL
    parts = parsed.path.strip("/").split("/")
    owner, repo = parts[0], parts[1]

    if len(parts) > 3 and parts[2] in ("tree", "blob"):
        branch = parts[3]
        path = "/".join(parts[4:]) if len(parts) > 4 else ""
    else:
        branch = "main"
        path = ""

    return owner, repo, branch, path


async def fetch_github_content(url: str, client: httpx.AsyncClient) -> tuple[str, str]:
    """
    Fetch content from a GitHub URL.
    Returns (content, filename).
    """
    owner, repo, branch, path = parse_github_url(url)

    # If path points to a directory or no path, look for SKILL.md
    if not path or not path.endswith(".md"):
        if path:
            skill_path = f"{path}/SKILL.md"
        else:
            # Check common locations
            for try_path in ["SKILL.md", "skills/SKILL.md", ".claude/skills/SKILL.md"]:
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{try_path}"
                try:
                    resp = await client.get(raw_url)
                    if resp.status_code == 200:
                        return resp.text, try_path.split("/")[-1]
                except Exception:
                    continue
            skill_path = "SKILL.md"
    else:
        skill_path = path

    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{skill_path}"
    resp = await client.get(raw_url)
    resp.raise_for_status()
    return resp.text, skill_path.split("/")[-1]


async def fetch_skill_from_skillsmp(skill_id: str, api_key: str, client: httpx.AsyncClient) -> dict:
    """Fetch skill details from SkillsMP API."""
    url = f"https://skillsmp.com/api/v1/skills/{skill_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = await client.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def extract_skill_name_from_content(content: str) -> Optional[str]:
    """Extract skill name from SKILL.md frontmatter."""
    # Look for YAML frontmatter
    match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if match:
        frontmatter = match.group(1)
        name_match = re.search(r'^name:\s*["\']?([^"\'\n]+)["\']?', frontmatter, re.MULTILINE)
        if name_match:
            return name_match.group(1).strip()
    return None


def extract_skill_name_from_url(url: str) -> str:
    """Extract a skill name from a GitHub URL."""
    owner, repo, branch, path = parse_github_url(url)
    if path:
        # Use the directory name
        parts = path.rstrip("/").split("/")
        for part in reversed(parts):
            if part and part != "SKILL.md":
                return part
    return repo


@cli.command()
@click.argument("source")
@click.option("-g", "--global", "global_install", is_flag=True, help="Install globally to ~/.claude/skills/")
@click.option("-n", "--name", help="Override the skill name (directory name)")
@click.option("--dry-run", is_flag=True, help="Show what would be installed without installing")
def install(source, global_install, name, dry_run):
    """
    Install a skill from a GitHub URL or SkillsMP.

    SOURCE can be:
    - A GitHub URL (https://github.com/owner/repo or path to SKILL.md)
    - A SkillsMP skill ID (e.g., "zenobi-us/postgres-pro")

    Examples:
        skillseeker install https://github.com/user/repo
        skillseeker install https://github.com/user/repo/tree/main/skills/my-skill
        skillseeker install zenobi-us/postgres-pro --global
    """
    async def do_install():
        async with httpx.AsyncClient(timeout=30.0) as client:
            content = None
            skill_name = name
            source_type = "unknown"

            # Determine source type
            if source.startswith("http://") or source.startswith("https://"):
                if "github.com" in source or "raw.githubusercontent.com" in source:
                    source_type = "github"
                    console.print(f"[dim]Fetching from GitHub: {source}[/dim]")
                    try:
                        content, filename = await fetch_github_content(source, client)
                        if not skill_name:
                            # Try to get name from content first
                            skill_name = extract_skill_name_from_content(content)
                            if not skill_name:
                                skill_name = extract_skill_name_from_url(source)
                    except httpx.HTTPStatusError as e:
                        console.print(f"[red]Error fetching from GitHub: {e}[/red]")
                        return
                else:
                    console.print(f"[red]Unsupported URL: {source}[/red]")
                    console.print("Supported: GitHub URLs (github.com, raw.githubusercontent.com)")
                    return
            else:
                # Assume SkillsMP skill ID
                source_type = "skillsmp"
                api_key = os.environ.get("SKILLSMP_API_KEY")
                if not api_key:
                    console.print("[red]SkillsMP API key required. Set SKILLSMP_API_KEY in .env[/red]")
                    return

                console.print(f"[dim]Fetching from SkillsMP: {source}[/dim]")
                try:
                    # Search for the skill
                    search_client = SkillsmpClient(api_key)
                    results = await search_client.search(source, use_ai=False)
                    await search_client.close()

                    if not results:
                        console.print(f"[red]Skill not found: {source}[/red]")
                        return

                    # Find exact match or first result
                    skill_data = None
                    for r in results:
                        if r.name == source or source in r.name:
                            skill_data = r
                            break
                    if not skill_data:
                        skill_data = results[0]

                    if not skill_name:
                        skill_name = skill_data.name

                    # If we have a GitHub URL, fetch from there
                    if skill_data.github_url:
                        console.print(f"[dim]Found GitHub URL: {skill_data.github_url}[/dim]")
                        try:
                            content, _ = await fetch_github_content(skill_data.github_url, client)
                        except Exception:
                            pass

                    # If no content yet, create a basic SKILL.md from the data
                    if not content:
                        content = f"""---
name: {skill_data.name}
description: {skill_data.description}
---

# {skill_data.name}

{skill_data.description}

## Source

- **Author**: {skill_data.author or 'Unknown'}
- **SkillsMP**: {skill_data.url}
"""
                        if skill_data.github_url:
                            content += f"- **GitHub**: {skill_data.github_url}\n"

                except Exception as e:
                    console.print(f"[red]Error fetching from SkillsMP: {e}[/red]")
                    return

            if not content:
                console.print("[red]Could not fetch skill content[/red]")
                return

            if not skill_name:
                console.print("[red]Could not determine skill name. Use --name to specify.[/red]")
                return

            # Sanitize skill name
            skill_name = re.sub(r'[^a-z0-9-]', '-', skill_name.lower())
            skill_name = re.sub(r'-+', '-', skill_name).strip('-')

            install_path = get_skill_install_path(skill_name, global_install)
            location = "global (~/.claude/skills)" if global_install else "local (.claude/skills)"

            if dry_run:
                console.print(Panel.fit(
                    f"[bold]Skill:[/bold] {skill_name}\n"
                    f"[bold]Source:[/bold] {source_type}\n"
                    f"[bold]Location:[/bold] {location}\n"
                    f"[bold]Path:[/bold] {install_path}\n\n"
                    f"[dim]Content preview (first 500 chars):[/dim]\n"
                    f"{content[:500]}...",
                    title="Dry Run - Would Install"
                ))
                return

            # Create directory and write SKILL.md
            install_path.mkdir(parents=True, exist_ok=True)
            skill_file = install_path / "SKILL.md"

            if skill_file.exists():
                if not Confirm.ask(f"[yellow]Skill already exists at {skill_file}. Overwrite?[/yellow]"):
                    console.print("[dim]Installation cancelled[/dim]")
                    return

            skill_file.write_text(content)

            console.print(Panel.fit(
                f"[green]✓ Installed successfully![/green]\n\n"
                f"[bold]Skill:[/bold] {skill_name}\n"
                f"[bold]Location:[/bold] {install_path}\n\n"
                f"[dim]Claude Code will automatically discover this skill.[/dim]",
                title="Skill Installed"
            ))

    asyncio.run(do_install())


@cli.command()
@click.option("-g", "--global", "global_only", is_flag=True, help="List only global skills")
@click.option("-l", "--local", "local_only", is_flag=True, help="List only local skills")
def installed(global_only, local_only):
    """List installed Claude Code skills."""
    global_path = Path.home() / ".claude" / "skills"
    local_path = Path.cwd() / ".claude" / "skills"

    table = Table(title="Installed Skills", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Location", style="green")
    table.add_column("Description", max_width=50)

    found = False

    def scan_skills(base_path: Path, location: str):
        nonlocal found
        if not base_path.exists():
            return
        for skill_dir in base_path.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    found = True
                    content = skill_file.read_text()
                    # Extract description from frontmatter
                    desc = ""
                    match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
                    if match:
                        frontmatter = match.group(1)
                        desc_match = re.search(r'^description:\s*["\']?([^"\'\n]+)', frontmatter, re.MULTILINE)
                        if desc_match:
                            desc = desc_match.group(1).strip()[:50]
                    table.add_row(skill_dir.name, location, desc or "[dim]No description[/dim]")

    if not local_only:
        scan_skills(global_path, "global")
    if not global_only:
        scan_skills(local_path, "local")

    if found:
        console.print(table)
    else:
        console.print("[yellow]No skills installed.[/yellow]")
        console.print(f"\n[dim]Global path: {global_path}[/dim]")
        console.print(f"[dim]Local path: {local_path}[/dim]")


@cli.command()
@click.argument("skill_name")
@click.option("-g", "--global", "global_install", is_flag=True, help="Uninstall from global location")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation")
def uninstall(skill_name, global_install, yes):
    """Uninstall a skill by name."""
    import shutil

    install_path = get_skill_install_path(skill_name, global_install)
    location = "global" if global_install else "local"

    if not install_path.exists():
        console.print(f"[yellow]Skill '{skill_name}' not found in {location} location.[/yellow]")
        console.print(f"[dim]Path: {install_path}[/dim]")
        return

    if not yes:
        if not Confirm.ask(f"[yellow]Remove skill '{skill_name}' from {location}?[/yellow]"):
            console.print("[dim]Uninstall cancelled[/dim]")
            return

    shutil.rmtree(install_path)
    console.print(f"[green]✓ Uninstalled '{skill_name}' from {location}[/green]")


if __name__ == "__main__":
    cli()
