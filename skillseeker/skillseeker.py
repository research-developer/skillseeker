#!/usr/bin/env python3
"""
Skillseeker - Claude Skills/MCP Aggregator CLI
Aggregates skills, plugins, agents, and MCP servers from multiple marketplaces.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import click
import httpx
import questionary
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False

# Load environment variables from .env file
load_dotenv()

class Verbosity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


VERBOSITY_ORDER = {Verbosity.INFO: 0, Verbosity.WARNING: 1, Verbosity.ERROR: 2}
_current_verbosity = Verbosity.INFO
WARNING_PATTERNS = (
    re.compile(r"\[yellow\]", re.IGNORECASE),
    re.compile(r"\bwarning\b", re.IGNORECASE),
)
ERROR_PATTERNS = (
    re.compile(r"\[red\]", re.IGNORECASE),
    re.compile(r"\berror\b", re.IGNORECASE),
)


def set_verbosity(level: Verbosity | str):
    """Set global verbosity for console output."""
    global _current_verbosity
    try:
        _current_verbosity = Verbosity(level)
    except ValueError as exc:
        valid_levels = ", ".join(v.value for v in Verbosity)
        raise ValueError(
            f"Invalid verbosity level {level!r}. Expected one of: {valid_levels}"
        ) from exc


def get_verbosity() -> Verbosity:
    """Get the current verbosity level."""
    return _current_verbosity


def _should_log(level: Verbosity) -> bool:
    return VERBOSITY_ORDER[level] >= VERBOSITY_ORDER[_current_verbosity]


def infer_verbosity(args, kwargs) -> Verbosity:
    """
    Infer verbosity for a console call.

    Callers can pass `_verbosity` explicitly to avoid the heuristic.
    Otherwise, we fall back to detecting common Rich color tags or
    keywords so existing console.print calls keep working without
    refactoring each site.
    """
    explicit = kwargs.get("_verbosity")
    if explicit is not None:
        try:
            return Verbosity(explicit)
        except ValueError:
            return Verbosity.INFO

    if args:
        first = args[0]
        if isinstance(first, str):
            for level, patterns in (
                (Verbosity.ERROR, ERROR_PATTERNS),
                (Verbosity.WARNING, WARNING_PATTERNS),
            ):
                if any(pattern.search(first) for pattern in patterns):
                    return level

    return Verbosity.INFO


class VerboseConsole(Console):
    """Console that respects global verbosity settings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._raw_console_print = super().print

    def print(self, *args, **kwargs):
        level = infer_verbosity(args, kwargs)
        if "_verbosity" in kwargs:
            clean_kwargs = {k: v for k, v in kwargs.items() if k != "_verbosity"}
        else:
            clean_kwargs = kwargs
        if _should_log(level):
            self._raw_console_print(*args, **clean_kwargs)


console = VerboseConsole()


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
    ALL = "all"


@dataclass
class Resource:
    """Unified schema for all resource types"""
    name: str
    description: str
    type: str
    source: str
    url: str
    identifier: Optional[str] = None  # Unique ID for installation (slug, qualified_name, etc.)
    author: Optional[str] = None
    github_url: Optional[str] = None
    stars: Optional[int] = None
    downloads: Optional[int] = None
    last_updated: Optional[str] = None
    install_command: Optional[str] = None
    category: Optional[str] = None
    language: Optional[str] = None
    tags: list = field(default_factory=list)


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
                slug = item.get("slug") or item.get("id")
                resource = Resource(
                    name=item.get("name", "Unknown"),
                    description=item.get("description", ""),
                    type=ResourceType.SKILL.value,
                    source=Source.SKILLSMP.value,
                    url=item.get("url", f"https://skillsmp.com/skills/{slug or ''}"),
                    identifier=slug,
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
                    identifier=qualified_name,
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


class Aggregator:
    """Main aggregator class that searches and combines resources from multiple sources."""

    def __init__(
        self,
        skillsmp_api_key: Optional[str] = None,
        smithery_api_key: Optional[str] = None,
    ):
        # API keys
        self.skillsmp_api_key = skillsmp_api_key or os.environ.get("SKILLSMP_API_KEY")
        self.smithery_api_key = smithery_api_key or os.environ.get("SMITHERY_API_KEY")

        # API clients
        self.skillsmp = SkillsmpClient(self.skillsmp_api_key)
        self.smithery = SmitheryClient(self.smithery_api_key)

        self.results: list[Resource] = []

    async def search_source(self, source: Source, query: Optional[str] = None) -> list[Resource]:
        """Search a single source."""
        if source == Source.SKILLSMP:
            if self.skillsmp_api_key:
                return await self.skillsmp.search(query or "", use_ai=True)
            else:
                console.print("[dim]SkillsMP: No API key, get one at skillsmp.com/docs/api[/dim]")
                return []

        elif source == Source.SMITHERY:
            if self.smithery_api_key:
                return await self.smithery.search(query)
            else:
                console.print("[dim]Smithery: No API key, get one at smithery.ai/account/api-keys[/dim]")
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


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard, with fallback to pbcopy/xclip."""
    if HAS_PYPERCLIP:
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            pass

    # Fallback to system commands
    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
            return True
        elif sys.platform.startswith("linux"):
            subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
            return True
    except Exception:
        pass

    return False


def get_installable_identifier(resource: Resource) -> str:
    """Get the best identifier for installing a resource."""
    # Prefer the identifier field (slug for SkillsMP, qualified_name for Smithery)
    if resource.identifier:
        return resource.identifier

    # Fallback: For skills from SkillsMP, prefer github_url
    if resource.source == Source.SKILLSMP.value:
        if resource.github_url:
            return resource.github_url

    # Fallback: For Smithery MCP servers, extract from install command
    if resource.source == Source.SMITHERY.value:
        if resource.install_command:
            # "npx @smithery/cli install owner/repo" -> "owner/repo"
            parts = resource.install_command.split()
            if len(parts) >= 4:
                return parts[-1]

    return resource.url or resource.name


def interactive_select(results: list[Resource], global_install: bool = False) -> None:
    """Show interactive multi-select menu for search results."""
    if not results:
        console.print("[yellow]No results to select from.[/yellow]")
        return

    # Build choices for questionary
    choices = []
    for i, r in enumerate(results[:50]):  # Limit to 50 for usability
        stars = f"⭐{r.stars}" if r.stars else ""
        source_tag = f"[{r.source}]"
        desc = r.description[:40] + "..." if len(r.description) > 40 else r.description
        label = f"{r.name} {stars} {source_tag} - {desc}"
        choices.append(questionary.Choice(title=label, value=i))

    # Show multi-select
    console.print("\n[bold cyan]Select resources to install:[/bold cyan]")
    console.print("[dim]Use arrow keys to navigate, Space to select, Enter to confirm[/dim]\n")

    selected_indices = questionary.checkbox(
        "Select resources:",
        choices=choices,
        instruction="(Space to select, Enter to confirm)",
    ).ask()

    if not selected_indices:
        console.print("[dim]No items selected.[/dim]")
        return

    selected_resources = [results[i] for i in selected_indices]

    # Show what was selected
    console.print(f"\n[green]Selected {len(selected_resources)} item(s):[/green]")
    for r in selected_resources:
        console.print(f"  • {r.name} ({r.source})")

    # Ask what to do with selection
    action = questionary.select(
        "\nWhat would you like to do?",
        choices=[
            questionary.Choice("Install selected items now", value="install"),
            questionary.Choice("Copy install identifiers to clipboard", value="copy"),
            questionary.Choice("Show install commands", value="show"),
            questionary.Choice("Cancel", value="cancel"),
        ],
    ).ask()

    if action == "cancel" or action is None:
        console.print("[dim]Cancelled.[/dim]")
        return

    # Get installable identifiers
    identifiers = [get_installable_identifier(r) for r in selected_resources]

    if action == "copy":
        # Format for pasting after "skillseeker install"
        clipboard_text = " ".join(f'"{i}"' if " " in i else i for i in identifiers)
        if copy_to_clipboard(clipboard_text):
            console.print(f"\n[green]✓ Copied to clipboard![/green]")
            console.print(f"[dim]Paste after: skillseeker install[/dim]")
            console.print(f"[cyan]{clipboard_text}[/cyan]")
        else:
            console.print("\n[yellow]Could not copy to clipboard. Here are the identifiers:[/yellow]")
            console.print(f"[cyan]{clipboard_text}[/cyan]")

    elif action == "show":
        console.print("\n[bold]Install commands:[/bold]")
        location = "--global" if global_install else ""
        for ident in identifiers:
            console.print(f"  skillseeker install {location} {ident}".strip())

    elif action == "install":
        console.print("\n[bold]Installing selected items...[/bold]\n")
        install_multiple_resources(identifiers, global_install)


async def _install_multiple_resources_async(sources: list[str], global_install: bool = False) -> tuple[int, int]:
    """Async helper to install multiple resources within a single event loop."""
    success_count = 0
    fail_count = 0

    for source in sources:
        console.print(f"\n[cyan]Installing: {source}[/cyan]")
        try:
            # Call the install logic directly within the same event loop
            await do_single_install(source, global_install, name_override=None, dry_run=False)
            success_count += 1
        except Exception as e:
            console.print(f"[red]Failed to install {source}: {e}[/red]")
            fail_count += 1

    return success_count, fail_count


def install_multiple_resources(sources: list[str], global_install: bool = False) -> None:
    """Install multiple resources."""
    success_count, fail_count = asyncio.run(
        _install_multiple_resources_async(sources, global_install)
    )
    console.print(f"\n[bold]Installation complete:[/bold]")
    console.print(f"  [green]✓ Succeeded: {success_count}[/green]")
    if fail_count:
        console.print(f"  [red]✗ Failed: {fail_count}[/red]")


async def do_single_install(source: str, global_install: bool, name_override: Optional[str], dry_run: bool) -> None:
    """Core install logic for a single source."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        content = None
        skill_name = name_override

        # Determine source type
        if source.startswith("http://") or source.startswith("https://"):
            if "github.com" in source or "raw.githubusercontent.com" in source:
                source_type = "github"
                console.print(f"[dim]Fetching from GitHub: {source}[/dim]")
                try:
                    content, filename = await fetch_github_content(source, client)
                    if not skill_name:
                        skill_name = extract_skill_name_from_content(content)
                        if not skill_name:
                            skill_name = extract_skill_name_from_url(source)
                except httpx.HTTPStatusError as e:
                    raise Exception(f"Error fetching from GitHub: {e}")
            else:
                raise Exception(f"Unsupported URL: {source}")
        else:
            # Assume SkillsMP skill ID
            source_type = "skillsmp"
            api_key = os.environ.get("SKILLSMP_API_KEY")
            if not api_key:
                raise Exception("SkillsMP API key required. Set SKILLSMP_API_KEY in .env")

            console.print(f"[dim]Fetching from SkillsMP: {source}[/dim]")
            # Search for the skill
            search_client = SkillsmpClient(api_key)
            results = await search_client.search(source, use_ai=False)
            await search_client.close()

            if not results:
                raise Exception(f"Skill not found: {source}")

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
                except httpx.HTTPError as e:
                    console.print(
                        f"[yellow]Warning: failed to fetch skill content from GitHub URL "
                        f"({skill_data.github_url}): {e}[/yellow]"
                    )
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: unexpected error while fetching skill content "
                        f"from GitHub URL ({skill_data.github_url}): {e}[/yellow]"
                    )

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

        if not content:
            raise Exception("Could not fetch skill content")

        if not skill_name:
            raise Exception("Could not determine skill name. Use --name to specify.")

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

        # Allow non-interactive/batch installs to skip overwrite prompts
        assume_yes = os.environ.get("SKILLSEEKER_ASSUME_YES", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
            "all",
        }

        if skill_file.exists():
            if not assume_yes and not Confirm.ask(
                f"[yellow]Skill already exists at {skill_file}. Overwrite?[/yellow]"
            ):
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
            desc = (
                f"  {r.description[:100]}..."
                if len(r.description) > 100
                else f"  {r.description}"
            )
            console.print(desc)
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
        console.print(
            f"\n[dim]Showing 50 of {len(results)} results. "
            "Use --format json for full output.[/dim]"
        )


# CLI Commands
@click.group()
@click.option(
    "-v",
    "--verbosity",
    type=click.Choice([v.value for v in Verbosity]),
    default=Verbosity.INFO.value,
    help="Set output verbosity: info, warning, or error"
)
@click.pass_context
def cli(ctx, verbosity):
    """Skillseeker - Search across multiple Claude skill and MCP marketplaces"""
    ctx.ensure_object(dict)
    set_verbosity(verbosity)
    ctx.obj["verbosity"] = verbosity


@cli.command()
@click.option("-q", "--query", help="Search query")
@click.option(
    "-s",
    "--source",
    type=click.Choice([s.value for s in Source]),
    multiple=True,
    default=["all"],
    help="Sources to search",
)
@click.option(
    "-t",
    "--type",
    "resource_type",
    type=click.Choice([t.value for t in ResourceType]),
    default="all",
    help="Resource type filter",
)
@click.option("--min-stars", type=int, help="Minimum GitHub stars")
@click.option("--min-downloads", type=int, help="Minimum downloads")
@click.option("-c", "--category", help="Category filter")
@click.option("-a", "--author", help="Author filter")
@click.option(
    "--sort",
    type=click.Choice(["stars", "downloads", "name"]),
    default="stars",
    help="Sort by field",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "simple"]),
    default="table",
    help="Output format",
)
@click.option("-o", "--output", type=click.Path(), help="Save results to file")
@click.option("-i", "--interactive", "interactive_mode", is_flag=True, help="Interactive mode: select results to install/copy")
@click.option("-g", "--global", "global_install", is_flag=True, help="Install globally (used with --interactive)")
@click.pass_context
def search(ctx, query, source, resource_type, min_stars, min_downloads, category, author, sort, output_format, output, interactive_mode, global_install):
    """Search and aggregate Claude skills, plugins, and MCP servers"""
    sources = [Source(s) for s in source]
    rtype = ResourceType(resource_type) if resource_type else None

    # Check for at least one usable source
    has_skillsmp = bool(os.environ.get("SKILLSMP_API_KEY"))
    has_smithery = bool(os.environ.get("SMITHERY_API_KEY"))

    if not (has_skillsmp or has_smithery):
        console.print("[red]Error: No API keys configured.[/red]")
        console.print("Set at least one of: SKILLSMP_API_KEY or SMITHERY_API_KEY")
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
    elif interactive_mode:
        # Show results first, then interactive selection
        display_results(results, output_format)
        console.print(f"\n[dim]Found {len(results)} resources[/dim]")
        interactive_select(results, global_install)
        return
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
@click.argument("sources", nargs=-1, required=True)
@click.option("-g", "--global", "global_install", is_flag=True, help="Install globally to ~/.claude/skills/")
@click.option("-n", "--name", help="Override the skill name (only works with single source)")
@click.option("--dry-run", is_flag=True, help="Show what would be installed without installing")
def install(sources, global_install, name, dry_run):
    """
    Install skill(s) from GitHub URLs or SkillsMP.

    SOURCES can be one or more of:
    - GitHub URLs (https://github.com/owner/repo or path to SKILL.md)
    - SkillsMP skill IDs (e.g., "zenobi-us/postgres-pro")

    Examples:
        skillseeker install https://github.com/user/repo
        skillseeker install https://github.com/user/repo/tree/main/skills/my-skill
        skillseeker install zenobi-us/postgres-pro --global
        skillseeker install skill1 skill2 skill3  # Install multiple
    """
    # Handle --name with multiple sources
    if name and len(sources) > 1:
        console.print("[yellow]Warning: --name only applies to the first source when installing multiple.[/yellow]")

    # Multiple sources: install each one
    if len(sources) > 1:

        async def _install_multiple_sources():
            success_count = 0
            fail_count = 0

            for i, source in enumerate(sources):
                console.print(f"\n[cyan]Installing ({i+1}/{len(sources)}): {source}[/cyan]")
                try:
                    # Only apply name override to first source
                    name_override = name if i == 0 else None
                    await do_single_install(source, global_install, name_override, dry_run)
                    success_count += 1
                except Exception as e:
                    console.print(f"[red]Failed to install {source}: {e}[/red]")
                    fail_count += 1

            return success_count, fail_count

        success_count, fail_count = asyncio.run(_install_multiple_sources())
        console.print(f"\n[bold]Installation complete:[/bold]")
        console.print(f"  [green]✓ Succeeded: {success_count}[/green]")
        if fail_count:
            console.print(f"  [red]✗ Failed: {fail_count}[/red]")
        return

    # Single source: use original inline logic for better UX
    source = sources[0]

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
                    console.print(
                        "Supported: GitHub URLs (github.com, raw.githubusercontent.com)",
                        _verbosity=Verbosity.ERROR,
                    )
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
                        desc_match = re.search(
                            r'^description:\s*["\']?([^"\'\n]+)',
                            frontmatter,
                            re.MULTILINE,
                        )
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
        console.print(f"[dim]Path: {install_path}[/dim]", _verbosity=Verbosity.WARNING)
        return

    if not yes:
        if not Confirm.ask(f"[yellow]Remove skill '{skill_name}' from {location}?[/yellow]"):
            console.print("[dim]Uninstall cancelled[/dim]")
            return

    shutil.rmtree(install_path)
    console.print(f"[green]✓ Uninstalled '{skill_name}' from {location}[/green]")


async def fetch_unresolved_comments(
    owner: str, repo: str, pr_number: int, github_token: str
) -> list[dict]:
    """Fetch unresolved review comments from a GitHub pull request."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }

        # Fetch review comments with the comfort-fade preview to get resolved status
        # https://docs.github.com/en/rest/pulls/comments
        headers["Accept"] = "application/vnd.github.comfort-fade-preview+json"

        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/comments"
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        comments = response.json()

        # Filter for top-level, unresolved comments
        # A comment is unresolved if:
        # 1. It's a top-level comment (in_reply_to_id is None)
        # 2. It's not marked as resolved (GitHub API comfort-fade preview)
        unresolved = []
        for comment in comments:
            # Only include top-level comments (thread starters)
            if comment.get("in_reply_to_id") is None:
                # Check if the thread is resolved
                # The API may not always include this field, so we treat missing as unresolved
                if not comment.get("resolved", False):
                    unresolved.append(comment)

        return unresolved


async def create_github_issue(
    owner: str, repo: str, title: str, body: str, github_token: str
) -> dict:
    """Create a GitHub issue."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "X-GitHub-Api-Version": "2022-11-28"
        }

        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        data = {
            "title": title,
            "body": body
        }

        response = await client.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()


def format_comments_as_issue(comments: list[dict], pr_number: int, pr_url: str) -> tuple[str, str]:
    """Format unresolved comments into an issue title and body."""
    title = f"Unresolved comments from PR #{pr_number}"

    body_parts = [
        f"This issue aggregates unresolved review comments from PR #{pr_number}",
        f"PR URL: {pr_url}",
        "",
        "## Unresolved Comments",
        ""
    ]

    for i, comment in enumerate(comments, 1):
        author = comment.get("user", {}).get("login", "unknown")
        path = comment.get("path", "unknown file")
        line = comment.get("line") or comment.get("original_line", "?")
        body = comment.get("body", "")
        comment_url = comment.get("html_url", "")

        body_parts.append(f"### Comment {i}")
        body_parts.append(f"**File:** `{path}:{line}`")
        body_parts.append(f"**Author:** @{author}")
        body_parts.append(f"**Link:** {comment_url}")
        body_parts.append("")
        body_parts.append(body)
        body_parts.append("")
        body_parts.append("---")
        body_parts.append("")

    return title, "\n".join(body_parts)


def parse_pr_identifier(pr_input: str) -> tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Parse PR identifier from various formats:
    - Full URL: https://github.com/owner/repo/pull/123
    - Short format: owner/repo#123
    - Just number: 123 (requires repo context from git)

    Returns (owner, repo, pr_number)
    """
    # Try parsing as URL
    if pr_input.startswith("http://") or pr_input.startswith("https://"):
        match = re.match(r'https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)', pr_input)
        if match:
            return match.group(1), match.group(2), int(match.group(3))

    # Try parsing as owner/repo#123
    match = re.match(r'([^/]+)/([^#]+)#(\d+)', pr_input)
    if match:
        return match.group(1), match.group(2), int(match.group(3))

    # Try parsing as just a number
    if pr_input.isdigit():
        return None, None, int(pr_input)

    return None, None, None


def get_repo_from_git() -> tuple[Optional[str], Optional[str]]:
    """Extract owner/repo from git remote."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True
        )
        remote_url = result.stdout.strip()

        # Parse git@github.com:owner/repo.git or https://github.com/owner/repo.git
        match = re.search(r'github\.com[:/]([^/]+)/([^/\.]+)', remote_url)
        if match:
            return match.group(1), match.group(2)
    except Exception:
        pass

    return None, None


@cli.command()
@click.argument("pr_identifier")
@click.option("--token", help="GitHub personal access token (or set GITHUB_TOKEN env var)")
@click.option("--dry-run", is_flag=True, help="Show what would be created without creating")
def create_issue(pr_identifier, token, dry_run):
    """
    Create a GitHub issue from unresolved PR review comments.

    PR_IDENTIFIER can be:
    - Full URL: https://github.com/owner/repo/pull/123
    - Short format: owner/repo#123
    - Just number: 123 (uses current git repo)

    Examples:
        skillseeker create-issue https://github.com/owner/repo/pull/123
        skillseeker create-issue owner/repo#123
        skillseeker create-issue 123
    """
    github_token = token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        console.print("[red]Error: GitHub token required.[/red]")
        console.print(
            "Set GITHUB_TOKEN environment variable or use --token option",
            _verbosity=Verbosity.ERROR,
        )
        sys.exit(1)

    # Parse PR identifier
    owner, repo, pr_number = parse_pr_identifier(pr_identifier)

    # If owner/repo not provided, try to get from git
    if not owner or not repo:
        git_owner, git_repo = get_repo_from_git()
        owner = owner or git_owner
        repo = repo or git_repo

    if not owner or not repo or not pr_number:
        console.print("[red]Error: Could not parse PR identifier.[/red]")
        console.print(
            "Expected format: https://github.com/owner/repo/pull/123, owner/repo#123, or 123",
            _verbosity=Verbosity.ERROR,
        )
        sys.exit(1)

    pr_url = f"https://github.com/{owner}/{repo}/pull/{pr_number}"
    console.print(f"[dim]Fetching unresolved comments from {pr_url}...[/dim]")

    async def run():
        try:
            # Fetch unresolved comments
            comments = await fetch_unresolved_comments(owner, repo, pr_number, github_token)

            if not comments:
                console.print("[yellow]No unresolved comments found.[/yellow]")
                return

            console.print(f"[green]Found {len(comments)} unresolved comment(s)[/green]")

            # Format as issue
            title, body = format_comments_as_issue(comments, pr_number, pr_url)

            if dry_run:
                console.print(Panel.fit(
                    f"[bold]Title:[/bold]\n{title}\n\n"
                    f"[bold]Body:[/bold]\n{body[:500]}...",
                    title="Dry Run - Would Create Issue"
                ))
                return

            # Create issue
            console.print("[dim]Creating issue...[/dim]")
            issue = await create_github_issue(owner, repo, title, body, github_token)

            console.print(Panel.fit(
                f"[green]✓ Issue created successfully![/green]\n\n"
                f"[bold]Issue:[/bold] #{issue['number']}\n"
                f"[bold]Title:[/bold] {issue['title']}\n"
                f"[bold]URL:[/bold] {issue['html_url']}",
                title="Issue Created"
            ))

        except httpx.HTTPStatusError as e:
            console.print(f"[red]GitHub API error: {e.response.status_code}[/red]")
            console.print(f"[red]{e.response.text}[/red]", _verbosity=Verbosity.ERROR)
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    asyncio.run(run())


if __name__ == "__main__":
    cli()
