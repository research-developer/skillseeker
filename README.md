# Skillseeker

A CLI tool that aggregates Claude skills, plugins, agents, and MCP servers from multiple marketplaces into a unified, searchable format.

## Features

- **Multi-source aggregation**: Scrapes from skillsmp.com, claude-plugins.dev, claudemarketplaces.com, and mcp.so
- **Parallel scraping**: Fetches from all sources concurrently for speed
- **Unified schema**: Normalizes data from different sources into a consistent format
- **Powerful filtering**: Filter by type, stars, downloads, category, author
- **Multiple output formats**: Table, JSON, or simple text
- **Extensible**: Easy to add new sources

## Installation

```bash
# Clone the repository
git clone https://github.com/youruser/skillseekerregator
cd skillseekerregator

# Install with pip
pip install -e .

# Or install dependencies directly
pip install click rich firecrawl-py httpx pydantic
```

## Configuration

Set your FireCrawl API key:

```bash
export FIRECRAWL_API_KEY="fc-YOUR-API-KEY"
```

Or use a `.env` file (recommended):

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and add your API key.

Or pass it via the CLI:

```bash
skillseeker --api-key "fc-YOUR-API-KEY" search
```

## Usage

### Basic Search

```bash
# Search all sources
skillseeker search

# Search with a query
skillseeker search -q "database"

# Search specific sources
skillseeker search -s skillsmp -s mcp_so -q "postgres"
```

### Filtering

```bash
# Filter by resource type
skillseeker search -t mcp_server

# Filter by minimum stars
skillseeker search --min-stars 100

# Filter by category
skillseeker search -c "database"

# Filter by author
skillseeker search -a "anthropic"

# Combine filters
skillseeker search -t skill --min-stars 50 -c development
```

### Output Formats

```bash
# Table format (default)
skillseeker search -f table

# JSON format
skillseeker search -f json

# Simple text format
skillseeker search -f simple

# Save to file
skillseeker search -f json -o results.json
```

### Sorting

```bash
# Sort by stars (default)
skillseeker search --sort stars

# Sort by downloads
skillseeker search --sort downloads

# Sort by name
skillseeker search --sort name
```

### Other Commands

```bash
# List available sources
skillseeker sources

# List resource types
skillseeker types

# Export unified schema
skillseeker schema -o my-schema.json
```

## Sources

| Source | URL | Description |
|--------|-----|-------------|
| skillsmp | https://skillsmp.com/ | 10,000+ GitHub-sourced Claude skills |
| claude_plugins | https://claude-plugins.dev/skills | Community skill directory |
| claudemarketplaces | https://claudemarketplaces.com/ | Plugin marketplace aggregator |
| mcp_so | https://mcp.so/ | MCP server directory |

## Resource Types

- `skill` - Claude Code skills
- `plugin` - Claude Code plugins
- `mcp_server` - Model Context Protocol servers
- `agent` - Claude agents/subagents
- `command` - Slash commands
- `all` - All types

## Unified Schema

All resources are normalized to this schema:

```json
{
  "name": "string",
  "description": "string", 
  "type": "skill|plugin|mcp_server|agent|command",
  "source": "skillsmp|claude_plugins|claudemarketplaces|mcp_so",
  "url": "string",
  "author": "string|null",
  "github_url": "string|null",
  "stars": "integer|null",
  "downloads": "integer|null",
  "last_updated": "datetime|null",
  "install_command": "string|null",
  "category": "string|null",
  "language": "string|null",
  "tags": ["string"]
}
```

## Examples

### Find popular MCP database servers

```bash
skillseeker search -t mcp_server -c database --min-stars 100 --sort stars
```

### Export all skills to JSON

```bash
skillseeker search -t skill -f json -o all-skills.json
```

### Search for authentication-related resources

```bash
skillseeker search -q "auth" -f table
```

### Find anthropic's official resources

```bash
skillseeker search -a anthropic -f simple
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
# Format code
black .
ruff check .
```

## Architecture

```
prompt > request > parallel scraping > normalize > filter > sort > return
     │         │              │           │         │        │
     └─────────┴──────────────┴───────────┴─────────┴────────┘
                         async pipeline
```

The CLI uses FireCrawl's LLM-powered extraction to understand and parse the different marketplace formats, then normalizes everything into a unified schema.

## License

MIT
