# Skillseeker

A CLI tool that aggregates Claude skills, plugins, agents, and MCP servers from multiple marketplaces into a unified, searchable format.

## Features

- **Multi-source aggregation**: Searches SkillsMP and Smithery
- **API-first approach**: Uses native REST APIs for fast, reliable results
- **Parallel searching**: Fetches from all sources concurrently for speed
- **Unified schema**: Normalizes data from different sources into a consistent format
- **Powerful filtering**: Filter by type, stars, downloads, category, author
- **Multiple output formats**: Table, JSON, or simple text

## Installation

```bash
# Clone the repository
git clone https://github.com/youruser/skillseeker
cd skillseeker

# Install with pip
pip install -e .

# Or install dependencies directly
pip install click rich httpx pydantic python-dotenv
```

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env` and configure your API keys:

```bash
# Required for SkillsMP (recommended - has AI semantic search)
SKILLSMP_API_KEY=sk_live_your_key_here

# Required for Smithery MCP registry
SMITHERY_API_KEY=your_smithery_key
```

**Where to get API keys:**
- SkillsMP: https://skillsmp.com/docs/api (requires account)
- Smithery: https://smithery.ai/account/api-keys

## Usage

### Basic Search

```bash
# Search all sources
skillseeker search -q "database"

# Search specific sources
skillseeker search -s skillsmp -q "web scraping"
skillseeker search -s smithery -q "postgres"

# Search without query (browse all)
skillseeker search -s smithery
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
skillseeker search -q "api" -f table

# JSON format
skillseeker search -q "api" -f json

# Simple text format
skillseeker search -q "api" -f simple

# Save to file
skillseeker search -q "api" -f json -o results.json
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

### Installing Skills

```bash
# Install a skill from SkillsMP (searches and fetches from GitHub)
skillseeker install postgres-pro

# Install globally (to ~/.claude/skills/)
skillseeker install postgres-pro --global

# Install from a GitHub URL directly
skillseeker install https://github.com/user/repo/tree/main/skills/my-skill

# Preview what would be installed without installing
skillseeker install postgres-pro --dry-run

# Override the skill name
skillseeker install https://github.com/user/repo --name my-custom-name
```

### Managing Installed Skills

```bash
# List all installed skills (global and local)
skillseeker installed

# List only global skills
skillseeker installed --global

# List only local/project skills
skillseeker installed --local

# Uninstall a local skill
skillseeker uninstall my-skill

# Uninstall a global skill
skillseeker uninstall my-skill --global

# Uninstall without confirmation
skillseeker uninstall my-skill -y
```

### Utility Commands

```bash
# List available sources and their API status
skillseeker sources

# List resource types
skillseeker types

# Export unified schema
skillseeker schema -o my-schema.json
```

## Sources

| Source | Type | API | Description |
|--------|------|-----|-------------|
| **skillsmp** | Skills | REST API | 10,000+ GitHub-sourced Claude skills with AI semantic search |
| **smithery** | MCP Servers | REST API | Smithery MCP server registry |

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
  "source": "skillsmp|smithery",
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
skillseeker search -s skillsmp -f json -o all-skills.json
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
black .
ruff check .
```

## Architecture

```
query > API clients (parallel) > normalize > filter > sort > return
           │
           ├── SkillsMP API (native REST)
           └── Smithery API (native REST)
```

The CLI uses native REST APIs for reliability and speed.

## License

MIT
