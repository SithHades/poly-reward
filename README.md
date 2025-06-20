# poly-reward

A Python project for managing reward systems with async capabilities and dependency injection architecture.

## Features

- Async HTTP client/server framework (aiohttp)
- Dependency injection (dependency-injector)
- CLOB (Central Limit Order Book) client integration
- YAML configuration support

## Requirements

- Python 3.13+
- uv package manager

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd poly-reward
```

2. Install dependencies:
```bash
uv sync
```

3. Run the application:
```bash
uv run python main.py
```

## Development

### Project Structure

```
poly-reward/
src/          # Source code
tests/        # Test files
config/       # Configuration files
main.py       # Application entry point
pyproject.toml # Project configuration
```

### Dependencies

- **aiohttp**: Async HTTP client/server framework
- **dependency-injector**: Dependency injection framework
- **py-clob-client**: CLOB (Central Limit Order Book) client
- **pyyaml**: YAML parser and emitter

### Common Commands

```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package-name>

# Remove dependency
uv remove <package-name>

# Run application
uv run python main.py

# Development mode
uv sync --dev
```

## License

This project is licensed under the MIT License.
