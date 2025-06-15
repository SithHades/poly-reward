# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
  
## Project Overview

This is a Python project called "poly-reward" that appears to be in early development. The project uses `uv` for dependency management and includes dependencies for async HTTP (`aiohttp`), dependency injection (`dependency-injector`), CLOB client functionality (`py-clob-client`), and YAML parsing (`pyyaml`).
 
## Development Environment
- **Python Version**: 3.13 (specified in `.python-version`)
- **Package Manager**: `uv` (evident from `uv.lock` file)
- **Dependencies**: Defined in `pyproject.toml`
 
## Common Commands
  
### Package Management
- Install dependencies: `uv sync`
- Add new dependency: `uv add <package-name>`
- Remove dependency: `uv remove <package-name>`
  
### Running the Application
- Run main application: `uv run python main.py`
- Run with uv: `uv run main.py`

### Development Workflow
- Install project in development mode: `uv sync --dev`
- Run Python scripts: `uv run python <script.py>`

## Project Structure

The project is currently minimal with:
- `main.py`: Entry point with basic "Hello World" functionality
- `pyproject.toml`: Project configuration and dependencies
- `uv.lock`: Locked dependency versions

## Key Dependencies

- **aiohttp**: Async HTTP client/server framework
- **dependency-injector**: Dependency injection framework
- **py-clob-client**: CLOB (Central Limit Order Book) client
- **pyyaml**: YAML parser and emitter

This suggests the project may be related to financial trading or order book management with async capabilities and dependency injection architecture.

## Task Master AI Integration

The project includes extensive task management workflow configuration in `.windsurfrules` for coordinating development using the `task-master-ai` system. Key commands:
- `task-master list`: View current tasks
- `task-master next`: Get next task to work on
- `task-master set-status --id=<id> --status=done`: Mark tasks complete
- `task-master expand --id=<id>`: Break down complex tasks
