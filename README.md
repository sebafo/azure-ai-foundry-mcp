# Azure AI Foundry Agent MCP

This project provides an MCP (Model Context Protocol) server for integrating with Azure AI Agent Service. It is designed to help you interact with Azure-hosted AI agents, query them, and manage agent-related workflows in a secure and scalable way.

## Features
- Automatic discovery and registration of all Azure AI Agents from your service
- Dynamically creates MCP tools for each agent in your Azure AI Agent Service
- Supports both local and web transport modes
- Regular background sync to detect new or changed agents

## Project Structure
- `azure_agent_mcp_server/` — Main server code and tools
- `.env` — Environment variables for configuration (see below)
- `pyproject.toml` — Project dependencies and metadata
- `uv.lock` — Lockfile for reproducible installs (managed by [uv](https://github.com/astral-sh/uv))

## Notes
- The most recent version of the AI Foundry SDK requires an AI Foundry Project. It doesn't support a hub based project currently.
For more information about Azure AI Foundry project types, see the [official documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/what-is-azure-ai-foundry#project-types).

## Getting Started

### 1. Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended for dependency management)

### 2. Setup
1. **Clone the repository**
2. **Configure environment variables:**
   - Copy the provided `.env` file or create your own. Example:
     ```env
     PROJECT_ENDPOINT=your-ai-foundry-project-endpoint
     ```
   - This variable is required for connecting to your Azure AI Agent Service.

3. **Install dependencies:**
   - Using [uv](https://github.com/astral-sh/uv):
     ```sh
     uv pip install -r pyproject.toml
     ```
   - Or, to sync with the lockfile:
     ```sh
     uv pip sync
     ```
   - Alternatively, you can use `pip` or `pipx` if you prefer.

### 3. Running the Server
The server can run in two modes:

* **Local mode** (default):
  ```sh
  uv run -m azure_agent_mcp_server  
  # Alternatively, you can run:
  # python -m azure_agent_mcp_server
  ```

* **Web mode** (accessible via HTTP):
  ```sh
  # Set SERVER_TYPE=web in your .env file, or run with:
  SERVER_TYPE=web uv run -m azure_agent_mcp_server 
  # Alteratively, you can run:
  # SERVER_TYPE=web python -m azure_agent_mcp_server
  ```

When started, the server will:
1. Connect to Azure AI Agent Service using the provided endpoint
2. Automatically discover all your agents
3. Create MCP tools for each agent
4. Periodically check for new or updated agents every 60 seconds

### 4. Querying Agents in VSCode / GitHub Copilot
1. Add MCP Server to VSCode settings:
   ```json
   "mcp": {
        "servers": {
            "Azure AI Agents Server": {
                "command": "uv",
                "args": [
                    "--directory",
                    "/YOUR/PROJECT/PATH",
                    "run",
                    "-m",
                    "azure_agent_mcp_server"
                ],
                "env": {
                    "PROJECT_ENDPOINT": "your-ai-foundry-project-endpoint"
                }
            }
        }
    },
   ```

2. After the server starts, it automatically discovers all agents from your Azure AI Agent Service and makes them available as MCP tools with names based on the agent names (converted to snake_case).

3. You can then use these tools directly in GitHub Copilot or any other MCP-compatible client.

## Environment Variables and Configuration

The MCP server can be configured using the following environment variables in your `.env` file:

- `PROJECT_ENDPOINT`: Azure AI Foundry project endpoint (required)
- `SERVER_TYPE`: Set to "local" (default) or "web" to choose the transport mode
- `SERVER_PORT`: Port number for web mode (default: 8000)
- `SERVER_PATH`: Path for web mode (default: "/")
- `UPDATE_INTERVAL`: How often (in seconds) to check for new or updated agents (default: 60)
- `LOG_LEVEL`: Set the logging level (default: "WARNING"). Options include "DEBUG", "INFO", "WARNING", "ERROR", and "CRITICAL".

Example `.env` file:
```env
PROJECT_ENDPOINT=your-ai-foundry-project-endpoint
SERVER_TYPE=web
SERVER_PORT=9000
UPDATE_INTERVAL=120
LOG_LEVEL=INFO
```

**Note**: Never commit secrets to version control.

## About uv
[uv](https://github.com/astral-sh/uv) is a fast, modern Python package and project manager. It replaces tools like `pip`, `pip-tools`, `pipx`, `poetry`, and `virtualenv`, and is recommended for reproducible, efficient dependency management in this project.

- See [uv documentation](https://docs.astral.sh/uv/) for more details.

## How Agent Tools Work

The system automatically:

1. Connects to Azure AI Agent Service on startup
2. Discovers all agents available in your service
3. Creates an MCP tool for each agent, converting the agent name to snake_case for the function name
4. Sets the tool description to match the agent description 
5. Periodically checks for new, updated, or deleted agents
6. Updates the available tools accordingly

Example:
- An agent named "Coding Guidelines" becomes a tool named `coding_guidelines`
- An agent named "Python Expert" becomes a tool named `python_expert`
