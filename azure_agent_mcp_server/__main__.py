"""
Azure AI Agent Service MCP Server

This server connects to Azure AI Agent Service and dynamically registers
agents as FastMCP tools that can be queried by clients.
"""

import os
import sys
import logging
import asyncio
import re
from typing import Dict, Any
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import Agent, MessageRole, MessageTextContent, ListSortOrder
from azure.identity.aio import DefaultAzureCredential
from azure.core.exceptions import ServiceRequestError, HttpResponseError, ResourceNotFoundError

# Configure structured logging with timestamp, module, and log level
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("azure_agent_mcp")

# Global variables for client and agent cache
agents_client = None
registered_agents: Dict[str, Dict[str, Any]] = {}  # Dictionary to keep track of registered agent tools by ID
server_initialized = False
server_type = None
server_port = None
server_path = None
update_interval = 300  # Default update interval in seconds

# Configuration constants
MAX_RETRIES = 2
BASE_BACKOFF_DELAY = 1  # Base delay for exponential backoff in seconds
MAX_POLL_DELAY = 5  # Maximum polling delay in seconds
DEFAULT_PORT = 8000
DEFAULT_PATH = "/"

def initialize_server() -> bool:
    """
    Initialize the Azure AI Agent client and server configuration.
    
    Returns:
        bool: True if initialization succeeded, False otherwise.
    """
    global agents_client, server_type, server_port, server_path, update_interval

    # Load environment variables from .env file if present
    load_dotenv()
    
    # Load configuration from environment variables
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    update_interval = int(os.getenv("UPDATE_INTERVAL", update_interval))

    # Configure server type and networking
    server_type = os.getenv("SERVER_TYPE", "local").lower()
    server_port = int(os.getenv("SERVER_PORT", DEFAULT_PORT))
    server_path = os.getenv("SERVER_PATH", DEFAULT_PATH)

    log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
    valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}

    if log_level not in valid_levels:
        log_level = "WARNING"  # fallback or raise an error

    logger.setLevel(log_level)

    # Validate server type configuration
    if server_type not in ["local", "web"]:
        logger.error(f"Invalid server type: {server_type}. Must be 'local' or 'web'.")
        return False

    # Validate essential environment variables
    if not project_endpoint:
        logger.error("Missing required environment variable: PROJECT_ENDPOINT")
        return False

    try:
        # Initialize the Azure AI Agent client with managed identity authentication
        agents_client = AgentsClient(
            credential=DefaultAzureCredential(),
            endpoint=project_endpoint
        )
        logger.info(f"Successfully initialized Azure AI Agent client for endpoint: {project_endpoint}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to initialize AgentsClient: {str(e)}")
        return False


async def query_agent(agent_id: str, query: str) -> str:
    """
    Query an Azure AI Agent with retry logic and exponential backoff.
    
    Args:
        agent_id: The ID of the agent to query
        query: The text query to send to the agent
        
    Returns:
        str: The formatted response from the agent
        
    Raises:
        Exception: If the query fails after all retry attempts
    """
    response_message = None
    
    # Implement retry with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            # Create a new thread for each conversation
            thread = await agents_client.threads.create()
            thread_id = thread.id
            logger.debug(f"Created thread {thread_id} for agent {agent_id}")

            # Add the user's message to the thread
            await agents_client.messages.create(
                thread_id=thread_id, role=MessageRole.USER, content=query
            )

            # Start the agent run
            run = await agents_client.runs.create(thread_id=thread_id, agent_id=agent_id)
            run_id = run.id
            logger.debug(f"Created run {run_id} for agent {agent_id}")

            # Poll with dynamic backoff until completion
            poll_count = 0
            while run.status in ["queued", "in_progress", "requires_action"]:
                # Calculate dynamic sleep time based on poll count with exponential backoff
                poll_delay = min(BASE_BACKOFF_DELAY * (1.5 ** poll_count), MAX_POLL_DELAY)
                await asyncio.sleep(poll_delay)
                
                # Get updated run status
                run = await agents_client.runs.get(thread_id=thread_id, run_id=run.id)
                poll_count += 1
                logger.debug(f"Run {run_id} status: {run.status} (poll #{poll_count})")

            if run.status == "failed":
                error_msg = f"Agent run failed: {run.last_error}" if hasattr(run, 'last_error') else "Agent run failed with no error details"
                logger.error(f"Run {run_id} failed: {error_msg}")
                
                if attempt < MAX_RETRIES - 1:
                    backoff_time = BASE_BACKOFF_DELAY * (2 ** attempt)
                    logger.info(f"Retrying in {backoff_time} seconds (attempt {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(backoff_time)
                    continue
                    
                return f"Error: {error_msg}"

            # Get the agent's response messages
            messages = agents_client.messages.list(
                thread_id=thread.id,
                order=ListSortOrder.DESCENDING,
            )
            
            async for msg in messages:
                if not msg.content:
                    logger.warning(f"Empty message content received from agent {agent_id}")
                    continue
                    
                last_part = msg.content[-1]
                if isinstance(last_part, MessageTextContent):
                    if msg.role == MessageRole.AGENT:
                        response_message = msg
                        logger.debug(f"Agent response received for run {run_id}")
                        break

            # Process the successful response
            return _format_agent_response(response_message)
            
        except ResourceNotFoundError as e:
            logger.error(f"Resource not found error for agent {agent_id}: {str(e)}")
            return f"Error: The agent {agent_id} could not be found or accessed."
            
        except (ServiceRequestError, HttpResponseError) as e:
            logger.error(f"Service error querying agent {agent_id} (attempt {attempt+1}/{MAX_RETRIES}): {str(e)}")
            
            if attempt < MAX_RETRIES - 1:
                backoff_time = BASE_BACKOFF_DELAY * (2 ** attempt)
                logger.info(f"Retrying in {backoff_time} seconds")
                await asyncio.sleep(backoff_time)
            else:
                logger.error(f"Failed to query agent {agent_id} after {MAX_RETRIES} attempts")
                return f"Error: Failed to get a response after multiple attempts: {str(e)}"
                
        except Exception as e:
            logger.error(f"Unexpected error querying agent {agent_id}: {str(e)}")
            raise

    # This should not be reached if retries are working correctly
    return "Error: Failed to get a response after multiple attempts."


def _format_agent_response(response_message) -> str:
    """
    Format the agent's response message with proper markdown and citations.
    
    Args:
        response_message: The response message from the agent
        
    Returns:
        str: Formatted response text with citations if available
    """
    if not response_message:
        return "No response received from the agent."
        
    result = ""
    citations = []

    # Collect text content
    for text_message in response_message.text_messages:
        result += text_message.text.value + "\n"

    # Collect citations
    try:
        for annotation in response_message.url_citation_annotations:
            citation = f"[{annotation.url_citation.title}]({annotation.url_citation.url})"
            if citation not in citations:
                citations.append(citation)
    except (AttributeError, KeyError) as e:
        logger.debug(f"No citations found in response: {str(e)}")

    # Add citations section if any were found
    if citations:
        result += "\n\n## Sources\n"
        for citation in citations:
            result += f"- {citation}\n"

    return result.strip()


def to_snake_case(text: str) -> str:
    """
    Convert a string to snake_case, preserving existing underscores.
    
    Args:
        text: The text to convert
        
    Returns:
        str: The text converted to snake_case
    """
    # Replace non-alphanumeric characters (except underscores) with spaces
    text = re.sub(r'[^a-zA-Z0-9_\s]', '', text)
    # Replace spaces and runs of underscores with a single underscore and convert to lowercase
    return re.sub(r'[\s_]+', '_', text).lower()


def create_agent_tool(agent: Agent, function_name: str) -> None:
    """
    Create a tool for an agent and register it with the MCP framework.
    
    Args:
        agent: The agent object containing ID, name and description
        function_name: The function name to use for the tool
    """
    async def agent_tool(query: str, ctx: Context = None) -> str:
        """Query the specified Azure AI Agent."""
        if not server_initialized:
            return "Error: Azure AI Agent server is not initialized. Check server logs for details."

        try:
            response = await query_agent(agent.id, query)
            return f"## Response from {agent.name} Agent\n\n{response}"
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in agent tool {function_name}: {error_msg}")
            return f"Error querying {agent.name} agent: {error_msg}"
        
    # Set function metadata
    agent_tool.__name__ = function_name
    agent_tool.__doc__ = agent.description
    
    # Register with MCP framework
    mcp.add_tool(
        fn=agent_tool,
        name=function_name,
        description=agent.description
    )
    
    # Store in registry for tracking
    registered_agents[agent.id] = {
        "name": agent.name,
        "description": agent.description,
        "function_name": function_name
    }


async def sync_agents() -> Dict[str, Agent]:
    """
    Sync agents between Azure AI Agent Service and MCP tools.
    
    This function queries the Azure AI Agent Service for available agents,
    registers new agents as tools, updates existing agents, and removes
    deleted agents.
    
    Returns:
        Dict[str, Agent]: Dictionary mapping agent IDs to their Agent objects
    """
    if not server_initialized:
        logger.warning("Cannot sync agents: Server not initialized.")
        return {}

    try:
        # Get current agents from the service
        current_agents: Dict[str, Agent] = {}
        agent_count = 0
        
        # Process the AsyncIterable response
        async for agent in agents_client.list_agents():
            current_agents[agent.id] = agent
            agent_count += 1
            
        if agent_count == 0:
            logger.warning("No agents found in the Azure AI Agent Service.")
            return current_agents
            
        logger.info(f"Found {agent_count} agents in Azure AI Agent Service")
        
        # Add or update agents
        _add_or_update_agents(current_agents)
        
        # Remove deleted agents
        _remove_deleted_agents(current_agents)
                
        return current_agents
        
    except Exception as e:
        logger.error(f"Failed to sync agents: {str(e)}")
        return {}


def _add_or_update_agents(current_agents: Dict[str, Agent]) -> None:
    """
    Add new agents as tools and update existing agents if needed.
    
    Args:
        current_agents: Dictionary of active agents from Azure AI Agent Service
    """
    for agent_id, agent in current_agents.items():
        if agent_id not in registered_agents:
            # New agent - add it
            function_name = to_snake_case(agent.name)
            create_agent_tool(agent, function_name)
            
            logger.info(f"Added agent tool: {agent.name} (ID: {agent.id}, Function: {function_name})")
            logger.debug(f"Agent details - Name: {agent.name}, ID: {agent.id}, Description: {agent.description}")
            
        elif (agent.name != registered_agents[agent_id]["name"] or 
              agent.description != registered_agents[agent_id]["description"]):
            # Update existing agent if name or description changed
            old_function_name = registered_agents[agent_id]["function_name"]
            mcp.remove_tool(old_function_name)
            
            # Create updated tool
            function_name = to_snake_case(agent.name)
            create_agent_tool(agent, function_name)
            
            logger.info(f"Updated agent tool: {agent.name} (old function: {old_function_name}, new function: {function_name})")


def _remove_deleted_agents(current_agents: Dict[str, Agent]) -> None:
    """
    Remove tools for agents that no longer exist in Azure AI Agent Service.
    
    Args:
        current_agents: Dictionary of active agents from Azure AI Agent Service
    """
    for agent_id in list(registered_agents.keys()):
        if agent_id not in current_agents:
            agent_name = registered_agents[agent_id]["name"]
            function_name = registered_agents[agent_id]["function_name"]
            
            logger.info(f"Removing agent tool: {agent_name} (function: {function_name})")
            mcp.remove_tool(function_name)
            del registered_agents[agent_id]


async def register_agents() -> None:
    """Register all available agents as MCP tools."""
    logger.info("Registering agents as tools...")
    await sync_agents()
    logger.info(f"Registered {len(registered_agents)} agents as MCP tools")


async def update_tools() -> None:
    """Update tools based on changes in the Azure AI Agents."""
    logger.debug("Checking for agent updates...")
    current_agents = await sync_agents()
    logger.debug(f"Agent sync complete, {len(current_agents)} agents available")


async def periodic_update_task() -> None:
    """Run the update_tools function periodically."""
    while True:
        try:
            await asyncio.sleep(update_interval)
            logger.debug(f"Running scheduled agent sync (interval: {update_interval}s)")
            await update_tools()
        except asyncio.CancelledError:
            logger.info("Periodic update task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in periodic update task: {str(e)}")


async def shutdown() -> None:
    """Perform cleanup operations before server shutdown."""
    logger.info("Shutting down Azure AI Agent MCP Server...")
    # Add any cleanup code here if needed


async def main() -> None:
    """Main entry point for the async server."""
    # Register agents
    await register_agents()
    
    # Start the periodic update task
    update_task = asyncio.create_task(periodic_update_task())
    logger.info(f"MCP server is running with periodic updates every {update_interval} seconds")
    
    try:
        # Run the MCP server
        if server_type == "web":
            await mcp.run_async(transport="streamable-http", host="0.0.0.0", port=server_port, path=server_path)
        else:
            await mcp.run_async()
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
    finally:
        # Ensure cleanup on exit
        update_task.cancel()
        try:
            await update_task
        except asyncio.CancelledError:
            pass
        
        await shutdown()


# Initialize MCP server
mcp = FastMCP(name="azure-agent")
server_initialized = initialize_server()

if __name__ == "__main__":
    status = "successfully initialized" if server_initialized else "initialization failed"
    logger.info(f"{'='*50}")
    logger.info(f"Azure AI Agent MCP Server {status}")
    
    if not server_initialized:
        logger.error("Server initialization failed. Exiting...")
        sys.exit(1)
        
    logger.info("Starting server...")
    
    # Run the main async function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Uncaught exception: {str(e)}")
        sys.exit(1)
