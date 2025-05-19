"""Azure AI Agent Service MCP Server"""

import os
import sys
import logging
import asyncio
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import Agent, MessageRole
from azure.identity.aio import DefaultAzureCredential

# Set up logging configuration
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("azure_agent_mcp")
#logger.setLevel(logging.INFO) # Set to DEBUG/INFO for detailed logs

# Global variables for client and agent cache
ai_client = None
registered_agents = {}  # Dictionary to keep track of registered agent tools by ID

def initialize_server() -> bool:
    """Initialize the Azure AI Agent client asynchronously"""
    global ai_client, server_type, server_port, server_path, update_interval

    # Load environment variables
    project_connection_string = os.getenv("PROJECT_CONNECTION_STRING")
    update_interval = int(os.getenv("UPDATE_INTERVAL", 60))  # Default to 60 seconds

    # Check if the server type is valid
    server_type = os.getenv("SERVER_TYPE", "local").lower()
    server_port = int(os.getenv("SERVER_PORT", 8000))
    server_path = os.getenv("SERVER_PATH", "/")

    if server_type not in ["local", "web"]:
        logger.error(f"Invalid server type: {server_type}. Must be 'local' or 'web'.")
        return False

    # Validate essential environment variables
    if not project_connection_string:
        logger.error("Missing required environment variable: PROJECT_CONNECTION_STRING")
        return False

    try:
        credential = DefaultAzureCredential()
        ai_client = AgentsClient(
            credential=credential,
            endpoint=project_connection_string,
            user_agent="mcp-azure-ai-agent",
        )
    
        return True
    
    except Exception as e:
        logger.error(f"Failed to initialize AIProjectClient: {str(e)}")
        return False


async def query_agent(agent_id: str, query: str) -> str:
    """Query an Azure AI Agent and get the response."""
    try:
        # Always create a new thread
        thread = await ai_client.threads.create()
        thread_id = thread.id

        # Add message to thread
        await ai_client.messages.create(
            thread_id=thread_id, role=MessageRole.USER, content=query
        )

        # Process the run asynchronously
        run = await ai_client.runs.create(thread_id=thread_id, agent_id=agent_id)

        # Poll until the run is complete
        while run.status in ["queued", "in_progress", "requires_action"]:
            await asyncio.sleep(1)  # Non-blocking sleep
            run = await ai_client.runs.get(thread_id=thread_id, run_id=run.id)

        if run.status == "failed":
            error_msg = f"Agent run failed: {run.last_error}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        # Get the agent's response
        response_messages = await ai_client.messages.list(thread_id=thread_id)
        response_message = response_messages.get_last_message_by_role(MessageRole.AGENT)

        result = ""
        citations = []

        if response_message:
            # Collect text content
            for text_message in response_message.text_messages:
                result += text_message.text.value + "\n"

            # Collect citations
            for annotation in response_message.url_citation_annotations:
                citation = (
                    f"[{annotation.url_citation.title}]({annotation.url_citation.url})"
                )
                if citation not in citations:
                    citations.append(citation)

        # Add citations if any
        if citations:
            result += "\n\n## Sources\n"
            for citation in citations:
                result += f"- {citation}\n"

        return result.strip()

    except Exception as e:
        logger.error(f"Agent query failed - ID: {agent_id}, Error: {str(e)}")
        raise
    
# Helper function to convert a string to snake_case
def to_snake_case(text: str) -> str:
    """Convert a string to snake_case, preserving existing underscores."""
    import re
    # Replace non-alphanumeric characters (except underscores) with spaces
    text = re.sub(r'[^a-zA-Z0-9_\s]', '', text)
    # Replace spaces with underscores and convert to lowercase
    return re.sub(r'\s+', '_', text).lower()

# Function to create a function/tool for each agent
def create_agent_tool(agent: Agent, function_name: str):
    """Create a tool for each agent using a provided function name."""
    
    async def agent_tool(query: str, ctx: Context = None) -> str:
        """Query the specified Azure AI Agent."""
        if not server_initialized:
            return "Error: Azure AI Agent server is not initialized. Check server logs for details."

        try:
            response = await query_agent(agent.id, query)
            return f"## Response from {agent.name} Agent\n\n{response}"
        except Exception as e:
            return f"Error querying {agent.name} agent: {str(e)}"
        
    agent_tool.__name__ = function_name  # Set the function name for the tool
    agent_tool.__doc__ = agent.description  # Set the docstring for the tool

    # Register the tool with MCP
    mcp.add_tool(
        fn=agent_tool,
        name=function_name,  # Use the provided snake_case function name
        description=agent.description
    )
    
    # Store in our registry
    registered_agents[agent.id] = {
        "name": agent.name,
        "description": agent.description,
        "function_name": function_name
    }

async def sync_agents() -> dict:
    """
    Sync agents between Azure AI Agent Service and MCP tools.
    
    Returns:
        dict: Dictionary mapping agent IDs to their details
    """
    if not server_initialized:
        return {}

    try:
        # Get current agents from the service - now returns AsyncIterable["_models.Agent"]
        agents_response = ai_client.list_agents()
        # Check if the response is valid
        current_agents = {}
        
        # Process the AsyncIterable response
        agent_count = 0
        async for agent in agents_response:
            current_agents[agent.id] = agent
            agent_count += 1
            
        if agent_count == 0:
            logger.warning("No agents found in the Azure AI Agent Service.")
            return current_agents
            
        logger.info(f"Found {agent_count} agents")
        
        # Find agents to add (new agents) or update
        for agent_id, agent in current_agents.items():
            if agent_id not in registered_agents:
                # New agent - add it
                function_name = to_snake_case(agent.name)
                create_agent_tool(agent, function_name)
                
                # Log at INFO level for normal operation
                logger.info(f"Added agent tool: {agent.name} (ID: {agent.id}, Function: {function_name})")
                
                # Additional details at DEBUG level
                logger.debug(f"Agent details - Name: {agent.name}, ID: {agent.id}, " +
                           f"Description: {agent.description}, Function: {function_name}")
                
            elif (agent.name != registered_agents[agent_id]["name"] or 
                  agent.description != registered_agents[agent_id]["description"]):
                # Update existing agent if name or description changed
                old_function_name = registered_agents[agent_id]["function_name"]
                mcp.remove_tool(old_function_name)
                
                # Create updated tool
                function_name = to_snake_case(agent.name)
                create_agent_tool(agent, function_name)
                
                logger.info(f"Updated agent tool: {agent.name} (old function: {old_function_name}, new function: {function_name})")
        
        # Find agents to remove (deleted agents)
        for agent_id in list(registered_agents.keys()):
            if agent_id not in current_agents:
                agent_name = registered_agents[agent_id]["name"]
                function_name = registered_agents[agent_id]["function_name"]
                
                logger.info(f"Removing agent tool: {agent_name} (function: {function_name})")
                mcp.remove_tool(function_name)
                del registered_agents[agent_id]
                
        return current_agents
        
    except Exception as e:
        logger.error(f"Failed to sync agents: {str(e)}")
        return {}

# Register all agents as tools
async def register_agents() -> None:
    """Register all available agents as tools."""
    await sync_agents()

async def update_tools() -> None:
    """Update tools regularly based on changes in the Azure AI Agents."""
    await sync_agents()

async def periodic_update_task():
    """Run the update_tools function periodically."""
    while True:
        try:
            await asyncio.sleep(update_interval)
            await update_tools()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic update task: {str(e)}")

# Initialize MCP and server
load_dotenv()
server_initialized = initialize_server()
mcp = FastMCP(
    name="azure-agent"
)

if __name__ == "__main__":
    status = "successfully initialized" if server_initialized else "initialization failed"
    logger.info(f"{'='*50}")
    logger.info(f"Azure AI Agent MCP Server {status}")
    
    if not server_initialized:
        logger.error("Server initialization failed. Exiting...")
        sys.exit(1)
        
    logger.info(f"Starting server...")
    
    # Create and run the async loop
    async def main():
        # Register agents
        await register_agents()
        
        # Start the periodic update task
        update_task = asyncio.create_task(periodic_update_task())
        logger.info(f"MCP server is running with periodic updates every {update_interval} seconds...")
        
        try:
            # Run the MCP server
            if server_type == "web":
                await mcp.run_async(transport="streamable-http", host="127.0.0.1", port=server_port, path=server_path)
            else:
                await mcp.run_async()
        finally:
            # Make sure to cancel the periodic task when the server shuts down
            update_task.cancel()
            try:
                await update_task
            except asyncio.CancelledError:
                pass
            
    # Run the main async function
    asyncio.run(main())
