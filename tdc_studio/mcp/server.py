"""TDC-Studio MCP Server Core Implementation."""

from __future__ import annotations

import logging

from mcp.server.mcpserver import MCPServer

from tdc_studio.mcp.prompts import register_prompts
from tdc_studio.mcp.resources import register_resources
from tdc_studio.mcp.tools import register_all_tools

logger = logging.getLogger("tdc_studio.mcp")

SERVER_INSTRUCTIONS = """You are connected to TDC-Studio: An AI-native Drug Discovery & Therapeutics MLOps Studio.
You have access to specialized machine learning models (SLMs) trained on Therapeutics Data Commons (TDC) benchmarks:
- 25-Task ADMET predictions (Absorption, Distribution, CYP450, Clearance, Toxicity)
- Explainable AI (XAI) toxicophore identification and bioisostere recommendations
- ChemBERTa + ESM-2 Drug-Target Interaction (DTI) binding affinity (Kd, Ki, IC50) and contact residue maps
- Repeat-dose PBPK simulation (QD/BID steady-state peak, trough, and hERG safety margins)
- Mechanism-based CYP drug-drug interaction (DDI) risk assessment (Midazolam, Warfarin fold changes)
- Retrosynthesis A* multi-step pathway planning to commercial catalog building blocks
- Closed-loop Self-Correcting Lead Optimization
- Automated ICH CTD Nonclinical IND Candidate Dossier compilation into standalone interactive HTML and JSON

Always apply sound medicinal chemistry principles when interpreting model predictions.
"""


def create_mcp_server() -> MCPServer:
    """Instantiate and configure the complete TDC-Studio MCP server."""
    server = MCPServer(
        name="tdc-studio",
        version="0.1.0",
        instructions=SERVER_INSTRUCTIONS,
    )

    # 1. Register domain tools
    register_all_tools(server)

    # 2. Register guidelines and reference resources
    register_resources(server)

    # 3. Register standard workflow prompts
    register_prompts(server)

    return server


def run_mcp_server(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the TDC-Studio MCP server with specified transport protocol.

    Args:
        transport: Communication transport protocol ('stdio' or 'sse').
        host: Host IP for SSE transport.
        port: Port number for SSE transport.
    """
    server = create_mcp_server()

    if transport == "stdio":
        logger.info("Starting TDC-Studio MCP server on stdio transport...")
        server.run(transport="stdio")
    elif transport == "sse":
        logger.info("Starting TDC-Studio MCP server on SSE transport (http://%s:%d)...", host, port)
        server.run(transport="sse", host=host, port=port)
    else:
        raise ValueError(f"Unsupported transport '{transport}'. Supported: 'stdio', 'sse'.")


if __name__ == "__main__":
    run_mcp_server(transport="stdio")
