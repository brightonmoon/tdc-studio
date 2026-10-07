"""Registration hub for all TDC-Studio MCP tools."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from tdc_studio.mcp.tools.admet import register_admet_tools
from tdc_studio.mcp.tools.dossier import register_dossier_tools
from tdc_studio.mcp.tools.dti import register_dti_tools
from tdc_studio.mcp.tools.optimizer import register_optimizer_tools
from tdc_studio.mcp.tools.pbpk import register_pbpk_tools
from tdc_studio.mcp.tools.retro import register_retro_tools


def register_all_tools(server: MCPServer) -> None:
    """Register all 6 domains of SLM tools on the MCP server."""
    register_admet_tools(server)
    register_dti_tools(server)
    register_pbpk_tools(server)
    register_retro_tools(server)
    register_optimizer_tools(server)
    register_dossier_tools(server)
