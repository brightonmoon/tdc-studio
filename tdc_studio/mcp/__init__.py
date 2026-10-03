"""TDC-Studio Model Context Protocol (MCP) Server Package.

Provides standardized MCP Tools, Resources, and Prompts enabling AI agents
and LLMs to collaboratively design, evaluate, and document novel therapeutics
using specialized domain models (SLMs).
"""

from tdc_studio.mcp.server import create_mcp_server

__all__ = ["create_mcp_server"]
