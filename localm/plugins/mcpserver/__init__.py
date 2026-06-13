"""
localm MCP server plugin - expose local models to any MCP client.

Run with ``localm mcp`` (stdio transport). MCP clients launch it on demand
via their standard server config; ``localm mcp --print-config`` prints the
JSON block to paste into a client's mcpServers section.
"""
