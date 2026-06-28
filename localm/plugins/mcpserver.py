import sys
import asyncio
import json
import urllib.request
import urllib.error

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    import mcp.types as types
except ImportError:
    print("MCP SDK not installed. Run 'uv pip install mcp'.", file=sys.stderr)
    sys.exit(1)

app = Server("localm")

def _get_base_url():
    from localm import instances
    from localm.config import home_dir
    existing = instances.find_attachable(home_dir(), None)
    if not existing:
        return None, None
    return instances.attach_url(existing), existing.get("token")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="list_models",
            description="List available local models in localm registry",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="generate_text",
            description="Generate text using a local model. Fails if no localm server is running.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model ID"},
                    "prompt": {"type": "string", "description": "The user prompt"}
                },
                "required": ["model", "prompt"]
            },
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if name == "list_models":
        from localm.model_manager import load_registry
        models = load_registry()
        model_names = list(models.keys())
        return [types.TextContent(type="text", text=f"Available models: {', '.join(model_names)}")]
        
    elif name == "generate_text":
        model_id = arguments.get("model")
        prompt = arguments.get("prompt")
        
        base_url, token = _get_base_url()
        if not base_url:
            return [types.TextContent(type="text", text="Error: No localm server is currently running. Please start one via 'localm serve' or 'localm gui'.")]
            
        # Call the localm v1 API
        req_data = json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024
        }).encode("utf-8")
        
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        req = urllib.request.Request(f"{base_url}v1/chat/completions", data=req_data, headers=headers)
        try:
            with urllib.request.urlopen(req) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                content = resp_data["choices"][0]["message"]["content"]
                return [types.TextContent(type="text", text=content)]
        except urllib.error.HTTPError as e:
            return [types.TextContent(type="text", text=f"API Error: {e.code} - {e.read().decode('utf-8')}")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error connecting to localm: {str(e)}")]
            
    else:
        raise ValueError(f"Unknown tool: {name}")

async def run():
    # Run the stdio server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

def main():
    # CLI entrypoint for 'localm mcp'
    asyncio.run(run())

if __name__ == "__main__":
    main()
