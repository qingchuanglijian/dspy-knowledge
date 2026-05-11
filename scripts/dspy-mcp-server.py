#!/usr/bin/env python3
"""
DSPy Knowledge MCP Server
Exposes DSPy knowledge base via Model Context Protocol (MCP).
Compatible with Claude Desktop, Cursor, and any MCP client.

Usage:
  python dspy-mcp-server.py

Environment:
  DSPY_KB_PATH — Path to knowledge base (default: ~/Wiki/pages/dspy-framework)
"""

import json
import sys
import os
from pathlib import Path

KB_PATH = Path(os.environ.get("DSPY_KB_PATH", 
    Path.home() / "Wiki/pages/dspy-framework"))

def read_doc(path):
    full = KB_PATH / path
    if full.exists():
        return full.read_text(encoding="utf-8")
    return f"Document not found: {path}"

def search_index(keyword):
    index_file = KB_PATH / "source-index/index.json"
    if not index_file.exists():
        return {"error": "Source index not found"}
    with open(index_file) as f:
        index = json.load(f)
    
    results = []
    keyword_lower = keyword.lower()
    for module in index.get("modules", []):
        for cls in module.get("classes", []):
            if keyword_lower in cls["name"].lower():
                results.append({
                    "type": "class",
                    "name": f"{module['module']}.{cls['name']}",
                    "file": cls.get("file", module["filepath"]),
                    "doc": cls.get("docstring", "")[:200]
                })
    return {"results": results[:10]}

# MCP Protocol handlers
def handle_list_tools():
    return {
        "tools": [
            {
                "name": "get_dspy_concept",
                "description": "Get documentation for a DSPy core concept (Signatures, Modules, LM, Optimizers, Evaluators)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string", "enum": ["signatures", "modules", "lm", "optimizers", "evaluators", "pitfalls", "migration"]},
                    },
                    "required": ["concept"]
                }
            },
            {
                "name": "get_dspy_pattern",
                "description": "Get a DSPy design pattern (rag, react-agent, multi-hop)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "enum": ["rag", "react-agent", "multi-hop"]},
                    },
                    "required": ["pattern"]
                }
            },
            {
                "name": "query_dspy_source",
                "description": "Search DSPy source code for classes or functions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string"},
                    },
                    "required": ["keyword"]
                }
            },
            {
                "name": "get_api_cheatsheet",
                "description": "Get DSPy 3.x API quick reference",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]
    }

def handle_call_tool(name, arguments):
    if name == "get_dspy_concept":
        concept = arguments.get("concept", "signatures")
        mapping = {
            "signatures": "core-concepts.md",
            "modules": "core-concepts.md",
            "lm": "core-concepts.md",
            "optimizers": "core-concepts.md",
            "evaluators": "core-concepts.md",
            "pitfalls": "common-pitfalls.md",
            "migration": "migration-2x-to-3x.md",
        }
        return {"content": [{"type": "text", "text": read_doc(mapping.get(concept, "README.md"))}]}
    
    elif name == "get_dspy_pattern":
        pattern = arguments.get("pattern", "rag")
        return {"content": [{"type": "text", "text": read_doc(f"patterns/{pattern}.md")}]}
    
    elif name == "query_dspy_source":
        keyword = arguments.get("keyword", "")
        return {"content": [{"type": "text", "text": json.dumps(search_index(keyword), indent=2)}]}
    
    elif name == "get_api_cheatsheet":
        return {"content": [{"type": "text", "text": read_doc("api-cheatsheet-3x.md")}]}
    
    return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}]}

def main():
    for line in sys.stdin:
        try:
            req = json.loads(line)
            method = req.get("method")
            
            if method == "initialize":
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "dspy-knowledge", "version": "1.0.0"}
                    }
                }), flush=True)
            
            elif method == "tools/list":
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "result": handle_list_tools()
                }), flush=True)
            
            elif method == "tools/call":
                result = handle_call_tool(req["params"]["name"], req["params"].get("arguments", {}))
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "id": req["id"],
                    "result": result
                }), flush=True)
        
        except Exception as e:
            print(json.dumps({
                "jsonrpc": "2.0",
                "id": req.get("id"),
                "error": {"code": -32603, "message": str(e)}
            }), flush=True)

if __name__ == "__main__":
    main()
