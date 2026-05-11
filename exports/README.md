# Portable Export Formats

**Purpose:** Make DSPy knowledge usable by ANY AI agent, not just Hermes.  
**Philosophy:** Knowledge should be **framework-agnostic**.

---

## Available Formats

| Format | File | Use Case | Target Agent |
|--------|------|----------|-------------|
| **MCP Protocol** | `mcp-dspy-knowledge.json` | Structured tool definitions | Claude Desktop, Cursor, any MCP client |
| **Standard Prompt** | `system-prompt-dspy.md` | Direct system prompt | Any LLM API (OpenAI, Anthropic, etc.) |
| **Cursor Rules** | `.cursorrules-dspy` | Cursor IDE context | Cursor |
| **Claude Project** | `CLAUDE.md-dspy` | Claude Code/Projects | Claude Code, Claude Projects |
| **JSON API** | `api-schema.json` | Programmatic access | Custom agents, scripts |

---

## How to Use Each Format

### 1. MCP Protocol (Most Powerful)

The MCP (Model Context Protocol) format turns the DSPy knowledge base into a **tool server** that any MCP-compatible client can query.

**For Claude Desktop:**
```json
// claude_desktop_config.json
{
  "mcpServers": {
    "dspy-knowledge": {
      "command": "python",
      "args": ["/Users/tokens/Wiki/pages/dspy-framework/scripts/dspy-mcp-server.py"]
    }
  }
}
```

**Capabilities exposed:**
- `search_dspy_docs(query)` — Search the knowledge base
- `get_pattern(name)` — Get a specific pattern (RAG, ReAct, etc.)
- `get_api_reference(component)` — Get API docs for any component
- `query_source_code(keyword)` — Query the DSPy source index

**For other agents:** Any agent supporting MCP (Cursor, future standards) can use the same server.

---

### 2. Standard Prompt (Simplest)

Copy-paste ready system prompt. Works with ANY LLM API.

```bash
# Include in your API call
curl https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "'$(cat system-prompt-dspy.md)'"},
      {"role": "user", "content": "Write a RAG pipeline with DSPy"}
    ]
  }'
```

**Content includes:**
- Core philosophy (Signatures > Prompts)
- 3.x API syntax
- Common pitfalls
- Pattern templates (RAG, ReAct, Multi-hop)

---

### 3. Cursor Rules

For Cursor IDE users. Place in project root or `~/.cursorrules`.

```bash
cp .cursorrules-dspy /path/to/your/project/.cursorrules
# Or append to existing
cat .cursorrules-dspy >> ~/.cursorrules
```

**Effect:** Cursor will use DSPy 3.x patterns when generating code, avoid deprecated APIs, and suggest correct imports.

---

### 4. Claude Project

For Claude Code or Claude Projects.

```bash
cp CLAUDE.md-dspy /path/to/your/project/CLAUDE.md
```

**Effect:** Claude Code will reference DSPy conventions when answering questions about your codebase.

---

### 5. JSON API

For programmatic access from custom agents.

```python
import json

with open("api-schema.json") as f:
    dspy_api = json.load(f)

# Access any component
signature_docs = dspy_api["components"]["Signature"]
print(signature_docs["syntax"])
print(signature_docs["examples"])
print(signature_docs["pitfalls"])
```

---

## Sync Strategy

When the knowledge base updates, regenerate all formats:

```bash
cd ~/Wiki/pages/dspy-framework
python scripts/export-all.py  # Regenerates all portable formats
```

**Auto-sync:** Set up a cronjob to run weekly:
```bash
hermes cron create --schedule "0 9 * * 1" \
  --script "~/Wiki/pages/dspy-framework/scripts/export-all.py" \
  --name "dspy-knowledge-sync"
```

---

## Contributing New Formats

To add a new export format:

1. Create a template in `exports/templates/`
2. Add a generator in `scripts/export-all.py`
3. Document it in this README

**Principle:** Every piece of knowledge should be expressible in at least 3 formats (structured, prompt, programmatic).

---

**Maintained by:** default profile  
**Update frequency:** Weekly sync with upstream DSPy
