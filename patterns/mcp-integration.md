---
pattern_id: P19-MCPIntegration
difficulty: intermediate
source_tutorial: Tool Use / External Tool Integration from dspy.ai
api_modules: []
---

## 1. 核心思想 (Core Concept)

Integrate DSPy with the Model Context Protocol (MCP) to give LLM programs access to standardized external tools (search, databases, APIs, file systems). MCP acts as a universal adapter — DSPy generates the reasoning, MCP handles tool execution. The key insight: DSPy handles the "what" (which tool, which arguments), MCP handles the "how" (authentication, serialization, transport).

## 2. 类图与数据流 (Architecture)

```
User Input
    ↓
DSPy Module (reasoning + tool selection)
    ↓
MCP Client (protocol serialization)
    ↓
MCP Server (tool execution)
    ↓
Tool Result
    ↓
DSPy Module (reasoning with result)
    ↓
Final Output
```

Key transformation: kwargs → structured reasoning → tool selection → MCP transport → structured execution → synthesis → final answer.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class MockMCP:
    def call_tool(self, name: str, args: str) -> str:
        return "22°C, sunny" if "weather" in args else "N/A"

class PickTool(dspy.Signature):
    """Select a tool and its arguments for the query."""
    query: str = dspy.InputField()
    tool_name: str = dspy.OutputField(desc="e.g., weather_lookup")
    tool_args: str = dspy.OutputField(desc="JSON arguments")

class Synthesize(dspy.Signature):
    """Answer the query using the tool result."""
    query: str = dspy.InputField()
    tool_result: str = dspy.InputField()
    answer: str = dspy.OutputField()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
mcp = MockMCP()

picker = dspy.ChainOfThought(PickTool)
synth = dspy.Predict(Synthesize)

q = "Weather in Paris?"
p = picker(query=q)
tr = mcp.call_tool(p.tool_name, p.tool_args)
out = synth(query=q, tool_result=tr)

print(f"{p.tool_name}({p.tool_args}) -> {tr}")
print(out.answer)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Hard-coding tool schemas instead of using MCP discovery

**Wrong:**
```python
# Fragile — breaks when server updates schemas
TOOLS = {"search": {"args": {"q": "string"}}}
```

**Correct:**
```python
tools = mcp_client.list_tools()  # Dynamic discovery
```

> **Debugging tip:** Always call `list_tools()` at startup and cache schemas. Validate arguments against the discovered schema before sending.

### Anti-pattern 2: Not handling MCP server failures / timeouts

**Wrong:**
```python
result = mcp.call_tool("search", args)  # Blocks forever on failure
```

**Correct:**
```python
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor() as e:
    fut = e.submit(mcp.call_tool, "search", args)
    result = fut.result(timeout=10)
```

> **Debugging tip:** Wrap every MCP call in a timeout. If the tool fails, return a structured error to DSPy so the module can retry or fall back.

### Anti-pattern 3: Treating every MCP call as a single `forward()` step

**Wrong:**
```python
class AllAtOnce(dspy.Signature):
    query: str = dspy.InputField()
    answer: str = dspy.OutputField()  # Hides tool use
```

**Correct:**
```python
# Use ReAct: reason → act (tool) → observe → repeat
class ReActStep(dspy.Signature):
    query: str = dspy.InputField()
    observation: str = dspy.InputField()
    thought: str = dspy.OutputField()
    action: str = dspy.OutputField()
```

> **Debugging tip:** If a task needs more than one tool call, use a ReAct loop inside a Custom Module rather than a single Signature.

### Anti-pattern 4: Ignoring MCP's structured error responses

**Wrong:**
```python
try:
    result = mcp.call_tool(name, args)
except Exception:
    result = "Error"  # Loses context
```

**Correct:**
```python
try:
    result = mcp.call_tool(name, args)
except mcp.MCPError as e:
    result = f"Error {e.code}: {e.message}"  # Rich context preserved
```

> **Debugging tip:** Feed the full error code and message back into the next DSPy step. The LM can often suggest a corrected argument set.

### Anti-pattern 5: Not validating MCP tool outputs before feeding downstream

**Wrong:**
```python
tool_result = mcp.call_tool(name, args)
next_step(result=tool_result)  # May be malformed
```

**Correct:**
```python
tool_result = mcp.call_tool(name, args)
assert "expected_key" in tool_result, "Malformed tool output"
next_step(result=tool_result)
```

> **Debugging tip:** Add lightweight validation (key presence, type checks) between the MCP layer and the next DSPy module. Fail fast on unexpected shapes.

## 5. 组合指南 (Composition)

MCP integration is most powerful when combined with modules that orchestrate multi-step reasoning and routing.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| MCP + ReAct | Multi-step tool agent that reasons, calls tools, observes, and repeats |
| MCP + RAG | Use an MCP search tool as the retriever in a RAG pipeline |
| MCP + Router | Route queries to different MCP servers based on domain |
| MCP + Custom Module | Embed an MCP tool call as a sub-module inside a larger pipeline |

**Example: MCP + ReAct loop**
```python
class AgentStep(dspy.Signature):
    """Decide the next tool call or final answer."""
    query: str = dspy.InputField()
    history: str = dspy.InputField()
    thought: str = dspy.OutputField()
    action: str = dspy.OutputField(desc="Tool call or 'FINISH'")
    action_input: str = dspy.OutputField()

agent = dspy.ChainOfThought(AgentStep)
# Loop: agent → mcp.call_tool → agent → ... until action == "FINISH"
```

## 6. 进阶变体 (Advanced Variants)

- **Multi-server MCP:** Aggregate `list_tools()` from many MCP servers into a unified registry so one DSPy agent can access search, database, and file-system tools simultaneously.
- **Streaming MCP:** Stream partial tool results (e.g., search snippets) back to DSPy in real-time for incremental reasoning.
- **MCP caching layer:** Cache tool results by argument hash to avoid redundant API calls and reduce latency.
- **Auth/permissions per tool:** Gate MCP tool access based on user roles before invoking `call_tool()`.
- **MCP + sandbox:** Execute generated code inside a local sandbox exposed via MCP for safe interactive computing.
