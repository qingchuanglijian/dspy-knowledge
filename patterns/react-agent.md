---
pattern_id: P03-ReAct
difficulty: intermediate
source_tutorial: Building AI Agents with DSPy
api_modules: [ReAct]
---

## 1. 核心思想 (Core Concept)

ReAct (Reason + Act) is a paradigm where an LLM agent interleaves internal reasoning with external tool execution. Instead of generating an answer in one shot, the agent thinks about what it needs to know, invokes a tool to gather information, observes the result, and then reasons again. This loop continues until the agent produces a final answer. The design philosophy treats tools as sensory extensions: the model "acts" to reduce uncertainty and "reasons" to plan the next action. In DSPy, `dspy.ReAct` automates this loop by prompting the model to emit `[Action]` and `[Observation]` tags, handling the dispatch and feedback transparently.

## 2. 类图与数据流 (Architecture)

```
Question → [Reason] → Action → [Tool] → Observation → [Reason] → ... → Answer
```

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for '{query}'..."

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

class AgentTask(dspy.Signature):
    """Answer questions using available tools."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Final answer after using tools")

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

react = dspy.ReAct(AgentTask, tools=[search, calculator], max_iters=5)

result = react(question="What is 12 times 34?")
print(result.answer)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### 4.1 Tools with Poorly Designed Signatures
Agents rely on docstrings and type hints to decide how to call a tool.

**Wrong**
```python
def get_data(x):  # ❌ No type hints, vague name, no docstring
    ...
```

**Correct**
```python
def get_stock_price(ticker: str) -> str:
    """Return the current stock price for a given ticker symbol (e.g., 'AAPL')."""
    ...
```

**Debug tip:** Inspect the compiled prompt with `dspy.inspect_history()` to see how the tool schema is rendered.

### 4.2 Not Limiting `max_iters`
An unbounded loop can burn tokens indefinitely.

**Wrong**
```python
react = dspy.ReAct(AgentTask, tools=[search])  # ❌ max_iters omitted
```

**Correct**
```python
react = dspy.ReAct(AgentTask, tools=[search], max_iters=5)  # ✅ Explicit guard
```

**Debug tip:** Watch for repeated `[Action]` → `[Tool]` cycles with identical arguments; that signals the agent is stuck.

### 4.3 Tools Returning Unstructured Text
Raw HTML or prose breaks the contract model and degrades downstream reasoning.

**Wrong**
```python
def search(query: str) -> str:
    return "<html>...</html>"  # ❌ Unstructured raw text
```

**Correct**
```python
def search(query: str) -> str:
    results = fetch(query)
    return "\n".join(f"{r['title']}: {r['snippet']}" for r in results)  # ✅ Structured string
```

**Debug tip:** If observations exceed context limits or the agent starts hallucinating facts, check the tool return format first.

## 5. 组合指南 (Composition)

ReAct rarely lives in isolation. It is usually composed with other patterns to handle memory, routing, or multi-step workflows.

| Pattern | When to Compose | Mini Example |
|---|---|---|
| **ReAct + Memory** | Conversational agents that need prior turns | Pass `history` into the `AgentTask` signature and append each turn. |
| **ReAct + Router** | Delegate to specialized sub-agents | A router module picks the domain; ReAct handles tool use inside that domain. |
| **ReAct inside Custom Module** | ReAct is one step in a larger pipeline | Plan → ReAct → Verify; ReAct executes the planned step with tools. |

```python
class Router(dspy.Module):
    def __init__(self):
        self.route = dspy.Predict("question -> domain")
        self.math_agent = dspy.ReAct(AgentTask, tools=[calculator], max_iters=3)
        self.web_agent = dspy.ReAct(AgentTask, tools=[search], max_iters=5)

    def forward(self, question: str):
        r = self.route(question=question)
        if r.domain == "math":
            return self.math_agent(question=question)
        return self.web_agent(question=question)
```

## 6. 进阶变体 (Advanced Variants)

- **Tool result caching:** Memoize expensive tool calls with `@functools.lru_cache` or a custom cache keyed by arguments to reduce latency and cost.
- **Parallel tool execution:** When the agent needs multiple independent tools, dispatch them concurrently (e.g., via `asyncio.gather`) and feed the combined observations back into the next reasoning step.
- **Structured tool output:** Have tools return `pydantic.BaseModel` instances, then serialize them to JSON strings. This preserves type safety and makes downstream parsing reliable.
