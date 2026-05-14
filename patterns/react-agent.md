---
pattern_id: P03-ReAct
difficulty: intermediate
source_tutorial: Building AI Agents with DSPy
api_modules: [ReAct]
---

# DSPy Pattern: ReAct Agent

**Purpose:** Build an agent that interleaves reasoning (Re) and acting (Ac) using tools.  
**Complexity:** Intermediate  
**Dependencies:** Tool definitions (functions the agent can call)

---

## Architecture

```
Question → [Reason] → Action → [Tool] → Observation → [Reason] → ... → Answer
```

## Basic Implementation

```python
import dspy
from typing import Callable, List

# 1. Define tools as functions
def search(query: str) -> str:
    """Search the web for information."""
    # Implementation
    return "Search results..."

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# 2. Signature
class AgentTask(dspy.Signature):
    """Answer questions using available tools."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Final answer after using tools")

# 3. ReAct Module
react = dspy.ReAct(
    AgentTask,
    tools=[search, calculator],
    max_iters=5
)

# 4. Usage
lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

result = react(question="What is the population of Paris times 2?")
print(result.answer)
```

## Custom Tool Pattern

```python
class CustomTool:
    def __init__(self, name: str, func: Callable, desc: str):
        self.name = name
        self.func = func
        self.desc = desc
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def weather_api(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny, 25°C in {city}"

tools = [
    CustomTool("search", search, "Search the web"),
    CustomTool("calculator", calculator, "Evaluate math expressions"),
    CustomTool("weather", weather_api, "Get weather for a city"),
]

react = dspy.ReAct(AgentTask, tools=[t.func for t in tools])
```

## Multi-Step Agent with Custom Logic

```python
class AdvancedAgent(dspy.Module):
    def __init__(self, tools: List[Callable]):
        super().__init__()
        self.tools = {t.__name__: t for t in tools}
        self.plan = dspy.ChainOfThought(PlanSignature)
        self.execute = dspy.ReAct(ExecuteSignature, tools=tools)
    
    def forward(self, question: str):
        # First, plan the approach
        plan_result = self.plan(question=question)
        steps = plan_result.steps.split("\n")
        
        # Then execute with ReAct
        result = self.execute(question=question, plan=plan_result.steps)
        return result
```

## Evaluation

```python
def agent_correctness(example, pred) -> float:
    """Check if agent answer is correct."""
    # Exact match or LLM judge
    return 1.0 if example.answer in pred.answer else 0.0

evaluator = Evaluate(devset=devset, metric=agent_correctness, num_threads=4)
score = evaluator(agent)
```

## Pitfalls

1. **Tool description quality** — Tool docstrings become part of the prompt. Write clear descriptions.
2. **Max iterations** — Always set `max_iters` to prevent infinite loops.
3. **Tool failures** — Wrap tool calls in try/except and return error messages.
4. **State confusion** — Each `forward()` call is independent. Don't rely on external mutable state.

## 4. Common Anti-patterns and Diagnosis

Agents are powerful, but small implementation mistakes compound quickly across reasoning loops. Below are the three most common errors we see when agents generate ReAct code.

### 4.1 Tools with Poorly Designed Signatures
Agents rely on tool docstrings and type hints to decide *how* to call a tool. Vague descriptions or missing annotations cause incorrect invocations.

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

**Debug tip:** If the agent passes malformed arguments or ignores a tool, inspect the compiled prompt (`dspy.inspect_history()`) to see how the tool schema is rendered.

### 4.2 Not Limiting `max_iters`
An unbounded ReAct loop can burn tokens indefinitely on unanswerable questions. Always set an explicit ceiling.

**Wrong**
```python
react = dspy.ReAct(AgentTask, tools=[search])  # ❌ max_iters omitted or too high
```

**Correct**
```python
react = dspy.ReAct(AgentTask, tools=[search], max_iters=5)  # ✅ Explicit guard
```

**Debug tip:** Watch for repeated `[Action]` → `[Tool]` cycles with identical arguments; that signals the agent is stuck.

### 4.3 Tools Returning Unstructured Text
DSPy modules expect typed, predictable outputs. A tool that dumps raw HTML or prose breaks the contract model and degrades downstream reasoning.

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

## 5. Composition Guide

ReAct rarely lives in isolation. It is usually composed with other patterns to handle memory, routing, or multi-step workflows.

| Pattern | When to Compose | Mini Example |
|---|---|---|
| **ReAct + Memory** | Conversational agents that need prior turns | Pass `history` into the `AgentTask` signature and append each turn. |
| **ReAct + Router** | Delegate to specialized sub-agents | A router module picks the domain; ReAct handles tool use inside that domain. |
| **ReAct inside Custom Module** | ReAct is one step in a larger pipeline | Plan → ReAct → Verify; ReAct executes the planned step with tools. |

```python
# ReAct + Router sketch
class Router(dspy.Module):
    def __init__(self):
        self.route = dspy.Predict(RouteSignature)
        self.math_agent = dspy.ReAct(MathTask, tools=[calc], max_iters=3)
        self.web_agent = dspy.ReAct(WebTask, tools=[search], max_iters=5)

    def forward(self, question: str):
        r = self.route(question=question)
        if r.domain == "math":
            return self.math_agent(question=question)
        return self.web_agent(question=question)
```

Use ReAct alone for simple tool-calling tasks with a single tool. Compose it when the user journey involves multiple domains, conversation history, or verification steps.

## 6. Advanced Variants

- **Tool result caching:** Memoize expensive tool calls with `@functools.lru_cache` or a custom cache keyed by arguments to reduce latency and cost.
- **Parallel tool execution:** When the agent needs multiple independent tools, dispatch them concurrently (e.g., via `asyncio.gather`) and feed the combined observations back into the next reasoning step.
- **Structured tool output:** Have tools return `pydantic.BaseModel` instances, then serialize them to JSON strings. This preserves type safety and makes downstream parsing reliable.

---

**Next:** [`multi-hop.md`](./multi-hop.md) for multi-step reasoning without tools.
