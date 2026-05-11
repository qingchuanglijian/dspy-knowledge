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

---

**Next:** [`multi-hop.md`](./multi-hop.md) for multi-step reasoning without tools.
