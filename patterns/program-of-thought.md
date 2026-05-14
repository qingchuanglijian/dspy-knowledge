---
pattern_id: P06-ProgramOfThought
difficulty: intermediate
source_tutorial: Program Of Thought from dspy.ai
api_modules: [ProgramOfThought]
---

## 1. 核心思想 (Core Concept)

Instead of reasoning in natural language, the LM generates executable Python code to solve the problem. DSPy executes the code safely and returns the result. This is critical for math, data manipulation, and structured computation tasks where natural-language reasoning is brittle and imprecise. DSPy designs it this way to separate *what* to compute (the prompt) from *how* to compute (the generated code), giving the LM a deterministic, verifiable execution environment rather than relying on stochastic text generation for exact answers.

## 2. 类图与数据流 (Architecture)

```
Input (question/str)
    ↓
┌─────────────────────┐
│   MathSignature     │
│  question → answer  │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ ProgramOfThought    │
│  - generates code   │
│  - executes safely  │
└─────────────────────┘
    ↓
Generated Python code
    ↓
[Code Execution Sandbox]
    ↓
Output (answer/float)
```

Key transformations: natural-language question → Python code → executed result → typed output field.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class MathSolver(dspy.Signature):
    """Solve the math problem by generating and executing Python code."""
    question: str = dspy.InputField(desc="The math problem to solve")
    answer: float = dspy.OutputField(desc="The final numerical answer")

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

solver = dspy.ProgramOfThought(MathSolver)

result = solver(question="What is the sum of squares of the first 10 natural numbers?")
print(result.answer)
# Expected: 385.0
```

## 4. 常见反模式与诊断 (Anti-patterns)

### 1. Using PoT for tasks that don't need computation
**Wrong:**
```python
# PoT adds execution overhead for pure semantic tasks
sentiment = dspy.ProgramOfThought(SentimentSignature)
```
**Correct:**
```python
# Use Predict or ChainOfThought for semantic tasks
sentiment = dspy.ChainOfThought(SentimentSignature)
```
**Debugging tip:** If the generated code is just `print(...)` or trivial string manipulation, you chose the wrong module.

### 2. Not handling code execution failures
**Wrong:**
```python
result = pot(question="Solve this complex integral")
answer = result.answer  # Crashes if generated code throws
```
**Correct:**
```python
class SafeSolver(dspy.Module):
    def __init__(self):
        super().__init__()
        self.pot = dspy.ProgramOfThought(MathSolver)
        self.fallback = dspy.ChainOfThought(MathSolver)

    def forward(self, question: str):
        try:
            return self.pot(question=question)
        except Exception:
            return self.fallback(question=question)
```
**Debugging tip:** Wrap PoT calls in a try/except block inside a custom module, or use `dspy.inspect_history()` to inspect the generated code that failed.

### 3. Not constraining the Signature output type
**Wrong:**
```python
class MathSolver(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()  # Loose type
```
**Correct:**
```python
class MathSolver(dspy.Signature):
    question: str = dspy.InputField()
    answer: float = dspy.OutputField(desc="Final numerical answer")
```
**Debugging tip:** If execution succeeds but downstream code gets a TypeError, check that the OutputField type matches the computation result (e.g., `float`, `int`, `list[float]`).

## 5. 组合指南 (Composition)

- **PoT + ReAct:** The agent reasons about which tool to use, then generates code to combine tool outputs. Useful when the agent needs both external data and computation.
- **PoT + Refine:** Run PoT; if execution fails or the answer seems wrong, feed the error back into a Refine module to rewrite the code. Ideal for iterative debugging of generated programs.
- **PoT inside Custom Module:** Embed a PoT step as the computation node in a larger pipeline (e.g., Extract → Compute → Format). Keeps the pipeline modular and lets you swap PoT for CoT per-step.

## 6. 进阶变体 (Advanced Variants)

- **Custom code execution sandbox:** Override the default Deno-based sandbox with a restricted Docker container or a local Python `exec` with timeout and whitelist restrictions for sensitive environments.
- **Timeout handling:** Wrap code execution with a `signal.alarm` or `concurrent.futures` timeout to prevent infinite loops from generated code.
- **Multi-step computation:** Chain multiple PoT modules where the output of one computation becomes the input context of the next, useful for multi-stage financial or engineering calculations.
