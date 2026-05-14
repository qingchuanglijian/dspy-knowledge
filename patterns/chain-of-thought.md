---
pattern_id: P02-ChainOfThought
difficulty: foundation
source_tutorial: Math Reasoning from dspy.ai
api_modules: [ChainOfThought]
---

## 1. 核心思想 (Core Concept)

ChainOfThought forces the LM to generate explicit reasoning steps before producing the final answer. DSPy automatically injects a `rationale` field into the Signature — you don't define it. The `rationale` is exposed in the output, making the model's reasoning inspectable and debuggable. This pattern is essential whenever the task requires arithmetic, multi-step logic, or nuanced judgment that a single leap would miss.

## 2. 类图与数据流 (Architecture)

```
Input (typed fields)
    ↓
Signature (docstring + InputField/OutputField)
    ↓
dspy.ChainOfThought(Signature)
    ↓
forward(**kwargs)
    ↓
LM prompt with "Reasoning: Let's think step by step..." injected
    ↓
Output fields:
  - rationale (auto-generated reasoning)
  - <your OutputField(s)>
```

Key transformation: kwargs → prompt with rationale instruction → LM generates reasoning + final answer → parsed into typed fields including `rationale`.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class MathSolver(dspy.Signature):
    """Solve the math word problem step by step."""
    question: str = dspy.InputField(desc="The math word problem")
    answer: int = dspy.OutputField(desc="The final numerical answer")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

solver = dspy.ChainOfThought(MathSolver)
result = solver(question="Alice has 5 apples. She buys 3 more, then gives away 2. How many does she have?")

print("Rationale:", result.rationale)
print("Answer:", result.answer)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Using CoT for tasks where reasoning doesn't help

**Wrong:**
```python
class SimpleLabel(dspy.Signature):
    """Return the language of the text."""
    text: str = dspy.InputField()
    language: str = dspy.OutputField()

# CoT adds latency and cost for an obvious single-token answer
labeler = dspy.ChainOfThought(SimpleLabel)
```

**Correct:**
```python
# Use Predict for tasks where the answer is direct and obvious
labeler = dspy.Predict(SimpleLabel)
```

> **Debugging tip:** If the "rationale" is just a restatement of the input with no real intermediate steps, switch to `dspy.Predict`. CoT shines on problems that *require* steps.

### Anti-pattern 2: Not reading/consuming the `rationale` field

**Wrong:**
```python
result = solver(question="...")
print(result.answer)   # rationale is silently discarded
```

**Correct:**
```python
result = solver(question="...")
print("Reasoning:", result.rationale)
print("Answer:", result.answer)

# Or log it for downstream debugging/auditing
if result.answer != expected:
    print("MISMATCH — Rationale:", result.rationale)
```

> **Debugging tip:** Always inspect `rationale` during development. It reveals whether the model is reasoning correctly or hallucinating intermediate steps. In production, log rationale for audit trails.

### Anti-pattern 3: Making the Signature too complex

**Wrong:**
```python
class Overloaded(dspy.Signature):
    """Analyze the report."""
    report: str = dspy.InputField()
    market: str = dspy.InputField()
    competitors: str = dspy.InputField()
    summary: str = dspy.OutputField()
    risks: str = dspy.OutputField()
    opportunities: str = dspy.OutputField()
    score: float = dspy.OutputField()
```

**Correct:**
```python
class StepOne(dspy.Signature):
    """Summarize the report."""
    report: str = dspy.InputField()
    summary: str = dspy.OutputField()

class StepTwo(dspy.Signature):
    """Identify risks from the summary."""
    summary: str = dspy.InputField()
    risks: str = dspy.OutputField()
```

> **Debugging tip:** If rationale becomes noisy or contradicts the final output, split the Signature. Each CoT call should handle one logical reasoning chain.

## 5. 组合指南 (Composition)

ChainOfThought is most powerful when composed with other modules to build reasoning pipelines.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| CoT + Predict | Two-stage pipeline: CoT reasons through the problem, Predict formats the final decision — e.g., evaluate arguments then output a strict JSON verdict |
| CoT + ReAct | CoT handles internal reasoning, ReAct interleaves tool calls; use CoT for reasoning-heavy sub-steps inside an agent |
| CoT inside Custom Module | Each sub-step of a larger pipeline is a CoT call — e.g., extract entities (CoT), then extract relations (CoT), then summarize (Predict) |

**Example: reason then decide**
```python
class EvaluateArgs(dspy.Signature):
    """Evaluate the strength of each argument."""
    claim: str = dspy.InputField()
    argument_a: str = dspy.InputField()
    argument_b: str = dspy.InputField()
    stronger_argument: str = dspy.OutputField(desc="Either 'A' or 'B'")

evaluator = dspy.ChainOfThought(EvaluateArgs)
result = evaluator(
    claim="DSPy improves prompt engineering",
    argument_a="It automates prompt optimization",
    argument_b="It requires more setup code"
)
print(result.rationale)
print(result.stronger_argument)
```

## 6. 进阶变体 (Advanced Variants)

- **Self-consistency:** Run `ChainOfThought` multiple times with `temperature > 0`, then majority-vote on the `answer` field while inspecting rationales for consensus.
- **Verified CoT:** After generating `rationale`, use a second `Predict` or `ChainOfThought` module to verify the reasoning steps before accepting the final answer.
- **Tool-augmented CoT:** Inside a Custom Module, alternate `ChainOfThought` (reason about what to do) with tool calls (execute), then feed tool outputs back into the next CoT step.
