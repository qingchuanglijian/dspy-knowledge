---
pattern_id: P10-Parallel
difficulty: intermediate
source_tutorial: API Reference / Parallel module
api_modules: [Parallel]
---

## 1. 核心思想 (Core Concept)

Parallel executes multiple independent predictions concurrently from a single input, collapsing sequential latency into one wall-clock wait. It is the DSPy primitive for horizontal scaling of reasoning: instead of calling modules A, B, and C in sequence, you declare them parallel and receive all results together. The design philosophy is latency reduction through independence — any time two or more predictions do not depend on each other, they should run in parallel.

## 2. 类图与数据流 (Architecture)

```
Input (shared kwargs)
    ↓
dspy.Parallel(module_a, module_b, module_c)
    ↓
forward(**kwargs)  →  [module_a(**kwargs), module_b(**kwargs), module_c(**kwargs)]
    ↓
Output (tuple of dspy.Prediction objects)
```

Key transformation: Single input broadcast → concurrent LM calls → aggregated tuple of typed predictions.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class Sentiment(dspy.Signature):
    """Classify the sentiment of the text."""
    text: str = dspy.InputField()
    sentiment: str = dspy.OutputField()

class Entities(dspy.Signature):
    """Extract named entities from the text."""
    text: str = dspy.InputField()
    entities: str = dspy.OutputField()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

parallel = dspy.Parallel(dspy.Predict(Sentiment), dspy.Predict(Entities))
sentiment_pred, entities_pred = parallel(text="Apple unveiled new AI chips.")

print(sentiment_pred.sentiment)
print(entities_pred.entities)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Running parallel when tasks have hidden dependencies

**Wrong:**
```python
step1 = dspy.Predict(Analyze)
step2 = dspy.Predict(Decide)  # Needs analysis result
parallel = dspy.Parallel(step1, step2)  # Race condition
```

**Correct:**
```python
analysis = dspy.Predict(Analyze)(text=text)
decision = dspy.Predict(Decide)(analysis=analysis.summary)
```

> **Debugging tip:** Draw a DAG of your pipeline. If any arrow points from one parallel branch to another, they are not independent. Use sequential composition instead.

### Anti-pattern 2: Not aggregating results properly after parallel execution

**Wrong:**
```python
results = parallel(text="...")
print(results[0])  # Only uses first result, ignores others
```

**Correct:**
```python
results = parallel(text="...")
for i, pred in enumerate(results):
    print(f"Branch {i}: {pred}")
```

> **Debugging tip:** Always unpack or iterate over the full tuple returned by `Parallel`. If you only need one branch, you don't need Parallel.

### Anti-pattern 3: Running parallel for trivially small tasks

**Wrong:**
```python
# Two single-token labels on a tiny input
parallel = dspy.Parallel(dspy.Predict(LabelA), dspy.Predict(LabelB))
```

**Correct:**
```python
# Combine into one Signature
class Combined(dspy.Signature):
    """Label both category and priority."""
    text: str = dspy.InputField()
    category: str = dspy.OutputField()
    priority: str = dspy.OutputField()
labeler = dspy.Predict(Combined)
```

> **Debugging tip:** If each parallel call costs less than ~50ms of LM time, the overhead of parallelization may exceed the savings. Batch or merge instead.

### Anti-pattern 4: Using parallel without a timeout

**Wrong:**
```python
parallel = dspy.Parallel(module_a, module_b)  # No timeout configured
results = parallel(text="...")
```

**Correct:**
```python
# Configure timeout at the LM or framework level
lm = dspy.LM("openai/gpt-4o-mini", max_tokens=500, timeout=30.0)
dspy.configure(lm=lm)
```

> **Debugging tip:** One slow LM call in a Parallel batch blocks the entire result tuple. Set aggressive timeouts and fall back to sequential execution if latencies diverge.

## 5. 组合指南 (Composition)

Parallel is most valuable when combined with modules that evaluate, judge, or select from the parallel outputs.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| Parallel + Judge | Run multiple aspect analyses in parallel, then score each aspect with a Judge module |
| Parallel + BestOfN | Generate N variants concurrently, then select the best via a metric |
| Parallel + MultiChainComparison | Run different reasoning chains in parallel, then compare and pick the winner |
| Parallel inside Custom Module | Use Parallel as a sub-module for fan-out operations in larger pipelines |

**Example: multi-aspect evaluation**
```python
class Tone(dspy.Signature):
    text: str = dspy.InputField()
    tone: str = dspy.OutputField()

class Grammar(dspy.Signature):
    text: str = dspy.InputField()
    grammar_score: int = dspy.OutputField()

parallel = dspy.Parallel(dspy.Predict(Tone), dspy.Predict(Grammar))
tone_pred, grammar_pred = parallel(text="Draft paragraph here")
print(tone_pred.tone, grammar_pred.grammar_score)
```

## 6. 进阶变体 (Advanced Variants)

- **Dynamic parallelism:** Decide which branches to run at runtime based on input characteristics, constructing `dspy.Parallel(*selected_modules)` inside `forward()`.
- **Result aggregation strategies:** After parallel execution, apply concatenate, majority vote, or merge functions on the output fields rather than treating each branch independently.
- **Async parallel with asyncio:** For I/O-bound pipelines, wrap `dspy.Parallel` calls in `asyncio.gather` or use an async-compatible LM backend to maximize throughput beyond thread-level parallelism.
