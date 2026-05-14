---
pattern_id: P14-BootstrapFewShot
difficulty: intermediate
source_tutorial: Compilation / Optimizers from dspy.ai
api_modules: [BootstrapFewShot]
---

## 1. 核心思想 (Core Concept)

BootstrapFewShot is automatic few-shot demonstration selection. Given a metric and a small set of training examples, it runs the program, collects execution traces where the metric is satisfied, and injects those as few-shot demos into the prompt. It is the "hello world" of DSPy optimization — no manual prompt engineering, just better examples. The philosophy is to treat prompts as compiled artifacts: you write the task definition once, and the optimizer discovers the best demonstrations from data.

## 2. 类图与数据流 (Architecture)

```
Uncompiled Module (Signature + Module)
    ↓
Training Set (list of Examples)
    ↓
Metric (gold → pred → bool/float)
    ↓
BootstrapFewShot.telecompile(module, trainset, metric)
    ↓
Compiled Module (prompts augmented with demos)
    ↓
forward(**kwargs) → Better prediction
```

Key transformation: training examples + metric → filtered successful traces → injected demos → compiled module.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class QA(dspy.Signature):
    """Answer the question briefly."""
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="A short answer")

def metric(gold, pred, trace=None):
    return gold.answer.lower() in pred.answer.lower()

trainset = [
    dspy.Example(question="2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="Largest mammal?", answer="blue whale").with_inputs("question"),
]

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

module = dspy.Predict(QA)
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=2)
compiled = optimizer.compile(module, trainset=trainset)

result = compiled(question="What is 3+3?")
print(result.answer)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Too few training examples (< 10)

**Wrong:**
```python
trainset = [dspy.Example(question="2+2?", answer="4").with_inputs("question")]
optimizer = dspy.BootstrapFewShot(metric=metric)
```

**Correct:**
```python
trainset = load_examples(min_size=20)  # cover the input distribution
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
```

> **Debugging tip:** If compiled performance is no better than uncompiled, check `len(trainset)`. BootstrapFewShot needs diversity to select good demos. Aim for 20–200 examples.

### Anti-pattern 2: Noisy or binary metric that doesn't reflect true quality

**Wrong:**
```python
def bad_metric(gold, pred, trace=None):
    return len(pred.answer) > 0  # Always true, useless signal
```

**Correct:**
```python
def good_metric(gold, pred, trace=None):
    return gold.answer.lower() in pred.answer.lower()
```

> **Debugging tip:** If the metric always returns `True`, every trace is selected and demos become random. Validate your metric on a handful of examples before compiling.

### Anti-pattern 3: Not setting max_bootstrapped_demos

**Wrong:**
```python
optimizer = dspy.BootstrapFewShot(metric=metric)  # defaults may overflow context
```

**Correct:**
```python
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)
```

> **Debugging tip:** Too many demos overflow the context window; too few underfit. Start with 3–5 and tune against a dev set.

### Anti-pattern 4: Re-compiling on every run instead of saving compiled state

**Wrong:**
```python
for item in requests:
    compiled = optimizer.compile(module, trainset=trainset)  # wasteful
    result = compiled(**item)
```

**Correct:**
```python
compiled = optimizer.compile(module, trainset=trainset)
for item in requests:
    result = compiled(**item)
```

> **Debugging tip:** Compilation is expensive. Save the compiled module object (or its demos) and reuse it. In production, compile once at load time.

## 5. 组合指南 (Composition)

BootstrapFewShot wraps any module and improves it with data. Use it whenever you have a metric and at least a modest training set.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| BootstrapFewShot + Predict | Optimize a simple classifier or extractor with real labels |
| BootstrapFewShot + Custom Module | Optimize multi-stage pipelines end-to-end — the metric judges final output |
| BootstrapFewShot + Evaluation | Measure before/after accuracy on a hold-out dev set to verify gains |

**Example: optimizing a multi-stage pipeline**

```python
class Pipeline(dspy.Module):
    def __init__(self):
        self.extract = dspy.Predict(Extract)
        self.answer = dspy.Predict(Answer)

    def forward(self, context, question):
        extracted = self.extract(context=context)
        return self.answer(context=extracted.fact, question=question)

pipeline = Pipeline()
compiled = dspy.BootstrapFewShot(metric=metric).compile(pipeline, trainset=trainset)
```

## 6. 进阶变体 (Advanced Variants)

- **BootstrapFewShotWithRandomSearch:** Combines bootstrapping with random search over demo combinations to find the subset that maximizes validation score.
- **Custom metric functions:** Use a metric that inspects `trace` to reward intermediate reasoning steps, not just final output.
- **Incremental bootstrapping:** Start with a small trainset, compile, evaluate, then add hard negatives and re-compile.
- **Demo diversity enforcement:** Post-process selected demos to ensure they cover different input clusters, avoiding redundant examples.
