---
pattern_id: P15-MIPROv2
difficulty: advanced
source_tutorial: Compilation / Optimizers from dspy.ai
api_modules: [MIPROv2]
---

## 1. 核心思想 (Core Concept)

MIPROv2 treats prompt engineering as hyperparameter search over the joint space of prompt instructions and few-shot demonstrations. It uses Bayesian optimization to propose candidate prompt+demo combinations, evaluates them on a validation set, and uses the results to guide subsequent proposals. Unlike simple few-shot bootstrapping, MIPROv2 explores globally and can discover non-obvious prompt formulations. It's DSPy's most powerful built-in optimizer for production quality when you have sufficient labeled data and compute budget.

## 2. 类图与数据流 (Architecture)

```
Program (module with signatures)
    ↓
TrainSet + ValSet + Metric
    ↓
MIPROv2.telecompile(program, trainset, valset, metric, num_trials=...)
    ↓
Optimized instructions + few-shot demos
    ↓
Compiled Program (forward with optimized prompts)
```

Key transformation: Python module + data + metric → Bayesian search → compiled module with tuned prompts and demonstrations.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class Classify(dspy.Signature):
    """Classify the topic of the article."""
    text: str = dspy.InputField()
    label: str = dspy.OutputField(desc="One of: tech, sports, politics")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

program = dspy.Predict(Classify)

train = [
    dspy.Example(text="New AI model released", label="tech").with_inputs("text"),
    dspy.Example(text="Team wins championship", label="sports").with_inputs("text"),
    dspy.Example(text="Election results announced", label="politics").with_inputs("text"),
]
val = train  # small demo; use distinct set in production

def metric(gold, pred, trace=None):
    return gold.label == pred.label

optimizer = dspy.MIPROv2(metric=metric, num_threads=1)
compiled = optimizer.compile(program, trainset=train, valset=val, num_trials=5)

result = compiled(text="Senator proposes new bill")
print(result.label)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Running with fewer than 50 examples

**Wrong:**
```python
# Only 3 examples — Bayesian optimization has no signal to converge
optimizer.compile(program, trainset=train[:3], valset=val[:3], num_trials=30)
```

**Correct:**
```python
# Provide at least 50-100 labeled examples for meaningful optimization
optimizer.compile(program, trainset=large_train, valset=large_val, num_trials=30)
```

> **Debugging tip:** If optimization finishes but validation score is flat, check `len(trainset)`. Few-shot optimizers need statistical signal; MIPROv2 is designed for medium-to-large datasets.

### Anti-pattern 2: Using a cheap metric that doesn't correlate with true quality

**Wrong:**
```python
def bad_metric(gold, pred, trace=None):
    # Rewards length, not correctness
    return len(pred.label) > 3
```

**Correct:**
```python
def good_metric(gold, pred, trace=None):
    return gold.label == pred.label
```

> **Debugging tip:** The optimizer maximizes exactly what the metric returns. If the metric is noisy or gameable, the resulting prompts will be too. Validate your metric on a held-out set before running MIPROv2.

### Anti-pattern 3: Not saving the optimized program

**Wrong:**
```python
compiled = optimizer.compile(program, trainset=train, valset=val)
# Re-compiling next run is non-deterministic and expensive
```

**Correct:**
```python
compiled = optimizer.compile(program, trainset=train, valset=val)
compiled.save("optimized_program.json")
# Later: loaded = program.load("optimized_program.json")
```

> **Debugging tip:** Compilation is stochastic and costly. Always `save()` the compiled module. To freeze behavior, also fix the random seed and LM temperature.

## 5. 组合指南 (Composition)

MIPROv2 composes with any DSPy module to lift its quality through automated prompt optimization.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| MIPROv2 + Predict | Optimize a simple classifier or extractor for production accuracy |
| MIPROv2 + ChainOfThought | Optimize reasoning tasks where rationale quality matters |
| MIPROv2 + Custom Module | Optimize multi-step pipelines; each submodule can be compiled independently |

**Example: optimizing a reasoning pipeline**
```python
class Reason(dspy.Signature):
    """Solve the math problem step by step."""
    question: str = dspy.InputField()
    answer: int = dspy.OutputField()

solver = dspy.ChainOfThought(Reason)
compiled_solver = optimizer.compile(solver, trainset=math_train, valset=math_val)
```

## 6. 进阶变体 (Advanced Variants)

- **Multi-objective optimization:** Extend the metric to return a tuple or dict balancing accuracy, latency, and cost, then use a multi-objective Bayesian optimizer backend.
- **Warm start with BootstrapFewShot:** Run `BootstrapFewShot` first to generate initial demonstrations, then feed its output into MIPROv2 as a starting point to reduce search space.
- **Hierarchical MIPRO:** Optimize sub-modules independently (e.g., retriever prompts vs. generator prompts) rather than the full program at once, reducing dimensionality.
- **Early stopping:** Monitor validation metric convergence and stop trials early when improvement drops below a threshold, saving compute budget.
