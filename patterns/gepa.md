---
pattern_id: P16-GEPA
difficulty: advanced
source_tutorial: Reflective Prompt Evolution / Compilation from dspy.ai
api_modules: [COPRO]
---

## 1. 核心思想 (Core Concept)

GEPA (Gradient-Evolutionary Prompt Adaptation) is the pattern of iteratively evolving prompts through reflection and mutation. Unlike MIPROv2's global Bayesian search, GEPA-style evolution starts from an initial prompt, evaluates it, reflects on failures, and mutates the prompt based on that reflection. It's particularly effective in rapidly evolving domains where manual prompt tuning can't keep up. The core philosophy is directional improvement: every mutation is guided by explicit failure analysis rather than random perturbation.

## 2. 类图与数据流 (Architecture)

```
Initial Prompt + Program
    ↓
Evaluate on validation set → Collect failures
    ↓
Reflect: why did it fail?
    ↓
Mutate prompt based on reflection
    ↓
Repeat until convergence or budget exhausted
    ↓
Optimized Prompt + Program
```

Key transformation: Prompt + failures → LLM reflection → mutated prompt → next generation. Evolution is local and directed by error signal.

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
    dspy.Example(text="GPU speeds up training", label="tech").with_inputs("text"),
    dspy.Example(text="Striker scores goal", label="sports").with_inputs("text"),
    dspy.Example(text="New bill passed in congress", label="politics").with_inputs("text"),
]
val = train

def metric(gold, pred, trace=None):
    return gold.label == pred.label

# COPRO implements the GEPA reflective prompt evolution pattern
optimizer = dspy.COPRO(metric=metric, breadth=2, depth=2)
compiled = optimizer.compile(program, trainset=train, valset=val)

result = compiled(text="Senator proposes new bill")
print(result.label)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Mutating prompts randomly without reflection signal

**Wrong:**
```python
# Custom loop that shuffles words without analyzing failures
def mutate(prompt):
    words = prompt.split()
    random.shuffle(words)
    return " ".join(words)
```

**Correct:**
```python
# Use reflection on failures to drive mutation (COPRO does this internally)
# Or manually: analyze errors → identify ambiguity → rewrite instruction
```

> **Debugging tip:** If evolved prompts don't improve, check whether mutation is tied to failure analysis. Random mutation wastes budget; reflection ensures each generation fixes real errors.

### Anti-pattern 2: Not maintaining a population of prompts

**Wrong:**
```python
# Single-line evolution: one prompt, mutate, replace
current = initial_prompt
for _ in range(10):
    current = mutate(current)  # No diversity, easy to get stuck
```

**Correct:**
```python
# Keep a population and select top performers (COPRO maintains candidates)
# Or maintain a list: population = [prompt_a, prompt_b, ...]
# scored by validation metric
```

> **Debugging tip:** If evolution plateaus early, you likely have a population of one. Maintain 3-5 diverse prompt variants and only propagate high-scoring mutations.

### Anti-pattern 3: Evolving prompts but ignoring demonstrations

**Wrong:**
```python
# Optimizer only mutates the Signature docstring
# while few-shot examples remain random or stale
```

**Correct:**
```python
# Evolve instructions and demos together (COPRO optimizes both)
# Or alternate: evolve prompt → bootstrap new demos → evolve prompt again
```

> **Debugging tip:** Instructions and demonstrations co-evolve for best results. If the prompt improves but accuracy stalls, the demo set may be outdated or biased. Refresh demonstrations after major prompt mutations.

## 5. 组合指南 (Composition)

GEPA-style evolution pairs well with bootstrapping and modular architectures to create adaptive systems.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| GEPA + BootstrapFewShot | Evolve prompts while simultaneously bootstrapping high-quality demonstrations from successful traces |
| GEPA + Custom Module | Evolve each sub-module's prompt independently, creating specialized prompts for each pipeline stage |
| GEPA + Evaluation | Track evolution trajectory across generations to detect overfitting or convergence |

**Example: evolving a multi-stage pipeline**
```python
class Extract(dspy.Signature):
    """Extract key entities from the text."""
    text: str = dspy.InputField()
    entities: str = dspy.OutputField()

extractor = dspy.Predict(Extract)
compiled_extractor = optimizer.compile(extractor, trainset=extract_train, valset=extract_val)
```

## 6. 进阶变体 (Advanced Variants)

- **Population-based evolution:** Maintain a diverse pool of 5-10 prompts, apply cross-over between high-performing variants, and cull low-performing ones each generation.
- **LLM-as-critic for reflection:** Use a separate critic LM to analyze failures and propose mutations, decoupling reflection from the main program's LM.
- **Co-evolution of prompts and metrics:** Evolve the evaluation rubric alongside the prompt when the task definition itself is fuzzy or evolving.
- **Checkpointing and rollback:** Archive every generation. If a mutation degrades performance, rollback to the last known good prompt rather than continuing from a bad state.
