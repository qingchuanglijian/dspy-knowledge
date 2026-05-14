---
pattern_id: P11-MultiChainComparison
difficulty: advanced
source_tutorial: API Reference / MultiChainComparison module
api_modules: [MultiChainComparison]
---

## 1. 核心思想 (Core Concept)

MultiChainComparison runs multiple distinct reasoning chains or generation strategies on the same input, then compares their outputs to select or synthesize the best result. Unlike BestOfN — which samples the same chain N times — this pattern uses fundamentally different approaches (e.g., ChainOfThought vs. direct Predict vs. ReAct) and lets a comparator decide which reasoning path produced the superior output. It is a meta-reasoning pattern: the model not only solves the problem, but evaluates how it was solved.

## 2. 类图与数据流 (Architecture)

```
Input (shared kwargs)
    ↓
dspy.MultiChainComparison([chain_a, chain_b, chain_c])
    ↓
forward(**kwargs)
    ↓
Each chain runs independently → outputs collected
    ↓
Comparator evaluates all outputs against a metric
    ↓
Output (best prediction selected or merged)
```

Key transformation: Same input → diverse reasoning paths → structured comparison → single best result.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class Answer(dspy.Signature):
    """Answer the question accurately."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

cot = dspy.ChainOfThought(Answer)
direct = dspy.Predict(Answer)

compare = dspy.MultiChainComparison([cot, direct])
result = compare(question="If a train travels 60 km in 30 minutes, what is its average speed in km/h?")

print(result.answer)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Hard-coding comparison logic

**Wrong:**
```python
def pick_best(results):
    # Brittle, domain-specific heuristic
    return max(results, key=lambda r: len(r.answer))
```

**Correct:**
```python
# Use a configurable DSPy module or metric
class CompareAnswers(dspy.Signature):
    """Select the most accurate answer."""
    question: str = dspy.InputField()
    answer_a: str = dspy.InputField()
    answer_b: str = dspy.InputField()
    winner: str = dspy.OutputField(desc="Either 'A' or 'B'")

judge = dspy.Predict(CompareAnswers)
```

> **Debugging tip:** If you find yourself writing `if/else` logic to compare chain outputs, replace it with a Signature-based judge module. DSPy can compile the judge just like any other module.

### Anti-pattern 2: Comparing chains with different output schemas

**Wrong:**
```python
class ShortAnswer(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class LongAnswer(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    explanation: str = dspy.OutputField()

# Cannot compare fairly — schemas differ
compare = dspy.MultiChainComparison([short, long])
```

**Correct:**
```python
class Unified(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

cot = dspy.ChainOfThought(Unified)
direct = dspy.Predict(Unified)
compare = dspy.MultiChainComparison([cot, direct])
```

> **Debugging tip:** Normalize all chains to produce the same output fields before comparison. If one chain naturally produces extra metadata, strip it or wrap the chain in an adapter module.

### Anti-pattern 3: Not handling ties or ambiguous winners

**Wrong:**
```python
result = compare(question="...")
best = result.answer  # Silent random pick if chains tie
```

**Correct:**
```python
result = compare(question="...")
print(result.rationale)  # Inspect why the winner was chosen

# Or add a tie-breaker module
if hasattr(result, 'confidence') and result.confidence < 0.6:
    result = cot(question=question)  # Fall back to CoT
```

> **Debugging tip:** Always inspect the comparison rationale. If the judge frequently reports ties or near-ties, add a third chain or increase the judge's temperature to break symmetry.

### Anti-pattern 4: Running too many chains

**Wrong:**
```python
# 8 different chains — comparator context explodes
compare = dspy.MultiChainComparison([c1, c2, c3, c4, c5, c6, c7, c8])
```

**Correct:**
```python
# Start with 2-3 diverse approaches
compare = dspy.MultiChainComparison([cot, direct, react])
```

> **Debugging tip:** The comparison step receives all chain outputs in its context window. If the combined outputs exceed the model's context limit, the comparator degrades. Cap at 3-4 chains or summarize outputs before comparison.

## 5. 组合指南 (Composition)

MultiChainComparison is a meta-pattern that improves any downstream task by increasing the diversity of candidate solutions.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| MultiChainComparison + BestOfN | Use diverse chains, where each chain generates N samples, then compare the full pool |
| MultiChainComparison + Judge | Use an LLM-as-judge module as the comparator instead of the built-in metric |
| MultiChainComparison + Ensemble | Formal voting across chain outputs rather than single winner selection |
| MultiChainComparison inside Custom Module | Use as a sub-routine that selects the best generation strategy for a given input type |

**Example: diverse chains with judge**
```python
class Judge(dspy.Signature):
    """Pick the better answer."""
    question: str = dspy.InputField()
    answer_a: str = dspy.InputField()
    answer_b: str = dspy.InputField()
    winner: str = dspy.OutputField(desc="A or B")

compare = dspy.MultiChainComparison([cot, direct])
result = compare(question="What is 15 * 24?")
print(result.answer)
```

## 6. 进阶变体 (Advanced Variants)

- **Weighted voting by chain reliability:** Track historical accuracy per chain and weight its vote proportionally during comparison.
- **Learned combination function:** Train a small classifier to combine outputs from multiple chains rather than selecting one winner.
- **Progressive elimination tournament:** Run chains in pairs, eliminate the loser, and recurse until one champion remains — reduces comparison context load.
