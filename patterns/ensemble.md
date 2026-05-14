---
pattern_id: P18-Ensemble
difficulty: intermediate
source_tutorial: Compilation / Optimizers from dspy.ai
api_modules: [Ensemble]
---

## 1. 核心思想 (Core Concept)

Ensemble combines multiple DSPy programs and aggregates their outputs for improved robustness. Like a Random Forest for LLMs, diversity in error modes leads to better aggregate performance. The simplest meta-optimization is to run several programs on the same input and vote on the answer. Because different modules may fail on different examples, the ensemble smooths out individual weaknesses without changing the underlying prompts.

## 2. 类图与数据流 (Architecture)

```
Input (typed fields)
    ↓
┌─────────────────────────────────────┐
│  Program A → Prediction             │
│  Program B → Prediction             │
│  Program C → Prediction             │
└─────────────────────────────────────┘
    ↓
Aggregation (vote / average / weighted)
    ↓
Final Output (Prediction)
```

Key transformation: single input → parallel execution across diverse members → aggregated output.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class QA(dspy.Signature):
    """Answer the question briefly."""
    question: str = dspy.InputField(desc="The question")
    answer: str = dspy.OutputField(desc="Short answer")

class Ensemble(dspy.Module):
    def __init__(self, members):
        self.members = members

    def forward(self, **kwargs):
        preds = [m(**kwargs) for m in self.members]
        answers = [p.answer for p in preds]
        vote = max(set(answers), key=answers.count)
        return dspy.Prediction(answer=vote, all_answers=answers)

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

members = [dspy.Predict(QA), dspy.ChainOfThought(QA), dspy.Predict(QA)]
ensemble = Ensemble(members)
result = ensemble(question="What is the capital of Germany?")
print(result.answer)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Ensembling identical programs

**Wrong:**
```python
members = [dspy.Predict(QA), dspy.Predict(QA), dspy.Predict(QA)]
```

**Correct:**
```python
members = [dspy.Predict(QA), dspy.ChainOfThought(QA), dspy.Predict(ConciseQA)]
```

> **Debugging tip:** If all members share the same architecture and temperature=0, they produce identical outputs and the ensemble adds only latency. Introduce diversity: different Signatures, modules, or prompt variations.

### Anti-pattern 2: Not weighting by program reliability

**Wrong:**
```python
vote = max(set(answers), key=answers.count)  # every member counts equally
```

**Correct:**
```python
weights = {"A": 0.5, "B": 0.3, "C": 0.2}
# or compute weights from validation accuracy
```

> **Debugging tip:** If one member is systematically worse, it can flip close votes. Track per-member accuracy on a dev set and weight or prune low performers.

### Anti-pattern 3: Ensemble size too large without pruning

**Wrong:**
```python
members = [build_member(seed) for seed in range(50)]  # 50x latency
```

**Correct:**
```python
members = select_top_k(candidates, k=5, by=dev_accuracy)
```

> **Debugging tip:** After 3–5 diverse members, returns diminish. Profile latency versus accuracy gains and prune members that rarely change the outcome.

### Anti-pattern 4: Using simple majority vote for structured outputs

**Wrong:**
```python
# answers are JSON strings; exact-match vote fails on equivalent variants
vote = max(set(answers), key=answers.count)
```

**Correct:**
```python
import json

def aggregate_json(preds):
    parsed = [json.loads(p.answer) for p in preds]
    # domain-aware merging, e.g., field-level majority
    return merge_dicts(parsed)
```

> **Debugging tip:** For structured outputs, parse before voting. Use schema-aware aggregation (e.g., field-level majority) instead of string equality.

## 5. 组合指南 (Composition)

Ensemble is a meta-pattern that sits on top of other modules. Use it when a single program is brittle and you can afford multiple inference calls.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| Ensemble + BestOfN | Generate N candidates from different programs, then score and pick the best |
| Ensemble + MultiChainComparison | Treat diverse chains as ensemble members; aggregate after comparison |
| Ensemble + BootstrapFewShot | Optimize each member independently with its own demos, then ensemble the compiled versions |

**Example: diverse members with independent optimization**

```python
member1 = dspy.BootstrapFewShot(metric=metric).compile(dspy.Predict(QA), trainset)
member2 = dspy.BootstrapFewShot(metric=metric).compile(dspy.ChainOfThought(QA), trainset)
ensemble = Ensemble([member1, member2])
result = ensemble(question="What is 12 * 13?")
print(result.answer)
```

## 6. 进阶变体 (Advanced Variants)

- **Weighted ensemble by validation accuracy:** Learn per-member weights by minimizing validation error; use soft voting instead of hard majority.
- **Stacking:** Train a lightweight meta-classifier (or another DSPy module) that takes member outputs as features and predicts the final answer.
- **Dynamic ensemble:** Select a subset of members per input based on an estimated difficulty score — fast and cheap for easy queries, deep ensemble for hard ones.
- **Cascade ensemble:** Run a cheap member first; if confidence is low, progressively invoke more expensive members until threshold is met.
