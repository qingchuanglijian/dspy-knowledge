---
pattern_id: P09-Refine
difficulty: intermediate
source_tutorial: Output Refinement from dspy.ai
api_modules: [Refine]
---

## 1. 核心思想 (Core Concept)

Refine iteratively improves an output by feeding it back into the model with feedback. Like human editing: draft → critique → revise → repeat until good enough. DSPy automates this loop — the module generates an initial prediction, the metric checks it, and if the score is insufficient, the prediction is passed back as context for another attempt. The key insight is that models are better at fixing concrete text than generating perfect text in one shot.

## 2. 类图与数据流 (Architecture)

```
Input (typed fields)
    ↓
Draft Module (dspy.Predict or ChainOfThought)
    ↓
metric(example, pred, trace) → score
    ↓
if score is insufficient:
    pred → fed back into module as additional context
    ↓
    revised Prediction
    ↓
    (repeat up to N)
    ↓
Output (best-scoring Prediction across iterations)
```

Key transformation: kwargs → initial draft → metric evaluation → if failing, draft becomes part of next input → iterative revision → returns best attempt.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class DraftTweet(dspy.Signature):
    """Write a tweet about the topic under 100 characters."""
    topic: str = dspy.InputField(desc="The tweet topic")
    tweet: str = dspy.OutputField(desc="Short tweet text")

def under_limit(example, pred, trace=None) -> float:
    return 1.0 if len(pred.tweet or "") <= 100 else 0.0

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

refiner = dspy.Refine(module=dspy.Predict(DraftTweet), metric=under_limit, N=3)
result = refiner(topic="DSPy framework")

print(result.tweet)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: No termination condition

**Wrong:**
```python
# Missing N or a metric that never returns a passing score
refiner = dspy.Refine(
    module=dspy.Predict(DraftTweet),
    metric=lambda e, p, t: 0.0,
)
# Infinite loop of diminishing improvements
```

**Correct:**
```python
refiner = dspy.Refine(
    module=dspy.Predict(DraftTweet),
    metric=under_limit,
    N=3,
)
```

> **Debugging tip:** Always set `N`. Even with a good metric, token cost grows linearly with iterations. Cap the loop. If the metric never passes, inspect the draft quality at each iteration to see whether the model is actually improving or just rewriting.

### Anti-pattern 2: Refining already-good outputs

**Wrong:**
```python
def harsh_metric(example, pred, trace=None) -> float:
    # Demands perfection; forces refinement even when output is good
    return 1.0 if len(pred.tweet or "") == 42 else 0.0

refiner = dspy.Refine(
    module=dspy.Predict(DraftTweet),
    metric=harsh_metric,
    N=5,
)
```

**Correct:**
```python
def good_enough(example, pred, trace=None) -> float:
    return 1.0 if len(pred.tweet or "") <= 100 else 0.0

refiner = dspy.Refine(
    module=dspy.Predict(DraftTweet),
    metric=good_enough,
    N=2,
)
```

> **Debugging tip:** A good metric should accept "good enough" outputs. Overly strict metrics force unnecessary revisions that introduce new errors. If the final refined output is worse than the draft, your metric is too strict.

### Anti-pattern 3: Not passing the original input/context to the refinement step

**Wrong:**
```python
class RefineOnly(dspy.Signature):
    """Improve the draft."""
    draft: str = dspy.InputField()
    tweet: str = dspy.OutputField()
# The refiner never sees the original topic, so it drifts from intent
```

**Correct:**
```python
class DraftTweet(dspy.Signature):
    """Write a tweet about the topic under 100 characters."""
    topic: str = dspy.InputField()
    tweet: str = dspy.OutputField()
# Refine uses the same Signature; topic is preserved across iterations
```

> **Debugging tip:** If the refined output drifts away from the original request (e.g., changes topic, drops constraints), verify that the original input fields are still present in the Signature. Refine must re-invoke the module with the full original context.

### Anti-pattern 4: Using the same Signature for draft and refine

**Wrong:**
```python
class Generic(dspy.Signature):
    """Write and improve text."""
    topic: str = dspy.InputField()
    tweet: str = dspy.OutputField()

refiner = dspy.Refine(module=dspy.Predict(Generic), metric=metric, N=3)
# One Signature handles both creation and revision — muddled instructions
```

**Correct:**
```python
class DraftSig(dspy.Signature):
    """Write a first draft tweet. Focus on coverage."""
    topic: str = dspy.InputField()
    draft: str = dspy.OutputField()

class RefineSig(dspy.Signature):
    """Revise the draft to be shorter and punchier."""
    topic: str = dspy.InputField()
    draft: str = dspy.InputField()
    tweet: str = dspy.OutputField()

class TwoStageRefiner(dspy.Module):
    def __init__(self):
        self.draft = dspy.Predict(DraftSig)
        self.refine = dspy.Predict(RefineSig)

    def forward(self, topic):
        pred = self.draft(topic=topic)
        for _ in range(2):
            pred = self.refine(topic=topic, draft=pred.draft)
        return pred

refiner = TwoStageRefiner()
```

> **Debugging tip:** Separate `DraftSignature` and `RefineSignature`. The draft should focus on coverage; the refine should focus on quality and constraints. If you see the model deleting good content instead of polishing, split the Signatures and orchestrate them inside a custom Module.

## 5. 组合指南 (Composition)

Refine is a quality gate. Wrap it around any module whose outputs benefit from iterative polishing.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| Refine + BestOfN | Run `BestOfN` to pick the strongest candidate, then feed the winner into `Refine` for final polishing |
| Refine + RAG | Generate an initial answer from retrieved context, then use `Refine` to iteratively improve the answer as additional context is retrieved |
| Refine inside Custom Module | Use `Refine` as a sub-module in a larger pipeline — e.g., draft a report section, refine it, then pass it to the next stage |

**Example: refine after retrieval**
```python
class Answer(dspy.Signature):
    """Answer the question using the context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

def completeness(example, pred, trace=None) -> float:
    return 1.0 if len(pred.answer or "") > 50 else 0.0

refiner = dspy.Refine(
    module=dspy.Predict(Answer),
    metric=completeness,
    N=2,
)
result = refiner(context="DSPy is a framework...", question="What is DSPy?")
print(result.answer)
```

## 6. 进阶变体 (Advanced Variants)

- **Learned stopping classifier:** Train a small classifier to predict whether another refinement iteration will help, stopping early when marginal gain is near zero.
- **Critique-then-revise two-step module:** Use one Signature to generate a critique of the draft, then a second Signature to revise based on that critique (more structured than single-module Refine).
- **Human-in-the-loop refinement:** Replace the metric with a human approval signal. Pause the loop, show the draft to a user, and continue only if they request changes.
