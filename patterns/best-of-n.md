---
pattern_id: P08-BestOfN
difficulty: intermediate
source_tutorial: Output Refinement from dspy.ai
api_modules: [BestOfN]
---

## 1. 核心思想 (Core Concept)

BestOfN generates N candidate outputs from the same input, then selects the best one using a scoring metric or judge. It is a brute-force quality amplifier — it trades compute for reliability. Instead of trusting a single LM generation, you exploit the fact that models often produce good answers sporadically, and a principled selector can cherry-pick the winner.

## 2. 类图与数据流 (Architecture)

```
Input (typed fields)
    ↓
Signature (docstring + InputField/OutputField)
    ↓
dspy.BestOfN(module=dspy.Predict(Signature), metric=metric, N=3)
    ↓
forward(**kwargs) → runs module N times internally
    ↓
metric scores each candidate → highest score wins
    ↓
Output (best Prediction)
```

Key transformation: kwargs → N parallel/sequential prompts → N predictions → metric evaluates each → returns the prediction with the highest score.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class Summarize(dspy.Signature):
    """Summarize the article in one sentence."""
    article: str = dspy.InputField(desc="The article text")
    summary: str = dspy.OutputField(desc="A concise one-sentence summary")

def score(example, pred, trace=None) -> float:
    return 1.0 if len(pred.summary or "") > 20 else 0.0

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

best_of_n = dspy.BestOfN(module=dspy.Predict(Summarize), metric=score, N=3)
result = best_of_n(
    article="DSPy is a framework for programming language models. "
            "It unifies prompting and LM usage into typed Signatures."
)

print(result.summary)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: N is too large without a defined "best" metric

**Wrong:**
```python
# No metric — BestOfN has no principled way to choose
best_of_n = dspy.BestOfN(module=dspy.Predict(Summarize), N=10)
```

**Correct:**
```python
def relevance_metric(example, pred, trace=None) -> float:
    return float(pred.summary and "framework" in pred.summary.lower())

best_of_n = dspy.BestOfN(
    module=dspy.Predict(Summarize),
    metric=relevance_metric,
    N=3,
)
```

> **Debugging tip:** If your metric returns identical scores for every candidate, BestOfN degenerates to random selection. Verify metric variance by testing it on a few sample predictions first.

### Anti-pattern 2: Using BestOfN for trivial tasks where single generation is sufficient

**Wrong:**
```python
class LanguageDetect(dspy.Signature):
    """Detect the language of the text."""
    text: str = dspy.InputField()
    language: str = dspy.OutputField()

# Overkill: deterministic answer, no benefit from multiple samples
detector = dspy.BestOfN(
    module=dspy.Predict(LanguageDetect),
    metric=lambda e, p, t: 1.0,
    N=5,
)
```

**Correct:**
```python
detector = dspy.Predict(LanguageDetect)
```

> **Debugging tip:** BestOfN multiplies token cost by N. If your task is deterministic (classification, extraction, formatting), use `Predict` or `ChainOfThought`. Reserve BestOfN for open-ended generation where quality varies stochastically.

### Anti-pattern 3: Hard-coding the selection logic instead of making it a configurable metric/module

**Wrong:**
```python
candidates = [generator(input=text) for _ in range(3)]
# Brittle: selection logic lives outside DSPy, not reusable or optimizable
best = max(candidates, key=lambda p: len(p.output))
```

**Correct:**
```python
def quality_score(example, pred, trace=None) -> float:
    return float(len(pred.output or ""))

best_of_n = dspy.BestOfN(
    module=dspy.Predict(Generator),
    metric=quality_score,
    N=3,
)
result = best_of_n(input=text)
```

> **Debugging tip:** Treat the metric as a first-class DSPy citizen. Pass it to `BestOfN` so DSPy can later compile or optimize it. Inline selection logic breaks the declarative contract.

## 5. 组合指南 (Composition)

BestOfN is a quality wrapper. Use it around any module where generation variance is high and you have a way to score outputs.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| BestOfN + Judge | Use an LLM-as-judge module as the metric to score each candidate on coherence, helpfulness, or correctness |
| BestOfN + MultiChainComparison | Generate N candidates, then run a voting module to select the majority answer (great for factual QA) |
| BestOfN + Refine | Select the best of N candidates, then feed the winner into `Refine` for further polishing |

**Example: judge-assisted selection**
```python
class Judge(dspy.Signature):
    """Rate the summary quality from 1 to 10."""
    article: str = dspy.InputField()
    summary: str = dspy.InputField()
    score: float = dspy.OutputField(desc="Quality score 1-10")

def judge_metric(example, pred, trace=None) -> float:
    result = dspy.Predict(Judge)(
        article=example.article, summary=pred.summary
    )
    return result.score

best_of_n = dspy.BestOfN(
    module=dspy.Predict(Summarize),
    metric=judge_metric,
    N=3,
)
```

## 6. 进阶变体 (Advanced Variants)

- **Temperature sweep:** Vary `temperature` across the N candidates (e.g., 0.3, 0.7, 1.0) to trade off coherence and creativity without changing the prompt.
- **Diversity enforcement:** Use different system prompts or few-shot examples per candidate to ensure the N outputs explore distinct angles, not just paraphrases.
- **Early stopping:** Maintain a score threshold inside the metric and raise a custom signal so `BestOfN` stops generating once a candidate exceeds the threshold, saving tokens.
