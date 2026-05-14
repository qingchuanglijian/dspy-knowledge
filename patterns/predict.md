---
pattern_id: P01-Predict
difficulty: foundation
source_tutorial: Classification, Entity Extraction from dspy.ai
api_modules: [Predict]
---

## 1. 核心思想 (Core Concept)

Predict is the simplest input-to-output mapping in DSPy. A Signature wrapped in direct prediction. Agent often treats this as a "prompt template" but it's actually a typed contract — the docstring and type annotations together define what the LM must produce, not how. DSPy compiles this contract into prompts automatically. This separation of "what" from "how" is the core philosophy: you declare intent, DSPy handles prompt engineering.

## 2. 类图与数据流 (Architecture)

```
Input (typed fields)
    ↓
Signature (docstring + InputField/OutputField)
    ↓
dspy.Predict(Signature)
    ↓
forward(**kwargs) → dspy.Prediction
    ↓
Output (typed fields accessible as attributes)
```

Key transformation: Python kwargs → structured prompt → LM completion → parsed typed output fields.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of the given text as positive, negative, or neutral."""
    text: str = dspy.InputField(desc="The text to classify")
    sentiment: str = dspy.OutputField(desc="One of: positive, negative, neutral")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

classify = dspy.Predict(SentimentClassifier)
result = classify(text="I absolutely love this new feature!")

print(result.sentiment)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Writing verbose prompt strings inside `forward()`

**Wrong:**
```python
class BadClassifier(dspy.Signature):
    text: str = dspy.InputField()

def forward(self, text):
    # Never do this — bypasses DSPy's prompt compilation
    return self.lm(f"Classify sentiment of: {text}. Reply with one word.")
```

**Correct:**
```python
class GoodClassifier(dspy.Signature):
    """Classify the sentiment of the given text as positive, negative, or neutral."""
    text: str = dspy.InputField(desc="The text to classify")
    sentiment: str = dspy.OutputField(desc="One of: positive, negative, neutral")
```

> **Debugging tip:** If you find yourself concatenating strings before calling a module, move that text into the Signature docstring or field descriptions.

### Anti-pattern 2: Not using type annotations (3.x requires them)

**Wrong:**
```python
class Untyped(dspy.Signature):
    """Summarize the text."""
    text = dspy.InputField()   # Missing `: str`
    summary = dspy.OutputField()  # Missing `: str`
```

**Correct:**
```python
class Typed(dspy.Signature):
    """Summarize the text."""
    text: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="A one-sentence summary")
```

> **Debugging tip:** DSPy 3.x uses type annotations to generate the JSON schema sent to the LM. Missing annotations cause runtime errors or schema failures.

### Anti-pattern 3: Using `prefix=` parameter (deprecated in 3.x)

**Wrong:**
```python
class PrefixBad(dspy.Signature):
    text: str = dspy.InputField(prefix="Input Text:")
    label: str = dspy.OutputField(prefix="Sentiment:")
```

**Correct:**
```python
class PrefixGood(dspy.Signature):
    """Classify sentiment."""
    text: str = dspy.InputField(desc="The input text")
    label: str = dspy.OutputField(desc="The sentiment label")
```

> **Debugging tip:** Remove all `prefix=` from field definitions. DSPy 3.2+ emits deprecation warnings. Rely on `desc=` and the class docstring instead.

## 5. 组合指南 (Composition)

Predict is the atomic unit of DSPy. Use it alone for straightforward classification, extraction, or transformation tasks where a single LM call suffices.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| Predict + ChainOfThought | First stage reasons (CoT), second stage decides (Predict) — e.g., evaluate arguments then output a verdict |
| Predict + Refine | Predict generates a draft, Refine iteratively improves it |
| Predict + BestOfN | Run Predict N times, select the best output via a metric |
| Predict inside Custom Module | Use Predict as a sub-module for simple transformations in larger pipelines |

**Example: two-stage extraction**
```python
class ExtractNames(dspy.Signature):
    """Extract person names from the text."""
    text: str = dspy.InputField()
    names: str = dspy.OutputField(desc="Comma-separated person names")

class ClassifyRole(dspy.Signature):
    """Classify the role of each person."""
    names: str = dspy.InputField()
    roles: str = dspy.OutputField(desc="JSON mapping name to role")

extract = dspy.Predict(ExtractNames)
classify = dspy.Predict(ClassifyRole)

names = extract(text=doc).names
roles = classify(names=names).roles
```

## 6. 进阶变体 (Advanced Variants)

- **Parallel Predict:** Run multiple `dspy.Predict` instances with different Signatures on the same input, then aggregate results (ensemble classification).
- **Conditional Predict:** Use `Predict` inside a Custom Module with Python `if/else` logic to choose which Signature to invoke based on input characteristics.
- **Typed JSON output:** Define `output: dict` or a Pydantic model annotation to force structured JSON responses directly from `Predict`.
