# DSPy 2.x → 3.x Migration Guide

**Purpose:** Checklist for upgrading existing DSPy 2.x code to 3.x.  
**Last updated:** 2026-05-11

---

## Pre-Migration Checklist

- [ ] Pin current DSPy version: `pip show dspy`
- [ ] Run existing tests and record baseline scores
- [ ] Check which LM providers you use (OpenAI, Azure, local, etc.)
- [ ] Identify all `InputField`/`OutputField` usages with `prefix=`
- [ ] Identify all `dspy.OpenAI()` / `dspy.AzureOpenAI()` calls
- [ ] Identify all `dspy.settings.configure()` calls

---

## Step 1: Update Dependency

```bash
# Before
pip install dspy-ai>=2.0

# After
pip install dspy>=3.0

# Or pin for reproducibility
pip install dspy==3.2.0
```

---

## Step 2: Replace LM Configuration

### ❌ 2.x Style (DEPRECATED)

```python
import dspy

openai_lm = dspy.OpenAI(model="gpt-4o-mini", api_key="...")
dspy.settings.configure(lm=openai_lm)

azure_lm = dspy.AzureOpenAI(
    api_base="...",
    api_version="...",
    model="gpt-4",
)
```

### ✅ 3.x Style

```python
import dspy

# OpenAI
lm = dspy.LM("openai/gpt-4o-mini", api_key="...")
dspy.configure(lm=lm)

# Azure
lm = dspy.LM("azure/your-deployment", api_base="...", api_key="...")
dspy.configure(lm=lm)

# Local (vLLM/Ollama)
lm = dspy.LM("openai/localhost:8000", api_base="http://localhost:8000")
dspy.configure(lm=lm)
```

---

## Step 3: Remove `prefix=` from Fields

### ❌ 2.x Style (DEPRECATED in 3.2+)

```python
class MySignature(dspy.Signature):
    text: str = dspy.InputField(prefix="Document:")
    summary: str = dspy.OutputField(prefix="Summary:")
```

### ✅ 3.x Style

```python
class MySignature(dspy.Signature):
    """Summarize the document."""
    text: str = dspy.InputField(desc="Document to summarize")
    summary: str = dspy.OutputField(desc="Concise summary")
```

**Migration command:**
```bash
# Find all occurrences
grep -rn "prefix=" your_project/

# Replace systematically:
# dspy.InputField(prefix="...") → dspy.InputField(desc="...")
# dspy.OutputField(prefix="...") → dspy.OutputField(desc="...")
```

---

## Step 4: Add Type Annotations

### ❌ 2.x (Still works but not recommended)

```python
class MySignature(dspy.Signature):
    text = dspy.InputField()  # No type annotation
    summary = dspy.OutputField()
```

### ✅ 3.x (Required)

```python
class MySignature(dspy.Signature):
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()
```

**Migration command:**
```bash
# Find signatures without type annotations
grep -rn "dspy.InputField()" your_project/
grep -rn "dspy.OutputField()" your_project/
```

---

## Step 5: Update Optimizer Imports

### ❌ 2.x

```python
from dspy.teleprompt import MIPRO
```

### ✅ 3.x

```python
from dspy.teleprompt import MIPROv2  # MIPRO is deprecated
```

---

## Step 6: Test Everything

```python
# Run your evaluation pipeline
score = evaluator(your_program)
print(f"Post-migration score: {score:.2%}")

# Compare with baseline
# If score drops > 5%, check:
# 1. Did `prefix=` carry semantic meaning? (Move to docstring)
# 2. Are type annotations correct?
# 3. Is LM configured with same parameters?
```

---

## Common Migration Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `TypeError: InputField() got an unexpected keyword argument 'prefix'` | DSPy 3.2+ removed `prefix` | Remove `prefix=`, use `desc=` |
| `AttributeError: module 'dspy' has no attribute 'OpenAI'` | Using 3.x API | Replace with `dspy.LM()` |
| `Signature requires type annotations` | Missing `: str` annotations | Add type hints to all fields |
| `dspy.settings has no attribute 'configure'` | Old API | Use `dspy.configure(lm=...)` |
| Lower scores after migration | `prefix` had semantic value | Move that value to docstring |

---

**Post-migration:** Run full test suite and compare scores with pre-migration baseline.
