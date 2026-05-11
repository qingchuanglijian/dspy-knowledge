# DSPy Common Pitfalls

**Purpose:** Traps discovered through real project usage. Read before every code review.  
**Last updated:** 2026-05-11

---

## 💥 P0: Critical

### 1. `prefix` Argument Deprecated (3.2+)

**Problem:** Code using `prefix=` in `InputField`/`OutputField` silently does nothing in 3.2+ and generates warnings.

```python
# ❌ WRONG — prefix has NO EFFECT in 3.2+
text: str = dspy.InputField(prefix="Document:")

# ✅ CORRECT — use desc or docstring
text: str = dspy.InputField(desc="Document to summarize")
```

**Detection:**
```bash
grep -rn "prefix=" your_project/
```

**Fix:** Replace all `prefix=` with `desc=` or move semantic guidance to Signature docstring.

---

### 2. Missing Type Annotations

**Problem:** DSPy 3.x requires type annotations on Signature fields. Without them, schema generation fails.

```python
# ❌ WRONG
text = dspy.InputField()

# ✅ CORRECT
text: str = dspy.InputField()
```

**Fix:** Add `: str`, `: int`, `: float`, `: list[str]`, etc. to every field.

---

### 3. JSON Parse Failures in Modules

**Problem:** LLMs sometimes output malformed JSON. If you don't handle this, your pipeline crashes.

```python
# ❌ WRONG — crashes on bad JSON
factors = json.loads(result.matched_risk_factors_json)

# ✅ CORRECT — graceful fallback
try:
    factors = json.loads(result.matched_risk_factors_json)
    return factors if isinstance(factors, list) else []
except (json.JSONDecodeError, AttributeError):
    return []
```

---

## 🚨 P1: High

### 4. Heavy Dependency Installation

**Problem:** `pip install dspy>=3.0` can take several minutes and install many sub-dependencies.

**Impact:** Subagent timeouts, slow CI builds.

**Fixes:**
- Pre-install in base Docker image
- Use `uv pip install dspy` (faster)
- Pin exact version: `dspy==3.2.0`

---

### 5. Category Code Mismatch (Config-Driven Pipelines)

**Problem:** When using YAML configs to filter extraction targets, a typo in `applicable_categories` silently excludes factors.

```yaml
# ❌ WRONG — typo
applicable_categories:
  - FAMIL_CONFLICT  # Should be FAMILY_CONFLICT

# ✅ CORRECT — exact match
applicable_categories:
  - FAMILY_CONFLICT
```

**Fix:** Use enums/constants, never string literals.

---

### 6. LM Not Configured Before Module Use

**Problem:** Calling a module before `dspy.configure(lm=...)` raises cryptic errors.

```python
# ❌ WRONG
predict = dspy.Predict(MySignature)
result = predict(text="...")  # No LM configured!

# ✅ CORRECT
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
predict = dspy.Predict(MySignature)
result = predict(text="...")
```

---

### 7. Confusing `dspy.inspect_history()` Output

**Problem:** `inspect_history()` shows raw prompt/response, but formatting can be hard to read.

**Fix:** Use `print(dspy.inspect_history(n=1))` for better formatting, or parse the history object directly.

---

## ⚠️ P2: Medium

### 8. Mutable Default Arguments in Modules

```python
# ❌ WRONG
class BadModule(dspy.Module):
    def __init__(self, items=[]):
        self.items = items  # Shared across instances!

# ✅ CORRECT
class GoodModule(dspy.Module):
    def __init__(self, items=None):
        self.items = items or []
```

---

### 9. Forgetting `super().__init__()`

```python
# ❌ WRONG
class MyModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(Signature)  # Missing super()!

# ✅ CORRECT
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(Signature)
```

---

### 10. Temperature Mismatch Between Dev and Prod

**Problem:** Using `temperature=0.7` in dev but `temperature=0.0` in prod causes evaluation drift.

**Fix:** Centralize LM config:
```python
class LMConfig:
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0  # Deterministic for evaluation
    max_tokens: int = 4096
```

---

## Checklist Before Code Review

- [ ] No `prefix=` in any `InputField`/`OutputField`
- [ ] All Signature fields have type annotations
- [ ] JSON parsing wrapped in try/except
- [ ] `super().__init__()` called in custom Modules
- [ ] LM configured before first module use
- [ ] No mutable default arguments
- [ ] Category codes match exactly (config-driven pipelines)
- [ ] Temperature is intentional, not accidental

---

**Next:** Review [`core-concepts.md`](./core-concepts.md) for architecture refresher.
