# DSPy 3.x API Cheatsheet

**Purpose:** Quick syntax reference during implementation.  
**Version:** DSPy 3.2+ (latest stable)

---

## Signature Definition

```python
import dspy

class MySignature(dspy.Signature):
    """Task description that becomes the meta-prompt."""
    
    # Inputs
    input_field: str = dspy.InputField(desc="Description")
    
    # Outputs  
    output_field: str = dspy.OutputField(desc="Description")
    # ❌ DEPRECATED in 3.2+: prefix="..." — has no effect, produces warning
```

| Element | Syntax | Notes |
|---------|--------|-------|
| Input field | `name: type = dspy.InputField(desc="...")` | `desc` is optional but recommended |
| Output field | `name: type = dspy.OutputField(desc="...")` | Type annotations **required** in 3.x |
| Docstring | `"""Task description"""` | Becomes part of the prompt |

---

## Module Instantiation

```python
# Simple prediction
predict = dspy.Predict(Signature)
result = predict(input_field="value")

# Chain of Thought (reasoning exposed)
cot = dspy.ChainOfThought(Signature)
result = cot(input_field="value")
# result.rationale — the reasoning steps

# Program of Thought (generates code)
pot = dspy.ProgramOfThought(Signature)
result = pot(input_field="value")

# ReAct (reasoning + tools)
react = dspy.ReAct(Signature, tools=[tool1, tool2])
result = react(input_field="value")
```

---

## LM Configuration

```python
# Global (simplest)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# With kwargs
lm = dspy.LM(
    "openai/gpt-4o",
    api_key="...",
    api_base="...",
    max_tokens=4096,
    temperature=0.0,
)
dspy.configure(lm=lm)

# Per-module override
module.set_lm(lm)

# Context manager (temporary override)
with dspy.context(lm=other_lm):
    result = module(input="...")
```

---

## Optimizers

```python
from dspy.teleprompt import BootstrapFewShot, MIPROv2, Ensemble

# BootstrapFewShot
optimizer = BootstrapFewShot(
    metric=my_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=8,
)
optimized = optimizer.compile(program, trainset=trainset)

# MIPROv2
optimizer = MIPROv2(
    metric=my_metric,
    num_candidates=10,
    init_temperature=1.0,
    verbose=True,
)
optimized = optimizer.compile(program, trainset=trainset, valset=valset)

# Ensemble
ensemble = Ensemble(programs=[prog1, prog2], reduce_fn=dspy.majority_vote)
```

---

## Evaluation

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=devset,
    metric=my_metric,
    num_threads=4,
    display_progress=True,
    display_table=5,
)
score = evaluator(program)
```

---

## Debugging

```python
# Inspect last N predictions
dspy.inspect_history(n=3)

# Pretty print prediction
print(result)

# Access fields
print(result.output_field)
print(result.rationale)  # For ChainOfThought
```

---

## Common Patterns

### JSON Output

```python
class JSONOutput(dspy.Signature):
    """Extract structured data."""
    text: str = dspy.InputField()
    json_output: str = dspy.OutputField(desc="Valid JSON string")

# In module:
try:
    data = json.loads(result.json_output)
except json.JSONDecodeError:
    data = {}  # graceful fallback
```

### Classification

```python
class Classify(dspy.Signature):
    """Classify the text into one category."""
    text: str = dspy.InputField()
    category: str = dspy.OutputField(desc="One of: A, B, C")
    confidence: float = dspy.OutputField(desc="0.0 to 1.0")
```

### Multi-hop Retrieval

```python
class MultiHopRAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generate = dspy.ChainOfThought(QASignature)
        self.refine = dspy.ChainOfThought(RefineSignature)
    
    def forward(self, question):
        passages = []
        for hop in range(3):
            query = question if hop == 0 else refined.sub_query
            new_passages = self.retriever(query)
            passages.extend(new_passages)
            refined = self.refine(question=question, passages=passages)
        return self.generate(context="\n".join(passages), question=question)
```

---

## 2.x → 3.x Quick Mapping

| 2.x | 3.x | Notes |
|-----|-----|-------|
| `dspy.OpenAI(...)` | `dspy.LM("openai/...")` | Unified interface |
| `dspy.AzureOpenAI(...)` | `dspy.LM("azure/...")` | Set `api_base` |
| `dspy.settings.configure(lm=...)` | `dspy.configure(lm=...)` | Simpler |
| `dspy.Predict(Signature)(input=...)` | Same | No change |
| `InputField(prefix="...")` | `InputField(desc="...")` | `prefix` deprecated |
| `OutputField(prefix="...")` | `OutputField(desc="...")` | `prefix` deprecated |
| `dspy.teleprompt.BootstrapFewShot` | Same | No change |
| `dspy.teleprompt.MIPRO` | `dspy.teleprompt.MIPROv2` | v2 is default now |

---

**Next:** [`migration-2x-to-3x.md`](./migration-2x-to-3x.md) for detailed migration checklist.
