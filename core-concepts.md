# DSPy Core Concepts (3.x)

**Purpose:** Deep understanding of DSPy's 5 pillars — what they are, why they exist, and how they compose.

---

## 1. Signatures — Declarative Contracts

A Signature is a **typed contract** between your program and the language model. It describes:
- What inputs the LM receives
- What outputs the LM produces
- The semantic task (via docstring)

### Basic Syntax (DSPy 3.x)

```python
import dspy

class SentimentAnalysis(dspy.Signature):
    """Analyze the sentiment of the given text."""
    
    text: str = dspy.InputField(desc="The text to analyze")
    sentiment: str = dspy.OutputField(desc="One of: positive, negative, neutral")
    confidence: float = dspy.OutputField(desc="Confidence score from 0.0 to 1.0")
```

### Key Rules

| Rule | DSPy 2.x | DSPy 3.x | Impact |
|------|----------|----------|--------|
| Type annotations | Optional | **Required** | 3.x uses them for schema generation |
| `prefix` argument | Used in `InputField`/`OutputField` | **Deprecated** (3.2+) | Remove all `prefix=` — produces warnings |
| Docstring | Ignored | **Becomes meta-prompt** | Write clear task descriptions |
| Multiple outputs | Comma-separated string | **Multiple fields** | Cleaner, typed outputs |

### Hard Constraints Pattern

Use docstring for constraints that must never be violated:

```python
class RiskExtraction(dspy.Signature):
    """Extract risk factors from the report.
    
    Hard constraints:
    1. Only output factor codes from the provided list.
    2. Each match must include evidence text from the source.
    3. Do not infer risks not explicitly mentioned.
    """
    
    report_content: str = dspy.InputField()
    candidate_factors: str = dspy.InputField(desc="JSON list of valid factor codes")
    matched_factors: str = dspy.OutputField(desc="JSON array of matched factors with evidence")
```

---

## 2. Modules — Composable Logic

Modules are the **building blocks** of DSPy programs. They wrap Signatures with execution logic.

### Built-in Modules

```python
import dspy

# Simple prediction
predict = dspy.Predict(SentimentAnalysis)
result = predict(text="I love this product!")

# Chain of Thought (reasoning steps exposed)
cot = dspy.ChainOfThought(SentimentAnalysis)
result = cot(text="This movie was okay, but the ending ruined it.")
# result includes: rationale, sentiment, confidence

# Program of Thought (generates and executes code)
pot = dspy.ProgramOfThought(MathProblem)
result = pot(question="What is 123 * 456?")

# ReAct (interleaves reasoning + tool use)
react = dspy.ReAct(TaskSignature, tools=[search, calculator])
result = react(query="What is the population of Paris in 2024?")
```

### Custom Module Pattern

```python
class MultiStageExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_entities = dspy.ChainOfThought(EntityExtraction)
        self.extract_relations = dspy.ChainOfThought(RelationExtraction)
        self.summarize = dspy.Predict(SummarizeRelations)
    
    def forward(self, document: str) -> dict:
        entities = self.extract_entities(document=document)
        relations = self.extract_relations(
            document=document,
            entities=entities.entity_list
        )
        summary = self.summarize(relations=relations.relation_list)
        return {
            "entities": entities.entity_list,
            "relations": relations.relation_list,
            "summary": summary.summary
        }
```

### Module Composition Rules

1. **Always call `super().__init__()`** in `__init__`
2. **Always implement `forward()`** — it's the contract
3. **Return structured data** — don't print or side-effect
4. **Handle exceptions** — wrap JSON parsing in try/except

---

## 3. LM (Language Model) — Unified Backend

DSPy 3.x unifies all LM providers behind `dspy.LM()`.

### Configuration Patterns

```python
import dspy
import os

# Pattern 1: Global configuration (simplest)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Pattern 2: Per-module LM (isolation)
lm_cheap = dspy.LM("openai/gpt-4o-mini")
lm_powerful = dspy.LM("anthropic/claude-sonnet-4")

quick_predict = dspy.Predict(SimpleTask)
quick_predict.set_lm(lm_cheap)

deep_reason = dspy.ChainOfThought(ComplexTask)
deep_reason.set_lm(lm_powerful)

# Pattern 3: Environment-driven (production)
lm = dspy.LM(
    model=os.environ.get("DSPY_MODEL", "openai/gpt-4o-mini"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    api_base=os.environ.get("OPENAI_API_BASE"),
    max_tokens=4096,
    temperature=0.0,
)
dspy.configure(lm=lm)
```

### Supported Providers

| Provider | Model string example | Notes |
|----------|---------------------|-------|
| OpenAI | `openai/gpt-4o` | Default, most tested |
| Anthropic | `anthropic/claude-sonnet-4` | Via litellm |
| Azure OpenAI | `azure/your-deployment-name` | Set `api_base` |
| Local (vLLM) | `openai/localhost:8000` | Set `api_base` |
| Ollama | `ollama/llama3.2` | Local models |
| Any (litellm) | `provider/model-name` | 100+ providers via litellm |

### LM Context Management

```python
# Inspect recent predictions
dspy.inspect_history(n=3)

# Clear context (rarely needed)
# DSPy manages context automatically per-prediction
```

---

## 4. Optimizers — Automated Improvement

Optimizers automatically improve your program's prompts and/or examples.

### BootstrapFewShot (Starter)

```python
from dspy.teleprompt import BootstrapFewShot

def metric(example, pred) -> float:
    return 1.0 if pred.sentiment == example.sentiment else 0.0

optimizer = BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=4,  # Max auto-generated examples
    max_labeled_demos=8,       # Max labeled examples from trainset
)

optimized = optimizer.compile(program, trainset=trainset)
```

### MIPROv2 (Advanced)

```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=metric,
    num_candidates=10,
    init_temperature=1.0,
    verbose=True,
)

optimized = optimizer.compile(program, trainset=trainset, valset=valset)
```

### Ensemble

```python
from dspy.teleprompt import Ensemble

programs = [prog1, prog2, prog3]
ensemble = Ensemble(programs=programs, reduce_fn=dspy.majority_vote)
```

### Optimizer Selection Guide

| Optimizer | When to use | Data needed | Time |
|-----------|------------|-------------|------|
| `BootstrapFewShot` | Quick wins, small datasets | 20-50 examples | Minutes |
| `BootstrapFewShotWithRandomSearch` | Better few-shot selection | 50-200 examples | 10-30 min |
| `MIPROv2` | Best quality, production | 200+ examples | Hours |
| `Ensemble` | Combine multiple approaches | Multiple programs | Minutes |

---

## 5. Evaluators — Systematic Assessment

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=devset,
    metric=accuracy,
    num_threads=4,
    display_progress=True,
    display_table=5,  # Show top 5 failures
)

score = evaluator(optimized_program)
print(f"Accuracy: {score:.2%}")
```

### Metric Patterns

```python
# Simple exact match
def exact_match(example, pred) -> float:
    return 1.0 if example.answer == pred.answer else 0.0

# F1 for spans
def span_f1(example, pred) -> float:
    gold = set(example.entities)
    pred = set(pred.entities)
    if not pred:
        return 0.0
    precision = len(gold & pred) / len(pred)
    recall = len(gold & pred) / len(gold)
    return 2 * (precision * recall) / (precision + recall)

# LLM-as-judge
def llm_judge(example, pred) -> float:
    with dspy.context(lm=judge_lm):
        judge = dspy.Predict(JudgeSignature)
        result = judge(
            question=example.question,
            gold=example.answer,
            predicted=pred.answer
        )
        return result.score  # 0.0 to 1.0
```

---

## Composing the 5 Pillars

```python
# 1. Define the contract
class QASignature(dspy.Signature):
    """Answer the question based on the context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# 2. Build the module
class RAGModule(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generate = dspy.ChainOfThought(QASignature)
    
    def forward(self, question):
        passages = self.retriever(question)
        context = "\n".join(passages)
        return self.generate(context=context, question=question)

# 3. Configure the LM
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 4. Optimize
optimizer = BootstrapFewShot(metric=exact_match)
optimized = optimizer.compile(RAGModule(retriever), trainset=trainset)

# 5. Evaluate
evaluator = Evaluate(devset=devset, metric=exact_match)
score = evaluator(optimized)
```

---

**Next:** Read [`api-cheatsheet-3x.md`](./api-cheatsheet-3x.md) for quick syntax reference.
