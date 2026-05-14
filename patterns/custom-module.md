---
pattern_id: P07-CustomModule
difficulty: foundation-intermediate
source_tutorial: Building AI Applications by Customizing DSPy Modules from dspy.ai
api_modules: [Module]
---

## 1. 核心思想 (Core Concept)

The essence of DSPy is composing multiple built-in modules into a new `Module` by implementing `__init__()` (declare sub-modules) and `forward()` (define data flow). This is where "programming not prompting" becomes real: you write a Python class that behaves like any other software component, except its internal operations are powered by LMs. DSPy designs it this way so that AI pipelines are first-class objects—debuggable, optimizable, and reusable—rather than ad-hoc prompt strings.

## 2. 类图与数据流 (Architecture)

```
┌─────────────────────────────────────┐
│         CustomModule                │
│  ┌─────────────────────────────┐    │
│  │ __init__(self)              │    │
│  │   self.step1 = Predict(...) │    │
│  │   self.step2 = CoT(...)     │    │
│  │   self.step3 = Predict(...) │    │
│  └─────────────────────────────┘    │
│              ↓                      │
│  ┌─────────────────────────────┐    │
│  │ forward(self, text)         │    │
│  │   out1 = self.step1(text)   │    │
│  │   out2 = self.step2(out1)   │    │
│  │   return self.step3(out2)   │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

Key transformations: raw input → sub-module predictions → structured output. Each arrow is a typed Signature contract.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class Extract(dspy.Signature):
    """Extract named entities from the text."""
    text: str = dspy.InputField()
    entities: str = dspy.OutputField(desc="Comma-separated list of entities")

class Summarize(dspy.Signature):
    """Summarize the extracted entities into one sentence."""
    entities: str = dspy.InputField()
    summary: str = dspy.OutputField()

class EntityPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(Extract)
        self.summarize = dspy.Predict(Summarize)

    def forward(self, text: str) -> dspy.Prediction:
        ext = self.extract(text=text)
        return self.summarize(entities=ext.entities)

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

pipe = EntityPipeline()
result = pipe(text="Apple Inc. was founded by Steve Jobs in California.")
print(result.summary)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### 1. Writing forward() as a linear script without reusable sub-modules
**Wrong:**
```python
class BadPipeline(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, text: str):
        # Inline logic—can't optimize or reuse steps
        extract = dspy.Predict(Extract)(text=text)
        summary = dspy.Predict(Summarize)(entities=extract.entities)
        return summary
```
**Correct:**
```python
class GoodPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.Predict(Extract)
        self.summarize = dspy.Predict(Summarize)

    def forward(self, text: str):
        return self.summarize(entities=self.extract(text=text).entities)
```
**Debugging tip:** If `dspy.inspect_history()` only shows one step or optimizers can't improve individual stages, you likely inlined the modules.

### 2. Not calling super().__init__()
**Wrong:**
```python
class BrokenModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict(MySig)  # Missing super().__init__()
```
**Correct:**
```python
class FixedModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(MySig)
```
**Debugging tip:** If you see `AttributeError` about missing `_store` or sub-modules not being tracked by optimizers, you forgot `super().__init__()`.

### 3. Side effects in forward()
**Wrong:**
```python
def forward(self, text: str):
    result = self.predict(text=text)
    print(result)               # Side effect
    with open("log.txt", "a") as f:
        f.write(result.output)  # Side effect
    return result
```
**Correct:**
```python
def forward(self, text: str) -> dspy.Prediction:
    return self.predict(text=text)  # Pure transformation
```
**Debugging tip:** Side effects break DSPy's assumption that `forward()` is deterministic and replayable. Move I/O to a wrapper function outside the Module.

### 4. Hard-coding logic that should be parameterized
**Wrong:**
```python
class RAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generate = dspy.ChainOfThought(QASig)

    def forward(self, question: str):
        passages = self.retriever(question, k=3)  # Magic number
```
**Correct:**
```python
class RAG(dspy.Module):
    def __init__(self, retriever, k_passages: int = 3):
        super().__init__()
        self.retriever = retriever
        self.k = k_passages
        self.generate = dspy.ChainOfThought(QASig)

    def forward(self, question: str):
        passages = self.retriever(question, k=self.k)
```
**Debugging tip:** If you find yourself editing the module code to change behavior (e.g., number of passages), that value should be an `__init__` argument.

## 5. 组合指南 (Composition)

A Custom Module **is** the composition mechanism. Every other pattern lives inside it.

- **Predict → CoT → Predict:** Use a cheap model for initial filtering, a stronger model for reasoning, and a final formatting step. Declaring each as a sub-module lets you assign different LMs per step.
- **RAG → Judge → Refine:** Retrieve passages, judge answer quality, and refine if the judge returns a low score. The Custom Module orchestrates the loop.
- **Custom Module inside Custom Module:** Treat a sub-pipeline as its own Module for recursive composition. This mirrors software engineering practices and makes unit-testing possible.

Use Custom Modules alone for simple two-step pipelines; compose them when you need branching, loops, or stateful multi-stage logic.

## 6. 进阶变体 (Advanced Variants)

- **Stateful modules (with memory):** Store intermediate results or user context in `self.memory` during `forward()`, enabling multi-turn conversations or accumulating evidence across calls.
- **Conditional branching inside forward():** Use LM outputs to route data dynamically (e.g., if the confidence score is low, trigger a second retrieval step).
- **Parameterized module selection:** Pass module classes or instances as `__init__` arguments to build meta-modules that swap strategies at runtime (e.g., `SolverModule(strategy=dspy.ChainOfThought)` vs. `SolverModule(strategy=dspy.ProgramOfThought)`).
