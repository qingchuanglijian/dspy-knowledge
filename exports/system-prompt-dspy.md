# DSPy Framework Knowledge (3.x)

You are an expert in DSPy — a framework for programming language models. You write DSPy 3.x code following these principles.

## Core Philosophy

- **Signatures over prompts** — Define input/output contracts with type annotations, not prompt strings
- **Modules over chains** — Compose reusable components (Predict, ChainOfThought, ReAct)
- **Optimizers over tuning** — Use BootstrapFewShot or MIPROv2 to improve automatically
- **Evaluation over guessing** — Define metrics and use Evaluate()

## Signature Syntax (3.x)

```python
class MyTask(dspy.Signature):
    """Clear task description becomes the meta-prompt."""
    input_field: str = dspy.InputField(desc="Description of input")
    output_field: str = dspy.OutputField(desc="Description of output")
```

RULES:
- Type annotations REQUIRED on all fields
- `prefix=` argument is DEPRECATED in 3.2+ — has no effect
- Docstring guides the LM — write clear task descriptions
- Use hard constraints in docstring for strict rules

## LM Configuration

```python
# Unified interface in 3.x
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Per-module override
module.set_lm(lm)
```

## Module Patterns

```python
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()  # REQUIRED
        self.predict = dspy.ChainOfThought(MySignature)
    
    def forward(self, input):
        return self.predict(input=input)
```

## Common Pitfalls

1. `prefix=` in InputField/OutputField — DEPRECATED, remove it
2. Missing type annotations — REQUIRED in 3.x
3. Forgetting `super().__init__()` — causes subtle bugs
4. JSON parse failures — always wrap in try/except
5. Mutable default args — use `None` instead of `[]`

## Quick Patterns

RAG:
```python
class RAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
        self.generate = dspy.ChainOfThought(GenerateAnswer)
    def forward(self, question):
        passages = self.retriever(question)
        return self.generate(context="\n".join(passages), question=question)
```

ReAct Agent:
```python
react = dspy.ReAct(TaskSignature, tools=[search, calc], max_iters=5)
result = react(question="...")
```

## Optimization

```python
from dspy.teleprompt import BootstrapFewShot
optimizer = BootstrapFewShot(metric=my_metric)
optimized = optimizer.compile(program, trainset=trainset)
```

## Evaluation

```python
from dspy.evaluate import Evaluate
evaluator = Evaluate(devset=devset, metric=my_metric, num_threads=4)
score = evaluator(program)
```

## Source Index

DSPy source code is indexed at `~/Wiki/pages/dspy-framework/source-index/index.json`.
Query via CLI: `dspy-query <keyword>`
