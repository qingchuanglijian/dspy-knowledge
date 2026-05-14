---
pattern_id: P04-RAG
difficulty: beginner
source_tutorial: Retrieval-Augmented Generation
api_modules: [Predict, ChainOfThought]
---

# DSPy Pattern: RAG (Retrieval-Augmented Generation)

**Purpose:** Build a RAG pipeline with DSPy — retrieve relevant passages, then generate answers.  
**Complexity:** Beginner — Intermediate  
**Dependencies:** Any retriever (vector DB, BM25, hybrid)

---

## Architecture

```
Question → [Retriever] → Passages → [DSPy Module] → Answer
                    ↓
            (Can be any: vector search, BM25, hybrid)
```

## Basic Implementation

```python
import dspy
from typing import List

# 1. Signature
class GenerateAnswer(dspy.Signature):
    """Answer the question based on the provided context passages."""
    context: str = dspy.InputField(desc="Relevant passages concatenated with newlines")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Concise answer based on context")

# 2. Module
class RAG(dspy.Module):
    def __init__(self, retriever, k_passages: int = 3):
        super().__init__()
        self.retriever = retriever
        self.k = k_passages
        self.generate = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question: str) -> dspy.Prediction:
        passages = self.retriever(question, k=self.k)
        context = "\n".join(passages)
        return self.generate(context=context, question=question)

# 3. Usage
retriever = lambda q, k: ["Passage 1...", "Passage 2..."]  # Your retriever
rag = RAG(retriever, k_passages=3)

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

result = rag(question="What is DSPy?")
print(result.answer)
```

## Advanced: Multi-hop RAG

For questions requiring multiple retrieval steps:

```python
class GenerateSearchQuery(dspy.Signature):
    """Generate a search query to find missing information."""
    question: str = dspy.InputField()
    existing_context: str = dspy.InputField()
    search_query: str = dspy.OutputField(desc="Query to retrieve missing info")

class MultiHopRAG(dspy.Module):
    def __init__(self, retriever, max_hops: int = 3):
        super().__init__()
        self.retriever = retriever
        self.max_hops = max_hops
        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question: str):
        context = []
        for hop in range(self.max_hops):
            if hop == 0:
                query = question
            else:
                query_result = self.generate_query(
                    question=question,
                    existing_context="\n".join(context)
                )
                query = query_result.search_query
            
            passages = self.retriever(query)
            context.extend(passages)
        
        return self.generate_answer(
            context="\n".join(context),
            question=question
        )
```

## Evaluation

```python
def answer_exact_match(example, pred) -> float:
    return 1.0 if example.answer.lower() == pred.answer.lower() else 0.0

def answer_contains(example, pred) -> float:
    return 1.0 if example.answer.lower() in pred.answer.lower() else 0.0

# Run evaluation
evaluator = Evaluate(devset=devset, metric=answer_exact_match, num_threads=4)
score = evaluator(rag)
```

## Optimization

```python
# Compile with few-shot examples
optimizer = BootstrapFewShot(metric=answer_exact_match)
optimized_rag = optimizer.compile(rag, trainset=trainset)

# Evaluate optimized
score = evaluator(optimized_rag)
```

## Pitfalls

1. **Empty retrieval** — Always handle `passages=[]` → return "No relevant information found"
2. **Context too long** — Truncate or rank passages by relevance
3. **Retriever not returning strings** — Ensure retriever returns `List[str]`
4. **No attribution** — Consider adding `citations` output field

## 4. Common Anti-patterns and Diagnosis

RAG is conceptually simple, but subtle implementation errors destroy optimizability and accuracy. Here are the four mistakes agents make most often.

### 4.1 Coupling Retriever and Generator in One Module
Tightly binding retrieval and generation prevents independent optimization of each stage.

**Wrong**
```python
class RAG(dspy.Module):
    def __init__(self):
        self.generate = dspy.Predict(GenerateAnswer)  # ❌ Retriever logic buried inside forward

    def forward(self, question: str):
        passages = some_vector_db.query(question)  # ❌ Hard-wired, non-configurable
        return self.generate(context="\n".join(passages), question=question)
```

**Correct**
```python
class RAG(dspy.Module):
    def __init__(self, retriever, k_passages: int = 3):
        self.retriever = retriever
        self.k = k_passages
        self.generate = dspy.ChainOfThought(GenerateAnswer)
```

**Debug tip:** If you cannot swap retrievers or tune `k` without rewriting the module, the coupling is too tight.

### 4.2 Not Using ChainOfThought for the Generator
A direct `Predict` lacks explicit reasoning and often skips steps or hallucinates.

**Wrong**
```python
self.generate = dspy.Predict(GenerateAnswer)  # ❌ No reasoning trace
```

**Correct**
```python
self.generate = dspy.ChainOfThought(GenerateAnswer)  # ✅ Forces step-by-step reasoning
```

**Debug tip:** Check the `rationale` field in the prediction; if it is empty or missing, you are using `Predict` instead of `ChainOfThought`.

### 4.3 Hard-coding Retriever `k`
Baking the number of passages into the module body makes it impossible to adapt to different question complexities.

**Wrong**
```python
passages = self.retriever(question, k=3)  # ❌ Magic number
```

**Correct**
```python
passages = self.retriever(question, k=self.k)  # ✅ Configurable at init/compile time
```

**Debug tip:** Evaluate with varying `k` values; if performance plateaus early, your hard-coded value may be suboptimal.

### 4.4 Passing Raw Passage Objects
Feeding list-of-dicts or database records into the signature instead of a formatted context string confuses the LM.

**Wrong**
```python
context = self.retriever(question)  # ❌ Returns [{"text": "..."}, ...]
return self.generate(context=context, question=question)
```

**Correct**
```python
context = "\n\n".join(p["text"] for p in self.retriever(question))
return self.generate(context=context, question=question)
```

**Debug tip:** If the model produces JSON-ish output or complains about "object" references, you likely passed raw structures instead of a string.

## 5. Composition Guide

RAG becomes significantly more powerful when combined with iterative retrieval, verification, or refinement patterns.

| Pattern | When to Compose | Mini Example |
|---|---|---|
| **RAG + Multi-Hop** | Iterative retrieval for complex questions | Use the answer from hop *n* to generate a follow-up query for hop *n+1*. |
| **RAG + Judge** | Verify the generated answer against retrieved context | A separate `Judge` module scores `answer` vs `context`; retry if inconsistent. |
| **RAG + Refine** | Iterative answer improvement | Generate an initial answer, then feed it back with the context to polish or expand. |

```python
# RAG + Judge sketch
class VerifiableRAG(dspy.Module):
    def __init__(self, retriever):
        self.rag = RAG(retriever)
        self.judge = dspy.Predict(JudgeSignature)

    def forward(self, question: str):
        pred = self.rag(question=question)
        verdict = self.judge(context=pred.context, answer=pred.answer)
        if verdict.is_supported:
            return pred
        return self.rag(question=question + " (provide more detail)")
```

Use standalone RAG for straightforward single-hop questions. Compose when the query requires disambiguation, verification, or progressive narrowing.

## 6. Advanced Variants

- **Hybrid retrieval:** Combine BM25 sparse retrieval with dense vector search, then merge and de-duplicate results before passing to the generator.
- **Reranking:** Add a lightweight cross-encoder or DSPy `Predict` reranker that scores the top-*k* retrieved passages and keeps only the most relevant subset.
- **Query rewriting:** Insert a `RewriteQuery` module before retrieval to expand acronyms, add synonyms, or convert conversational history into a standalone search query.

---

**Next:** [`react-agent.md`](./react-agent.md) for agentic reasoning with tools.
