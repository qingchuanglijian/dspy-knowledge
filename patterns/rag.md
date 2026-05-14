---
pattern_id: P04-RAG
difficulty: beginner
source_tutorial: Retrieval-Augmented Generation from dspy.ai
api_modules: [Retrieve, ChainOfThought]
---

## 1. 核心思想 (Core Concept)

Retrieve relevant passages from a knowledge base, then generate an answer conditioned on them. RAG grounds generation in external knowledge, reducing hallucination by ensuring the model has access to up-to-date or domain-specific facts at inference time.

## 2. 类图与数据流 (Architecture)

```
Query → Retrieve(k passages) → Concatenate(query + passages) → Generate(answer)
```

- **Retrieve:** Takes a query and returns the top-k relevant passages from an external index (vector DB, BM25, or hybrid).
- **Concatenate:** Joins the query and retrieved passages into a single context string.
- **Generate:** A DSPy module (typically `ChainOfThought`) produces the final answer grounded in the provided context.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy
from typing import List, Protocol


class Retriever(Protocol):
    def __call__(self, query: str, k: int) -> List[str]: ...


class GenerateAnswer(dspy.Signature):
    """Answer the question based on the provided context passages."""

    context: str = dspy.InputField(desc="Relevant passages concatenated with newlines")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="Concise answer based on context")


class RAG(dspy.Module):
    def __init__(self, retriever: Retriever, k_passages: int = 3) -> None:
        super().__init__()
        self.retriever = retriever
        self.k = k_passages
        self.generate = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        passages: List[str] = self.retriever(question, k=self.k)
        context = "\n".join(passages)
        return self.generate(context=context, question=question)


# Mock retriever for standalone testing
def mock_retriever(query: str, k: int) -> List[str]:
    return [
        "DSPy is a framework for programming language models.",
        "It emphasizes modular, composable signatures and modules.",
    ][:k]


if __name__ == "__main__":
    lm = dspy.LM("openai/gpt-4o-mini")
    dspy.configure(lm=lm)

    rag = RAG(retriever=mock_retriever, k_passages=2)
    result = rag(question="What is DSPy?")
    print(result.answer)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### 4.1 Retrieve then ignore context
Generating an answer without actually using the retrieved passages. The model hallucinates or relies on parametric knowledge instead of the provided context.

**Symptom:** Answer contradicts retrieved passages or mentions facts not present in context.  
**Fix:** Use `ChainOfThought` and prompt for explicit grounding; add a post-hoc attribution check.

### 4.2 Flatten all retrieved passages into one blob
Concatenating every passage into an unstructured wall of text removes boundaries, making attribution hard and increasing noise.

**Symptom:** Answers blend information across passages incorrectly; citations are impossible.  
**Fix:** Number passages, insert separators, or pass a `List[str]` and let the module reference passage indices.

### 4.3 Hard-code retriever inside the module
Baking retriever logic (e.g., `some_vector_db.query(...)`) directly into `forward()` prevents swapping retrievers for testing or different environments.

**Symptom:** Unit tests require a live vector DB; cannot evaluate with a mock retriever.  
**Fix:** Inject the retriever via `__init__` (as shown in the MVP) so any callable `Retriever` can be substituted.

## 5. 组合指南 (Composition)

| Composition | Purpose | Sketch |
|---|---|---|
| **RAG + ChainOfThought** | Reasoning-aware retrieval | Use `ChainOfThought` not just for the final answer, but also to reformulate the query before retrieval based on reasoning steps. |
| **RAG + MultiHop** | Iterative retrieval | After an initial retrieval and partial answer, generate a follow-up query, retrieve again, and accumulate context across hops. |
| **RAG + Custom Module** | Multi-stage pipeline | Insert a `Judge` or `Refine` module between retrieval and generation (or after generation) to verify support or polish the answer. |

```python
class VerifiableRAG(dspy.Module):
    def __init__(self, retriever: Retriever) -> None:
        super().__init__()
        self.rag = RAG(retriever)
        self.judge = dspy.ChainOfThought("context, answer -> is_supported: bool")

    def forward(self, question: str) -> dspy.Prediction:
        pred = self.rag(question=question)
        verdict = self.judge(context=pred.context, answer=pred.answer)
        if verdict.is_supported:
            return pred
        return self.rag(question=question + " (provide more detail)")
```

## 6. 进阶变体 (Advanced Variants)

- **Hybrid retrieval:** Combine dense vector search with sparse BM25 retrieval, merge and de-duplicate results before passing to the generator. Often improves recall over either method alone.
- **Reranking:** Add a lightweight cross-encoder (or a DSPy `Predict` reranker) that scores the top-k retrieved passages and keeps only the most relevant subset, reducing context noise.
- **Query expansion:** Insert a `RewriteQuery` module before retrieval to expand acronyms, add synonyms, or convert conversational history into a standalone search query.
- **Source attribution:** Extend the signature with a `citations` output field that references the indices of passages used to derive the answer, enabling traceability and verification.
