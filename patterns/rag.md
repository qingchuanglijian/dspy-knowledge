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

---

**Next:** [`react-agent.md`](./react-agent.md) for agentic reasoning with tools.
