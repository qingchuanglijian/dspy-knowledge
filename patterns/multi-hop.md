---
pattern_id: P05-MultiHopRAG
difficulty: advanced
source_tutorial: Multi-Hop Retrieval from dspy.ai
api_modules: [MultiHop, ChainOfThought]
---

## 1. 核心思想 (Core Concept)

Complex questions often require connecting information from multiple sources. Multi-hop iteratively retrieves documents, uses intermediate findings to reformulate queries, and retrieves again until the answer can be synthesized.

## 2. 类图与数据流 (Architecture)

```
Query → Retrieve → Reason → Generate sub-query → Retrieve → ... → Synthesize final answer
```

## 3. 最小可运行代码 (MVP Code)

```python
import dspy


class GenerateFollowUp(dspy.Signature):
    """Generate a follow-up query based on intermediate findings."""
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Retrieved passages so far")
    follow_up: str = dspy.OutputField(desc="Next query or 'DONE' if sufficient")
    reasoning: str = dspy.OutputField(desc="Step-by-step reasoning")


class SynthesizeAnswer(dspy.Signature):
    """Synthesize the final answer from all gathered context."""
    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="All retrieved passages")
    answer: str = dspy.OutputField()


class MultiHopRAG(dspy.Module):
    def __init__(self, retriever, max_hops: int = 3):
        super().__init__()
        self.retriever = retriever
        self.max_hops = max_hops
        self.follow_up = dspy.ChainOfThought(GenerateFollowUp)
        self.synthesize = dspy.ChainOfThought(SynthesizeAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        context: list[str] = []
        current_query = question

        for _ in range(self.max_hops):
            passages = self.retriever(current_query)
            if not passages:
                break
            context.extend(passages)

            follow = self.follow_up(question=question, context="\n\n".join(context))
            if follow.follow_up.strip().upper() == "DONE":
                break
            current_query = follow.follow_up

        return self.synthesize(question=question, context="\n\n".join(context))


# Usage

def dummy_retriever(query: str) -> list[str]:
    return [f"Passage about: {query}"]


lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

rag = MultiHopRAG(retriever=dummy_retriever, max_hops=3)
result = rag(question="What is the relationship between A and B?")
print(result.answer)
```

## 4. 常见反模式与诊断 (Anti-patterns)

| # | Anti-pattern | Symptom | Fix |
|---|---|---|---|
| 1 | **Too many hops without a stopping condition** | Infinite retrieval loops, spiraling token usage | Add a `max_hops` ceiling and a per-hop `is_sufficient` / `DONE` gate |
| 2 | **Not using intermediate reasoning to guide next hop** | Random follow-up queries that ignore prior context | Pass accumulated context into every `GenerateFollowUp` call and require explicit `reasoning` |
| 3 | **Ignoring retrieval failures** | Empty results crash the pipeline or produce hallucinated answers | Check `if not passages: break` after each retrieval step |

## 5. 组合指南 (Composition)

| Combination | Role of Multi-hop | Role of Partner | When to Use |
|---|---|---|---|
| **MultiHop + RAG** | Orchestrates the overall hop loop | Each hop is a self-contained RAG step (retrieve then generate) | Linear multi-document questions where each hop can be answered independently |
| **MultiHop + ChainOfThought** | Structures the iterative retrieval | Provides explicit reasoning traces inside every hop | Debugging complex reasoning paths; improves sub-query quality |
| **MultiHop + Custom Module** | Acts as the retriever interface | Maintains shared state (e.g., visited docs, running summary) across hops | Long-running research tasks that need persistent memory |

## 6. 进阶变体 (Advanced Variants)

- **Learned stopping classifier** — Replace the hard-coded `DONE` heuristic with a small `dspy.Predict` classifier trained on labeled examples to decide when context is sufficient.
- **Adaptive hop count** — Start with a large retrieval `k` for broad exploration, then shrink `k` in later hops once the topic is narrowed.
- **Parallel hops (branching)** — Spawn multiple independent follow-up queries at each hop, explore branches in parallel, and select the best-supported answer.
- **Confidence-based early stopping** — After each hop, score answer confidence; exit immediately if confidence exceeds a calibrated threshold.
