---
pattern_id: P05-MultiHopRAG
difficulty: advanced
source_tutorial: Multi-Hop RAG, RL for Multi-Hop Research
api_modules: [MultiChainComparison, ChainOfThought, Predict]
---

# DSPy Pattern: Multi-hop Reasoning

**Purpose:** Answer complex questions requiring multiple reasoning steps or information sources.  
**Complexity:** Advanced  
**Dependencies:** Retriever or knowledge base

---

## Architecture

```
Question → [Hop 1: Retrieve + Reason] → Sub-question → [Hop 2: Retrieve + Reason] → ... → Final Answer
```

## Implementation

```python
import dspy

# 1. Signatures
class GenerateSubQuestion(dspy.Signature):
    """Break down the question into a sub-question for this hop."""
    question: str = dspy.InputField()
    previous_context: str = dspy.InputField(desc="Information gathered so far")
    sub_question: str = dspy.OutputField(desc="Specific question for this hop")

class SynthesizeAnswer(dspy.Signature):
    """Synthesize final answer from all gathered information."""
    question: str = dspy.InputField()
    all_context: str = dspy.InputField(desc="All retrieved passages")
    answer: str = dspy.OutputField()

# 2. Module
class MultiHopReasoner(dspy.Module):
    def __init__(self, retriever, max_hops: int = 3):
        super().__init__()
        self.retriever = retriever
        self.max_hops = max_hops
        self.gen_sub_q = dspy.ChainOfThought(GenerateSubQuestion)
        self.synthesize = dspy.ChainOfThought(SynthesizeAnswer)
    
    def forward(self, question: str):
        context = []
        
        for hop in range(self.max_hops):
            # Generate sub-question
            if hop == 0:
                sub_q = question
            else:
                sub_q_result = self.gen_sub_q(
                    question=question,
                    previous_context="\n".join(context)
                )
                sub_q = sub_q_result.sub_question
            
            # Retrieve
            passages = self.retriever(sub_q)
            context.extend([f"[Hop {hop+1}] {p}" for p in passages])
        
        # Synthesize
        return self.synthesize(
            question=question,
            all_context="\n".join(context)
        )

# 3. Usage
retriever = lambda q: ["Passage about X...", "Passage about Y..."]
reasoner = MultiHopReasoner(retriever, max_hops=3)

lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=lm)

result = reasoner(question="What is the relationship between A and B?")
print(result.answer)
```

## Self-Ask Variant

The model asks itself questions iteratively:

```python
class SelfAsk(dspy.Module):
    def __init__(self, retriever, max_followups: int = 3):
        super().__init__()
        self.retriever = retriever
        self.max_followups = max_followups
        self.ask = dspy.ChainOfThought(SelfAskSignature)
        self.answer = dspy.ChainOfThought(FinalAnswerSignature)
    
    def forward(self, question: str):
        context = []
        current_q = question
        
        for _ in range(self.max_followups):
            result = self.ask(question=current_q, context="\n".join(context))
            
            if result.followup_question:
                passages = self.retriever(result.followup_question)
                context.extend(passages)
                current_q = result.followup_question
            else:
                break
        
        return self.answer(question=question, context="\n".join(context))
```

## Evaluation

```python
def multi_hop_correctness(example, pred) -> float:
    """Evaluate multi-hop answers."""
    # Often requires LLM-as-judge due to complexity
    judge = dspy.Predict(JudgeSignature)
    result = judge(
        question=example.question,
        gold=example.answer,
        predicted=pred.answer
    )
    return float(result.score)
```

## Pitfalls

1. **Looping** — Without exit condition, agent keeps asking questions. Always include "is_answerable" check.
2. **Context explosion** — Each hop adds more text. Summarize or compress periodically.
3. **Wrong sub-questions** — Poorly formed sub-questions derail retrieval. Use ChainOfThought to improve quality.

## 4. Common Anti-patterns and Diagnosis

Multi-hop reasoning is fragile because errors in early hops propagate. The four anti-patterns below are the most common sources of failure.

### 4.1 Each Hop is Independent
If a hop does not receive the accumulated context from prior hops, it retrieves the same information repeatedly and makes no progress.

**Wrong**
```python
for hop in range(self.max_hops):
    sub_q = self.gen_sub_q(question=question)  # ❌ No previous_context passed
    passages = self.retriever(sub_q)
```

**Correct**
```python
for hop in range(self.max_hops):
    sub_q = self.gen_sub_q(question=question, previous_context="\n".join(context))
    passages = self.retriever(sub_q)
    context.extend(passages)
```

**Debug tip:** Log each sub-question; if they are semantically identical across hops, state is not being passed.

### 4.2 Not Setting a `max_hops` Limit
Without an upper bound, ambiguous queries trigger infinite retrieval loops.

**Wrong**
```python
while not done:  # ❌ No guard; 'done' may never fire
    ...
```

**Correct**
```python
for hop in range(self.max_hops):  # ✅ Hard ceiling
    ...
```

**Debug tip:** Monitor token usage per query; a sudden spike often indicates unbounded looping.

### 4.3 Not Using a Stop Condition
Running all `max_hops` even when the answer is already found wastes tokens and can overwrite a correct answer.

**Wrong**
```python
for hop in range(self.max_hops):
    ...  # ❌ Always runs to max_hops
return self.synthesize(...)
```

**Correct**
```python
for hop in range(self.max_hops):
    ...
    if self.should_stop(question=question, context=context):  # ✅ Early exit
        break
return self.synthesize(...)
```

**Debug tip:** Compare answers synthesized after 1 hop vs `max_hops`; if quality degrades after an early correct retrieval, you need a stop gate.

### 4.4 Retrieving but Not Aggregating Context
Using only the last hop's passages discards evidence gathered earlier.

**Wrong**
```python
for hop in range(self.max_hops):
    passages = self.retriever(sub_q)
    context = passages  # ❌ Overwrites instead of accumulates
return self.synthesize(question=question, all_context="\n".join(context))
```

**Correct**
```python
for hop in range(self.max_hops):
    passages = self.retriever(sub_q)
    context.extend(passages)  # ✅ Accumulates across hops
return self.synthesize(question=question, all_context="\n".join(context))
```

**Debug tip:** If the final answer ignores facts clearly present in early hops, check whether `context` is being overwritten.

## 5. Composition Guide

Multi-hop systems benefit enormously from tool use, verification, and ensemble strategies.

| Pattern | When to Compose | Mini Example |
|---|---|---|
| **Multi-Hop + ReAct** | Each hop needs tool use (calculator, APIs) | The retrieval step inside a hop delegates to a `ReAct` agent that can call tools. |
| **Multi-Hop + Judge** | Verify after each hop before continuing | A `Judge` module scores retrieved context; if irrelevant, trigger query rewriting. |
| **Multi-Hop + BestOfN** | Try multiple hop strategies in parallel | Run *N* independent multi-hop chains and return the answer with the highest judge score. |

```python
# Multi-Hop + Judge sketch
class CheckedMultiHop(dspy.Module):
    def __init__(self, retriever):
        self.reasoner = MultiHopReasoner(retriever)
        self.judge = dspy.Predict(JudgeSignature)

    def forward(self, question: str):
        pred = self.reasoner(question=question)
        verdict = self.judge(question=question, answer=pred.answer, context=pred.all_context)
        if verdict.is_correct:
            return pred
        return self.reasoner(question=question + " (be more specific)")
```

Use multi-hop alone when the retrieval path is linear and deterministic. Compose when individual hops need external tools, when answers must be verified, or when you want to explore multiple reasoning paths.

## 6. Advanced Variants

- **Learned hop termination:** Train a small DSPy classifier (`dspy.Predict`) that takes the current context and sub-question as input and outputs a `should_stop` boolean. Compile it with labeled examples to replace heuristic stop rules.
- **Adaptive `k` per hop:** Use an early hop with a large `k` to explore broadly, then reduce `k` in later hops once the topic is narrowed. Expose `k` as a per-hop parameter.
- **Context compression:** Periodically summarize the accumulated context with a `Summarize` module so that later hops receive a fixed-size input regardless of hop count.

---

**Next:** Review [`../core-concepts.md`](../core-concepts.md) for optimizer integration with multi-hop.
