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

---

**Next:** Review [`../core-concepts.md`](../core-concepts.md) for optimizer integration with multi-hop.
