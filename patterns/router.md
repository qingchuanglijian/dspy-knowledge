---
pattern_id: P12-Router
difficulty: intermediate
source_tutorial: Privacy-Conscious Delegation from dspy.ai
api_modules: [Predict]
---

## 1. 核心思想 (Core Concept)

A Router is a classifier that reads the input and decides which sub-module or specialized pipeline should handle it. The Router is the "traffic cop" of multi-domain DSPy systems — its accuracy determines the entire system's reliability. If the Router misclassifies, the wrong specialist processes the request and every downstream step compounds the error. In DSPy, a Router is typically implemented as a `Predict` module with a Signature whose output is a route label. The design philosophy is separation of concerns: instead of building one monolithic module that handles everything, you build focused specialists and let a lightweight classifier delegate. This makes each module easier to optimize, evaluate, and maintain.

## 2. 类图与数据流 (Architecture)

```
Input (query / document / request)
    ↓
Router Signature
    query: str = InputField
    route: str = OutputField   ← e.g. "billing", "tech_support", "general"
    ↓
dspy.Predict(RouterSignature)
    ↓
forward(query=...)
    ↓
Python if/elif/else or dict dispatch
    ↓
Specialized Module A / B / C
    ↓
Final Output
```

Key transformation: Input → single classification call → route label → dynamic dispatch to the correct specialist pipeline.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class RouteQuery(dspy.Signature):
    """Route the user query to the correct department."""
    query: str = dspy.InputField(desc="The user's question")
    route: str = dspy.OutputField(desc="One of: billing, tech_support, general")

class BillingAnswer(dspy.Signature):
    """Answer billing questions."""
    query: str = dspy.InputField()
    answer: str = dspy.OutputField()

class TechAnswer(dspy.Signature):
    """Answer technical questions."""
    query: str = dspy.InputField()
    answer: str = dspy.OutputField()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

router = dspy.Predict(RouteQuery)
billing = dspy.Predict(BillingAnswer)
tech = dspy.Predict(TechAnswer)

query = "How do I reset my password?"
route = router(query=query).route

if route == "billing":
    result = billing(query=query)
elif route == "tech_support":
    result = tech(query=query)
else:
    result = dspy.Predict(dspy.Signature("Answer general questions.\nquery: str = InputField\nanswer: str = OutputField"))(query=query)

print(f"Route: {route}\nAnswer: {result.answer}")
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Router accuracy too low → cascade errors

**Wrong:**
```python
# Tiny model, no examples, complex routes — router guesses randomly
router = dspy.Predict(RouteQuery)
# If route is wrong, everything downstream fails silently
```

**Correct:**
```python
# Compile / bootstrap with labeled examples; start with a strong LM
router = dspy.Predict(RouteQuery)
# Evaluate router accuracy in isolation before wiring it into the full system
```

> **Debugging tip:** If downstream outputs are nonsensical, log the `route` field first. A low router accuracy is the cheapest diagnosis — fix routing before debugging specialists.

### Anti-pattern 2: No fallback / default route

**Wrong:**
```python
if route == "billing":
    result = billing(query=query)
elif route == "tech_support":
    result = tech(query=query)
# Missing else — inputs that don't match crash or get silently dropped
```

**Correct:**
```python
if route == "billing":
    result = billing(query=query)
elif route == "tech_support":
    result = tech(query=query)
else:
    result = general(query=query)  # Always provide a safe fallback
```

> **Debugging tip:** If some inputs return `None` or raise `KeyError` on dispatch, you are missing a default branch. Log unhandled routes and map them to a catch-all module.

### Anti-pattern 3: Routing based on keyword matching instead of semantic understanding

**Wrong:**
```python
# Brittle hard-coded rules
if "password" in query.lower():
    route = "tech_support"
elif "charge" in query.lower():
    route = "billing"
# Fails on paraphrasing like "I forgot my login credentials"
```

**Correct:**
```python
# Let the LM understand intent via the Router Signature
route = router(query=query).route
```

> **Debugging tip:** If users complain that "I can't log in" goes to billing, you are using surface-level heuristics. Move all routing logic into the DSPy Signature so the LM handles synonyms and paraphrases.

### Anti-pattern 4: Overly granular routes

**Wrong:**
```python
class OverlyGranular(dspy.Signature):
    """Route to one of 25 micro-categories."""
    query: str = dspy.InputField()
    route: str = dspy.OutputField()  # 25 options, each with almost no training data
```

**Correct:**
```python
class CoarseRouter(dspy.Signature):
    """Route to one of 3 departments."""
    query: str = dspy.InputField()
    route: str = dspy.OutputField(desc="One of: billing, tech_support, general")
# Add a second-level router inside tech_support if needed
```

> **Debugging tip:** If the router accuracy plateaus below 85%, merge rare categories into an "other" bucket or split routing into a hierarchical two-stage classifier.

## 5. 组合指南 (Composition)

The Router pattern unlocks multi-domain systems by delegating to focused specialists.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| Router + ReAct | Each route gets its own ReAct agent with domain-specific tools — e.g., billing agent calls Stripe API, tech agent calls docs search |
| Router + RAG | Different retrievers / knowledge bases per domain — e.g., route "policy" to HR vector store, "code" to engineering vector store |
| Router + Custom Module | Each route is a full pipeline (extract → validate → format) tailored to that domain |
| Router + Parallel | Route to multiple modules and merge outputs — e.g., "compare" queries need both product and competitor pipelines |

**Example: domain-specific RAG**
```python
class RouteQuestion(dspy.Signature):
    """Route the question to the correct knowledge base."""
    question: str = dspy.InputField()
    kb: str = dspy.OutputField(desc="One of: docs, api_ref, faq")

router = dspy.Predict(RouteQuestion)
kb = router(question="How do I authenticate?").kb

# Each retriever targets a different vector store
retrievers = {"docs": docs_rag, "api_ref": api_rag, "faq": faq_rag}
answer = retrievers.get(kb, fallback_rag)(question=question)
```

## 6. 进阶变体 (Advanced Variants)

- **Hierarchical routing:** A coarse router delegates to a department, then a fine-grained router inside that department picks the exact specialist. Reduces cognitive load per classification step.
- **Learned router with confidence thresholding:** Train the router with `dspy.BootstrapFewShot`, then reject low-confidence routes and escalate to a human or a general-purpose fallback.
- **Dynamic route discovery:** Instead of a fixed route list, let the router generate a route name and use a registry pattern (`MODULES[route]`) to look up the handler at runtime. New specialists can be registered without changing the router code.
