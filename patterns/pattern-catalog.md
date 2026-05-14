---
pattern_id: CATALOG
purpose: Master index — choose the right DSPy pattern for your task
difficulty_map: foundation | intermediate | advanced
last_updated: 2026-05-14
---

# DSPy Pattern Catalog

> **One-page cheat-sheet:** match your task to the right pattern, then jump to the full recipe.

---

## Batch 1 — Core Foundation (✅ Available)

| ID | Pattern | Difficulty | One-liner | When to use |
|---|---|---|---|---|
| P01 | [Predict](./predict.md) | foundation | Atomic input→output contract | Simple classification, extraction, or generation with a single step |
| P02 | [ChainOfThought](./chain-of-thought.md) | foundation | Force explicit reasoning before answering | Any task where the *process* matters: math, logic, multi-step reasoning |
| P03 | [ReAct Agent](./react-agent.md) | intermediate | Interleave reasoning + tool calls | The agent needs to search, calculate, or call external APIs to answer |
| P04 | [RAG](./rag.md) | beginner | Retrieve passages, then generate answer | Question-answering over a knowledge base or document collection |
| P05 | [Multi-Hop RAG](./multi-hop.md) | advanced | Iterative retrieval across multiple hops | Complex questions that require connecting information from multiple sources |
| P06 | [ProgramOfThought](./program-of-thought.md) | intermediate | Generate & execute code for exact computation | Math, data manipulation, structured computation where NL reasoning is imprecise |
| P07 | [Custom Module](./custom-module.md) | foundation-intermediate | Compose sub-modules into a new Module | Any non-trivial pipeline: multi-stage extraction, verification loops, branching |

---

## Batch 2 — Quality & Control (✅ Available)

| ID | Pattern | Difficulty | One-liner | When to use |
|---|---|---|---|---|
| P08 | [BestOfN](./best-of-n.md) | intermediate | Generate N candidates, pick the best | Quality-critical outputs where a single generation may be suboptimal |
| P09 | [Refine](./refine.md) | intermediate | Iteratively improve an output | Answers that benefit from polish: summaries, long-form text, code |
| P10 | [Parallel](./parallel.md) | intermediate | Execute multiple predictions concurrently | Need multiple independent analyses of the same input |
| P11 | [MultiChainComparison](./multi-chain-comparison.md) | advanced | Run multiple chains, compare or vote | High-stakes decisions where consensus or diversity helps |
| P12 | [Router / Delegation](./router.md) | intermediate | Route input to the best sub-module | Input has distinct domains (math vs. legal vs. medical) |
| P13 | [Conversation History](./conversation-history.md) | intermediate | Manage multi-turn state | Chatbots, iterative clarification, conversational agents |

---

## Batch 3 — Optimization & RL (✅ Available)

| ID | Pattern | Difficulty | One-liner | When to use |
|---|---|---|---|---|
| P14 | [BootstrapFewShot](./bootstrap-few-shot.md) | intermediate | Auto-select optimal few-shot demos | You have 20–200 labeled examples and want better prompts |
| P15 | [MIPROv2](./mipro-v2.md) | advanced | Meta-prompt optimization via bayesian search | Production quality, 200+ examples, hours of compile time |
| P16 | [GEPA](./gepa.md) | advanced | Reflective prompt evolution | Rapidly evolving domains where manual prompt tuning is too slow |
| P17 | [RL Optimization](./rl-optimization.md) | expert | Reinforcement learning for agent strategy | Agent behavior tuning with delayed rewards |
| P18 | [Ensemble](./ensemble.md) | intermediate | Combine multiple programs | You have several approaches and want robustness via voting |

---

## Batch 4 — Engineering & Deployment (✅ Available)

| ID | Pattern | Difficulty | One-liner | When to use |
|---|---|---|---|---|
| P19 | [MCP Integration](./mcp-integration.md) | intermediate | Standardize tool access via Model Context Protocol | Connecting DSPy to external tool ecosystems |
| P20 | [Streaming](./streaming.md) | intermediate | Stream partial outputs to the user | Real-time UIs where users shouldn't wait for the full response |
| P21 | [Async](./async.md) | intermediate | High-concurrency async execution | Server workloads with many concurrent requests |
| P22 | [Saving & Loading](./saving-loading.md) | beginner | Persist compiled module state | Deploying optimized programs to production |

---

## 🔍 Decision Tree: "Which Pattern Do I Need?"

```
Start
├─── Do you need to retrieve information first?
│   ├─── Yes → Is one retrieval enough?
│   │   ├─── Yes → [P04 RAG]
│   │   └─── No  → [P05 Multi-Hop RAG]
│   └─── No
├─── Do you need to use tools (APIs, search, calculator)?
│   ├─── Yes → [P03 ReAct Agent]
│   └─── No
├─── Is the task exact computation (math, data manipulation)?
│   ├─── Yes → [P06 ProgramOfThought]
│   └─── No
├─── Is quality critical and a single answer may be wrong?
│   ├─── Yes → Need diverse approaches or just more samples?
│   │   ├─── Diverse → [P11 MultiChainComparison]
│   │   └─── More samples → [P08 BestOfN]
│   └─── No
├─── Does the answer need iterative polish / refinement?
│   ├─── Yes → [P09 Refine]
│   └─── No
├─── Do you need multiple independent analyses?
│   ├─── Yes → [P10 Parallel]
│   └─── No
├─── Does the answer require explicit step-by-step reasoning?
│   ├─── Yes → [P02 ChainOfThought]
│   └─── No
├─── Is input from distinct domains needing different handlers?
│   ├─── Yes → [P12 Router]
│   └─── No
├─── Is this a multi-turn conversation?
│   ├─── Yes → [P13 Conversation History]
│   └─── No
├─── Is the pipeline multi-stage or branching?
│   ├─── Yes → [P07 Custom Module] + compose with P01–P13
│   └─── No
└─── Simple single-step task → [P01 Predict]
```

---

## 🔗 Composition Matrix

> "Patterns don't live in isolation — they compose."

| Primary Pattern | Common Companions | Resulting Capability |
|---|---|---|
| P04 RAG | + P02 CoT | Reasoning-aware retrieval |
| P04 RAG | + P05 Multi-Hop | Deep research over documents |
| P04 RAG | + P09 Refine | Iteratively improved answers |
| P03 ReAct | + P13 Conversation History | Persistent multi-turn agent |
| P03 ReAct | + P12 Router | Domain-specialized agent fleet |
| P02 CoT | + P01 Predict | Two-stage: reason, then decide |
| P07 Custom Module | + any P01–P13 | Production pipeline |
| P06 PoT | + P09 Refine | Self-correcting computation |
| P08 BestOfN | + P11 MultiChainComparison | Quality-optimized generation |
| P10 Parallel | + P11 MultiChainComparison | Concurrent diverse analysis |
| P12 Router | + P08 BestOfN | Domain-aware quality boost |
| P13 Conversation History | + P04 RAG | Contextual retrieval over chat |
| P14 BootstrapFewShot | + P07 Custom Module | Optimized multi-stage pipeline |
| P15 MIPROv2 | + P18 Ensemble | Optimized + robust meta-system |
| P16 GEPA | + P14 BootstrapFewShot | Co-evolved prompts and demos |
| P17 RL Optimization | + P03 ReAct | Self-improving agent |
| P18 Ensemble | + P14 BootstrapFewShot | Diverse, independently optimized members |
| P19 MCP Integration | + P03 ReAct | Multi-step agent with standardized tools |
| P20 Streaming | + P13 Conversation History | Real-time chat with memory |
| P21 Async | + P10 Parallel | Maximum throughput concurrent execution |
| P22 Saving & Loading | + P15 MIPROv2 | Deploy expensive optimizations once |

---

## 📝 How to Read a Pattern Doc

Each pattern document follows this structure:

1. **Core Concept** — Why this pattern exists and what problem it solves
2. **Architecture** — Data flow diagram
3. **MVP Code** — Copy-paste runnable example (DSPy 3.x syntax)
4. **Anti-patterns** — 3+ mistakes agents typically make + correct alternatives
5. **Composition Guide** — How to combine with other patterns
6. **Advanced Variants** — Production extensions

---

## 📚 Related Docs

- [Core Concepts](../core-concepts.md) — DSPy 5 pillars: Signatures, Modules, LM, Optimizers, Evaluators
- [API Cheatsheet](../api-cheatsheet-3x.md) — Quick syntax reference
- [Common Pitfalls](../common-pitfalls.md) — 10 critical traps across all patterns

---

**Maintained by:** default profile  
**Last updated:** 2026-05-14  
**Coverage:** 22 / 22 patterns documented (✅ ALL BATCHES COMPLETE)
