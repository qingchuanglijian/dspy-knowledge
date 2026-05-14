---
pattern_id: P13-ConversationHistory
difficulty: intermediate
source_tutorial: Managing Conversation History from dspy.ai
api_modules: []
---

## 1. 核心思想 (Core Concept)

Maintain and inject multi-turn dialogue context into DSPy predictions so the model "remembers" prior exchanges. Unlike stateless predictors, conversation-aware modules accumulate history as a structured message list. The core philosophy is that many real-world tasks — clarification, follow-ups, and reference resolution — require context that a single query cannot capture. By treating history as explicit typed input, you keep the module deterministic and testable while giving the LM the conversational context it needs to produce coherent, grounded responses.

## 2. 类图与数据流 (Architecture)

```
User Turn N
    ↓
Append to history: list[dict]  (role / content / timestamp)
    ↓
Signature with history: list[dict] as InputField
    ↓
Custom Module.forward(history, latest_query)
    ↓
LM prompt: system instruction + formatted history + latest query
    ↓
Output: assistant response
    ↓
Append assistant response to history
    ↓
Wait for next user turn
```

Key transformation: Raw turns → structured history list → DSPy Signature input → LM sees full dialogue context → generated response enriches history for the next turn.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class ChatTurn(dspy.Signature):
    """Answer the user's latest message using the conversation history."""
    history: list[dict] = dspy.InputField(desc="List of messages with 'role' and 'content'")
    query: str = dspy.InputField(desc="The user's latest message")
    response: str = dspy.OutputField(desc="The assistant's reply")

class ChatBot(dspy.Module):
    def __init__(self):
        self.respond = dspy.Predict(ChatTurn)
        self.history: list[dict] = []

    def forward(self, query: str) -> str:
        result = self.respond(history=self.history, query=query)
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": result.response})
        return result.response

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

bot = ChatBot()
print(bot.forward("What is DSPy?"))
print(bot.forward("How does it differ from LangChain?"))
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Treating history as raw string concatenation

**Wrong:**
```python
history_str = ""
for turn in history:
    history_str += turn["content"] + "\n"
# No turn boundaries; model confuses user vs assistant voice
```

**Correct:**
```python
# Pass structured messages and let DSPy / the LM format them
history: list[dict] = [{"role": "user", "content": "Hi"},
                       {"role": "assistant", "content": "Hello!"}]
result = self.respond(history=history, query=query)
```

> **Debugging tip:** If the model starts replying to itself or attributes its own previous statements to the user, you have lost role boundaries. Use a structured `list[dict]` with explicit `role` keys.

### Anti-pattern 2: Never truncating or summarizing long history

**Wrong:**
```python
# History grows unbounded — eventually exceeds context window
self.history.append({"role": "user", "content": query})
self.history.append({"role": "assistant", "content": result.response})
```

**Correct:**
```python
MAX_TURNS = 10
self.history = self.history[-MAX_TURNS:]
# Or summarize old turns and replace them with a compact memory
```

> **Debugging tip:** If response quality degrades after many turns or you see context-overflow errors, implement a sliding window or summarization layer. Measure token count before each call.

### Anti-pattern 3: Including system prompts or instructions in the history accumulator

**Wrong:**
```python
self.history = [{"role": "system", "content": "You are a helpful assistant."}]
# System instructions leak into the history and get passed back as user input
```

**Correct:**
```python
# Keep system prompts outside history; history is only for dialogue turns
self.history = []  # user + assistant only
# System behavior belongs in the Signature docstring, not the message list
```

> **Debugging tip:** If the model starts repeating system instructions back to you or treating them as user messages, you have poisoned the history. Separate meta-instructions from conversational state.

### Anti-pattern 4: Not handling history immutability

**Wrong:**
```python
def forward(self, history, query):
    history.append({"role": "user", "content": query})  # Mutates caller's list!
    result = self.respond(history=history, query=query)
```

**Correct:**
```python
def forward(self, history, query):
    local_history = list(history)  # Shallow copy
    local_history.append({"role": "user", "content": query})
    result = self.respond(history=local_history, query=query)
```

> **Debugging tip:** If parallel tests or multi-user deployments show "bleed" between conversations, you are mutating shared history in-place. Always copy before appending, or store history per instance.

## 5. 组合指南 (Composition)

Conversation history turns any module into a stateful agent that can clarify, correct, and build on prior context.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| Conversation History + ReAct | Multi-turn agent with tool use — history lets the agent remember which tools it already called and what they returned |
| Conversation History + RAG | Retrieve based on the full conversation context, not just the last query — e.g., "Tell me more about that" requires knowing what "that" refers to |
| Conversation History + Custom Module | Stateful pipeline with memory — e.g., interview bot that tracks which topics were already covered |
| Conversation History + Router | Route based on conversation phase — e.g., greeting → intake → diagnosis → wrap-up, where the router sees the full dialogue |

**Example: contextual RAG follow-up**
```python
class ContextualRAG(dspy.Signature):
    """Answer using retrieved documents and conversation history."""
    history: list[dict] = dspy.InputField()
    query: str = dspy.InputField()
    context: str = dspy.InputField(desc="Retrieved documents")
    answer: str = dspy.OutputField()

rag = dspy.Predict(ContextualRAG)
history = [{"role": "user", "content": "What is DSPy?"},
           {"role": "assistant", "content": "DSPy is a framework for programming language models."}]
query = "What are its main modules?"
context = retriever("DSPy main modules").passages
result = rag(history=history, query=query, context=context)
```

## 6. 进阶变体 (Advanced Variants)

- **Sliding window truncation:** Keep only the last N turns; simple, predictable, but loses long-range context.
- **Summarization-based compression:** Periodically run a `Predict` module to summarize old turns into a compact "memory" block, then drop the raw turns. Preserves salient information with bounded context.
- **Selective memory:** Tag important turns during the conversation and always include them; evict routine chit-chat. Use a small classifier to score turn importance.
- **Structured message format:** Enrich messages with `timestamp`, `turn_id`, or `intent` fields. Enables time-aware reasoning ("What did I ask 10 minutes ago?") and explicit reference resolution.
