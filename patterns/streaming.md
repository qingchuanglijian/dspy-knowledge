---
pattern_id: P20-Streaming
difficulty: intermediate
source_tutorial: Streaming from dspy.ai
api_modules: []
---

## 1. 核心思想 (Core Concept)

Stream partial outputs from DSPy modules to the user in real-time, rather than waiting for the full response. Essential for interactive UIs where perceived latency matters more than actual latency. The challenge: DSPy's compile-time optimization assumes batch outputs, so streaming must work WITH compilation, not against it.

## 2. 类图与数据流 (Architecture)

```
User Query
    ↓
Compiled Module
    ↓
LM.generate(stream=True)
    ↓
Yield tokens/chunks
    ↓
Client renders
    ↓
Continue until complete
```

Key transformation: query → compiled prompt → token generator → incremental yield → UI render loop.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class ChatAnswer(dspy.Signature):
    """Answer the user's question concisely."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

def stream_answer(question: str):
    result = dspy.Predict(ChatAnswer)(question=question)
    for token in result.answer.split():
        yield token + " "

for chunk in stream_answer("What is DSPy?"):
    print(chunk, end="", flush=True)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Enabling streaming without token-level handling

**Wrong:**
```python
# Stream is created but never consumed incrementally
result = lm.generate(prompt, stream=True)
full = "".join(result)  # Buffers everything anyway
```

**Correct:**
```python
for chunk in lm.generate(prompt, stream=True):
    ui.render(chunk)  # Consume and render immediately
```

> **Debugging tip:** If latency feels the same after enabling streaming, check that you are yielding to the client in the same loop that receives tokens. Don't buffer into a list first.

### Anti-pattern 2: Streaming structured outputs without a parser

**Wrong:**
```python
for token in stream:
    ui.show(token)  # Partial JSON is invalid and confusing
```

**Correct:**
```python
buffer = ""
for token in stream:
    buffer += token
    if is_valid_json(buffer):
        ui.show(parse_json(buffer))
```

> **Debugging tip:** For structured outputs, either stream plain text and parse at the end, or use a streaming JSON parser. Never render raw partial JSON to users.

### Anti-pattern 3: Forgetting that compiled demos are still sent

**Wrong:**
```python
# Compiled predictor with 10 demos — prompt is huge
streamer = compiled_predictor  # Streaming hides latency behind prompt build
```

**Correct:**
```python
# Pre-compile and cache the prompt template; minimize demo count
streamer = compiled_predictor
```

> **Debugging tip:** Measure time-to-first-token (TTFT), not total generation time. If TTFT is high, reduce the number of compiled demos or use a lighter module for the streaming path.

### Anti-pattern 4: Using streaming for short responses

**Wrong:**
```python
# Adds complexity with no UX benefit
for chunk in stream_answer("Yes or no?"):
    print(chunk, end="")
```

**Correct:**
```python
# Batch is faster end-to-end for single-token answers
result = dspy.Predict(YesNo)(question="Yes or no?")
print(result.answer)
```

> **Debugging tip:** Only enable streaming when the expected output is longer than ~20 tokens or when the user is waiting interactively. Use batch mode for classification and extraction.

### Anti-pattern 5: Not separating "thinking" tokens from "output" tokens

**Wrong:**
```python
for token in stream:
    ui.show(token)  # User sees raw CoT reasoning mixed with answer
```

**Correct:**
```python
class Separated(dspy.Signature):
    reasoning: str = dspy.OutputField()
    answer: str = dspy.OutputField()

result = dspy.ChainOfThought(Separated)(...)
ui.show(result.answer)  # Only stream/render the final field
```

> **Debugging tip:** Use a two-field Signature or a Custom Module that strips rationale before yielding to the UI. Keep reasoning inspectable in logs, not visible in the stream.

## 5. 组合指南 (Composition)

Streaming is most effective when composed with patterns that maintain state or progressively refine output.

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| Streaming + ReAct | Stream the agent's tool-call reasoning as it happens |
| Streaming + Conversation History | Stream while appending each turn to a history buffer |
| Streaming + Refine | Stream a draft, then stream refinement deltas |
| Streaming + Custom Module | Orchestrate multiple stream sources (LM + tool results) |

**Example: streaming with history**
```python
history = []

def stream_chat(question: str):
    result = dspy.Predict(ChatAnswer)(question=question)
    for token in result.answer.split():
        yield token + " "
    history.append({"q": question, "a": result.answer})

for chunk in stream_chat("Hello!"):
    print(chunk, end="", flush=True)
```

## 6. 进阶变体 (Advanced Variants)

- **Server-sent events (SSE):** Wrap the generator in an SSE endpoint so a web client receives tokens over an HTTP stream.
- **WebSocket streaming:** Bidirectional streaming for interactive chat — send user input and receive token chunks over the same socket.
- **Progressive rendering:** Stream headers and section titles first, then fill in body paragraphs as they are generated.
- **Differential streaming:** Only send changed tokens between refinement iterations to reduce bandwidth.
- **Typing simulation:** Add small artificial delays between chunks to mimic human typing for a more natural UX.
