---
pattern_id: P21-Async
difficulty: intermediate
source_tutorial: Async execution from dspy.ai
api_modules: []
---

## 1. 核心思想 (Core Concept)

Execute DSPy predictions asynchronously to maximize throughput under high concurrency. Unlike parallel (which runs multiple predictions in one forward pass), async is about making the single forward() call non-blocking so the event loop can handle many requests concurrently. Essential for server workloads and real-time systems.

## 2. 类图与数据流 (Architecture)

```
Request arrives
    ↓
Event Loop
    ↓
async forward() dispatched
    ↓
LM API call (non-blocking)
    ↓
await response
    ↓
Parse
    ↓
Return to event loop
    ↓
Response sent
```

Key transformation: Python asyncio coroutines wrap DSPy module calls, yielding control to the event loop during I/O wait so many requests share a single thread.

## 3. 最小可运行代码 (MVP Code)

```python
import asyncio
import dspy

class Summarize(dspy.Signature):
    """Summarize the given text in one sentence."""
    text: str = dspy.InputField(desc="The text to summarize")
    summary: str = dspy.OutputField(desc="A one-sentence summary")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

summarizer = dspy.Predict(Summarize)

async def async_summarize(text: str) -> dspy.Prediction:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, summarizer, text)

async def main():
    texts = [
        "DSPy is a framework for programming language models.",
        "Async programming allows concurrent I/O operations.",
        "Python asyncio uses an event loop for concurrency.",
    ]
    tasks = [async_summarize(t) for t in texts]
    results = await asyncio.gather(*tasks)
    for text, result in zip(texts, results):
        print(f"Input: {text[:40]}... → {result.summary}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Running sync forward() in async context without executor

**Wrong:**
```python
async def bad(text: str):
    # Blocks the event loop — kills concurrency
    return summarizer(text=text)
```

**Correct:**
```python
async def good(text: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, summarizer, text)
```

> **Debugging tip:** If your async server stalls under load or coroutines never yield, check whether sync DSPy calls are blocking the event loop. Use `loop.run_in_executor` or an async-compatible LM client.

### Anti-pattern 2: Creating new event loops inside DSPy modules

**Wrong:**
```python
class BadModule(dspy.Module):
    def forward(self, text):
        loop = asyncio.new_event_loop()  # Nested loop!
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(some_async_call())
```

**Correct:**
```python
class GoodModule(dspy.Module):
    def forward(self, text):
        return summarizer(text=text)  # Keep sync; let caller handle async
```

> **Debugging tip:** `RuntimeError: Cannot run the event loop while another loop is running` means nested loops. Keep DSPy modules sync and wrap them at the application layer.

### Anti-pattern 3: Not batching async requests

**Wrong:**
```python
async def bad_batch(texts):
    results = []
    for t in texts:
        results.append(await async_summarize(t))  # Serial, one by one
    return results
```

**Correct:**
```python
async def good_batch(texts):
    tasks = [async_summarize(t) for t in texts]
    return await asyncio.gather(*tasks)  # Concurrent
```

> **Debugging tip:** If throughput doesn't scale with concurrency, you may be awaiting inside a loop. Use `asyncio.gather` or `asyncio.as_completed` to launch all requests concurrently.

### Anti-pattern 4: Ignoring backpressure

**Wrong:**
```python
async def bad_unbounded(texts):
    # Launches thousands of tasks at once
    return await asyncio.gather(*[async_summarize(t) for t in texts])
```

**Correct:**
```python
async def good_bounded(texts, limit: int = 10):
    sem = asyncio.Semaphore(limit)
    async def bounded(t):
        async with sem:
            return await async_summarize(t)
    return await asyncio.gather(*[bounded(t) for t in texts])
```

> **Debugging tip:** If you see rate-limit errors or timeout cascades, add an `asyncio.Semaphore` to cap concurrency. Start with your LM provider's RPM limit divided by 2.

### Anti-pattern 5: Using async for CPU-bound DSPy compilation

**Wrong:**
```python
async def bad_compile():
    # Compilation is CPU-bound; async provides zero benefit
    return await loop.run_in_executor(None, teleprompter.compile, module)
```

**Correct:**
```python
def good_compile():
    # Run compilation synchronously in a background thread/process
    return teleprompter.compile(module)
```

> **Debugging tip:** Compilation involves bootstrap reasoning and metric evaluation — CPU-bound work. Don't wrap it in async I/O; offload to a dedicated thread or process pool if needed.

## 5. 组合指南 (Composition)

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| Async + Parallel | Async concurrent execution of parallel branches — maximize GPU/TPU utilization |
| Async + Streaming | Async stream multiple responses to clients as they arrive |
| Async + Router | Async route incoming requests to multiple handlers without blocking |
| Async + BestOfN | Async generate N candidates concurrently, then score and select best |

**Example: async BestOfN**
```python
async def async_best_of_n(text: str, n: int = 3) -> dspy.Prediction:
    tasks = [async_summarize(text) for _ in range(n)]
    candidates = await asyncio.gather(*tasks)
    return max(candidates, key=lambda r: len(r.summary))
```

## 6. 进阶变体 (Advanced Variants)

- **Connection pooling for LM clients:** Reuse HTTP connections across async requests to reduce handshake overhead.
- **Adaptive concurrency limits:** Dynamically adjust semaphore size based on LM API rate-limit headers or latency feedback.
- **Circuit breaker for failing LM endpoints:** Pause requests to a failing endpoint and fall back to a backup model.
- **Request coalescing (deduplicate identical requests):** Cache in-flight promises so duplicate requests share the same response.
- **Async caching layer:** Store completed predictions in an async cache (e.g., Redis) to avoid redundant LM calls.
