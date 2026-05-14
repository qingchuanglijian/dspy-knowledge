---
pattern_id: P22-SavingLoading
difficulty: beginner
source_tutorial: Saving and Loading from dspy.ai
api_modules: [Module.save, Module.load]
---

## 1. 核心思想 (Core Concept)

Persist compiled DSPy modules (optimized prompts + demonstrations) to disk so they can be reused across sessions, shared between environments, and deployed to production without re-compiling. A compiled module is an asset — treat it like a trained model checkpoint.

## 2. 类图与数据流 (Architecture)

```
Raw Module
    ↓
Compile (BootstrapFewShot / MIPROv2)
    ↓
Compiled Module
    ↓
save(path) → JSON checkpoint
    ↓
load(path) → Compiled Module restored
    ↓
Deploy
```

Key transformation: Compilation produces optimized demonstrations and prompt weights; saving serializes this state; loading restores it without re-running the optimizer.

## 3. 最小可运行代码 (MVP Code)

```python
import dspy

class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of the given text as positive, negative, or neutral."""
    text: str = dspy.InputField(desc="The text to classify")
    sentiment: str = dspy.OutputField(desc="One of: positive, negative, neutral")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 1. Create and compile the module
trainset = [
    dspy.Example(text="I love this!", sentiment="positive").with_inputs("text"),
    dspy.Example(text="Terrible experience.", sentiment="negative").with_inputs("text"),
]

classifier = dspy.Predict(SentimentClassifier)
compiled = dspy.BootstrapFewShot(metric=lambda ex, pred, trace=None: ex.sentiment == pred.sentiment).compile(
    classifier, train=trainset
)

# 2. Save the compiled module
compiled.save("./checkpoints/sentiment_classifier.json")

# 3. Load it in a new session (or different machine)
loaded = dspy.Predict(SentimentClassifier)
loaded.load("./checkpoints/sentiment_classifier.json")

# 4. Use immediately without recompiling
result = loaded(text="This is amazing!")
print(result.sentiment)
```

## 4. 常见反模式与诊断 (Anti-patterns)

### Anti-pattern 1: Saving uncompiled modules

**Wrong:**
```python
classifier = dspy.Predict(SentimentClassifier)
classifier.save("uncompiled.json")  # Wastes disk space; no demos or optimized prompts
```

**Correct:**
```python
compiled = teleprompter.compile(classifier, train=trainset)
compiled.save("compiled.json")  # Persists the optimized state
```

> **Debugging tip:** If loading a checkpoint doesn't improve quality over a fresh module, you probably saved before compiling. Always call `.save()` on the object returned by `.compile()`.

### Anti-pattern 2: Not versioning saved checkpoints

**Wrong:**
```python
compiled.save("model.json")  # Overwrites previous checkpoint
```

**Correct:**
```python
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
compiled.save(f"checkpoints/sentiment_{timestamp}.json")
```

> **Debugging tip:** Keep a metadata file (or naming convention) mapping checkpoints to compilation parameters and dataset versions so you can roll back when a new compilation degrades.

### Anti-pattern 3: Saving with hard-coded LM credentials in the checkpoint

**Wrong:**
```python
# If your LM client embeds API keys inside the module graph,
# the key leaks into the JSON checkpoint.
compiled.save("checkpoint.json")
```

**Correct:**
```python
# Configure LM separately at load time; don't embed credentials
lm = dspy.LM("openai/gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
dspy.configure(lm=lm)
loaded.load("checkpoint.json")
```

> **Debugging tip:** Inspect saved JSON for secrets before committing to version control. DSPy 3.x stores module state, not LM credentials, but custom modules may inadvertently leak them.

### Anti-pattern 4: Loading a checkpoint with a different DSPy version

**Wrong:**
```python
# Saved with DSPy 2.x, loaded in 3.x
loaded.load("old_checkpoint.json")  # May raise or silently fail
```

**Correct:**
```python
# Pin the DSPy version used during compilation
# requirements.txt: dspy==3.0.0
# Save metadata alongside the checkpoint
dspy_version = dspy.__version__
```

> **Debugging tip:** If loading produces missing keys or schema errors, the serialization format may have changed. Recompile with the current DSPy version or maintain a compatibility layer.

### Anti-pattern 5: Saving everything into one giant file

**Wrong:**
```python
ensemble.save("monolith.json")  # 500 MB, slow I/O, hard to diff
```

**Correct:**
```python
for i, member in enumerate(ensemble.members):
    member.save(f"ensemble/member_{i}.json")
```

> **Debugging tip:** Split large pipelines into per-module checkpoints. This speeds up I/O, makes inspection easier, and isolates failures when loading.

## 5. 组合指南 (Composition)

**Typical compositions:**

| Composition | When to use |
|-------------|-------------|
| Saving + MIPROv2 | Save expensive optimization results from MIPROv2 to avoid hours of recompilation |
| Saving + Ensemble | Save each ensemble member independently for modular deployment |
| Saving + Custom Module | Save complex pipeline state when sub-modules contain learned parameters |
| Saving + BootstrapFewShot | Save bootstrapped demonstrations so they survive across restarts |

**Example: save after MIPROv2**
```python
optimizer = dspy.MIPROv2(metric=my_metric, num_trials=20)
optimized = optimizer.compile(module, train=trainset, val=valset)
optimized.save("checkpoints/mipro_optimized.json")
```

## 6. 进阶变体 (Advanced Variants)

- **Incremental saves:** Track parameter hashes and save only modules whose state changed since the last checkpoint.
- **Checkpoint compression:** Gzip large JSON checkpoints to reduce disk usage and transfer time.
- **Cloud storage backends (S3, GCS):** Override save/load to stream checkpoints to object storage for distributed deployments.
- **Automatic checkpointing during long compilations:** Hook into the teleprompter to save intermediate states every N trials, enabling resumption after crashes.
- **Diff-based versioning between compilations:** Compare two saved checkpoints to audit which demonstrations or prompt weights changed.
