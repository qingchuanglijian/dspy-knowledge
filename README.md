# DSPy Knowledge Base

> 编程——而非提示词——语言模型。  
> Programming—not prompting—language models.

**A comprehensive, portable knowledge base for the DSPy framework (3.x).**

This repository is designed to be consumed by AI coding agents (Codex, Claude Code, Hermes, Cursor, etc.) to ensure they write correct, modern DSPy code.

---

## 🚀 Quick Start

### For Any LLM (System Prompt)

Copy [`exports/system-prompt-dspy.md`](exports/system-prompt-dspy.md) into your agent's system prompt:

```bash
cat exports/system-prompt-dspy.md | pbcopy  # macOS
cat exports/system-prompt-dspy.md | xclip -selection clipboard  # Linux
```

### For Cursor IDE

```bash
cp exports/.cursorrules-dspy /path/to/your/project/.cursorrules
```

### For Claude Code / Claude Projects

```bash
cp exports/CLAUDE.md-dspy /path/to/your/project/CLAUDE.md
```

### For MCP Clients (Claude Desktop, etc.)

```bash
# Start the MCP server
python scripts/dspy-mcp-server.py
```

Then configure in your MCP client:
```json
{
  "mcpServers": {
    "dspy-knowledge": {
      "command": "python",
      "args": ["/path/to/dspy-knowledge/scripts/dspy-mcp-server.py"]
    }
  }
}
```

---

## 📚 What's Inside

### Core Documentation

| Document | Purpose |
|----------|---------|
| [`core-concepts.md`](core-concepts.md) | 5 pillars: Signatures, Modules, LM, Optimizers, Evaluators |
| [`api-cheatsheet-3x.md`](api-cheatsheet-3x.md) | Quick 3.x syntax reference |
| [`migration-2x-to-3x.md`](migration-2x-to-3x.md) | Upgrade checklist from 2.x |
| [`common-pitfalls.md`](common-pitfalls.md) | 10 critical traps to avoid |

### Design Patterns

| Pattern | File | Complexity | Status |
|---------|------|------------|--------|
| Predict (P01) | [`patterns/predict.md`](patterns/predict.md) | Foundation | ✅ Batch 1 |
| ChainOfThought (P02) | [`patterns/chain-of-thought.md`](patterns/chain-of-thought.md) | Foundation | ✅ Batch 1 |
| ReAct Agent (P03) | [`patterns/react-agent.md`](patterns/react-agent.md) | Intermediate | ✅ Batch 1 |
| RAG (P04) | [`patterns/rag.md`](patterns/rag.md) | Beginner | ✅ Batch 1 |
| Multi-hop Reasoning (P05) | [`patterns/multi-hop.md`](patterns/multi-hop.md) | Advanced | ✅ Batch 1 |
| ProgramOfThought (P06) | [`patterns/program-of-thought.md`](patterns/program-of-thought.md) | Intermediate | ✅ Batch 1 |
| Custom Module (P07) | [`patterns/custom-module.md`](patterns/custom-module.md) | Foundation—Intermediate | ✅ Batch 1 |
| **BestOfN (P08)** | [`patterns/best-of-n.md`](patterns/best-of-n.md) | Intermediate | ✅ **Batch 2** |
| **Refine (P09)** | [`patterns/refine.md`](patterns/refine.md) | Intermediate | ✅ **Batch 2** |
| **Parallel (P10)** | [`patterns/parallel.md`](patterns/parallel.md) | Intermediate | ✅ **Batch 2** |
| **MultiChainComparison (P11)** | [`patterns/multi-chain-comparison.md`](patterns/multi-chain-comparison.md) | Advanced | ✅ **Batch 2** |
| **Router (P12)** | [`patterns/router.md`](patterns/router.md) | Intermediate | ✅ **Batch 2** |
| **Conversation History (P13)** | [`patterns/conversation-history.md`](patterns/conversation-history.md) | Intermediate | ✅ **Batch 2** |
| **BootstrapFewShot (P14)** | [`patterns/bootstrap-few-shot.md`](patterns/bootstrap-few-shot.md) | Intermediate | ✅ **Batch 3** |
| **MIPROv2 (P15)** | [`patterns/mipro-v2.md`](patterns/mipro-v2.md) | Advanced | ✅ **Batch 3** |
| **GEPA (P16)** | [`patterns/gepa.md`](patterns/gepa.md) | Advanced | ✅ **Batch 3** |
| **RL Optimization (P17)** | [`patterns/rl-optimization.md`](patterns/rl-optimization.md) | Expert | ✅ **Batch 3** |
| **Ensemble (P18)** | [`patterns/ensemble.md`](patterns/ensemble.md) | Intermediate | ✅ **Batch 3** |
| **MCP Integration (P19)** | [`patterns/mcp-integration.md`](patterns/mcp-integration.md) | Intermediate | ✅ **Batch 4** |
| **Streaming (P20)** | [`patterns/streaming.md`](patterns/streaming.md) | Intermediate | ✅ **Batch 4** |
| **Async (P21)** | [`patterns/async.md`](patterns/async.md) | Intermediate | ✅ **Batch 4** |
| **Saving & Loading (P22)** | [`patterns/saving-loading.md`](patterns/saving-loading.md) | Beginner | ✅ **Batch 4** |
| Pattern Catalog | [`patterns/pattern-catalog.md`](patterns/pattern-catalog.md) | All levels | ✅ Master index |

### Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `scripts/dspy-query` | Search DSPy source code index | `dspy-query ChainOfThought` |
| `scripts/dspy-mcp-server.py` | MCP protocol server for any client | `python scripts/dspy-mcp-server.py` |
| `scripts/export-all.py` | Validate all portable formats | `python scripts/export-all.py` |

### Source Code Index

[`source-index/index.json`](source-index/index.json) contains an indexed view of 119 DSPy modules (175 classes, 95 functions) for quick lookup.

**To refresh the index after a DSPy update:**
```bash
# 1. Download latest source
git clone --depth 1 https://github.com/stanfordnlp/dspy.git source-dspy

# 2. Regenerate index
python -c "
import json, ast
from pathlib import Path

# (see scripts/export-all.py for full reindex logic)
# Or use the dspy-query tool directly
"
```

---

## 🌐 Portable Export Formats

All formats are auto-generated from the same source of truth:

| Format | Target | File |
|--------|--------|------|
| **System Prompt** | Any LLM API | [`exports/system-prompt-dspy.md`](exports/system-prompt-dspy.md) |
| **Cursor Rules** | Cursor IDE | [`exports/.cursorrules-dspy`](exports/.cursorrules-dspy) |
| **Claude Context** | Claude Code/Projects | [`exports/CLAUDE.md-dspy`](exports/CLAUDE.md-dspy) |
| **MCP Server** | MCP-compatible clients | [`scripts/dspy-mcp-server.py`](scripts/dspy-mcp-server.py) |

---

## 🧪 Testing Your Agent

After integrating the knowledge base, test your agent with these prompts:

1. **"Write a RAG pipeline with DSPy 3.x"** → Should use `dspy.LM()`, `ChainOfThought`, type annotations
2. **"Create a Signature for entity extraction"** → Should NOT use `prefix=`, should use `desc=`
3. **"How do I migrate from DSPy 2.x to 3.x?"** → Should mention `dspy.LM()`, `prefix` deprecation, type annotations
4. **"Optimize my DSPy program"** → Should suggest `BootstrapFewShot` or `MIPROv2`

If the agent produces 2.x-style code (❌ `dspy.OpenAI`, ❌ `prefix=`, ❌ missing types), the knowledge base isn't loaded correctly.

---

## 🔄 Sync with Upstream

DSPy evolves quickly. Update weekly:

```bash
# Pull latest source
cd source-dspy && git pull origin main && cd ..

# Reindex
python scripts/reindex-source.py  # (see scripts/)

# Verify exports
python scripts/export-all.py
```

Or set up a cron job:
```bash
# Runs every Monday at 9:00 AM
0 9 * * 1 cd /path/to/dspy-knowledge && git pull && python scripts/export-all.py
```

---

## 📝 Contributing

1. Update source docs (`.md` files in root and `patterns/`)
2. Run `python scripts/export-all.py` to regenerate portable formats
3. Submit PR with before/after test results

---

## 📁 Related Skills (Hermes)

If you're using [Hermes Agent](https://hermes-agent.nousresearch.com/), these skills auto-load:

| Skill | Trigger |
|-------|---------|
| `dspy-structured-extraction` | Structured data extraction tasks |
| `dspy-rag-pipeline` | RAG/retrieval tasks |
| `dspy-agent-patterns` | Agent/ReAct tasks |
| `dspy-optimization` | Optimizer/evaluation tasks |

---

## 📖 License

MIT — See [LICENSE](LICENSE).

Source code index derived from [DSPy](https://github.com/stanfordnlp/dspy) (Apache 2.0).

---

**Maintained by:** [tokens](https://github.com/tokens)  
**Last synced:** 2026-05-11  
**DSPy version:** 3.x
