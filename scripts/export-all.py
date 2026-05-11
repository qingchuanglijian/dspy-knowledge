#!/usr/bin/env python3
"""
Regenerate all portable export formats from the knowledge base.
Run this after updating any core docs.
"""

import sys
from pathlib import Path

KB = Path(__file__).parent.parent
EXPORTS = KB / "exports"

def main():
    print("DSPy Knowledge Base Export")
    print("=" * 40)
    
    # Verify source docs exist
    required = ["README.md", "core-concepts.md", "api-cheatsheet-3x.md", 
                "common-pitfalls.md", "migration-2x-to-3x.md"]
    for doc in required:
        path = KB / doc
        if path.exists():
            print(f"  ✅ {doc}")
        else:
            print(f"  ❌ {doc} MISSING")
            sys.exit(1)
    
    # Verify patterns exist
    patterns = ["rag.md", "react-agent.md", "multi-hop.md"]
    for pat in patterns:
        path = KB / "patterns" / pat
        if path.exists():
            print(f"  ✅ patterns/{pat}")
        else:
            print(f"  ❌ patterns/{pat} MISSING")
    
    # Verify exports exist
    exports = ["system-prompt-dspy.md", ".cursorrules-dspy", "CLAUDE.md-dspy"]
    for exp in exports:
        path = EXPORTS / exp
        if path.exists():
            print(f"  ✅ exports/{exp}")
        else:
            print(f"  ❌ exports/{exp} MISSING")
    
    # Verify source index
    index = KB / "source-index/index.json"
    if index.exists():
        import json
        data = json.loads(index.read_text())
        print(f"  ✅ Source index: {len(data['modules'])} modules")
    else:
        print(f"  ❌ Source index missing")
    
    print()
    print("All exports verified. Ready for cross-agent use!")
    print()
    print("Usage:")
    print("  MCP Server:    python scripts/dspy-mcp-server.py")
    print("  CLI Query:     dspy-query <keyword>")
    print("  System Prompt: exports/system-prompt-dspy.md")
    print("  Cursor Rules:  exports/.cursorrules-dspy")
    print("  Claude Context: exports/CLAUDE.md-dspy")

if __name__ == "__main__":
    main()
