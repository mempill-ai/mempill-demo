"""
console — mempill interactive console agent (TASK-6).

Entry points:
  python -m console           REPL (deterministic grammar)
  python -m console --llm     REPL with LLM extraction (requires ANTHROPIC_API_KEY)
  python -m console --scenario  auto-play 3-act story then REPL
  python -m console --selftest  assertion suite (no API key)
  python -m console --reset   delete persistent DB and exit
"""
