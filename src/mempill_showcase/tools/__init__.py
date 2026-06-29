"""
mempill_showcase.tools — LangChain BaseTool wrappers over the W1 memory core.

All tools are framework-agnostic: they hold a reference to a MempillAdapter
(or MemoryStore) and are suitable for both LangGraph nodes and CrewAI agents.

Exports:
  MempillRememberTool   — write a claim with succession encapsulation
  MempillRecallTool     — current-belief recall + bi-temporal query_at
  MempillAuditTool      — chronological audit ledger
  DateParserTool        — deterministic natural-date → ISO string (no LLM)
  RAGWriteTool          — write bulk research text to in-memory RAG store
  RAGReadTool           — read/search bulk research text from in-memory RAG store
  CalendarTool          — stub calendar event creator
  EmailDraftTool        — stub email draft composer
"""
from mempill_showcase.tools.mempill_remember_tool import MempillRememberTool
from mempill_showcase.tools.mempill_recall_tool import MempillRecallTool
from mempill_showcase.tools.mempill_audit_tool import MempillAuditTool
from mempill_showcase.tools.date_parser_tool import DateParserTool
from mempill_showcase.tools.rag_write_tool import RAGWriteTool
from mempill_showcase.tools.rag_read_tool import RAGReadTool
from mempill_showcase.tools.stub_tools import CalendarTool, EmailDraftTool

__all__ = [
    "MempillRememberTool",
    "MempillRecallTool",
    "MempillAuditTool",
    "DateParserTool",
    "RAGWriteTool",
    "RAGReadTool",
    "CalendarTool",
    "EmailDraftTool",
]
