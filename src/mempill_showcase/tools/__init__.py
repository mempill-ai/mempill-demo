"""
mempill_showcase.tools — LangChain BaseTool wrappers over the W1 memory core.

7 agent tools used by the ReAct ExecAssistant:
  RecallSubjectTool         — return ALL stored facts for a subject (cornerstone tool)
  RecallAtTool              — point-in-time valid-time recall
  RecallAsOfTool            — transaction-time as-of recall
  RememberFactTool          — write a free-form fact with succession encapsulation
  GetContestedTool          — surface competing beliefs for a contested predicate
  RequestAdjudicationTool   — HITL: interrupt the graph, get human verdict, resolve
  AuditTrailTool            — chronological audit ledger

Legacy tools (kept for existing tests and adapter layer — not in the agent's tool list):
  MempillRememberTool — write with canonical-key guard (old graph path)
  MempillRecallTool   — current-belief recall + bi-temporal query_at
  MempillAuditTool    — audit ledger (old graph path)
  DateParserTool      — deterministic natural-date → ISO string (no LLM)
  RAGWriteTool        — write bulk research text to in-memory RAG store
  RAGReadTool         — read/search bulk research text from in-memory RAG store
"""
from mempill_showcase.tools.recall_subject_tool import RecallSubjectTool
from mempill_showcase.tools.recall_at_tool import RecallAtTool
from mempill_showcase.tools.recall_as_of_tool import RecallAsOfTool
from mempill_showcase.tools.remember_fact_tool import RememberFactTool
from mempill_showcase.tools.get_contested_tool import GetContestedTool
from mempill_showcase.tools.request_adjudication_tool import RequestAdjudicationTool
from mempill_showcase.tools.audit_trail_tool import AuditTrailTool
from mempill_showcase.tools.mempill_remember_tool import MempillRememberTool
from mempill_showcase.tools.mempill_recall_tool import MempillRecallTool
from mempill_showcase.tools.mempill_audit_tool import MempillAuditTool
from mempill_showcase.tools.date_parser_tool import DateParserTool
from mempill_showcase.tools.rag_write_tool import RAGWriteTool
from mempill_showcase.tools.rag_read_tool import RAGReadTool

__all__ = [
    # Agent tools (7)
    "RecallSubjectTool",
    "RecallAtTool",
    "RecallAsOfTool",
    "RememberFactTool",
    "GetContestedTool",
    "RequestAdjudicationTool",
    "AuditTrailTool",
    # Legacy tools
    "MempillRememberTool",
    "MempillRecallTool",
    "MempillAuditTool",
    "DateParserTool",
    "RAGWriteTool",
    "RAGReadTool",
]
