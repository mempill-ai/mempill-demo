"""
mempill_showcase.tools — LangChain BaseTool wrappers over the W1 memory core.

7 agent tools used by the ReAct ExecAssistant:
  RecallSubjectTool         — return ALL stored facts for a subject (cornerstone tool)
  RecallAtTool              — point-in-time valid-time recall
  RecallAsOfTool            — transaction-time as-of recall
  RememberFactTool          — free-form write tool: the agent's primary write path;
                              accepts any subject/predicate and uses succession
                              encapsulation (recall-then-close pattern)
  GetContestedTool          — surface competing beliefs for a contested predicate
  RequestAdjudicationTool   — HITL: interrupt the graph, get human verdict, resolve
  AuditTrailTool            — chronological audit ledger

Legacy tools (kept for existing tests and adapter layer — not in the agent's tool list):
  MempillRememberTool — write path with soft normalisation (lowercase + separator);
                        the closed-vocabulary canonical-key guard was removed in Wave B;
                        subject/predicate are now accepted open-world and normalised only
  MempillRecallTool   — current-belief recall + bi-temporal query_at
  MempillAuditTool    — audit ledger (old graph path)
  DateParserTool      — deterministic natural-date → ISO string (no LLM)
  RAGWriteTool        — write bulk research text to in-memory RAG store
  RAGReadTool         — read/search bulk research text from in-memory RAG store

Picking the right write tool:
  Use RememberFactTool (remember_fact) for the active ReAct agent — it handles
  succession encapsulation and is registered in the agent's tool list.
  MempillRememberTool is kept for legacy tests; it also soft-normalises but is
  NOT wired into the live agent graph.
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
