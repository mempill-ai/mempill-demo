"""
mempill_showcase.tools.rag_read_tool — RAGReadTool

Minimal in-memory RAG store read/search interface for bulk research text.

Paired with RAGWriteTool and InMemoryRAGStore (shared instance).

Two modes:
  - Exact lookup by doc_id: supply doc_id, omit query.
  - Keyword search: supply query string, optionally filter by namespace.

Returns JSON with matching documents (doc_id, text, metadata).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from mempill_showcase.tools.rag_write_tool import InMemoryRAGStore

log = logging.getLogger(__name__)


class RAGReadInput(BaseModel):
    namespace: str = Field(
        default="research",
        description="Logical collection / namespace to search within",
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="Exact document ID for direct lookup. If supplied, query is ignored.",
    )
    query: Optional[str] = Field(
        default=None,
        description="Keyword search string. Returns documents whose text contains this string.",
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return for keyword search",
    )


class RAGReadTool(BaseTool):
    """Read or search bulk research text from the in-memory RAG store.

    Two modes:
      - doc_id supplied → exact lookup for that document.
      - query supplied  → naïve keyword search within namespace.

    Returns JSON with a 'documents' list. Each document has doc_id, text, metadata.
    Note: the naïve in-memory search is substring-based. A real ChromaDB integration
    (planned for W4+) would replace this with semantic similarity search.
    """

    name: str = "rag_read"
    description: str = (
        "Read or search bulk research text from the in-memory RAG store. "
        "Supply doc_id for exact lookup, or query for keyword search. "
        "Returns JSON with a 'documents' list (doc_id, text, metadata). "
        "Pair with rag_write to store research results."
    )
    args_schema: Type[BaseModel] = RAGReadInput

    model_config = {"arbitrary_types_allowed": True}

    store: InMemoryRAGStore

    def _run(
        self,
        namespace: str = "research",
        doc_id: Optional[str] = None,
        query: Optional[str] = None,
        max_results: int = 5,
        **kwargs: Any,
    ) -> str:
        documents: List[Dict[str, Any]] = []

        if doc_id is not None:
            doc = self.store.read(namespace, doc_id)
            if doc is not None:
                documents = [{"doc_id": doc_id, **doc}]
        elif query is not None:
            documents = self.store.search(namespace, query, max_results=max_results)
        else:
            # List all doc_ids in namespace without returning full text
            ids = self.store.list_ids(namespace)
            return json.dumps({
                "namespace": namespace,
                "doc_ids": ids,
                "documents": [],
                "note": "Supply doc_id or query to retrieve content",
            })

        log.debug(
            "RAGReadTool: namespace=%s doc_id=%s query=%r → %d results",
            namespace, doc_id, query, len(documents),
        )

        return json.dumps({
            "namespace": namespace,
            "result_count": len(documents),
            "documents": documents,
        }, default=str)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("RAGReadTool does not support async")
