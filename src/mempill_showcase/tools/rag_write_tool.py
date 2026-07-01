"""
mempill_showcase.tools.rag_write_tool — RAGWriteTool

Minimal in-memory RAG store write interface for bulk research text.

Design:
  - Bulk research content (articles, transcripts, email threads) goes HERE,
    NOT into mempill. Only distilled atomic claims move from this store to mempill.
  - W2 implementation: simple in-memory dict keyed by (namespace, doc_id).
    Real ChromaDB integration is optional in later waves (W4+).
  - Multiple tool instances can share the same in-memory store via a shared
    InMemoryRAGStore instance passed at construction.

Each document stored has:
  - doc_id: caller-assigned identifier (e.g. UUID or slug)
  - namespace: logical collection (e.g. "research", "transcripts")
  - text: the bulk text content
  - metadata: optional dict for tags, timestamps, source URLs, etc.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class InMemoryRAGStore:
    """Shared in-memory document store. One instance per session."""

    def __init__(self) -> None:
        # {namespace: {doc_id: {"text": str, "metadata": dict}}}
        self._store: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def write(
        self,
        namespace: str,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write a document. Returns the doc_id used."""
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        ns = self._store.setdefault(namespace, {})
        ns[doc_id] = {"text": text, "metadata": metadata or {}}
        return doc_id

    def read(self, namespace: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return a document dict or None if not found."""
        return self._store.get(namespace, {}).get(doc_id)

    def search(self, namespace: str, query: str, max_results: int = 5) -> list[Dict[str, Any]]:
        """Naïve keyword search: returns documents whose text contains the query string."""
        ns = self._store.get(namespace, {})
        results = []
        query_lower = query.lower()
        for doc_id, doc in ns.items():
            if query_lower in doc["text"].lower():
                results.append({"doc_id": doc_id, **doc})
            if len(results) >= max_results:
                break
        return results

    def list_ids(self, namespace: str) -> list[str]:
        """List all doc_ids in a namespace."""
        return list(self._store.get(namespace, {}).keys())

    def total_documents(self) -> int:
        """Total documents across all namespaces."""
        return sum(len(ns) for ns in self._store.values())


class RAGWriteInput(BaseModel):
    text: str = Field(description="Bulk text content to store (article, transcript, notes, etc.)")
    namespace: str = Field(
        default="research",
        description="Logical collection / namespace (e.g. 'research', 'transcripts')",
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="Optional document ID. Auto-generated UUID if not supplied.",
    )
    metadata: Optional[str] = Field(
        default=None,
        description="Optional JSON string of metadata tags (source URL, date, etc.)",
    )


class RAGWriteTool(BaseTool):
    """Write bulk research text to the in-memory RAG store.

    Use this for raw research output, article text, transcripts, and any
    bulk prose content. Do NOT write this content to mempill — only distilled
    atomic claims (subject / predicate / value / valid_from) should go to mempill.

    Returns JSON with doc_id and namespace so the caller can reference the
    document in subsequent RAGReadTool calls.
    """

    name: str = "rag_write"
    description: str = (
        "Store bulk research text (article, transcript, notes) in the in-memory RAG store. "
        "Supply text and optional namespace / doc_id / metadata. "
        "Returns JSON with doc_id and namespace. "
        "Use rag_read to retrieve or search stored documents."
    )
    args_schema: Type[BaseModel] = RAGWriteInput

    model_config = {"arbitrary_types_allowed": True}

    store: InMemoryRAGStore

    def _run(
        self,
        text: str,
        namespace: str = "research",
        doc_id: Optional[str] = None,
        metadata: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        meta_dict: Dict[str, Any] = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                # Treat as a plain string tag
                meta_dict = {"raw": metadata}

        stored_id = self.store.write(namespace, text, doc_id=doc_id, metadata=meta_dict)
        log.debug("RAGWriteTool: wrote doc_id=%s namespace=%s len=%d", stored_id, namespace, len(text))

        return json.dumps({
            "doc_id": stored_id,
            "namespace": namespace,
            "characters": len(text),
        })

    async def _arun(
        self,
        text: str,
        namespace: str = "research",
        doc_id: Optional[str] = None,
        metadata: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        return self._run(
            text=text,
            namespace=namespace,
            doc_id=doc_id,
            metadata=metadata,
            **kwargs,
        )
