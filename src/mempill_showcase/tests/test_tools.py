"""
mempill_showcase.tests.test_tools — W2 tool contract tests.

No API key, no LLM, no external services. Uses real in-memory mempill engine.

Covers:
  T1. MempillRememberTool — succession encapsulation:
        write Alice city=Austin (valid_from 2020), then city=NYC (valid_from 2025-02)
        via the tool → recall == NYC, query_at(valid_at=2021) == Austin.
        Proves the recall-then-close pattern works WITHOUT manual valid_until.

  T2. MempillRecallTool — returns status + display fields (Resolved, valid_from_display).

  T3. DateParserTool — deterministic parsing:
        "March 2020"        → "2020-03"     granularity=month
        "since June 15, 2026" → "2026-06-15" granularity=day
        "2023"              → "2023"        granularity=year

  T4. MempillAuditTool — returns audit entries after writes.

  T5. Canonical key enforcement — MempillRememberTool raises ValueError on off-key writes.
      (Wave B NOTE: stub_tools deleted; T7 CalendarTool/EmailDraftTool tests removed.)

  T6. RAGWriteTool + RAGReadTool — write bulk text, keyword-search retrieves it.
"""
from __future__ import annotations

import json

import pytest

from mempill_showcase.config.di import build_mempill_adapter
from mempill_showcase.scenarios.seed_data import AGENT_ID
from mempill_showcase.tools.date_parser_tool import DateParserTool
from mempill_showcase.tools.mempill_audit_tool import MempillAuditTool
from mempill_showcase.tools.mempill_recall_tool import MempillRecallTool
from mempill_showcase.tools.mempill_remember_tool import MempillRememberTool
from mempill_showcase.tools.rag_read_tool import RAGReadTool
from mempill_showcase.tools.rag_write_tool import InMemoryRAGStore, RAGWriteTool


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def adapter():
    """Fresh in-memory mempill adapter per test."""
    return build_mempill_adapter(in_memory=True)


@pytest.fixture()
def remember_tool(adapter):
    return MempillRememberTool(adapter=adapter)


@pytest.fixture()
def recall_tool(adapter):
    return MempillRecallTool(adapter=adapter)


@pytest.fixture()
def audit_tool(adapter):
    return MempillAuditTool(adapter=adapter)


@pytest.fixture()
def rag_store():
    return InMemoryRAGStore()


@pytest.fixture()
def rag_write_tool(rag_store):
    return RAGWriteTool(store=rag_store)


@pytest.fixture()
def rag_read_tool(rag_store):
    return RAGReadTool(store=rag_store)


@pytest.fixture()
def date_parser():
    return DateParserTool()


# ── T1: Succession encapsulation via MempillRememberTool ─────────────────────

class TestRememberToolSuccession:
    """T1 — remember tool encapsulates succession (recall-then-close pattern).

    Two writes: Austin (valid_from=2023-06) then NYC (valid_from=2025-02).
    After both writes:
      - recall(city) == "New York NY" (current belief, Resolved)
      - query_at(valid_at=2024-06-01) == "Austin TX" (bi-temporal — proves succession)

    The tool's recall-then-close pattern:
      1. Detects open-ended Austin incumbent (vt_start=2023-06, vt_end=open).
      2. Writes a bounded version of Austin (valid_until=2025-02) to close the window.
      3. Writes NYC challenger (valid_from=2025-02, open-ended).
      4. Calls engine.reconcile() (up to 3 passes) to fold non-overlapping windows.
      Result: both claims CommittedCheap, no HITL needed, bi-temporal queries correct.
    """

    def test_first_write_committed(self, remember_tool, adapter):
        """Write Austin city claim returns CommittedCheap."""
        raw = remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
            "confidence": 1.0,
            "provenance_channel": "UserAsserted",
        })
        result = json.loads(raw)
        assert result["disposition"] == "CommittedCheap", (
            f"First write should be CommittedCheap, got {result['disposition']!r}"
        )
        assert result["claim_ref"], "claim_ref must be non-empty"

    def test_second_write_nyc_succession_committed(self, remember_tool, adapter):
        """Write NYC (valid_from=2025-02) after Austin (2023-06) → CommittedCheap (succession).

        The recall-then-close pattern: tool closes Austin at 2025-02, then reconcile()
        folds Austin-bounded + NYC (non-overlapping) to CommittedCheap.
        """
        remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
            "confidence": 1.0,
            "provenance_channel": "UserAsserted",
        })
        raw2 = remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "New York NY",
            "valid_from": "2025-02",
            "confidence": 1.0,
            "provenance_channel": "UserAsserted",
        })
        result2 = json.loads(raw2)
        assert result2["disposition"] == "CommittedCheap", (
            f"NYC succession should be CommittedCheap (after recall-then-close), "
            f"got {result2['disposition']!r}. contested_with={result2.get('contested_with')}"
        )

    def test_current_recall_after_succession_is_nyc(self, remember_tool, recall_tool):
        """After Austin (2023-06) then NYC (2025-02) writes, current recall == 'New York NY'."""
        remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
            "provenance_channel": "UserAsserted",
        })
        remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "New York NY",
            "valid_from": "2025-02",
            "provenance_channel": "UserAsserted",
        })

        raw = recall_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
        })
        result = json.loads(raw)
        assert result["status"] == "Resolved", f"Expected Resolved, got {result['status']!r}"
        assert result["value"] == "New York NY", (
            f"Current recall should be 'New York NY', got {result['value']!r}"
        )
        assert not result["is_contested"], "Current recall must not be contested"

    def test_bitemoral_query_at_returns_austin_within_window(self, remember_tool, recall_tool):
        """query_at(valid_at=2024-06-01) returns Austin TX — proves succession bounding.

        Austin's valid window: 2023-06 to 2025-02 (bounded by the recall-then-close pattern).
        Querying at any date inside [2023-06, 2025-02) returns Austin.
        Querying at any date from 2025-02 onwards returns NYC.
        """
        remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
            "provenance_channel": "UserAsserted",
        })
        remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "New York NY",
            "valid_from": "2025-02",
            "provenance_channel": "UserAsserted",
        })

        # Query inside Austin's valid window
        raw = recall_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "valid_at": "2024-06-01T00:00:00Z",
        })
        result = json.loads(raw)
        assert result["status"] == "Resolved", (
            f"Bi-temporal query at 2024-06 should be Resolved, got {result['status']!r}"
        )
        assert result["value"] == "Austin TX", (
            f"Bi-temporal query at 2024-06 should return 'Austin TX', got {result['value']!r}"
        )


# ── T2: MempillRecallTool — status and display fields ────────────────────────

class TestRecallToolOutputFields:
    """T2 — recall tool returns structured result with status + display fields."""

    def test_recall_no_belief_returns_nobelief_status(self, recall_tool):
        """Recall on unknown subject/predicate returns NoBelief."""
        # bob-liu / dietary_restriction not in seed (empty store)
        raw = recall_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "bob-liu",
            "predicate": "city",
        })
        result = json.loads(raw)
        assert result["status"] == "NoBelief"
        assert result["value"] is None
        assert result["is_contested"] is False

    def test_recall_after_write_returns_display_fields(self, remember_tool, recall_tool):
        """Recall after write returns valid_from_display at correct granularity."""
        remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2023-06",
            "provenance_channel": "UserAsserted",
        })
        raw = recall_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
        })
        result = json.loads(raw)
        assert result["status"] == "Resolved"
        assert result["value"] == "Austin TX"
        # valid_from_display should be "2023-06" (month granularity)
        assert result["valid_from_display"] == "2023-06", (
            f"Expected '2023-06', got {result['valid_from_display']!r}"
        )

    def test_recall_all_required_keys_present(self, remember_tool, recall_tool):
        """Recall result always contains all expected keys."""
        remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "employer",
            "value": "Acme Corp",
            "valid_from": "2023",
        })
        raw = recall_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "employer",
        })
        result = json.loads(raw)
        required_keys = {
            "subject", "predicate", "value", "status",
            "is_contested", "is_resolved", "conf",
            "valid_from_display", "valid_until_display",
            "provenance", "claim_ref", "alternatives",
        }
        missing = required_keys - set(result.keys())
        assert not missing, f"Missing keys in recall result: {missing}"


# ── T3: DateParserTool — deterministic parsing ───────────────────────────────

class TestDateParserTool:
    """T3 — deterministic natural-date parsing (no LLM)."""

    def test_month_year_string(self, date_parser):
        """'March 2020' → iso_date='2020-03' granularity='month'."""
        raw = date_parser.invoke({"date_string": "March 2020"})
        result = json.loads(raw)
        assert "error" not in result, f"Parse error: {result.get('error')}"
        assert result["iso_date"] == "2020-03", f"Got {result['iso_date']!r}"
        assert result["granularity"] == "month"

    def test_since_prefix_with_full_date(self, date_parser):
        """'since June 15, 2026' → iso_date='2026-06-15' granularity='day'."""
        raw = date_parser.invoke({"date_string": "since June 15, 2026"})
        result = json.loads(raw)
        assert "error" not in result, f"Parse error: {result.get('error')}"
        assert result["iso_date"] == "2026-06-15", f"Got {result['iso_date']!r}"
        assert result["granularity"] == "day"

    def test_iso_year_only(self, date_parser):
        """'2020' → iso_date='2020' granularity='year'."""
        raw = date_parser.invoke({"date_string": "2020"})
        result = json.loads(raw)
        assert "error" not in result
        assert result["iso_date"] == "2020"
        assert result["granularity"] == "year"

    def test_iso_year_month(self, date_parser):
        """'2025-02' → iso_date='2025-02' granularity='month'."""
        raw = date_parser.invoke({"date_string": "2025-02"})
        result = json.loads(raw)
        assert "error" not in result
        assert result["iso_date"] == "2025-02"
        assert result["granularity"] == "month"

    def test_iso_full_date(self, date_parser):
        """'2023-06-01' → iso_date='2023-06-01' granularity='day'."""
        raw = date_parser.invoke({"date_string": "2023-06-01"})
        result = json.loads(raw)
        assert "error" not in result
        assert result["iso_date"] == "2023-06-01"
        assert result["granularity"] == "day"

    def test_since_month_year(self, date_parser):
        """'since June 2026' → iso_date='2026-06' granularity='month'."""
        raw = date_parser.invoke({"date_string": "since June 2026"})
        result = json.loads(raw)
        assert "error" not in result, f"Parse error: {result.get('error')}"
        assert result["iso_date"] == "2026-06", f"Got {result['iso_date']!r}"
        assert result["granularity"] == "month"

    def test_original_preserved(self, date_parser):
        """Result always echoes the original input."""
        raw = date_parser.invoke({"date_string": "March 2020"})
        result = json.loads(raw)
        assert result["original"] == "March 2020"


# ── T4: MempillAuditTool — returns entries after writes ──────────────────────

class TestAuditTool:
    """T4 — audit tool returns chronological ledger."""

    def test_audit_returns_entries_after_writes(self, remember_tool, audit_tool):
        """After 2 writes, audit returns at least 2 entries."""
        remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2020",
        })
        remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "New York NY",
            "valid_from": "2025-02",
        })

        raw = audit_tool.invoke({"agent_id": AGENT_ID, "limit": 50})
        result = json.loads(raw)
        assert result["agent_id"] == AGENT_ID
        assert result["entry_count"] >= 2, (
            f"Expected at least 2 audit entries, got {result['entry_count']}"
        )
        # Each entry must have the required fields
        for entry in result["entries"]:
            assert "claim_ref" in entry
            assert "event_kind" in entry
            assert "disposition" in entry
            assert "recorded_at" in entry

    def test_audit_empty_on_fresh_adapter(self, audit_tool):
        """Fresh adapter has no audit entries."""
        raw = audit_tool.invoke({"agent_id": AGENT_ID, "limit": 50})
        result = json.loads(raw)
        assert result["entry_count"] == 0

    def test_audit_claim_ref_filter(self, remember_tool, audit_tool):
        """Filtering by claim_ref returns only entries for that claim."""
        raw_write = remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2020",
        })
        write_result = json.loads(raw_write)
        target_ref = write_result["claim_ref"]

        # Write another claim
        remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "bob-liu",
            "predicate": "city",
            "value": "San Francisco",
            "valid_from": "2022",
        })

        raw = audit_tool.invoke({
            "agent_id": AGENT_ID,
            "limit": 50,
            "claim_ref": target_ref,
        })
        result = json.loads(raw)
        for entry in result["entries"]:
            assert entry["claim_ref"] == target_ref, (
                f"Filtered audit should only return entries for {target_ref!r}, "
                f"got {entry['claim_ref']!r}"
            )


# ── T5: Soft normalisation (replaces canonical key enforcement) ───────────────

class TestCanonicalKeyEnforcement:
    """T5 — MempillRememberTool uses soft normalisation (Wave B: no closed vocabulary).

    The canonical_keys guard was removed in Wave B. The tool now accepts any
    subject/predicate and soft-normalises (lowercase + separator).
    """

    def test_free_form_subject_normalised(self, remember_tool):
        """Free-form subject 'Alice Chen' is normalised to 'alice-chen' and stored."""
        raw = remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "Alice Chen",  # free-form — no longer raises
            "predicate": "city",
            "value": "Austin TX",
        })
        result = json.loads(raw)
        assert result["disposition"] in ("CommittedCheap", "Contested", "QueuedForAdjudication"), (
            f"Free-form subject should write without error, got {result['disposition']!r}"
        )

    def test_free_form_predicate_accepted(self, remember_tool):
        """Free-form predicate 'home address' is normalised to 'home_address' and stored."""
        raw = remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "home address",  # free-form — no longer raises
            "value": "123 Main St",
        })
        result = json.loads(raw)
        assert result["disposition"] in ("CommittedCheap", "Contested", "QueuedForAdjudication"), (
            f"Free-form predicate should write without error, got {result['disposition']!r}"
        )

    def test_canonical_keys_pass(self, remember_tool):
        """Canonical subject + predicate writes without error."""
        raw = remember_tool.invoke({
            "agent_id": AGENT_ID,
            "subject": "alice-chen",
            "predicate": "city",
            "value": "Austin TX",
            "valid_from": "2020",
        })
        result = json.loads(raw)
        assert result["disposition"] in ("CommittedCheap", "Contested", "QueuedForAdjudication")


# ── T6: RAGWriteTool + RAGReadTool ───────────────────────────────────────────

class TestRAGTools:
    """T6 — write bulk text to RAG store, search retrieves it."""

    def test_write_returns_doc_id(self, rag_write_tool):
        """RAGWriteTool returns doc_id and namespace."""
        raw = rag_write_tool.invoke({
            "text": "Marcus Webb named CTO at Acme Corp in January 2025.",
            "namespace": "research",
        })
        result = json.loads(raw)
        assert "doc_id" in result
        assert result["namespace"] == "research"
        assert result["characters"] > 0

    def test_keyword_search_finds_document(self, rag_write_tool, rag_read_tool):
        """Write then keyword-search returns the document."""
        rag_write_tool.invoke({
            "text": "Alice Chen relocated from Austin to New York in February 2025.",
            "namespace": "research",
            "doc_id": "alice-relocation-2025",
        })

        raw = rag_read_tool.invoke({
            "namespace": "research",
            "query": "Alice Chen",
        })
        result = json.loads(raw)
        assert result["result_count"] >= 1
        texts = [doc["text"] for doc in result["documents"]]
        assert any("Alice Chen" in t for t in texts), (
            f"Search should find 'Alice Chen' in results, got: {texts}"
        )

    def test_exact_doc_id_lookup(self, rag_write_tool, rag_read_tool):
        """Direct doc_id lookup returns the exact document."""
        raw_write = rag_write_tool.invoke({
            "text": "Acme Corp: new CTO Marcus Webb. Board approved Jan 15 2025.",
            "namespace": "research",
            "doc_id": "acme-cto-article",
        })
        write_result = json.loads(raw_write)
        doc_id = write_result["doc_id"]

        raw = rag_read_tool.invoke({
            "namespace": "research",
            "doc_id": doc_id,
        })
        result = json.loads(raw)
        assert result["result_count"] == 1
        assert result["documents"][0]["doc_id"] == doc_id
        assert "Marcus Webb" in result["documents"][0]["text"]

    def test_total_documents_grows(self, rag_store, rag_write_tool):
        """RAG store total_documents increases with each write."""
        initial = rag_store.total_documents()
        rag_write_tool.invoke({"text": "First doc", "namespace": "research"})
        rag_write_tool.invoke({"text": "Second doc", "namespace": "research"})
        assert rag_store.total_documents() == initial + 2


# T7 (CalendarTool / EmailDraftTool) removed — stub_tools.py deleted in Wave B.
