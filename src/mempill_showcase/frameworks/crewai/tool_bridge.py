"""
mempill_showcase.frameworks.crewai.tool_bridge — LangChain BaseTool → CrewAI adapter.

W2 tools are LangChain `BaseTool` subclasses.  CrewAI agents accept instances of
`crewai.tools.base_tool.BaseTool`.  This module bridges the two without pulling in
heavy LangChain Community dependencies.

Approach: for each LangChain tool, dynamically create a subclass of CrewAI's
`BaseTool` that delegates `_run()` to the LangChain tool's `_run()`.  The
`args_schema` is forwarded from the LangChain tool so CrewAI gets the same field
descriptions the LangChain tool already specifies.

Usage:
    from mempill_showcase.frameworks.crewai.tool_bridge import as_crewai_tool
    crewai_remember = as_crewai_tool(remember_tool)

The resulting object is a `crewai.tools.base_tool.BaseTool` subclass instance that:
  - carries the same name and description as the LangChain tool
  - delegates _run(**kwargs) to lc_tool._run(**kwargs)
  - validates inputs via the LangChain tool's Pydantic args_schema
  - is accepted by crewai.Agent(tools=[...])

No LLM or external call is made at bridge time; the bridge is pure Python glue
and can be tested without an API key.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Type

log = logging.getLogger(__name__)

# Cache to avoid recreating the same dynamic class multiple times
_BRIDGE_CLASS_CACHE: dict[str, type] = {}


def as_crewai_tool(lc_tool: Any) -> Any:
    """Wrap a LangChain BaseTool as a crewai.tools.base_tool.BaseTool instance.

    Args:
        lc_tool: any instance of langchain_core.tools.BaseTool

    Returns:
        An instance of a dynamically-created crewai.tools.base_tool.BaseTool
        subclass that delegates _run(**kwargs) to lc_tool._run(**kwargs).

    The delegation is transparent:
      - CrewAI supplies kwargs from the parsed args_schema.
      - The wrapper calls lc_tool._run(**kwargs) and returns the result string.
      - No LLM or external call is made at bridge time.
    """
    from crewai.tools.base_tool import BaseTool as CrewBaseTool

    # Extract metadata from the LangChain tool
    tool_name: str = getattr(lc_tool, "name", lc_tool.__class__.__name__)
    tool_description: str = getattr(lc_tool, "description", "")
    args_schema: Optional[Type] = getattr(lc_tool, "args_schema", None)

    # Look up or create the dynamic subclass
    cache_key = f"{lc_tool.__class__.__qualname__}_{id(lc_tool)}"
    if cache_key not in _BRIDGE_CLASS_CACHE:
        # Capture the LC tool in a closure
        _lc = lc_tool
        _schema = args_schema

        # Pydantic v2 requires annotations for all field overrides.
        # Build the class using a proper class statement via exec so that
        # Python 3.14's annotation mechanism correctly populates __annotate_func__.
        # We store the LC tool reference in a closure via class variable _lc_impl.

        _captured_lc = _lc
        _captured_schema = _schema

        # We can't use `type()` directly with pydantic v2 + Python 3.14 annotation rules.
        # Instead, use a class factory function so we get proper annotation metadata.
        def _make_class(name: str, description: str, schema: Optional[type], lc: Any) -> type:
            """Create a properly-annotated CrewAI BaseTool subclass via exec."""
            ns: dict[str, Any] = {
                "CrewBaseTool": CrewBaseTool,
                "_lc_impl": lc,
                "_tool_name": name,
                "_tool_desc": description,
                "Optional": Optional,
                "Type": Type,
                "Any": Any,
            }
            if schema is not None:
                ns["_schema"] = schema
                schema_line = "    args_schema: Type[Any] = _schema"
            else:
                schema_line = ""

            # Use exec to create a properly-annotated class
            code = (
                f"class BridgedTool(CrewBaseTool):\n"
                f"    name: str = _tool_name\n"
                f"    description: str = _tool_desc\n"
                f"{schema_line}\n"
                f"    def _run(self, **kwargs: Any) -> Any:\n"
                f"        return _lc_impl._run(**kwargs)\n"
            )
            exec(code, ns)
            return ns["BridgedTool"]

        DynamicBridgedTool = _make_class(tool_name, tool_description, _captured_schema, _captured_lc)
        # Force pydantic to rebuild so forward references resolve
        DynamicBridgedTool.model_rebuild(force=True, _types_namespace={"Type": Type, "Any": Any, "Optional": Optional})
        _BRIDGE_CLASS_CACHE[cache_key] = DynamicBridgedTool

    DynClass = _BRIDGE_CLASS_CACHE[cache_key]

    # Instantiate with fixed name/description
    instance = DynClass(name=tool_name, description=tool_description)
    log.debug("as_crewai_tool: bridged %s → CrewAI BaseTool (%s)", tool_name, DynClass.__name__)
    return instance
