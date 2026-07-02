"""
Classify stage: one Claude call per scene, with prompt caching.

The big (synthetic) taxonomy sits in a cached system block and the scene text is
the uncached user message.

To fake C independent cache scopes from one API key, a small per-scope marker is
put at the very start of the cached block. The taxonomy after it is identical,
but the cache is keyed on the exact prefix, so every marker gets its own cache
entry. With more scopes each one sees less traffic, its prefix expires between
hits, and p_hit drops.

classify_scene returns (label, usage); usage has the raw token counts the cost
formula needs.
"""
import json
import os
import pathlib
import time

_TAXONOMY_PATH = pathlib.Path(__file__).with_name("taxonomy_synth.json")
_TAXONOMY_JSON = None  # minified once per process


def _taxonomy() -> str:
    global _TAXONOMY_JSON
    if _TAXONOMY_JSON is None:
        with open(_TAXONOMY_PATH) as f:
            _TAXONOMY_JSON = json.dumps(json.load(f), separators=(",", ":"))
    return _TAXONOMY_JSON


# $ per token, from the published $/MTok prices. Check these against the current
# price list before trusting the cost numbers.
PRICING = {
    "claude-haiku-4-5": {"in": 1.00e-6, "out": 5.00e-6},
    "claude-sonnet-5":  {"in": 3.00e-6, "out": 15.00e-6},
}
P_CACHE_WRITE_MULT = 1.25   # cache writes cost ~1.25x input (5 min TTL)
P_CACHE_READ_MULT = 0.10    # cache reads cost ~0.10x input


def cost_from_usage(usage: dict, model: str) -> float:
    p = PRICING[model]
    return (
        usage["input_tokens"] * p["in"]
        + usage["cache_creation_input_tokens"] * p["in"] * P_CACHE_WRITE_MULT
        + usage["cache_read_input_tokens"] * p["in"] * P_CACHE_READ_MULT
        + usage["output_tokens"] * p["out"]
    )


_SYSTEM_PREAMBLE = (
    "You are a scene-segment matcher. You receive a synthetic taxonomy of "
    "narrative categories (with ids, labels, descriptions, example_keywords) "
    "and scene-level dimensions (tone / angle / persona). Choose the single "
    "best category_id + subcategory_id for the scene, and set tone/angle/"
    "persona from the dimension lists. Use the exact ids from the taxonomy."
)

_OUTPUT_INSTRUCTIONS = (
    "Return ONLY valid JSON, no prose:\n"
    '{"category_id":"...","subcategory_id":"...","act":0,'
    '"tone":"...","angle":"...","persona":"...","confidence":0.0}'
)

_client = None


def _anthropic():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def classify_scene(scene_text, scope_id=0, caching=True, model=None, max_retries=3,
                   mode=None):
    """Classify one scene. Returns (label, usage).

    mode "injection" (default): taxonomy in a cached system block. This is what
    the cache-scope sweep uses.

    mode "agentic": the model fetches the taxonomy through a get_taxonomy tool,
    so it comes back as a tool_result and is re-sent on every cycle with no
    caching. Much more expensive (~44k input tokens per scene in our runs), only
    used for the agentic-vs-injection cost comparison.
    """
    mode = mode or os.environ.get("CLASSIFY_MODE", "injection")
    model = model or os.environ.get("CLASSIFY_MODEL", "claude-haiku-4-5")

    # mock backend: no API call, for the memory/latency runs where classify
    # doesn't matter. MOCK_CLASSIFY_MS fakes the call time.
    if os.environ.get("CLASSIFY_BACKEND", "real") == "mock":
        ms = float(os.environ.get("MOCK_CLASSIFY_MS", "0"))
        if ms:
            time.sleep(ms / 1000.0)
        return (
            {"category_id": "demonstration", "subcategory_id": "live_use", "act": 2,
             "tone": "neutral", "angle": "none", "persona": "unknown", "confidence": 0.5},
            {"input_tokens": 0, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
             "latency_ms": ms, "scope_id": scope_id, "cache_hit": False,
             "arrival_ts": time.time(), "model": "mock", "cycles": 1},
        )

    if mode == "agentic":
        return _classify_agentic(scene_text, scope_id, model, max_retries)
    scene_text = (scene_text or "").strip()

    # no speech -> fixed b_roll label, no API call
    if not scene_text:
        return (
            {"category_id": "b_roll", "subcategory_id": "general_visual", "act": 0,
             "tone": "neutral", "angle": "none", "persona": "unknown", "confidence": 1.0},
            {"input_tokens": 0, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
             "latency_ms": 0.0, "scope_id": scope_id, "cache_hit": False,
             "arrival_ts": time.time(), "model": model},
        )

    # marker first, then the taxonomy, so each scope gets its own cache entry.
    # The salt keeps a new sweep from hitting caches left warm by the last one,
    # and cost_sweep offsets scope_id per C so different C values never share one.
    salt = os.environ.get("COST_RUN_SALT", "0")
    scope_marker = f"[run:{salt} scope:{scope_id:06d}] evaluation marker\n"
    system_blocks = [{"type": "text", "text": _SYSTEM_PREAMBLE}]
    taxonomy_block = {
        "type": "text",
        "text": scope_marker + "TAXONOMY:\n" + _taxonomy(),
    }
    if caching:
        taxonomy_block["cache_control"] = {"type": "ephemeral"}
    system_blocks.append(taxonomy_block)

    user = f"SCENE:\n{scene_text}\n\n{_OUTPUT_INSTRUCTIONS}"

    arrival = time.time()
    t0 = time.perf_counter()
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = _anthropic().messages.create(
                model=model,
                max_tokens=256,
                system=system_blocks,
                messages=[{"role": "user", "content": user}],
            )
            break
        except Exception as e:  # network / 429 / 5xx
            last_exc = e
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"classify failed after {max_retries} attempts: {last_exc}")
    latency_ms = (time.perf_counter() - t0) * 1000.0

    u = resp.usage
    usage = {
        "input_tokens": u.input_tokens,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "output_tokens": u.output_tokens,
        "scope_id": scope_id,
        "arrival_ts": arrival,
        "latency_ms": latency_ms,
        "model": model,
        "cycles": 1,
    }
    usage["cache_hit"] = usage["cache_read_input_tokens"] > 0
    usage["cost_usd"] = cost_from_usage(usage, model)

    # best-effort parse, the sweeps mostly care about usage
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    try:
        label = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except Exception:
        label = {"category_id": None, "subcategory_id": None, "act": None,
                 "tone": None, "angle": None, "persona": None, "confidence": 0.0,
                 "parse_error": True}
    return label, usage


# agentic path: taxonomy comes from a tool, gets re-sent every cycle, no cache
_TOOLS = [
    {
        "name": "get_taxonomy",
        "description": "Return the full scene taxonomy (categories, subcategories, acts).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "save_classification_result",
        "description": "Save the final classification JSON.",
        "input_schema": {
            "type": "object",
            "properties": {"classification_result": {"type": "string"}},
            "required": ["classification_result"],
        },
    },
]


def _classify_agentic(scene_text, scope_id, model, max_retries=3):
    client = _anthropic()
    system = (
        _SYSTEM_PREAMBLE
        + " Call get_taxonomy to fetch the taxonomy, then call "
        "save_classification_result with your final JSON."
    )
    messages = [{"role": "user", "content": f"SCENE:\n{scene_text}\n\n{_OUTPUT_INSTRUCTIONS}"}]

    tot = {"input_tokens": 0, "cache_creation_input_tokens": 0,
           "cache_read_input_tokens": 0, "output_tokens": 0}
    cycles = 0
    label = None
    arrival = time.time()
    t0 = time.perf_counter()

    for _ in range(8):  # max cycles
        for attempt in range(max_retries):
            try:
                resp = client.messages.create(
                    model=model, max_tokens=1024, system=system,
                    tools=_TOOLS, messages=messages,
                )
                break
            except Exception as e:
                last = e
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"agentic classify failed: {last}")

        cycles += 1
        u = resp.usage
        tot["input_tokens"] += u.input_tokens
        tot["cache_creation_input_tokens"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        tot["cache_read_input_tokens"] += getattr(u, "cache_read_input_tokens", 0) or 0
        tot["output_tokens"] += u.output_tokens

        messages.append({"role": "assistant", "content": resp.content})
        tool_uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
        if resp.stop_reason != "tool_use" or not tool_uses:
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            try:
                label = json.loads(text[text.index("{"): text.rindex("}") + 1])
            except Exception:
                label = {"parse_error": True}
            break

        results = []
        for tu in tool_uses:
            if tu.name == "get_taxonomy":
                out = _taxonomy()  # re-sent in full on every cycle
            elif tu.name == "save_classification_result":
                out = "saved"
                try:
                    label = json.loads(tu.input.get("classification_result", "{}"))
                except Exception:
                    label = {"parse_error": True}
            else:
                out = ""
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out})
        messages.append({"role": "user", "content": results})

    latency_ms = (time.perf_counter() - t0) * 1000.0
    usage = dict(tot, scope_id=scope_id, arrival_ts=arrival, latency_ms=latency_ms,
                 model=model, cycles=cycles, cache_hit=False)
    usage["cost_usd"] = cost_from_usage(usage, model)
    return label or {"parse_error": True}, usage
