"""One place for LLM calls: rate limiting for free tiers, retries, JSON parsing."""

import json
import os
import re
import time

DEFAULT_MODEL = os.environ.get("UNCHARTED_MODEL", "gemini/gemini-3.8-flash")

_last_call = [0.0]


def chat(prompt, model=None, rpm=8, max_tokens=8000, temperature=0.4, system=None):
    """Send one prompt, return the reply text.

    rpm paces calls so a free tier key does not get throttled. Google shows the
    real limits for your project in AI Studio; raise rpm if yours allows it.
    """
    from litellm import completion

    model = model or DEFAULT_MODEL
    gap = 60.0 / max(rpm, 1)
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    last = None
    for attempt in range(6):
        wait = _last_call[0] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            extra = {}
            if model.startswith(("ollama/", "ollama_chat/")):
                # Ollama defaults to a 4096 token window and silently truncates past it, which
                # cuts the spec/files/pytest output out of fix prompts.
                extra["num_ctx"] = int(os.environ.get("UNCHARTED_NUM_CTX", "16384"))
            r = completion(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                           **extra)
            text = r.choices[0].message.content or ""
            if text.strip():
                return text
            last = RuntimeError("empty reply")
        except Exception as e:  # 429s, timeouts, overloaded
            last = e
            msg = str(e).lower()
            daily = ("quota" in msg and "day" in msg) or "per day" in msg or "(tpd)" in msg or "(rpd)" in msg
            if daily:
                # Exit instead of failing: otherwise every remaining build is recorded as a model failure.
                raise SystemExit(f"Daily free tier limit used up for {model}. Rerun after it resets; finished "
                                 "work is skipped.") from e
        time.sleep(min(120, 5 * 2 ** attempt))
    raise RuntimeError(f"LLM call failed after retries: {last}")


def parse_json(text):
    """Pull the outermost JSON object out of a reply, tolerating code fences and chatter."""
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = fence.group(1) if fence else text[text.find("{"): text.rfind("}") + 1]
    if not raw:
        raise ValueError("no JSON object in reply")
    try:
        return json.loads(raw, strict=False)  # strict=False: allow raw newlines/tabs inside strings
    except json.JSONDecodeError as first:
        # Common small-model slips: trailing commas, smart quotes around keys.
        fixed = re.sub(r",\s*([}\]])", r"\1", raw).replace("“", '"').replace("”", '"')
        try:
            return json.loads(fixed, strict=False)
        except json.JSONDecodeError:
            raise ValueError(f"reply was not valid JSON: {first}") from first
