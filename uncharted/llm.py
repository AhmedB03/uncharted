"""One place for LLM calls: rate limiting for free tiers, retries, JSON parsing."""

import json
import os
import re
import time

DEFAULT_MODEL = os.environ.get("UNCHARTED_MODEL", "gemini/gemini-3.8-flash")

_last_call = [0.0]


class ContentFiltered(RuntimeError):
    """The provider refused the prompt outright; the same prompt will be refused again."""


def reasoning_off(model):
    """Extra completion arguments that stop an OpenRouter model from thinking before it answers.

    DeepSeek v4 and Claude Haiku 5.5 think by default on OpenRouter, and the hidden reasoning can
    use up all of max_tokens and leave the reply empty. UNCHARTED_REASONING=on keeps it, and
    low, medium or high keep it at that effort (Claude Sonnet 5.5 refuses to run with it off).
    """
    mode = os.environ.get("UNCHARTED_REASONING", "off")
    if model.startswith("openrouter/") and mode != "on":
        if mode in ("low", "medium", "high"):
            return {"extra_body": {"reasoning": {"effort": mode}}}
        return {"extra_body": {"reasoning": {"enabled": False}}}
    return {}


def chat(prompt, model=None, rpm=8, max_tokens=8000, temperature=0.4, system=None, images=None,
         reasoning=None):
    """Send one prompt, return the reply text.

    rpm paces calls so a free tier key does not get throttled. Google shows the
    real limits for your project in AI Studio; raise rpm if yours allows it.
    images is a list of PNG or JPEG files as bytes, sent after the prompt in order (the model must
    take pictures). reasoning=True lets the model think whatever UNCHARTED_REASONING says: some
    endpoints (Claude Sonnet 5.5 on OpenRouter) refuse a call that switches it off.
    """
    from litellm import completion

    model = model or DEFAULT_MODEL
    gap = 60.0 / max(rpm, 1)
    content = prompt
    if images:
        import base64
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {
                "url": f"data:image/{'jpeg' if img[:2] == bytes([255, 216]) else 'png'};base64,"
                       + base64.b64encode(img).decode()}}
            for img in images]
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": content}]
    last = None
    for attempt in range(6):
        wait = _last_call[0] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            extra = {} if reasoning else reasoning_off(model)
            if model.startswith(("ollama/", "ollama_chat/")):
                # Ollama defaults to a 4096 token window and silently truncates past it, which
                # cuts the spec/files/pytest output out of fix prompts.
                extra["num_ctx"] = int(os.environ.get("UNCHARTED_NUM_CTX", "16384"))
            # Without a timeout one stalled provider connection blocks the whole run for 10+ minutes.
            # A long reply needs longer: 32,000 tokens at 40 a second is over 13 minutes, and a call
            # cut off at 240 s is paid for in full and then sent again.
            timeout = float(os.environ.get("UNCHARTED_LLM_TIMEOUT", "1800" if model.startswith("ollama")
                                           else str(max(240, max_tokens // 40 + 60))))
            r = completion(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                           timeout=timeout, **extra)
            text = r.choices[0].message.content or ""
            if text.strip():
                return text
            if getattr(r.choices[0], "finish_reason", None) == "content_filter":
                # The provider's filter blocks the same prompt every time; retrying only burns minutes.
                raise ContentFiltered(f"{model} content filter blocked this prompt")
            out_of_room = getattr(r.choices[0], "finish_reason", None) == "length"
            last = RuntimeError("empty reply: max_tokens used up before any answer (hidden reasoning?)"
                                if out_of_room else "empty reply")
        except ContentFiltered:
            raise
        except Exception as e:  # 429s, timeouts, overloaded
            if type(e).__name__ == "BadRequestError":
                # The request itself is wrong (no picture input, a setting the endpoint refuses):
                # it fails the same way every time, and six tries with backoff take five minutes.
                raise RuntimeError(f"{model} rejected the request: {str(e)[-300:]}") from e
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


def openrouter_spent():
    """Total USD used on the OpenRouter key so far, or None if it cannot be read."""
    import json
    import urllib.request
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/credits",
                                     headers={"Authorization": f"Bearer {key}"})
        return float(json.load(urllib.request.urlopen(req, timeout=30))["data"]["total_usage"])
    except Exception:
        return None
