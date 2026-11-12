"""The play check: does a built web page do what its spec promises when someone opens it?

Unit tests and the page check in web.py can all pass on a page that does not play: a ragdoll that
never falls, a camera that frames the wrong corner, particles that never form the groups the spec
is about. This step records a short session in headless Chrome (a few seconds left alone, then a
click, a drag and some keys), measures how much of the window changed between frames, and shows
the frames to a model that takes pictures, next to the spec's own account of the first ten
seconds. The model answers with what it saw and a short list of plain faults.

`build` runs it once a web project passes its tests and sends the faults back to the fixer (see
--play-rounds there). On its own it re-checks projects that are already built:

    python uncharted.py play --out factory_out/fun --model openrouter/anthropic/claude-sonnet-5.5

The frames are kept in `<out>/play/<slug>/` and the verdict in the registry under "play". One
check is 7 screenshots plus a short prompt: about 2-3 cents on Sonnet 5.5, under 1 cent on Haiku.
"""

import argparse
import base64

from .. import llm
from . import web
from .registry import Registry

PLAY_TEST = "page::plays_like_the_spec"
# A pixel counts as changed when any colour channel moved by more than this (out of 255).
CHANGED_BY = 12

# Runs on a blank page of its own. Takes PNGs as base64; returns the share of the window that
# changed between each frame and the one before it, and the frames again as JPEG (a busy canvas is
# 2 MB a frame as PNG, which makes the request to the model slow and close to provider limits).
FRAME_CHANGES = """async (frames) => {
  const data = [], jpegs = [];
  for (const b64 of frames) {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const img = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
    const canvas = new OffscreenCanvas(img.width, img.height), g = canvas.getContext("2d");
    g.drawImage(img, 0, 0);
    data.push(g.getImageData(0, 0, img.width, img.height).data);
    const jpeg = new Uint8Array(await (await canvas.convertToBlob({ type: "image/jpeg", quality: 0.85 })).arrayBuffer());
    let s = "";
    for (let i = 0; i < jpeg.length; i += 0x8000) s += String.fromCharCode.apply(null, jpeg.subarray(i, i + 0x8000));
    jpegs.push(btoa(s));
  }
  const out = [];
  for (let f = 1; f < data.length; f++) {
    const a = data[f - 1], b = data[f];
    let changed = 0, total = 0;
    for (let i = 0; i < a.length; i += 8) {
      total++;
      if (Math.abs(a[i] - b[i]) > %d || Math.abs(a[i + 1] - b[i + 1]) > %d || Math.abs(a[i + 2] - b[i + 2]) > %d) changed++;
    }
    out.push(changed / total);
  }
  return { changes: out, jpegs };
}""" % (CHANGED_BY, CHANGED_BY, CHANGED_BY)

PROMPT = """You are play-testing a small web toy or game that an AI wrote from the spec below. It already
passes its unit tests. Unit tests cannot tell whether the page works for a person, so you are shown
{n} screenshots from one session in headless Chrome at 1280x720 and asked what a stranger who
opened the page would see.

WHAT THE SPEC PROMISES
Title: {title}
Pitch: {pitch}
The first ten seconds: {first}
Controls: {controls}

THE SESSION (the screenshots are attached in this order)
{session}

The tester does not know where things are on the page, so the inputs are generic: one click in the
middle, one drag across the middle, a few common keys. Do not fault the page because a particular
control from the spec was never tried, or because a frame happens to miss a short event. Judge
what these frames can show:
- Is the main subject of the spec on screen, inside the window, at a size a person can see and use?
- Does what the spec says happens on its own really happen across the frames with no input
  (things fall, move, grow, play)? Use the "changed" figures: 0.0% means the picture stood still.
- Did the picture respond to the click, the drag or the keys in a way that fits the controls?
- Is anything plainly broken: the scene off to one side or mostly outside the window, objects
  frozen where they should move, text on top of other text, a blank or nearly blank playfield,
  everything collapsed into one spot, an error message on the page?

Be strict about what is plainly wrong and silent about taste and polish: a plain page that does
what the spec says plays. Something dim, small or unlabelled, two markers that overlap before the
first move, or a panel that is still empty because nothing has been played yet, is not a fault.
List a problem only when it stops the page doing what the spec says, or would stop a stranger from
using it. Every problem must name what is on screen and what should be there instead, in words the
programmer can act on without seeing the screenshots.

Reply with JSON only:
{{"seen": "two sentences on what the frames actually show",
  "plays": true or false,
  "problems": ["at most 4 specific faults, the worst first; empty when it plays"]}}"""


def record(root):
    """Open index.html and use it for about ten seconds. Returns [(what happened, png bytes)], the
    share of the window that changed before each frame after the first, and the frames as JPEG."""
    from playwright.sync_api import sync_playwright
    w, h = web.VIEWPORT["width"], web.VIEWPORT["height"]
    frames = []
    with sync_playwright() as pw:
        browser = web.launch(pw)
        try:
            session = web.Session(browser, root)
            page, _ = session.page()
            page.goto(web.ORIGIN + "/index.html", wait_until="load", timeout=15000)

            def shot(wait_ms, what):
                page.wait_for_timeout(wait_ms)
                frames.append((what, page.screenshot()))

            shot(1000, "1.0 s after the page opened, no input yet")
            shot(2500, "2.5 s later, still no input")
            shot(2500, "another 2.5 s later, still no input")
            page.mouse.move(w / 2, h / 2)
            page.mouse.click(w / 2, h / 2)
            shot(500, "0.5 s after one click in the middle of the window")
            page.mouse.move(w / 2 - 200, h / 2 - 90)
            page.mouse.down()
            page.mouse.move(w / 2 + 200, h / 2 + 110, steps=12)
            page.mouse.up()
            shot(600, "after a drag from upper left of the middle to lower right of it")
            for key in ("ArrowRight", "ArrowRight", "ArrowUp", "Space", "Enter", "d", "w"):
                page.keyboard.press(key, delay=60)
            shot(600, "after the keys ArrowRight, ArrowRight, ArrowUp, Space, Enter, d, w")
            shot(2000, "2 s later, no further input")
            blank = browser.new_page()
            got = blank.evaluate(FRAME_CHANGES, [base64.b64encode(png).decode() for _, png in frames])
        finally:
            browser.close()
    return frames, got["changes"], [base64.b64decode(j) for j in got["jpegs"]]


def describe(frames, changes):
    lines = []
    for i, (what, _) in enumerate(frames):
        note = f" (changed since screenshot {i}: {changes[i - 1]:.1%} of the window)" if i else ""
        lines.append(f"{i + 1}. {what}{note}")
    return "\n".join(lines)


def check(root, spec, model, rpm=8, keep=None):
    """Record a session and ask `model` whether it matches the spec.

    Returns {"ok", "seen", "problems", "changes", "model"}, or None when no verdict could be had
    (the page would not open, the model takes no pictures, the reply was not JSON): the check is
    advice on top of the tests, so a broken check never fails a build by itself.
    """
    try:
        frames, changes, jpegs = record(root)
    except Exception as e:
        print(f"    play check skipped: the session could not be recorded ({str(e).splitlines()[0]})")
        return None
    if keep:
        keep.mkdir(parents=True, exist_ok=True)
        for i, (_, png) in enumerate(frames):
            (keep / f"frame_{i + 1}.png").write_bytes(png)
    prompt = PROMPT.format(n=len(frames), title=spec.get("title", ""), pitch=spec.get("pitch", ""),
                           first=spec.get("first_ten_seconds") or "(not given: go by the pitch and the controls)",
                           controls=spec.get("controls", ""), session=describe(frames, changes))
    try:
        # Thinking stays on: comparing seven frames is the kind of call it helps, and Sonnet 5.5 on
        # OpenRouter refuses to run without it. max_tokens leaves room for it before the answer.
        reply = llm.parse_json(llm.chat(prompt, model, rpm, max_tokens=6000, temperature=0.0,
                                        images=jpegs, reasoning=True))
    except Exception as e:
        print(f"    play check skipped: no verdict from {model or llm.DEFAULT_MODEL} ({str(e)[:160]})")
        return None
    problems = [str(p).strip() for p in (reply.get("problems") or []) if str(p).strip()][:4]
    ok = bool(reply.get("plays")) and not problems
    if not ok and not problems:
        problems = ["the reviewer said the page does not play as the spec describes, without naming a fault: "
                    + str(reply.get("seen", ""))[:300]]
    return {"ok": ok, "seen": str(reply.get("seen", ""))[:600], "problems": problems,
            "changes": [round(c, 4) for c in changes], "model": model or llm.DEFAULT_MODEL}


def report(verdict, spec):
    """The verdict as test output for the fixer, in the same shape as a failed test."""
    return (f"FAILED {PLAY_TEST} - every unit test passes, but the page does not yet play the way the spec "
            "says. A reviewer opened index.html in Chrome at 1280x720, left it alone for 6 seconds, then "
            "clicked the middle, dragged across the middle and pressed ArrowRight, ArrowUp, Space, Enter, d, w.\n"
            f"  What the spec promises: {spec.get('first_ten_seconds') or spec.get('pitch', '')}\n"
            f"  What was on screen: {verdict['seen']}\n"
            "  Problems to fix:\n" + "\n".join(f"  - {p}" for p in verdict["problems"]) + "\n"
            "Fix the page itself (main loop, drawing, camera and scale, starting state, input handling) so "
            "these are no longer true. Keep every test passing; add a test for whatever logic you had to "
            "correct.")


def keep_dir(root):
    """`<out>/play/<slug>` for a project at `<out>/projects/<slug>`: outside the project, so the
    frames are never published with it."""
    return root.parent.parent / "play" / root.name


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="factory_out")
    p.add_argument("--slug", action="append", default=None, help="Check just this project (repeatable)")
    p.add_argument("--model", default=llm.DEFAULT_MODEL, help="Must take pictures")
    p.add_argument("--rpm", type=int, default=8)
    args = p.parse_args(argv)

    reg = Registry(args.out)
    slugs = args.slug or [s for s, i in reg.ideas.items() if i.get("status") == "built" and i.get("target") == "web"]
    before = llm.openrouter_spent() if args.model.startswith("openrouter/") else None
    plays = 0
    for slug in slugs:
        root = reg.project_dir(slug)
        if not (root / "index.html").is_file():
            print(f"  {slug}: no index.html, skipped")
            continue
        verdict = check(root, reg.ideas[slug]["spec"], args.model, args.rpm, keep=keep_dir(root))
        if verdict is None:
            continue
        reg.update(slug, play=verdict)
        plays += verdict["ok"]
        print(f"  {slug}: {'PLAYS' if verdict['ok'] else 'DOES NOT PLAY'}\n    seen: {verdict['seen']}")
        for problem in verdict["problems"]:
            print(f"    - {problem}")
    after = llm.openrouter_spent() if before is not None else None
    cost = f" (${after - before:.3f} on OpenRouter)" if before is not None and after is not None else ""
    print(f"Done: {plays}/{len(slugs)} play{cost}. Frames are in {reg.root / 'play'}")


if __name__ == "__main__":
    main()
