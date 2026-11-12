"""The web build target: a project is one static page, checked in headless Chrome.

A `web` spec (see spec.PROMPT_WEB) is built as `index.html` plus ES modules in `src/` and tests in
`tests/tests.js`. Nothing is installed and nothing is run outside the browser:

  * the tests run inside a real page (`tests/index.html`, supplied here along with a small
    assertion harness), so they exercise the same modules the page loads;
  * then `index.html` itself is opened, clicked, dragged and typed at. It has to draw something,
    change in response, throw no errors, and ask for nothing outside its own folder.

The browser is the sandbox. Files are handed to it straight from disk under a made-up origin, and
every other request (http, websocket, DNS) is refused, so AI written JavaScript never sees the
network or the file system. Needs `pip install playwright` and Chrome or Edge (or
`playwright install chromium`).
"""

import base64
import re
from pathlib import Path

ORIGIN = "https://page.test"
VIEWPORT = {"width": 1280, "height": 720}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".svg": "image/svg+xml", ".txt": "text/plain; charset=utf-8",
         ".png": "image/png", ".wasm": "application/wasm"}
# The share of the window that must differ from the background colour for the page to count as drawn.
MIN_PAINTED = 0.005
TESTS_TIMEOUT_MS = 60000
PAGE_TEST = "page::opens_draws_and_responds"

HARNESS_JS = r"""// Supplied by the Uncharted factory and rewritten before every run: do not edit.
const tests = [];
let assertions = 0;

function plain(v) {
  if (ArrayBuffer.isView(v)) return Array.from(v);
  if (v instanceof Map) return { Map: [...v] };
  if (v instanceof Set) return { Set: [...v] };
  return v;
}

function show(v) {
  let s;
  try {
    s = JSON.stringify(v, (k, x) => (typeof x === "number" && !Number.isFinite(x) ? String(x) : plain(x)));
  } catch (e) {
    s = undefined;
  }
  if (s === undefined) s = String(v);
  return s.length > 400 ? s.slice(0, 400) + "..." : s;
}

function same(a, b) {
  if (a === b || (a !== a && b !== b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  a = plain(a);
  b = plain(b);
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && same(a[k], b[k]));
}

const label = (message) => (message ? message + ": " : "");

export function test(name, fn) {
  tests.push({ name, fn });
}

export function assert(condition, message) {
  assertions++;
  if (!condition) throw new Error(message || "assert failed");
}

export function assertEqual(actual, expected, message) {
  assertions++;
  if (!same(actual, expected)) throw new Error(`${label(message)}expected ${show(expected)}, got ${show(actual)}`);
}

export function assertClose(actual, expected, tolerance = 1e-6, message) {
  assertions++;
  if (!(Math.abs(actual - expected) <= tolerance)) {
    throw new Error(`${label(message)}expected ${show(expected)} within ${tolerance}, got ${show(actual)}`);
  }
}

export function assertThrows(fn, message) {
  assertions++;
  const fail = () => {
    throw new Error(`${label(message)}expected an error, but none was thrown`);
  };
  let result;
  try {
    result = fn();
  } catch (e) {
    return undefined;
  }
  if (result && typeof result.then === "function") return result.then(fail, () => undefined);
  return fail();
}

export async function run(timeoutMs = 5000) {
  const results = [];
  for (const { name, fn } of tests) {
    assertions = 0;
    let error = null, timer;
    try {
      await Promise.race([
        Promise.resolve().then(fn),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error(`timed out after ${timeoutMs} ms`)), timeoutMs);
        }),
      ]);
      if (assertions === 0) error = "made no assertions: call assert/assertEqual/assertClose/assertThrows";
    } catch (e) {
      error = String((e && e.stack) || e);
    }
    clearTimeout(timer);
    results.push({ name, ok: error === null, error, assertions });
  }
  return results;
}
"""

TESTS_HTML = """<!doctype html>
<!-- Supplied by the Uncharted factory and rewritten before every run: do not edit. -->
<meta charset="utf-8">
<title>Tests</title>
<style>
  body { font: 15px/1.5 system-ui, sans-serif; margin: 2rem; background: #111; color: #ddd; }
  li { list-style: none; white-space: pre-wrap; }
  .ok { color: #7c6; } .bad { color: #f76; }
</style>
<h1 id="summary">Running tests...</h1>
<ul id="out"></ul>
<script type="module">
  import { run } from "./harness.js";
  let results;
  try {
    await import("./tests.js");
    results = await run();
    if (results.length === 0) {
      results = [{ name: "no_tests_registered", ok: false, assertions: 0,
                   error: "tests/tests.js loaded but registered no tests, so no test ran. Exported functions and " +
                          "node:test or other runners are never called: each test must be registered with " +
                          "test(name, fn) imported from ./harness.js." }];
    }
  } catch (e) {
    results = [{ name: "load_tests", ok: false, assertions: 0,
                 error: "tests/tests.js could not be loaded, so no test ran: " + String((e && e.stack) || e) }];
  }
  const out = document.getElementById("out");
  for (const r of results) {
    const li = document.createElement("li");
    li.className = r.ok ? "ok" : "bad";
    li.textContent = (r.ok ? "PASSED " : "FAILED ") + r.name + (r.ok ? "" : "\\n    " + r.error);
    out.append(li);
  }
  const bad = results.filter((r) => !r.ok).length;
  document.getElementById("summary").textContent = `${results.length - bad} passed, ${bad} failed`;
  window.__uncharted = { done: true, results };
</script>
"""

# Written into every web project; never shown to the model as files it may change.
FACTORY_FILES = {"tests/harness.js": HARNESS_JS, "tests/index.html": TESTS_HTML}

WRITE_PROMPT = """Implement this project completely, as one static web page. It will be opened and
tested automatically in headless Chrome.

SPEC:
{spec}

Layout rules:
- `index.html` is the page. Its CSS is inline, and it loads `./src/main.js` with
  `<script type="module">`.
- Code lives in `src/` as ES modules (relative imports that include the `.js` extension), split
  into modules of at most about 300 lines, so any single file can later be rewritten whole. Keep
  rules, state, generation and maths in modules that never touch `document` or `window` when
  imported; keep drawing and input in `src/main.js` and the modules only it imports.
- Vanilla JavaScript only: no framework, no library, no build step, no npm, no CDN. No external
  URL of any kind (scripts, fonts, images, sounds, fetch): every request outside the page's own
  folder is blocked and fails the build. Draw the graphics in code (Canvas 2D, WebGL or SVG) and
  synthesize any sound with Web Audio; do not load image or audio files.
- Relative URLs only (`./src/main.js`, never `/src/main.js`), so the page also works from a
  subfolder on GitHub Pages.
- The page works the moment it opens: something is drawn at once, filling the window, and the
  picture changes within a second of the first click, drag or key press. No start screen that
  waits for one particular button. Sound may only start after the first click or key press.
- Usable with a mouse or a finger as well as a keyboard (pointer events), at 1280x720 and on a
  phone. Show the controls on the page in one short line.
- Anything random takes a seed (write a small seeded generator such as mulberry32), so the tests
  can pin its behavior down.
- Tests go in `tests/tests.js`, an ES module that uses the supplied harness, like this:

      import {{ test, assert, assertEqual, assertClose, assertThrows }} from "./harness.js";
      import {{ step }} from "../src/sim.js";
      test("test_spec_01_blinker_flips", () => {{
        assertEqual(step([[0, 1, 0], [0, 1, 0], [0, 1, 0]]), [[0, 0, 0], [1, 1, 1], [0, 0, 0]]);
      }});

  assertEqual compares deeply (numbers, strings, arrays, plain objects, typed arrays);
  assertClose(actual, expected, tolerance) is for floats; a test function may be async. Every
  test must make at least one assertion about a real output. `tests/harness.js` and
  `tests/index.html` are supplied for you: do not write them.
- The spec's "tests" list is the acceptance checklist. For entry number N (counting from 1),
  write a test named `test_spec_NN_<short_name>` (two digits, e.g. `test_spec_01_blinker_flips`)
  that checks exactly that behavior with the inputs and expected outputs it gives. The project
  only counts as built when every one of these passes. Other tests may use any name.
- The tests must import the same `src/` modules the page runs: logic that is tested but never
  loaded by `index.html`, or shown on the page but never tested, does not count.
- Real, working code. No TODOs, no placeholders, no stubs.

""" + "{format}"

FILE_FORMAT = """Return every file as a block in exactly this format, with nothing else around it:

===== FILE: index.html =====
<full contents>
===== FILE: src/main.js =====
<full contents>
===== FILE: tests/tests.js =====
<full contents>
===== END ====="""

FIX_FORMAT = """Return the full new contents of every file you change, in exactly this format:

===== FILE: path/to/file.js =====
<full contents>
===== END =====
To delete a file, add a line `===== DELETE: path =====` before END."""

TRIVIAL = ("All tests pass but they look trivial: tests/tests.js must import the page's own modules from "
           "`../src/` and make at least 3 real assertions about their outputs. Strengthen the tests, then "
           "fix the code.")

# Runs on a blank page of its own, so the page under test cannot influence it.
PIXEL_STATS = """async (b64) => {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const img = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
  const canvas = new OffscreenCanvas(img.width, img.height), g = canvas.getContext("2d");
  g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, img.width, img.height).data, counts = new Map();
  for (let i = 0; i < d.length; i += 16) {
    const k = (d[i] << 16) | (d[i + 1] << 8) | d[i + 2];
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  return { colors: counts.size, painted: 1 - Math.max(...counts.values()) / (d.length / 16) };
}"""


def write_factory_files(root):
    for rel, text in FACTORY_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def launch(pw):
    """Headless Chrome, else Edge, else Playwright's own Chromium, with name lookups switched off."""
    last = None
    for channel in ("chrome", "msedge", None):
        try:
            return pw.chromium.launch(headless=True, args=["--host-resolver-rules=MAP * ~NOTFOUND"],
                                      **({"channel": channel} if channel else {}))
        except Exception as e:  # that browser is not installed
            last = e
    raise SystemExit("No browser to test web pages with. Install Chrome or Edge, or run "
                     f"`playwright install chromium`. ({str(last).splitlines()[0]})")


class Session:
    """One browser context that serves `root` as ORIGIN and refuses everything else."""

    def __init__(self, browser, root):
        self.root = root.resolve()
        self.blocked, self.missing, self.loaded = [], [], set()
        self.context = browser.new_context(viewport=VIEWPORT, accept_downloads=False, service_workers="block")
        self.context.route("**/*", self._serve)
        self.context.route_web_socket("**/*", self._refuse_socket)

    def _refuse_socket(self, ws):
        self.blocked.append(ws.url)
        ws.close()

    def _serve(self, route):
        url = route.request.url
        if not url.startswith(ORIGIN + "/"):
            if not url.startswith(("data:", "blob:", "about:")):
                self.blocked.append(url)
                return route.abort()
            return route.continue_()
        rel = url[len(ORIGIN) + 1:].split("?")[0].split("#")[0]
        path = (self.root / rel).resolve()
        if path.is_dir():
            path, rel = path / "index.html", rel.rstrip("/") + "/index.html"
        if self.root not in path.parents or not path.is_file():
            if not rel.endswith("favicon.ico"):
                self.missing.append(rel)
            return route.fulfill(status=404, body="not found")
        self.loaded.add(rel.lstrip("/"))
        return route.fulfill(status=200, body=path.read_bytes(),
                             content_type=TYPES.get(path.suffix.lower(), "application/octet-stream"))

    def page(self):
        page, errors = self.context.new_page(), []
        page.on("pageerror", lambda e: errors.append(clean(str(e))))
        # A refused or missing file also logs a console error; those are reported by name instead.
        page.on("console", lambda m: errors.append(clean(m.text)) if m.type == "error"
                and "Failed to load resource" not in m.text else None)
        return page, errors

    def take(self):
        """Everything requested since the last call: (blocked urls, missing files, loaded files)."""
        got = (sorted(set(self.blocked)), sorted(set(self.missing)), set(self.loaded))
        self.blocked, self.missing, self.loaded = [], [], set()
        return got


def clean(text):
    return text.replace(ORIGIN + "/", "")


def short(error, lines=4):
    """An error with its stack cut down to the frames inside the project."""
    rows = clean(error).splitlines()
    keep = rows[:1] + [r for r in rows[1:] if "harness.js" not in r and r.strip().startswith("at ")][: lines - 1]
    return "\n    ".join(keep)


def run_unit_tests(session):
    """Open tests/index.html. Returns (lines, passed, failed, modules the tests loaded)."""
    page, errors = session.page()
    lines, passed, failed = [], 0, 0
    try:
        page.goto(ORIGIN + "/tests/index.html", wait_until="load", timeout=15000)
        page.wait_for_function("window.__uncharted && window.__uncharted.done", timeout=TESTS_TIMEOUT_MS)
        results = page.evaluate("window.__uncharted.results")
    except Exception as e:
        results = [{"name": "load_tests", "ok": False, "error":
                    f"the tests did not finish within {TESTS_TIMEOUT_MS // 1000}s (an endless loop, or a promise "
                    f"that never settles): {str(e).splitlines()[0]}"}]
    for r in results:
        if r["ok"]:
            passed += 1
            lines.append(f"PASSED tests/tests.js::{r['name']}")
        else:
            failed += 1
            lines.append(f"FAILED tests/tests.js::{r['name']} - {short(r.get('error') or 'failed')}")
    blocked, missing, loaded = session.take()
    if missing:
        lines.append("Files the tests asked for that do not exist: " + ", ".join(missing))
    if errors and failed:
        lines.append("Errors while loading the tests:\n    " + "\n    ".join(dict.fromkeys(errors))[:1500])
    try:
        page.close()
    except Exception:
        pass
    return lines, passed, failed, {f for f in loaded if f.startswith("src/")}


def poke(page):
    """What a visitor does in the first seconds: click, drag, press the usual keys.

    Returns a screenshot after the click and one after the rest, since a page that toggles (a
    blinker, a pause button) can be back where it started by the end.
    """
    w, h = VIEWPORT["width"], VIEWPORT["height"]
    page.mouse.move(w / 2, h / 2)
    page.mouse.click(w / 2, h / 2)
    page.wait_for_timeout(400)
    clicked = page.screenshot()
    page.mouse.move(w / 2 - 160, h / 2 - 70)
    page.mouse.down()
    page.mouse.move(w / 2 + 160, h / 2 + 90, steps=10)
    page.mouse.up()
    for key in ("ArrowRight", "ArrowUp", "Space", "Enter", "d", "w"):
        page.keyboard.press(key, delay=50)
    page.wait_for_timeout(700)
    return clicked, page.screenshot()


def check_page(session, browser, root, tested):
    """Open index.html the way a visitor would. Returns the list of problems (empty when fine)."""
    if not (root / "index.html").is_file():
        return ["index.html does not exist"]
    page, errors = session.page()
    problems = []
    try:
        page.goto(ORIGIN + "/index.html", wait_until="load", timeout=15000)
        page.wait_for_timeout(1000)
        first = page.screenshot()
        clicked, after = poke(page)
        (root / "screenshot.png").write_bytes(after)
        blank = browser.new_page()
        stats = blank.evaluate(PIXEL_STATS, base64.b64encode(first).decode())
        blank.close()
        if stats["painted"] < MIN_PAINTED:
            problems.append(f"the page is blank when it opens: {stats['painted']:.1%} of the window differs from "
                            "the background. Draw the first frame at once, filling the window, without waiting "
                            "for input.")
        if first == clicked == after:
            problems.append("nothing on the page changed after a click, a drag across the middle and the keys "
                            "ArrowRight, ArrowUp, Space, Enter, d, w. It must react to the first click, drag or "
                            "key press (or animate on its own).")
    except Exception as e:
        problems.append(f"the page could not be opened and used: {str(e).splitlines()[0]}")
    blocked, missing, loaded = session.take()
    if errors:
        problems.append("JavaScript errors on the page:\n    " + "\n    ".join(dict.fromkeys(errors))[:1500])
    if blocked:
        problems.append("the page asked for URLs outside its own folder, which is not allowed (draw or "
                        "synthesize everything in code): " + ", ".join(blocked[:8]))
    if missing:
        problems.append("the page asked for files that do not exist (use relative URLs like ./src/main.js): "
                        + ", ".join(missing[:8]))
    unused = sorted(tested - loaded)
    if unused and not problems:
        problems.append("the tests check modules that index.html never loads: " + ", ".join(unused) + ". The page "
                        "must run the same code the tests check: import and use these modules from src/main.js.")
    try:
        page.close()
    except Exception:
        pass
    return problems


def run_tests(root, sandbox=None):
    """Return (passed_count, failed_count, output), in the same shape as the pytest runner."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("Web projects are tested in a real browser: run `pip install playwright` first.")
    write_factory_files(root)
    with sync_playwright() as pw:
        browser = launch(pw)
        try:
            session = Session(browser, root)
            lines, passed, failed, tested = run_unit_tests(session)
            problems = check_page(session, browser, root, tested)
        finally:
            browser.close()
    if problems:
        failed += 1
        lines.append(f"FAILED {PAGE_TEST} - " + "\n  - ".join(["index.html was opened in Chrome:"] + problems))
    else:
        passed += 1
        lines.append(f"PASSED {PAGE_TEST}")
    lines.append(f"{passed} passed, {failed} failed")
    return passed, failed, "\n".join(lines)


def hints(out):
    found = []
    if "Failed to resolve module specifier" in out:
        found.append("- An import names a package (like `three`) or leaves out `./` or `.js`. Nothing is "
                     "installed: import only your own files, by relative path with the extension.")
    if "could not be loaded, so no test ran" in out:
        found.append("- tests/tests.js failed to load, so ZERO tests ran. Fix that import or syntax error first: "
                     "check that every name it imports is really exported by the module it names.")
    if "registered no tests" in out:
        found.append("- ZERO tests ran: tests/tests.js never calls the supplied harness. Nothing is run with node, "
                     "npm or any other runner (delete package.json and other test files). tests/tests.js must "
                     "start with `import { test, assert, assertEqual, assertClose, assertThrows } from "
                     "\"./harness.js\";` and register every test as `test(\"test_spec_01_short_name\", () => { "
                     "assertEqual(actual, expected); });`, using the harness's assert functions (a test that "
                     "makes no harness assertion fails).")
    return ("HINTS:\n" + "\n".join(found) + "\n\n") if found else ""


def tests_look_real(root):
    path = root / "tests" / "tests.js"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return "../src/" in text and len(re.findall(r"\bassert\w*\(", text)) >= 3


class WebTarget:
    """The web side of what build.build_one needs; build.PythonTarget is the other one."""
    name = "web"
    sandbox = "chrome"
    output_label = "TEST OUTPUT"
    fix_format = FIX_FORMAT
    trivial = TRIVIAL

    def write_prompt(self, spec, pkg, idea):
        return WRITE_PROMPT.format(spec=spec, format=FILE_FORMAT)

    def prepare(self, root, pkg, args, idea):
        return {}

    def run_tests(self, root, sandbox):
        return run_tests(root)

    def tests_look_real(self, root, pkg):
        return tests_look_real(root)

    def hints(self, out, pkg, idea):
        return hints(out)

    def play_check(self, root, idea, args):
        from . import play  # play imports this module
        return play.check(root, idea["spec"], args.play_model or args.model, args.rpm, keep=play.keep_dir(root))

    def play_report(self, verdict, idea):
        from . import play
        return play.report(verdict, idea["spec"])
