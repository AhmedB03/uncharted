"""Build a static gallery page of everything the factory has made, ranked by stars.

    python uncharted.py gallery --refresh-stars
    # then host factory_out/gallery/index.html on GitHub Pages
"""

import argparse
import html
import json
import os

import requests

from uncharted.factory.registry import Registry, now


def refresh_stars(reg):
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {"Accept": "application/vnd.github+json", **({"Authorization": f"Bearer {token}"} if token else {})}
    for slug, idea in reg.ideas.items():
        if idea.get("repo"):
            r = requests.get(f"https://api.github.com/repos/{idea['repo']}", headers=headers, timeout=20)
            if r.status_code == 200:
                d = r.json()
                history = idea.get("star_history", [])
                history.append([now()[:10], d["stargazers_count"]])
                reg.update(slug, stars=d["stargazers_count"], forks=d["forks_count"], star_history=history[-60:])


def row(idea):
    spec = idea.get("spec", {})
    status = idea["status"]
    atoms = "".join(f'<span class="atom">{html.escape(a)}</span>' for a in idea.get("atoms", []))
    depth = max(0.0, min(1.0, (idea.get("unavailability_z", 0) + 1) / 4))  # z of -1..3 -> 0..1
    title = html.escape(spec.get("title") or idea["slug"])
    if idea.get("repo_url"):
        title = f'<a href="{html.escape(idea["repo_url"])}">{title}</a>'
    detail = {
        "published": f'{idea.get("tests_passed", 0)} tests, {idea.get("loc", 0)} lines',
        "built": f'{idea.get("tests_passed", 0)} tests passing, waiting to ship',
        "failed": f'gave up after {idea.get("iterations", 0)} tries',
        "infeasible": "no sensible project fits",
        "spec": "spec written, not built yet",
    }.get(status, status)
    detail = html.escape(detail)
    if idea.get("page_url"):  # a published web project: link straight to the playable page
        detail += f' · <a href="{html.escape(idea["page_url"])}">play it</a>'
    stars = idea.get("stars")
    star_cell = f'<span class="stars">{stars}</span><span class="unit">stars</span>' if stars is not None else ""
    return f"""
<li class="entry" data-status="{status}" data-stars="{stars or 0}" data-depth="{depth:.3f}">
  <div class="combo">{atoms}</div>
  <div class="body">
    <h2>{title}</h2>
    <p class="pitch">{html.escape(spec.get("pitch", ""))}</p>
    <p class="detail"><span class="badge {status}">{status}</span> {detail}</p>
  </div>
  <div class="metric">
    <div class="sounding" title="How far this combination sits from any existing developer community">
      <span style="height:{depth * 100:.0f}%"></span></div>
    <div class="count">{star_cell}</div>
  </div>
</li>"""


def render(reg):
    ideas = sorted(reg.ideas.values(), key=lambda i: (-(i.get("stars") or -1), i.get("rank", 9999)))
    counts = {s: sum(1 for i in ideas if i["status"] == s) for s in ("published", "built", "failed", "infeasible")}
    tried = counts["published"] + counts["built"] + counts["failed"]
    rate = f"{100 * (counts['published'] + counts['built']) / tried:.0f}%" if tried else "no"
    stars = sum(i.get("stars") or 0 for i in ideas)
    rows = "\n".join(row(i) for i in ideas) or '<li class="empty">Nothing yet. Run spec, then build.</li>'
    return PAGE.format(
        rows=rows, n=len(ideas), published=counts["published"], rate=rate, stars=stars,
        tried=tried, updated=html.escape(now()[:16].replace("T", " ")),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="factory_out")
    p.add_argument("--refresh-stars", action="store_true")
    args = p.parse_args(argv)
    reg = Registry(args.out)
    if args.refresh_stars:
        refresh_stars(reg)
    d = reg.root / "gallery"
    d.mkdir(exist_ok=True)
    (d / "index.html").write_text(render(reg), encoding="utf-8")
    (d / "registry.json").write_text(json.dumps(reg.data, indent=2))
    print(f"Wrote {d / 'index.html'}")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Uncharted Factory</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>
:root {{
  --paper: #e7efec; --ink: #13294b; --soft: #4b6078; --contour: #8fb3cf;
  --sounding: #b3261e; --land: #c8b07a; --line: #c4d4d6; --card: #f3f7f6;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper: #0f1d2e; --ink: #e3ecf2; --soft: #9db0c2; --contour: #2f5576;
    --sounding: #ff7a6b; --land: #a8935f; --line: #233a52; --card: #14263a;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--paper); color: var(--ink);
  font: 17px/1.6 "Source Serif 4", Georgia, serif; }}
a {{ color: inherit; text-decoration-color: var(--contour); text-underline-offset: 3px; }}
a:focus-visible, button:focus-visible {{ outline: 3px solid var(--sounding); outline-offset: 3px; }}
header {{ position: relative; overflow: hidden; padding: 72px 0 48px; border-bottom: 2px solid var(--ink); }}
header svg {{ position: absolute; inset: 0; width: 100%; height: 100%; z-index: 0; }}
header .inner {{ position: relative; z-index: 1; max-width: 980px; margin: 0 auto; padding: 0 24px; }}
h1 {{ font: 700 clamp(56px, 11vw, 132px)/0.86 "Barlow Condensed", sans-serif;
  letter-spacing: -0.01em; margin: 0 0 24px; }}
.lede {{ max-width: 34em; font-size: 20px; margin: 0 0 12px; }}
.tally {{ color: var(--soft); margin: 0; max-width: 40em; }}
main {{ max-width: 980px; margin: 0 auto; padding: 32px 24px 80px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }}
.controls button {{ font: 600 16px "Barlow Condensed", sans-serif; letter-spacing: 0.02em;
  padding: 6px 14px; border: 2px solid var(--ink); background: transparent; color: var(--ink);
  border-radius: 999px; cursor: pointer; }}
.controls button[aria-pressed="true"] {{ background: var(--ink); color: var(--paper); }}
ol {{ list-style: none; margin: 0; padding: 0; }}
.entry {{ display: grid; grid-template-columns: 1fr 88px; gap: 4px 24px;
  padding: 24px 0; border-bottom: 1px solid var(--line); }}
.entry[hidden] {{ display: none; }}
.combo {{ grid-column: 1 / -1; display: flex; flex-wrap: wrap; align-items: center; gap: 6px 0; }}
.atom {{ font: 600 19px/1.2 "Barlow Condensed", sans-serif; }}
.atom + .atom::before {{ content: "+"; color: var(--sounding); font-weight: 700; margin: 0 10px; }}
h2 {{ font: 600 22px/1.25 "Source Serif 4", serif; margin: 6px 0 4px; }}
.pitch {{ margin: 0 0 8px; max-width: 38em; }}
.detail {{ margin: 0; color: var(--soft); font-size: 15px; }}
.badge {{ font: 600 14px "Barlow Condensed", sans-serif; padding: 1px 8px; border-radius: 4px;
  border: 1.5px solid currentColor; margin-right: 6px; }}
.badge.published {{ color: var(--ink); background: var(--land); border-color: var(--land); }}
.badge.failed, .badge.infeasible {{ color: var(--sounding); }}
.metric {{ display: flex; align-items: flex-end; gap: 12px; justify-self: end; }}
.sounding {{ width: 10px; height: 64px; background: var(--card); border: 1px solid var(--line);
  display: flex; align-items: flex-end; }}
.sounding span {{ display: block; width: 100%; background: var(--sounding); }}
.count {{ text-align: right; }}
.stars {{ display: block; font: 700 34px/1 "Barlow Condensed", sans-serif; }}
.unit {{ color: var(--soft); font-size: 13px; }}
.empty {{ padding: 48px 0; color: var(--soft); }}
footer {{ max-width: 980px; margin: 0 auto; padding: 0 24px 48px; color: var(--soft); font-size: 15px; }}
.key {{ display: inline-flex; align-items: center; gap: 8px; }}
.key i {{ width: 10px; height: 22px; background: linear-gradient(to top, var(--sounding) 60%, var(--card) 60%);
  border: 1px solid var(--line); }}
@media (max-width: 560px) {{
  header {{ padding: 48px 0 32px; }} header .inner, main, footer {{ padding-left: 16px; padding-right: 16px; }}
  .entry {{ grid-template-columns: 1fr 64px; }} .stars {{ font-size: 28px; }}
}}
</style>
</head>
<body>
<header>
  <svg aria-hidden="true" preserveAspectRatio="none" viewBox="0 0 1000 400">
    <g style="fill:none;stroke:var(--contour);stroke-width:1.2;opacity:.55">
      <path d="M-20 330 C 180 260, 320 380, 520 300 S 860 230, 1020 290"/>
      <path d="M-20 290 C 200 220, 340 330, 540 255 S 870 190, 1020 245"/>
      <path d="M-20 250 C 220 180, 360 290, 560 210 S 880 150, 1020 200"/>
      <path d="M560 120 c 60 -40, 170 -30, 190 20 s -60 90, -150 70 s -90 -60, -40 -90 z"/>
      <path d="M600 125 c 40 -25, 110 -18, 122 14 s -40 56, -96 44 s -56 -38, -26 -58 z"/>
      <path d="M640 130 c 20 -12, 52 -8, 58 7 s -18 26, -45 20 s -27 -18, -13 -27 z"/>
    </g>
  </svg>
  <div class="inner">
    <h1>Uncharted Factory</h1>
    <p class="lede">Software nobody was going to write. A model finds technique combinations that fit
      together but sit far from every developer community, then an AI builds each one until its tests pass.</p>
    <p class="tally">{n} ideas logged, {tried} build attempts, {rate} reached passing tests,
      {published} shipped to GitHub with {stars} stars between them.</p>
  </div>
</header>
<main>
  <div class="controls" role="group" aria-label="Show">
    <button aria-pressed="true" data-filter="all">Everything</button>
    <button aria-pressed="false" data-filter="published">Shipped</button>
    <button aria-pressed="false" data-filter="built">Built</button>
    <button aria-pressed="false" data-filter="failed">Failed</button>
  </div>
  <ol id="list">{rows}</ol>
</main>
<footer>
  <p><span class="key"><i></i> The red bar is how uncharted a combination is: taller means further from
    anyone who already works with those techniques.</span></p>
  <p>Everything here was invented and written by AI and has not been reviewed by a person.
    Updated {updated} UTC. Built with Uncharted, on the Alien Space of Science.</p>
</footer>
<script>
document.querySelectorAll(".controls button").forEach(function (b) {{
  b.addEventListener("click", function () {{
    document.querySelectorAll(".controls button").forEach(function (x) {{ x.setAttribute("aria-pressed", x === b); }});
    var f = b.dataset.filter;
    document.querySelectorAll(".entry").forEach(function (e) {{
      e.hidden = f !== "all" && e.dataset.status !== f;
    }});
  }});
}});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
