# Copyright (c) 2026, WireCraft Contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Build a clickable acceptance page from ``verify_matrix.py`` output.

    python scripts/wirecraft/build_report.py results
"""

import glob
import html
import json
import os
import sys

STYLE = """
body{font:14px/1.45 system-ui,sans-serif;margin:24px;color:#1c2024;background:#fff}
h1{font-size:22px}h2{font-size:17px;margin-top:28px}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #e3e6ea;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f4f5f7;position:sticky;top:0}.ok{color:#0a7a3d;font-weight:600}.bad{color:#b3261e;font-weight:600}
.videos{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
figure{margin:0}figcaption{font-size:12px;word-break:break-all}video{width:100%;background:#111}
code{font-size:12px}input{padding:6px 8px;width:320px;margin:8px 0}
"""
SCRIPT = """
const box=document.getElementById('q');
box.oninput=()=>{const q=box.value.toLowerCase();
for(const r of document.querySelectorAll('tbody tr'))r.style.display=r.textContent.toLowerCase().includes(q)?'':'none';};
"""


def link(root, path, label):
    if not os.path.exists(path):
        return ""
    return f'<a href="{html.escape(os.path.relpath(path, root))}">{label}</a>'


def detail(result):
    if result["mode"] == "random":
        return (
            f"episodes {result.get('episodes', 0)}, partial resets {result.get('partial_resets', 0)}, "
            f"full resets {result.get('full_resets', 0)}"
        )
    rollouts = result.get("rollouts", [])
    notes = []
    for key in ("best_depth_mm", "seated", "in_channel_fraction"):
        values = [r[key] for r in rollouts if key in r]
        if values:
            notes.append(f"{key} {values}")
    return f"success {result.get('success_count', 0)}/{len(rollouts)}; " + "; ".join(notes)


def row(root, cell, result):
    spec = result.get("composition", {}).get("spec", {})
    state = '<span class="ok">pass</span>' if result.get("passed") else '<span class="bad">fail</span>'
    evidence = " ".join(
        filter(
            None,
            (
                link(root, os.path.join(cell, "rollout.mp4"), "video"),
                link(root, os.path.join(cell, "result.json"), "result"),
                link(root, os.path.join(cell, "params", "env.yaml"), "env"),
                link(root, os.path.join(cell, "params", "agent.yaml"), "agent"),
                link(root, os.path.join(cell, "params", "selection.yaml"), "layers"),
                link(root, os.path.join(cell, "run.log"), "log"),
            ),
        )
    )
    columns = (
        result.get("task", ""), spec.get("robot", ""), spec.get("task", ""), spec.get("wire", ""),
        spec.get("connector", ""), spec.get("variant", ""), result.get("mode", ""), result.get("seed", ""),
        result.get("num_envs", ""), result.get("steps", ""), result.get("faults", ""), result.get("exit_code", ""),
    )
    cells = "".join(f"<td>{html.escape(str(c))}</td>" for c in columns)
    return f"<tr>{cells}<td>{state}</td><td>{html.escape(detail(result))}</td><td>{evidence}</td></tr>"


def main() -> None:
    root = os.path.abspath(sys.argv[1])
    cells = []
    for path in sorted(glob.glob(os.path.join(root, "*", "*", "result.json"))):
        with open(path, encoding="utf-8") as file:
            cells.append((os.path.dirname(path), json.load(file)))
    passed = sum(bool(r.get("passed")) for _, r in cells)
    with open(os.path.join(root, "summary.json"), encoding="utf-8") as file:
        not_expected = json.load(file).get("not_expected", [])
    pending = "".join(f"<li><code>{html.escape(task)}</code></li>" for task in not_expected)
    heads = "task robot job wire connector variant mode seed envs steps faults exit state detail evidence".split()
    rows = "\n".join(row(root, cell, result) for cell, result in cells)
    videos = "\n".join(
        f'<figure><video controls preload="none" src="{html.escape(os.path.relpath(p, root))}"></video>'
        f"<figcaption>{html.escape(os.path.basename(os.path.dirname(p)))}</figcaption></figure>"
        for p in sorted(glob.glob(os.path.join(root, "*", "*", "rollout.mp4")))
    )
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>WireCraft acceptance</title>
<style>{STYLE}</style></head><body><h1>WireCraft acceptance</h1>
<p>{passed} of {len(cells)} cells passed.</p>
<h2>No finished scripted policy ({len(not_expected)})</h2>
<p>Random mode covers these; their scripted runs were not made.</p><ul>{pending}</ul>
<h2>Rollouts</h2><div class="videos">{videos}</div>
<h2>All cells</h2><input id="q" placeholder="Filter, e.g. franka deformable fail">
<table><thead><tr>{''.join(f'<th>{h}</th>' for h in heads)}</tr></thead><tbody>
{rows}</tbody></table><script>{SCRIPT}</script></body></html>"""
    with open(os.path.join(root, "index.html"), "w", encoding="utf-8") as file:
        file.write(page)
    print(os.path.join(root, "index.html"))


if __name__ == "__main__":
    main()
