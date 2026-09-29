#!/usr/bin/env python3
"""Render reproducible, dependency-free SVGs from branchcraft benchmark CSV."""
import csv
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)


def svg_header(width, height, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
            f'<title>{html.escape(title)}</title><desc>{html.escape(desc)}</desc>'
            '<style>text{font-family:Inter,Arial,sans-serif;fill:#dce8f5}'
            '.muted{fill:#8fa6bf}.title{font-size:24px;font-weight:700}'
            '.small{font-size:12px}.label{font-size:14px;font-weight:600}</style>'
            f'<rect width="{width}" height="{height}" rx="22" fill="#0d1727"/>')


def latency(rows):
    selected = sorted((r for r in rows if int(r["batch"]) in (1, 16)),
                      key=lambda r: (int(r["batch"]), int(r["context"])))
    w, h = 1100, 530
    p = [svg_header(w, h, "Decode latency by context length", "CUDA event medians; lower is better")]
    p += ['<text x="42" y="48" class="title">When does split-KV repay its extra launch?</text>',
          '<text x="42" y="75" class="muted small">RTX PRO 5000 Blackwell · FP16 KV · GQA 32:8 · D=128 · lower is better</text>']
    ymax = max(float(r[x]) for r in selected for x in ("single_us", "split_us")) * 1.16
    left, top, plot_w, plot_h = 90, 122, 940, 305
    for tick in range(5):
        y = top + plot_h * tick / 4
        val = ymax * (1 - tick / 4)
        p.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#29415b"/>')
        p.append(f'<text x="{left-13}" y="{y+4:.1f}" text-anchor="end" class="muted small">{val:.0f}</text>')
    contexts = sorted(set(int(r["context"]) for r in selected))
    group = plot_w / (len(contexts) * 2)
    for bi, batch in enumerate((1, 16)):
        subset = {int(r["context"]): r for r in selected if int(r["batch"]) == batch}
        for ci, context in enumerate(contexts):
            r = subset.get(context)
            if not r:
                continue
            xmid = left + (bi * len(contexts) + ci + 0.5) * group
            for key, dx, color in (("single_us", -15, "#477fdd"), ("split_us", 15, "#42d6b0")):
                value = float(r[key]); bar_h = value / ymax * plot_h
                p.append(f'<rect x="{xmid+dx-13:.1f}" y="{top+plot_h-bar_h:.1f}" width="26" height="{bar_h:.1f}" rx="4" fill="{color}"/>')
            p.append(f'<text x="{xmid:.1f}" y="{top+plot_h+24}" text-anchor="middle" class="muted small">{context//1024}k' if context >= 1024 else f'<text x="{xmid:.1f}" y="{top+plot_h+24}" text-anchor="middle" class="muted small">{context}')
            p.append('</text>')
    p += [f'<text x="{left+group*len(contexts)/2:.0f}" y="{top+plot_h+53}" text-anchor="middle" class="label">Batch 1</text>',
          f'<text x="{left+group*len(contexts)*1.5:.0f}" y="{top+plot_h+53}" text-anchor="middle" class="label">Batch 16</text>',
          f'<rect x="{left}" y="497" width="13" height="13" rx="2" fill="#477fdd"/><text x="{left+22}" y="508" class="small">single partition</text>',
          f'<rect x="{left+185}" y="497" width="13" height="13" rx="2" fill="#42d6b0"/><text x="{left+207}" y="508" class="small">selected split count</text>',
          '</svg>']
    (OUT / "latency.svg").write_text(''.join(p), encoding="utf-8")


def phase(rows):
    batches = sorted(set(int(r["batch"]) for r in rows))
    contexts = sorted(set(int(r["context"]) for r in rows))
    lookup = {(int(r["batch"]), int(r["context"])): r for r in rows}
    w, h = 1050, 575
    p = [svg_header(w, h, "Split-KV speedup map", "Each cell shows baseline latency divided by selected split latency")]
    p += ['<text x="45" y="48" class="title">The dispatch map</text>',
          '<text x="45" y="75" class="muted small">Speedup versus one partition. Teal helps; coral hurts; slate means the policy chose one.</text>']
    left, top, cw, ch = 160, 145, 190, 82
    for ci, context in enumerate(contexts):
        p.append(f'<text x="{left+ci*cw+cw/2:.0f}" y="125" text-anchor="middle" class="label">{context:,} tokens</text>')
    for bi, batch in enumerate(batches):
        y = top + bi * ch
        p.append(f'<text x="{left-22}" y="{y+49}" text-anchor="end" class="label">B={batch}</text>')
        for ci, context in enumerate(contexts):
            r = lookup.get((batch, context))
            if r is None:
                continue
            speed = float(r["speedup"]); splits = int(r["splits"])
            color = "#153f42" if speed >= 1.05 else "#51323e" if speed < .95 else "#293d55"
            fg = "#67e8c5" if speed >= 1.05 else "#ffad9f" if speed < .95 else "#dce8f5"
            x = left + ci * cw
            p.append(f'<rect x="{x+4}" y="{y+4}" width="{cw-10}" height="{ch-10}" rx="13" fill="{color}" stroke="#4a6179"/>')
            p.append(f'<text x="{x+cw/2-3:.0f}" y="{y+42}" text-anchor="middle" font-size="26" font-weight="700" fill="{fg}">{speed:.2f}×</text>')
            p.append(f'<text x="{x+cw/2-3:.0f}" y="{y+62}" text-anchor="middle" class="muted small">{splits} partition{"s" if splits != 1 else ""}</text>')
    p += ['<text x="45" y="531" class="muted small">Policy: aim for ~2 CTAs/SM, cap at 16 splits, keep ≥128 tokens/partition; S &lt; 512 stays single.</text>', '</svg>']
    (OUT / "phase-map.svg").write_text(''.join(p), encoding="utf-8")


def tree_latency(rows):
    selected = [r for r in rows if int(r["requests"]) in (1, 4, 8)]
    w, h = 1120, 575
    p = [svg_header(w, h, "Tree verifier versus expanded-cache decode", "Measured CUDA event latencies; lower is better")]
    p += ['<text x="42" y="48" class="title">One tree, seven candidate paths</text>',
          '<text x="42" y="75" class="muted small">One batched launch per path; shared prefix versus physically expanded prefix · lower is better</text>']
    left, top, plot_w, plot_h = 93, 125, 956, 335
    max_val = max(float(r[k]) for r in selected for k in ("tree_us", "expanded_us")) * 1.14
    for tick in range(5):
        y = top + plot_h * tick / 4
        value = max_val * (1 - tick / 4)
        p.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#29415b"/>')
        p.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="muted small">{value:.0f}</text>')
    contexts = sorted(set(int(r["prefix_tokens"]) for r in selected))
    batches = sorted(set(int(r["requests"]) for r in selected))
    group = plot_w / (len(contexts) * len(batches))
    for bi, batch in enumerate(batches):
        for ci, context in enumerate(contexts):
            r = next(x for x in selected if int(x["requests"]) == batch and int(x["prefix_tokens"]) == context)
            mid = left + (bi * len(contexts) + ci + .5) * group
            for key, dx, color in (("tree_us", -14, "#42d6b0"), ("expanded_us", 14, "#477fdd")):
                value = float(r[key]); bar = value / max_val * plot_h
                p.append(f'<rect x="{mid+dx-12:.1f}" y="{top+plot_h-bar:.1f}" width="24" height="{bar:.1f}" rx="4" fill="{color}"/>')
            p.append(f'<text x="{mid:.1f}" y="{top+plot_h+23}" text-anchor="middle" class="muted small">{context}</text>')
        x = left + (bi + .5) * len(contexts) * group
        p.append(f'<text x="{x:.1f}" y="{top+plot_h+51}" text-anchor="middle" class="label">{batch} request{"s" if batch != 1 else ""}</text>')
    p += [f'<rect x="{left}" y="542" width="13" height="13" rx="2" fill="#42d6b0"/><text x="{left+22}" y="553" class="small">shared tree</text>',
          f'<rect x="{left+180}" y="542" width="13" height="13" rx="2" fill="#477fdd"/><text x="{left+202}" y="553" class="small">expanded paths</text>', '</svg>']
    (OUT / "tree-latency.svg").write_text(''.join(p), encoding="utf-8")


def tree_memory(rows):
    selected = sorted((r for r in rows if int(r["prefix_tokens"]) == 1024),
                      key=lambda r: int(r["requests"]))
    w, h = 920, 470
    p = [svg_header(w, h, "Live KV representation by request count", "Shared prefix tree uses far fewer live KV bytes than expanded paths")]
    p += ['<text x="40" y="48" class="title">The memory cost of repeating a prefix</text>',
          '<text x="40" y="75" class="muted small">1024-token prefix · seven-node tree · live KV bytes, excluding allocator reserve</text>']
    max_val = max(float(r["expanded_kv_mib"]) for r in selected)
    for i, r in enumerate(selected):
        y = 129 + i * 105
        p.append(f'<text x="40" y="{y+22}" class="label">{r["requests"]} request{"s" if int(r["requests"]) != 1 else ""}</text>')
        for j, (key, color) in enumerate((("shared_kv_mib", "#42d6b0"), ("expanded_kv_mib", "#477fdd"))):
            value = float(r[key]); width = max(4, value / max_val * 580)
            yy = y + j * 31
            p.append(f'<rect x="180" y="{yy}" width="{width:.1f}" height="23" rx="5" fill="{color}"/>')
            p.append(f'<text x="{190+width:.1f}" y="{yy+17}" class="small">{value:.1f} MiB</text>')
    p += ['<rect x="40" y="430" width="13" height="13" rx="2" fill="#42d6b0"/><text x="62" y="441" class="small">shared prefix + draft nodes</text>',
          '<rect x="310" y="430" width="13" height="13" rx="2" fill="#477fdd"/><text x="332" y="441" class="small">seven expanded sequences per request</text>', '</svg>']
    (OUT / "tree-memory.svg").write_text(''.join(p), encoding="utf-8")


def main():
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "benchmarks/decode_results.csv"
    with source.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("benchmark CSV is empty")
    latency(rows); phase(rows)
    tree_source = ROOT / "benchmarks/tree_results.csv"
    if tree_source.exists():
        with tree_source.open(newline="", encoding="utf-8") as fh:
            tree_rows = list(csv.DictReader(fh))
        tree_latency(tree_rows); tree_memory(tree_rows)
    print("Wrote benchmark SVGs in assets/")


if __name__ == "__main__":
    main()
