from __future__ import annotations

import html
from pathlib import Path

CSS = """
:root{color-scheme:dark;
 --page:#0d0d0d; --surface-1:#1a1a19; --line:#2c2c2a; --baseline:#383835;
 --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --series-1:#3987e5;
 --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);
 font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:34px 22px 70px}
h1{font-size:21px;margin:0 0 4px;letter-spacing:-.3px}
.sub{color:var(--muted);font-size:13px;margin-bottom:26px}
h2{font-size:11px;text-transform:uppercase;letter-spacing:.9px;color:var(--muted);
   margin:34px 0 12px}
.card{background:var(--surface-1);border:1px solid var(--line);border-radius:10px;padding:20px}
.hero{display:flex;gap:44px;flex-wrap:wrap}
.hero .n{font-size:46px;font-weight:700;line-height:1;letter-spacing:-1.6px}
.hero .k{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-top:7px}
.hero .s{font-size:12px;color:var(--muted);margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.6px;
   color:var(--muted);font-weight:600;padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:8px 10px;border-bottom:1px solid #232321;vertical-align:top}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}
.r{text-align:right}
.note{font-size:13px;color:var(--ink2);margin:-4px 0 14px}
details summary{cursor:pointer;color:var(--muted);font-size:12px;padding:8px 0}
.tt{position:fixed;display:none;pointer-events:none;background:#000;border:1px solid var(--baseline);
    border-radius:8px;padding:8px 11px;font-size:12px;box-shadow:0 6px 22px #000c;z-index:9}
"""

JS = """
document.querySelectorAll('#pchart .bar').forEach(g=>{
  const tt=document.getElementById('tt');
  g.onmousemove=e=>{
    tt.innerHTML=g.dataset.tip;tt.style.display='block';
    tt.style.left=Math.min(e.clientX+14,innerWidth-260)+'px';tt.style.top=(e.clientY+14)+'px';};
  g.onmouseleave=()=>tt.style.display='none';
});
"""


def _phone_chart(per_phoneme: dict, top: int = 20) -> str:
    rows = sorted(per_phoneme.items(), key=lambda kv: -kv[1]["error_rate"])[:top]
    if not rows:
        return '<p class="note">No phoneme-level data.</p>'
    W, rowH, gap, padL, padR, padT = 860, 18, 6, 46, 92, 24
    H = padT + len(rows) * (rowH + gap)
    max_rate = max(0.05, max(v["error_rate"] for _, v in rows))
    plot = W - padL - padR
    ticks = [0, max_rate / 2, max_rate]
    parts = [(f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}" role="img" '
          f'aria-label="Phoneme error rate, worst first">')]
    for t in ticks:
        x = padL + t / max_rate * plot
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{padT-8}" y2="{H}" '
                 f'stroke="var(--line)" stroke-width="1"/>'
                 f'<text x="{x:.1f}" y="{padT-13}" fill="var(--muted)" font-size="10" '
                 f'text-anchor="middle">{t*100:.0f}%</text>')
    parts.append(f'<line x1="{padL}" x2="{padL}" y1="{padT-8}" y2="{H}" '
             f'stroke="var(--baseline)" stroke-width="1"/>')
    for i, (ph, v) in enumerate(rows):
        y = padT + i * (rowH + gap)
        w = max(3.0, v["error_rate"] / max_rate * plot)
        r = 4
        tip = (f"<b>{ph}</b> -{v['error_rate']*100:.0f}% error<br>"
               f"{v['n']} tokens · {v['sub']} sub · {v['del']} del")
        parts.append(
            f'<g class="bar" data-tip="{html.escape(tip, quote=True)}">'
            f'<rect x="0" y="{y-3}" width="{W}" height="{rowH+6}" fill="transparent"/>'
            f'<text x="{padL-9}" y="{y+rowH-5}" fill="var(--ink2)" font-size="11" '
            f'text-anchor="end" font-family="ui-monospace,monospace">{ph}</text>'
            f'<path d="M{padL},{y} H{padL+w-r} a{r},{r} 0 0 1 {r},{r} '
            f'V{y+rowH-r} a{r},{r} 0 0 1 -{r},{r} H{padL} Z" fill="var(--series-1)"/>'
            f'<text x="{padL+w+8}" y="{y+rowH-5}" fill="var(--muted)" font-size="11" '
            f'class="mono">{v["error_rate"]*100:.0f}%  n={v["n"]}</text></g>')
    parts.append("</svg>")
    return "".join(parts)


def _tbl(head: list[str], rows: list[list[str]], right: set[int] | None = None) -> str:
    right = right or set()
    header = "".join(f'<th class="{"r" if i in right else ""}">{html.escape(x)}</th>'
                for i, x in enumerate(head))
    body = "".join(
        "<tr>" + "".join(
            f'<td class="{"r mono" if i in right else ""}">{c}</td>'
            for i, c in enumerate(r)) + "</tr>"
        for r in rows) or f'<tr><td colspan="{len(head)}" style="color:var(--muted)">No data.</td></tr>'
    return f"<table><tr>{header}</tr>{body}</table>"


def build(res, meta: dict, out_path: str | Path) -> Path:
    result = res.dict() if hasattr(res, "dict") else res
    per_phoneme = result["per_phoneme"]

    hero = f"""<div class="hero">
      <div><div class="n">{result['wer']*100:.1f}%</div><div class="k">word error rate</div>
        <div class="s">{result['word']['errors']} errors / {result['word']['n_ref']} words</div></div>
      <div><div class="n">{result['cer']*100:.1f}%</div><div class="k">character error rate</div>
        <div class="s">less brittle on short utterances</div></div>
      <div><div class="n">{result['per']*100:.1f}%</div><div class="k">phoneme error rate</div>
        <div class="s">what to fix, not just how bad</div></div>
      <div><div class="n">{result['n_utts']}</div><div class="k">utterances scored</div>
        <div class="s">{meta.get('split','test')} split</div></div>
    </div>"""

    grp = _tbl(["Prompt set", "WER", "Words", "Sub", "Del", "Ins"],
               [[g, f"{v['rate']*100:.1f}%", str(v['n_ref']), str(v['sub']),
                 str(v['dele']), str(v['ins'])]
                for g, v in sorted(result["by_group"].items(), key=lambda x: -x[1]["rate"])],
               right={1, 2, 3, 4, 5})

    conf = _tbl(["Reference", "Heard as", "Count"],
                [[f'<span class="mono">{c["ref"]}</span>',
                  f'<span class="mono">{c["hyp"]}</span>', str(c["n"])]
                 for c in result["confusions"]], right={2})

    worst = _tbl(["WER", "Reference", "Hypothesis"],
                 [[f'{u["wer"]*100:.0f}%', html.escape(u["ref"]),
                   f'<span style="color:var(--muted)">{html.escape(u["hyp"]) or "—"}</span>']
                  for u in result["worst"]], right={0})

    diff = meta.get("difficulty") or {}
    difftbl = _tbl(["Phone", "Sampling weight", "Error rate"],
                   [[f'<span class="mono">{p}</span>', f"{w:.2f}×",
                     f'{per_phoneme.get(p, {}).get("error_rate", 0)*100:.0f}%']
                    for p, w in sorted(diff.items(), key=lambda x: -x[1])[:15]],
                   right={1, 2})

    oov = ", ".join(result["oov_words"]) or "none"

    return _write(out_path, f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>willspeak -eval report</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Evaluation report</h1>
<div class="sub">{html.escape(meta.get('label', 'run'))} · model
  <span class="mono">{html.escape(str(meta.get('model', '—')))}</span>
  · {html.escape(str(meta.get('date', '')))}</div>

<div class="card">{hero}</div>

<h2>By prompt set</h2>
<p class="note">Critical-set numbers matter most and should be read against a
closed-set classifier, not open ASR. If phonetic WER is far below functional WER,
the model has learned the script rather than the speaker.</p>
<div class="card">{grp}</div>

<h2>Which sounds the model gets wrong</h2>
<p class="note">Worst {min(20, len(per_phoneme))} phonemes by error rate. This is the feedback
loop: feed these weights back into the recording scheduler and the next session
concentrates on the sounds that are actually failing, instead of marching down a
script. Hover a bar for its counts.</p>
<div class="card"><div id="pchart">{_phone_chart(per_phoneme)}</div>
<details><summary>Show as a table</summary>
{_tbl(["Phone", "Tokens", "Correct", "Sub", "Del", "Error rate"],
      [[f'<span class="mono">{p}</span>', str(v["n"]), str(v["correct"]),
        str(v["sub"]), str(v["del"]), f'{v["error_rate"]*100:.1f}%']
       for p, v in sorted(per_phoneme.items(), key=lambda kv: -kv[1]["error_rate"])],
      right={1, 2, 3, 4, 5})}
</details></div>

<h2>Recording weights for the next session</h2>
<p class="note">Pass these to <span class="mono">Scheduler(store, difficulty=…)</span>.
A weight of 5× means prompts containing that sound get five times the priority.</p>
<div class="card">{difftbl}</div>

<h2>Most common phoneme confusions</h2>
<p class="note">Read these as articulation patterns. A consistent pattern
(say, every /t/ heard as /d/) is a systematic shift a fine-tune can learn from
more data. Scattered, inconsistent confusions mean intra-speaker variability, and
that is what the exemplar matcher exists to handle.</p>
<div class="card">{conf}</div>

<h2>Worst utterances</h2>
<div class="card">{worst}</div>

<h2>Notes</h2>
<div class="card"><p class="note" style="margin:0">
Out-of-vocabulary words (no CMUdict entry, excluded from phoneme metrics):
<span class="mono">{html.escape(oov)}</span></p></div>
</div><div class="tt" id="tt"></div><script>{JS}</script></body></html>""")


def _write(path, text: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out
