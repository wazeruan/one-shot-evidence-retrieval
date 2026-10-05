"""Dependency-free, headless experiment graphs from measured query records."""
import html
import json
import math
import os
from pathlib import Path


def chart(title, series, unit='', bounds=None):
    width, height = 720, 260
    left, top, plot_w, plot_h = 65, 25, 625, 180
    values = [float(v) for _, points, _ in series for v in points]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Graph values must be finite')
    low, high = bounds if bounds else (0, max(values, default=1) or 1)
    if high <= low:
        raise ValueError('Invalid graph bounds')
    n = max((len(points) for _, points, _ in series), default=0)
    svg = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    for i in range(5):
        value = low + (high-low)*i/4
        y = top + plot_h*(1-i/4)
        svg.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#ddd"/><text x="{left-8}" y="{y+4}" text-anchor="end">{value:.3g}</text>')
    for name, points, color in series:
        coordinates = [(left + plot_w*i/max(1,n-1), top + plot_h*(1-(float(v)-low)/(high-low))) for i,v in enumerate(points)]
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="'+ ' '.join(f'{x:.2f},{y:.2f}' for x,y in coordinates)+'"/>')
        if len(coordinates)==1:
            x,y=coordinates[0]; svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
    svg.append(f'<text x="{left}" y="{top+plot_h+25}">1</text><text x="{left+plot_w}" y="{top+plot_h+25}" text-anchor="end">{n}</text><text x="360" y="250" text-anchor="middle">Completed query order</text></svg>')
    legend=' '.join(f'<span style="color:{color}">{html.escape(name)}</span>' for name,_,color in series)
    return f'<section><h2>{html.escape(title)} {html.escape(unit)}</h2><div>{legend}</div>'+''.join(svg)+'</section>'


def running_mean(values):
    total=0.0
    result=[]
    for i,v in enumerate(values,1):
        total+=float(v); result.append(total/i)
    return result


def write_graphs(output, records, total):
    if not records:
        return
    output=Path(output)
    panels=[]
    for metric, title in [('support_recall','Running supporting-document recall'),('complete_support','Running complete evidence coverage')]:
        panels.append(chart(title, [(method, running_mean([r['metrics'][method][metric] for r in records]), color)
            for method,color in [('baseline','#2563eb'),('judged','#dc2626')]], bounds=(0,1)))
    panels.append(chart('Per-query latency', [('cosine search',[r['timing_ms']['baseline_online'] for r in records],'#2563eb'),
        ('additional judging',[r['timing_ms']['judge_extra'] for r in records],'#dc2626')], unit='(ms)'))
    winners=[max(r['bundles'],key=lambda b:b['combined']) for r in records]
    panels.append(chart('Selected bundle judge predictions',[(key,[w[key]['score'] for w in winners],color)
        for key,color in [('coverage','#059669'),('correctness','#9333ea')]], bounds=(0,1)))
    panels.append(chart('Selected bundle yes/no vocabulary mass',[(key,[w[key]['yes_no_mass'] for w in winners],color)
        for key,color in [('coverage','#059669'),('correctness','#9333ea')]], bounds=(0,1)))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Retrieval experiment progress</title><style>body{font:16px system-ui;background:#f5f7fa;color:#18212f;max-width:1500px;margin:auto;padding:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}section{background:white;padding:16px;border-radius:12px}h2{font-size:18px}svg{width:100%;font-size:12px}span{margin-right:18px}</style>'''
    page+=f'<h1>One-shot retrieval evaluation</h1><p>Completed {len(records)} / {total} queries. Reload this file for updates.</p><p>Evaluation curves, not training losses. Judge predictions are uncalibrated; correctness is not independently verified. Latency is measured sequentially, not under production load. Initial indexing and model loading are excluded from per-query curves.</p><main>'+''.join(panels)+'</main></html>'
    tmp=output/f'.progress.{os.getpid()}.tmp'
    tmp.write_text(page); tmp.replace(output/'progress.html')
    payload={'completed':len(records),'total':total,'query_ids':[r['query_id'] for r in records],
             'metrics':[{'query_id':r['query_id'],'metrics':r['metrics'],'timing_ms':r['timing_ms']} for r in records]}
    tmp=output/f'.metrics.{os.getpid()}.tmp'
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n'); tmp.replace(output/'progress.json')
