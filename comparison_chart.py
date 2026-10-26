"""Accessible animated HTML bars for the app's saved-model metric comparison.

The chart uses the evaluated comparison table directly. It performs no model
training, network requests, JavaScript execution, or metric recalculation.
"""

import hashlib
from html import escape
import json
import math

from utils import METRICS


def build_metric_chart_html(comparison, metric, animation_key=""):
    """Return self-contained HTML/CSS for a fixed-scale model comparison.

    ``comparison`` must have ``Model`` and the selected metric column. Missing
    or non-finite scores remain N/A. MCC uses a signed -1..1 axis; every other
    metric uses 0..1. A changed metric, table, or replay key restarts the CSS
    animation. All labels and values remain readable without animation.
    """
    if metric not in METRICS:
        raise ValueError(f"Choose one of the supported metrics: {', '.join(METRICS)}.")
    if "Model" not in comparison.columns or metric not in comparison.columns:
        raise ValueError("The comparison table must contain Model and the selected metric column.")
    if comparison.empty:
        raise ValueError("The comparison table must contain at least one model.")

    signed = metric == "MCC"
    minimum, maximum = (-1.0, 1.0) if signed else (0.0, 1.0)
    records = []
    for model_name, raw_score in comparison[["Model", metric]].itertuples(index=False, name=None):
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = None
        if score is not None and not math.isfinite(score):
            score = None
        if score is not None and not minimum <= score <= maximum:
            raise ValueError(f"{metric} scores must fall between {minimum:g} and {maximum:g}.")
        records.append((str(model_name), score))

    available = [score for _, score in records if score is not None]
    best = max(available) if available else None
    fingerprint = json.dumps([metric, records, str(animation_key)], ensure_ascii=True, allow_nan=False)
    token = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    chart_id = f"metric-chart-{token}"
    animation_name = f"metric-bars-enter-{token}"

    rows = []
    for index, (model_name, score) in enumerate(records):
        is_best = score is not None and best is not None and math.isclose(score, best, rel_tol=1e-10, abs_tol=1e-12)
        classes = "metric-chart-row" + (" is-best" if is_best else "")
        if score is None:
            classes += " is-unavailable"
        value = f"{score:.4f}" if score is not None else "N/A"
        raw_value = str(score) if score is not None else ""
        badge = '<span class="metric-chart-best">Best</span>' if is_best else ""

        if score is None:
            fill = '<span class="metric-chart-unavailable" aria-hidden="true">Not available</span>'
        else:
            width = abs(score) * (50 if signed else 100)
            left = (50 - width if score < 0 else 50) if signed else 0
            negative = " is-negative" if score < 0 else ""
            fill = (
                f'<span class="metric-bar-fill{negative}" aria-hidden="true" '
                f'data-left="{left:.8f}" data-width="{width:.8f}" '
                f'style="left:{left:.8f}%;width:{width:.8f}%;animation-delay:{index * .065:.3f}s"></span>'
            )
        zero = '<span class="metric-chart-zero" aria-hidden="true"></span>' if signed else ""
        accessible_score = f"{metric}: {value}" if score is not None else f"{metric}: not available"
        rows.append(
            f'<div class="{classes}" role="listitem" data-model="{escape(model_name, quote=True)}" '
            f'data-value="{raw_value}" data-available="{str(score is not None).lower()}">'
            f'<div class="metric-chart-label"><span>{escape(model_name)}</span>{badge}</div>'
            f'<div class="metric-chart-track" aria-hidden="true">{fill}{zero}</div>'
            f'<span class="metric-chart-score" aria-label="{escape(accessible_score, quote=True)}">{value}</span>'
            '</div>'
        )

    tick_values = [-1, -.5, 0, .5, 1] if signed else [0, .25, .5, .75, 1]
    ticks = "".join(
        f'<span style="left:{position * 25}%">{value:g}</span>'
        for position, value in enumerate(tick_values)
    )
    caption = (
        "Higher values are better. Teal highlights the best score; tied models share the highlight. "
        + ("MCC uses a fixed −1 to 1 scale, with zero in the center and negative scores extending left."
           if signed else "The axis stays fixed from 0 to 1 when you switch models or upload test data.")
    )
    if any(score is None for _, score in records):
        caption += " N/A means the metric could not be calculated; it is not a zero score."

    styles = f"""
<style>
#{chart_id}.metric-comparison-chart {{
  margin:1rem 0 .3rem;padding:1.15rem 1.1rem .95rem;
  border:1px solid #e1e9ef;border-radius:14px;
  background:linear-gradient(145deg,#f7fafc,#ffffff);color:#172b45;
  font-family:inherit;box-sizing:border-box;
}}
#{chart_id} * {{box-sizing:border-box;}}
#{chart_id} .metric-chart-heading {{
  display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;
  gap:.35rem 1rem;margin-bottom:1.05rem;
}}
#{chart_id} .metric-chart-title {{font-size:.95rem;font-weight:700;letter-spacing:-.015em;}}
#{chart_id} .metric-chart-count {{font-size:.73rem;color:#63748b;}}
#{chart_id} .metric-chart-row,#{chart_id} .metric-chart-axis-row {{
  display:grid;grid-template-columns:minmax(0,230px) minmax(0,1fr) 84px;
  align-items:center;gap:18px;
}}
#{chart_id} .metric-chart-row {{padding:.8rem .6rem;border-radius:9px;}}
#{chart_id} .metric-chart-row + .metric-chart-row {{margin-top:.1rem;}}
#{chart_id} .metric-chart-row.is-best {{background:#eaf5f0;}}
#{chart_id} .metric-chart-label {{
  display:flex;align-items:center;flex-wrap:wrap;gap:.45rem;
  min-width:0;color:#25415a;font-size:.82rem;font-weight:550;line-height:1.45;
  overflow-wrap:anywhere;
}}
#{chart_id} .metric-chart-best {{
  display:inline-block;padding:.1rem .4rem;border-radius:5px;
  background:#d1ece2;color:#166350;font-size:.61rem;font-weight:750;letter-spacing:.02em;
}}
#{chart_id} .metric-chart-track {{
  position:relative;min-width:0;height:16px;overflow:hidden;border-radius:5px;
  background-color:#e7eef3;
  background-image:linear-gradient(90deg,transparent calc(100% - 1px),#ccdae4 0);
  background-size:25% 100%;
}}
#{chart_id} .metric-bar-fill {{
  display:block;position:absolute;height:100%;border-radius:4px;
  background:linear-gradient(90deg,#78afbd,#3a7e91);transform-origin:left center;
  animation:{animation_name} .85s cubic-bezier(.18,.75,.24,1) both;
}}
#{chart_id} .metric-bar-fill.is-negative {{
  transform-origin:right center;background:linear-gradient(90deg,#bd7168,#d4a096);
}}
#{chart_id} .is-best .metric-bar-fill {{background:linear-gradient(90deg,#38ac96,#087f75);}}
#{chart_id} .metric-chart-zero {{
  position:absolute;left:50%;top:0;bottom:0;width:2px;background:#40627a88;
  transform:translateX(-1px);pointer-events:none;
}}
#{chart_id} .metric-chart-score {{
  text-align:right;white-space:nowrap;font-size:.86rem;font-weight:650;
  color:#29485f;font-variant-numeric:tabular-nums;
}}
#{chart_id} .is-best .metric-chart-score {{color:#086d5f;}}
#{chart_id} .is-unavailable .metric-chart-track {{background:#eef2f5;}}
#{chart_id} .is-unavailable .metric-chart-score {{color:#63748b;}}
#{chart_id} .metric-chart-unavailable {{
  display:block;text-align:center;color:#718295;font-size:.6rem;line-height:16px;
}}
#{chart_id} .metric-chart-axis-row {{padding:0 .6rem;margin:.4rem 0 .75rem;}}
#{chart_id} .metric-chart-axis {{grid-column:2;position:relative;height:17px;color:#718295;font-size:.64rem;}}
#{chart_id} .metric-chart-axis span {{position:absolute;top:0;transform:translateX(-50%);}}
#{chart_id} .metric-chart-axis span:first-child {{transform:none;}}
#{chart_id} .metric-chart-axis span:last-child {{transform:translateX(-100%);}}
#{chart_id} .metric-chart-caption {{
  border-top:1px solid #e2eaf0;padding-top:.7rem;color:#63748b;
  font-size:.73rem;line-height:1.65;
}}
@keyframes {animation_name} {{from {{transform:scaleX(0);}} to {{transform:scaleX(1);}}}}
@media (max-width:600px) {{
  #{chart_id}.metric-comparison-chart {{padding:1rem .65rem .85rem;}}
  #{chart_id} .metric-chart-row {{grid-template-columns:minmax(0,1fr) 84px;gap:.5rem 12px;padding:.75rem .45rem;}}
  #{chart_id} .metric-chart-label {{grid-column:1 / -1;font-size:.82rem;}}
  #{chart_id} .metric-chart-track {{grid-column:1;}}
  #{chart_id} .metric-chart-score {{grid-column:2;}}
  #{chart_id} .metric-chart-axis-row {{grid-template-columns:minmax(0,1fr) 84px;gap:12px;padding:0 .45rem;}}
  #{chart_id} .metric-chart-axis {{grid-column:1;}}
}}
@media (prefers-reduced-motion:reduce) {{
  #{chart_id} .metric-bar-fill {{animation:none !important;transform:scaleX(1) !important;}}
}}
</style>
"""
    return (
        styles
        + f'<figure id="{chart_id}" class="metric-comparison-chart" data-metric="{metric}" '
        f'data-domain-min="{minimum:g}" data-domain-max="{maximum:g}" '
        f'aria-labelledby="{chart_id}-title" aria-describedby="{chart_id}-caption">'
        '<div class="metric-chart-heading">'
        f'<span id="{chart_id}-title" class="metric-chart-title">{metric} by model</span>'
        f'<span class="metric-chart-count">{len(records)} saved classifiers</span></div>'
        f'<div role="list" aria-label="{metric} scores">'
        + "".join(rows)
        + '</div><div class="metric-chart-axis-row" aria-hidden="true">'
        f'<div class="metric-chart-axis">{ticks}</div></div>'
        f'<figcaption id="{chart_id}-caption" class="metric-chart-caption">{caption}</figcaption>'
        '</figure>'
    )
