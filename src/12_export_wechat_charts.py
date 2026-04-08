"""Export ECharts charts as static PNG images optimized for WeChat articles.

Reads the same analysis data as the interactive report, but renders charts
with WeChat-optimized settings: 1080px width, larger fonts, direct labels.
Uses Playwright to screenshot each chart from a temporary HTML page.

Output: output/wechat_charts/*.png (6 charts, each 1080px wide)
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))

# Reuse data functions from report generator
from importlib import import_module
report_mod = import_module("10_generate_report")

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "wechat_charts"

# ── WeChat chart specs (per wechat-chart-design skill) ────────────────
CHART_W = 1080
COLORS = {
    "green": "#10b981", "gray": "#94a3b8", "blue": "#3b82f6",
    "amber": "#f59e0b", "red": "#ef4444", "primary": "#1b2838",
    "text": "#3f3f3f", "text_light": "#64748b", "bg": "#FFFFFF",
}
# Font sizes for 1080px static export (WeChat readable)
F = {
    "title": 32, "axis_name": 22, "axis_label": 20,
    "data_label": 18, "legend": 20, "annotation": 18,
    "mark_label": 16, "small": 14,
}


def title_js(text):
    """Generate ECharts title config for WeChat charts."""
    return f"""title: {{text:'{text}', left:'center', top:8,
        textStyle:{{fontSize:{F["title"]}, fontWeight:'bold', color:'{COLORS["text"]}'}}}},"""


def build_html(charts_js: list[dict]) -> str:
    """Build a single HTML page with multiple ECharts containers."""
    containers = ""
    scripts = ""
    for c in charts_js:
        h = c.get("height", 675)
        containers += f'<div id="{c["id"]}" style="width:{CHART_W}px;height:{h}px;background:#fff;margin:20px 0;"></div>\n'
        scripts += f"""
(function() {{
  var el = document.getElementById('{c["id"]}');
  var chart = echarts.init(el);
  chart.setOption({c["option_js"]});
}})();
"""

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>body {{ margin:0; padding:20px; background:#f0f0f0; }}</style>
</head><body>
{containers}
<script>
{scripts}
</script>
</body></html>"""


# ── Chart builders ────────────────────────────────────────────────────

def chart_avg_vs_peak(peak_data_full):
    """Scatter: daily velocity vs peak week velocity for full backdrop."""
    s_data = [g for g in peak_data_full if g["group"] == "success"]
    b_data = [g for g in peak_data_full if g["group"] != "success"]

    # Confidence ellipse (log-space)
    def conf_ellipse_js(data, x_key, y_key):
        pts = [d for d in data if d[x_key] > 0 and d[y_key] > 0]
        if len(pts) < 3:
            return "[]"
        n = len(pts)
        xs = [np.log10(max(d[x_key], 0.3)) for d in pts]
        ys = [np.log10(d[y_key]) for d in pts]
        mx, my = np.mean(xs), np.mean(ys)
        cxx = np.var(xs, ddof=1)
        cyy = np.var(ys, ddof=1)
        cxy = np.cov(xs, ys, ddof=1)[0][1]
        tr = cxx + cyy
        det = cxx * cyy - cxy * cxy
        disc = np.sqrt(max(tr * tr / 4 - det, 0))
        l1 = tr / 2 + disc
        l2 = max(tr / 2 - disc, 0.001)
        ang = np.arctan2(2 * cxy, cxx - cyy) / 2
        sigma = 2
        a, b = sigma * np.sqrt(l1), sigma * np.sqrt(l2)
        result = []
        for i in range(81):
            t = 2 * np.pi * i / 80
            ex, ey = a * np.cos(t), b * np.sin(t)
            rx = ex * np.cos(ang) - ey * np.sin(ang) + mx
            ry = ex * np.sin(ang) + ey * np.cos(ang) + my
            result.append([round(10**rx, 2), round(10**ry, 2)])
        return json.dumps(result)

    s_ellipse = conf_ellipse_js(s_data, "avg_daily", "peak_week")
    b_ellipse = conf_ellipse_js(b_data, "avg_daily", "peak_week")

    b_scatter = json.dumps([[max(d["avg_daily"], 0.3), d["peak_week"]]
                            for d in b_data if d["avg_daily"] > 0])
    s_scatter = json.dumps([[max(d["avg_daily"], 0.3), d["peak_week"], d["name"]]
                            for d in s_data if d["avg_daily"] > 0])

    option_js = f"""{{
        {title_js('每款游戏的日常增速 vs 巅峰周增速')}
        tooltip: {{trigger:'item', confine:true, textStyle:{{fontSize:{F["small"]}}},
            formatter: function(p) {{
                var d = p.data;
                if (!d[2]) return '';
                return '<b>'+d[2]+'</b><br>日常增速: '+Math.round(d[0])+'/天<br>巅峰周增速: '+Math.round(d[1])+'/天';
            }}
        }},
        legend: {{data:['爆款','Backdrop样本'], bottom:5, textStyle:{{fontSize:{F["legend"]}}}}},
        grid: {{left:80, right:50, top:65, bottom:75, containLabel:true}},
        xAxis: {{type:'log', name:'日常增速（中位数，人/天）', nameLocation:'middle', nameGap:40,
            nameTextStyle:{{fontSize:{F["axis_name"]}}},
            min:0.3, axisLabel:{{fontSize:{F["axis_label"]},
            formatter:function(v){{return v>=1000?Math.round(v/1000)+'K':v;}}}}}},
        yAxis: {{type:'log', name:'巅峰周增速（人/天）', nameLocation:'end', nameGap:15,
            nameTextStyle:{{fontSize:{F["axis_name"]}}},
            min:1, axisLabel:{{fontSize:{F["axis_label"]},
            formatter:function(v){{return v>=1000?Math.round(v/1000)+'K':v;}}}}}},
        series: [
            {{name:'Backdrop样本', type:'line', data:{b_ellipse}, smooth:true, symbol:'none',
                lineStyle:{{color:'{COLORS["gray"]}', width:1.5, type:'dashed'}},
                areaStyle:{{color:'rgba(148,163,184,0.08)'}}, silent:true, z:1}},
            {{name:'爆款', type:'line', data:{s_ellipse}, smooth:true, symbol:'none',
                lineStyle:{{color:'{COLORS["green"]}', width:1.5, type:'dashed'}},
                areaStyle:{{color:'rgba(16,185,129,0.10)'}}, silent:true, z:1}},
            {{name:'Backdrop样本', type:'scatter', data:{b_scatter},
                symbolSize:5, itemStyle:{{color:'{COLORS["gray"]}', opacity:0.2}}, z:2}},
            {{name:'爆款', type:'scatter', data:{s_scatter},
                symbolSize:14, itemStyle:{{color:'{COLORS["green"]}', opacity:0.9}}, z:3,
                label:{{show:true, formatter:function(p){{return p.data[2];}},
                    fontSize:{F["small"]}, position:'right', color:'#065f46'}},
                markLine: {{silent:true, symbol:'none',
                    lineStyle:{{color:'#b0b8c4', type:'dashed', width:1.5}},
                    data:[
                        {{yAxis:900, label:{{formatter:'巅峰周≥900', position:'insideEndTop',
                            fontSize:{F["mark_label"]}, color:'{COLORS["text_light"]}'}}}},
                        {{xAxis:40, label:{{formatter:'日常≥40', position:'insideEndTop',
                            fontSize:{F["mark_label"]}, color:'{COLORS["text_light"]}'}}}},
                    ]
                }}
            }},
        ]
    }}"""
    return {"id": "chart_avg_vs_peak", "height": 720, "option_js": option_js}


def chart_trajectories(report_data):
    """Line chart: follower growth trajectories vs control envelope."""
    traj = report_data["trajectories"]
    ctrl = report_data["control_envelope"]
    game_colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"]

    series_js = []
    # Control envelope
    series_js.append(f"""{{
        name:'对照组中位数', type:'line', data:{json.dumps(ctrl)}, smooth:true,
        lineStyle:{{color:'{COLORS["gray"]}', width:2.5, type:'dashed'}}, symbol:'none',
        itemStyle:{{color:'{COLORS["gray"]}'}},
        areaStyle:{{color:'rgba(148,163,184,0.08)'}}
    }}""")
    idx = 0
    for name, data in traj.items():
        c = game_colors[idx % 5]
        series_js.append(f"""{{
            name:{json.dumps(name)}, type:'line', data:{json.dumps(data)}, smooth:true,
            lineStyle:{{color:'{c}', width:3}}, symbol:'none',
            itemStyle:{{color:'{c}'}}
        }}""")
        idx += 1

    option_js = f"""{{
        {title_js('3款低调爆款的关注者增长轨迹 vs 对照组')}
        tooltip: {{trigger:'axis', confine:true, textStyle:{{fontSize:{F["small"]}}}}},
        legend: {{bottom:0, textStyle:{{fontSize:{F["legend"]}}}}},
        grid: {{left:90, right:40, top:60, bottom:95}},
        xAxis: {{type:'value', name:'距上线天数', nameLocation:'middle', nameGap:32,
            nameTextStyle:{{fontSize:{F["axis_name"]}}},
            inverse:true, min:-7, max:200,
            axisLabel:{{fontSize:{F["axis_label"]},
                formatter:function(v){{return v>0?'T-'+v:v===0?'上线':'T+'+Math.abs(v);}}}}}},
        yAxis: {{type:'value', name:'关注者总数', nameLocation:'end', nameGap:15,
            nameTextStyle:{{fontSize:{F["axis_name"]}}},
            axisLabel:{{fontSize:{F["axis_label"]},
                formatter:function(v){{return v>=1000?Math.round(v/1000)+'K':v;}}}}}},
        series: [{','.join(series_js)}]
    }}"""
    return {"id": "chart_trajectories", "height": 700, "option_js": option_js}


def chart_peak_annotated(report_data):
    """Line chart with peak annotations for selected games."""
    games = report_data["peak_annotated"]
    game_colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444"]

    series_js = []
    annotations = []
    for idx, game in enumerate(games):
        c = game_colors[idx % 4]
        series_js.append(f"""{{
            name:{json.dumps(game["name"])}, type:'line',
            data:{json.dumps(game["data"])}, smooth:true,
            lineStyle:{{color:'{c}', width:2.5}}, symbol:'none'
        }}""")
        if game.get("peak_dtl"):
            annotations.append({
                "name": game["name"] + " 巅峰",
                "coord": [game["peak_dtl"], game["peak_week"]],
                "symbol": "pin", "symbolSize": 40,
                "itemStyle": {"color": c},
                "label": {"show": True,
                          "formatter": str(round(game["peak_week"])) + "/天",
                          "fontSize": F["data_label"], "position": "top"},
            })

    ann_json = json.dumps(annotations, ensure_ascii=False)

    option_js = f"""{{
        tooltip: {{trigger:'axis', confine:true, textStyle:{{fontSize:{F["small"]}}},
            formatter: function(params) {{
                var s = 'T-'+params[0].value[0]+' 天<br>';
                params.forEach(function(p){{if(p.value[1]!=null)s+=p.marker+p.seriesName+': '+Math.round(p.value[1])+'/天<br>';}});
                return s;
            }}
        }},
        {title_js('4款游戏的7日滚动均增轨迹与巅峰周')}
        legend: {{bottom:0, textStyle:{{fontSize:{F["legend"]}}}}},
        grid: {{left:90, right:110, top:60, bottom:95}},
        xAxis: {{type:'value', name:'距上线天数', inverse:true, min:0, max:300,
            nameLocation:'middle', nameGap:32,
            nameTextStyle:{{fontSize:{F["axis_name"]}}},
            axisLabel:{{fontSize:{F["axis_label"]}, formatter:function(v){{return 'T-'+v;}}}}}},
        yAxis: {{type:'value', name:'增速（人/天）', nameLocation:'end', nameGap:15,
            nameTextStyle:{{fontSize:{F["axis_name"]}}},
            axisLabel:{{fontSize:{F["axis_label"]},
                formatter:function(v){{return v>=1000?Math.round(v/1000)+'K':v;}}}}}},
        series: [{','.join(series_js)},
            {{type:'scatter', data:{json.dumps([a["coord"] for a in annotations])},
                symbolSize:0, itemStyle:{{opacity:0}},
                markPoint:{{data:{ann_json}}},
                silent:true, z:10}}
        ]
    }}"""
    return {"id": "chart_peak_annotated", "height": 675, "option_js": option_js}


def chart_windows(report_data):
    """Grouped bar chart: velocity by time window."""
    wd = report_data["velocity_windows"]
    windows = ['T-360→T-180', 'T-180→T-90', 'T-90→T-30', 'T-30→上线']
    s_vals = []
    c_vals = []
    for w in windows:
        sd = next((d for d in wd if d["window"] == w and d["group"] == "success"), None)
        cd = next((d for d in wd if d["window"] == w and d["group"] == "control"), None)
        s_vals.append(sd["median"] if sd else 0)
        c_vals.append(cd["median"] if cd else 0)
    ratios = [f"{s / c:.1f}" if c > 0 else "-" for s, c in zip(s_vals, c_vals)]

    option_js = f"""{{
        {title_js('不同时间窗口的关注者增速对比')}
        tooltip: {{trigger:'axis', confine:true, textStyle:{{fontSize:{F["small"]}}}}},
        legend: {{data:['爆款','对照组'], bottom:5, textStyle:{{fontSize:{F["legend"]}}}}},
        grid: {{left:80, right:30, top:70, bottom:60}},
        xAxis: {{type:'category', data:{json.dumps(windows)},
            axisLabel:{{fontSize:{F["axis_label"]}}}}},
        yAxis: {{type:'value', name:'增速（人/天）',
            nameTextStyle:{{fontSize:{F["axis_name"]}}},
            axisLabel:{{fontSize:{F["axis_label"]}}}}},
        series: [
            {{name:'爆款', type:'bar', data:{json.dumps(s_vals)},
                itemStyle:{{color:'{COLORS["green"]}'}}, barGap:'20%',
                label:{{show:true, position:'top', fontSize:{F["data_label"]},
                    formatter:function(p){{return Math.round(p.value);}}}}}},
            {{name:'对照组', type:'bar', data:{json.dumps(c_vals)},
                itemStyle:{{color:'{COLORS["gray"]}'}},
                label:{{show:true, position:'top', fontSize:{F["data_label"]},
                    formatter:function(p){{return Math.round(p.value);}}}}}},
        ],
        graphic: {json.dumps([
            {"type": "text", "left": f"{15 + i * 21}%", "top": 48,
             "style": {"text": f"{r}倍", "fill": COLORS["red"],
                       "fontSize": F["data_label"] + 2, "fontWeight": "bold"}}
            for i, r in enumerate(ratios)
        ])}
    }}"""
    return {"id": "chart_windows", "height": 580, "option_js": option_js}


def chart_threshold(report_data):
    """PR curve: hit rate vs catch rate by threshold."""
    td = report_data["threshold_sweep"]
    prec_data = json.dumps([[d["threshold"], d["precision"]] for d in td])
    rec_data = json.dumps([[d["threshold"], d["recall"]] for d in td])

    option_js = f"""{{
        {title_js('命中率 vs 不漏球率随门槛变化')}
        tooltip: {{trigger:'item', confine:true, textStyle:{{fontSize:{F["small"]}}},
            formatter: function(p) {{
                var d = p.data;
                if (!d || d.length < 2) return '';
                return p.marker+p.seriesName+': '+(d[1]*100).toFixed(1)+'%<br>门槛: ≥'+Math.round(d[0])+'/天';
            }}
        }},
        legend: {{data:['命中率','不漏球率'], bottom:0, textStyle:{{fontSize:{F["legend"]}}}}},
        grid: {{left:70, right:40, top:65, bottom:100}},
        xAxis: {{type:'log', name:'巅峰周增速门槛（人/天）', nameLocation:'middle', nameGap:32,
            nameTextStyle:{{fontSize:{F["axis_name"]}}},
            min:50, max:10000,
            axisLabel:{{fontSize:{F["axis_label"]}}}}},
        yAxis: {{type:'value', name:'比率', nameLocation:'end', nameGap:15,
            nameTextStyle:{{fontSize:{F["axis_name"]}}}, max:1,
            axisLabel:{{fontSize:{F["axis_label"]},
                formatter:function(v){{return (v*100)+'%';}}}}}},
        series: [
            {{name:'命中率', type:'line', data:{prec_data},
                lineStyle:{{color:'{COLORS["green"]}', width:3}}, symbol:'none', smooth:true,
                itemStyle:{{color:'{COLORS["green"]}'}},
                markLine: {{silent:true, symbol:'none', data:[
                    {{xAxis:400, lineStyle:{{color:'{COLORS["green"]}', type:'dashed', width:2}},
                        label:{{formatter:'广角 400', fontSize:{F["mark_label"]},
                            color:'{COLORS["green"]}', lineHeight:18}}}},
                    {{xAxis:900, lineStyle:{{color:'{COLORS["amber"]}', type:'dashed', width:2}},
                        label:{{formatter:'精选 900', fontSize:{F["mark_label"]},
                            color:'{COLORS["amber"]}', lineHeight:18}}}},
                    {{xAxis:1900, lineStyle:{{color:'{COLORS["red"]}', type:'dashed', width:2}},
                        label:{{formatter:'高确信 1900', fontSize:{F["mark_label"]},
                            color:'{COLORS["red"]}', lineHeight:18}}}},
                ]}}}},
            {{name:'不漏球率', type:'line', data:{rec_data},
                lineStyle:{{color:'{COLORS["blue"]}', width:3}}, symbol:'none', smooth:true,
                itemStyle:{{color:'{COLORS["blue"]}'}}}},
        ]
    }}"""
    return {"id": "chart_threshold", "height": 720, "option_js": option_js}


def chart_strategy_quality(report_data):
    """Grouped bar: revenue quality by strategy tier."""
    sq = report_data["strategy_quality"]
    tiers = sq["tiers"]
    labels = json.dumps([t["label"] for t in tiers])

    option_js = f"""{{
        tooltip: {{trigger:'axis', confine:true, textStyle:{{fontSize:{F["small"]}}}}},
        {title_js('三种策略的商业表现对比')}
        legend: {{data:['收入>$10M占比','收入>$50M占比','Top-100占比'],
            bottom:0, textStyle:{{fontSize:{F["legend"]}}}}},
        grid: {{left:70, right:30, top:60, bottom:70}},
        xAxis: {{type:'category', data:{labels},
            axisLabel:{{fontSize:{F["axis_label"]}}}}},
        yAxis: {{type:'value', max:1,
            axisLabel:{{fontSize:{F["axis_label"]},
                formatter:function(v){{return (v*100)+'%';}}}}}},
        series: [
            {{name:'收入>$10M占比', type:'bar',
                data:{json.dumps([t["pct_10m"] for t in tiers])},
                itemStyle:{{color:'{COLORS["green"]}'}}, barGap:'15%',
                label:{{show:true, position:'top', fontSize:{F["data_label"]},
                    formatter:function(p){{return Math.round(p.value*100)+'%';}}}}}},
            {{name:'收入>$50M占比', type:'bar',
                data:{json.dumps([t["pct_50m"] for t in tiers])},
                itemStyle:{{color:'{COLORS["blue"]}'}},
                label:{{show:true, position:'top', fontSize:{F["data_label"]},
                    formatter:function(p){{return Math.round(p.value*100)+'%';}}}}}},
            {{name:'Top-100占比', type:'bar',
                data:{json.dumps([t["pct_top100"] for t in tiers])},
                itemStyle:{{color:'{COLORS["amber"]}'}},
                label:{{show:true, position:'top', fontSize:{F["data_label"]},
                    formatter:function(p){{return Math.round(p.value*100)+'%';}}}}}},
        ]
    }}"""
    return {"id": "chart_strategy_quality", "height": 580, "option_js": option_js}


# ── Main ──────────────────────────────────────────────────────────────

def run():
    print("Loading data...")
    af, fwd, bd, success_names, name_lookup = report_mod.load_all_data()

    print("Extracting chart data...")
    success_ids = set(success_names.keys())
    peak_data = report_mod.extract_peak_week_data(fwd, name_lookup)
    peak_data_full = report_mod.extract_peak_week_from_histories(success_ids, name_lookup)

    # Build report_data dict with what we need
    report_data = {}
    report_data["trajectories"] = report_mod.extract_trajectories(
        fwd, ["3164500", "3241660", "2379780"], name_lookup)  # Schedule I, R.E.P.O., Balatro
    report_data["control_envelope"] = report_mod.extract_control_envelope(fwd)
    report_data["peak_annotated"] = report_mod.extract_trajectory_with_peak(
        fwd, ["1808500", "2767030", "1363080", "3164500"], name_lookup, peak_data)
        # ARC Raiders, Marvel Rivals, Manor Lords, Schedule I
    report_data["velocity_windows"] = report_mod.extract_velocity_windows(fwd)
    report_data["threshold_sweep"] = report_mod.extract_threshold_sweep(peak_data_full, avg_floor=20)
    report_data["strategy_quality"] = report_mod.extract_strategy_quality(peak_data_full)

    print("Building chart configurations...")
    charts = [
        chart_avg_vs_peak(peak_data_full),
        chart_trajectories(report_data),
        chart_peak_annotated(report_data),
        chart_windows(report_data),
        chart_threshold(report_data),
        chart_strategy_quality(report_data),
    ]

    html = build_html(charts)

    # Write temp HTML
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8")
    tmp.write(html)
    tmp.close()
    tmp_path = tmp.name
    print(f"  Temp HTML: {tmp_path}")

    # Screenshot with Playwright
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Rendering charts with Playwright...")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": CHART_W + 60, "height": 900})
        page.goto(f"file://{tmp_path}")
        page.wait_for_timeout(3000)  # Wait for ECharts to render

        for c in charts:
            el = page.query_selector(f"#{c['id']}")
            if el:
                path = OUTPUT_DIR / f"{c['id']}.png"
                el.screenshot(path=str(path))
                print(f"  ✓ {c['id']}.png ({CHART_W}×{c['height']})")
            else:
                print(f"  ✗ {c['id']} not found!")

        browser.close()

    # Cleanup
    Path(tmp_path).unlink(missing_ok=True)
    print(f"\nDone! Charts exported to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()
