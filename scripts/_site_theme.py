"""The public site's paint (2026-10-01): the Midcentury research-post look
(midcentury-research-post/references/paint.md) applied to the static pages
in results/pages -- index (the project post), leaderboard.html and every
population page. Black ground, white type, one lime accent for "ours /
best / active", white/14 hairlines, greys for comparison series, mono
uppercase labels, radii 6 (controls) / 8 (stages) / square elsewhere, one
easing curve, no gradients or shadows, nothing visible under 16 px,
reduced motion honoured. Structure borrowed from physera.ai/research and
research.withdavid.ai: a sticky numbered page nav on the left, an eyebrow
over every heading, a stats row under the hook, and leaderboard tables
with the confidence interval beside the point estimate.

ITC Avant Garde Gothic is licensed and not shipped here; the stack falls
back to Jost (Google Fonts, the closest free geometric) and shows Avant
Garde when the viewer has it installed.
"""
import html
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mcphysbench.branding import PUBLIC_NAME

FONT_LINK = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link href="https://fonts.googleapis.com/css2?family=Jost:wght@400;500;600&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">')

CSS = """
:root{--bg:#000;--fg:#fff;--fg2:rgba(255,255,255,.9);--fg3:rgba(255,255,255,.8);--muted:#8a8a8a;--accent:#A9FF3C;
--border:rgba(255,255,255,.14);--step:#111;--step2:#1a1a1a;--g1:#f6f6f6;--g2:#bdc0b7;--g3:#8a8a8a;--err:#FB2E00;
--ease:cubic-bezier(.16,1,.3,1);
--sans:"ITC Avant Garde Gothic Std","ITC Avant Garde Gothic","Avant Garde",Jost,"Century Gothic",system-ui,sans-serif;
--mono:"Roboto Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
html{color-scheme:dark;scroll-behavior:smooth}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*,*::before,*::after{transition:none!important;animation:none!important}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:400 16px/1.55 var(--sans);-webkit-font-smoothing:antialiased;overflow-x:hidden}
a{color:var(--accent);text-decoration:underline;text-underline-offset:3px;text-decoration-thickness:1px}
a:hover{color:var(--fg)}
.frame.wide{max-width:1520px}
.frame{max-width:1280px;margin:0 auto;padding:40px 24px 120px;display:grid;grid-template-columns:200px minmax(0,1fr);gap:0 56px}
.rail{position:sticky;top:40px;align-self:start}
.rail .brand{display:block;font:500 16px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--fg);text-decoration:none;margin:0 0 28px}
.rail .brand:hover{color:var(--accent)}
.rail a.item{display:block;font:400 16px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);text-decoration:none;
padding:7px 0 7px 16px;border-left:1px solid var(--border);transition:color .4s var(--ease),border-color .4s var(--ease)}
.rail a.item:hover{color:var(--fg)}
.rail a.item.on{color:var(--fg);border-left-color:var(--accent)}
.rail a.item .n{color:var(--muted);margin-right:10px}
.rail a.item.on .n{color:var(--accent)}
.rail .links{margin-top:28px;display:flex;flex-direction:column;gap:6px}
.rail .links a{font:400 16px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);text-decoration:none}
.rail .links a:hover{color:var(--accent)}
article{min-width:0}
.prose{max-width:760px}
.kicker{font:500 16px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--accent);margin:0 0 20px}
h1{font-size:48px;line-height:1.05;font-weight:500;letter-spacing:-.01em;margin:0 0 20px;max-width:900px}
h1 .sub{display:block;font-size:24px;line-height:1.3;color:var(--fg3);font-weight:400;margin-top:14px;letter-spacing:0}
.hook{font-size:20px;line-height:1.5;color:var(--fg2);margin:0 0 36px;max-width:760px}
.eyebrow{font:400 16px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:88px 0 10px}
.eyebrow:first-of-type{margin-top:72px}
h2{font-size:32px;line-height:1.15;font-weight:500;letter-spacing:-.01em;margin:0 0 18px}
h3{font-size:24px;line-height:1.25;font-weight:500;margin:40px 0 12px}
p{margin:0 0 16px;color:var(--fg2)}
p.note{color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:24px 32px;border-top:1px solid var(--border);padding:20px 0 0;margin:0 0 8px}
.stats .v{font-size:36px;line-height:1.1;font-weight:500;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.stats .k{font:400 16px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-top:8px}
.stats .s{color:var(--muted);margin-top:4px}
.btn{display:inline-block;padding:11px 18px;border-radius:6px;background:var(--accent);color:#000;font-weight:500;text-decoration:none;
border:1px solid var(--accent);transition:transform .4s var(--ease),background .4s var(--ease)}
.btn:hover{color:#000;transform:translateY(-1px)}
.btn.outline{background:none;color:var(--fg);border-color:var(--border)}
.btn.outline:hover{color:var(--fg);border-color:var(--fg)}
.scroll{overflow-x:auto;max-width:100%;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:12px 14px 12px 0;border-bottom:1px solid var(--border);text-align:left;vertical-align:middle;white-space:nowrap}
th{font:400 16px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:400}
th.r,td.r{text-align:right}
td.name{font-weight:500}
td.name a{color:var(--fg);text-decoration:none}td.name a:hover{color:var(--accent)}
tr.ref td{color:var(--g2)}
tr.grey td{color:var(--muted)}
tr.top td.name{color:var(--accent)}
.rank{display:inline-block;min-width:28px;font:500 16px/1.3 var(--mono);color:var(--fg)}
tr.top .rank{color:var(--accent)}
.tag{display:inline-block;font:400 16px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);
border:1px solid var(--border);border-radius:6px;padding:1px 8px;margin-left:8px;white-space:normal;max-width:100%}
.tag.err{color:var(--err);border-color:var(--err)}
.tag.ok{color:var(--accent);border-color:var(--accent)}
.ci{color:var(--muted)}
.sig{color:var(--accent)}.neg{color:var(--err)}.ns{color:var(--muted)}
.bar{position:relative;height:6px;background:var(--step2);overflow:hidden;width:140px;margin-top:6px}
.bar>i{position:absolute;left:0;top:0;bottom:0;background:var(--g2)}
tr.top .bar>i{background:var(--accent)}
.stack{display:flex;height:10px;overflow:hidden;background:var(--step2);min-width:160px}
.stack>i{display:block;height:100%}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.chip{display:inline-block;font:400 16px/1.3 var(--mono);padding:1px 8px;border-radius:6px;background:var(--step2);color:var(--fg3);white-space:nowrap}
.chip i{display:inline-block;width:10px;height:10px;margin-right:7px;vertical-align:-1px;font-style:normal}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 24px}
.tabs button{font:400 16px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.04em;padding:8px 12px;border-radius:6px;border:1px solid var(--border);
background:none;color:var(--fg3);cursor:pointer;transition:color .4s var(--ease),border-color .4s var(--ease)}
.tabs button:hover{color:var(--fg);border-color:var(--fg)}
.tabs button.on{background:var(--accent);border-color:var(--accent);color:#000}
.js .panel{display:none}.js .panel.on{display:block}
.panel>.head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 16px;margin:0 0 12px}
.panel>.head h3{margin:0}
.rules{background:var(--step);border:1px solid var(--border);padding:20px 24px;white-space:pre-wrap;color:var(--fg3);font-size:16px;line-height:1.55}
.noteblock{background:var(--step);padding:18px 22px;margin:0 0 24px;max-width:760px}
.noteblock .lbl{font:400 16px/1.3 var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 8px}
.noteblock p:last-child{margin:0}
.stage{border:1px solid var(--border);border-radius:8px;overflow:hidden;background:#000}
.stage video{display:block;width:100%;height:auto;background:#000}
figure{margin:0}
figcaption{font:400 16px/1.4 var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-top:10px}
figcaption b{color:var(--fg);font-weight:500}
.reel{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px 24px;margin:0 0 8px}
.reel a{text-decoration:none;color:inherit;display:block}
.reel a .stage{transition:border-color .4s var(--ease)}
.reel a:hover .stage{border-color:var(--fg)}
.reel figcaption{display:flex;flex-direction:column;gap:2px}
.reel figcaption span{text-transform:none;letter-spacing:0}
.reel figcaption .ideal b{color:var(--accent)}
.split{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:24px;align-items:start}
@media(max-width:900px){.split{grid-template-columns:minmax(0,1fr)}}
.section{margin:0 0 16px}
.foot{margin-top:96px;padding-top:20px;border-top:1px solid var(--border);color:var(--muted);font:400 16px/1.5 var(--mono);text-transform:uppercase;letter-spacing:.04em}
@media(max-width:1023px){
.frame{grid-template-columns:minmax(0,1fr);padding:28px 16px 96px}
.rail{position:static;display:flex;flex-wrap:wrap;align-items:center;gap:4px 18px;margin:0 0 40px;padding-bottom:16px;border-bottom:1px solid var(--border)}
.rail .brand{margin:0 24px 0 0}
.rail a.item{border-left:0;padding:4px 0}
.rail a.item.on{color:var(--accent)}
.rail .links{margin:0 0 0 auto;flex-direction:row;gap:18px}
h1{font-size:40px}h2{font-size:28px}.stats .v{font-size:30px}
.eyebrow{margin-top:64px}}
"""

JS_NAV = """
document.documentElement.classList.add('js');
(function(){
  var items=[].slice.call(document.querySelectorAll('.rail a.item'));
  var secs=items.map(function(a){return document.getElementById(a.getAttribute('href').slice(1));});
  if(!secs.length||secs.some(function(s){return !s;}))return;
  var cur=null,tick=false;
  function update(){
    tick=false;
    var y=window.scrollY+window.innerHeight*0.25,id=secs[0].id;
    for(var i=0;i<secs.length;i++){if(secs[i].getBoundingClientRect().top+window.scrollY<=y)id=secs[i].id;}
    if(window.innerHeight+window.scrollY>=document.body.scrollHeight-2)id=secs[secs.length-1].id;
    if(cur!==id){cur=id;items.forEach(function(a){a.classList.toggle('on',a.getAttribute('href')==='#'+id);});}
  }
  window.addEventListener('scroll',function(){if(!tick){tick=true;requestAnimationFrame(update);}},{passive:true});
  window.addEventListener('resize',update);update();
})();
"""

JS_TABS = """
(function(){
  var groups=document.querySelectorAll('[data-tabs]');
  groups.forEach(function(g){
    var btns=[].slice.call(g.querySelectorAll('.tabs button'));
    var panels=[].slice.call(g.querySelectorAll('.panel'));
    function show(id,push){
      btns.forEach(function(b){b.classList.toggle('on',b.dataset.panel===id);});
      panels.forEach(function(p){p.classList.toggle('on',p.id===id);});
      if(push&&history.replaceState)history.replaceState(null,'','#'+id);
    }
    btns.forEach(function(b){b.addEventListener('click',function(){show(b.dataset.panel,true);});});
    var h=location.hash.slice(1);
    show(panels.some(function(p){return p.id===h;})?h:panels[0].id,false);
  });
})();
"""


def videos_scored(results_dir):
    """All-time count of model-generated videos that were scored: every
    episode in every real-model population file, superseded model versions
    included (they were evaluated; the site just no longer ranks them).
    The user wants this number forward-facing (2026-10-05)."""
    import json
    from mcphysbench.adapters import MODEL_REGISTRY
    keys = sorted(MODEL_REGISTRY, key=len, reverse=True)
    total = 0
    for f in pathlib.Path(results_dir).glob("l0_demo_*.json"):
        name = f.stem[len("l0_demo_"):]
        if not any(name.endswith("_" + k) for k in keys):
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        total += len(d.get("seeds") or d.get("survival", {}).get("events", []))
    return total


def esc(s):
    return html.escape(str(s))


def eyebrow(n, label, anchor):
    """A numbered section label over a heading; `anchor` is the section id the page nav links to."""
    return f'<div class="eyebrow" id="{esc(anchor)}">{n:02d} / {esc(label)}</div>'


def nav_items(items):
    """items: list of (anchor, label). Numbers match the eyebrows."""
    return "".join(f'<a class="item" href="#{esc(a)}"><span class="n">{i:02d}</span>{esc(l)}</a>' for i, (a, l) in enumerate(items, 1))


def stats_row(stats):
    """stats: list of (value, label, note-or-empty)."""
    return '<div class="stats">' + "".join(
        f'<div><div class="v">{esc(v)}</div><div class="k">{esc(k)}</div>' + (f'<div class="s">{esc(s)}</div>' if s else "") + "</div>"
        for v, k, s in stats) + "</div>"


def noteblock(label, body_html):
    return f'<div class="noteblock"><div class="lbl">{esc(label)}</div>{body_html}</div>'


def shell(title, brand_href, nav_html, rail_links, body_html, extra_css="", extra_js="", wide=False, head_extra=""):
    """The page frame: fonts, paint, the rail (brand, numbered nav, links) and the article."""
    links = "".join(f'<a href="{esc(h)}">{esc(l)}</a>' for h, l in rail_links)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{esc(title)}</title>{FONT_LINK}{head_extra}<style>{CSS}{extra_css}</style></head><body>"
            f"<div class='frame{' wide' if wide else ''}'><nav class='rail'><a class='brand' href='{esc(brand_href)}'>{esc(PUBLIC_NAME)}</a>{nav_html}"
            f"<div class='links'>{links}</div></nav><article>{body_html}</article></div>"
            f"<script>{JS_NAV}{extra_js}</script></body></html>")
