"""Local, self-contained scene/evidence viewer; no fabricated scene assets."""

import json
from pathlib import Path
import re
import numpy as np


def write_viewer(output, report, before, after):
    output = Path(output)
    folder = output / "visualization"
    folder.mkdir(parents=True, exist_ok=True)

    def sampled(points):
        p = np.asarray(points, float).reshape(-1, 3)
        if not np.isfinite(p).all():
            raise ValueError("Finite nonempty scene clouds required")
        return p[:: max(1, len(p) // 5000)].tolist()

    events = []
    for event in report["events"]:
        names = []
        for o in event.get("evidence", {}).get("observations", []):
            for key in ["evidence_file", "mask_file"]:
                name = o.get("provenance", {}).get(key)
                if name and re.fullmatch(r"[A-Za-z0-9_-]+\.(?:png|jpg|jpeg)", name):
                    names.append(name)
        events.append(
            {
                k: event.get(k)
                for k in [
                    "id",
                    "semantic_label",
                    "change",
                    "before_position",
                    "after_position",
                    "explanation",
                    "decision_status",
                    "scale_status",
                ]
            }
            | {"evidence_files": list(dict.fromkeys(names))}
        )
    payload = {
        "scene_id": report.get("scene_id", "scene"),
        "before": sampled(before),
        "after": sampled(after),
        "events": events,
        "scale_status": report["events"][0].get("scale_status")
        if report["events"]
        else "Consult registration and scale diagnostics",
    }
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SpatialMemory-X scene evidence</title><style>body{margin:0;background:#111b27;color:#eff4fa;font:15px system-ui}header{padding:16px 24px;border-bottom:1px solid #334}h1{font-size:24px;margin:0 0 8px}main{display:grid;grid-template-columns:1fr 300px}canvas{width:100%;height:72vh;touch-action:none}aside{padding:18px;overflow:auto;max-height:72vh}label{margin-right:16px}select{padding:6px}button{display:block;width:100%;margin:6px 0;text-align:left;padding:8px;background:#253447;color:white;border:1px solid #456;border-radius:4px}a{color:#83c5ff;display:block}p{line-height:1.5}@media(max-width:760px){main{grid-template-columns:1fr}aside{max-height:35vh}canvas{height:55vh}}</style><header><h1>SpatialMemory-X · Scene evidence</h1><p id="summary"></p><label><input id="before" type="checkbox" checked>Before</label><label><input id="after" type="checkbox" checked>After</label><label>Decisions <select id="filter"><option value="ALL">All</option><option value="CHANGES">Change predictions</option><option>UNKNOWN</option><option>UNCHANGED</option></select></label><p>Drag to rotate · Scroll to zoom · Select a decision for its evidence</p></header><main><canvas id="view"></canvas><aside><div id="decisions"></div><div id="detail"></div></aside></main><script>const data=PAYLOAD;const c=document.getElementById('view'),ctx=c.getContext('2d'),detail=document.getElementById('detail');let rx=-.3,ry=.2,scale=1,last=null,fitted=false;const center=data.before.reduce((a,p)=>a.map((v,i)=>v+p[i]/data.before.length),[0,0,0]),colors={MOVED:'#ffbd53',ADDED:'#65df99',REMOVED:'#ff747c',UNKNOWN:'#adb6c2',UNCHANGED:'#71b7ff'};function selected(){const f=document.getElementById('filter').value;return data.events.filter(e=>f==='ALL'||e.change===f||(f==='CHANGES'&&['MOVED','ADDED','REMOVED'].includes(e.change)))}function project(p){let[x,y,z]=p.map((v,i)=>v-center[i]);let X=x*Math.cos(ry)+z*Math.sin(ry),Z=-x*Math.sin(ry)+z*Math.cos(ry),Y=y*Math.cos(rx)-Z*Math.sin(rx);return[c.width/2+X*scale,c.height/2-Y*scale]}function draw(){c.width=c.clientWidth;c.height=c.clientHeight;if(!fitted){let r=Math.max(...data.before.concat(data.after).map(p=>Math.hypot(...p.map((v,i)=>v-center[i]))),.1);scale=Math.min(c.width,c.height)*.38/r;fitted=true}ctx.clearRect(0,0,c.width,c.height);for(let[key,color]of[['before','#45aaff'],['after','#ffad50']]){if(!document.getElementById(key).checked)continue;ctx.fillStyle=color;for(let p of data[key]){const[x,y]=project(p);ctx.fillRect(x,y,1.5,1.5)}}for(let e of selected()){ctx.strokeStyle=ctx.fillStyle=colors[e.change]||'#fff';if(e.change==='MOVED'&&e.before_position&&e.after_position){const a=project(e.before_position),b=project(e.after_position);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}for(let p of[e.before_position,e.after_position]){if(!p)continue;let[x,y]=project(p);ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill()}}}function choose(e){detail.replaceChildren();for(let t of[e.semantic_label+' · '+e.change,e.explanation,e.scale_status||'Scale unverified']){let p=document.createElement('p');p.textContent=t;detail.append(p)}for(let name of e.evidence_files){let a=document.createElement('a');a.textContent=name;a.href='../evidence/supporting_frames/'+encodeURIComponent(name);detail.append(a)}}function list(){const div=document.getElementById('decisions');div.replaceChildren();for(let e of selected()){let b=document.createElement('button');b.textContent=e.semantic_label+' · '+e.change;b.onclick=()=>choose(e);div.append(b)}draw()}const changed=data.events.filter(e=>['MOVED','ADDED','REMOVED'].includes(e.change)).length;document.getElementById('summary').textContent=data.scene_id+' · '+changed+' change predictions · '+data.events.filter(e=>e.change==='UNKNOWN').length+' unknown · '+data.scale_status;c.onpointerdown=e=>{last=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(last){ry+=(e.clientX-last[0])*.01;rx+=(e.clientY-last[1])*.01;last=[e.clientX,e.clientY];draw()}};c.onpointerup=()=>last=null;c.addEventListener('wheel',e=>{e.preventDefault();scale*=Math.exp(-e.deltaY*.001);draw()},{passive:false});for(let id of['before','after'])document.getElementById(id).onchange=draw;document.getElementById('filter').onchange=list;onresize=()=>{fitted=false;draw()};list();</script></html>"""
    (folder / "index.html").write_text(
        html.replace("PAYLOAD", json.dumps(payload).replace("<", "\\u003c"))
    )
    return folder / "index.html"
